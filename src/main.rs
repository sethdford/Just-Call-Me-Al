// extern crate toml;

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use tracing::{info, warn, Level, error};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use std::sync::Arc;
use std::path::PathBuf;
use anyhow::{Result, anyhow};
use clap::Parser;
use csm::models::{CSMModel, CSMImpl, MoshiSpeechModel};
use csm::models::moshi_speech_model::SpeechModelError;
use csm::vocoder::{Vocoder, MimiVocoder};
use csm::models::moshi_speech_model::STTOutput;
use tch::Device;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use std::net::SocketAddr;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};

// Axum imports
use axum::{
    extract::{
        ws::WebSocketUpgrade,
        State,
    },
    response::{Html, IntoResponse, Response},
    routing::{get, get_service, MethodRouter},
    Router,
    http::{Uri, StatusCode},
    body::Body,
};
use tower_http::{
    services::ServeDir,
    trace::{DefaultMakeSpan, TraceLayer},
};

// CLI Arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,
    #[arg(short, long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value = "config.toml")]
    config: String,
}

// Config struct
#[derive(Deserialize, Debug, Clone)]
struct Config {
    #[serde(default = "default_moshi_model_path")]
    moshi_model_path: String,
    #[serde(default = "default_tokenizer_path")]
    tokenizer_path: String,
    #[serde(default = "default_mimi_model_path")]
    mimi_model_path: String,
    #[serde(default = "default_asr_delay")]
    asr_delay_in_tokens: usize,
}

fn default_moshi_model_path() -> String { "models/moshi/language_model.safetensors".to_string() }
fn default_tokenizer_path() -> String { "models/moshi/tokenizer.model".to_string() }
fn default_mimi_model_path() -> String { "models/mimi/model.safetensors".to_string() }
fn default_asr_delay() -> usize { 6 }

// Application State
#[derive(Clone)]
struct AppState {
    csm_model: Arc<CSMImpl>,
    vocoder: Arc<Box<dyn Vocoder + Send + Sync>>,
    speech_model: Option<Arc<MoshiSpeechModel>>,
}

// WebSocket Message Types
#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum ServerMessage {
    Info { message: String },
    Error { message: String },
    Transcript { 
        text: String, 
        start_time: f64, 
        stop_time: f64, 
    },
    PartialTranscript { partial: String },
    SpeechEnded,
    SpeechCodes { codes: Vec<i32> }, // Keep original CSM output if needed
    SynthesizedAudio { data: Vec<f32>, sample_rate: u32 }, // New variant for TTS audio output
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ClientMessageType {
    Synthesize { text: String, emotion: Option<String>, style: Option<String> },
    AudioData { data: Vec<f32>, sample_rate: u32, #[serde(default)] request_codes: bool },
    EndSpeech,
    RequestReset,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .init();

    let args = Args::parse();

    // Load main config
    let config_str = std::fs::read_to_string(&args.config)
        .map_err(|e| anyhow!("Failed to read config file '{}': {}", args.config, e))?;
    let config: Config = ::toml::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse config file '{}': {}", args.config, e))?;

    // Determine device
    let device = match args.device.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::cuda_if_available(),
        "metal" => Device::Mps,
        _ => return Err(anyhow!("Invalid device specified")),
    };
    info!("Using device: {:?}", device);

    // --- Load CSM Model --- 
    let csm_model_dir = args.model_dir.join("csm-1b"); // Define the model directory
    // let csm_config_path = args.model_dir.join("csm-1b/config.json"); // Config loaded internally by CSMImpl::new
    // let csm_weights_path = args.model_dir.join("csm-1b/model.safetensors"); // Weights loaded internally by CSMImpl::new
    // let csm_config = CsmModelConfig::from_file(&csm_config_path)?;
    let csm_impl = CSMImpl::new(&csm_model_dir, device)?; // Use CSMImpl::new with model dir and device
    let csm_model = Arc::new(csm_impl);
    info!("CSM model loaded.");
    
    // --- Create Vocoder Instance --- 
    let vocoder_sample_rate = 24000;
    let mimi_vocoder_path = args.model_dir.join("mimi/model.safetensors");
    info!("Initializing MimiVocoder with model: {:?}", mimi_vocoder_path);

    let mut vocoder = MimiVocoder::new(
        vocoder_sample_rate,
        device,
    )?;
    vocoder.load_model(mimi_vocoder_path)?;
    let vocoder: Arc<Box<dyn Vocoder + Send + Sync>> = Arc::new(Box::new(vocoder));
    info!("Using MimiVocoder with sample rate: {}", vocoder.sample_rate());
    // --------------------------------

    // Initialize Moshi Speech Model if files exist
    let speech_model = if std::path::Path::new(&config.moshi_model_path).exists()
           && std::path::Path::new(&config.tokenizer_path).exists()
           && std::path::Path::new(&config.mimi_model_path).exists() {
        info!("Initializing Moshi Speech model...");
        let moshi_device = match device {
            Device::Cpu => moshi::candle::Device::Cpu,
            Device::Cuda(_) => {
                if cfg!(feature = "cuda") {
                    match moshi::candle::Device::new_cuda(0) {
                        Ok(cuda_device) => cuda_device,
                        Err(e) => {
                            warn!("Failed to create CUDA device for Speech Model: {}, falling back to CPU", e);
                            moshi::candle::Device::Cpu
                        }
                    }
                } else {
                    moshi::candle::Device::Cpu
                }
            },
            Device::Mps => moshi::candle::Device::Cpu, // Fallback for Moshi
            _ => moshi::candle::Device::Cpu,
        };
        
        // Create the necessary configs
        let mimi_config = moshi::mimi::Config::v0_1(None); // Use defaults for now
        let lm_config = moshi::lm::Config::v0_1(); // Use defaults for now
        
        match MoshiSpeechModel::new(
            &config.moshi_model_path,
            &config.tokenizer_path,
            &config.mimi_model_path, 
            mimi_config,
            lm_config,
            config.asr_delay_in_tokens,
            2, // Provide a default acoustic_delay (e.g., 2)
            moshi_device,
        ) {
            Ok(model) => {
                info!("Moshi Speech model initialized successfully");
                Some(Arc::new(model))
            }
            Err(e) => {
                error!("Failed to initialize Moshi Speech model: {}", e);
                None
            }
        }
    } else {
        warn!("Required Moshi model files not found. Speech capabilities will be disabled.");
        None
    };

    // Create application state
    let app_state = Arc::new(AppState {
        csm_model,
        vocoder,
        speech_model,
    });

    // Setup router
    let app = Router::new()
        .route("/ws", get(ws_handler)) // WebSocket route
        // Use ServeDir directly as the fallback service
        .fallback_service(ServeDir::new("app/out")) 
        .with_state(app_state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::default().include_headers(true)),
        );

    // Run server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// Route Handlers
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

// WebSocket Handler
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let session_id = Uuid::new_v4();
    info!(session_id = %session_id, "WebSocket client connected");
    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(TokioMutex::new(sender));

    // Track if the client wants audio codes
    let request_codes = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Check if Speech model (and thus STT) is available
    let stt_available = state.speech_model.is_some();
    let speech_model_instance = state.speech_model.clone(); // Clone Option<Arc<MoshiSpeechModel>>

    // Create channels for STT if available
    let (audio_tx, text_rx_option): (
        Option<mpsc::Sender<Vec<f32>>>,
        Option<mpsc::Receiver<Result<Vec<STTOutput>, SpeechModelError>>> // Re-apply explicit type annotation
    ) = if stt_available {
        if let Some(model) = &speech_model_instance {
            match model.start_streaming(24000) { // Call on MoshiSpeechModel
                Ok((tx, rx)) => (Some(tx), Some(rx)),
                Err(e) => {
                    error!("Failed to start STT streaming: {}", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // Send initial message
    let init_msg = serde_json::to_string(&ServerMessage::Info { message: format!("Connected. STT Available: {}", stt_available) }).unwrap();
    if let Err(e) = sender.lock().await.send(Message::Text(init_msg)).await {
        error!("Error sending initial message: {}", e);
        return;
    }

    // Clone sender and state for various tasks
    let sender_for_stt = sender.clone();
    let request_codes_for_stt = request_codes.clone();
    let sender_for_receiver = sender.clone();
    let state_for_receiver = state.clone(); // Clone state for receiver loop
    let audio_tx_for_receiver = audio_tx.clone(); // Clone audio_tx for receiver loop

    // Spawn STT output processing task
    let stt_task = if let Some(mut rx) = text_rx_option { // Take ownership of rx
        Some(tokio::spawn(async move {
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(outputs) => {
                        // Iterate through the Vec<STTOutput>
                        for output in outputs {
                            // Access fields directly from the STTOutput struct
                            let transcript_msg = ServerMessage::Transcript {
                                text: output.word.text,
                                start_time: output.word.start_time,
                                stop_time: output.word.stop_time,
                                // Omit audio_codes for now
                            };

                            if let Ok(json_msg) = serde_json::to_string(&transcript_msg) {
                                let mut guard = sender_for_stt.lock().await;
                                if let Err(e) = guard.send(Message::Text(json_msg)).await {
                                    error!(session_id = %session_id, "Error sending STT transcript message: {}", e);
                                    break; // Exit loop on send error
                                }
                            } else {
                                error!(session_id = %session_id, "Failed to serialize STT transcript message");
                            }
                            
                            // TODO: Decide if/how to handle SpeechEnded/PartialTranscript if Moshi provides them via STTOutput 
                            // The current STTOutput struct only has 'word' and 'audio_codes'.
                            // If Moshi signals the end via an empty Vec or a specific error, 
                            // that logic should be handled outside this inner loop.
                        }
                    }
                    Err(e) => {
                        error!(session_id = %session_id, "Error receiving STT output: {}", e);
                        let error_resp = ServerMessage::Error { message: format!("STT Processing Error: {}", e) };
                        if let Ok(err_json) = serde_json::to_string(&error_resp) {
                           let mut guard = sender_for_stt.lock().await;
                           let _ = guard.send(Message::Text(err_json)).await; // Ignore error
                        }
                        break; // Stop processing on error
                    }
                }
            }
            info!(session_id = %session_id, "STT receiver task finished");
        }))
    } else {
        None
    };

    // Receive loop
    let receive_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<ClientMessageType>(&text) {
                        Ok(ClientMessageType::AudioData { data, sample_rate, request_codes: req_codes }) => {
                             request_codes_for_stt.store(req_codes, std::sync::atomic::Ordering::Relaxed);
                            if let Some(ref tx) = audio_tx_for_receiver {
                                if sample_rate != 24000 {
                                    warn!(session_id = %session_id, "Received audio with unexpected sample rate: {}, expected 24000", sample_rate);
                                    // TODO: Add resampling if needed
                                    continue;
                                }
                                if tx.send(data).await.is_err() {
                                    error!(session_id = %session_id, "Audio channel closed, cannot send audio data.");
                                    break;
                                }
                            } else {
                                warn!(session_id = %session_id, "Received audio data, but STT is not available.");
                            }
                        },
                        Ok(ClientMessageType::Synthesize { text, .. }) => {
                             info!(session_id = %session_id, "Received synthesize request for text: {}", text);
                             if let Some(model) = &state_for_receiver.speech_model {
                                 // Clone Arc for the synthesis task
                                 let model_clone = model.clone();
                                 let sender_clone = sender_for_receiver.clone();
                                 let text_clone = text.clone(); // Clone text for the async block
                                 let session_id_clone = session_id; // Clone session_id for the async block

                                 tokio::spawn(async move {
                                     info!(session_id = %session_id_clone, "Spawning TTS task for: {}", text_clone);
                                     match model_clone.synthesize_audio(&text_clone, None, None, None).await {
                                         Ok(audio_data) => {
                                             info!(session_id = %session_id_clone, "TTS synthesis successful, sending audio ({} samples).", audio_data.len());
                                             let response = ServerMessage::SynthesizedAudio { 
                                                 data: audio_data, 
                                                 sample_rate: 24000 // Assuming TTS output SR is 24kHz
                                             };
                                             if let Ok(json_msg) = serde_json::to_string(&response) {
                                                let mut guard = sender_clone.lock().await;
                                                if let Err(e) = guard.send(Message::Text(json_msg)).await {
                                                    error!(session_id = %session_id_clone, "Failed to send synthesized audio: {}", e);
                                                }
                                             } else {
                                                 error!(session_id = %session_id_clone, "Failed to serialize synthesized audio response.");
                                             }
                                         },
                                         Err(e) => {
                                             error!(session_id = %session_id_clone, "TTS synthesis failed: {}", e);
                                             let error_resp = ServerMessage::Error { message: format!("TTS Synthesis failed: {}", e) };
                                             if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                                let mut guard = sender_clone.lock().await;
                                                let _ = guard.send(Message::Text(err_json)).await;
                                             }
                                         }
                                     }
                                     info!(session_id = %session_id_clone, "TTS task finished for: {}", text_clone);
                                 });
                             } else {
                                 warn!(session_id = %session_id, "Received synthesize request, but TTS model is not available.");
                                 let error_resp = ServerMessage::Error { message: "TTS functionality is not available.".to_string() };
                                 if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                     let mut sender_guard = sender_for_receiver.lock().await;
                                     let _ = sender_guard.send(Message::Text(err_json)).await;
                                 }
                             }
                         },
                        Ok(ClientMessageType::EndSpeech) => {
                            info!(session_id = %session_id, "Received end_speech signal from client.");
                            // Optionally signal the STT model or other components
                            let end_msg = ServerMessage::SpeechEnded;
                            let mut sender_guard = sender_for_receiver.lock().await;
                            let _ = sender_guard.send(Message::Text(serde_json::to_string(&end_msg).unwrap())).await;
                        },
                        Ok(ClientMessageType::RequestReset) => {
                            info!(session_id = %session_id, "Received reset signal from client.");
                            if let Some(model) = &state_for_receiver.speech_model {
                                model.reset().await;
                                info!(session_id = %session_id, "Speech model state reset.");
                                let reset_msg = ServerMessage::Info { message: "Model state reset.".to_string() };
                                let mut sender_guard = sender_for_receiver.lock().await;
                                let _ = sender_guard.send(Message::Text(serde_json::to_string(&reset_msg).unwrap())).await;
                            } else {
                                warn!(session_id = %session_id, "Received reset request, but speech model is not available.");
                            }
                        },
                        Err(e) => {
                             warn!(session_id = %session_id, raw_text = text, "Failed to parse client message: {}", e);
                             let error_resp = ServerMessage::Error { message: format!("Invalid message format: {}", e) };
                             if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                 let mut sender_guard = sender_for_receiver.lock().await;
                                 let _ = sender_guard.send(Message::Text(err_json)).await;
                             }
                        }
                    }
                },
                Ok(Message::Binary(data)) => {
                    warn!(session_id = %session_id, "Received unexpected binary message ({} bytes), ignoring.", data.len());
                },
                Ok(Message::Close(_)) => {
                    info!(session_id = %session_id, "Client requested close.");
                    break;
                },
                Ok(Message::Ping(p)) => {
                    let mut sender_guard = sender_for_receiver.lock().await;
                    if sender_guard.send(Message::Pong(p)).await.is_err() {
                        break; // Exit if pong fails
                    }
                },
                Ok(Message::Pong(_)) => {
                    // Pong received, do nothing
                },
                Err(e) => {
                    error!(session_id = %session_id, "Error receiving message: {}", e);
                    break;
                }
            }
        }
        info!(session_id = %session_id, "Receiver task finished.");
    });

    // Wait for tasks to complete (or abort if needed)
    if let Some(task) = stt_task {
        let _ = task.await; // Wait for STT task
    }
    let _ = receive_task.await; // Wait for receiver task

    info!(session_id = %session_id, "WebSocket client disconnected");
} 
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
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "init")]
    Init { stt_available: bool },
    #[serde(rename = "audio_info")]
    AudioInfo { sample_rate: u32, channels: u16 },
    #[serde(rename = "audio_chunk")]
    AudioChunk { data: String }, // Base64 encoded audio
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error(String),
    #[serde(rename = "transcript")]
    Transcript { text: String, start_time: f64, stop_time: f64, audio_codes: serde_json::Value },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ClientMessageType {
    #[serde(rename = "synthesize")]
    Synthesize { text: String, emotion: Option<String>, style: Option<String> },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "stop_audio")]
    StopAudio,
    #[serde(rename = "audio_data")]
    AudioData { data: Vec<f32>, sample_rate: u32, #[serde(default)] request_codes: bool },
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
    let init_msg = serde_json::to_string(&ServerMessage::Init { stt_available }).unwrap();
    if let Err(e) = sender.lock().await.send(Message::Text(init_msg)).await {
        error!("Error sending initial message: {}", e);
        return;
    }

    // Clone sender and request_codes flag for STT task
    let sender_for_stt = sender.clone();
    let request_codes_for_stt = request_codes.clone();

    // Create STT output task (receives Result<Vec<STTOutput>, SpeechModelError>)
    let stt_task = if let Some(mut rx) = text_rx_option { // Take ownership of rx
        Some(tokio::spawn(async move {
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(outputs) => {
                        if outputs.is_empty() { continue; }

                        for output in outputs {
                            let word = output.word;
                            let include_codes = request_codes_for_stt.load(std::sync::atomic::Ordering::Relaxed);
                            let codes_json = if include_codes {
                                // TODO: Implement Tensor serialization or conversion if needed.
                                // For now, just send null as Tensor cannot be directly serialized.
                                // match output.audio_codes {
                                //     Some(codes) => json!(codes), // This causes the error
                                //     None => json!(null),
                                // }
                                json!(null) // Send null for now
                            } else {
                                json!(null)
                            };

                            let transcript_msg = ServerMessage::Transcript {
                                text: word.text,
                                start_time: word.start_time,
                                stop_time: word.stop_time,
                                audio_codes: codes_json,
                            };
                            let msg = serde_json::to_string(&transcript_msg).unwrap();

                            let sender_clone = sender_for_stt.clone();
                            let msg_clone = msg.clone();
                            tokio::spawn(async move {
                                let mut guard = sender_clone.lock().await;
                                if let Err(e) = guard.send(Message::Text(msg_clone)).await {
                                    error!("Failed to send transcript: {}", e);
                                }
                            });
                        }
                    }
                    Err(e) => {
                        error!(session_id = %session_id, "Error receiving STT output: {}", e);
                        let error_resp = ServerMessage::Error(format!("STT Processing Error: {}", e));
                        if let Ok(err_json) = serde_json::to_string(&error_resp) {
                           let mut guard = sender_for_stt.lock().await;
                           let _ = guard.send(Message::Text(err_json)).await;
                        }
                        // Consider whether to break the loop on error
                    }
                }
            }
            info!(session_id = %session_id, "STT processing task finished.");
        }))
    } else {
        None
    };

    // Create task to handle incoming messages
    let receiver_task = tokio::spawn(async move {
        // Clone Arcs needed for this task
        let audio_tx_for_receiver = audio_tx.clone(); // Clone Option<Sender>
        let sender_for_receiver = sender.clone(); // Clone Arc<Mutex<Sender>>
        let state_for_receiver = state.clone(); // Clone Arc<AppState>
        let request_codes_for_receiver = request_codes.clone(); // Clone Arc<AtomicBool>

        while let Some(result) = receiver.next().await {
            match result {
                Ok(message) => {
                    match message {
                        Message::Text(text) => {
                            match serde_json::from_str::<ClientMessageType>(&text) {
                                Ok(ClientMessageType::Synthesize { text, emotion: _, style: _ }) => {
                                    // TODO: Handle emotion/style later. For now, use standard synthesize.
                                    info!(session_id = %session_id, "Received synthesize request for text: {}", text);
                                    let csm_model_clone = state_for_receiver.csm_model.clone();
                                    let sender_for_synthesis = sender_for_receiver.clone(); // Clone sender for the spawned task
                                    
                                    tokio::spawn(async move {
                                        // Call synthesize with correct args (temp, top_k, seed)
                                        // Using None for now, extract from ClientMessage later if needed.
                                        let result = csm_model_clone.synthesize(
                                            &text,
                                            None, // temperature
                                            None, // top_k
                                            None, // seed
                                        ).await;
                                        
                                        // Handle result: Send audio via WebSocket
                                        match result {
                                            Ok(audio_i16) => {
                                                info!(session_id = %session_id, "Synthesized {} audio samples (i16)", audio_i16.len());
                                                // Convert i16 to f32 for WebSocket transmission if needed, or send as i16 bytes
                                                let bytes: Vec<u8> = audio_i16.into_iter()
                                                    .flat_map(|s| s.to_le_bytes())
                                                    .collect();
                                                let mut guard = sender_for_synthesis.lock().await; // Use the cloned sender
                                                if let Err(e) = guard.send(Message::Binary(bytes)).await {
                                                     error!(session_id = %session_id, "Failed to send synthesized audio: {}", e);
                                                }
                                            }
                                            Err(e) => {
                                                error!(session_id = %session_id, "Synthesis failed: {}", e);
                                                let error_resp = ServerMessage::Error(format!("Synthesis failed: {}", e));
                                                if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                                     let mut guard = sender_for_synthesis.lock().await; // Use the cloned sender
                                                    let _ = guard.send(Message::Text(err_json)).await;
                                                }
                                            }
                                        }
                                    });
                                },
                                Ok(ClientMessageType::Ping) => {
                                    let pong_msg = ServerMessage::Pong;
                                    if let Ok(json_msg) = serde_json::to_string(&pong_msg) {
                                        let mut sender_guard = sender_for_receiver.lock().await;
                                        if let Err(e) = sender_guard.send(Message::Text(json_msg)).await {
                                            error!(session_id = %session_id, "Receive task: Failed to send pong response: {}. Closing.", e);
                                            break;
                                        }
                                    }
                                },
                                Ok(ClientMessageType::StopAudio) => {
                                    info!(session_id = %session_id, "Receive task: Received stop_audio signal from client.");
                                    break;
                                },
                                Ok(ClientMessageType::AudioData { data, sample_rate: _, request_codes: req_codes }) => {
                                    request_codes_for_receiver.store(req_codes, std::sync::atomic::Ordering::Relaxed);
                                    if let Some(tx) = &audio_tx_for_receiver {
                                        if let Err(e) = tx.send(data).await {
                                            error!("Failed to send audio data to STT channel: {}", e);
                                        }
                                    } else if !stt_available {
                                        warn!(session_id = %session_id, "Received audio_data but STT is not available/initialized.");
                                    }
                                },
                                Err(e) => {
                                     warn!(session_id = %session_id, raw_text = text, "Failed to parse client message: {}", e);
                                     let error_resp = ServerMessage::Error(format!("Invalid message format: {}", e));
                                     if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                         let mut sender_guard = sender_for_receiver.lock().await;
                                         let _ = sender_guard.send(Message::Text(err_json)).await.map_err(|e| {
                                             error!(session_id = %session_id, "Receive task: Failed to send error response: {}. Closing.", e);
                                         });
                                     }
                                }
                            }
                        }
                        Message::Binary(data) => {
                             if let Some(tx) = &audio_tx_for_receiver { // Use cloned tx
                                 if data.len() % 2 == 0 {
                                     let samples: Vec<f32> = data
                                         .chunks_exact(2)
                                         .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                                         .collect();
                                     if let Err(e) = tx.send(samples).await {
                                         error!("Failed to send binary audio data to STT channel: {}", e);
                                     }
                                 } else {
                                      warn!(session_id = %session_id, "Received binary data with odd length, cannot decode as i16 PCM.");
                                 }
                             } else {
                                warn!(session_id = %session_id, "Received binary audio data but STT is not available.");
                            }
                        }
                        Message::Ping(ping) => {
                            let mut sender_guard = sender_for_receiver.lock().await;
                            if let Err(e) = sender_guard.send(Message::Pong(ping)).await {
                                error!(session_id = %session_id, "Receive task: Failed to send pong (from ping): {}. Closing.", e);
                                break;
                            }
                        }
                        Message::Pong(_) => { /* Keepalive */ }
                        Message::Close(_) => {
                            info!(session_id = %session_id, "Client requested close.");
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!(session_id = %session_id, "WebSocket receive error: {}", e);
                    break;
                }
            }
        }
        info!(session_id = %session_id, "Receive task FINISHED.");
    });

    // Wait for tasks to complete
    if let Some(stt_task) = stt_task {
        tokio::select! {
            res = receiver_task => { if let Err(e) = res { error!("Receiver task join error: {}", e); } },
            res = stt_task => { if let Err(e) = res { error!("STT task join error: {}", e); } },
        }
    } else {
        receiver_task.await.unwrap_or_else(|e| error!("Error in receiver task: {}", e));
    }

    info!(session_id = %session_id, "WebSocket client disconnected and handle_socket finished.");
} 
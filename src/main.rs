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

// Project modules
mod context; // Add context module
mod llm_integration; // Add LLM integration module
use context::ConversationHistory; // Import the history struct
use llm_integration::{LlmProcessor, LlmConfig, LlmType, create_llm_service}; // Import LLM types

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
    #[serde(default = "default_llm_type")]
    llm_type: String,
    #[serde(default = "default_llm_model_path")]
    llm_model_path: String,
}

fn default_moshi_model_path() -> String { "models/moshi/language_model.safetensors".to_string() }
fn default_tokenizer_path() -> String { "models/moshi/tokenizer.model".to_string() }
fn default_mimi_model_path() -> String { "models/mimi/model.safetensors".to_string() }
fn default_asr_delay() -> usize { 6 }
fn default_llm_type() -> String { "mock".to_string() }
fn default_llm_model_path() -> String { "models/llm/model.safetensors".to_string() }

// Application State
#[derive(Clone)]
struct AppState {
    csm_model: Arc<CSMImpl>,
    vocoder: Arc<Box<dyn Vocoder + Send + Sync>>,
    speech_model: Option<Arc<MoshiSpeechModel>>,
    llm_processor: Arc<dyn LlmProcessor>,
    // Note: AppState is typically shared. For per-connection state like history,
    // it's better managed within the WebSocket handler (handle_socket).
    // Leaving this commented out as a design note.
    // conversation_history: Arc<TokioMutex<ConversationHistory>>, // Use Mutex for shared state
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

    // Initialize LLM processor
    let llm_config = match config.llm_type.to_lowercase().as_str() {
        "llama" => LlmConfig {
            llm_type: LlmType::Llama,
            model_path: Some(args.model_dir.join(&config.llm_model_path)),
            use_gpu: device != Device::Cpu,
            embedding_dim: 768,
            ..Default::default()
        },
        "mistral" => LlmConfig {
            llm_type: LlmType::Mistral,
            model_path: Some(args.model_dir.join(&config.llm_model_path)),
            use_gpu: device != Device::Cpu,
            embedding_dim: 1024,
            ..Default::default()
        },
        "local" => LlmConfig {
            llm_type: LlmType::Local,
            model_path: Some(args.model_dir.join(&config.llm_model_path)),
            use_gpu: device != Device::Cpu,
            embedding_dim: 768,
            ..Default::default()
        },
        _ => {
            info!("Using mock LLM processor for development");
            LlmConfig::default() // Uses Mock LLM type
        }
    };
    
    let llm_processor = create_llm_service(llm_config)
        .map_err(|e| anyhow!("Failed to initialize LLM processor: {}", e))?;
    info!("LLM processor initialized.");

    // Create application state
    let app_state = Arc::new(AppState {
        csm_model,
        vocoder,
        speech_model,
        llm_processor,
        // conversation_history: Arc::new(TokioMutex::new(ConversationHistory::new(None))), // Initialize shared history if needed
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
    info!("WebSocket connection upgrade requested");
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

// WebSocket Handler
async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let session_id = Uuid::new_v4();
    info!("New WebSocket connection: {}", session_id);

    // --- Per-Connection State --- 
    let mut conversation_history = ConversationHistory::new(None); // Each connection gets its own history
    let mut current_asr_stream: Option<mpsc::Sender<Vec<f32>>> = None;
    let mut current_asr_task: Option<tokio::task::JoinHandle<Result<STTOutput, SpeechModelError>>> = None;

    // Send initial info message
    let _ = socket.send(Message::Text(serde_json::to_string(&ServerMessage::Info {
        message: "Connection established".to_string()
    }).unwrap())).await;

    // --- Message Loop --- 
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                info!("Received text message from {}: {}", session_id, text);
                match serde_json::from_str::<ClientMessageType>(&text) {
                    Ok(client_msg) => {
                        match client_msg {
                            ClientMessageType::Synthesize { text, emotion, style } => {
                                info!("Synthesis request: '{}' (Emotion: {:?}, Style: {:?})", text, emotion, style);

                                // 1. Add current user turn to history
                                let user_turn = context::ConversationTurn::new(context::Speaker::User, text.clone());
                                conversation_history.add_turn(user_turn);

                                // 2. Use LLM to generate a contextual response
                                let llm_response = match state.llm_processor.generate_response(&conversation_history) {
                                    Ok(response) => {
                                        info!("LLM generated response: {}", response);
                                        response
                                    },
                                    Err(e) => {
                                        error!("LLM response generation failed: {}", e);
                                        format!("Sorry, I couldn't process that correctly. Error: {}", e)
                                    }
                                };

                                // 3. Use LLM to generate contextual embeddings
                                let _contextual_embedding = match state.llm_processor.generate_embeddings(&conversation_history) {
                                    Ok(embedding) => {
                                        info!("Generated contextual embedding with dimension {}", embedding.dim());
                                        Some(embedding)
                                    },
                                    Err(e) => {
                                        error!("Contextual embedding generation failed: {}", e);
                                        None
                                    }
                                };

                                // 4. Pass LLM response and contextual embedding to the model
                                // For now, we'll just use a placeholder response
                                
                                // --- Placeholder Synthesis Logic --- 
                                let sample_rate = state.vocoder.sample_rate();
                                let placeholder_audio: Vec<f32> = vec![0.1; (sample_rate / 4) as usize];
                                
                                info!("Sending placeholder synthesized audio...");
                                let send_result = socket.send(Message::Binary(placeholder_audio.iter().flat_map(|f| f.to_le_bytes()).collect())).await;
                                if let Err(e) = send_result {
                                    warn!("Failed to send placeholder audio: {}", e);
                                    break;
                                }

                                // 5. Add model turn to history with the LLM-generated response
                                let model_turn = context::ConversationTurn::new(context::Speaker::Model, llm_response);
                                conversation_history.add_turn(model_turn);
                            },
                            ClientMessageType::AudioData { data, sample_rate, request_codes } => {
                                // Handle incoming audio for STT (if speech_model exists)
                                if let Some(speech_model) = state.speech_model.as_ref() {
                                    // If no ASR stream exists, create one
                                    if current_asr_stream.is_none() {
                                        let (tx, rx) = mpsc::channel(100); // Buffer size
                                        current_asr_stream = Some(tx);
                                        
                                        let model_clone = Arc::clone(speech_model);
                                        let mut history_clone = conversation_history.clone(); // Clone history for the task

                                        current_asr_task = Some(tokio::spawn(async move {
                                            info!("Starting ASR task...");
                                            // TODO: Pass conversation history to transcribe_stream
                                            let result = model_clone.transcribe_stream(rx).await;
                                            info!("ASR task finished: {:?}", result);
                                            
                                            // Add transcript to history if successful
                                            if let Ok(stt_output) = &result {
                                                if !stt_output.full_transcript.is_empty() {
                                                     history_clone.add_turn(context::ConversationTurn::new(
                                                         context::Speaker::User, 
                                                         stt_output.full_transcript.clone()
                                                     ));
                                                    // TODO: We might need a way to update the history owned by the main task here.
                                                    // Using Arc<Mutex<History>> shared between tasks is one option.
                                                }
                                            }
                                            result
                                        }));

                                        // Send partial transcripts back to client
                                        // TODO: Need a way to receive partial results from transcribe_stream
                                        // let partial_rx = ... // Get receiver for partial transcripts
                                        // tokio::spawn(async move {
                                        //     while let Some(partial) = partial_rx.recv().await {
                                        //         let _ = socket.send(... ServerMessage::PartialTranscript ...).await;
                                        //     }
                                        // });
                                    }

                                    // Send audio data to the ASR stream
                                    if let Some(tx) = &current_asr_stream {
                                        if let Err(e) = tx.send(data).await {
                                            error!("Failed to send audio to ASR stream: {}", e);
                                            // Close the stream and maybe reset state
                                            current_asr_stream = None;
                                            current_asr_task = None;
                                        }
                                    }
                                } else {
                                    warn!("Received audio data but no speech model configured/loaded.");
                                }
                            }
                            ClientMessageType::EndSpeech => {
                                info!("Received EndSpeech signal.");
                                // Close the ASR stream sender to signal the end
                                if let Some(tx) = current_asr_stream.take() {
                                    drop(tx); // Dropping sender closes the channel
                                    info!("ASR stream sender dropped.");
                                }
                                // Wait for the ASR task to finish and get the final transcript
                                if let Some(task) = current_asr_task.take() {
                                    match task.await {
                                        Ok(Ok(stt_output)) => {
                                            info!("Final Transcript: {}", stt_output.full_transcript);
                                             // Send final transcript
                                            let _ = socket.send(Message::Text(serde_json::to_string(&ServerMessage::Transcript {
                                                 text: stt_output.full_transcript, // Send full transcript
                                                 start_time: stt_output.start_time, // Use times from STTOutput
                                                 stop_time: stt_output.stop_time,
                                             }).unwrap())).await;
                                            // History update should happen inside the task or via shared state
                                        }
                                        Ok(Err(e)) => error!("ASR task failed: {}", e),
                                        Err(e) => error!("ASR task join error: {}", e),
                                    }
                                }
                                let _ = socket.send(Message::Text(serde_json::to_string(&ServerMessage::SpeechEnded).unwrap())).await;
                            }
                            ClientMessageType::RequestReset => {
                                info!("Received RequestReset signal.");
                                // Reset conversation history
                                conversation_history.clear();
                                // Reset ASR state
                                if let Some(tx) = current_asr_stream.take() { drop(tx); }
                                if let Some(task) = current_asr_task.take() { task.abort(); }
                                let _ = socket.send(Message::Text(serde_json::to_string(&ServerMessage::Info { message: "State reset".to_string() }).unwrap())).await;
                                info!("Conversation state reset for {}", session_id);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse client message: {}, message: {}", e, text);
                        let _ = socket.send(Message::Text(serde_json::to_string(&ServerMessage::Error {
                            message: format!("Invalid message format: {}", e)
                        }).unwrap())).await;
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                warn!("Received unexpected binary message from {}", session_id);
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => { /* Ignore */ }
            Ok(Message::Close(_)) => {
                info!("Client {} disconnected gracefully.", session_id);
                break;
            }
            Err(e) => {
                error!("WebSocket error for session {}: {}", session_id, e);
                break;
            }
        }
    }

    info!("Cleaning up WebSocket connection: {}", session_id);
    // Ensure any running ASR task is cleaned up
    if let Some(task) = current_asr_task { task.abort(); }
}

// Placeholder for static file handling (if needed beyond public dir)
// async fn static_handler(uri: Uri) -> impl IntoResponse { ... } 
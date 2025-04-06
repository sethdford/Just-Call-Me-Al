// extern crate toml;

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Standard library imports
use std::sync::Arc;
use std::path::PathBuf;
use std::net::SocketAddr;
use std::str::FromStr;

// Third-party imports
use anyhow::{Result, anyhow};
use clap::Parser;
use tch::Device;
use serde::{Deserialize, Serialize};
use tracing::{info, error, warn, trace};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use tokio::sync::{mpsc, Mutex as TokioMutex};
use futures::{SinkExt, stream::StreamExt};

// Consolidated CSM Project Imports
use csm::models::{
    CSMImpl,
    CSMModel,
    Device as CsmDevice,
    MoshiSpeechModel,
    config::{CsmModelConfig, MoshiModelPaths},
    ModelError,
};
use csm::vocoder::{MimiVocoder, Vocoder};

// Imports for REST API (always included now)
use axum::{
    Router,
    routing::get,
    extract::{
        ws::{Message, WebSocket},
        State,
        WebSocketUpgrade,
    },
    response::IntoResponse,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    Json,
};
use tower_http::{
    services::ServeDir,
    trace::{DefaultMakeSpan, TraceLayer},
};
use tokio::net::TcpListener;
use uuid::Uuid;
use axum_extra::headers::{authorization::Bearer, Authorization, UserAgent};
use axum_extra::TypedHeader;

// CSM crate imports (Ensure these paths are correct)
// REMOVE ALL DUPLICATE IMPORTS BELOW THIS LINE
// use csm::models::CSMImpl; // REMOVE
// use csm::vocoder::{Vocoder, MimiVocoder}; // REMOVE (already imported above)
// use csm::models::Device as CsmDevice; // REMOVE
// use csm::models::moshi_speech_model::MoshiSpeechModel; // REMOVE
// use csm::models::config::{CsmModelConfig, MoshiModelPaths}; // REMOVE

// Re-add LlmProcessor and LlmConfig, remove duplicate imports
use crate::llm_integration::{create_optimized_llm_service, LlmType, LlmConfig, LlmProcessor};
use crate::llm_integration::create_llm_history;

// Project modules
mod context; // Local context module
mod llm_integration; // Local LLM integration module

// CLI Arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
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
struct LoggingConfig {
    level: String, // Example field
}

#[derive(Deserialize, Debug, Clone)]
struct ModelConfig {
    #[serde(default = "default_asr_lm_path")]
    _asr_lm_path: String,
    #[serde(default = "default_moshi_model_path")]
    _moshi_model_path: String,
    #[serde(default = "default_tokenizer_path")]
    _tokenizer_path: String,
    #[serde(default = "default_mimi_model_path")]
    _mimi_model_path: String,
    #[serde(default = "default_asr_delay")]
    _asr_delay_in_tokens: usize,
    #[serde(default = "default_csm_model_dir")]
    _csm_model_dir: String, // Assuming this was intended here
    #[serde(default = "default_device")]
    _device: String, // Assuming this was intended here
    #[serde(default = "default_llm_type")]
    llm_type: String,
    #[serde(default = "default_llm_model_path")]
    llm_model_path: String,
}

#[derive(Deserialize, Debug, Clone, Default)]
struct MimiEncoderConfig { 
    #[serde(default = "default_mimi_sample_rate")]
    sample_rate: f64, 
    // Add other necessary fields with defaults
}

#[derive(Deserialize, Debug, Clone, Default)]
struct RVQConfig {
    #[serde(default = "default_rvq_num_codebooks")]
    num_codebooks: usize,
    // Add other necessary fields with defaults
}

#[derive(Deserialize, Debug, Clone)]
struct AudioConfig {
    sample_rate: u32, // Example field
    #[serde(default)]
    mimi_encoder: MimiEncoderConfig,
    #[serde(default)]
    rvq: RVQConfig,
}

#[derive(Deserialize, Debug, Clone)]
struct ServerConfig {
    enabled: bool,
    port: u16,
    #[serde(default = "default_static_dir")]
    static_dir: String,
}

#[derive(Deserialize, Debug, Clone)]
struct Config {
    logging: LoggingConfig,
    model: ModelConfig,
    server: ServerConfig,
    audio: AudioConfig,
    // Remove fields now in ModelConfig
    // #[serde(default = "default_moshi_model_path")]
    // _moshi_model_path: String, 
    // ... other fields ...
}

// --- Default Functions ---
fn default_moshi_model_path() -> String { "moshi/language_model.safetensors".to_string() }
fn default_tokenizer_path() -> String { "moshi/tokenizer.model".to_string() }
fn default_mimi_model_path() -> String { "mimi/model.safetensors".to_string() }
fn default_asr_delay() -> usize { 6 }
fn default_csm_model_dir() -> String { "csm-1b".to_string() } 
fn default_device() -> String { "cpu".to_string() } 
fn default_llm_type() -> String { "mock".to_string() } 
fn default_llm_model_path() -> String { "llm/model.safetensors".to_string() }
fn default_static_dir() -> String { "static".to_string() }
fn default_asr_lm_path() -> String { "models/asr_lm/language_model.kenlm".to_string() }
fn default_mimi_sample_rate() -> f64 { 24000.0 }
fn default_rvq_num_codebooks() -> usize { 8 }

// --- Config Loading --- 
fn load_config(config_path: &str) -> Result<Config> {
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow!("Failed to read config file '{}': {}", config_path, e))?;
    
    // Parse unconditionally. Serde ignores #[cfg]d out fields.
    let config: Config = ::toml::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse config file '{}': {}", config_path, e))?;
    
    Ok(config)
}

// Application State
#[derive(Clone)]
struct AppState {
    _csm_model: Arc<CSMImpl>,
    vocoder: Arc<Box<dyn Vocoder + Send + Sync>>,
    speech_model: Option<Arc<MoshiSpeechModel>>,
    llm_processor: Arc<dyn LlmProcessor>,
    audio_processor: Arc<ThreadSafeAudioProcessor>,
}

// Thread-safe wrapper for AudioProcessor 
struct ThreadSafeAudioProcessor {
    inner: tokio::sync::Mutex<csm::audio::AudioProcessor>,
}

impl ThreadSafeAudioProcessor {
    fn new(processor: csm::audio::AudioProcessor) -> Self {
        Self {
            inner: tokio::sync::Mutex::new(processor),
        }
    }
    
    async fn bytes_to_samples(&self, bytes: &[u8]) -> Result<Vec<f32>, anyhow::Error> {
        let guard = self.inner.lock().await;
        guard.bytes_to_samples(bytes)
    }
    
    // Add other methods from AudioProcessor as needed, ensuring they're wrapped with async/await
    // and proper locking via the mutex
}

// WebSocket Message Types
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
    Ready, // Sent when the server is ready to accept audio
    Error { message: String },
    _ConnectionStatus { connected: bool, message: String },
    FullTranscript { transcript: String },
    _PartialTranscript { partial: String },
    SpeechEnded,
    _SpeechCodes { codes: Vec<i32> },
    EndOfStream, // Added to signal end of audio chunks
    AudioData { data: Vec<u8> }, // Sending binary data (bytes) now
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessageType {
    // Prefix unused fields
    AudioData { _data: Vec<f32>, _sample_rate: u32, #[serde(default)] _request_codes: bool },
    TextData { text: String, _temperature: Option<f64>, _top_k: Option<i64>, _seed: Option<u64> },
    Stop,
}

// Let's create placeholder types to avoid errors
#[derive(Debug)]
struct SpeechModelError(String);

impl std::fmt::Display for SpeechModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Speech Model Error: {}", self.0)
    }
}

struct STTOutput {
    full_transcript: String,
    _start_time: f64,
    _stop_time: f64,
}

// --- Server Setup & Handlers ---
async fn run_server(
    config: Config,
    csm_model: Arc<CSMImpl>,
    vocoder: Arc<Box<dyn Vocoder + Send + Sync>>,
    llm_processor: Arc<dyn LlmProcessor>,
    speech_model: Option<Arc<MoshiSpeechModel>>,
    audio_processor: Arc<ThreadSafeAudioProcessor>,
) -> Result<()> {
    // Change port from 8000 to 8001
    let server_port = 8001;
    let addr = SocketAddr::from(([0, 0, 0, 0], server_port));
    info!("REST API server listening on {}", addr);
    
    let state = AppState { 
        _csm_model: csm_model,
        vocoder,
        llm_processor,
        speech_model,
        audio_processor,
     };

    let app = Router::new()
        .route("/synthesize", axum::routing::post(synthesize_handler))
        .route("/ws", get(websocket_handler))
        .nest_service("/", ServeDir::new(&config.server.static_dir))
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::default().include_headers(true)),
        );
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Deserialize)]
struct SynthesizeRequest {
    text: String,
    temperature: Option<f32>,
    top_k: Option<i64>,
    seed: Option<i64>,
}

#[axum::debug_handler]
async fn synthesize_handler(
    State(state): State<AppState>,
    Json(payload): Json<SynthesizeRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let csm_model = state._csm_model;
    
    match csm_model.synthesize(
        &payload.text, 
        None, // conversation_history
        None, // temperature
        None, // top_k
        None, // seed
    ).await {
        Ok(audio_output) => {
            // Convert audio samples to wav bytes
            let wav_bytes = encode_wav_bytes(&audio_output.samples, audio_output.sample_rate)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                
            let content_type = HeaderValue::from_static("audio/wav");
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, content_type);
            Ok((headers, wav_bytes))
        }
        Err(e) => {
            error!("Synthesis failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Helper function to encode audio samples to wav bytes
fn encode_wav_bytes(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>, std::io::Error> {
    use std::io::{Cursor, Write};
    
    let mut buffer = Vec::new();
    
    // WAV Header (44 bytes)
    let channels: u16 = 1; // Mono
    let bits_per_sample: u16 = 16; // 16-bit PCM
    let bytes_per_sample = bits_per_sample / 8;
    let byte_rate = sample_rate * (bits_per_sample as u32) * (channels as u32) / 8;
    let block_align = channels * bytes_per_sample;
    let data_size = (samples.len() * bytes_per_sample as usize) as u32;
    let file_size = data_size + 36; // File size minus 8 bytes for RIFF and file size
    
    // RIFF header
    buffer.write_all(b"RIFF")?; // ChunkID
    buffer.write_all(&file_size.to_le_bytes())?; // ChunkSize
    buffer.write_all(b"WAVE")?; // Format
    
    // fmt sub-chunk
    buffer.write_all(b"fmt ")?; // Subchunk1ID
    buffer.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM)
    buffer.write_all(&1u16.to_le_bytes())?; // AudioFormat (1 for PCM)
    buffer.write_all(&channels.to_le_bytes())?; // NumChannels
    buffer.write_all(&sample_rate.to_le_bytes())?; // SampleRate
    buffer.write_all(&byte_rate.to_le_bytes())?; // ByteRate
    buffer.write_all(&block_align.to_le_bytes())?; // BlockAlign
    buffer.write_all(&bits_per_sample.to_le_bytes())?; // BitsPerSample
    
    // data sub-chunk
    buffer.write_all(b"data")?; // Subchunk2ID
    buffer.write_all(&data_size.to_le_bytes())?; // Subchunk2Size
    
    // Write sample data
    let mut cursor = Cursor::new(&mut buffer);
    cursor.set_position(44); // Position after header
    
    for sample in samples {
        cursor.write_all(&sample.to_le_bytes())?;
    }
    
    Ok(buffer)
}

#[axum::debug_handler]
async fn websocket_handler(
    ws: WebSocketUpgrade,
    _user_agent: Option<TypedHeader<UserAgent>>,
    _auth: Option<TypedHeader<Authorization<Bearer>>>,
    state: State<AppState>,
) -> impl IntoResponse {
    info!("Websocket connection upgrading...");
    
    // Use a struct impl with a static method to handle the Send issue
    struct WebSocketHandler;
    
    impl WebSocketHandler {
        async fn handle(socket: WebSocket, state: AppState) {
            let session_id = Uuid::new_v4();
            info!(session_id = %session_id, "New WebSocket connection");

            // Split the socket and wrap sender in Arc<Mutex>
            let (sender, mut receiver) = socket.split();
            let ws_sender = Arc::new(TokioMutex::new(sender));

            // --- Per-Connection State ---
            let mut conversation_history = context::ConversationHistory::new(None);
            let mut audio_buffer = Vec::new();
            let mut current_asr_task: Option<tokio::task::JoinHandle<Result<(), anyhow::Error>>> = None;
            let (transcript_tx, mut transcript_rx) = mpsc::channel::<String>(32);

            // Send initial info message
            {
                let mut sender_guard = ws_sender.lock().await;
                if sender_guard.send(Message::Text(serde_json::to_string(&ServerMessage::Ready).unwrap())).await.is_err() {
                    warn!(session_id = %session_id, "Failed to send Ready message");
                    return;
                }
            }

            // Main WebSocket event loop
            loop {
                tokio::select! {
                    biased;

                    msg = receiver.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                info!(message = %text, "Received text message");
                                match serde_json::from_str::<ClientMessageType>(&text) {
                                    Ok(ClientMessageType::TextData { text, .. }) => {
                                        info!(text_to_synthesize = %text, "Processing TTS request");
                                        conversation_history.add_turn(
                                            context::ConversationTurn::new(context::Speaker::User, text.clone())
                                        );

                                        let llm_history = create_llm_history(&conversation_history);
                                        let llm_response = match state.llm_processor.generate_response(&llm_history) {
                                             Ok(response) => {
                                                 info!(response = %response, "LLM generated response");
                                                 response
                                             },
                                             Err(e) => {
                                                 error!(error = %e, "LLM response generation failed");
                                                 format!("Sorry, I couldn't process that. Error: {}", e)
                                             }
                                        };
                                        let model_turn = context::ConversationTurn::new(context::Speaker::Assistant, llm_response.clone());
                                        conversation_history.add_turn(model_turn);

                                        let _contextual_embedding = match state.llm_processor.generate_embeddings(&llm_history) {
                                            Ok(embedding) => {
                                                info!("Generated contextual embedding with dimension {}", embedding.dim());
                                                Some(embedding)
                                            },
                                            Err(e) => {
                                                error!("Contextual embedding generation failed: {}", e);
                                                None
                                            }
                                        };

                                        info!("Starting streaming synthesis...");
                                        let csm_model = state._csm_model.clone();
                                        let (chunk_tx, audio_receiver) = mpsc::channel::<Result<Vec<u8>, ModelError>>(32);

                                        let sender_clone = Arc::clone(&ws_sender);
                                        let _sending_task = tokio::spawn(async move {
                                            info!("Audio chunk sending task started.");
                                            send_audio_chunks(sender_clone, audio_receiver).await;
                                            info!("Audio chunk sending task finished.");
                                        });

                                        let session_id_clone = session_id.clone();
                                        tokio::spawn(async move {
                                            info!("Synthesis task started.");
                                            match csm_model.synthesize_streaming(
                                                &llm_response,
                                                None, // prosody
                                                None, // style_preset
                                                chunk_tx,
                                            ).await {
                                                Ok(_) => info!(session_id = %session_id_clone, "Streaming synthesis completed successfully"),
                                                Err(e) => error!(session_id = %session_id_clone, error = %e, "Streaming synthesis failed"),
                                            }
                                            info!("Synthesis task finished.");
                                        });
                                    },
                                    Ok(ClientMessageType::AudioData { .. }) => {
                                        warn!("Received audio data, but ASR processing is not implemented in this branch yet.");
                                        let mut sender_guard = ws_sender.lock().await;
                                        let _ = send_error_message(&mut *sender_guard, "ASR processing not implemented").await;
                                    },
                                    Ok(ClientMessageType::Stop) => {
                                        info!("Received Stop signal.");
                                        let mut sender_guard = ws_sender.lock().await;
                                        let _ = sender_guard.send(Message::Text(serde_json::to_string(&ServerMessage::SpeechEnded).unwrap())).await;
                                    },
                                    Err(e) => {
                                        warn!(error = %e, "Failed to deserialize client message");
                                        let mut sender_guard = ws_sender.lock().await;
                                        let _ = send_error_message(&mut *sender_guard, &format!("Invalid message format: {}", e)).await;
                                    }
                                }
                            },
                            Some(Ok(Message::Binary(data))) => {
                                info!(bytes = data.len(), "Received binary audio data");

                                if let Some(speech_model) = &state.speech_model {
                                    match state.audio_processor.bytes_to_samples(&data).await {
                                        Ok(samples) => {
                                            audio_buffer.extend_from_slice(&samples);
                                            if audio_buffer.len() >= 32000 || data.is_empty() {
                                                let samples_to_process = audio_buffer.clone();
                                                audio_buffer.clear();
                                                let speech_model_clone = speech_model.clone();
                                                let session_id_clone = session_id.clone();
                                                let transcript_tx_clone = transcript_tx.clone();

                                                if let Some(task) = current_asr_task.take() {
                                                     task.abort();
                                                     info!("Aborted previous ASR task.");
                                                }
                                                current_asr_task = Some(tokio::spawn(async move {
                                                    match speech_model_clone.process_audio(&samples_to_process, 16000).await {
                                                        Ok(result) => {
                                                            let full_text: String = result.words.iter()
                                                                .map(|word| word.text.clone())
                                                                .collect::<Vec<String>>()
                                                                .join(" ");
                                                            
                                                            if !full_text.trim().is_empty() {
                                                                info!("ASR result for {}: {}", session_id_clone, full_text);
                                                                if transcript_tx_clone.send(full_text).await.is_err() {
                                                                    error!("Failed to send transcript to main loop for {}", session_id_clone);
                                                                }
                                                            }
                                                            Ok(())
                                                        },
                                                        Err(e) => {
                                                            error!("ASR processing failed for {}: {}", session_id_clone, e);
                                                            Err(anyhow!("ASR processing failed: {}", e))
                                                        }
                                                    }
                                                }));
                                            }
                                        },
                                        Err(e) => {
                                            error!(error = %e, "Failed to convert bytes to samples");
                                            let mut sender_guard = ws_sender.lock().await;
                                            let _ = send_error_message(&mut *sender_guard, &format!("Audio format error: {}", e)).await;
                                        }
                                    }
                                } else {
                                    warn!("Received audio data but no speech model configured/loaded.");
                                    let mut sender_guard = ws_sender.lock().await;
                                    let _ = send_error_message(&mut *sender_guard, "Speech model not available.").await;
                                }
                            },
                            Some(Ok(Message::Ping(ping))) => {
                                let mut sender_guard = ws_sender.lock().await;
                                if sender_guard.send(Message::Pong(ping)).await.is_err() {
                                    break;
                                }
                            },
                            Some(Ok(Message::Pong(_))) => { /* Ignore */ },
                            Some(Ok(Message::Close(_))) => {
                                info!("Client disconnected gracefully.");
                                break;
                            },
                            Some(Err(e)) => {
                                error!(error = %e, "WebSocket receive error");
                                break;
                            },
                            None => {
                                info!("WebSocket receiver stream ended.");
                                break;
                            }
                        }
                    },

                    Some(transcript) = transcript_rx.recv() => {
                        info!(transcript = %transcript, "Received transcript in main loop");
                        let server_msg = ServerMessage::FullTranscript { transcript: transcript.clone() };
                        let mut sender_guard = ws_sender.lock().await;
                        if sender_guard.send(Message::Text(serde_json::to_string(&server_msg).unwrap())).await.is_err() {
                            error!("Failed to send transcript to client");
                            break;
                        }
                    },

                    else => {
                        break;
                    }
                }
            }

            info!(session_id = %session_id, "WebSocket connection closing");
            if let Some(task) = current_asr_task.take() {
                task.abort();
                info!("Aborted ASR task during cleanup.");
            }
        }
    }
    
    // On upgrade, pass the state into our static handler
    ws.on_upgrade(|socket| async move {
        WebSocketHandler::handle(socket, state.0).await;
    })
}

async fn send_audio_chunks(
    ws_sender: Arc<TokioMutex<impl SinkExt<Message, Error = axum::Error> + Unpin + Send>>,
    mut receiver: tokio::sync::mpsc::Receiver<Result<Vec<u8>, ModelError>>,
) {
    while let Some(chunk_result) = receiver.recv().await {
        match chunk_result {
            Ok(chunk) => {
                if chunk.is_empty() {
                    info!("Received empty chunk (EOS signal from model). Sending EndOfStream message.");
                    let eos_msg = ServerMessage::EndOfStream {};
                    match serde_json::to_string(&eos_msg) {
                        Ok(eos_json) => {
                           let mut sender_guard = ws_sender.lock().await;
                            if let Err(e) = sender_guard.send(Message::Text(eos_json)).await {
                               error!(error = %e, "Failed to send EndOfStream message");
                           }
                        }
                        Err(e) => {
                           error!(error = %e, "Failed to serialize EndOfStream message");
                        }
                    }
                    break;
                }

                trace!(bytes = chunk.len() ,"Sending audio chunk");
                let mut sender_guard = ws_sender.lock().await;
                if let Err(e) = sender_guard.send(Message::Binary(chunk)).await {
                    error!(error = %e, "Failed to send audio chunk");
                    break;
                }
            }
            Err(e) => {
                error!(error = %e, "Error receiving chunk from synthesis stream");
                let mut sender_guard = ws_sender.lock().await;
                let _ = send_error_message(
                    &mut *sender_guard,
                    &format!("Error during synthesis stream: {}", e),
                ).await;
                break;
            }
        }
    }
    info!("send_audio_chunks finished.");
}

async fn send_error_message(
    ws_sender: &mut (impl SinkExt<Message, Error = axum::Error> + Unpin),
    error_message: &str,
) -> Result<(), axum::Error> {
    warn!(message = error_message, "Sending error message to client");
    let error_payload = ServerMessage::Error { message: error_message.to_string() };
    let error_json = serde_json::to_string(&error_payload)
        .unwrap_or_else(|e| format!("{{\"type\":\"Error\",\"message\":\"Serialization failed: {}\"}}", e));
    ws_sender.send(Message::Text(error_json)).await
}

// Placeholder for static file handling (if needed beyond public dir)
// async fn static_handler(uri: Uri) -> impl IntoResponse { ... } 

// Add other handlers/server logic inside #[cfg(feature = "rest_api")] as needed

// --- Config Loading (remains unconditional, but `server` part is unused without feature) ---
// REMOVE DUPLICATED STRUCT AND FUNCTION BELOW
/*
#[derive(Deserialize, Debug, Clone)] // Added Clone
struct Config {
    logging: LoggingConfig,
    model: ModelConfig,
    #[cfg(feature = "rest_api")] // Only include server config if feature is enabled
    server: ServerConfig, 
    audio: AudioConfig,
}

#[cfg(feature = "rest_api")] // Only include server config if feature is enabled
#[derive(Deserialize, Debug, Clone)]
struct ServerConfig {
    enabled: bool,
    port: u16,
    #[serde(default = "default_static_dir")]
    static_dir: String,
}

// ... rest of config structs ...

// Default function for static_dir (remains unconditional)
fn default_static_dir() -> String {
    "static".to_string()
}

fn load_config() -> Result<Config> {
    // ... (loading logic remains the same)
    // The parsing will ignore the `server` field if the feature isn't enabled
    // and the struct definition doesn't include it.
    let config_path = "config.toml";
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow!("Failed to read config file '{}': {}", config_path, e))?;
    
    let config: Config = ::toml::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse config file '{}': {}", config_path, e))?;
    
    Ok(config)
}
*/

// Existing config structs and load_config function (KEEP THESE)
// ... 

// --- Implement FromStr for LlmType ---
impl FromStr for LlmType {
    type Err = anyhow::Error; // Use anyhow::Error for flexibility

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "llama" => Ok(LlmType::Llama),
            "mistral" => Ok(LlmType::Mistral),
            "local" => Ok(LlmType::Local),
            "mock" => Ok(LlmType::Mock),
            _ => Err(anyhow!("Invalid LLM type: {}", s)),
        }
    }
}

// --- Main Application Entry Point ---
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse Command Line Arguments
    let args = Args::parse();

    // 2. Load Configuration
    // Use the path provided in args, falling back to default if needed.
    let config = load_config(&args.config)
        .map_err(|e| anyhow!("Configuration loading failed: {}", e))?;

    // 3. Setup Logging (using tracing_subscriber)
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.logging.level)); // Use level from config

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    info!("Logging initialized with level: {}", config.logging.level);
    info!("Starting CSM application...");
    info!("Arguments: {:?}", args);
    info!("Configuration: {:?}", config);

    // Create model directories if they don't exist
    if !args.model_dir.exists() {
        info!("Creating model directory at {:?}", args.model_dir);
        std::fs::create_dir_all(&args.model_dir)?;
    }
    
    // Create subdirectories for models
    let mimi_dir = args.model_dir.join("mimi");
    let rvq_dir = args.model_dir.join("rvq");
    
    if !mimi_dir.exists() {
        info!("Creating mimi model directory at {:?}", mimi_dir);
        std::fs::create_dir_all(&mimi_dir)?;
    }
    
    if !rvq_dir.exists() {
        info!("Creating rvq model directory at {:?}", rvq_dir);
        std::fs::create_dir_all(&rvq_dir)?;
    }

    // 4. Determine Device (using tch::Device)
    let device_result: Result<Device, anyhow::Error> = match args.device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            if tch::Cuda::is_available() {
                info!("CUDA is available. Using device 0.");
                Ok(Device::Cuda(0))
            } else {
                Err(anyhow!("CUDA device requested but not available."))
            }
        },
        #[cfg(target_os = "macos")]
        "mps" | "metal" => {
            // tch-rs does not directly support Metal, warn and fallback
            warn!("tch backend does not support Metal/MPS, falling back to CPU.");
            Ok(Device::Cpu)
        },
        #[cfg(not(target_os = "macos"))]
        "mps" | "metal" => {
            warn!("MPS/Metal is only available on macOS, falling back to CPU.");
            Ok(Device::Cpu)
        },
        _ => {
            warn!("Unsupported device '{}', falling back to CPU.", args.device);
            Ok(Device::Cpu)
        }
    };

    let device = device_result?; // Now this is a tch::Device
    info!("Using tch device: {:?}", device);

    // Map tch::Device to internal CsmDevice enum
    let csm_device = match device {
        Device::Cpu => CsmDevice::Cpu,
        Device::Cuda(ordinal) => CsmDevice::Cuda(ordinal), // Correctly use ordinal
        // Add other mappings if needed
        _ => {
            // Should not happen with current tch versions unless Vulkan etc. are added
            warn!("Unsupported tch::Device variant encountered, mapping to CsmDevice::Cpu");
            CsmDevice::Cpu
        }
    };
    info!("Internal CsmDevice: {:?}", csm_device);

    // 5. Load CSM Model
    // Use the directory specified in the main config.toml under model._csm_model_dir
    let csm_model_dir_str = config.model._csm_model_dir.clone(); // Use the field from ModelConfig
    info!("Loading CSM model from directory specified in config: {}", csm_model_dir_str);
    
    // Construct the full path by joining the base model directory (from args) with the specific model dir (from config)
    let csm_model_dir = args.model_dir.join(csm_model_dir_str); // THIS LINE IS MODIFIED
    info!("Full path to CSM model directory: {}", csm_model_dir.display()); 

    // Create the CSM model implementation
    // Pass the CORRECTED csm_model_dir to the CSMImpl constructor
    let csm_model = Arc::new(CSMImpl::new(&csm_model_dir, csm_device.clone())?);
    // -------------------
    info!("CSM model loaded successfully.");

    // 6. Load Vocoder (MimiVocoder::new expects sample_rate, tch::Device)
    // Remove the incorrect path argument for now.
    // Weights are likely loaded separately or found by convention.
    // let vocoder_path = args.model_dir.join("mimi/model.safetensors"); 
    info!("Initializing Vocoder...");
    let sample_rate = config.audio.sample_rate; // Get sample rate from config
    let vocoder = Arc::new(Box::new(
        // Pass sample_rate and the tch::Device
        MimiVocoder::new(sample_rate, device)
            .map_err(|e| anyhow!("Failed to initialize Vocoder: {}", e))?
    ) as Box<dyn Vocoder + Send + Sync>);
    // Add logging after successful initialization
    info!("Vocoder initialized successfully for sample rate {} Hz.", sample_rate);

    // 7. Load Moshi Speech Model (Optional, for ASR)
    // Create config specifically for Moshi initialization
    let moshi_config = CsmModelConfig {
        model_type: Some("moshi".to_string()), // Identify model type
        moshi_model_paths: Some(MoshiModelPaths { // Use paths from main config
            model_dir: config.model._moshi_model_path.clone(),
            tokenizer_path: config.model._tokenizer_path.clone(),
            asr_lm_path: config.model._asr_lm_path.clone(),
        }),
        sample_rate: Some(config.audio.sample_rate), // Pass sample rate
        device_type: Some(args.device.clone()), // Pass device string
        ..Default::default() // Other fields are not directly needed for Moshi init
    };

    let speech_model: Option<Arc<MoshiSpeechModel>> = match MoshiSpeechModel::new(&moshi_config, csm_device.clone()) {
        Ok(_) => {
            info!("Moshi Speech Model loaded successfully (placeholder).");
            None
        },
        Err(e) => {
            warn!("Failed to load Moshi speech model: {}. ASR features will be disabled.", e);
            None
        }
    };

    // 8. Initialize LLM Processor
    info!("Initializing LLM Processor...");
    let use_gpu = device != Device::Cpu;
    let llm_config = LlmConfig {
        llm_type: config.model.llm_type.parse::<LlmType>()?,
        model_path: Some(PathBuf::from(&config.model.llm_model_path)),
        use_gpu,
        embedding_dim: 768,
        max_context_window: 4096,
        _model_id: None,
        _api_key: None,
        _temperature: 0.7,
        _parameters: std::collections::HashMap::new(),
    };
    let llm_processor = create_optimized_llm_service(llm_config)?;
    info!("LLM Processor initialized.");

    // 8.5 Initialize Audio Processor
    info!("Initializing Audio Processor...");
    // Create MimiEncoderConfig and RVQConfig from the main Config struct
    let mimi_encoder_config = csm::audio::codec::MimiEncoderConfig {
        sample_rate: config.audio.mimi_encoder.sample_rate,
        // Populate actual fields based on MimiEncoderConfig definition
        input_channels: 1, // Example, adjust from config if needed
        dimension: 512, // Example, adjust from config if needed
        hidden_size: 512, // Example, adjust from config if needed
        hidden_channels: 512, // Example, adjust from config if needed
        causal: true, // Example, adjust from config if needed
        kernel_size: 7, // Example, adjust from config if needed
        compress: 2, // Example, adjust from config if needed
        num_hidden_layers: 8, // Example, adjust from config if needed
        num_attention_heads: 8, // Example, adjust from config if needed
        head_dim: 64, // Example, adjust from config if needed
        intermediate_size: 2048, // Example, adjust from config if needed
        norm_eps: 1e-5, // Example, adjust from config if needed
        rope_theta: 10000.0, // Example, adjust from config if needed
        frame_length_ms: 80.0, // Example, adjust from config if needed
        hop_length_ms: 80.0, // Example, adjust from config if needed
        activation: "GELU".to_string(), // Added missing field (example)
    };
    let rvq_config = csm::rvq::RVQConfig {
        num_codebooks: config.audio.rvq.num_codebooks,
        // Populate actual fields based on RVQConfig definition
        vector_dim: 512, // Example, ensure matches Mimi output
        codebook_size: 2048, // Example, adjust from config if needed
        normalize: true, // Added missing field (example)
        learning_rate: 0.01, // Added missing field (example)
        device: device, // Added missing field, use the determined tch::Device
        // REMOVED non-existent fields: kmeans_iters, threshold_ema_dead_code
    };
    // Construct paths (using defaults or config)
    let mimi_weights_path = args.model_dir.join("mimi/model.safetensors");
    let rvq_weights_path = args.model_dir.join("rvq/model.safetensors"); // Adjust path if needed

    // Check if Mimi weights exist
    let mimi_weights_option = if mimi_weights_path.exists() {
        info!("Found Mimi weights file at {:?}", mimi_weights_path);
        Some(mimi_weights_path.as_path())
    } else {
        warn!("Mimi weights file not found at {:?}", mimi_weights_path);
        None
    };
    
    // Check if RVQ weights exist
    let rvq_weights_option = if rvq_weights_path.exists() {
        info!("Found RVQ weights file at {:?}", rvq_weights_path);
        Some(rvq_weights_path.as_path())
    } else {
        warn!("RVQ weights file not found at {:?}", rvq_weights_path);
        None
    };
    
    let audio_processor_inner = csm::audio::AudioProcessor::new(
        mimi_encoder_config,
        rvq_config,
        mimi_weights_option,
        rvq_weights_option, // Pass None if weights don't exist
        device, // Use the determined tch::Device
    ).map_err(|e| anyhow!("Failed to create AudioProcessor: {}", e))?;

    // Wrap it in our thread-safe wrapper
    let audio_processor = Arc::new(ThreadSafeAudioProcessor::new(audio_processor_inner));
    info!("Audio Processor initialized.");
    
    // 9. Run Server (Now always compiled)
    if config.server.enabled {
        info!("Starting REST API server...");
        // Pass the initialized audio_processor Arc
        run_server(config, csm_model, vocoder, llm_processor, speech_model, audio_processor).await
            .map_err(|e| anyhow!("Server failed to run: {}", e))?;
    } else {
        info!("REST API server is disabled in the configuration.");
    }

    info!("Application finished.");
    Ok(())
} 
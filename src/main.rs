use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn, Level, error, trace};
use anyhow::{Result, anyhow};
use clap::Parser;
use csm::models::{CSMModel, CSMImpl};
use tch::Device;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use std::net::SocketAddr;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

// Add vocoder imports
// Make vocoder module public at crate root
pub mod vocoder;
use crate::vocoder::{Vocoder, MimiVocoder};

// Axum imports
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse},
    routing::{get, get_service},
    Router,
};
use tower_http::{
    services::ServeDir,
    trace::{DefaultMakeSpan, TraceLayer},
};
use futures_util::{
    sink::SinkExt,
    stream::StreamExt,
};

// CLI Arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

    #[arg(short = 'd', long, default_value = "cpu")]
    device: String,

    #[arg(long, default_value_t = 3000)]
    port: u16,
}

// Application State
#[derive(Clone)]
struct AppState {
    csm_model: Arc<CSMImpl>,
    vocoder: Arc<Box<dyn Vocoder + Send + Sync>>,
}

// Client -> Server Message Structure
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ClientMessageType {
    #[serde(rename = "synthesize")]
    Synthesize {
        text: String,
        emotion: Option<String>,
        style: Option<String>,
    },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "stop_audio")]
    StopAudio,
}

// Server -> Client Message Structure
#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "audio_info")]
    AudioInfo { sample_rate: u32, channels: u16 },
    #[serde(rename = "error")]
    Error(String),
    #[serde(rename = "pong")]
    Pong,
}

// WebSocket Handler
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let session_id = Uuid::new_v4();
    info!(session_id = %session_id, "WebSocket client connected");
    let (sender, mut receiver) = socket.split();

    let sender = Arc::new(TokioMutex::new(sender));

    // CORRECTED: Channel for Tokens (Synthesis Task -> Vocoder Task)
    let (token_tx, mut token_rx) = mpsc::channel::<Vec<(i64, Vec<i64>)>>(100);
    // Channel for Audio Chunks (Vocoder Task -> Send Task)
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<i16>>(100);

    // --- Task to send AUDIO messages to the client ---
    let sender_clone = sender.clone();
    let vocoder_sample_rate = state.vocoder.sample_rate();
    let send_task_handle = tokio::spawn(async move {
        let mut audio_info_sent = false;
        info!(session_id = %session_id, "Send task STARTED.");
        loop {
            tokio::select! {
                Some(audio_chunk_i16) = audio_rx.recv() => {
                    trace!(session_id = %session_id, "Send task received audio chunk ({} samples).", audio_chunk_i16.len());
                    let mut sender_guard = sender_clone.lock().await;

                    // Send AudioInfo JSON first
                    if !audio_info_sent {
                        let info_msg = ServerMessage::AudioInfo { sample_rate: vocoder_sample_rate, channels: 1 };
                        match serde_json::to_string(&info_msg) {
                            Ok(json_msg) => {
                                if let Err(e) = sender_guard.send(Message::Text(json_msg)).await {
                                    error!(session_id = %session_id, "Send task: Failed to send audio info: {}. Closing.", e);
                                    break;
                                }
                                audio_info_sent = true;
                                info!(session_id = %session_id, "Sent audioInfo to client (Rate: {}).", vocoder_sample_rate);
                            },
                            Err(e) => {
                                error!(session_id = %session_id, "Failed to serialize audio info: {}", e);
                            }
                        }
                    }
                    
                    // Convert Vec<i16> to Vec<u8> for binary message
                    let mut audio_bytes = Vec::with_capacity(audio_chunk_i16.len() * 2);
                    for sample_i16 in audio_chunk_i16 {
                        audio_bytes.extend_from_slice(&sample_i16.to_le_bytes());
                    }
                    
                    // Send binary audio data
                    if let Err(e) = sender_guard.send(Message::Binary(audio_bytes)).await {
                        error!(session_id = %session_id, "Send task: Failed to send audio data chunk: {}. Closing.", e);
                        break;
                    }
                }
                else => {
                    info!(session_id = %session_id, "Send task: Audio chunk channel closed.");
                    // Optionally send completion message
                    let mut sender_guard = sender_clone.lock().await;
                    let completion_msg = serde_json::json!({ "type": "completed" }).to_string();
                    let _ = sender_guard.send(Message::Text(completion_msg)).await;
                    break;
                }
            }
        }
        info!(session_id = %session_id, "Send task FINISHED.");
    });
    
    // --- Task to run Vocoder ---
    let vocoder_clone = state.vocoder.clone();
    let audio_tx_clone = audio_tx.clone();
    let vocoder_task_handle = tokio::spawn(async move {
        info!(session_id = %session_id, "Vocoder task STARTED.");
        while let Some(token_chunk_vec) = token_rx.recv().await {
             trace!(session_id = %session_id, "Vocoder task received token chunk vec (len {}).", token_chunk_vec.len());
             if let Some(token_tuple) = token_chunk_vec.into_iter().next() {
                match vocoder_clone.synthesize_chunk(token_tuple).await {
                    Ok(audio_chunk) => {
                        if !audio_chunk.is_empty() {
                            if audio_tx_clone.send(audio_chunk).await.is_err() {
                                warn!(session_id = %session_id, "Vocoder task: Failed to send audio chunk to sender task.");
                                break;
                            }
                        }
                    },
                    Err(e) => {
                        error!(session_id = %session_id, "Vocoder failed to synthesize chunk: {}", e);
                    }
                }
             } else {
                 warn!(session_id = %session_id, "Received empty token chunk vector from synthesis task.");
             }
        }
        info!(session_id = %session_id, "Vocoder task FINISHED (token channel closed).");
    });

    // --- Task to receive messages from the client and run Synthesis ---
    let csm_model_clone = state.csm_model.clone();
    let receive_task_handle = tokio::spawn(async move {
        info!(session_id = %session_id, "Receive task STARTED.");
        let token_tx_clone = token_tx.clone();
        while let Some(msg_result) = receiver.next().await {
            match msg_result {
                Ok(Message::Text(text)) => {
                    info!(session_id = %session_id, "Received text message.");
                    match serde_json::from_str::<ClientMessageType>(&text) {
                        Ok(ClientMessageType::Synthesize { text, emotion, style }) => {
                            // --- Construct final text with control tokens ---
                            let mut final_text = String::new();

                            // Prepend style token if valid
                            if let Some(style) = style {
                                match style.to_lowercase().as_str() {
                                    "normal" => final_text.push_str("[STYLE:normal] "),
                                    "whispering" => final_text.push_str("[STYLE:whispering] "),
                                    _ => warn!(session_id = %session_id, "Ignoring invalid style: {}", style),
                                }
                            }

                            // Prepend emotion token if valid
                            if let Some(emotion) = emotion {
                                match emotion.to_lowercase().as_str() {
                                    "neutral" => final_text.push_str("[EMOTION:neutral] "),
                                    "happy" => final_text.push_str("[EMOTION:happy] "),
                                    "sad" => final_text.push_str("[EMOTION:sad] "),
                                    _ => warn!(session_id = %session_id, "Ignoring invalid emotion: {}", emotion),
                                }
                            }

                            final_text.push_str(&text);

                            info!(session_id = %session_id, final_text = final_text, "Starting synthesis task with prepended control tokens.");
                            let model_for_task = csm_model_clone.clone();
                            let tx_for_synthesis_task = token_tx_clone.clone();

                            tokio::spawn(async move {
                                let result = model_for_task.synthesize_streaming(
                                    &final_text,
                                    None,
                                    None,
                                    None,
                                    tx_for_synthesis_task,
                                ).await;

                                if let Err(e) = result {
                                    error!(session_id = %session_id, final_text = final_text, "Synthesis task failed: {}", e);
                                } else {
                                    info!(session_id = %session_id, final_text = final_text, "Synthesis task completed.");
                                }
                            });
                        },
                        Ok(ClientMessageType::Ping) => {
                            let mut sender_guard = sender.lock().await;
                            let pong_msg = ServerMessage::Pong;
                            if let Ok(json_msg) = serde_json::to_string(&pong_msg) {
                                if let Err(e) = sender_guard.send(Message::Text(json_msg)).await {
                                    error!(session_id = %session_id, "Receive task: Failed to send pong response: {}. Closing.", e);
                                    break;
                                }
                            }
                        },
                        Ok(ClientMessageType::StopAudio) => {
                            info!(session_id = %session_id, "Receive task: Received stop_audio signal from client.");
                            // Explicitly drop the sender to signal completion to vocoder/sender tasks
                            drop(token_tx_clone);
                            // Break the loop after stopping
                            break;
                        },
                        Err(e) => {
                             warn!(session_id = %session_id, raw_text = text, "Failed to parse client message: {}", e);
                             let mut sender_guard = sender.lock().await;
                             let error_resp = ServerMessage::Error(format!("Invalid message format: {}", e));
                             if let Ok(err_json) = serde_json::to_string(&error_resp) {
                                 let _ = sender_guard.send(Message::Text(err_json)).await.map_err(|e| {
                                     error!(session_id = %session_id, "Receive task: Failed to send error response: {}. Closing.", e);
                                 });
                             }
                        }
                    }
                }
                Ok(Message::Binary(data)) => {
                    info!(session_id = %session_id, "Received binary message (audio chunk) of size: {} bytes.", data.len());
                }
                Ok(Message::Ping(ping)) => {
                    let mut sender_guard = sender.lock().await;
                    if let Err(e) = sender_guard.send(Message::Pong(ping)).await {
                        error!(session_id = %session_id, "Receive task: Failed to send pong (from ping): {}. Closing.", e);
                        break;
                    }
                }
                Ok(Message::Pong(_)) => { /* Keepalive */ }
                Ok(Message::Close(_)) => {
                    info!(session_id = %session_id, "Client requested close.");
                    break;
                }
                Err(e) => {
                    warn!(session_id = %session_id, "WebSocket receive error: {}", e);
                    break;
                }
            }
        }
        info!(session_id = %session_id, "Receive task FINISHED.");
        drop(token_tx);
    });

    // Wait for tasks to complete and log results
    tokio::select! {
        res = send_task_handle => { 
            info!(session_id = %session_id, "Send task exited.");
            if let Err(e) = res { error!(session_id = %session_id, "Send task JoinError: {}", e); }
        }
        res = vocoder_task_handle => { 
            info!(session_id = %session_id, "Vocoder task exited."); 
            if let Err(e) = res { error!(session_id = %session_id, "Vocoder task JoinError: {}", e); }
        }
        res = receive_task_handle => { 
            info!(session_id = %session_id, "Receive task exited."); 
            if let Err(e) = res { error!(session_id = %session_id, "Receive task JoinError: {}", e); }
        }
    }

    info!(session_id = %session_id, "WebSocket client disconnected and handle_socket finished.");
}

// WebSocket Upgrade Handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("Upgrading connection to WebSocket...");
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

// Handler to serve the main HTML page
async fn root_handler() -> impl IntoResponse {
    match tokio::fs::read_to_string("web/index.html").await {
        Ok(content) => Html(content),
        Err(e) => {
            error!("Failed to read index.html: {}", e);
            Html("<h1>Error</h1><p>Could not load the application interface.</p>".to_string())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false) // Don't include event targets
        .compact() // Use compact format
        .init();

    let args = Args::parse();

    // Determine device (ensure tch feature flags are handled in lib.rs or models mod)
    let device = match args.device.as_str() {
        "cuda" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0) // Default to device 0
            } else {
                warn!("CUDA specified but not available, falling back to CPU.");
                Device::Cpu
            }
        }
        "mps" => {
             if tch::utils::has_mps() { // Assuming this check exists and works for your tch version
                 Device::Mps
             } else {
                warn!("MPS specified but not available or tch not built with MPS support, falling back to CPU.");
                Device::Cpu
            }
        }
        "cpu" | _ => Device::Cpu,
    };
    info!("Using device: {:?}", device);

    // Load the CSM model using the wrapper
    // Point to the subdirectory containing the model files
    let csm_model_path = args.model_dir.join("csm-1b"); 
    info!("Loading CSM model from directory: {:?}", csm_model_path);
    let csm_impl = CSMImpl::new(&csm_model_path, device)
        .map_err(|e| anyhow!("Failed to initialize CSMImpl: {}", e))?;

    // --- Create Vocoder Instance --- 
    let vocoder_sample_rate = 24000;
    let mimi_path = args.model_dir.join("mimi/model.safetensors");
    info!("Initializing MimiVocoder with model: {:?}", mimi_path);

    let mut vocoder = MimiVocoder::new(
        vocoder_sample_rate,
        device,
    )?;
    vocoder.load_model(mimi_path)?;
    
    let vocoder: Arc<Box<dyn Vocoder + Send + Sync>> = Arc::new(Box::new(vocoder));
    info!("Using MimiVocoder with sample rate: {}", vocoder.sample_rate());
    // --------------------------------

    let app_state = Arc::new(AppState {
        csm_model: Arc::new(csm_impl),
        vocoder: vocoder.clone(),
    });

    // Build the Axum router
    let app = Router::new()
        // Route for the WebSocket connection
        .route("/ws", get(ws_handler))
        // Route to serve the main index.html file
        .route("/", get(root_handler))
        // Service to serve static files (JS, CSS) from the 'web' directory
        // Note: Updated ServeDir path to be relative to workspace root
        .nest_service("/static", get_service(ServeDir::new("web")))
        // Add state
        .with_state(app_state)
        // Add tracing layer
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::default().include_headers(true)),
        );

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port)); // Listen on all interfaces
    info!("Server listening on http://{}", addr);

    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| anyhow!("Failed to bind TCP listener: {}", e))?;

    // Run the server
    axum::serve(listener, app.into_make_service())
        .await
        .map_err(|e| anyhow!("Server failed: {}", e))?;

    Ok(())
}

// Declare modules
pub mod models; 
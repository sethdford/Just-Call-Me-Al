use tokio::sync::mpsc;
use warp::ws::{Message, WebSocket};
use futures::{FutureExt, StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error};

use crate::models::{CSMModel, CSMImpl};

// Simplified request for streaming text synthesis
#[derive(Debug, Serialize, Deserialize)]
struct StreamRequest {
    text: String,
    // Add optional temp/top_k/seed later if needed
}

// Response containing token chunks
#[derive(Debug, Serialize)]
struct TokenResponse {
    r#type: String, // e.g., "tokens"
    tokens: Vec<Vec<i64>>, // A chunk of timesteps, each with N tokens
}

// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    r#type: String, // e.g., "error"
    message: String,
}

pub async fn handle_websocket(ws: WebSocket, csm: Arc<CSMImpl>) {
    info!("New WebSocket connection established.");
    let (mut ws_sender, mut ws_receiver) = ws.split();

    while let Some(result) = ws_receiver.next().await {
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => {
                error!("WebSocket receive error: {}", e);
                break;
            }
        };

        info!("Received message: {:?}", msg);

        if msg.is_text() {
        if let Ok(text) = msg.to_str() {
                let request: StreamRequest = match serde_json::from_str(text) {
                Ok(req) => req,
                    Err(e) => {
                        error!("Failed to parse StreamRequest: {}", e);
                        send_ws_error(&mut ws_sender, format!("Invalid request format: {}", e)).await;
                        continue;
                    }
                };

                info!("Received synthesis request for text: \"{}\"", request.text);

                // --- Create channel for token streaming --- 
                let (token_tx, mut token_rx) = mpsc::channel::<Vec<Vec<i64>>>(100); 

                let csm_clone = csm.clone(); // Clone Arc for the synthesis task
                let text_clone = request.text.clone();

                // --- Task to run synthesis --- 
                tokio::spawn(async move {
                    info!("Spawning synthesis task...");
                    match csm_clone.synthesize_streaming(&text_clone, None, None, None, token_tx).await {
                        Ok(_) => info!("Synthesize streaming finished successfully."),
                        Err(e) => error!("Synthesize streaming failed: {}", e),
                    }
                    // Sender (`token_tx`) is dropped here, closing the channel
                });

                // --- Task to forward tokens to WebSocket --- 
                let mut ws_sender_clone = ws_sender.clone();
                tokio::spawn(async move {
                    info!("Spawning WebSocket sender task for tokens...");
                    while let Some(token_chunk) = token_rx.recv().await {
                        let response = TokenResponse {
                            r#type: "tokens".to_string(),
                            tokens: token_chunk,
                                };
                        match serde_json::to_string(&response) {
                            Ok(json) => {
                                if let Err(e) = ws_sender_clone.send(Message::text(json)).await {
                                    error!("WebSocket send error: {}", e);
                                    break; // Stop if sending fails
                                }
                            }
                            Err(e) => {
                                error!("Failed to serialize TokenResponse: {}", e);
                                // Optionally send an error back to the client
                            }
                        }
                    }
                    info!("Token receiver channel closed, WebSocket sender task finished.");
                    // Optionally send a completion message
                    // let completion_msg = ...;
                    // let _ = ws_sender_clone.send(Message::text(completion_msg)).await;
                });

            } else {
                 warn!("Received non-text message, ignoring.");
            }
        } else if msg.is_close() {
            info!("Received close message.");
            break;
        } else {
            warn!("Received unexpected message type: {:?}", msg);
                    }
                }

    info!("WebSocket connection closed.");
        }

// Helper to send structured error messages
async fn send_ws_error<S>(sink: &mut S, message: String)
where
    S: SinkExt<Message, Error = warp::Error> + Unpin,
{
    let error_response = ErrorResponse {
        r#type: "error".to_string(),
        message,
    };
    match serde_json::to_string(&error_response) {
        Ok(json) => {
            if let Err(e) = sink.send(Message::text(json)).await {
                error!("Failed to send error response over WebSocket: {}", e);
    }
}
        Err(e) => {
            error!("Failed to serialize ErrorResponse: {}", e);
            // Fallback to plain text error
            let fallback_msg = format!("{{\"type\":\"error\",\"message\":\"Internal serialization error\"}}");
            if let Err(e) = sink.send(Message::text(fallback_msg)).await {
                 error!("Failed to send fallback error response over WebSocket: {}", e);
            }
        }
    }
} 
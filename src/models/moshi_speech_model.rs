use moshi::lm::LmModel;
use moshi::mimi::Mimi;
use moshi::lm::Config as LmConfig;
use moshi::mimi::Config as MimiConfig;
use candle_core::{Tensor, Device, Error as CandleError};
use candle_nn::VarBuilder;
use moshi::nn::MaybeQuantizedVarBuilder;
use sentencepiece::SentencePieceProcessor;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use thiserror::Error;
use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use moshi::transformer::CaSrc;

// --- Error Enum (copied from stt/mod.rs, maybe move to a common place later) ---
#[derive(Error, Debug)]
pub enum SpeechModelError {
    #[error("Failed to initialize model: {0}")]
    Init(#[from] anyhow::Error),
    #[error("Initialization error (streaming): {0}")]
    InitializationError(moshi::candle::Error),
    #[error("Error processing audio: {0}")]
    Processing(String),
    #[error("Device error: {0}")]
    Device(CandleError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
}

// --- Output Structs (copied from stt/mod.rs) ---
#[derive(Debug, Clone)]
pub struct RecognizedWord {
    pub text: String,
    pub start_time: f64,
    pub stop_time: f64,
}

#[derive(Debug, Clone)]
pub struct STTOutput {
    pub word: RecognizedWord,
    pub audio_codes: Option<Tensor>,
}

// --- The main Model Struct ---
#[derive(Clone)]
pub struct MoshiSpeechModel {
    mimi: Arc<Mutex<Mimi>>,
    lm: Arc<Mutex<LmModel>>,
    tokenizer: Arc<SentencePieceProcessor>,
    device: Device,
    mimi_config: MimiConfig,
    lm_config: LmConfig,
    asr_delay_in_tokens: usize,
    acoustic_delay: usize,
}

impl MoshiSpeechModel {
    pub fn new(
        model_path: &str,         // LM safetensors
        tokenizer_path: &str,    // SentencePiece model
        mimi_path: &str,         // Mimi safetensors (encoder + decoder)
        mimi_config: MimiConfig, // Mimi config
        lm_config: LmConfig,     // LM config
        asr_delay_in_tokens: usize,
        acoustic_delay: usize,
        device: Device,
    ) -> Result<Self, SpeechModelError> { // Use new Error type
        // --- Loading Logic (similar to MoshiSTT::new) ---
        let tokenizer = Arc::new(SentencePieceProcessor::open(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?);

        let mimi_vb = unsafe {
             VarBuilder::from_mmaped_safetensors(&[mimi_path], candle_core::DType::F32, &device)
                 .map_err(|e| anyhow::anyhow!("Failed to load Mimi model: {}", e))?
        };
        let mimi_instance = Mimi::new(mimi_config.clone(), mimi_vb)
            .map_err(|e| anyhow::anyhow!("Failed to initialize Mimi: {}", e))?;
        let mimi = Arc::new(Mutex::new(mimi_instance));

        let lm_raw_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)
                .map_err(|e| anyhow::anyhow!("Failed to load LM model: {}", e))?
        };
        let lm_vb = MaybeQuantizedVarBuilder::Real(lm_raw_vb);
        let lm_instance = LmModel::new(&lm_config, lm_vb)
            .map_err(|e| anyhow::anyhow!("Failed to initialize LM: {}", e))?;
        let lm = Arc::new(Mutex::new(lm_instance));

        Ok(Self {
            mimi,
            lm,
            tokenizer,
            device,
            mimi_config,
            lm_config,
            asr_delay_in_tokens,
            acoustic_delay,
        })
    }

    // --- STT Methods --- 
    
    /// Process PCM audio data and return recognized words and optionally the audio codes for the chunk.
    pub async fn process_audio(&self, pcm_data: &[f32], sample_rate: u32) -> Result<(Vec<STTOutput>, Option<Tensor>), SpeechModelError> { // Use new Error type
        if sample_rate != 24000 {
            return Err(SpeechModelError::Processing(format!(
                "Expected sample rate of 24000, got {}", sample_rate
            )));
        }

        let pcm_tensor = Tensor::from_slice(pcm_data, (1, pcm_data.len()), &self.device)?;

        // Lock models
        let mut mimi_guard = self.mimi.lock().await; // Mut needed for encode
        let lm_guard = self.lm.lock().await;

        // Get audio codes for the entire chunk first
        let codes_tensor = match mimi_guard.encode(&pcm_tensor) { // Call encode on Mimi
            Ok(codes) => Some(codes),
            Err(e) => {
                eprintln!("Failed to encode audio chunk to get codes: {}", e);
                None
            }
        };

        // Create ASR state for this processing task
        // Note: We clone the models again here for State::new
        let mut asr_state = moshi::asr::State::new(self.asr_delay_in_tokens, mimi_guard.clone(), lm_guard.clone())?;

        // Process the entire audio tensor in one step with ASR state
        let words: Vec<moshi::asr::Word> = match asr_state.step_pcm(pcm_tensor, |_, _| Ok(())) {
             Ok(w) => w,
             Err(e) => return Err(SpeechModelError::Processing(format!("ASR step_pcm failed: {}", e))),
        };

        // Convert the Moshi Words to STTOutput
        let stt_outputs = words.into_iter()
            .filter_map(|w| {
                if w.tokens.is_empty() { return None; }
                match self.tokenizer.decode_piece_ids(&w.tokens) { // Use decode_piece_ids
                    Ok(text) => Some(STTOutput {
                        word: RecognizedWord {
                            text,
                            start_time: w.start_time,
                            stop_time: w.stop_time,
                        },
                        audio_codes: None, // Codes are returned chunk-level, not per-word
                    }),
                    Err(e) => {
                        eprintln!("Failed to decode word tokens: {}", e);
                        None
                    }
                }
            })
            .collect();

        Ok((stt_outputs, codes_tensor)) // Return both words and chunk codes
    }

    /// Start streaming audio processing
    pub fn start_streaming(&self, sample_rate: u32) -> Result<(mpsc::Sender<Vec<f32>>, mpsc::Receiver<Result<Vec<STTOutput>, SpeechModelError>>), SpeechModelError> { // Updated Receiver type in signature
        if sample_rate != 24000 {
            return Err(SpeechModelError::Processing(format!(
                "Expected sample rate of 24000, got {}", sample_rate
            )));
        }

        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(100);
        // Updated channel type to send Result
        let (text_tx, text_rx) = mpsc::channel::<Result<Vec<STTOutput>, SpeechModelError>>(100);

        // Clone necessary parts from self
        let mimi_clone = self.mimi.clone();
        let lm_clone = self.lm.clone();
        let tokenizer_clone = self.tokenizer.clone();
        let asr_delay = self.asr_delay_in_tokens;
        let device_clone = self.device.clone();

        tokio::spawn(async move {
            // Lock models needed for ASR state creation
            let mimi_guard = mimi_clone.lock().await;
            let lm_guard = lm_clone.lock().await;
            // Manually map the error here
            let mut asr_state = match moshi::asr::State::new(asr_delay, mimi_guard.clone(), lm_guard.clone()) {
                Ok(state) => state,
                Err(e) => {
                    eprintln!("Failed to create ASR state for streaming: {}", e);
                    // Send the mapped error over the channel
                    let _ = text_tx.send(Err(SpeechModelError::InitializationError(e))).await;
                    drop(text_tx);
                    return;
                }
            };
            drop(mimi_guard);
            drop(lm_guard);

            while let Some(pcm_data) = audio_rx.recv().await {
                if pcm_data.is_empty() { continue; }
                match Tensor::from_slice(&pcm_data, (1, pcm_data.len()), &device_clone) {
                    Ok(pcm_tensor) => {
                        // Process the current chunk
                        match asr_state.step_pcm(pcm_tensor, |_, _| Ok(())) {
                            Ok(words) => {
                                if !words.is_empty() {
                                    let stt_outputs: Vec<STTOutput> = words.into_iter()
                                        .filter_map(|w| {
                                            if w.tokens.is_empty() { return None; }
                                            match tokenizer_clone.decode_piece_ids(&w.tokens) {
                                                Ok(text) => Some(STTOutput {
                                                    word: RecognizedWord {
                                                        text,
                                                        start_time: w.start_time,
                                                        stop_time: w.stop_time,
                                                    },
                                                    audio_codes: None, // Cannot easily associate codes here
                                                }),
                                                Err(e) => {
                                                    eprintln!("Failed to decode word tokens: {}", e);
                                                    None
                                                }
                                            }
                                        })
                                        .collect();
                                    // Send Ok(outputs) on success
                                    if text_tx.send(Ok(stt_outputs)).await.is_err() {
                                        eprintln!("STT streaming receiver dropped.");
                                        break; // Exit loop if receiver is gone
                                    }
                                }
                                // Send Ok(empty_vec) even if no words were finalized in this step?
                                // else {
                                //     if text_tx.send(Ok(vec![])).await.is_err() {
                                //         eprintln!("STT streaming receiver dropped.");
                                //         break;
                                //     }
                                // }
                            }
                            Err(e) => {
                                eprintln!("Error processing audio chunk in streaming: {}", e);
                                // Send Err on processing failure
                                if text_tx.send(Err(SpeechModelError::Processing(format!("ASR step_pcm failed: {}", e)))).await.is_err() {
                                    eprintln!("STT streaming receiver dropped.");
                                    break; // Exit loop if receiver is gone
                                }
                                // Optionally break or continue after error?
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error creating tensor from PCM data in streaming: {}", e);
                        // Send Err on tensor creation failure
                        if text_tx.send(Err(SpeechModelError::Tensor(e))).await.is_err() {
                            eprintln!("STT streaming receiver dropped.");
                            break; // Exit loop if receiver is gone
                        }
                        // Optionally break or continue after error?
                    }
                }
            }
            // Audio channel closed, task ends.
        });

        Ok((audio_tx, text_rx))
    }

    /// Reset the STT state (currently a no-op for the combined model)
    pub async fn reset(&self) -> Result<(), SpeechModelError> {
        // Resetting for the combined model is currently a no-op.
        // The state (ASR state) is created within the processing methods (process_audio, start_streaming)
        // For a persistent, resettable state across calls, architecture changes would be needed.
        Ok(())
    }
    
    // --- Generation Methods --- 
    
    /// Get null inputs for unconditional generation
    pub async fn get_unconditional_inputs(&self, num_samples: usize) -> Result<(Tensor, Tensor, Tensor), SpeechModelError> { // Use new Error type
        let _lm_guard = self.lm.lock().await; // Prefix with underscore if guard not used directly
        let _mimi_guard = self.mimi.lock().await; // Prefix with underscore if guard not used directly

        // Get text padding token ID from the tokenizer
        let text_pad_token = self.tokenizer.pad_id().unwrap_or(0); // Use pad_id(), default to 0 if None
        let text_input_ids = Tensor::full(text_pad_token, (num_samples, 0), &self.device)?;
        
        // Use audio_vocab_size from lm_config as the null audio token
        let audio_null_token = self.lm_config.audio_vocab_size; // Use lm_config field
        let user_audio_codes = Tensor::full(audio_null_token as u32, (num_samples, 0, 1), &self.device)?;
        let moshi_audio_codes = Tensor::full(audio_null_token as u32, (num_samples, 0, 1), &self.device)?;

        Ok((text_input_ids, user_audio_codes, moshi_audio_codes))
    }

    /// Synthesize audio from text using the Moshi model.
    pub async fn synthesize_audio(
        &self, 
        text: &str,
        temperature: Option<f64>,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Vec<f32>, SpeechModelError> { 
        // 1. Text Tokenization
        let text_tokens = self.tokenizer.encode(text)
            .map_err(|e| SpeechModelError::Processing(format!("Tokenizer encode failed: {}", e)))?
            .into_iter()
            .map(|p| p.id)
            .collect::<Vec<_>>();
        
        let text_tokens_len = text_tokens.len();
        if text_tokens_len == 0 {
            return Err(SpeechModelError::Processing("Input text is empty after tokenization".to_string()));
        }
        // [1, TextSeqLen]
        let text_tensor = Tensor::from_vec(text_tokens.clone(), (1, text_tokens_len), &self.device)?;

        // 2. Acquire Locks
        let mut lm_guard = self.lm.lock().await;
        let mut mimi_guard = self.mimi.lock().await;

        // 3. Conditioning: Prepare CaSrc using Tokens
        // LmModel::forward_ca will handle embedding the tokens internally.
        let ca_src = CaSrc::Tokens(text_tensor.clone()); // Clone text_tensor for CaSrc

        // 4. Generation Loop Parameters & Setup
        let max_duration_s = 60.0; // Limit generation duration
        let max_steps = (max_duration_s * self.mimi_config.frame_rate) as usize;
        let audio_codebooks = lm_guard.generated_audio_codebooks();
        let audio_pad_token = lm_guard.audio_pad_token();
        let acoustic_delay = self.acoustic_delay;

        let sampling = match (top_k, temperature) {
            (Some(k), Some(t)) => Sampling::TopK { k, temperature: t },
            (Some(k), None)    => Sampling::TopK { k, temperature: 0.8 },
            (None, Some(t))    => Sampling::TopK { k: 100, temperature: t },
            (None, None)       => Sampling::TopK { k: 100, temperature: 0.8 },
        };
        let seed = seed.unwrap_or(299792458);
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        let mut generated_audio_tokens: Vec<Vec<u32>> = vec![vec![u32::MAX; audio_codebooks]; max_steps + acoustic_delay];
        let mut end_of_gen: Option<usize> = None;
        let audio_vocab_size = lm_guard.audio_pad_token() + 1;
        let quantizer_bins = audio_vocab_size - 2;

        lm_guard.reset_state(); 
        mimi_guard.reset_state();
        
        let forced_audio_tokens_helper = moshi::lm::ForcedAudioTokens::new(
            acoustic_delay,
            audio_pad_token,
            &[audio_codebooks],
        );

        println!("Starting TTS generation loop (max_steps={})...", max_steps);

        // --- Generation Loop ---
        for step_idx in 0..max_steps {
            // Check if generation should stop
            if Some(step_idx) == end_of_gen {
                println!("Stopping generation at step {} due to end_of_gen flag.", step_idx);
                break;
            }

            // Prepare audio inputs for this step
            let mut current_step_audio_inputs = Vec::with_capacity(audio_codebooks);
            for codebook in 0..audio_codebooks {
                let delay = if codebook == 0 { 0 } else { acoustic_delay };
                let mut token = if step_idx < delay {
                    audio_pad_token
                } else {
                    generated_audio_tokens[step_idx - delay][codebook]
                };
                if token == u32::MAX { 
                    token = audio_pad_token;
                }
                current_step_audio_inputs.push(Some(Tensor::new(&[token], &self.device)?.unsqueeze(0)?));
            }

            // Call LmModel forward_ca with text condition tokens
            let (_text_logits, audio_logits) = lm_guard.forward_ca(None, current_step_audio_inputs, &ca_src)?;

            // Sample next audio tokens
            let forced_tokens = forced_audio_tokens_helper.forced_tokens(step_idx);
            let next_audio_tokens = match lm_guard.depformer_sample(
                &audio_logits, 
                None,          
                &forced_tokens,
                &mut logits_processor,
            )? {
                Some(tokens) => tokens,
                None => { 
                    println!("depformer_sample returned None at step {} - stopping.", step_idx);
                    if end_of_gen.is_none() { 
                       end_of_gen = Some(step_idx + acoustic_delay + 1); 
                    }
                    break;
                }
            };

            // Store the generated tokens and check for stop condition
            for (c_idx, token) in next_audio_tokens.into_iter().enumerate() {
                let delay = if c_idx == 0 { 0 } else { acoustic_delay };
                let write_idx = step_idx.saturating_sub(delay);
                if write_idx < generated_audio_tokens.len() {
                    // Only write if the slot is still UNGENERATED 
                    // (handles potential overlap if loop breaks early)
                    if generated_audio_tokens[write_idx][c_idx] == u32::MAX {
                         generated_audio_tokens[write_idx][c_idx] = token;
                    }
                    
                    // Check for stop token only if end_of_gen hasn't been triggered yet
                    if end_of_gen.is_none() && token >= quantizer_bins {
                        println!(
                            "End-of-generation token {} (>= {}) detected at step {}, codebook {}",
                            token, quantizer_bins, step_idx, c_idx
                        );
                        // Set stop point to allow final tokens to be generated based on delay
                        end_of_gen = Some(step_idx + acoustic_delay + 1);
                    }
                }
            }
        }
        println!("TTS generation loop finished.");

        // 5. Vocoding
        // Determine the actual number of steps generated before stopping
        let num_valid_steps = end_of_gen.unwrap_or(max_steps);
        let num_valid_steps = std::cmp::min(num_valid_steps, generated_audio_tokens.len()); // Ensure bounds

        println!("Effective number of steps generated: {}", num_valid_steps);
        if num_valid_steps <= acoustic_delay { // Need at least delay steps for any output
            return Err(SpeechModelError::Processing("Generated insufficient audio steps before stopping".to_string()));
        }

        // Extract valid tokens, excluding initial delay padding if necessary
        // The first `acoustic_delay` steps might contain padding/artifacts 
        let relevant_tokens = &generated_audio_tokens[..(num_valid_steps - acoustic_delay)]; 

        // Check if any valid tokens were generated after the delay
        if relevant_tokens.is_empty() || relevant_tokens.iter().all(|step| step[0] == u32::MAX) {
             return Err(SpeechModelError::Processing("No valid audio tokens generated after acoustic delay".to_string()));
        }

        let codes_flat: Vec<u32> = relevant_tokens.iter().flatten().cloned().collect();
        let final_steps = relevant_tokens.len();

        let codes_tensor = Tensor::from_vec(codes_flat, (final_steps, audio_codebooks), &self.device)?
                             .transpose(0, 1)? // -> [K, T]
                             .unsqueeze(0)?;   // -> [1, K, T]

        println!("Decoding audio tokens (Shape: {:?}) with Mimi...", codes_tensor.shape());
        let waveform_tensor = mimi_guard.decode(&codes_tensor)?;

        // 6. Convert to Vec<f32>
        let waveform_vec = waveform_tensor.flatten_all()?.to_vec1::<f32>()?;
        println!("Vocoding complete. Generated {} audio samples.", waveform_vec.len());

        Ok(waveform_vec)
    }
} 
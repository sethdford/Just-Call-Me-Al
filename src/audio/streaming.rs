use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Instant, Duration};
use std::fmt;
use tokio::sync::Mutex as TokioMutex;
use anyhow::{Result, anyhow};
use tracing::{debug, warn};
use crate::models::CSMModel; // Keep only needed traits/types from models
 // Add import for ModelError
 // Add import for mpsc
 // Add import for info macro
 // Import the attribute macro

const MAX_BUFFER_SIZE: usize = 48000 * 5; // 5 seconds at 48kHz
const PROCESS_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Default, Debug)]
pub struct StreamData {
    // Raw audio samples
    pub samples: VecDeque<f32>,
    // Semantic tokens from backbone
    pub semantic_tokens: Option<Vec<u32>>,
    // Acoustic tokens from decoder (N codebooks)
    pub acoustic_tokens: Option<Vec<Vec<u32>>>,
    // Whether we have data to process
    pub has_data: bool,
    // Whether this is the final chunk
    pub is_final: bool,
    // Last processing time
    pub last_process: Option<Instant>,
    // Error count for backoff
    pub error_count: u32,
}

impl StreamData {
    pub fn extend_samples(&mut self, new_samples: &[f32]) -> Result<()> {
        // Check buffer size limit
        if self.samples.len() + new_samples.len() > MAX_BUFFER_SIZE {
            warn!("Buffer overflow, dropping {} samples", new_samples.len());
            return Err(anyhow!("Buffer overflow"));
        }
        
        self.samples.extend(new_samples.iter().cloned());
        self.has_data = true;
        Ok(())
    }

    pub fn set_tokens(&mut self, semantic: Vec<u32>, acoustic: Vec<Vec<u32>>) {
        self.semantic_tokens = Some(semantic);
        self.acoustic_tokens = Some(acoustic);
        self.last_process = Some(Instant::now());
        self.error_count = 0; // Reset error count on successful processing
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.semantic_tokens = None;
        self.acoustic_tokens = None;
        self.has_data = false;
        self.error_count = 0;
    }

    pub fn is_empty(&self) -> bool {
        !self.has_data || self.samples.is_empty()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_stale(&self) -> bool {
        match self.last_process {
            Some(last) => last.elapsed() > PROCESS_TIMEOUT,
            None => false,
        }
    }

    pub fn should_backoff(&self) -> bool {
        self.error_count > 3
    }

    pub fn get_backoff_duration(&self) -> Duration {
        Duration::from_millis(100 * (1 << self.error_count.min(6)))
    }
}

/// A stream of audio data being processed.
/// This represents a stream of audio data that can be added to and processed.
pub struct AudioStream {
    sample_rate: usize,
    channels: usize,
    bit_depth: usize,
    frame_size: usize,
    data: Arc<TokioMutex<StreamData>>,
    processing: Arc<TokioMutex<()>>,
    is_final: Arc<TokioMutex<bool>>,
}

impl AudioStream {
    /// Create a new audio stream.
    pub fn new(
        sample_rate: usize, 
        channels: usize, 
        bit_depth: usize, 
        frame_size: usize,
    ) -> Self {
        debug!("Creating new AudioStream with sample_rate={}, channels={}, bit_depth={}, frame_size={}", 
            sample_rate, channels, bit_depth, frame_size);
        Self {
            sample_rate,
            channels,
            bit_depth,
            frame_size,
            data: Arc::new(TokioMutex::new(StreamData::default())),
            processing: Arc::new(TokioMutex::new(())),
            is_final: Arc::new(TokioMutex::new(false)),
        }
    }

    /// Get the sample rate of the stream.
    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Get the number of channels in the stream.
    pub fn get_channels(&self) -> usize {
        self.channels
    }

    /// Get the bits per sample of the stream.
    pub fn get_bit_depth(&self) -> usize {
        self.bit_depth
    }

    /// Get the frame size of the stream.
    pub fn get_frame_size(&self) -> usize {
        self.frame_size
    }

    /// Process the stream with the given CSM model.
    pub async fn process_with_model(&self, model: Arc<TokioMutex<Box<dyn CSMModel + Send>>>) -> Result<Vec<f32>> {
        // Try to acquire the processing lock. If Err, it's already locked.
        let _guard = match self.processing.try_lock() {
            Ok(guard) => guard, // Lock acquired, guard held until end of scope
            Err(_) => { // Use Err(_) to catch the TryLockError
                // Lock is held by another task
                return Err(anyhow!("Already processing stream"));
            }
        };
        
        // Call the internal function. The guard is released when it goes out of scope.
        self.process_with_model_internal(model).await
    }

    async fn process_with_model_internal(&self, _model: Arc<TokioMutex<Box<dyn CSMModel + Send>>>) -> Result<Vec<f32>> {
        let mut data = self.data.lock().await;
        
        // Check if there is data to process
        if data.is_empty() {
            // No new data to process
            return Ok(Vec::new());
        }

        // Add a small artificial delay to simulate processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // --- Placeholder: Actual processing logic would go here --- 
        // This section should contain the logic to process the audio chunks
        // present in `data.samples` using the provided `_model`.
        // For now, we just drain the buffer to simulate consumption.
        let processed_chunk = data.samples.drain(..).collect::<Vec<f32>>();
        debug!("Simulated processing of {} samples.", processed_chunk.len());
        // ----------------------------------------------------------
        
        Ok(processed_chunk) // Return the simulated processed chunk
    }

    /// Process the stream with any Model.
    pub async fn process_with_any_model(&self, _model: &mut Box<dyn CSMModel + Send>) -> Result<Vec<f32>> {
        let data = self.data.lock().await;
        
        if data.is_empty() {
            return Err(anyhow!("No data to process"));
        }
        
        debug!("Processing audio stream with Model trait");
        warn!("process_with_any_model needs refactoring to use CSMModel::synthesize/synthesize_streaming");

        // Comment out original calls
        // let i16_samples: Vec<i16> = data.samples.iter()
        //     .map(|&sample| (sample * 32767.0) as i16)
        //     .collect();
        // let utterance = model.process_audio(&i16_samples).await?;
        // data.samples.make_contiguous();
        // let samples_slice = data.samples.as_slices().0;
        // let processed_audio = model.process(samples_slice).await?;
        
        Ok(Vec::from_iter(data.samples.iter().cloned())) // Return current buffer for now
    }

    /// Stop the stream.
    pub async fn stop(&self) -> Result<()> {
        let mut is_final = self.is_final.lock().await;
        *is_final = true;
        
        // Process any remaining audio
        if let Ok(mut data) = self.data.try_lock() {
            if !data.is_empty() {
                data.is_final = true;
            }
        }
        
        Ok(())
    }

    pub async fn write_audio(&self, audio: &[f32]) -> Result<()> {
        let mut data = self.data.lock().await;
        data.extend_samples(audio)
    }

    pub async fn read_audio(&self) -> Result<Vec<f32>> {
        let data = self.data.lock().await;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        Ok(Vec::from_iter(data.samples.iter().cloned()))
    }

    pub async fn clear_audio(&self) -> Result<()> {
        let mut data = self.data.lock().await;
        data.clear();
        Ok(())
    }

    pub async fn process_audio(&self, input: &[u8]) -> Result<()> {
        let mut data = self.data.lock().await;
        let samples = self.convert_bytes_to_samples(input)?;
        debug!("Processing {} bytes of audio data into {} samples", input.len(), samples.len());
        let _ = data.extend_samples(&samples);
        
        warn!("process_audio needs refactoring to use CSMModel::synthesize/synthesize_streaming");
        // Comment out original calls
        // let backbone = self.backbone.lock().await;
        // let semantic_tokens = backbone.process(&samples).await?;
        // let decoder = self.decoder.lock().await;
        // let acoustic_tokens = decoder.process(&semantic_tokens).await?;
        // data.set_tokens(
        //     semantic_tokens.iter().map(|&x| x as u32).collect(),
        //     vec![acoustic_tokens.iter().map(|&x| x as u32).collect()]
        // );
        
        Ok(())
    }

    pub async fn get_data(&self) -> Result<Vec<f32>> {
        let data = self.data.lock().await;
        Ok(Vec::from_iter(data.samples.iter().cloned()))
    }

    pub async fn append_data(&self, new_data: &[f32]) -> Result<()> {
        let mut data = self.data.lock().await;
        let _ = data.extend_samples(new_data);
        Ok(())
    }

    pub async fn set_final(&self) -> Result<()> {
        let mut is_final = self.is_final.lock().await;
        *is_final = true;
        Ok(())
    }

    pub async fn is_final(&self) -> Result<bool> {
        let is_final = self.is_final.lock().await;
        Ok(*is_final)
    }

    pub async fn process_frame(&mut self, _model: &Box<dyn CSMModel + Send + Sync>) -> Result<Vec<f32>> {
        let data = self.data.lock().await;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        warn!("process_frame needs refactoring to use CSMModel::synthesize/synthesize_streaming");
        // Comment out original call
        // data.samples.make_contiguous();
        // let processed = model.process(data.samples.as_slices().0).await?;
        // Ok(processed)
        
        Ok(Vec::from_iter(data.samples.iter().cloned())) // Return current buffer for now
    }

    // Comment out functions using removed types
    /*
    pub async fn synthesize(&self, text: &str, prosody: Option<ProsodyParams>) -> Result<Vec<f32>> {
        let decoder_model = self.decoder.lock().await;
        decoder_model.synthesize(text, prosody).await
    }

    pub async fn synthesize_with_emotion(
        &self,
        text: &str,
        emotion: &EmotionalContext,
        personality: Option<&VoicePersonality>
    ) -> Result<Vec<f32>> {
        let decoder_model = self.decoder.lock().await;
        let mut prosody = ProsodyParams::default();
        prosody.with_emotion(emotion);
        if let Some(p) = personality {
            prosody.with_personality(p);
        }
        decoder_model.synthesize(text, Some(prosody)).await
    }

    pub async fn synthesize_with_personality(
        &self,
        text: &str,
        personality: &VoicePersonality,
        prosody: Option<ProsodyParams>
    ) -> Result<Vec<f32>> {
        let decoder_model = self.decoder.lock().await;
        let mut final_prosody = prosody.unwrap_or_default();
        final_prosody.with_personality(personality);
        decoder_model.synthesize(text, Some(final_prosody)).await
    }

    pub async fn synthesize_in_conversation(
        &self,
        text: &str,
        state: &mut ConversationState,
        emotion: Option<EmotionalContext>
    ) -> Result<Vec<f32>> {
        let backbone = self.backbone.lock().await;
        let decoder = self.decoder.lock().await;
        
        // Build conversation context
        let context_prefix = if !state.turn_context.is_empty() {
            format!("[CONTEXT:{}]", state.turn_context)
        } else {
            String::new()
        };
        
        // Add style marker
        let style_prefix = format!("[STYLE:{}]", state.speaking_style);
        
        // Combine with emotional context if provided
        let emotion = emotion.unwrap_or_default();
        let emotional_prefix = format!("[{}:{}]", emotion.emotion, emotion.intensity);
        
        // Build full text with context
        let full_text = format!("{}{}{}{}", context_prefix, style_prefix, emotional_prefix, text);
        
        // Create prosody parameters
        let mut prosody = ProsodyParams::default();
        prosody.with_emotion(&emotion);
        
        // Generate with full context
        let result = if let Some(ref personality) = state.current_personality {
            self.synthesize_with_personality(&full_text, personality, Some(prosody)).await?
        } else {
            self.synthesize_with_emotion(&full_text, &emotion, None).await?
        };
        
        // Update conversation history
        state.add_to_history(text, emotion);
        
        Ok(result)
    }
    */

    fn convert_bytes_to_samples(&self, input: &[u8]) -> Result<Vec<f32>> {
        // Convert bytes to i16 samples
        let mut i16_samples = Vec::with_capacity(input.len() / 2);
        for chunk in input.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            i16_samples.push(sample);
        }
        
        // Convert i16 to f32 samples
        let f32_samples = i16_samples.iter()
            .map(|&x| x as f32 / i16::MAX as f32)
            .collect();
            
        Ok(f32_samples)
    }
}

/// Implementation of Display for AudioStream
impl fmt::Display for AudioStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AudioStream(rate={}, channels={})", self.sample_rate, self.channels)
    }
}

/// A trait for audio processing that occurs on chunks of audio data.
pub trait ChunkedAudioProcessing {
    /// Process a chunk of audio data.
    fn process_chunk(&mut self, chunk: &[f32]) -> Result<()>;
    
    /// Finalize the processing.
    fn finalize(&mut self) -> Result<Vec<f32>>;
}

/// An audio processor that buffers and streams audio.
pub struct StreamingAudioProcessor {
    buffer: VecDeque<f32>,
    chunk_size: usize,
}

impl StreamingAudioProcessor {
    /// Create a new streaming audio processor.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            chunk_size,
        }
    }

    /// Add a chunk of audio to the buffer.
    pub fn add_audio_chunk(&mut self, chunk: &[f32]) {
        self.buffer.extend(chunk);
    }

    /// Process the buffered audio.
    pub fn process_buffered_audio(&mut self) -> Result<Vec<f32>> {
        if self.buffer.len() < self.chunk_size {
            return Err(anyhow!("Not enough audio data to process"));
        }

        let mut chunk = Vec::with_capacity(self.chunk_size);
        for _ in 0..self.chunk_size {
            if let Some(sample) = self.buffer.pop_front() {
                chunk.push(sample);
            } else {
                break;
            }
        }

        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::models::ModelError;
    use tokio::sync::mpsc;
    use tracing::info;
    use async_trait::async_trait;

    #[derive(Clone)] 
    struct TestCSMModel;

    impl TestCSMModel {
        fn new() -> Self { TestCSMModel }
    }

    #[async_trait]
    impl CSMModel for TestCSMModel {
        async fn synthesize(
            &self,
            _text: &str,
            _temperature: Option<f64>,
            _top_k: Option<i64>,
            _seed: Option<u64>,
        ) -> Result<Vec<i16>, ModelError> {
            info!("TestCSMModel synthesize called");
            Ok(vec![0i16; 512])
        }

        async fn synthesize_streaming(
            &self,
            _text: &str,
            _temperature: Option<f64>,
            _top_k: Option<i64>,
            _seed: Option<u64>,
            audio_token_tx: mpsc::Sender<Vec<(i64, Vec<i64>)>>,
        ) -> Result<(), ModelError> {
            info!("TestCSMModel synthesize_streaming called (sending mock tokens)");
            // Send mock token data
            for i in 0..2 { 
                let mock_acoustic_tokens: Vec<i64> = vec![(i+5) as i64; 8]; // Example: 8 codebooks
                let mock_semantic_token = (i+5) as i64; // Dummy semantic token
                let chunk_to_send = vec![(mock_semantic_token, mock_acoustic_tokens)]; 
                if audio_token_tx.send(chunk_to_send).await.is_err() {
                    warn!("TestCSMModel: Failed to send token chunk.");
                    return Err(ModelError::ChannelSendError("TestCSMModel send failed".to_string()));
                }
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_stream_processing() -> Result<()> {
        let sample_rate = 16000;
        let channels = 1;
        let bit_depth = 16;
        let frame_size = 256;
        let stream = AudioStream::new(sample_rate, channels, bit_depth, frame_size);

        // Generate test audio
        let test_audio: Vec<f32> = (0..1000)
            .map(|i| (i as f32 / 100.0).sin())
            .collect();

        // Write audio in chunks
        for chunk in test_audio.chunks(100) {
            stream.write_audio(chunk).await?;
        }

        // Process stream
        let model = Arc::new(TokioMutex::new(Box::new(TestCSMModel::new()) as Box<dyn CSMModel + Send>));
        let processed = stream.process_with_model(model).await?;

        assert!(!processed.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_overflow() -> Result<()> {
        let sample_rate = 16000;
        let channels = 1;
        let bit_depth = 16;
        let frame_size = 256;
        let stream = AudioStream::new(sample_rate, channels, bit_depth, frame_size);

        // Try to write more than MAX_BUFFER_SIZE
        let large_chunk = vec![0.0f32; MAX_BUFFER_SIZE + 1000];
        assert!(stream.write_audio(&large_chunk).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_processing() -> Result<()> {
        let sample_rate = 16000;
        let channels = 1;
        let bit_depth = 16;
        let frame_size = 256;
        let stream = Arc::new(AudioStream::new(sample_rate, channels, bit_depth, frame_size));
        
        // Write some test audio
        stream.write_audio(&vec![0.0f32; 1000]).await?;
        
        // Try to process concurrently
        let mut handles = Vec::new();
        for _ in 0..3 {
            let stream = stream.clone();
            let handle = tokio::spawn(async move {
                stream.process_with_model(Arc::new(TokioMutex::new(Box::new(TestCSMModel::new()) as Box<dyn CSMModel + Send>))).await
            });
            handles.push(handle);
        }
        
        // Only one should succeed, others should get "Already processing" error
        let results = futures::future::join_all(handles).await;
        let successes = results.iter()
            .filter(|r| r.as_ref().map_or(false, |r| r.is_ok()))
            .count();
        assert_eq!(successes, 1);
        
        Ok(())
    }
}

// Comment out ConversationState struct
/*
#[derive(Clone, Debug)]
pub struct ConversationState {
    pub history: Vec<(String, EmotionalContext)>,
    pub current_personality: Option<VoicePersonality>,
    pub speaking_style: String,
    pub turn_context: String,
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            current_personality: None,
            speaking_style: "casual".to_string(),
            turn_context: String::new(),
        }
    }
}

impl ConversationState {
    pub fn with_style(&mut self, style: &str) -> &mut Self {
        self.speaking_style = style.to_string();
        self
    }

    pub fn with_personality(&mut self, personality: VoicePersonality) -> &mut Self {
        self.current_personality = Some(personality);
        self
    }

    pub fn with_context(&mut self, context: &str) -> &mut Self {
        self.turn_context = context.to_string();
        self
    }

    pub fn add_to_history(&mut self, text: &str, emotion: EmotionalContext) {
        self.history.push((text.to_string(), emotion));
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }
}
*/

// ProsodyParams implementations are defined in src/models/mod.rs
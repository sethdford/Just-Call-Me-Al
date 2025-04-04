// app/public/audio-worklet-processor.js

class EnhancedAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        
        // --- Core Configuration ---
        this.bufferSize = 4096;          // Main buffer size
        this.fadeLength = 128;           // For crossfading between audio chunks
        this.processorName = 'EnhancedAudioProcessor';
        
        // --- Buffer Management ---
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferFilled = 0;
        this.previousFrame = new Float32Array(128);
        this.isFirstChunk = true;
        
        // --- Playback Control ---
        this.isPlaying = false;
        this.isPaused = false;
        this.lastPlaybackTime = 0;
        
        // --- Dynamic Playback Rate (from StreamProcessor) ---
        this.playbackRate = 1.0;         // Current rate
        this.playbackRateMin = 0.9;      // Minimum rate (slower)
        this.playbackRateMax = 1.1;      // Maximum rate (faster)
        this.playbackRateAffordance = 0.2; // Buffer threshold as percentage
        this.playbackSmoothing = 0.9;    // Smoothing factor for rate changes
        
        // --- Buffering Controls ---
        this.targetBufferMs = 250;       // Target buffer size in milliseconds
        this.minBufferMs = 150;          // Minimum acceptable buffer
        this.maxBufferMs = 500;          // Maximum buffer size
        this.skipSilence = true;         // Skip digital silence
        
        // --- Recording Capabilities ---
        this.isRecording = false;
        this.recordedChunks = [];
        this.recordingStartTime = 0;
        this.recordingTrackId = null;
        
        // --- Network Quality Metrics ---
        this.underrunCount = 0;          // Buffer underruns
        this.chunkReceiveCount = 0;      // Number of chunks received
        this.lastChunkTime = 0;          // Timing between chunks
        this.chunkIntervals = [];        // Last 10 intervals
        
        // --- Track Management ---
        this.trackOffsets = {};          // Track playback positions
        
        // Set up message port for main thread communication
        this.port.onmessage = this.handleMessage.bind(this);
        
        console.log(`[AudioWorklet] ${this.processorName} initialized with buffer size: ${this.bufferSize}`);
    }
    
    handleMessage(event) {
        const { data } = event;
        
        try {
            switch (data.type) {
                case 'reset':
                    this.resetState();
                    break;
                    
                case 'config':
                    this.updateConfig(data);
                    break;
                    
                case 'audio_data':
                    this.handleAudioData(data);
                    break;
                    
                case 'start':
                    this.startPlayback(data);
                    break;
                    
                case 'pause':
                    this.pausePlayback();
                    break;
                    
                case 'resume':
                    this.resumePlayback();
                    break;
                    
                case 'stop':
                    this.stopPlayback();
                    break;
                    
                case 'record_start':
                    this.startRecording(data.trackId);
                    break;
                    
                case 'record_stop':
                    this.stopRecording();
                    break;
                    
                case 'export_recording':
                    this.exportRecording(data.requestId);
                    break;
                    
                default:
                    console.warn(`[AudioWorklet] Unknown message type: ${data.type}`);
            }
        } catch (err) {
            this.reportError('Message handling error', err);
        }
    }
    
    // --- State Management ---
    
    resetState() {
        this.bufferFilled = 0;
        this.isFirstChunk = true;
        this.isPlaying = false;
        this.isPaused = false;
        this.underrunCount = 0;
        
        // Don't clear recorded data on reset unless explicitly requested
        
        this.reportBufferState();
        console.log('[AudioWorklet] Processor state reset');
    }
    
    updateConfig(config) {
        // Update buffer configuration
        if (config.bufferSize !== undefined) this.bufferSize = config.bufferSize;
        if (config.fadeLength !== undefined) this.fadeLength = config.fadeLength;
        
        // Update playback rate configuration
        if (config.playbackRateMin !== undefined) this.playbackRateMin = config.playbackRateMin;
        if (config.playbackRateMax !== undefined) this.playbackRateMax = config.playbackRateMax;
        if (config.playbackRateAffordance !== undefined) this.playbackRateAffordance = config.playbackRateAffordance;
        if (config.playbackSmoothing !== undefined) this.playbackSmoothing = config.playbackSmoothing;
        
        // Update buffer targets
        if (config.targetBufferMs !== undefined) this.targetBufferMs = config.targetBufferMs;
        if (config.minBufferMs !== undefined) this.minBufferMs = config.minBufferMs;
        if (config.maxBufferMs !== undefined) this.maxBufferMs = config.maxBufferMs;
        
        // Update features
        if (config.skipSilence !== undefined) this.skipSilence = config.skipSilence;
        
        // Resize buffer if needed
        const targetSampleCount = Math.ceil((this.targetBufferMs / 1000) * sampleRate);
        if (targetSampleCount > this.buffer.length) {
            const newBuffer = new Float32Array(targetSampleCount);
            newBuffer.set(this.buffer.subarray(0, this.bufferFilled));
            this.buffer = newBuffer;
        }
        
        console.log('[AudioWorklet] Configuration updated');
    }
    
    // --- Audio Data Handling ---
    
    handleAudioData(data) {
        const audioData = data.audioData;
        const trackId = data.trackId || 'default';
        
        // Track network timing for quality metrics
        const now = performance.now ? performance.now() : Date.now();
        if (this.lastChunkTime > 0) {
            const interval = now - this.lastChunkTime;
            this.chunkIntervals.push(interval);
            if (this.chunkIntervals.length > 10) this.chunkIntervals.shift();
        }
        this.lastChunkTime = now;
        this.chunkReceiveCount++;
        
        // Check for silence if enabled
        const isSilence = this.skipSilence ? this.isDigitalSilence(audioData) : false;
        
        // Skip silence blocks at the beginning of playback
        if (isSilence && !this.isPlaying && this.bufferFilled === 0) {
            console.log('[AudioWorklet] Skipping initial silence block');
            return;
        }
        
        // Add to buffer
        this.addAudioData(audioData, trackId, isSilence);
        
        // Update track offsets
        if (trackId) {
            this.trackOffsets[trackId] = (this.trackOffsets[trackId] || 0) + audioData.length;
        }
        
        // Record if active
        if (this.isRecording) {
            // Record all audio or just the specific track
            if (this.recordingTrackId === null || this.recordingTrackId === trackId) {
                this.recordAudio(audioData, trackId);
            }
        }
        
        // Report buffer state
        this.reportBufferState();
        
        // Report network quality periodically
        if (this.chunkReceiveCount % 10 === 0) {
            this.reportNetworkQuality();
        }
    }
    
    isDigitalSilence(audioData) {
        // Check a subset of samples for efficiency
        const checkEvery = Math.max(1, Math.floor(audioData.length / 100));
        for (let i = 0; i < audioData.length; i += checkEvery) {
            if (Math.abs(audioData[i]) > 0.0001) return false;
        }
        return true;
    }
    
    addAudioData(audioData, trackId = null, isSilence = false) {
        // Check if we need to resize the buffer
        if (this.bufferFilled + audioData.length > this.buffer.length) {
            // Need more space - resize with some headroom
            const newSize = Math.max(this.buffer.length * 2, this.bufferFilled + audioData.length * 2);
            const newBuffer = new Float32Array(newSize);
            newBuffer.set(this.buffer.subarray(0, this.bufferFilled));
            this.buffer = newBuffer;
            console.log(`[AudioWorklet] Buffer resized to ${newSize} samples`);
        }
        
        // Apply fade-in to new data if not the first chunk and not silence
        if (!this.isFirstChunk && !isSilence) {
            const fadeLength = Math.min(this.fadeLength, audioData.length / 2);
            for (let i = 0; i < fadeLength; i++) {
                const factor = i / fadeLength;
                audioData[i] *= factor;
            }
        } else if (this.isFirstChunk) {
            this.isFirstChunk = false;
        }
        
        // Copy audio data to our buffer
        this.buffer.set(audioData, this.bufferFilled);
        this.bufferFilled += audioData.length;
        
        // Save last samples for future crossfading
        const frameSize = Math.min(this.previousFrame.length, audioData.length);
        const offset = audioData.length - frameSize;
        for (let i = 0; i < frameSize; i++) {
            this.previousFrame[i] = audioData[offset + i];
        }
    }
    
    // --- Playback Control ---
    
    startPlayback(options = {}) {
        this.isPlaying = true;
        this.isPaused = false;
        this.lastPlaybackTime = performance.now ? performance.now() : Date.now();
        
        if (options.resetBuffer) {
            this.bufferFilled = 0;
            this.isFirstChunk = true;
        }
        
        console.log('[AudioWorklet] Playback started');
        this.port.postMessage({ type: 'playback_started' });
    }
    
    pausePlayback() {
        this.isPaused = true;
        console.log('[AudioWorklet] Playback paused');
        this.port.postMessage({ type: 'playback_paused' });
    }
    
    resumePlayback() {
        this.isPaused = false;
        this.lastPlaybackTime = performance.now ? performance.now() : Date.now();
        console.log('[AudioWorklet] Playback resumed');
        this.port.postMessage({ type: 'playback_resumed' });
    }
    
    stopPlayback() {
        this.isPlaying = false;
        this.isPaused = false;
        this.bufferFilled = 0;
        console.log('[AudioWorklet] Playback stopped');
        this.port.postMessage({ type: 'playback_stopped' });
    }
    
    // --- Recording Functionality ---
    
    startRecording(trackId = null) {
        this.isRecording = true;
        this.recordingTrackId = trackId;
        this.recordingStartTime = performance.now ? performance.now() : Date.now();
        this.recordedChunks = [];
        console.log('[AudioWorklet] Recording started');
        this.port.postMessage({ type: 'recording_started', trackId });
    }
    
    stopRecording() {
        this.isRecording = false;
        const now = performance.now ? performance.now() : Date.now();
        const duration = (now - this.recordingStartTime) / 1000;
        console.log(`[AudioWorklet] Recording stopped (${duration.toFixed(1)}s)`);
        this.port.postMessage({ 
            type: 'recording_stopped', 
            duration,
            trackId: this.recordingTrackId
        });
    }
    
    recordAudio(audioData, trackId) {
        // Save a copy of the audio data for the recording
        this.recordedChunks.push(audioData.slice(0));
    }
    
    exportRecording(requestId) {
        try {
            // Merge all recorded chunks
            const totalSamples = this.recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const merged = new Float32Array(totalSamples);
            
            let offset = 0;
            for (const chunk of this.recordedChunks) {
                merged.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Convert to 16-bit PCM
            const pcmData = this.floatTo16BitPCM(merged);
            
            // Send the recording data back to the main thread
            this.port.postMessage({
                type: 'recording_exported',
                requestId,
                audioData: pcmData,
                sampleRate,
                duration: totalSamples / sampleRate,
                format: '16bitPCM'
            });
            
        } catch (err) {
            this.reportError('Export recording error', err);
        }
    }
    
    // --- Audio Processing ---
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const input = inputs[0]; // Use input if available (passthrough mode)
        
        // Get output channels
        const outputChannels = output.length;
        if (outputChannels === 0) return true;
        
        try {
            if (input && input.length > 0) {
                // INPUT MODE: Pass through input audio with enhancements
                this.processInputAudio(input, output, outputChannels);
            } else {
                // BUFFER MODE: Output audio from internal buffer
                this.processBufferedAudio(output, outputChannels);
            }
            
            // Always return true to keep the processor alive
            return true;
        } catch (e) {
            this.reportError('Audio processing error', e);
            return true; // Continue despite errors
        }
    }
    
    processInputAudio(input, output, outputChannels) {
        // Apply our processing to input audio and send to output
        for (let channel = 0; channel < Math.min(outputChannels, input.length); channel++) {
            if (output[channel] && input[channel]) {
                // Apply crossfading/smoothing to avoid clicks
                if (channel === 0) { // Only process for first channel
                    this.blendWithPrevious(input[channel]);
                    this.savePreviousFrame(input[channel]);
                }
                
                // Add to recording if active
                if (this.isRecording && channel === 0) {
                    this.recordAudio(input[channel], 'input');
                }
                
                // Copy processed input to output
                output[channel].set(input[channel]);
            }
        }
    }
    
    processBufferedAudio(output, outputChannels) {
        if (this.isPaused || !this.isPlaying) {
            // If paused or not playing, output silence
            for (let channel = 0; channel < outputChannels; channel++) {
                output[channel].fill(0);
            }
            return;
        }
        
        // Determine how much data to output based on playback rate
        let targetPlaybackRate = 1.0;
        
        // Calculate adaptive playback rate if enabled
        if (this.playbackRateMin < this.playbackRateMax) {
            const targetSamples = (this.targetBufferMs / 1000) * sampleRate;
            const bufferPercentage = this.bufferFilled / targetSamples;
            
            if (Math.abs(bufferPercentage - 1.0) > this.playbackRateAffordance) {
                if (bufferPercentage < 1.0) {
                    // Buffer is low, slow down playback
                    targetPlaybackRate = this.playbackRateMin + 
                        (1.0 - this.playbackRateMin) * (bufferPercentage / (1.0 - this.playbackRateAffordance));
                } else {
                    // Buffer is high, speed up playback
                    const excessPercentage = (bufferPercentage - 1.0) / this.playbackRateAffordance;
                    targetPlaybackRate = 1.0 + (this.playbackRateMax - 1.0) * 
                        Math.min(1.0, excessPercentage);
                }
            }
            
            // Apply smoothing to playback rate changes
            this.playbackRate = this.playbackRate * this.playbackSmoothing + 
                targetPlaybackRate * (1.0 - this.playbackSmoothing);
                
            // Clamp to valid range
            this.playbackRate = Math.max(this.playbackRateMin, 
                                Math.min(this.playbackRateMax, this.playbackRate));
        }
        
        // Calculate how many samples to output
        const frameSize = output[0].length;
        const outputSamplesNeeded = Math.ceil(frameSize * this.playbackRate);
        
        if (this.bufferFilled > 0) {
            const samplesAvailable = Math.min(this.bufferFilled, outputSamplesNeeded);
            
            // Check for buffer underrun
            if (samplesAvailable < frameSize && this.isPlaying) {
                this.underrunCount++;
                if (this.underrunCount % 5 === 0) {
                    console.warn(`[AudioWorklet] Buffer underrun #${this.underrunCount}: needed ${frameSize}, had ${samplesAvailable}`);
                }
            }
            
            // Get samples from buffer
            const samplesOutput = new Float32Array(samplesAvailable);
            for (let i = 0; i < samplesAvailable; i++) {
                samplesOutput[i] = this.buffer[i];
            }
            
            // Apply time-stretching/compression based on playback rate
            let processedSamples;
            if (Math.abs(this.playbackRate - 1.0) > 0.01) {
                processedSamples = this.resampleAudio(samplesOutput, frameSize);
            } else {
                processedSamples = samplesOutput;
            }
            
            // Fill output channels
            const framesOutput = Math.min(processedSamples.length, frameSize);
            for (let channel = 0; channel < outputChannels; channel++) {
                // Copy available samples
                for (let i = 0; i < framesOutput; i++) {
                    output[channel][i] = processedSamples[i];
                }
                
                // Zero out any remaining samples
                for (let i = framesOutput; i < frameSize; i++) {
                    output[channel][i] = 0;
                }
            }
            
            // Update buffer state
            this.buffer.copyWithin(0, samplesAvailable, this.bufferFilled);
            this.bufferFilled -= samplesAvailable;
            
            // Add to recording if active
            if (this.isRecording) {
                // Only record what was actually output (after resampling)
                const outputRecording = new Float32Array(framesOutput);
                for (let i = 0; i < framesOutput; i++) {
                    outputRecording[i] = processedSamples[i];
                }
                this.recordAudio(outputRecording, 'output');
            }
            
        } else {
            // No buffered data, output silence
            for (let channel = 0; channel < outputChannels; channel++) {
                output[channel].fill(0);
            }
            
            // Report buffer status when empty
            if (this.isPlaying) {
                this.underrunCount++;
                this.reportBufferState();
            }
        }
        
        // Report buffer state if running low
        if (this.bufferFilled < frameSize * 2 && this.isPlaying) {
            this.reportBufferState();
        }
    }
    
    // --- Utility Functions ---
    
    blendWithPrevious(currentBuffer) {
        if (this.isFirstChunk) {
            this.isFirstChunk = false;
            return;
        }
        
        // Simple crossfade between chunks to avoid discontinuities
        const blendLength = Math.min(this.previousFrame.length, currentBuffer.length);
        for (let i = 0; i < blendLength; i++) {
            const fadeFactor = i / blendLength;
            currentBuffer[i] = (currentBuffer[i] * fadeFactor) + 
                              (this.previousFrame[i] * (1 - fadeFactor));
        }
    }
    
    savePreviousFrame(buffer) {
        const frameSize = Math.min(this.previousFrame.length, buffer.length);
        const offset = buffer.length - frameSize;
        
        for (let i = 0; i < frameSize; i++) {
            this.previousFrame[i] = buffer[offset + i];
        }
    }
    
    resampleAudio(inputBuffer, targetLength) {
        // Simple linear interpolation resampling
        const output = new Float32Array(targetLength);
        const inputLength = inputBuffer.length;
        const ratio = inputLength / targetLength;
        
        for (let i = 0; i < targetLength; i++) {
            const inputIdx = i * ratio;
            const inputIdx1 = Math.floor(inputIdx);
            const inputIdx2 = Math.min(inputIdx1 + 1, inputLength - 1);
            const frac = inputIdx - inputIdx1;
            
            // Linear interpolation
            output[i] = inputBuffer[inputIdx1] * (1 - frac) + inputBuffer[inputIdx2] * frac;
        }
        
        return output;
    }
    
    floatTo16BitPCM(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        
        for (let i = 0, offset = 0; i < float32Array.length; i++, offset += 2) {
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, intSample, true); // true = little-endian
        }
        
        return buffer;
    }
    
    // --- Reporting Functions ---
    
    reportBufferState() {
        const bufferMs = (this.bufferFilled / sampleRate) * 1000;
        const targetMs = this.targetBufferMs;
        const percentFilled = (bufferMs / targetMs) * 100;
        
        this.port.postMessage({
            type: 'buffer_state',
            samples: this.bufferFilled,
            milliseconds: bufferMs.toFixed(1),
            percentFilled: percentFilled.toFixed(1),
            targetMs,
            playbackRate: this.playbackRate.toFixed(3),
            underruns: this.underrunCount
        });
    }
    
    reportNetworkQuality() {
        // Calculate network quality metrics
        let avgInterval = 0;
        let jitter = 0;
        
        if (this.chunkIntervals.length > 0) {
            avgInterval = this.chunkIntervals.reduce((a, b) => a + b, 0) / this.chunkIntervals.length;
            
            // Calculate jitter as average deviation from mean interval
            jitter = this.chunkIntervals.reduce((sum, interval) => {
                return sum + Math.abs(interval - avgInterval);
            }, 0) / this.chunkIntervals.length;
        }
        
        // Estimate network quality (0-1) based on jitter and underruns
        const jitterFactor = avgInterval > 0 ? Math.min(1, Math.max(0, 1 - (jitter / avgInterval))) : 0;
        const underrunFactor = Math.min(1, Math.max(0, 1 - (this.underrunCount / 50)));
        const quality = 0.5 * jitterFactor + 0.5 * underrunFactor;
        
        this.port.postMessage({
            type: 'network_quality',
            quality: quality.toFixed(2),
            metrics: {
                avgChunkInterval: avgInterval.toFixed(1),
                jitter: jitter.toFixed(1),
                underruns: this.underrunCount,
                chunks: this.chunkReceiveCount
            }
        });
    }
    
    reportError(context, error) {
        console.error(`[AudioWorklet] ${context}:`, error);
        this.port.postMessage({
            type: 'error',
            context,
            message: error.toString(),
            stack: error.stack
        });
    }
}

try {
    // Register our enhanced processor
    registerProcessor('audio-worklet-processor', EnhancedAudioProcessor);
    console.log('[AudioWorklet] EnhancedAudioProcessor registered successfully');
} catch (e) {
    console.error('[AudioWorklet] Failed to register processor:', e);
} 
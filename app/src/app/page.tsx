"use client"; // Required for components with hooks and event handlers

import React, { useState, useEffect, useRef, useCallback } from 'react';
// We will add state management (useState, useEffect, etc.) later

// Define the structure for messages if needed
interface Message {
  type: 'user' | 'bot' | 'system';
  text: string;
}

interface RecognizedWord {
  text: string;
  start_time: number;
  stop_time: number;
}

// --- Constants ---
const AUDIO_WORKLET_PROCESSOR_URL = '/audio-worklet-processor.js'; // Path in public directory
const TARGET_SAMPLE_RATE = 16000; // Or your desired sample rate

// --- Audio Player Class (Adapted) ---
class AudioPlayer {
    private audioContext: AudioContext;
    private workletNode: AudioWorkletNode | null = null;
    private audioQueue: AudioBuffer[] = [];
    private isPlaying: boolean = false;
    private startTime: number = 0;
    private currentSource: AudioBufferSourceNode | null = null;

    // State update callbacks provided by the React component
    private updateStatusCallback: (message: string) => void;
    private logCallback: (message: string, level?: string) => void;
    private showErrorCallback: (message: string) => void;

    constructor(
        audioContext: AudioContext,
        workletNode: AudioWorkletNode,
        updateStatusCallback: (message: string) => void,
        logCallback: (message: string, level?: string) => void,
        showErrorCallback: (message: string) => void
    ) {
        this.audioContext = audioContext;
        this.workletNode = workletNode; // We might use this later for custom processing
        this.updateStatusCallback = updateStatusCallback;
        this.logCallback = logCallback;
        this.showErrorCallback = showErrorCallback;
        this.logCallback("AudioPlayer initialized.", "debug");
    }

    // Decodes raw Float32 buffer and queues it for playback
    async processAudioData(rawF32ArrayBuffer: ArrayBuffer) {
        this.logCallback(`[AudioPlayer] processAudioData called with ${rawF32ArrayBuffer.byteLength} bytes`, "debug");
        if (!this.audioContext) {
            this.logCallback("AudioContext not available for processing data.", "warn");
            return;
        }

        try {
            // Assuming the incoming data is already Float32Array
            const float32Data = new Float32Array(rawF32ArrayBuffer);
            // Calculate duration based on TARGET_SAMPLE_RATE
            const numberOfFrames = float32Data.length;
            const audioBuffer = this.audioContext.createBuffer(1, numberOfFrames, TARGET_SAMPLE_RATE);
            audioBuffer.copyToChannel(float32Data, 0);

            this.audioQueue.push(audioBuffer);
            this.logCallback(`[AudioPlayer] Queued chunk: Frames=${numberOfFrames}, Duration=${audioBuffer.duration.toFixed(3)}s. Queue size: ${this.audioQueue.length}`, "debug");

            if (!this.isPlaying) {
                this.playNextChunk();
            }
        } catch (error) {
            console.error("Error decoding audio data:", error);
            this.showErrorCallback("Error processing received audio.");
        }
    }

    playNextChunk() {
        if (!this.audioContext || this.audioContext.state !== 'running') {
            this.isPlaying = false;
            this.logCallback(`[AudioPlayer] Context not running (${this.audioContext?.state}). Stopping playback.`, 'warn');
            this.updateStatusCallback("Audio playback paused. Click page to resume.");
            return;
        }

        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            this.logCallback("[AudioPlayer] Queue empty, playback finished.", "info");
            this.updateStatusCallback("Playback finished. Ready.");
            return;
        }

        this.isPlaying = true;
        this.updateStatusCallback("Playing audio...");
        const audioBuffer = this.audioQueue.shift()!;
        this.currentSource = this.audioContext.createBufferSource();
        this.currentSource.buffer = audioBuffer;
        this.currentSource.connect(this.audioContext.destination);

        const currentTime = this.audioContext.currentTime;
        const startTime = (this.startTime <= currentTime) ? currentTime : this.startTime;

        this.logCallback(`[AudioPlayer] Starting playback for chunk. Context state: ${this.audioContext.state}. Start time: ${startTime.toFixed(3)}. Duration: ${audioBuffer.duration.toFixed(3)}`, "debug");
        this.currentSource.start(startTime);

        this.startTime = startTime + audioBuffer.duration;

        this.currentSource.onended = () => {
            this.logCallback("[AudioPlayer] Chunk playback ended.", "debug");
            this.playNextChunk(); // Automatically play the next chunk
        };
    }

    stop() {
        this.logCallback("[AudioPlayer] Stop requested.", "info");
        if (this.currentSource) {
            try {
                this.currentSource.stop();
                this.currentSource.disconnect();
            } catch (e) {
                 this.logCallback("Error stopping current source (may have already finished)", "warn");
            }
            this.currentSource = null;
        }
        this.audioQueue = []; // Clear the queue
        this.isPlaying = false;
        this.startTime = this.audioContext ? this.audioContext.currentTime : 0;
        this.updateStatusCallback("Audio stopped. Ready.");
    }
}

export default function Home() {
  // --- State Variables ---
  const [status, setStatus] = useState<string>('Initializing...');
  const [error, setError] = useState<string | null>(null);
  const [webSocket, setWebSocket] = useState<WebSocket | null>(null);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [audioPlayer, setAudioPlayer] = useState<AudioPlayer | null>(null);
  const [audioWorkletNode, setAudioWorkletNode] = useState<AudioWorkletNode | null>(null);
  const [textInputValue, setTextInputValue] = useState<string>('');
  const [isReady, setIsReady] = useState<boolean>(false); // Controls button enable/disable
  const [activeCharacter, setActiveCharacter] = useState<string>('john');
  const [messages, setMessages] = useState<Message[]>([]); // For conversation messages
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [transcript, setTranscript] = useState<string>('');
  const [recognizedWords, setRecognizedWords] = useState<RecognizedWord[]>([]);
  const [isMicActive, setIsMicActive] = useState<boolean>(false);
  const [isSTTAvailable, setIsSTTAvailable] = useState<boolean>(false);

  // Refs
  const isInitializingAudio = useRef<boolean>(false); // Prevent race conditions
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // --- Helper Functions (useCallback for stability) ---
  const log = useCallback((message: string, level: string = 'info') => {
    console.log(`[${level.toUpperCase()}] ${message}`);
  }, []);

  const updateStatus = useCallback((message: string) => {
    setStatus(message);
    log(`Status updated: ${message}`, 'debug');
  }, [log]);

  const showError = useCallback((message: string) => {
    setError(message);
    log(`Error shown: ${message}`, 'error');
    // Optionally clear the error after a delay
    // setTimeout(() => setError(null), 5000);
  }, [log]);

  // --- Audio Initialization Logic ---
  const initializeAudio = useCallback(async () => {
    if (audioContext || isInitializingAudio.current) {
      log("Audio already initialized or initialization in progress.", "debug");
      return;
    }
    isInitializingAudio.current = true;
    log("Initializing AudioContext...");
    updateStatus("Initializing audio system...");

    try {
      const context = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: TARGET_SAMPLE_RATE, // Ensure context matches target rate
        latencyHint: 'interactive'
      });
      setAudioContext(context);
      log(`AudioContext created. State: ${context.state}, Sample Rate: ${context.sampleRate}`);

      // Handle suspended state (requires user interaction)
      if (context.state === 'suspended') {
        updateStatus("Audio paused. Click anywhere to enable audio.");
        log("AudioContext is suspended. Waiting for user interaction.", "warn");
        const resume = async () => {
          try {
            await context.resume();
            log(`AudioContext resumed. State: ${context.state}`);
            if (context.state === 'running') {
              updateStatus("Audio Ready.");
              // Try initializing worklet/player *after* resume succeeds
              await loadWorkletAndPlayer(context);
            } else {
              showError("Failed to resume audio context.");
            }
          } catch (err) {
            console.error("Error resuming AudioContext:", err);
            showError("Could not enable audio.");
          } finally {
            document.body.removeEventListener('click', resume);
            document.body.removeEventListener('keydown', resume);
          }
        };
        document.body.addEventListener('click', resume, { once: true });
        document.body.addEventListener('keydown', resume, { once: true });
      } else {
        // If context is already running, proceed directly
        await loadWorkletAndPlayer(context);
      }

    } catch (err) {
      console.error("Failed to initialize AudioContext:", err);
      showError("AudioContext not supported or initialization failed.");
    } finally {
      isInitializingAudio.current = false;
    }
  }, [audioContext, updateStatus, log, showError]);

  // Helper to load worklet and create player
  const loadWorkletAndPlayer = useCallback(async (context: AudioContext) => {
    if (!context || context.state !== 'running') {
      log("Cannot load worklet/player, context not running.", "warn");
      return;
    }
    try {
      log(`Loading AudioWorklet processor from: ${AUDIO_WORKLET_PROCESSOR_URL}`);
      await context.audioWorklet.addModule(AUDIO_WORKLET_PROCESSOR_URL);
      log("AudioWorklet module added successfully.");

      // Example: Create worklet node (if needed, otherwise can be omitted)
      const workletNodeInstance = new AudioWorkletNode(context, 'audio-worklet-processor');
      setAudioWorkletNode(workletNodeInstance);
      log("AudioWorkletNode created.");

      // Instantiate AudioPlayer
      const player = new AudioPlayer(context, workletNodeInstance, updateStatus, log, showError);
      setAudioPlayer(player);
      updateStatus("Audio system ready."); // Final status update
      setIsReady(true); // Enable buttons fully now

    } catch (err) {
      console.error("Failed to load AudioWorklet or create AudioPlayer:", err);
      showError(`Failed to load audio processor: ${err}`);
      setIsReady(false); // Keep buttons disabled if audio fails
    }
  }, [updateStatus, log, showError]);

  // Handle WebSocket message parsing for STT
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    log("WebSocket message received.", "debug");
    
    // Handle binary messages (audio data)
    if (audioPlayer && event.data instanceof Blob) {
      log("[WS] Received binary message (Blob)", "debug");
      event.data.arrayBuffer().then(buffer => {
        audioPlayer.processAudioData(buffer);
      }).catch(err => {
        console.error("Error reading Blob data:", err);
        showError("Error processing received audio data.");
      });
      return;
    }
    
    // Handle text messages (JSON)
    if (typeof event.data === 'string') {
      log(`[WS] Received text message: ${event.data}`, "debug");
      try {
        const messageData = JSON.parse(event.data);
        
        switch (messageData.type) {
          case 'init':
            log(`Server supports STT: ${messageData.stt_available}`);
            setIsSTTAvailable(messageData.stt_available);
            break;
            
          case 'transcript':
            log(`Received transcript: ${messageData.text}`);
            // Add the recognized word to the state
            setRecognizedWords(prev => [...prev, {
              text: messageData.text,
              start_time: messageData.start_time,
              stop_time: messageData.stop_time
            }]);
            // Append to the full transcript
            setTranscript(prev => prev + ' ' + messageData.text);
            break;
            
          case 'transcription':
            // Example: Add to messages state
            setMessages(prev => [...prev, { type: 'bot', text: messageData.text }]);
            break;
            
          case 'error':
            showError(`Server error: ${messageData.message}`);
            break;
            
          default:
            log(`Unknown message type: ${messageData.type}`, 'warn');
        }
      } catch (e) {
        log(`Non-JSON text message or parse error: ${event.data}`, 'warn');
      }
    } else {
      log("Received unexpected message type.", "warn");
    }
  }, [audioPlayer, log, showError]);

  // Update the WebSocket connection useEffect to use handleWebSocketMessage
  useEffect(() => {
    let wsInstance: WebSocket | null = null;

    const connectWebSocket = async () => {
      updateStatus('Connecting to server...');
      const wsUrl = `ws://localhost:8000/ws`; // Point to the Rust backend port
      log(`Attempting to connect WebSocket: ${wsUrl}`);
      setIsReady(false); // Disable buttons while connecting

      try {
        wsInstance = new WebSocket(wsUrl);
        // Assign handlers *before* setting state
        wsInstance.onopen = () => {
          log("WebSocket connection established.");
          updateStatus('Connection successful. Initializing audio...');
          setWebSocket(wsInstance);
          initializeAudio(); // Initialize audio *after* connection opens
        };

        wsInstance.onmessage = handleWebSocketMessage;

        wsInstance.onerror = (event) => {
          console.error("WebSocket error:", event);
          showError("WebSocket connection error.");
          setWebSocket(null); // Clear WS state on error
          setIsReady(false);
        };

        wsInstance.onclose = (event) => {
          log(`WebSocket connection closed: Code=${event.code}, Reason=${event.reason}`);
          if (webSocket) { // Only update status if it wasn't an error/initial fail
            updateStatus("Connection closed.");
          }
          setWebSocket(null);
          setIsReady(false);
          setAudioPlayer(null); // Clear player on disconnect
          setAudioContext(null); // Clear context
          setAudioWorkletNode(null);
          // TODO: Add reconnection logic if desired
        };

      } catch (err) {
        console.error("WebSocket connection failed:", err);
        showError("Failed to establish WebSocket connection.");
        setIsReady(false);
      }
    };

    connectWebSocket();

    // Cleanup function
    return () => {
      if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
        log("Closing WebSocket connection.");
        wsInstance.close(1000, "Client disconnecting");
      }
      if (audioContext && audioContext.state !== 'closed') {
        log("Closing AudioContext.");
        audioContext.close();
      }
      setWebSocket(null);
      setAudioContext(null);
      setAudioPlayer(null);
      setAudioWorkletNode(null);
      setIsReady(false);
    };
  }, [handleWebSocketMessage, initializeAudio, log, updateStatus, showError]); // Add handleWebSocketMessage to dependencies

  // --- Send Message Function ---
  const sendWebSocketMessage = useCallback((message: object) => {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
      try {
        const messageString = JSON.stringify(message);
        log(`[WS] Sending message: ${messageString}`, "debug");
        webSocket.send(messageString);
      } catch (e) {
        console.error("Failed to send WebSocket message:", e);
        showError(`Failed to send message: ${e}`);
      }
    } else {
      showError('Error: WebSocket connection not ready or lost.');
      log('WebSocket is not open. Cannot send message.', 'error');
    }
  }, [webSocket, log, showError]); // Dependency: webSocket state

  // Toggle microphone function for STT
  const toggleMicrophone = useCallback(async () => {
    if (isMicActive) {
      // Stop the microphone
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach(track => track.stop());
        micStreamRef.current = null;
      }
      
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      
      setIsMicActive(false);
      updateStatus('Microphone stopped');
    } else {
      // Start the microphone
      try {
        if (!audioContext) {
          showError('Audio context not initialized');
          return;
        }
        
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        
        // Ensure AudioContext is running
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        
        // Create a source from the microphone stream
        const source = audioContext.createMediaStreamSource(micStreamRef.current);
        
        // Create a script processor for processing audio data
        processorRef.current = audioContext.createScriptProcessor(4096, 1, 1);
        
        // Process audio data
        processorRef.current.onaudioprocess = (e) => {
          if (!isMicActive || !webSocket || webSocket.readyState !== WebSocket.OPEN) return;
          
          const inputData = e.inputBuffer.getChannelData(0);
          
          // Resample to 24kHz if needed (simple linear interpolation)
          let data: Float32Array;
          if (audioContext?.sampleRate !== 24000) {
            const ratio = 24000 / audioContext.sampleRate;
            const newLength = Math.floor(inputData.length * ratio);
            data = new Float32Array(newLength);
            
            for (let i = 0; i < newLength; i++) {
              const srcIdx = i / ratio;
              const srcIdxFloor = Math.floor(srcIdx);
              const srcIdxCeil = Math.ceil(srcIdx);
              
              if (srcIdxCeil >= inputData.length) {
                data[i] = inputData[srcIdxFloor];
              } else {
                const t = srcIdx - srcIdxFloor;
                data[i] = inputData[srcIdxFloor] * (1 - t) + inputData[srcIdxCeil] * t;
              }
            }
          } else {
            data = inputData;
          }
          
          // Send the audio data to the server
          sendWebSocketMessage({
            type: 'audio_data',
            data: Array.from(data),
            sample_rate: 24000
          });
        };
        
        // Connect the processor
        source.connect(processorRef.current);
        processorRef.current.connect(audioContext.destination);
        
        setIsMicActive(true);
        updateStatus('Microphone active');
      } catch (err) {
        console.error('Error accessing microphone:', err);
        showError(`Microphone error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }, [isMicActive, audioContext, webSocket, sendWebSocketMessage, updateStatus, showError]);

  // Function to clear the transcript
  const clearTranscript = useCallback(() => {
    setTranscript('');
    setRecognizedWords([]);
  }, []);

  // --- Event Handlers ---
  const handleSynthesize = useCallback(() => {
    if (!isReady || !webSocket) {
      showError("Not connected, cannot synthesize.");
      return;
    }
    const text = textInputValue.trim();
    if (text) {
      log(`Synthesizing text: ${text}`);
      sendWebSocketMessage({
        type: 'synthesize',
        text: text,
        character: activeCharacter // Send selected character
      });
      // Example: Add user message to display
      // setMessages(prev => [...prev, { type: 'user', text: text }]);
      // Clear input after sending?
      // setTextInputValue('');
    } else {
      showError('Please enter text to synthesize.');
    }
  }, [isReady, webSocket, textInputValue, activeCharacter, sendWebSocketMessage, log, showError]);

  const handleStopAudio = useCallback(() => {
    if (audioPlayer) {
      audioPlayer.stop();
      // Optionally send a stop message to the server if needed
      // sendWebSocketMessage({ action: 'stop_audio' });
    }
  }, [audioPlayer]);

  // TODO: Implement handleStartConversation, handleStopConversation

  // --- JSX Structure ---
  return (
    <div id="app-wrapper">
      <header id="app-header">
        <div className="logo">CSM Demo</div>
        {/* Status/Error will be managed by state and rendered here */}
      </header>

      <main id="app-main">
        <div id="status-error-container">
          <div id="status">{status}</div> {/* Display status from state */}
          {error && <div id="error" style={{ display: 'block' }}>{error}</div>} {/* Display error from state */}
        </div>

        <div id="demo-selector">
          <h3>Conversational Voice Demo</h3>
          <div className="character-choice">
            {/* Character buttons - state management needed */}
             <button
               id="john-button"
               className={`character-button ${activeCharacter === 'john' ? 'active' : ''}`}
               onClick={() => setActiveCharacter('john')}
               disabled={!isReady}
             >
               {/* SVG for John */}
               <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 19 19" className="icon"><path fill="currentColor" d="M5.408 13.373C2.586 10.561.448 7.163.448 4.359c0-1.24.42-2.363 1.366-3.271C2.391.53 3.054.238 3.7.238c.528 0 1.016.205 1.348.674l2.1 2.96c.332.458.488.839.488 1.19 0 .45-.264.84-.694 1.29l-.693.712a.5.5 0 0 0-.146.362c0 .146.058.283.107.39.312.606 1.201 1.641 2.158 2.598.967.957 2.002 1.846 2.608 2.168a.9.9 0 0 0 .39.107.53.53 0 0 0 .371-.156l.694-.683c.449-.44.85-.694 1.289-.694.351 0 .742.156 1.191.469l2.998 2.129c.46.332.645.8.645 1.289 0 .664-.322 1.338-.84 1.914-.889.977-1.992 1.416-3.252 1.416-2.803 0-6.23-2.178-9.053-5"></path></svg>
               <span>John</span>
             </button>
             <button
               id="peter-button"
               className={`character-button ${activeCharacter === 'peter' ? 'active' : ''}`}
               onClick={() => setActiveCharacter('peter')}
               disabled={!isReady}
             >
               {/* SVG for Peter */}
               <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 19 19" className="icon"><path fill="currentColor" d="M5.408 13.373C2.586 10.561.448 7.163.448 4.359c0-1.24.42-2.363 1.366-3.271C2.391.53 3.054.238 3.7.238c.528 0 1.016.205 1.348.674l2.1 2.96c.332.458.488.839.488 1.19 0 .45-.264.84-.694 1.29l-.693.712a.5.5 0 0 0-.146.362c0 .146.058.283.107.39.312.606 1.201 1.641 2.158 2.598.967.957 2.002 1.846 2.608 2.168a.9.9 0 0 0 .39.107.53.53 0 0 0 .371-.156l.694-.683c.449-.44.85-.694 1.289-.694.351 0 .742.156 1.191.469l2.998 2.129c.46.332.645.8.645 1.289 0 .664-.322 1.338-.84 1.914-.889.977-1.992 1.416-3.252 1.416-2.803 0-6.23-2.178-9.053-5"></path></svg>
               <span>Peter</span>
             </button>
          </div>
        </div>

        <div id="controls">
          <textarea
            id="textInput"
            placeholder="Enter text to synthesize..."
            value={textInputValue}
            onChange={(e) => setTextInputValue(e.target.value)}
            disabled={!isReady}
          />
          <div className="button-group">
            <button id="synthesizeButton" onClick={handleSynthesize} disabled={!isReady || !textInputValue.trim()}>Synthesize</button>
            <button id="stopButton" onClick={handleStopAudio} disabled={!isReady}>Stop Audio</button>
          </div>
          <div className="button-group conversation-controls">
            <button id="startConversation" disabled={!isReady}>Start Conversation</button>
            <button id="stopConversation" disabled>Stop Conversation</button> {/* TODO: Add handler and state */}
          </div>
        </div>

        {/* Add STT UI */}
        {isSTTAvailable && (
          <div id="stt-container">
            <h3>Speech-to-Text</h3>
            <div id="transcript-box">
              <p>{transcript}</p>
            </div>
            <div className="button-group stt-controls">
              <button 
                id="microphone-button" 
                onClick={toggleMicrophone} 
                disabled={!isReady}
                className={isMicActive ? 'active' : ''}
              >
                {isMicActive ? 'Stop Microphone' : 'Start Microphone'}
              </button>
              <button id="clear-transcript" onClick={clearTranscript} disabled={!isReady}>
                Clear Transcript
              </button>
            </div>
          </div>
        )}

        <div id="messages">
          {messages.map((msg, index) => (
             <div key={index} className={`message ${msg.type}`}>{msg.text}</div>
          ))}
        </div>
      </main>

      <footer id="app-footer">
        Copyright © 2025 CSM Demo. All rights reserved.
      </footer>

      {/* Script tag is no longer needed here, logic moves into the component */}
    </div>
  );
}

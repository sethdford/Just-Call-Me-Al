"use client"; // Required for components with hooks and event handlers

import React, { useState, useEffect, useRef, useCallback, Component, ErrorInfo, ReactNode } from 'react';
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

// WebSocket message type definitions
interface WSMessageBase {
  type: string;
}

interface WSAudioData extends WSMessageBase {
  type: 'AudioData';
  data: number[];
  sampleRate: number;
  request_codes: boolean;
}

// Add proper AudioWorklet message event type
interface AudioWorkletMessage {
  data: Float32Array;
}

interface WSSynthesize extends WSMessageBase {
  type: 'Synthesize';
  text: string;
  emotion: string | null;
  style: string | null;
}

interface WSEndSpeech extends WSMessageBase {
  type: 'EndSpeech';
}

interface WSError extends WSMessageBase {
  type: 'error';
  message: string;
}

interface WSInit extends WSMessageBase {
  type: 'init';
  stt_available: boolean;
}

interface WSTranscript extends WSMessageBase {
  type: 'transcript';
  text: string;
  start_time: number;
  stop_time: number;
}

interface WSTranscription extends WSMessageBase {
  type: 'transcription';
  text: string;
}

type WSMessage = WSAudioData | WSSynthesize | WSEndSpeech | WSError | WSInit | WSTranscript | WSTranscription;

// --- Constants ---
const AUDIO_WORKLET_PROCESSOR_URL = '/audio-worklet-processor.js'; // Path in public directory
const TARGET_SAMPLE_RATE = 24000; // Match backend if possible
const AUDIO_BUFFER_SIZE = 2048; // Smaller buffer size for reduced latency
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_INTERVAL_BASE = 1000; // Base reconnect interval in ms

// --- Error Boundary Component ---
class ErrorBoundary extends Component<{children: ReactNode, fallback?: ReactNode}> {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Application error:", error, errorInfo);
    // Could send to error logging service here
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>Please refresh the page to try again.</p>
          <button onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

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

// --- WebSocket Connection Handler with Reconnection ---
function useWebSocketWithReconnect(wsUrl: string, onMessage: (event: MessageEvent) => void) {
  const [wsState, setWsState] = useState<'connecting' | 'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    setWsState(reconnectAttemptRef.current > 0 ? 'reconnecting' : 'connecting');
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log("WebSocket connection established");
        setWsState('connected');
        reconnectAttemptRef.current = 0; // Reset reconnect attempts on successful connection
      };
      
      ws.onmessage = onMessage;
      
      ws.onerror = (event) => {
        console.error("WebSocket error:", event);
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket closed: ${event.code} ${event.reason}`);
        setWsState('disconnected');
        
        // Only attempt reconnection if not a normal closure
        if (event.code !== 1000 && event.code !== 1001) {
          scheduleReconnect();
        }
      };
    } catch (err) {
      console.error("WebSocket connection failed:", err);
      setWsState('disconnected');
      scheduleReconnect();
    }
  }, [wsUrl, onMessage]);
  
  const scheduleReconnect = useCallback(() => {
    if (reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
      console.log("Maximum reconnect attempts reached");
      return;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    const delay = RECONNECT_INTERVAL_BASE * Math.pow(2, reconnectAttemptRef.current);
    console.log(`Scheduling reconnect in ${delay}ms (attempt ${reconnectAttemptRef.current + 1})`);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectAttemptRef.current += 1;
      connectWebSocket();
    }, delay);
  }, [connectWebSocket]);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close(1000, "Client disconnecting");
    }
    
    wsRef.current = null;
    setWsState('disconnected');
  }, []);
  
  // Connect on component mount and url changes
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      disconnect();
    };
  }, [wsUrl, connectWebSocket, disconnect]);
  
  return { wsRef, wsState, reconnect: connectWebSocket, disconnect };
}

export default function Home() {
  // Wrap the component with ErrorBoundary
  return (
    <ErrorBoundary>
      <HomeContent />
    </ErrorBoundary>
  );
}

function HomeContent() {
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
  const [isRecording, setIsRecording] = useState(false);
  const [isSttProcessing, setIsSttProcessing] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState<string>('');
  const [audioQueue, setAudioQueue] = useState<Float32Array[]>([]);
  const [ttsInput, setTtsInput] = useState<string>(''); // State for TTS input
  const [isPlayingTts, setIsPlayingTts] = useState<boolean>(false); // State to track TTS playback

  // Refs
  const isInitializingAudio = useRef<boolean>(false); // Prevent race conditions
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const sourceNode = useRef<AudioBufferSourceNode | null>(null); // Ref for audio playback source
  const audioChunks = useRef<Blob[]>([]); // Store raw audio chunks

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
    // setTimeout(() => setError(null), 7000);
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
        sampleRate: TARGET_SAMPLE_RATE,
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
              setIsReady(true); // Enable buttons now that context is running
            } else {
              showError("Failed to resume audio context.");
              setIsReady(false);
            }
          } catch (error: unknown) {
            console.error("Error resuming AudioContext:", error);
            showError("Failed to resume audio.");
            setIsReady(false);
          } finally {
            document.body.removeEventListener('click', resume);
            document.body.removeEventListener('keydown', resume);
          }
        };
        document.body.addEventListener('click', resume, { once: true });
        document.body.addEventListener('keydown', resume, { once: true });
      } else {
        // If context is already running, proceed directly
        updateStatus("Audio Ready.");
        setIsReady(true); // Enable buttons
      }

    } catch (err) {
      console.error("Failed to initialize AudioContext:", err);
      showError("AudioContext not supported or initialization failed.");
      setIsReady(false);
    } finally {
      isInitializingAudio.current = false;
    }
  }, [audioContext, updateStatus, log, showError]);

  // Handle WebSocket message parsing for STT
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    log("WebSocket message received.", "debug");
    
    // Handle binary messages (audio data)
    if (audioPlayer && event.data instanceof Blob) {
      log("[WS] Received binary message (Blob)", "debug");
      // Use streaming processing for lower latency
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          audioPlayer.processAudioData(reader.result);
        }
      };
      reader.readAsArrayBuffer(event.data);
      return;
    }
    
    // Handle text messages (JSON)
    if (typeof event.data === 'string') {
      log(`[WS] Received text message: ${event.data}`, "debug");
      try {
        const messageData = JSON.parse(event.data) as WSMessage;
        
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
            setMessages(prev => [...prev, { type: 'bot', text: (messageData as any).text }]);
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

  // Use our custom WebSocket hook with reconnection
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || `ws://localhost:8000/ws`;
  const { wsRef: connectionWsRef, wsState: connectionState, reconnect } = useWebSocketWithReconnect(
    wsUrl,
    handleWebSocketMessage
  );

  // Update based on connection state
  useEffect(() => {
    // Update UI based on connection state
    switch (connectionState) {
      case 'connecting':
        updateStatus('Connecting to server...');
        setIsReady(false);
        break;
      case 'connected':
        updateStatus('Connection successful. Initializing audio...');
        initializeAudio();
        setWebSocket(connectionWsRef.current);
        break;
      case 'disconnected':
        updateStatus('Connection closed.');
        setIsReady(false);
        setIsRecording(false);
        setIsSttProcessing(false);
        setIsPlayingTts(false);
        break;
      case 'reconnecting':
        updateStatus('Attempting to reconnect...');
        setIsReady(false);
        break;
    }
  }, [connectionState, updateStatus, initializeAudio]);

  // --- Send Message Function (moved outside useEffect for clarity) ---
  const sendWebSocketMessage = useCallback((message: WSMessage) => {
    if (connectionWsRef.current && connectionWsRef.current.readyState === WebSocket.OPEN) {
      try {
        const messageString = JSON.stringify(message);
        log(`[WS] Sending: ${messageString}`, "debug");
        connectionWsRef.current.send(messageString);
      } catch (e) {
        console.error("Failed to send WebSocket message:", e);
        showError(`Failed to send message: ${e}`);
      }
    } else {
      showError('Error: WebSocket connection not ready or lost.');
      log('WebSocket is not open. Cannot send message.', 'error');
      
      // Attempt reconnection if disconnected
      if (connectionState === 'disconnected') {
        reconnect();
      }
    }
  }, [log, showError, connectionState, reconnect]);

  // --- Recording Logic (moved stopRecording before startRecording) ---
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop()); // Stop mic access
      console.log('Microphone access stopped.');
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      // Send stop message to the AudioWorklet
      if (processorRef.current.port) {
        processorRef.current.port.postMessage({ type: 'stop' });
      }
      processorRef.current = null;
      console.log('AudioWorkletNode disconnected.');
    }

    setIsRecording(false);
    setIsSttProcessing(false); // Reset STT status
    console.log('Recording stopped');

    // Send EndSpeech signal
    if (connectionWsRef.current && connectionWsRef.current.readyState === WebSocket.OPEN) {
      const endSpeechMsg: WSEndSpeech = {
        type: 'EndSpeech'
      };
      connectionWsRef.current.send(JSON.stringify(endSpeechMsg));
      console.log('Sent EndSpeech signal');
    }
  }, []); // Empty dependency array since it only uses refs

  const startRecording = useCallback(async () => {
    const currentAudioContext = audioContextRef.current; // Capture ref value
    const currentWs = connectionWsRef.current; // Capture ref value

    if (!currentAudioContext) { showError("Audio system not ready."); return; }
    if (currentAudioContext.state === 'suspended') { 
      try { 
        await currentAudioContext.resume(); 
      } catch (error: unknown) { 
        showError("Failed to resume audio."); 
        return; 
      } 
    }
    if (currentAudioContext.state !== 'running') { showError("Audio context not running."); return; }
    if (!currentWs || currentWs.readyState !== WebSocket.OPEN) { showError('Server connection not ready.'); return; }
    if (isRecording) { log("Already recording.", "warn"); return; }

    // Ensure resources are clean before starting
    stopRecording(); 
    await new Promise(resolve => setTimeout(resolve, 50)); // Small delay

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      micStreamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream); // Store dummy recorder for stream track access

      if (!audioContextRef.current) throw new Error("AudioContext lost after getting stream"); // Re-check

      // Add AudioWorklet module 
      try {
        await audioContextRef.current.audioWorklet.addModule(AUDIO_WORKLET_PROCESSOR_URL);
        log("AudioWorklet module loaded.");
      } catch (error: unknown) { 
        // Handle case where module is already loaded
        if (!(error instanceof Error) || !error.message.includes('already been loaded')) {
          console.error('Worklet load err:', error); 
          showError('Audio processor fail.'); 
          stream.getTracks().forEach(track => track.stop()); 
          micStreamRef.current=null; 
          return; 
        }
      }

      const source = audioContextRef.current.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContextRef.current, 'audio-processor', {
        processorOptions: {
          bufferSize: AUDIO_BUFFER_SIZE,
          targetSampleRate: TARGET_SAMPLE_RATE,
          sourceSampleRate: audioContextRef.current.sampleRate
        }
      });
      
      processorRef.current = workletNode;

      // Setup message port with proper typing
      workletNode.port.onmessage = (event: MessageEvent) => {
        if (connectionWsRef.current?.readyState === WebSocket.OPEN) {
          // The AudioWorklet sends Float32Array data
          const audioData = event.data;
          
          const message: WSAudioData = {
            type: 'AudioData', 
            data: Array.from(audioData instanceof Float32Array ? audioData : new Float32Array(0)), 
            sampleRate: audioContextRef.current?.sampleRate || TARGET_SAMPLE_RATE,
            request_codes: false
          };
          sendWebSocketMessage(message);
        } else {
          if (isRecording) { 
            log("WebSocket closed during recording, stopping worklet processing.", "warn");
            stopRecording(); 
          }
        }
      };

      // Add proper error handler
      workletNode.onprocessorerror = (event) => { 
        console.error("AudioWorkletProcessor error:", event);
        showError("Audio processing error."); 
        stopRecording();
      };

      // Connect nodes with null checks
      if (source && workletNode && currentAudioContext) {
        source.connect(workletNode);
        workletNode.connect(currentAudioContext.destination);
        log("Audio nodes connected.");
      }
      
      setIsRecording(true);
      setTranscript('');
      setPartialTranscript('');
      updateStatus('Listening...');
      log('Recording started successfully');

    } catch (error) {
      console.error('Start recording error:', error);
      showError(`Mic start fail: ${error instanceof Error ? error.message : String(error)}`);
      setIsRecording(false);
      stopRecording();
    }
  }, [isRecording, log, updateStatus, showError, sendWebSocketMessage, stopRecording]); // Dependencies for useCallback

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
        
        // Try to load AudioWorklet if not already loaded
        try {
          await audioContext.audioWorklet.addModule(AUDIO_WORKLET_PROCESSOR_URL);
        } catch (error: unknown) {
          // Ignore already loaded errors
          if (!(error instanceof Error) || !error.message?.includes('already been loaded')) {
            throw error;
          }
        }
        
        // Create AudioWorkletNode with optimized buffer size
        processorRef.current = new AudioWorkletNode(audioContext, 'audio-processor', {
          processorOptions: {
            bufferSize: AUDIO_BUFFER_SIZE,
            sampleRate: audioContext.sampleRate,
            targetSampleRate: TARGET_SAMPLE_RATE
          }
        });
        
        // Process audio data with proper typing
        processorRef.current.port.onmessage = (e: MessageEvent<AudioWorkletMessage>) => {
          if (!connectionWsRef.current || connectionWsRef.current.readyState !== WebSocket.OPEN) return;
          
          // Send audio data to the server with proper typing
          const audioData: WSAudioData = {
            type: 'AudioData',
            data: Array.from(e.data.data),
            sampleRate: TARGET_SAMPLE_RATE,
            request_codes: false
          };
          
          sendWebSocketMessage(audioData);
        };
        
        // Add error handler
        processorRef.current.onprocessorerror = (event) => {
          console.error('Audio processing error:', event);
          showError("Audio processing error.");
          stopRecording();
        };
        
        // Connect the processor with null checks
        if (processorRef.current) {
          source.connect(processorRef.current);
          processorRef.current.connect(audioContext.destination);
        }
        
        setIsMicActive(true);
        updateStatus('Microphone active');
      } catch (err) {
        console.error('Error accessing microphone:', err);
        showError(`Microphone error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }, [isMicActive, audioContext, connectionWsRef, sendWebSocketMessage, updateStatus, showError, stopRecording]);

  // Function to clear the transcript
  const clearTranscript = useCallback(() => {
    setTranscript('');
    setRecognizedWords([]);
  }, []);

  // --- Event Handlers ---
  const handleSynthesize = useCallback(() => {
    if (!isReady || !connectionWsRef.current) {
      showError("Not connected, cannot synthesize.");
      return;
    }
    const text = textInputValue.trim();
    if (text) {
      log(`Synthesizing text: ${text}`);
      sendWebSocketMessage({
        type: 'Synthesize',
        text: text,
        character: activeCharacter, // Send selected character
        emotion: null,
        style: null
      } as WSMessage);
      // Example: Add user message to display
      // setMessages(prev => [...prev, { type: 'user', text: text }]);
      // Clear input after sending?
      // setTextInputValue('');
    } else {
      showError('Please enter text to synthesize.');
    }
  }, [isReady, connectionWsRef, textInputValue, activeCharacter, sendWebSocketMessage, log, showError]);

  const handleStopAudio = useCallback(() => {
    if (audioPlayer) {
      audioPlayer.stop();
      // Optionally send a stop message to the server if needed
      // sendWebSocketMessage({ action: 'stop_audio' });
    }
  }, [audioPlayer]);

  // --- Web Audio API Setup & Playback ---
  const playAudioQueue = useCallback(async () => {
    if (!audioContext || isPlaying || audioQueue.length === 0) {
      return;
    }

    setIsPlaying(true);
    const currentAudioContext = audioContext;
    let nextStartTime = currentAudioContext.currentTime;

    // Process the entire queue at once for smoother playback
    const queueToPlay = [...audioQueue];
    setAudioQueue([]); // Clear the queue immediately

    try {
      // Combine all queued Float32Arrays into one
      let totalLength = queueToPlay.reduce((sum, arr) => sum + arr.length, 0);
      if (totalLength === 0) {
        setIsPlaying(false);
        return;
      }

      const combinedData = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of queueToPlay) {
        combinedData.set(chunk, offset);
        offset += chunk.length;
      }

      const buffer = currentAudioContext.createBuffer(
        1, // num channels (mono)
        combinedData.length, // length
        24000 // sample rate (assuming 24kHz from backend)
      );
      buffer.copyToChannel(combinedData, 0); // Fill buffer

      const source = currentAudioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(currentAudioContext.destination);

      source.onended = () => {
        setIsPlaying(false);
        // Check if more audio arrived while playing
        // This check is now less critical as we process the whole queue
        // but kept for potential edge cases or future streaming improvements
        if (audioQueue.length > 0) {
          console.log('More audio arrived, restarting playback...');
          // Debounce or simply recall playAudioQueue
          setTimeout(playAudioQueue, 50); // Small delay
        }
      };
      
      source.start(nextStartTime);
      sourceNode.current = source; // Store the source node

    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlaying(false);
      setAudioQueue([]); // Clear queue on error
    }
  }, [audioQueue, isPlaying]);

  // Trigger playback when the queue has items and not already playing
  useEffect(() => {
    if (audioQueue.length > 0 && !isPlaying) {
      playAudioQueue();
    }
  }, [audioQueue, isPlaying, playAudioQueue]);

  // --- TTS Send Logic ---
  const handleSynthesizeTts = () => {
    if (!ttsInput.trim()) {
      alert('Please enter text to synthesize.');
      return;
    }
    if (connectionWsRef.current && connectionWsRef.current.readyState === WebSocket.OPEN) {
      log('Sending Synthesize message:', 'info');
      sendWebSocketMessage({
        type: 'Synthesize',
        text: ttsInput,
        emotion: null,
        style: null
      } as WSSynthesize);
      // Optionally clear input: setTtsInput(''); 
    } else {
      alert('WebSocket not connected. Cannot send synthesis request.');
    }
  };

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

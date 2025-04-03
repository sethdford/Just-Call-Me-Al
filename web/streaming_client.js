// Constants
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3000;
const AUDIO_PROCESSOR_NAME = 'audio_processor';
const AUDIO_PROCESSOR_PATH = '/static/audio-worklet-processor.js';

// State variables
let isRecording = false;
let microphoneStream = null;
let audioWorkletNode = null;
let audioContext = null;
let socket = null;
let sampleRate = 16000;
let audioQueue = [];
let isPlaying = false;
let connectionRetries = 0;

// UI elements
let textInput = null;
let synthesizeButton = null;
let recordButton = null;
let statusDiv = null;
let errorDiv = null;

// Function declarations
function handleMessageTimeout() {
    helpers.log('Server response timeout reached', 'error');
    helpers.showError('Server did not respond in time.');
    messageTimeoutId = null;
}

function sendSynthesizeRequest(text, options = {}) {
    if (!text || text.trim().length === 0) {
        helpers.showError('Please enter some text to synthesize.');
        return;
    }

    // Clear any existing timeouts
    if (messageTimeoutId) {
        clearTimeout(messageTimeoutId);
        messageTimeoutId = null;
    }

    // Reset audio state
    resetAudioState();

    // Construct message with optional style/emotion
    const message = {
        type: 'synthesize',
        text: text.trim(),
        style: options.style || null,
        emotion: options.emotion || null
    };

    // Set timeout for server response
    messageTimeoutId = setTimeout(() => {
        handleMessageTimeout();
    }, TIMEOUT_CONFIG.messageTimeoutMs);

    // Send the message if socket is connected
    if (connectionState.socket && connectionState.socket.readyState === WebSocket.OPEN) {
        try {
            connectionState.socket.send(JSON.stringify(message));
            helpers.updateStatus('Synthesizing...');
            helpers.clearError();
            updateActivity(); // Update activity timestamp
        } catch (error) {
            helpers.showError(`Failed to send message: ${error.message}`);
            if (messageTimeoutId) {
                clearTimeout(messageTimeoutId);
                messageTimeoutId = null;
            }
        }
    } else {
        helpers.showError('Not connected to server. Please wait for connection or refresh the page.');
    }
}

function processTextMessage(message) {
    try {
        // Clear message timeout since we received a response
        if (messageTimeoutId) {
            clearTimeout(messageTimeoutId);
            messageTimeoutId = null;
        }

        // Handle error messages first
        if (message.Error) {
            helpers.showError(message.Error);
            return;
        }

        // Handle different message types
        if (message.type === 'completed') {
            helpers.updateStatus('Synthesis completed.');
            return;
        }

        if (message.type === 'error') {
            helpers.showError(message.error || 'Unknown server error');
            return;
        }

        if (message.sample_rate) {
            // Audio info message - store sample rate
            audioSampleRate = message.sample_rate;
            helpers.log(`Received audio info: ${message.sample_rate}Hz`, 'info');
            return;
        }

        // For other message types, log for debugging
        helpers.log(`Received message: ${JSON.stringify(message)}`, 'debug');

    } catch (error) {
        helpers.log(`Error processing message: ${error.message}`, 'error');
        helpers.showError(`Failed to process server message: ${error.message}`);
    }
}

// Initialize UI elements if in browser environment
if (typeof document !== 'undefined') {
    textInput = document.getElementById('text-input');
    synthesizeButton = document.getElementById('synthesize-button');
    recordButton = document.getElementById('record-button');
    statusDiv = document.getElementById('status');
    errorDiv = document.getElementById('error');

    // Remove token display div if it exists
    const tokenOutputDiv = document.getElementById('token-output');
    if (tokenOutputDiv) {
        tokenOutputDiv.remove();
    }
}

// Default helper functions
function defaultLog(message, level = 'info') {
    console.log(`[${level}] ${message}`);
    if (level === 'status') {
        const statusElement = document.getElementById('statusDiv');
        if (statusElement) statusElement.textContent = message;
    }
}

function defaultUpdateStatus(message) {
    defaultLog(message, 'status');
}

function defaultShowError(message) {
    defaultLog(message, 'error');
    const errorDiv = document.getElementById('errorDiv');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
}

function defaultClearError() {
    const errorDiv = document.getElementById('errorDiv');
    if (errorDiv) {
        errorDiv.textContent = '';
        errorDiv.style.display = 'none';
    }
}

// WebSocket connection configuration
const CONNECTION_CONFIG = {
    url: `ws://${window.location.hostname}:${window.location.port}/ws`,
    initialBackoffDelayMs: 1000,
    maxBackoffDelayMs: 30000,
    backoffFactor: 1.5,
    firstMessageTimeoutMs: 10000,
    MAX_RETRIES: 5
};

// Connection state
const connectionState = {
    socket: null,
    retryCount: 0,
    firstMessageTimer: null,
    retryTimeout: null
};

// Helper functions that can be overridden for testing
let helpers = {
    log: defaultLog,
    updateStatus: defaultUpdateStatus,
    showError: defaultShowError,
    clearError: defaultClearError
};

// Add timeout configuration
const TIMEOUT_CONFIG = {
    messageTimeoutMs: 15000,
    recordInputTimeoutMs: 60000,
    playOutputTimeoutMs: 30000,
    inactivityTimeoutMs: 300000
};

// Add timeout tracking
let messageTimeoutId = null;
let recordTimeoutId = null;
let playTimeoutId = null;
let inactivityTimeoutId = null;
let lastActivityTime = Date.now();

// Add cleanup function for timeouts
function clearAllTimeouts() {
    if (messageTimeoutId) {
        clearTimeout(messageTimeoutId);
        messageTimeoutId = null;
    }
    if (recordTimeoutId) {
        clearTimeout(recordTimeoutId);
        recordTimeoutId = null;
    }
    if (playTimeoutId) {
        clearTimeout(playTimeoutId);
        playTimeoutId = null;
    }
    if (inactivityTimeoutId) {
        clearTimeout(inactivityTimeoutId);
        inactivityTimeoutId = null;
    }
}

// Add activity tracking
function updateActivity() {
    lastActivityTime = Date.now();
    if (inactivityTimeoutId) {
        clearTimeout(inactivityTimeoutId);
    }
    inactivityTimeoutId = setTimeout(handleInactivity, TIMEOUT_CONFIG.inactivityTimeoutMs);
}

// Extracted logic for the inactivity timeout callback
function handleInactivity() {
    helpers.log('Inactivity timeout reached. Closing connection.', 'warn');
    if (connectionState.socket && connectionState.socket.readyState === WebSocket.OPEN) {
        connectionState.socket.close();
    }
    inactivityTimeoutId = null;
}

// --- Define Helper Functions BEFORE they are used ---

function clearConnectionTimers() {
    if (connectionState.firstMessageTimer) {
        clearTimeout(connectionState.firstMessageTimer);
        connectionState.firstMessageTimer = null;
    }
    if (connectionState.retryTimeout) {
        clearTimeout(connectionState.retryTimeout);
        connectionState.retryTimeout = null;
    }
}

function calculateBackoffDelay(retryAttempt) {
    const delay = CONNECTION_CONFIG.initialBackoffDelayMs * Math.pow(CONNECTION_CONFIG.backoffFactor, retryAttempt);
    return Math.min(delay, CONNECTION_CONFIG.maxBackoffDelayMs);
}

function connectWebSocket(retryAttempt = 0) {
    if (retryAttempt >= CONNECTION_CONFIG.MAX_RETRIES) {
        helpers.updateStatus('Maximum retry attempts reached. Please refresh to try again.');
        helpers.showError('Maximum retry attempts reached. Please refresh the page to try again.');
        return;
    }

    if (connectionState.socket && (connectionState.socket.readyState === WebSocket.CONNECTING || connectionState.socket.readyState === WebSocket.OPEN)) {
        helpers.log(`Connection attempt ${retryAttempt + 1}: Already connected or connecting.`, 'info');
        return;
    }

    clearConnectionTimers();
    clearAllTimeouts();
    connectionState.retryCount = retryAttempt;

    try {
        helpers.updateStatus(retryAttempt > 0 ? `Reconnecting (attempt ${retryAttempt + 1})...` : 'Connecting...');
        connectionState.socket = new WebSocket(CONNECTION_CONFIG.url);
        const currentSocket = connectionState.socket;

        currentSocket.onopen = () => {
            helpers.clearError();
            helpers.updateStatus('Connected. Ready to synthesize.');
            connectionState.retryCount = 0;
            
            // Send an initial ping to verify connection
            try {
                currentSocket.send(JSON.stringify({ type: 'ping' }));
            } catch (e) {
                helpers.log('Failed to send initial ping', 'warn');
            }
            
            // Set up first message timeout
            connectionState.firstMessageTimer = setTimeout(() => {
                helpers.log('First message timeout reached', 'error');
                if (currentSocket && currentSocket.readyState === WebSocket.OPEN) {
                    currentSocket.close();
                }
            }, CONNECTION_CONFIG.firstMessageTimeoutMs);

            // Start inactivity tracking
            updateActivity();
        };

        currentSocket.onclose = (event) => {
            const { shouldRetry, nextRetryAttempt } = handleWebSocketClose(event, retryAttempt);
            if (shouldRetry) {
                const delay = calculateBackoffDelay(retryAttempt);
                connectionState.retryTimeout = setTimeout(() => {
                    connectWebSocket(nextRetryAttempt);
                }, delay);
            }
        };

        currentSocket.onerror = (error) => {
            helpers.log(`WebSocket error: ${error.message || 'Unknown error'}`, 'error');
        };

        // Add message handler
        currentSocket.onmessage = (event) => {
            // Clear first message timer on any message
            if (connectionState.firstMessageTimer) {
                clearTimeout(connectionState.firstMessageTimer);
                connectionState.firstMessageTimer = null;
            }
            updateActivity();
            
            // Process the message
            try {
                if (event.data instanceof Blob) {
                    // Handle binary audio data
                    // Initialize audio context if needed
                    if (!audioContext) {
                        try {
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            helpers.log("[Audio] Context initialized at sample rate:", audioContext.sampleRate);
                        } catch (error) {
                            helpers.log(`Error initializing AudioContext: ${error.message}`, 'error');
                            return;
                        }
                    }
                    
                    // Convert Blob to ArrayBuffer for processing
                    event.data.arrayBuffer().then(arrayBuffer => {
                        processAudioData(arrayBuffer);
                    }).catch(error => {
                        helpers.log(`Error processing audio data: ${error.message}`, 'error');
                    });
                } else {
                    // Handle text messages (JSON)
                    const message = JSON.parse(event.data);
                    processTextMessage(message);
                }
            } catch (error) {
                helpers.log(`Error processing message: ${error.message}`, 'error');
            }
        };

    } catch (error) {
        helpers.showError(`Failed to create WebSocket: ${error.message}`);
        const delay = calculateBackoffDelay(retryAttempt);
        connectionState.retryTimeout = setTimeout(() => {
            connectWebSocket(retryAttempt + 1);
        }, delay);
    }
}

// Extracted logic for handling WebSocket close events
function handleWebSocketClose(event, currentRetryAttempt) {
    clearConnectionTimers();
    clearAllTimeouts(); // Clear activity timers on close

    if (event.code === 1000 || event.code === 1001) { // Normal closure or going away
        helpers.updateStatus('Disconnected.');
        return { shouldRetry: false, reason: 'normal' };
    }

    // Handle max retries
    if (currentRetryAttempt >= CONNECTION_CONFIG.MAX_RETRIES) {
        helpers.updateStatus('Maximum retry attempts reached. Please refresh to try again.');
        helpers.showError('Maximum retry attempts reached. Please refresh the page to try again.');
        return { shouldRetry: false, reason: 'max_retries' };
    }

    // Calculate retry details
    const nextRetryAttempt = currentRetryAttempt + 1;
    const delay = calculateBackoffDelay(nextRetryAttempt);
    helpers.log(`WebSocket closed (Code: ${event.code}). Retry attempt ${nextRetryAttempt} in ${delay}ms...`);
    helpers.updateStatus(`Connection lost. Reconnecting (attempt ${nextRetryAttempt})...`);

    return { shouldRetry: true, delay: delay, nextRetryAttempt: nextRetryAttempt };
}

// Restore audio playback functions
function playAudio(float32Data, sampleRate) {
    updateActivity(); 

    if (!audioContext) {
        helpers.showError("AudioContext not available for playback.");
        return;
    }

    const audioBuffer = audioContext.createBuffer(
        1, // Number of channels (assuming mono)
        float32Data.length,
        sampleRate // Use sample rate received from server
    );

    audioBuffer.getChannelData(0).set(float32Data);

    audioQueue.push(audioBuffer);
    if (!isPlaying) {
        if (playTimeoutId) {
            clearTimeout(playTimeoutId);
        }
        playTimeoutId = setTimeout(() => {
            helpers.log('Audio playback timeout reached', 'warn');
            helpers.showError('Audio playback took too long to complete.');
            resetAudioState(); 
        }, TIMEOUT_CONFIG.playOutputTimeoutMs);
        playNextChunk();
    }
}

function playNextChunk() {
    // Check if playback was active *before* this check
    const wasPlaying = isPlaying;

    if (audioQueue.length === 0) {
        isPlaying = false;
        // Only clear timeout if playback was actually active and just finished
        if (wasPlaying && playTimeoutId) {
            clearTimeout(playTimeoutId);
            playTimeoutId = null;
        }
        helpers.updateStatus("Playback finished. Ready.");
        return;
    }

    if (!audioContext) {
        helpers.showError("AudioContext lost during playback.");
        isPlaying = false;
        audioQueue = [];
        return;
    }

    isPlaying = true;
    helpers.updateStatus("Playing audio...");

    const bufferToPlay = audioQueue.shift();
    const source = audioContext.createBufferSource();
    source.buffer = bufferToPlay;
    source.connect(audioContext.destination);
    source.onended = playNextChunk;
    source.start();
}

// --- Message Handling --- 
function handleServerMessage(data) {
    // Implement message handling logic here
    console.log('Received message:', data);
}

// --- UI Initialization and Event Binding ---
function initializeUI() {
    helpers.log("Initializing UI elements and listeners...", "info");
    
    // Get UI elements
    textInput = document.getElementById('text-input');
    synthesizeButton = document.getElementById('synthesize-button');
    recordButton = document.getElementById('record-button');
    statusDiv = document.getElementById('status');
    errorDiv = document.getElementById('error');
    
    if (!textInput || !synthesizeButton) {
        helpers.log("Could not find synthesize button or text input element.", "error");
        return;
    }

    // Initialize WebSocket connection
    connectWebSocket();

    // Add button listeners
    synthesizeButton.addEventListener('click', () => {
        // Initialize audio context on user interaction (browsers require this)
        if (!audioContext) {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                helpers.log(`[Audio] Context initialized at sample rate: ${audioContext.sampleRate}Hz`, 'info');
            } catch (error) {
                helpers.log(`Error initializing AudioContext: ${error.message}`, 'error');
            }
        }
        
        if (!connectionState.socket || connectionState.socket.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            return; // Wait for connection before sending
        }
        const text = textInput.value.trim();
        if (text) {
            sendSynthesizeRequest(text);
        }
    });

    if (recordButton) {
        recordButton.addEventListener('click', toggleRecording);
    }
}

// Start initialization when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeUI);
    } else {
        // DOM already loaded, initialize immediately
        initializeUI();
    }
}

// --- Script Execution Start ---

// Check if running in a browser-like environment to attach UI listeners
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    // Use DOMContentLoaded to ensure elements are ready before attaching listeners
    document.addEventListener('DOMContentLoaded', initializeUI);
} else {
    helpers.log('Not attaching UI listeners in non-browser environment.', 'debug');
}

// Attempt initial connection regardless of environment
connectWebSocket();

// Update exports
if (typeof window !== 'undefined') {
    window.streamingClient = {
        log: defaultLog,
        updateStatus: defaultUpdateStatus,
        showError: defaultShowError,
        clearError: defaultClearError,
        clearConnectionTimers,
        calculateBackoffDelay,
        connectWebSocket,
        CONNECTION_CONFIG,
        TIMEOUT_CONFIG,
        connectionState,
        clearAllTimeouts,
        setHelpers: (newHelpers) => {
            helpers = { ...helpers, ...newHelpers };
        },
        resetAudioState,
        // Internal state getters
        get messageTimeoutId() { return messageTimeoutId; },
        get recordTimeoutId() { return recordTimeoutId; },
        get playTimeoutId() { return playTimeoutId; },
        get inactivityTimeoutId() { return inactivityTimeoutId; },
        get lastActivityTime() { return lastActivityTime; },
        get audioQueue() { return audioQueue; },
        get isPlaying() { return isPlaying; },
        // Functions
        updateActivity,
        playAudio,
        playNextChunk,
        sendSynthesizeRequest,
        handleWebSocketClose,
        handleInactivity,
        handleMessageTimeout,
        processAudioData,
        // Testing utilities
        _setInternalStateForTesting: (newState) => {
            if (newState.audioContext !== undefined) audioContext = newState.audioContext;
            if (newState.messageTimeoutId !== undefined) messageTimeoutId = newState.messageTimeoutId;
            if (newState.recordTimeoutId !== undefined) recordTimeoutId = newState.recordTimeoutId;
            if (newState.playTimeoutId !== undefined) playTimeoutId = newState.playTimeoutId;
            if (newState.inactivityTimeoutId !== undefined) inactivityTimeoutId = newState.inactivityTimeoutId;
        }
    };
}

class AudioPlayer {
  constructor() {
    this.audioContext = null;
    this.processor = null;
    this.bufferConfig = {
      minBufferMs: 100,
      maxBufferMs: 1000,
      targetBufferMs: 300,
      bufferPaddingMultiplier: 1.5,
      poorRttThresholdMs: 200
    };
    this.networkMetrics = {
      quality: 1.0,
      avgRtt: 0,
      rttVar: 0,
      underruns: 0,
      bufferHealth: 1.0
    };
    this.statusCallbacks = new Set();
  }

  async initAudioContext() {
    if (this.audioContext) return;
    
    this.audioContext = new AudioContext();
    await this.audioContext.audioWorklet.addModule('audio_processor.js');
    
    this.processor = new AudioWorkletNode(this.audioContext, 'StreamProcessor');
    this.processor.connect(this.audioContext.destination);
    
    // Configure adaptive buffering
    this.processor.port.postMessage({
      event: 'configure',
      config: this.bufferConfig
    });
    
    // Handle messages from processor
    this.processor.port.onmessage = (event) => {
      const msg = event.data;
      
      if (msg.event === 'networkQuality') {
        this.networkMetrics = msg.metrics;
        this.networkMetrics.quality = msg.quality;
        this.updateStatus('Network quality: ' + (msg.quality * 100).toFixed(0) + '%');
        
        // Adjust buffer settings based on network quality
        if (msg.quality < 0.5) {
          // Poor network conditions - increase buffer
          this.bufferConfig.targetBufferMs = Math.min(
            this.bufferConfig.maxBufferMs,
            this.bufferConfig.targetBufferMs * 1.2
          );
        } else if (msg.quality > 0.8 && this.networkMetrics.underruns === 0) {
          // Good network conditions - decrease buffer
          this.bufferConfig.targetBufferMs = Math.max(
            this.bufferConfig.minBufferMs,
            this.bufferConfig.targetBufferMs * 0.9
          );
        }
        
        // Update processor config
        this.processor.port.postMessage({
          event: 'configure',
          config: this.bufferConfig
        });
      } else if (msg.event === 'bufferUnderrun') {
        this.updateStatus('Buffer underrun detected');
        // Increase target buffer on underrun
        this.bufferConfig.targetBufferMs = Math.min(
          this.bufferConfig.maxBufferMs,
          this.bufferConfig.targetBufferMs * 1.2
        );
        this.processor.port.postMessage({
          event: 'configure',
          config: this.bufferConfig
        });
      } else if (msg.event === 'bufferOverrun') {
        // Optional: Decrease buffer size on consistent overruns
        if (this.networkMetrics.quality > 0.7) {
          this.bufferConfig.targetBufferMs = Math.max(
            this.bufferConfig.minBufferMs,
            this.bufferConfig.targetBufferMs * 0.9
          );
          this.processor.port.postMessage({
            event: 'configure',
            config: this.bufferConfig
          });
        }
      }
      // ... handle other existing events ...
    };
  }

  updateStatus(message) {
    for (const callback of this.statusCallbacks) {
      callback(message);
    }
  }

  addStatusCallback(callback) {
    this.statusCallbacks.add(callback);
  }

  removeStatusCallback(callback) {
    this.statusCallbacks.delete(callback);
  }
} 

// Function to reset audio state for testing
function resetAudioState() {
    audioQueue = [];
    isPlaying = false;
} 

// --- Audio Capture State --- (New)
// let microphoneStream = null;
// let audioWorkletNode = null;
// let isRecording = false;
// const AUDIO_PROCESSOR_NAME = 'audio_processor';
// const AUDIO_PROCESSOR_PATH = '/static/audio-worklet-processor.js'; // Updated path

// --- Initialization Functions --- (New / Modified)

// Ensure AudioContext is initialized (lazily)
async function ensureAudioContext() {
    if (!audioContext) {
        try {
            helpers.log('Initializing AudioContext...', 'info');
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            // Check state and resume if needed (for browser autoplay policies)
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            helpers.log(`AudioContext initialized. State: ${audioContext.state}, Sample Rate: ${audioContext.sampleRate}`, 'info');
        } catch (err) {
            helpers.showError(`Error initializing AudioContext: ${err.message}`);
            console.error(err);
            throw err; // Re-throw to prevent further audio operations
        }
    }
    return audioContext;
}

// Load the AudioWorklet processor
async function loadAudioWorklet(context) {
    if (!context) {
        throw new Error("AudioContext not available for loading worklet.");
    }
    try {
        helpers.log(`Loading AudioWorklet processor from ${AUDIO_PROCESSOR_PATH}...`, 'info');
        await context.audioWorklet.addModule(AUDIO_PROCESSOR_PATH);
        helpers.log('AudioWorklet processor loaded successfully.', 'info');
    } catch (err) {
        helpers.showError(`Failed to load AudioWorklet processor: ${err.message}`);
        console.error(err);
        throw err; // Re-throw
    }
}

// --- Microphone and Recording Functions --- (New)

async function startMicrophoneAndWorklet() {
    try {
        const context = await ensureAudioContext();
        await loadAudioWorklet(context);

        helpers.log("Requesting microphone access...", 'info');
        microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        helpers.log("Microphone access granted.", 'info');

        const source = context.createMediaStreamSource(microphoneStream);
        audioWorkletNode = new AudioWorkletNode(context, AUDIO_PROCESSOR_NAME);
        helpers.log("AudioWorkletNode created.", 'info');

        // Handle messages (audio chunks) from the worklet
        audioWorkletNode.port.onmessage = (event) => {
            if (event.data.event === 'chunk') {
                const monoData = event.data.data.mono; // Use mono data (ArrayBuffer)
                if (connectionState.socket && connectionState.socket.readyState === WebSocket.OPEN) {
                    // Send the raw ArrayBuffer over WebSocket
                    connectionState.socket.send(monoData);
                    // console.log(`Sent ${monoData.byteLength} bytes of audio data.`); // Optional debug log
                    updateActivity();
                } else {
                    // Buffer or discard if socket not open?
                    // console.warn("WebSocket not open, discarding audio chunk.");
                }
            } else if (event.data.event === 'error') {
                helpers.showError(`AudioWorklet Error: ${event.data.data.message}`);
                console.error('AudioWorklet Error:', event.data.data);
                stopRecording(); // Stop recording on worklet error
            } else if (event.data.event === 'receipt') {
                // Handle receipts from worklet commands if needed
                 console.log('Received receipt from worklet:', event.data.id);
            }
        };

        audioWorkletNode.port.onerror = (error) => {
            helpers.showError(`AudioWorklet port error: ${error.message}`);
            console.error('AudioWorklet port error:', error);
            stopRecording();
        };

        // Connect microphone source to the worklet node
        source.connect(audioWorkletNode);
        // Optional: Connect worklet to destination for monitoring (can be loud/feedback)
        // audioWorkletNode.connect(context.destination);
        helpers.log("Microphone connected to AudioWorkletNode.", 'info');

        return true; // Indicate success

    } catch (err) {
        helpers.showError(`Error starting microphone/worklet: ${err.message}`);
        console.error(err);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
             helpers.showError("Microphone access denied. Please allow microphone access in your browser settings.");
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
             helpers.showError("No microphone found. Please ensure a microphone is connected and enabled.");
        }
        // Clean up partial resources if error occurred
        stopMicrophone(); 
        return false; // Indicate failure
    }
}

function startRecording() {
    if (!audioWorkletNode) {
        helpers.showError("Audio processing not initialized.");
        return;
    }
    if (isRecording) {
        helpers.log("Already recording.", 'warn');
        return;
    }
    try {
        audioWorkletNode.port.postMessage({ event: 'clear' }); // Clear any old buffer
        audioWorkletNode.port.postMessage({ event: 'start' });
        isRecording = true;
        helpers.updateStatus("Recording...");
        if (recordButton) recordButton.textContent = "Stop Recording";
        // Start recording timeout
        clearTimeout(recordTimeoutId);
        recordTimeoutId = setTimeout(() => {
             helpers.log('Recording timeout reached', 'warn');
             helpers.showError('Recording stopped due to inactivity.');
             stopRecording();
        }, TIMEOUT_CONFIG.recordInputTimeoutMs);
        updateActivity();
    } catch (err) {
        helpers.showError(`Failed to start recording: ${err.message}`);
        console.error(err);
    }
}

// Function to stop recording
async function stopRecording() {
    if (typeof isRecording === 'undefined' || !isRecording) {
        return;
    }
    
    try {
        // Send stop signal BEFORE stopping local resources
        if (connectionState.socket && connectionState.socket.readyState === WebSocket.OPEN) {
            connectionState.socket.send(JSON.stringify({ type: 'stop_audio' }));
            helpers.log('Sent stop_audio signal to server.', 'info');
        }

        isRecording = false;
        if (microphoneStream) {
            microphoneStream.getTracks().forEach(track => track.stop());
            microphoneStream = null;
        }
        if (audioWorkletNode) {
            audioWorkletNode.disconnect();
            audioWorkletNode = null;
        }
        helpers.log('Recording stopped.', 'info');
    } catch (error) {
        helpers.showError(`Error stopping recording: ${error.message}`);
        console.error('Error stopping recording:', error);
    }
}

// Stop microphone stream and disconnect worklet
function stopMicrophone() {
    stopRecording(); // Ensure recording command is sent

    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
        helpers.log("Microphone stream stopped.", 'info');
    }
    if (audioWorkletNode) {
        // Disconnect nodes to stop processing
        try {
           audioWorkletNode.disconnect(); 
        } catch (e) {
            console.warn("Error disconnecting worklet node:", e);
        }
        audioWorkletNode = null;
        helpers.log("AudioWorkletNode disconnected and removed.", 'info');
    }
    // Optionally close AudioContext if no longer needed for playback
    // if (audioContext && !isPlaying && audioQueue.length === 0) {
    //     audioContext.close().then(() => { audioContext = null; console.log('AudioContext closed.'); });
    // }
}

// Toggle recording state
async function toggleRecording() {
    // Ensure websocket is connected first
    if (!connectionState.socket || connectionState.socket.readyState !== WebSocket.OPEN) {
         helpers.showError("Please connect to WebSocket first.");
         // Optionally try to connect here: connectWebSocket();
         return;
    }

    if (!isRecording) {
        // Initialize microphone and worklet if not already done
        if (!microphoneStream || !audioWorkletNode) {
            const success = await startMicrophoneAndWorklet();
            if (!success) return; // Don't proceed if init failed
        }
        startRecording();
    } else {
        stopRecording();
    }
}

// Modify WebSocket onclose handler to stop microphone
function handleWebSocketClose(event, retryAttempt) {
    clearConnectionTimers();
    clearAllTimeouts();
    stopMicrophone(); // Stop mic when connection closes

    // ... (rest of the existing close handling logic)
    const wasClean = event.wasClean;
    const code = event.code;
    const reason = event.reason || 'No reason provided';

    helpers.log(`WebSocket closed. Code: ${code}, Reason: "${reason}", Clean: ${wasClean}`, 'warn');
    connectionState.socket = null;

    if (wasClean || code === 1000 || code === 1005) {
        helpers.updateStatus('Disconnected.');
        return { shouldRetry: false };
    } else if (code === 1001) { // Going Away
        helpers.updateStatus('Server is going away. Disconnected.');
        return { shouldRetry: false };
    } else {
        helpers.showError(`WebSocket error (Code: ${code}): ${reason}`);
        if (retryAttempt < CONNECTION_CONFIG.MAX_RETRIES) {
            const delay = calculateBackoffDelay(retryAttempt);
            helpers.updateStatus(`Connection lost. Retrying in ${Math.round(delay / 1000)}s...`);
            return { shouldRetry: true, nextRetryAttempt: retryAttempt + 1 };
        } else {
            helpers.updateStatus('Connection failed after multiple retries.');
            return { shouldRetry: false };
        }
    }
}

// --- Main Initialization --- (Ensure DOM is ready)
// Make sure this runs after the functions it calls are defined
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        helpers.log("Initializing UI elements and listeners...", "info");
        
        // Get UI elements
        const textInput = document.getElementById('text-input');
        const synthesizeButton = document.getElementById('synthesize-button');
        const recordButton = document.getElementById('record-button');
        
        // Verify required elements exist
        if (!textInput || !synthesizeButton) {
            helpers.log("Could not find synthesize button or text input element.", "error");
            return;
        }

        // Initialize WebSocket connection
        connectWebSocket();

        // Add button listeners
        synthesizeButton.addEventListener('click', () => {
            if (!connectionState.socket || connectionState.socket.readyState !== WebSocket.OPEN) {
                connectWebSocket();
            }
            if (textInput.value.trim()) {
                sendSynthesizeRequest(textInput.value);
            }
        });

        if (recordButton) {
            recordButton.addEventListener('click', toggleRecording);
        }
    });
} else {
    // Environment without DOM (e.g., testing)
    helpers.log("DOM not found. Skipping UI setup.", "info");
}

// --- Optional: Export for testing ---
// if (typeof module !== 'undefined' && module.exports) {
//     module.exports = {
//         connectWebSocket,
//         sendSynthesizeRequest,
//         handleWebSocketClose,
//         ensureAudioContext,
//         loadAudioWorklet,
//         startMicrophoneAndWorklet,
//         startRecording,
//         stopRecording,
//         stopMicrophone,
//         toggleRecording,
//         helpers, // Expose helpers for mocking
//         connectionState, // Expose state for inspection
//         // Add other exports as needed
//         setHelpers: (newHelpers) => { helpers = { ...helpers, ...newHelpers }; }
//     };
// } 

// --- Audio Processing for Binary WebSocket Messages ---
// Process audio data from WebSocket binary message
function processAudioData(data) {
  helpers.log('[Audio] Received binary audio data chunk', 'info');
  
  // Convert binary WebSocket message to audio data (Int16 PCM)
  const audioData = new Int16Array(data);
  helpers.log(`[Audio] Created Int16Array with ${audioData.length} samples`, 'debug');
  
  // Create audio buffer (mono channel, 24kHz sample rate for Sesame's CSM)
  const buffer = audioContext.createBuffer(1, audioData.length, 24000);
  const channel = buffer.getChannelData(0);
  
  // Convert Int16 values to Float32 (Web Audio API format)
  for (let i = 0; i < audioData.length; i++) {
    // Scale from [-32768, 32767] to [-1.0, 1.0]
    channel[i] = audioData[i] / 32768;
  }
  
  // Add to queue and play
  audioQueue.push(buffer);
  helpers.log(`[Audio] Added buffer to queue, queue length: ${audioQueue.length}`, 'debug');
  
  // Start playback if not already playing
  if (!isPlaying) {
    helpers.log('[Audio] Starting playback', 'debug');
    playNextChunk();
  }
} 

let audioPlayer;
let webSocket;
let sessionId;
let micStream;        // Added for microphone input
let isListening = false; // Added to track recording state

// Function to generate a unique session ID
function generateSessionId() {
    // Reuse existing implementation if present, otherwise:
    return 'session_' + Math.random().toString(36).substr(2, 9);
}

// Function to update status display
function updateStatus(message) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.textContent = message;
    } else {
        console.log("Status:", message); // Fallback
    }
}

// Ensure AudioContext is created (potentially moved inside connectWebSocket or init)
function ensureAudioContext() {
    if (!audioContext) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            console.log("AudioContext created with sample rate:", audioContext.sampleRate);
            // Handle state changes (e.g., suspended)
            audioContext.onstatechange = () => {
                console.log("AudioContext state changed:", audioContext.state);
                updateStatus(`AudioContext state: ${audioContext.state}`);
            };
        } catch (e) {
            console.error("Error creating AudioContext:", e);
            updateStatus("Error: Failed to create AudioContext. Check browser compatibility.");
            throw new Error("Failed to create AudioContext");
        }
    }
    // Resume context if suspended (requires user interaction)
    if (audioContext.state === 'suspended') {
        console.log("AudioContext is suspended, attempting to resume...");
        audioContext.resume().then(() => {
             console.log("AudioContext resumed successfully.");
             updateStatus("AudioContext resumed.");
        }).catch(err => {
             console.error("Failed to resume AudioContext:", err);
             updateStatus("Error: Click the page to enable audio.");
        });
    }
    return audioContext;
}

// WebSocket connection function (Ensure it initializes AudioContext and Player)
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        sessionId = generateSessionId();
        webSocket = new WebSocket(`${wsUrl}?sessionId=${sessionId}`);

        webSocket.onopen = () => {
            updateStatus('WebSocket Connected.');
            console.log('WebSocket connection established');
            try {
                ensureAudioContext(); // Ensure AudioContext is ready
                if (!audioPlayer) {
                    audioPlayer = new AudioPlayer(audioContext);
                    console.log("AudioPlayer initialized.");
                }
                resolve();
            } catch (e) {
                reject(e); // Reject if AudioContext creation fails
            }
        };

        webSocket.onmessage = handleWebSocketMessage;

        webSocket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            updateStatus('WebSocket Error. Check console.');
            reject(error);
        };

        webSocket.onclose = (event) => {
            handleWebSocketClose(event);
            // Update UI to reflect disconnected state
            const startConversationButton = document.getElementById('startConversation');
            const stopConversationButton = document.getElementById('stopConversation');
            const synthesizeButton = document.getElementById('synthesizeButton'); // Assuming this exists
            const stopButton = document.getElementById('stopButton'); // Assuming this exists

            if (startConversationButton) startConversationButton.disabled = true;
            if (stopConversationButton) stopConversationButton.disabled = true;
            if (synthesizeButton) synthesizeButton.disabled = true;
            if (stopButton) stopButton.disabled = true;

            updateStatus(`WebSocket Closed: Code=${event.code}. ${event.reason || 'No reason provided.'} Please refresh.`);
            reject(`WebSocket Closed: Code=${event.code}, Reason=${event.reason}`);
        };
    });
}

// Handle incoming WebSocket messages (Combined Handler)
function handleWebSocketMessage(event) {
    if (event.data instanceof Blob) {
        // Binary message - audio data for playback
        event.data.arrayBuffer().then(buffer => {
            processAudioData(buffer); // Process using AudioPlayer
        }).catch(error => {
            console.error("Error reading Blob data:", error);
            updateStatus("Error processing audio data.");
        });
    } else {
        // Text message - JSON control messages or transcriptions
        try {
            const message = JSON.parse(event.data);
            console.log('Received Text Message:', message);

            if (message.type === 'status') {
                updateStatus(`Server: ${message.message}`);
            } else if (message.type === 'audio_info') {
                console.log('Audio info received:', message);
            } else if (message.type === 'error') {
                console.error('Server error:', message.message);
                updateStatus(`Server Error: ${message.message}`);
            } else if (message.type === 'transcription') {
                displayTranscription(message.text); // Display received transcription
            } else {
                console.warn('Received unknown message type:', message.type);
            }
        } catch (e) {
            // console.error('Failed to parse WebSocket message or unknown format:', event.data, e);
             // It might just be a plain text status message from older parts of the code
             console.log("Received plain text message:", event.data);
             // Optionally display plain text messages if needed
             // displayTranscription(event.data); // Or handle differently
        }
    }
}

// Function to send messages via WebSocket
function sendWebSocketMessage(message) {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        webSocket.send(JSON.stringify(message));
    } else {
        console.error('WebSocket is not open. Cannot send message:', message);
        updateStatus('Error: WebSocket connection lost. Please refresh.');
        // Attempt to reconnect or disable relevant UI elements
    }
}

// --- Audio Playback Logic ---

// Assume AudioPlayer class exists and is defined correctly above or below
class AudioPlayer {
    constructor(audioContext) {
        this.audioContext = audioContext;
        this.audioQueue = [];
        this.isPlaying = false;
        this.startTime = 0;
        this.bufferSource = null;
        console.log("AudioPlayer instantiated.");
    }

    async processAudioData(arrayBuffer) {
        if (!this.audioContext || this.audioContext.state !== 'running') {
             console.warn("AudioContext not running, cannot process audio data. State:", this.audioContext?.state);
             // Try resuming, might require user interaction
             if (this.audioContext) await this.audioContext.resume();
             if (this.audioContext?.state !== 'running') {
                 updateStatus("Audio playback paused. Click page to resume.");
                 // Queue data or handle error? For now, just log and return.
                 console.error("Cannot process audio data - AudioContext not running.");
                 return;
             }
        }

        // Basic validation
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
            console.warn("Received empty or invalid audio data.");
            return;
        }
        if (arrayBuffer.byteLength % 2 !== 0) {
            console.error("Received audio data with odd byte length:", arrayBuffer.byteLength);
            return;
        }

        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        try {
            const audioBuffer = this.audioContext.createBuffer(
                1, float32Array.length, this.audioContext.sampleRate
            );
            audioBuffer.copyToChannel(float32Array, 0);

            this.audioQueue.push(audioBuffer);
            if (!this.isPlaying) {
                this.playNextChunk();
            }
        } catch (e) {
            console.error("Error creating or processing audio buffer:", e);
        }
    }

    playNextChunk() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            // console.log("Audio queue empty.");
            return;
        }
        if (this.audioContext.state !== 'running') {
            this.isPlaying = false;
            console.warn("AudioContext not running, pausing playback.");
            updateStatus("Audio playback paused. Click page to resume.");
            return; // Don't proceed if context isn't running
        }


        this.isPlaying = true;
        const audioBuffer = this.audioQueue.shift();
        this.bufferSource = this.audioContext.createBufferSource();
        this.bufferSource.buffer = audioBuffer;
        this.bufferSource.connect(this.audioContext.destination);

        const currentTime = this.audioContext.currentTime;
        const startTime = (this.startTime <= currentTime) ? currentTime : this.startTime;

        this.bufferSource.start(startTime);
        // console.log(`Playing audio chunk. Start: ${startTime.toFixed(3)}, Duration: ${audioBuffer.duration.toFixed(3)}`);

        this.startTime = startTime + audioBuffer.duration;

        this.bufferSource.onended = () => {
             // console.log("Audio chunk finished.");
             // Ensure stop() wasn't called in the meantime
             if (this.isPlaying) {
                 this.playNextChunk();
             }
        };
    }

    stop() {
        console.log("AudioPlayer stopping playback...");
        this.isPlaying = false; // Signal that playback should stop
        if (this.bufferSource) {
             try {
                 this.bufferSource.onended = null; // Remove listener to prevent auto-play next
                 this.bufferSource.stop();
                 console.log("Current buffer source stopped.");
             } catch (e) {
                 console.warn("Error stopping buffer source:", e); // Might already be stopped
             }
             this.bufferSource = null;
        }
        this.audioQueue = []; // Clear pending chunks
        this.startTime = 0;   // Reset schedule time
        console.log("Audio queue cleared.");
    }
}
// Ensure audioPlayer is instantiated correctly after AudioContext is ready


// Function to process received audio data (calls AudioPlayer)
function processAudioData(arrayBuffer) {
    if (!audioPlayer) {
        console.error("AudioPlayer not initialized.");
        updateStatus("Error: Audio system not ready.");
        return;
    }
    // Ensure context is running before processing
    if (audioContext && audioContext.state === 'running') {
        audioPlayer.processAudioData(arrayBuffer);
    } else if (audioContext && audioContext.state === 'suspended') {
        console.warn("AudioContext suspended. Attempting to resume for processing...");
        audioContext.resume().then(() => {
            if (audioContext.state === 'running') {
                console.log("AudioContext resumed. Processing data...");
                audioPlayer.processAudioData(arrayBuffer);
            } else {
                 console.error("Failed to resume AudioContext for processing.");
                 updateStatus("Audio paused. Click page to resume processing.");
            }
        }).catch(e => {
             console.error("Error resuming AudioContext:", e);
             updateStatus("Error resuming audio. Click page.");
        });
    } else {
         console.error("AudioContext not ready or in unexpected state:", audioContext?.state);
         updateStatus("Error: AudioContext not ready.");
    }
}

// Function to send synthesize request (ensure it stops player first)
function sendSynthesizeRequest() {
    const textInput = document.getElementById('textInput'); // Ensure ID matches HTML
    const text = textInput ? textInput.value.trim() : '';

    if (text && webSocket && webSocket.readyState === WebSocket.OPEN) {
        console.log(`Sending synthesize request for text: "${text}"`);
        updateStatus(`Synthesizing: "${text}"`);
        // Stop any ongoing playback before starting new synthesis
        if (audioPlayer) {
            audioPlayer.stop();
        }
        sendWebSocketMessage({
            type: 'synthesize',
            text: text,
            session_id: sessionId
        });
    } else if (!text) {
        updateStatus('Please enter text to synthesize.');
    } else {
        updateStatus('WebSocket not connected. Please wait or refresh.');
        console.error('WebSocket not connected.');
    }
}

// Function to send stop audio signal
function sendStopAudio() {
    // This stops server-side generation
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        console.log("Sending stop audio signal to server");
        sendWebSocketMessage({
            type: 'stop_audio',
            session_id: sessionId
        });
        updateStatus("Stop signal sent.");
    } else {
        updateStatus('WebSocket not connected.');
        console.error('WebSocket not connected.');
    }
    // Also stop client-side playback immediately
    if (audioPlayer) {
        console.log("Stopping client-side audio playback.");
        audioPlayer.stop();
    }
}

// --- Conversation / Microphone Logic ---

// Load audio worklet processor
async function loadAudioWorklet() {
  if (!audioContext) {
      console.error("AudioContext not available for loading worklet.");
      updateStatus("Error: AudioContext not initialized.");
      return false; // Indicate failure
  }
   // Resume context if needed before loading
   if (audioContext.state === 'suspended') {
       await audioContext.resume().catch(e => {
           console.warn("Could not resume AudioContext before loading worklet:", e);
           updateStatus("Click page to enable audio features.");
           // Don't necessarily stop here, but loading might fail if context remains suspended
       });
   }

  try {
    // Correct path assuming audio-worklet-processor.js is served from /static/
    await audioContext.audioWorklet.addModule('/static/audio-worklet-processor.js');
    console.log("Audio worklet '/static/audio-worklet-processor.js' loaded successfully");
    return true; // Indicate success
  } catch (e) {
    console.error("Failed to load audio worklet:", e);
    updateStatus("Error: Failed to load audio processor. Check path and script.");
    return false; // Indicate failure
  }
}

// Start the conversation (recording + sending audio)
async function startConversation() {
    console.log("Attempting to start conversation...");
    if (isListening) {
        console.warn("Conversation already started.");
        return;
    }

    // 1. Ensure WebSocket is ready
    if (!webSocket || webSocket.readyState !== WebSocket.OPEN) {
        updateStatus("WebSocket not connected. Cannot start conversation.");
        console.error("WebSocket not ready for conversation.");
        // Optionally try to reconnect here: await connectWebSocket();
        return;
    }

    // 2. Ensure AudioContext is running
    try {
        ensureAudioContext(); // Make sure it's created and try to resume
        if (audioContext.state !== 'running') {
            updateStatus("Audio system not ready. Click the page to enable audio.");
            console.error("AudioContext is not running:", audioContext.state);
            return;
        }
    } catch (e) {
        updateStatus("Failed to initialize audio system.");
        return;
    }


    // 3. Ensure Worklet is loaded (Load it if necessary)
     // Check if the processor is registered (a more robust check)
     let workletLoaded = false;
     try {
         // Attempting to create a node will throw if the processor isn't registered
         const testNode = new AudioWorkletNode(audioContext, 'audio-processor');
         testNode.disconnect(); // Clean up immediately
         workletLoaded = true;
         console.log("Audio worklet processor 'audio-processor' seems available.");
     } catch (e) {
         console.log("Audio worklet processor 'audio-processor' not found, attempting to load...");
         workletLoaded = await loadAudioWorklet();
     }

     if (!workletLoaded) {
         updateStatus("Failed to load or verify audio processor. Cannot start conversation.");
         return;
     }


    // 4. Get Microphone Access and Connect Pipeline
    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                channelCount: 1,
                // Request desired sample rate, browser might adjust
                sampleRate: audioContext.sampleRate
            }
        });
        console.log("Microphone access granted. Stream:", micStream);

        // Create the actual worklet node instance
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        console.log("AudioWorkletNode created:", audioWorkletNode);

        const microphoneSource = audioContext.createMediaStreamSource(micStream);
        console.log("Microphone source created:", microphoneSource);

        microphoneSource.connect(audioWorkletNode);
        console.log("Microphone source connected to AudioWorkletNode.");
        // Do NOT connect workletNode to destination for recording/sending

        // 5. Set up message handling FROM the worklet
        audioWorkletNode.port.onmessage = handleWorkletMessage;
        console.log("Worklet message handler set up.");

        // 6. Send 'start' command TO the worklet
        // Use the command structure expected by your audio-worklet-processor.js
        audioWorkletNode.port.postMessage({ command: 'start' });
        isListening = true;
        console.log("Sent 'start' command to worklet.");

        // 7. Update UI
        const startBtn = document.getElementById('startConversation');
        const stopBtn = document.getElementById('stopConversation');
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        updateStatus("Conversation started. Listening...");

        // 8. Notify Server (Optional)
        sendWebSocketMessage({ type: 'start_conversation', session_id: sessionId });
        console.log("Sent 'start_conversation' message to server.");

    } catch (e) {
        console.error("Failed to start conversation pipeline:", e);
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
            updateStatus("Error: Microphone permission denied.");
        } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
            updateStatus("Error: No microphone found.");
        } else if (e.name === 'TypeError' && e.message.includes('AudioWorkletNode')) {
             updateStatus("Error: Audio Worklet Node creation failed. Is the processor loaded?");
        } else {
            updateStatus("Error starting microphone/worklet. See console.");
        }
        // Clean up UI if start failed
        const startBtn = document.getElementById('startConversation');
        const stopBtn = document.getElementById('stopConversation');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        isListening = false; // Ensure state is correct
        // Clean up resources if partially started
        if (micStream) micStream.getTracks().forEach(track => track.stop());
        if (audioWorkletNode) audioWorkletNode.disconnect();
        micStream = null;
        audioWorkletNode = null;
    }
}

// Handle messages received FROM the audio worklet
function handleWorkletMessage(event) {
    // Adapt based on the exact message structure from your audio-worklet-processor.js
    const { event: eventType, data, message } = event.data;

    if (eventType === 'data') { // Assuming worklet sends { event: 'data', data: Int16Array }
        if (isListening && webSocket && webSocket.readyState === WebSocket.OPEN) {
            // Server expects raw binary audio data (Int16Array buffer)
             if (data instanceof Int16Array || data instanceof ArrayBuffer) {
                 webSocket.send(data.buffer ? data.buffer : data); // Send the underlying ArrayBuffer
                 // console.log(`Sent audio chunk: ${data.byteLength} bytes`);
             } else {
                 console.warn("Received unexpected data format from worklet:", data);
             }
        }
    } else if (eventType === 'info') {
        console.log("Audio Worklet Info:", message);
        // updateStatus(`Worklet: ${message}`); // Optional: Display info messages
    } else if (eventType === 'error') {
        console.error("Audio worklet error:", message);
        updateStatus(`Worklet Error: ${message}`);
        // Potentially stop conversation on critical worklet errors
        // stopConversation();
    } else {
        // console.log("Received unhandled message from worklet:", event.data);
    }
}

// Stop the conversation (stop recording/sending)
function stopConversation() {
    console.log("Attempting to stop conversation...");
    if (!isListening) {
        console.warn("Conversation not active or already stopping.");
        return;
    }
    isListening = false; // Set state immediately to prevent sending more data

    // 1. Send 'stop' command TO the worklet
    if (audioWorkletNode && audioWorkletNode.port) {
        try {
            audioWorkletNode.port.postMessage({ command: 'stop' });
            console.log("Sent 'stop' command to worklet.");
        } catch(e) {
             console.warn("Could not send stop command to worklet, might already be disconnected.", e);
        }
    }

    // 2. Stop Microphone Tracks
    if (micStream) {
        micStream.getTracks().forEach(track => {
             track.stop();
             console.log(`Microphone track stopped: ${track.label || track.id}`);
        });
        micStream = null; // Release the stream object
    } else {
         console.log("No active microphone stream to stop.");
    }

    // 3. Disconnect Worklet Node (important for cleanup)
    if (audioWorkletNode) {
         try {
             // Disconnect from any inputs first (microphoneSource should be disconnected automatically when track stops)
             audioWorkletNode.disconnect();
             console.log("AudioWorkletNode disconnected.");
         } catch (e) {
             console.warn("Error disconnecting AudioWorkletNode:", e);
         }
         audioWorkletNode = null; // Release the node object
    } else {
         console.log("No active AudioWorkletNode to disconnect.");
    }

    // 4. Update UI
    const startBtn = document.getElementById('startConversation');
    const stopBtn = document.getElementById('stopConversation');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    updateStatus("Conversation stopped.");

    // 5. Notify Server
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        sendWebSocketMessage({ type: 'stop_audio', session_id: sessionId });
        console.log("Sent 'stop_audio' message to server.");
    } else {
         console.warn("WebSocket not open when trying to send stop_audio.");
    }
}

// --- Transcription Display ---

function displayTranscription(text) {
    const messagesDiv = document.getElementById('messages');
    if (!messagesDiv) {
         console.error("Cannot display transcription: 'messages' element not found.");
         return;
    }

    const messageEl = document.createElement('div');
    // Add classes for potential styling
    messageEl.classList.add('message', 'server-message'); // Example classes
    messageEl.textContent = `Server: ${text}`; // Indicate it's from the server

    // Append and scroll
    messagesDiv.appendChild(messageEl);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Auto-scroll
}


// --- Initialization and Event Listeners ---

function setupEventListeners() {
    const synthesizeButton = document.getElementById('synthesizeButton');
    const stopButton = document.getElementById('stopButton');
    const startConversationButton = document.getElementById('startConversation');
    const stopConversationButton = document.getElementById('stopConversation');

    if (synthesizeButton) {
        synthesizeButton.addEventListener('click', sendSynthesizeRequest);
    } else console.warn("Button 'synthesizeButton' not found.");

    if (stopButton) {
        stopButton.addEventListener('click', sendStopAudio);
    } else console.warn("Button 'stopButton' not found.");

    if (startConversationButton) {
        startConversationButton.addEventListener('click', startConversation);
    } else console.warn("Button 'startConversation' not found.");

    if (stopConversationButton) {
        stopConversationButton.addEventListener('click', stopConversation);
    } else console.warn("Button 'stopConversation' not found.");

    // Add interaction listener to resume AudioContext (important!)
    const resumeAudioContext = async () => {
        if (audioContext && audioContext.state === 'suspended') {
            console.log("User interaction detected, attempting to resume AudioContext...");
            await audioContext.resume();
        }
        // Remove the listener after the first interaction
        document.body.removeEventListener('click', resumeAudioContext);
        document.body.removeEventListener('keydown', resumeAudioContext);
    };
    document.body.addEventListener('click', resumeAudioContext, { once: true });
    document.body.addEventListener('keydown', resumeAudioContext, { once: true });
}


// Initialize on DOMContentLoaded
window.addEventListener('DOMContentLoaded', async () => {
    updateStatus('Document loaded. Initializing...');
    setupEventListeners(); // Setup buttons first

    // Disable buttons until connection is ready
    const startConversationButton = document.getElementById('startConversation');
    const stopConversationButton = document.getElementById('stopConversation');
    const synthesizeButton = document.getElementById('synthesizeButton');
    const stopButton = document.getElementById('stopButton');

    if(startConversationButton) startConversationButton.disabled = true;
    if(stopConversationButton) stopConversationButton.disabled = true;
    if(synthesizeButton) synthesizeButton.disabled = true;
    if(stopButton) stopButton.disabled = true;


    try {
        await connectWebSocket(); // Connect WebSocket, which also initializes AudioContext/Player
        // Enable relevant buttons now that WS is connected
        if(synthesizeButton) synthesizeButton.disabled = false;
        if(stopButton) stopButton.disabled = false;
        if(startConversationButton) startConversationButton.disabled = false;
        // stopConversationButton remains disabled until conversation starts

        // Pre-load the worklet after successful connection
        await loadAudioWorklet();

        updateStatus('Ready. Enter text or start conversation.');
    } catch (error) {
        console.error("Initialization failed:", error);
        updateStatus(`Initialization failed: ${error.message || 'Unknown error'}. Check console and refresh.`);
        // Keep buttons disabled
    }
});

// Handle WebSocket close event (already defined above, ensure it disables buttons)
function handleWebSocketClose(event) {
    // Implementation from previous steps, ensure buttons are disabled
    console.log(`WebSocket closed: Code=${event.code}, Reason='${event.reason}' Clean=${event.wasClean}`);
    updateStatus(`WebSocket disconnected: ${event.reason || 'Connection closed'}.`);
    const startConversationButton = document.getElementById('startConversation');
    const stopConversationButton = document.getElementById('stopConversation');
    const synthesizeButton = document.getElementById('synthesizeButton');
    const stopButton = document.getElementById('stopButton');

    if (startConversationButton) startConversationButton.disabled = true;
    if (stopConversationButton) stopConversationButton.disabled = true;
    if (synthesizeButton) synthesizeButton.disabled = true;
    if (stopButton) stopButton.disabled = true;
    isListening = false; // Ensure listening state is reset
}


// --- Exports ---
// Make functions accessible globally if needed
window.streamingClient = {
     // Existing functions
     connectWebSocket,
     sendSynthesizeRequest,
     sendStopAudio,
     processAudioData,
     handleWebSocketClose,
     // New functions for conversation
     startConversation,
     stopConversation
     // Add others if necessary, e.g., ensureAudioContext, loadAudioWorklet
};

console.log("streaming_client.js loaded."); 
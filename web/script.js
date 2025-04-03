// Default helper functions
function defaultLog(message, level = 'info') {
    console.log(`[${level}] ${message}`);
    if (level === 'status') {
        const statusDiv = document.getElementById('statusDiv');
        if (statusDiv) statusDiv.textContent = message;
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

// DOM elements - with fallbacks for testing environment
const textInput = typeof document !== 'undefined' ? document.getElementById('text-input') : null;
const synthesizeButton = typeof document !== 'undefined' ? document.getElementById('synthesize-button') : null;
const statusDiv = typeof document !== 'undefined' ? document.getElementById('status') : null;
const errorDiv = typeof document !== 'undefined' ? document.getElementById('error') : null;

// Remove token display div element and related function
if (typeof document !== 'undefined') {
    const tokenOutputDiv = document.getElementById('token-output');
    if (tokenOutputDiv) {
        tokenOutputDiv.remove();
    }
}

let socket = null;
// Restore audio playback related variables
let audioContext = null;
let sampleRate = 16000; // Default, will be updated by AudioInfo
let audioQueue = []; 
let isPlaying = false;
let connectionRetries = 0;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

// WebSocket connection configuration
const CONNECTION_CONFIG = {
    url: 'ws://localhost:8765',
    initialBackoffDelayMs: 1000,
    maxBackoffDelayMs: 30000,
    backoffFactor: 1.5,
    firstMessageTimeoutMs: 5000,
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
    messageTimeoutMs: 10000,      // 10 seconds for server response
    recordInputTimeoutMs: 60000,  // 1 minute for user recording
    playOutputTimeoutMs: 30000,   // 30 seconds for audio playback
    inactivityTimeoutMs: 300000   // 5 minutes of inactivity
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
    // >>> DEBUG LOG REMOVED
    lastActivityTime = Date.now();
    if (inactivityTimeoutId) {
        clearTimeout(inactivityTimeoutId);
        // >>> DEBUG LOG REMOVED
    }
    // >>> DEBUG LOG REMOVED
    inactivityTimeoutId = setTimeout(handleInactivity, TIMEOUT_CONFIG.inactivityTimeoutMs);
    // >>> DEBUG LOG REMOVED
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
        // >>> DEBUG LOG REMOVED
        helpers.log(`Connection attempt ${retryAttempt + 1}: Already connected or connecting.`, 'info');
        return;
    }

    clearConnectionTimers();
    clearAllTimeouts();
    connectionState.retryCount = retryAttempt;

    try {
        helpers.updateStatus(retryAttempt > 0 ? `Reconnecting (attempt ${retryAttempt + 1})...` : 'Connecting...');
        connectionState.socket = new WebSocket(CONNECTION_CONFIG.url);
        const currentSocket = connectionState.socket; // Capture instance for handler assignment

        currentSocket.onopen = () => {
            // >>> DEBUG LOG REMOVED
            helpers.clearError();
            helpers.updateStatus('Connected. Ready to synthesize.');
            connectionState.retryCount = 0;
            
            // Set up first message timeout
            connectionState.firstMessageTimer = setTimeout(() => {
                helpers.log('First message timeout reached', 'error');
                if (connectionState.socket) {
                    connectionState.socket.close();
                }
            }, CONNECTION_CONFIG.firstMessageTimeoutMs);

            // Start inactivity tracking
            updateActivity();
        };
        // >>> DEBUG LOG REMOVED

        currentSocket.onmessage = (event) => {
            // Clear first message timer if it exists
            if (connectionState.firstMessageTimer) {
                clearTimeout(connectionState.firstMessageTimer);
                connectionState.firstMessageTimer = null;
                helpers.log('First message received, timer cleared.', 'info');
            }

            // Clear message timeout if it exists
            if (messageTimeoutId) {
                clearTimeout(messageTimeoutId);
                messageTimeoutId = null;
            }

            // Update activity timestamp
            updateActivity();

            // Handle message...
            if (event.data instanceof Blob) {
                // Audio data received, clear play timeout if exists
                if (playTimeoutId) {
                    clearTimeout(playTimeoutId);
                    playTimeoutId = null;
                }
                // Process audio data...
            } else {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'completed') {
                        // Synthesis completed, set play timeout
                        playTimeoutId = setTimeout(() => {
                            helpers.log('Audio playback timeout reached', 'warn');
                            helpers.showError('Audio playback took too long to complete.');
                            // Stop playback and reset state
                            audioQueue = [];
                            isPlaying = false;
                        }, TIMEOUT_CONFIG.playOutputTimeoutMs);
                    }
                } catch (e) {
                    helpers.log('Failed to parse message: ' + e, 'error');
                }
            }
        };
        // >>> DEBUG LOG REMOVED

        currentSocket.onclose = (event) => {
            // >>> DEBUG LOG REMOVED
            const decision = handleWebSocketClose(event, retryAttempt);

            if (decision.shouldRetry) {
                // >>> DEBUG LOG REMOVED
                connectionState.retryTimeout = setTimeout(() => {
                     // >>> DEBUG LOG REMOVED
                     connectWebSocket(decision.nextRetryAttempt);
                }, decision.delay);
            } else {
                 // >>> DEBUG LOG REMOVED
            }
        };
        // >>> DEBUG LOG REMOVED

        currentSocket.onerror = (event) => {
            // >>> DEBUG LOG REMOVED
            helpers.log('WebSocket error: ' + event, 'error');
            helpers.showError('Failed to connect to the server. Will retry automatically.');
        };
        // >>> DEBUG LOG REMOVED

    } catch (error) {
        helpers.log('Error creating WebSocket connection: ' + error, 'error');
        helpers.showError('Failed to create WebSocket connection: ' + error.message);
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
    // >>> DEBUG LOG REMOVED
    if (!isPlaying) {
        // Set play timeout when starting new playback
        if (playTimeoutId) {
            clearTimeout(playTimeoutId);
        }
        // >>> DEBUG LOG REMOVED
        playTimeoutId = setTimeout(() => {
            helpers.log('Audio playback timeout reached', 'warn');
            helpers.showError('Audio playback took too long to complete.');
            resetAudioState(); 
        }, TIMEOUT_CONFIG.playOutputTimeoutMs);
        // >>> DEBUG LOG REMOVED
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
    // This function will only be called in a browser environment
    helpers.log('Initializing UI elements and listeners...');
    const synthButton = document.getElementById('synthesizeButton');
    const textInput = document.getElementById('textInput');

    if (synthButton && textInput) {
        synthesizeButton.onclick = () => {
            const text = textInput.value.trim();
            if (!text) {
                helpers.showError("Please enter some text to synthesize.");
                return;
            }
            if (!connectionState.socket || connectionState.socket.readyState !== WebSocket.OPEN) {
                helpers.showError("WebSocket is not connected. Attempting to reconnect...");
                if (!connectionState.socket || connectionState.socket.readyState >= WebSocket.CLOSING) {
                    connectWebSocket(); 
                }
                return;
            }
            sendSynthesizeRequest(text); // Call dedicated function
        };
        helpers.log('UI listeners attached.');
    } else {
        helpers.log("Could not find synthesize button or text input element.", 'error');
    }
}

// --- Request Sending ---
function sendSynthesizeRequest(text) {
    if (!connectionState.socket || connectionState.socket.readyState !== WebSocket.OPEN) {
         helpers.log('Cannot send request: WebSocket not open.', 'warn');
         // Optionally trigger reconnect or show error
         helpers.showError('Cannot send request: Connection lost.');
         if (!connectionState.socket || connectionState.socket.readyState >= WebSocket.CLOSING) {
             connectWebSocket();
         }
         return;
     }
     
    helpers.clearError();
    helpers.updateStatus('Sending text for synthesis...');
    const message = { text: text }; 

    try {
        const jsonMessage = JSON.stringify(message);
        helpers.log(`Sending JSON: ${jsonMessage}`);
        connectionState.socket.send(jsonMessage);

        // Set message timeout
        if (messageTimeoutId) {
            clearTimeout(messageTimeoutId);
        }
        messageTimeoutId = setTimeout(handleMessageTimeout, TIMEOUT_CONFIG.messageTimeoutMs);
    } catch (e) {
        helpers.showError("Failed to construct or send message: " + e);
        helpers.updateStatus("Failed to send request");
    }
}

// Extracted logic for message timeout callback
function handleMessageTimeout() {
    helpers.log('Server response timeout reached', 'error');
    helpers.showError('Server did not respond in time.');
    messageTimeoutId = null;
}

// --- Script Execution Start ---

// Check if running in a browser-like environment to attach UI listeners
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    // Use DOMContentLoaded to ensure elements are ready before attaching listeners
    document.addEventListener('DOMContentLoaded', initializeUI);
} else {
    helpers.log('Not attaching UI listeners in non-browser environment.', 'debug');
}

// Attempt initial connection regardless of environment (for tests)
connectWebSocket();

// Update exports
module.exports = {
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
    // For testing
    setHelpers: (newHelpers) => {
        helpers = { ...helpers, ...newHelpers };
    },
    resetAudioState,
    // Export internal state for testing
    get messageTimeoutId() { return messageTimeoutId; },
    get recordTimeoutId() { return recordTimeoutId; },
    get playTimeoutId() { return playTimeoutId; },
    get inactivityTimeoutId() { return inactivityTimeoutId; },
    get lastActivityTime() { return lastActivityTime; },
    get audioQueue() { return audioQueue; },
    get isPlaying() { return isPlaying; },
    updateActivity,
    playAudio,
    playNextChunk,
    sendSynthesizeRequest,
    // Export newly extracted logic for testing
    handleWebSocketClose,
    handleInactivity,
    handleMessageTimeout,
    // Allow setting internal state for tests
    _setInternalStateForTesting: (newState) => {
        if (newState.audioContext !== undefined) audioContext = newState.audioContext;
        if (newState.messageTimeoutId !== undefined) messageTimeoutId = newState.messageTimeoutId;
        if (newState.recordTimeoutId !== undefined) recordTimeoutId = newState.recordTimeoutId;
        if (newState.playTimeoutId !== undefined) playTimeoutId = newState.playTimeoutId;
        if (newState.inactivityTimeoutId !== undefined) inactivityTimeoutId = newState.inactivityTimeoutId;
    }
}; 

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
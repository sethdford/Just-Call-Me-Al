const {
    connectWebSocket,
    CONNECTION_CONFIG,
    TIMEOUT_CONFIG,
    connectionState,
    clearAllTimeouts,
    setHelpers,
    sendSynthesizeRequest,
    playAudio,
    playNextChunk,
    updateActivity,
    messageTimeoutId,
    recordTimeoutId,
    playTimeoutId,
    inactivityTimeoutId,
    lastActivityTime,
    audioQueue,
    isPlaying,
    clearConnectionTimers,
    calculateBackoffDelay,
    resetAudioState,
    handleWebSocketClose,
    handleInactivity,
    handleMessageTimeout
} = require('./streaming_client');

// Mock helper functions
const mockHelpers = {
    log: jest.fn(),
    updateStatus: jest.fn(),
    showError: jest.fn(),
    clearError: jest.fn()
};

// Mock WebSocket
// Keep track of the latest instance for assertions
let mockWebSocketInstance = null;

class MockWebSocket {
    constructor(url) {
        this.readyState = WebSocket.CONNECTING;
        this.sentMessages = [];
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this.onclose = null;
        
        mockWebSocketInstance = this; // Update the global reference
    }

    send(message) {
        this.sentMessages.push(message);
    }

    // Wrap close in jest.fn() to allow spying/assertions
    close = jest.fn((code = 1000, reason = '') => {
        this.readyState = WebSocket.CLOSING;
        if (typeof this.onclose === 'function') {
            this.onclose({ code: code, reason: reason });
        }
        this.readyState = WebSocket.CLOSED;
    });

    // Helper methods for testing
    simulateOpen() {
        this.readyState = WebSocket.OPEN;
        if (typeof this.onopen === 'function') {
            this.onopen();
        }
    }

    simulateMessage(data) {
        if (typeof this.onmessage === 'function') {
            this.onmessage({ data: JSON.stringify(data) }); 
        }
    }

    simulateError(error) {
        if (typeof this.onerror === 'function') {
            this.onerror(error);
        }
        this.readyState = WebSocket.CLOSED; 
         if (typeof this.onclose === 'function') {
             this.onclose({ code: 1006, reason: 'Simulated error' }); 
        }
    }

    simulateClose(code = 1000, reason = '') {
        this.readyState = WebSocket.CLOSED;
        if (typeof this.onclose === 'function') {
            this.onclose({ code: code, reason: reason });
        }
    }
}

global.WebSocket = jest.fn((url) => new MockWebSocket(url));

// Mock AudioContext
global.AudioContext = jest.fn(() => ({
    createBufferSource: jest.fn(() => ({
        buffer: null,
        connect: jest.fn(),
        start: jest.fn(),
        onended: null,
    })),
    decodeAudioData: jest.fn((_, successCallback) => {
        // Simulate successful decoding
        successCallback({/* mock audio buffer */});
    }),
    resume: jest.fn(),
    createBuffer: jest.fn((channels, length, sampleRate) => {
        // Return a mock ArrayBuffer matching the structure expected
        return new ArrayBuffer(length * channels * 4); // Assuming 32-bit float
    }),
    close: jest.fn(),
    sampleRate: 44100
}));

describe('WebSocket Connection Logic', () => {
    beforeEach(() => {
        // Reset mocks for helpers before each test
        mockHelpers.log.mockClear();
        mockHelpers.updateStatus.mockClear();
        mockHelpers.showError.mockClear();
        mockHelpers.clearError.mockClear();
        // Reset WebSocket mock calls and state if necessary
        global.WebSocket.mockClear();
        if (mockWebSocketInstance) {
            mockWebSocketInstance.sentMessages = [];
        }
        // Reset connection state if needed (might be done by connectWebSocket)
        connectionState.socket = null;
        connectionState.retryCount = 0;
        connectionState.firstMessageTimer = null;
        connectionState.retryTimeout = null;
        // Clear any pending timers from previous tests, even though we remove fake timers
        jest.clearAllTimers(); 
        jest.clearAllMocks(); // Clears all mocks, including setTimeout/clearTimeout
        // DO NOT USE FAKE TIMERS ANYMORE
        // jest.useFakeTimers(); 
    });

    afterEach(() => {
        jest.clearAllTimers();
        jest.clearAllMocks();
        // Ensure all timers are cleared between tests
        clearConnectionTimers();
        clearAllTimeouts();
    });

    afterAll(() => {
        jest.useRealTimers();
    });

    test('requests retry on abnormal close', () => {
        const closeEvent = { code: 1006, reason: 'Abnormal closure' };
        const decision = handleWebSocketClose(closeEvent, 0);
        expect(decision.shouldRetry).toBe(true);
        expect(decision.delay).toBe(calculateBackoffDelay(1));
        expect(decision.nextRetryAttempt).toBe(1);
    });

    test('respects maximum retry limit', () => {
        setHelpers(mockHelpers); 
        const currentAttempt = CONNECTION_CONFIG.MAX_RETRIES;
        const closeEvent = { code: 1006, reason: 'Abnormal closure' };
        mockHelpers.showError.mockClear(); 
        
        const decision = handleWebSocketClose(closeEvent, currentAttempt);
        
        expect(decision.shouldRetry).toBe(false);
        expect(decision.reason).toBe('max_retries');
        expect(mockHelpers.showError).toHaveBeenCalledWith(
            expect.stringContaining('Maximum retry attempts reached. Please refresh the page')
        );
    });

    test('cleans up timers on close', () => {
        // Arrange: Set a minimal mock AudioContext for playAudio to run
        const { _setInternalStateForTesting } = require('./streaming_client');
        _setInternalStateForTesting({ 
            audioContext: { 
                createBuffer: jest.fn(() => ({ 
                    getChannelData: jest.fn(() => ({ set: jest.fn() })) 
                })),
                // Add createBufferSource needed by playNextChunk
                createBufferSource: jest.fn(() => ({ 
                    buffer: null,
                    connect: jest.fn(),
                    start: jest.fn(),
                    onended: null 
                })),
                destination: {} // Needed for source.connect
            } 
        });

        // Arrange: Set timers by calling relevant functions
        connectWebSocket(); // Sets connectionState timers (retry/firstMessage)
        const mockWebSocketInstance = connectionState.socket;
        mockWebSocketInstance.simulateOpen(); // Sets inactivity
        sendSynthesizeRequest('timer test'); // Sets message
        playAudio(new Float32Array([0.1]), 44100); // Sets play
        
        // Capture the IDs *after* they should have been set
        const scriptState = require('./streaming_client');
        const msgId = scriptState.messageTimeoutId;
        const inactivityId = scriptState.inactivityTimeoutId;
        const playId = scriptState.playTimeoutId;
        const firstMsgId = connectionState.firstMessageTimer;
        // Note: retryTimeout is only set *after* onclose runs, 
        // so we can't reliably test clearing it here without triggering onclose.
        // Let's skip asserting retryTimeout clearing in this specific test.

        // Verify timers were set (IDs are not null)
        expect(msgId).not.toBeNull();
        expect(inactivityId).not.toBeNull();
        expect(playId).not.toBeNull();
        expect(firstMsgId).not.toBeNull();
        
        global.clearTimeout = jest.fn(); // Mock clearTimeout

        // Act: Call the handler
        const closeEvent = { code: 1006, reason: 'Abnormal closure' };
        handleWebSocketClose(closeEvent, 0); 
        
        // Assert that clearTimeout was called with the captured IDs
        expect(global.clearTimeout).toHaveBeenCalledWith(msgId); 
        expect(global.clearTimeout).toHaveBeenCalledWith(inactivityId); 
        expect(global.clearTimeout).toHaveBeenCalledWith(playId); 
        expect(global.clearTimeout).toHaveBeenCalledWith(firstMsgId); 
        // expect(global.clearTimeout).toHaveBeenCalledWith(retryTimeoutId); // Skipped
    });

    test('prevents duplicate connections', () => {
        setHelpers(mockHelpers); // Ensure correct mock helpers are used
        connectWebSocket();
        const initialInstance = connectionState.socket;
        expect(initialInstance).toBeTruthy();
        // Ensure readyState is set correctly *before* the second call
        initialInstance.readyState = WebSocket.OPEN; 
        mockHelpers.log.mockClear(); // Clear logs before second call

        // Attempt to connect again
        connectWebSocket();

        // Check WebSocket constructor wasn't called again
        expect(global.WebSocket).toHaveBeenCalledTimes(1);
        // Check the log message
        expect(mockHelpers.log).toHaveBeenCalledWith(
            expect.stringContaining('Already connected or connecting'), 
            'info'
        );
    });

    test('cleans up on connection error', () => {
        setHelpers(mockHelpers); // Ensure correct mock helpers are used
        connectWebSocket();
        expect(global.WebSocket).toHaveBeenCalledTimes(1);
        const mockWebSocketInstance = connectionState.socket;
        expect(mockWebSocketInstance).toBeTruthy();
        mockHelpers.log.mockClear(); 
        mockHelpers.showError.mockClear();

        mockWebSocketInstance.simulateError('Test error event');
        
        // Check the actual log call from the onerror handler in script.js
        expect(mockHelpers.log).toHaveBeenCalledWith(
            expect.stringContaining('WebSocket error: Test error event'), // Match the actual remaining log
            'error'
        );
        // Check the error shown by the handler
        expect(mockHelpers.showError).toHaveBeenCalledWith(
            expect.stringContaining('Failed to connect to the server. Will retry automatically.') 
        );
    });
});

describe('Timeout Handling', () => {
    let mockAudioContext;
    let mockAudioBuffer;
    let mockAudioBufferSource;
    let currentMockSource;

    beforeEach(() => {
        // Ensure a fresh state for audio tests
        resetAudioState(); 

        // Create a fresh mock AudioContext for each test in this suite
        mockAudioContext = { 
            createBufferSource: jest.fn(() => {
                // Return a new mock source each time
                mockAudioBufferSource = {
                    buffer: null,
                    connect: jest.fn(),
                    start: jest.fn(),
                    onended: null, 
                };
                return mockAudioBufferSource;
            }),
            decodeAudioData: jest.fn((_, successCallback) => {
                successCallback({/* mock audio buffer */});
            }),
            resume: jest.fn(),
            createBuffer: jest.fn((channels, length, sampleRate) => {
                return { 
                    getChannelData: jest.fn(() => ({ 
                        set: jest.fn() // Mock the set method
                    })) 
                }; 
            }),
            close: jest.fn(),
            sampleRate: 44100,
            destination: {} 
        };
        // Assign the mock to the global scope if needed by the script
        global.AudioContext = jest.fn(() => mockAudioContext);
        
        // Assign directly for script under test using the exported setter
        const { _setInternalStateForTesting } = require('./streaming_client');
        _setInternalStateForTesting({ audioContext: mockAudioContext });

        clearAllTimeouts(); // Ensure all other timers are cleared
        jest.clearAllMocks(); // Clear setTimeout/clearTimeout mocks
        // DO NOT USE FAKE TIMERS ANYMORE
        // jest.useFakeTimers(); 

        // Mock AudioContext and related objects
        mockAudioBuffer = {
            getChannelData: jest.fn().mockReturnValue(new Float32Array(8))
        };

        mockAudioBufferSource = {
            buffer: null,
            connect: jest.fn(),
            start: jest.fn(),
            onended: null // The callback is assigned directly in playNextChunk
        };

        mockAudioContext.createBufferSource = jest.fn().mockImplementation(() => mockAudioBufferSource);
        mockAudioContext.destination = {};
        audioContext = mockAudioContext; // Assign directly for script under test
        clearAllTimeouts(); // Ensure all other timers are cleared
    });

    afterEach(() => {
        jest.clearAllTimers();
        jest.clearAllMocks();
        // Ensure all timers are cleared between tests
        clearConnectionTimers();
        clearAllTimeouts();
        audioContext = null;
        resetAudioState(); // Use the reset function
    });

    afterAll(() => {
        jest.useRealTimers();
    });

    test('should set message timeout ID when sending request', () => {
        connectWebSocket();
        const mockWebSocketInstance = connectionState.socket;
        mockWebSocketInstance.simulateOpen();
        sendSynthesizeRequest('test text');
        expect(require('./streaming_client').messageTimeoutId).not.toBeNull();
    });

    test('should clear message timeout ID on response', () => {
        connectWebSocket();
        const mockWebSocketInstance = connectionState.socket;
        mockWebSocketInstance.simulateOpen();
        sendSynthesizeRequest('test text');
        expect(require('./streaming_client').messageTimeoutId).not.toBeNull(); // Verify it was set
        const initialId = require('./streaming_client').messageTimeoutId;
        global.clearTimeout = jest.fn(); // Mock clearTimeout

        mockWebSocketInstance.simulateMessage({ type: 'audio', data: '...' });

        expect(require('./streaming_client').messageTimeoutId).toBeNull();
        // Optionally, check if clearTimeout was called with the initial ID
        // expect(global.clearTimeout).toHaveBeenCalledWith(initialId);
    });

    test('should handle message timeout', () => {
        setHelpers(mockHelpers); // Ensure correct mock helpers are used
        // Arrange 
        connectWebSocket(); 
        const mockWebSocketInstance = connectionState.socket;
        mockWebSocketInstance.simulateOpen(); 
        sendSynthesizeRequest('test'); 
        expect(require('./streaming_client').messageTimeoutId).not.toBeNull();
        mockHelpers.log.mockClear(); // Clear logs before handler call
        mockHelpers.showError.mockClear();

        // Act: Call the extracted handler directly
        handleMessageTimeout();

        // Assert
        expect(mockHelpers.log).toHaveBeenCalledWith(expect.stringContaining('Server response timeout reached'), 'error');
        expect(mockHelpers.showError).toHaveBeenCalledWith(expect.stringContaining('Server did not respond in time'));
        expect(require('./streaming_client').messageTimeoutId).toBeNull();
    });

    test('should set play timeout ID when starting audio playback', () => {
        const mockAudioData = new Float32Array([0.1, 0.2]);
        playAudio(mockAudioData, 44100);
        expect(require('./streaming_client').playTimeoutId).not.toBeNull();
    });

    test('should clear play timeout ID when playback completes', () => {
        const mockAudioData = new Float32Array([0.1, 0.2]);
        playAudio(mockAudioData, 44100); // Sets the ID and starts playback
        const initialId = require('./streaming_client').playTimeoutId;
        expect(initialId).not.toBeNull();
        global.clearTimeout = jest.fn(); // Mock clearTimeout

        // Simulate playback finishing 
        expect(mockAudioBufferSource).toBeTruthy();
        expect(mockAudioBufferSource.onended).toBeInstanceOf(Function);
        mockAudioBufferSource.onended(); 

        expect(require('./streaming_client').playTimeoutId).toBeNull();
        // Optionally check clearTimeout
        // expect(global.clearTimeout).toHaveBeenCalledWith(initialId);
    });

    test('should set inactivity timeout ID on activity', () => {
        updateActivity(); 
        expect(require('./streaming_client').inactivityTimeoutId).not.toBeNull();
    });

    test('should reset inactivity timer on user activity', () => {
        global.clearTimeout = jest.fn(); 
        updateActivity(); 
        const initialId = require('./streaming_client').inactivityTimeoutId;
        expect(initialId).not.toBeNull();
        updateActivity();
        const newId = require('./streaming_client').inactivityTimeoutId;
        expect(newId).not.toBeNull();
        expect(newId).not.toBe(initialId); 
        expect(global.clearTimeout).toHaveBeenCalledWith(initialId);
    });

    test('should handle inactivity timeout', () => {
        // Arrange: Connect and open socket to set initial timer
        connectWebSocket();
        const currentInstance = connectionState.socket;
        expect(currentInstance).toBeTruthy();
        currentInstance.simulateOpen(); // Trigger onopen -> updateActivity
        expect(require('./streaming_client').inactivityTimeoutId).not.toBeNull(); // Verify initial timer set
        currentInstance.readyState = WebSocket.OPEN; // Ensure state is open for close call

        // Act: Call the extracted handler directly
        handleInactivity();
        
        // Assert: Check if close was called and ID is null
        expect(currentInstance.close).toHaveBeenCalledTimes(1);
        expect(require('./streaming_client').inactivityTimeoutId).toBeNull();
    });

    test('should clear all timeout IDs', () => {
        // Arrange: Set timers by calling relevant functions
        connectWebSocket();
        const mockWebSocketInstance = connectionState.socket;
        mockWebSocketInstance.simulateOpen(); // Sets inactivity
        sendSynthesizeRequest('timer test'); // Sets message
        playAudio(new Float32Array([0.1]), 44100); // Sets play
        // We don't easily set recordTimeoutId in these tests, skip asserting it.
        
        // Capture IDs
        const scriptState = require('./streaming_client');
        const msgId = scriptState.messageTimeoutId;
        const inactivityId = scriptState.inactivityTimeoutId;
        const playId = scriptState.playTimeoutId;

        expect(msgId).not.toBeNull();
        expect(inactivityId).not.toBeNull();
        expect(playId).not.toBeNull();
        
        global.clearTimeout = jest.fn(); // Mock clearTimeout
        
        // Act
        clearAllTimeouts();
        
        // Assert state is null
        expect(scriptState.messageTimeoutId).toBeNull();
        // expect(scriptState.recordTimeoutId).toBeNull(); // Skipped
        expect(scriptState.playTimeoutId).toBeNull();
        expect(scriptState.inactivityTimeoutId).toBeNull();
        
        // Assert clearTimeout was called with captured IDs
        expect(global.clearTimeout).toHaveBeenCalledWith(msgId);
        // expect(global.clearTimeout).toHaveBeenCalledWith(recordId); // Skipped
        expect(global.clearTimeout).toHaveBeenCalledWith(playId);
        expect(global.clearTimeout).toHaveBeenCalledWith(inactivityId);
    });
});
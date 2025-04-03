// Mock DOM elements
const mockStatusDiv = { textContent: '', style: { display: 'none' } };
const mockErrorDiv = { textContent: '', style: { display: 'none' } };
const mockSynthButton = { onclick: null };
const mockTextInput = { value: '' };

// Assign directly to document for easier access/mocking
document.getElementById = jest.fn((id) => {
    if (id === 'statusDiv') return mockStatusDiv;
    if (id === 'errorDiv') return mockErrorDiv;
    if (id === 'synthesizeButton') return mockSynthButton;
    if (id === 'textInput') return mockTextInput;
    return null; // Default mock behavior
});

// Mock AudioWorkletProcessor
global.AudioWorkletProcessor = class {
  constructor() {
    this.port = {
      postMessage: jest.fn(),
      onmessage: null
    };
  }
};

// Mock performance.now()
global.performance = {
  now: () => Date.now()
};

// Mock Int16Array and Float32Array if not available in Node environment
if (typeof Int16Array === 'undefined') {
  global.Int16Array = Array;
}

if (typeof Float32Array === 'undefined') {
  global.Float32Array = Array;
}

// Mock AudioContext
global.AudioContext = class {
  constructor() {
    this.sampleRate = 48000;
    this.destination = { mock: 'destination' };
    this.createBuffer = jest.fn((channels, length, sampleRate) => ({
        getChannelData: jest.fn(() => new Float32Array(length)) // Mock buffer with getChannelData
    }));
    this.createBufferSource = jest.fn(() => {
      // Added minimal required properties/methods for playNextChunk
    });
  }
};

// Suppress console.error during tests
global.console.error = jest.fn();

// Mock registerProcessor
global.registerProcessor = jest.fn();

// --- WebSocket Mocking Removed - Now handled in script.test.js ---

// --- Keep performance.now mock ---
global.performance = {
    now: jest.fn(() => Date.now())
};

// --- Suppress console errors if needed ---
// Can still be done here if desired globally, or per-test
// let consoleErrorSpy;
// beforeEach(() => {
//   consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
// });
// afterEach(() => {
//   consoleErrorSpy.mockRestore();
// }); 
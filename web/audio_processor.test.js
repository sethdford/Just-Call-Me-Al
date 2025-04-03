const { StreamProcessor } = require('./audio_processor.js');

// StreamProcessor Tests

describe('StreamProcessor', () => {
  let processor;
  let mockPort;

  beforeEach(() => {
    // Mock AudioWorkletProcessor
    global.AudioWorkletProcessor = class {};
    global.performance = { now: () => Date.now() };

    // Mock port for communication
    mockPort = {
      postMessage: jest.fn(),
      onmessage: null
    };

    // Create processor instance
    processor = new StreamProcessor();
    processor.port = mockPort;
  });

  describe('Buffer Management', () => {
    test('initial buffer configuration is correct', () => {
      expect(processor.minBufferMs).toBe(100);
      expect(processor.maxBufferMs).toBe(1000);
      expect(processor.targetBufferMs).toBe(300);
      expect(processor.bufferPaddingMultiplier).toBe(1.5);
      expect(processor.poorRttThresholdMs).toBe(200);
    });

    test('updateBufferSize adjusts based on RTT', () => {
      // Test normal conditions
      processor.rttSamples = [100]; // Good RTT
      processor.updateBufferSize();
      const normalBuffers = processor.playbackMinBuffers;

      // Test poor network conditions
      processor.rttSamples = [250]; // Poor RTT > 200ms threshold
      processor.updateBufferSize();
      const poorNetworkBuffers = processor.playbackMinBuffers;

      expect(poorNetworkBuffers).toBeGreaterThan(normalBuffers);
    });

    test('buffer size respects min/max bounds', () => {
      // Test minimum bound
      processor.targetBufferMs = 50; // Below minimum
      processor.updateBufferSize();
      const minSamples = (processor.minBufferMs / 1000) * 48000;
      expect(processor.playbackMinBuffers * processor.bufferLength).toBeGreaterThanOrEqual(minSamples);

      // Test maximum bound
      processor.targetBufferMs = 2000; // Above maximum
      processor.updateBufferSize();
      const maxSamples = (processor.maxBufferMs / 1000) * 48000;
      expect(processor.playbackMinBuffers * processor.bufferLength).toBeLessThanOrEqual(maxSamples);
    });

    test('writeData handles silence detection correctly', () => {
      // Test non-silent data
      const nonSilentData = new Float32Array([0.1, 0.2, 0.3]);
      processor.writeData(nonSilentData, 'test-track');
      expect(processor.outputBuffers[0].isSilence).toBe(false);

      // Test silent data
      const silentData = new Float32Array([0, 0, 0]);
      processor.writeData(silentData, 'test-track');
      expect(processor.outputBuffers[1].isSilence).toBe(true);
    });
  });

  describe('RTT Monitoring', () => {
    test('RTT sample window maintains correct size', () => {
      for (let i = 0; i < 15; i++) {
        processor.updateRtt(100 + i);
      }
      expect(processor.rttSamples.length).toBe(processor.rttSampleSize);
      expect(processor.rttSamples[0]).toBe(105); // First sample after window shift
    });

    test('average RTT calculation is correct', () => {
      processor.rttSamples = [100, 200, 300];
      expect(processor.getAverageRtt()).toBe(200);
    });

    test('RTT updates trigger buffer size updates', () => {
      const updateBufferSizeSpy = jest.spyOn(processor, 'updateBufferSize');
      processor.updateRtt(150);
      expect(updateBufferSizeSpy).toHaveBeenCalled();
    });

    test('getAverageRtt handles empty samples array', () => {
      processor.rttSamples = [];
      expect(processor.getAverageRtt()).toBe(0);
    });
  });

  describe('Message Handling', () => {
    test('configuration updates are applied correctly', () => {
      const newConfig = {
        minBufferMs: 150,
        maxBufferMs: 800,
        targetBufferMs: 400,
        bufferPaddingMultiplier: 2.0,
        poorRttThresholdMs: 250,
        playbackRateMin: 0.8,
        playbackRateMax: 1.2,
        playbackRateAffordance: 0.3,
        playbackSmoothing: 0.8,
        playbackSkipDigitalSilence: false,
        playbackRecord: true
      };

      processor.doReceive({
        data: {
          event: 'configure',
          config: newConfig
        }
      });

      expect(processor.minBufferMs).toBe(150);
      expect(processor.maxBufferMs).toBe(800);
      expect(processor.targetBufferMs).toBe(400);
      expect(processor.bufferPaddingMultiplier).toBe(2.0);
      expect(processor.poorRttThresholdMs).toBe(250);
      expect(processor.playbackRateMin).toBe(0.8);
      expect(processor.playbackRateMax).toBe(1.2);
      expect(processor.playbackRateAffordance).toBe(0.3);
      expect(processor.playbackSmoothing).toBe(0.8);
      expect(processor.playbackSkipDigitalSilence).toBe(false);
      expect(processor.playbackRecord).toBe(true);
    });

    test('write event processes audio data correctly', () => {
      const testData = new Int16Array(128);
      testData[0] = 16384; // Non-zero value to test conversion

      processor.doReceive({
        data: {
          event: 'write',
          buffer: testData,
          trackId: 'test-track'
        }
      });

      expect(processor.outputBuffers.length).toBe(1);
      expect(processor.outputBuffers[0].trackId).toBe('test-track');
      expect(processor.outputBuffers[0].buffer[0]).toBe(0.5); // 16384/32768 = 0.5
    });

    test('offset event returns correct data', () => {
      processor.writeTrackId = 'test-track';
      processor.trackSampleOffsets['test-track'] = 1000;

      processor.doReceive({
        data: {
          event: 'offset',
          requestId: 'test-request',
          trackId: 'test-track'
        }
      });

      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'offset',
          requestId: 'test-request',
          trackId: 'test-track',
          offset: 1000
        })
      );
    });

    test('interrupt event sets hasInterrupted flag', () => {
      processor.doReceive({
        data: {
          event: 'interrupt',
          requestId: 'test-request'
        }
      });

      expect(processor.hasInterrupted).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('invalid events are caught and reported', () => {
      processor.receive({
        data: {
          event: 'invalid-event'
        }
      });

      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'error',
          data: expect.objectContaining({
            message: expect.stringContaining('Unhandled event')
          })
        })
      );
    });

    test('receive method catches and reports errors', () => {
      const error = new Error('Test error');
      jest.spyOn(processor, 'doReceive').mockImplementation(() => {
        throw error;
      });

      processor.receive({ data: { event: 'test' } });

      expect(mockPort.postMessage).toHaveBeenCalledWith({
        event: 'error',
        data: {
          message: error.message,
          stack: error.stack
        }
      });
    });

    test('handles missing data in receive', () => {
      processor.receive({});
      expect(mockPort.postMessage).not.toHaveBeenCalled();
    });
  });

  describe('Audio Processing', () => {
    let processor;
    let mockPort;
    let inputs;
    let outputs;
    let parameters;

    beforeEach(() => {
      mockPort = {
        postMessage: jest.fn(),
      };
      global.AudioWorkletProcessor = jest.fn();
      processor = new StreamProcessor();
      processor.port = mockPort;

      // Set up test data
      inputs = [];
      outputs = [[new Float32Array(128)]];
      parameters = {};

      // Add some test audio data
      processor.writeData(new Float32Array(128).fill(0.5));
      processor.isInPlayback = true;
    });

    test('process method handles basic audio processing', () => {
      // Process should continue running
      const result = processor.process(inputs, outputs, parameters);
      expect(result).toBe(true);

      // Check that output buffer was modified
      expect(outputs[0][0].some(sample => sample !== 0)).toBe(true);
    });

    test('process method handles empty buffer state', () => {
      processor.outputBuffers = [];
      
      // Process with no data should still continue
      const result = processor.process(inputs, outputs, parameters);
      expect(result).toBe(true);
      
      // Output should be silence
      expect(outputs[0][0].every(sample => sample === 0)).toBe(true);
    });

    test('process method respects playback rate', () => {
      // Setup test data with more samples to make the difference clearer
      const testBuffer = new Float32Array(256).fill(0.5);
      for (let i = 0; i < 5; i++) {
        processor.writeData(testBuffer.slice());
      }
      
      // Process with normal rate
      processor.playbackRate = 1.0;
      processor.playbackOutputOffset = 0;
      processor.process(inputs, outputs, parameters);
      const normalOffset = processor.playbackOutputOffset;

      // Reset and process with faster rate
      processor.outputBuffers = [];
      for (let i = 0; i < 5; i++) {
        processor.writeData(testBuffer.slice());
      }
      processor.playbackRate = 2.0;
      processor.playbackOutputOffset = 0;
      processor.process(inputs, outputs, parameters);
      const fastOffset = processor.playbackOutputOffset;

      // Fast playback should process more samples
      expect(fastOffset).toBeGreaterThan(normalOffset);
    });

    test('process method handles digital silence skipping', () => {
      // Add non-silent audio data
      const nonSilentBuffer = new Float32Array(128).fill(0.5);
      processor.writeData(nonSilentBuffer);
      
      // Process audio
      processor.process(inputs, outputs, parameters);
      
      // Verify that samples were written to output
      expect(outputs[0][0].some(sample => sample !== 0)).toBe(true);
    });

    test('process method records audio when enabled', () => {
      // Enable recording
      processor.playbackRecord = true;
      
      // Add test audio data
      const testBuffer = new Float32Array(128).fill(0.5);
      processor.writeData(testBuffer);
      
      // Process audio
      processor.process(inputs, outputs, parameters);
      
      // Check that audio was recorded
      expect(processor.playbackAudioChunks.length).toBeGreaterThan(0);
      expect(processor.playbackAudioChunks[0].length).toBe(128);
    });
  });

  describe('Utility Functions', () => {
    test('floatTo16BitPCM converts correctly', () => {
      const floatData = new Float32Array([0.5, -0.5, 1.0, -1.0]);
      const buffer = processor.floatTo16BitPCM(floatData);
      const view = new DataView(buffer);

      // Check conversion values
      expect(view.getInt16(0, true)).toBe(16383); // 0.5 * 0x7fff
      expect(view.getInt16(2, true)).toBe(-16384); // -0.5 * 0x8000
      expect(view.getInt16(4, true)).toBe(32767); // 1.0 * 0x7fff
      expect(view.getInt16(6, true)).toBe(-32768); // -1.0 * 0x8000
    });

    test('mergeAudioData combines chunks correctly', () => {
      const chunk1 = new Float32Array([0.1, 0.2]);
      const chunk2 = new Float32Array([0.3, 0.4]);
      processor.playbackAudioChunks = [chunk1, chunk2];

      const merged = processor.mergeAudioData(processor.playbackAudioChunks);
      expect(merged.length).toBe(4);
      
      // Use toBeCloseTo for floating point comparisons
      expect(merged[0]).toBeCloseTo(0.1, 5);
      expect(merged[1]).toBeCloseTo(0.2, 5);
      expect(merged[2]).toBeCloseTo(0.3, 5);
      expect(merged[3]).toBeCloseTo(0.4, 5);
    });

    test('determinePlaybackRate adjusts based on buffer state', () => {
      // Test with low buffer (should increase playback rate)
      processor.playbackRateMin = 0.8;
      processor.playbackRateMax = 1.2;
      processor.playbackRateAffordance = 0.2;

      // Test with buffer level below 0.5 (should slow down)
      const lowRate = processor.determinePlaybackRate(500, 2000); // 25% buffer level
      expect(lowRate).toBeLessThan(1.0);
      expect(lowRate).toBeGreaterThanOrEqual(processor.playbackRateMin);

      // Test with buffer level above 1.5 (should speed up)
      const highRate = processor.determinePlaybackRate(4000, 2000); // 200% buffer level
      expect(highRate).toBeGreaterThan(1.0);
      expect(highRate).toBeLessThanOrEqual(processor.playbackRateMax);

      // Test with normal buffer level (should be 1.0)
      const normalRate = processor.determinePlaybackRate(2000, 2000); // 100% buffer level
      expect(normalRate).toBe(1.0);
    });
  });

  describe('Adaptive Buffering', () => {
    let processor;
    let mockPort;
    let mockNow;

    beforeEach(() => {
      mockNow = 1000;
      mockPort = {
        postMessage: jest.fn(),
      };
      global.AudioWorkletProcessor = jest.fn();
      global.performance = {
        now: jest.fn(() => mockNow)
      };
      processor = new StreamProcessor();
      processor.port = mockPort;
    });

    test('should initialize with default buffer settings', () => {
      expect(processor.minBufferMs).toBe(100);
      expect(processor.maxBufferMs).toBe(1000);
      expect(processor.targetBufferMs).toBe(300);
      expect(processor.bufferPaddingMultiplier).toBe(1.5);
      expect(processor.poorRttThresholdMs).toBe(200);
    });

    test('should update buffer size based on RTT', () => {
      // Simulate good network conditions
      processor.updateRtt(50);
      expect(processor.networkQuality).toBeGreaterThan(0.8);
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'networkQuality',
          quality: expect.any(Number),
        })
      );

      // Simulate poor network conditions
      processor.updateRtt(300);
      expect(processor.networkQuality).toBeLessThan(0.5);
      expect(processor.playbackMinBuffers).toBeGreaterThan(36); // Default value
    });

    test('should handle buffer underruns', () => {
      // Simulate playback with low buffer
      processor.isInPlayback = true;
      processor.outputBuffers = [new Float32Array(128)];
      
      const outputs = [[new Float32Array(128)]];
      processor.process([], outputs, {});

      expect(processor.underrunCount).toBe(1);
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'bufferUnderrun',
          count: 1
        })
      );
    });

    test('should handle buffer overruns', () => {
      // Simulate playback with high buffer
      processor.isInPlayback = true;
      processor.playbackMinBuffers = 10;
      processor.outputBuffers = Array(20).fill(new Float32Array(128));
      
      const outputs = [[new Float32Array(128)]];
      processor.process([], outputs, {});

      expect(processor.overrunCount).toBe(1);
      expect(mockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'bufferOverrun',
          count: 1
        })
      );
    });

    test('should calculate RTT variance correctly', () => {
      processor.updateRtt(100);
      processor.updateRtt(200);
      processor.updateRtt(150);
      
      const variance = processor.getRttVariance();
      expect(variance).toBeGreaterThan(0);
      expect(variance).toBeLessThan(100);
    });

    test('should adjust buffer size based on recent underruns', () => {
      // Simulate recent underrun with poor network conditions
      processor.underrunCount = 1;
      mockNow = 2000;
      processor.lastUnderrunTime = mockNow - 1000;
      processor.rttSamples = [300, 350, 400]; // Add high RTT samples
      
      const initialBuffers = processor.playbackMinBuffers;
      processor.updateBufferSize();
      
      // Buffer size should increase due to both underrun and poor RTT
      expect(processor.playbackMinBuffers).toBeGreaterThan(initialBuffers);
    });

    test('should configure buffer settings via message', () => {
      const config = {
        minBufferMs: 200,
        maxBufferMs: 2000,
        targetBufferMs: 500,
        bufferPaddingMultiplier: 2.0,
        poorRttThresholdMs: 300
      };

      processor.receive({
        data: {
          event: 'configure',
          config
        }
      });

      expect(processor.minBufferMs).toBe(config.minBufferMs);
      expect(processor.maxBufferMs).toBe(config.maxBufferMs);
      expect(processor.targetBufferMs).toBe(config.targetBufferMs);
      expect(processor.bufferPaddingMultiplier).toBe(config.bufferPaddingMultiplier);
      expect(processor.poorRttThresholdMs).toBe(config.poorRttThresholdMs);
    });
  });
}); 
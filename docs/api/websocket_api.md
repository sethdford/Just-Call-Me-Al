# WebSocket API Documentation

This document describes the WebSocket API used for real-time communication between clients and the CSM server.

## Connection Establishment

Connect to the WebSocket server at:
```
ws://<server-address>:<port>/ws
```
or for secure connections:
```
wss://<server-address>:<port>/ws
```

By default, the server runs on port 8080.

## Message Format

All messages are exchanged in JSON format. Each message has a `type` field that determines the structure of the remaining fields.

### Client to Server Messages

#### Text Input

Used to send text that should be synthesized into speech.

```json
{
  "type": "text_input",
  "text": "Hello, how are you today?",
  "options": {
    "voice": "default",
    "speed": 1.0,
    "emotion": "neutral"
  }
}
```

Parameters:
- `text` (required): The text to be synthesized into speech
- `options` (optional): Configuration for speech synthesis
  - `voice`: Voice identifier (default: "default")
  - `speed`: Speaking rate multiplier (default: 1.0)
  - `emotion`: Emotional style (default: "neutral")

#### Audio Input

Used to send audio data for processing.

```json
{
  "type": "audio_input",
  "format": "raw",
  "sampleRate": 16000,
  "data": "<base64-encoded audio data>"
}
```

Parameters:
- `format` (required): Audio format ("raw", "wav", "opus")
- `sampleRate` (required): Sample rate in Hz
- `data` (required): Base64-encoded audio data

#### Control Message

Used to control ongoing operations.

```json
{
  "type": "control",
  "action": "cancel",
  "target": "current_synthesis"
}
```

Parameters:
- `action` (required): Control action ("cancel", "pause", "resume")
- `target` (required): Target of the action ("current_synthesis", "all")

#### Configuration

Used to update session configuration.

```json
{
  "type": "config",
  "settings": {
    "outputSampleRate": 24000,
    "bufferSettings": {
      "minBufferMs": 100,
      "targetBufferMs": 500,
      "maxBufferMs": 2000
    }
  }
}
```

Parameters:
- `settings` (required): Configuration settings to update

### Server to Client Messages

#### Audio Data

Sent when audio data is available.

```json
{
  "type": "audio_data",
  "format": "raw",
  "sampleRate": 24000,
  "timestamp": 1234567890,
  "data": "<base64-encoded audio data>",
  "final": false
}
```

Fields:
- `format`: Audio format ("raw", "wav", "opus")
- `sampleRate`: Sample rate in Hz
- `timestamp`: Timestamp in milliseconds
- `data`: Base64-encoded audio data
- `final`: Boolean indicating if this is the final chunk

#### Status Update

Sent to update client on server status.

```json
{
  "type": "status",
  "code": 200,
  "message": "Processing",
  "details": {
    "progress": 0.5,
    "estimatedTimeRemaining": 2000
  }
}
```

Fields:
- `code`: Status code (200: OK, 4xx: Client Error, 5xx: Server Error)
- `message`: Human-readable status message
- `details`: Additional status details (optional)

#### Error Message

Sent when an error occurs.

```json
{
  "type": "error",
  "code": 400,
  "message": "Invalid input format",
  "details": "Audio format 'mp3' is not supported"
}
```

Fields:
- `code`: Error code (same as HTTP status codes)
- `message`: Brief error description
- `details`: Detailed error information (optional)

## Protocol Flow Examples

### Text-to-Speech Synthesis

1. Client establishes WebSocket connection
2. Client sends text input message:
   ```json
   {
     "type": "text_input",
     "text": "Hello, how are you today?",
     "options": {
       "voice": "default",
       "speed": 1.0
     }
   }
   ```
3. Server responds with status update:
   ```json
   {
     "type": "status",
     "code": 200,
     "message": "Processing"
   }
   ```
4. Server streams audio data in chunks:
   ```json
   {
     "type": "audio_data",
     "format": "raw",
     "sampleRate": 24000,
     "timestamp": 1234567890,
     "data": "base64-encoded-chunk-1",
     "final": false
   }
   ```
   ```json
   {
     "type": "audio_data",
     "format": "raw",
     "sampleRate": 24000,
     "timestamp": 1234567891,
     "data": "base64-encoded-chunk-2",
     "final": false
   }
   ```
5. Server sends final audio chunk:
   ```json
   {
     "type": "audio_data",
     "format": "raw",
     "sampleRate": 24000,
     "timestamp": 1234567892,
     "data": "base64-encoded-chunk-3",
     "final": true
   }
   ```
6. Server indicates completion:
   ```json
   {
     "type": "status",
     "code": 200,
     "message": "Completed"
   }
   ```

### Error Handling

If an error occurs during processing:

```json
{
  "type": "error",
  "code": 500,
  "message": "Synthesis failed",
  "details": "Model inference error: out of memory"
}
```

## Timeouts and Connection Management

- The server expects a ping message every 30 seconds to keep the connection alive
- The server will close inactive connections after 120 seconds of inactivity
- If no response is received within 10 seconds of sending a message, clients should consider reconnecting

## Rate Limiting

- Maximum message rate: 10 messages per second
- Maximum text input length: 1000 characters
- Maximum audio input duration: 60 seconds 
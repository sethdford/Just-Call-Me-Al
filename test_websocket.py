import asyncio
import websockets
import json
import sys
import time
import wave
import io
import numpy as np

async def send_audio_file(uri):
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Wait for the ready message
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # You can replace this with your own audio file
        # This just generates 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 3  # seconds
        audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
        
        # Convert to int16 and then to bytes
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
        print(f"Sending {len(audio_bytes)} bytes of audio data")
        await websocket.send(audio_bytes)
        
        # Wait for response (might be an error if ASR is not available)
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            print(f"Received: {response}")
            response_json = json.loads(response)
            if response_json.get("type") == "Error":
                print("ASR error received, continuing with text test...")
            elif response_json.get("type") == "FullTranscript":
                print(f"Transcript: {response_json.get('transcript', '')}")
        except asyncio.TimeoutError:
            print("Timeout waiting for ASR response, continuing with text test...")
        except Exception as e:
            print(f"Error processing ASR response: {e}, continuing with text test...")
        
        # Send a text message for TTS
        text_msg = json.dumps({"type": "TextData", "text": "Hello, how are you?"})
        print(f"Sending text: {text_msg}")
        await websocket.send(text_msg)
        
        # Wait for response (synthesized audio)
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            print(f"Received response to text message - binary data length: {len(response)}")
        except asyncio.TimeoutError:
            print("Timeout waiting for synthesized audio")
        except Exception as e:
            print(f"Error receiving TTS response: {e}")
        
        # Send stop message
        stop_msg = json.dumps({"type": "Stop"})
        print(f"Sending stop message")
        await websocket.send(stop_msg)
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"Received final response: {response}")
        except asyncio.TimeoutError:
            print("No final response received")
        except Exception as e:
            print(f"Error receiving final response: {e}")

if __name__ == "__main__":
    uri = "ws://localhost:8001/ws"
    asyncio.run(send_audio_file(uri)) 
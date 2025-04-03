# CSM (Conversational Speech Model)

A high-performance Rust implementation of a conversational speech model based on the LLaMA architecture. This project enables real-time audio processing and generation, making it ideal for applications requiring natural, context-aware speech interactions.

## Inspiration

This project is inspired by [Sesame](https://www.sesame.com/), who are pioneering the development of lifelike computer companions through natural voice interactions. Their work on crossing the uncanny valley of conversational voice has been particularly influential in shaping this implementation.

### Technical Alignment with Sesame's Approach

1. **Natural Voice Processing**
   - Multi-codebook tokenization for rich speech representation
   - Context-aware audio generation
   - Real-time streaming capabilities
   - High-fidelity audio output

2. **Architecture Innovations**
   - LLaMA-based transformer with optimized attention mechanisms
   - Efficient key-value caching for faster inference
   - Rotary positional embeddings for better sequence modeling
   - Layer normalization and residual connections

3. **Crossing the Uncanny Valley**
   - Natural prosody and intonation modeling
   - Contextual awareness in responses
   - Emotional intelligence in voice generation
   - Consistent personality across interactions

4. **Performance Optimization**
   - Zero-copy audio processing
   - Efficient memory management
   - GPU acceleration support
   - Streaming inference capabilities

You can try their research demo at [Sesame's website](https://www.sesame.com/) to experience their vision of natural voice companions.

### Key Technical Differences

While inspired by Sesame's work, this implementation offers several unique advantages:

1. **Rust Implementation**
   - Memory safety guarantees
   - Zero-cost abstractions
   - Thread-safe design
   - Minimal runtime overhead

2. **Modular Architecture**
   - Pluggable model components
   - Configurable audio processing
   - Extensible tokenization system
   - Customizable generation parameters

3. **Development Focus**
   - Open-source implementation
   - Community-driven development
   - Cross-platform support
   - Easy integration capabilities

## Why Use CSM?

1. **Performance & Efficiency**
   - Written in Rust for maximum performance and memory safety
   - Leverages `tch-rs` for efficient tensor operations
   - Supports both CPU and GPU acceleration
   - Zero-cost abstractions and minimal runtime overhead

2. **Audio Processing Capabilities**
   - Real-time audio input/output processing
   - Multi-codebook audio tokenization for rich speech representation
   - High-quality WAV file support
   - Configurable audio parameters (sample rate, channels, bit depth)

3. **Advanced Model Architecture**
   - Based on the proven LLaMA transformer architecture
   - Multi-head attention with key-value caching
   - Rotary positional embeddings for better sequence modeling
   - Layer normalization and residual connections
   - Configurable model size and parameters

4. **Use Cases**
   - Voice assistants and chatbots
   - Real-time speech-to-speech translation
   - Audio content generation
   - Speech style transfer
   - Conversational AI applications
   - Audio data augmentation

## Features

- Audio processing with WAV file support
- LLaMA-based transformer architecture for text-to-semantic token generation
- Multi-codebook audio tokenization (RVQ) for rich speech representation
- **Vocoder based on Mimi Neural Audio Codec** (from Kyutai Labs' Moshi project) for high-fidelity audio synthesis
- Configurable model parameters
- CPU/GPU support via tch-rs
- Efficient memory management
- Thread-safe design
- Comprehensive error handling
- Real-time voice conversation through WebSocket
- Audio visualization
- `anyhow`
- `thiserror`
- `tracing`
- `serde`
- `serde_json`
- `clap`
- `rand`
- `rand_chacha`
- `rustfft` (for potential Mel Spectrogram implementation)
- `candle` / `candle-nn` (potentially used by integrated Mimi components)
- **Mimi Neural Audio Codec** (via Moshi project - Kyutai Labs) for vocoding

## Real-Time Voice Demo

The project includes a real-time voice demo that processes audio from your microphone.

### Features
- WebSocket server for real-time audio streaming
- Browser-based interface for microphone input
- Low-latency audio processing

### Prerequisites
- Install required dependencies (see [Installation](#installation))
- Ensure your microphone is properly configured
- Modern web browser that supports WebSockets and the Web Audio API

### Running the Demo
1. Build and start the WebSocket server:
```bash
cargo run --bin csm
```
2. Open `http://localhost:3000` in your web browser
3. Grant microphone permissions when prompted
4. Start speaking to see real-time transcription and responses

## Known Issues and Troubleshooting

### PyTorch Compatibility Issues

This project uses the `tch` crate (Rust bindings for PyTorch) which requires a compatible version of PyTorch's C++ library (libtorch) to be installed on your system.

#### Apple Silicon (M1/M2/M3) Compatibility

If you're running on Apple Silicon (arm64 architecture), you may encounter linking issues when building the project. This is because the default `libtorch` download is built for x86_64 architecture.

To fix this:

1. Download the correct PyTorch version for your architecture:
```bash
# For Apple Silicon (ARM64)
curl -O https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

2. Set the environment variable to point to the downloaded library:
```bash
export LIBTORCH=$(pwd)/libtorch
```

3. Build with version bypass if needed:
```bash
LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build
```

#### Version Compatibility

The `tch` crate version in this project (0.14.0) is compatible with PyTorch 2.1.0. Using a different version of PyTorch might cause version mismatch errors.

If you see errors like "undefined symbols" or "version mismatch", try:

1. Check the installed PyTorch version:
```bash
python -c "import torch; print(torch.__version__)"
```

2. Make sure the tch-rs version in Cargo.toml matches your PyTorch version:
- PyTorch 2.0.x → tch = "0.13.0"
- PyTorch 2.1.x → tch = "0.14.0"
- PyTorch 2.2.x → tch = "0.15.0"

3. If needed, modify the `tch` version in `Cargo.toml` to match your PyTorch installation.

#### PyTorch 2.5.1+ Compatibility

For PyTorch 2.5.1 and newer versions, especially on Apple Silicon Macs, we've created a patched version of tch-rs that addresses compatibility issues. See `PyTorch_2.5.1_Setup.md` for detailed instructions on using this solution.

## Prerequisites

- Rust 1.70 or later
- libtorch (PyTorch C++ API) - See note below for version compatibility
- FFmpeg (for potential future audio processing tasks, not strictly required currently)
- **Pre-trained Mimi Vocoder Weights:** The vocoder requires the weights file located at `models/mimi/model.safetensors`. Ensure this file is present before running.

> **Note on compatibility:** The project requires specific versions of PyTorch and the tch-rs crate to work correctly. The tch-rs crate has strict version compatibility requirements with PyTorch. If you encounter build errors, verify that you are using compatible versions:
> - For tch-rs 0.13.0: Use PyTorch 2.0.0
> - For tch-rs 0.14.0: Use PyTorch 2.1.0
> - For tch-rs 0.15.0: Use PyTorch 2.2.0
>
> You can modify the tch dependency version in Cargo.toml to match your installed PyTorch version.

### Installing libtorch

#### macOS

1. Download the latest libtorch release from [PyTorch website](https://pytorch.org/get-started/locally/):
```bash
# For CPU only
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
# OR for CUDA support (if applicable)
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

2. Extract the archive:
```bash
unzip libtorch-macos-2.1.0.zip
```

3. Set the LIBTORCH environment variable:
```bash
export LIBTORCH=$(pwd)/libtorch
```

4. Add to your .zshrc or .bashrc for persistence:
```bash
echo 'export LIBTORCH=<path-to-extracted-libtorch-folder>' >> ~/.zshrc
# OR
echo 'export LIBTORCH=<path-to-extracted-libtorch-folder>' >> ~/.bashrc
```

#### Linux

1. Download the latest libtorch release:
```bash
# For CPU only
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
# OR for CUDA support
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

2. Extract the archive:
```bash
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

3. Set the LIBTORCH environment variable:
```bash
export LIBTORCH=$(pwd)/libtorch
```

4. Add to your .bashrc for persistence:
```bash
echo 'export LIBTORCH=<path-to-extracted-libtorch-folder>' >> ~/.bashrc
```

#### Windows

1. Download the latest libtorch release from [PyTorch website](https://pytorch.org/get-started/locally/)

2. Extract the zip file to a known location

3. Set the LIBTORCH environment variable in PowerShell:
```powershell
$env:LIBTORCH = "C:\path\to\libtorch"
```

4. To make it persistent, add it to your system environment variables

#### Using PyTorch from Python (Alternative)

If you already have PyTorch installed in your Python environment, you can use it instead of downloading libtorch separately:

```bash
# Make sure PyTorch is installed
pip install torch

# Build with PyTorch environment
LIBTORCH_USE_PYTORCH=1 cargo build
```

### Installing FFmpeg

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows
Download from [FFmpeg website](https://ffmpeg.org/download.html) or install using Chocolatey:
```powershell
choco install ffmpeg
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sethdford/csm.git
cd csm
```

2. Build the project:
```bash
cargo build --release
```

## Usage

1. Generate model weights:
```bash
cargo run --bin create_model
```

2. Generate test audio:
```bash
cargo run --example generate_test_audio
```

3. Process audio with the model:
```bash
cargo run --bin csm -- -w models/llama-1B.pth -f llama-1B -i audio/conversational_a.wav -o output.wav
```

4. Start the real-time voice demo server:
```bash
cargo run --bin csm_server -- -w models/llama-1B.pth -f llama-1B
```

## Command Line Arguments

### CSM Binary
- `-w, --model-path`: Path to model weights file
- `-f, --model-flavor`: Model flavor (e.g., llama-1B)
- `-i, --input-file`: Input audio file
- `-o, --output-file`: Output audio file
- `-m, --max-new-tokens`: Maximum number of new tokens to generate (default: 100)
- `-t, --temperature`: Temperature for sampling (default: 0.8)
- `-k, --top-k`: Top-k for sampling (default: 50)
- `-p, --top-p`: Top-p for sampling (default: 0.9)

### CSM Server Binary
- `-w, --model-path`: Path to model weights file
- `-f, --model-flavor`: Model flavor (e.g., llama-1B)
- `-p, --port`: Port to run the WebSocket server on (default: 8080)

## Project Structure

```
src/
├── audio/         # Audio processing module
│   ├── mod.rs     # WAV handling and tokenization
│   └── streaming.rs # Real-time audio processing
├── models/        # Model implementations
│   ├── mod.rs     # Model traits and common types
│   └── llama.rs   # LLaMA model implementation
├── utils/         # Utility functions
│   └── mod.rs     # Tensor operations and helpers
├── server/        # WebSocket server implementation
│   ├── mod.rs     # Server module exports
│   └── websocket.rs # WebSocket communication
└── main.rs        # Main program entry point
```

## Performance Considerations

- Uses efficient tensor operations with `tch-rs`
- Implements streaming audio processing
- Optimized memory usage with zero-copy operations
- Supports batch processing for better throughput
- Configurable model size for different performance requirements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Sesame](https://www.sesame.com/) for their groundbreaking work in natural voice companions
- LLaMA model architecture by Meta AI
- PyTorch and tch-rs teams for the excellent tensor library
- Rust community for the amazing ecosystem
- Contributors and maintainers of all dependencies

## Related Projects

- [Sesame Research](https://www.sesame.com/research) - Original research on natural voice companions
- [Sesame Demo](https://www.sesame.com/demo) - Interactive demo of their voice technology
- [Sesame Team](https://www.sesame.com/team) - Meet the team behind the technology
- [Sesame Careers](https://www.sesame.com/careers) - Join the team advancing voice technology
- [Haadesx/realtime-voice-csm](https://github.com/Haadesx/realtime-voice-csm) - Python implementation that inspired our real-time demo 
- 
## Native Speech Synthesis with tch-rs

This project now includes a native Rust implementation of the Conversational Speech Model (CSM) using the `tch-rs` library (PyTorch C++ bindings). This allows using the CSM-1B model from Hugging Face directly in Rust without requiring a Python bridge.

### Benefits

- Better performance: Direct C++ bindings offer improved performance over Python bridges
- No Python dependency: No need for Python runtime or managing Python dependencies
- Simpler deployment: Single binary that includes all necessary functionality
- Resource efficiency: Lower memory overhead and better CPU/GPU utilization

### Usage

1. Download the CSM-1B model:
   ```bash
   ./scripts/download_csm_model.sh
   ```

2. Run the voice assistant with the native CSM model:
   ```bash
   cargo run --bin voice_assistant -- -c models/csm-1b -t real
   ```

For more detailed information, see [CSM_NATIVE.md](CSM_NATIVE.md).

# CSM Voice Assistant

This project implements a voice assistant with both speech-to-text and text-to-speech capabilities.

## Features

- Speech-to-text using Whisper or DeepSpeech
- Text-to-speech with multiple options:
  - Basic formant-based synthesis (primitive)
  - System TTS via the `tts` crate
  - Neural TTS with RVQ tokenization (high quality)
- WebSocket server for client integration
- Command-line interface

## Neural TTS System

The Neural Text-to-Speech system uses a modern two-transformer architecture:

1. **RVQ Tokenizer**: Converts audio to discrete tokens and back
2. **Backbone Transformer**: Text to semantic tokens
3. **Decoder Transformer**: Semantic tokens to acoustic tokens

Key features:
- High-quality speech synthesis
- Voice customization options
- Prosody control (rate, pitch, energy)
- Residual Vector Quantization for efficient audio representation

For more details, see the [Neural TTS documentation](docs/neural_tts.md).

## Setup

### Prerequisites

- Rust 1.65+
- libtorch (for neural models)
- OpenMP (for certain operations)

### Environment Setup

Set the following environment variables:

```bash
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
export LIBTORCH=/path/to/libtorch
export LIBTORCH_LIB=/path/to/libtorch/lib
```

### Building

```bash
cargo build --release
```

export LIBTORCH=/Users/sethford/Downloads/libtorch && export LIBTORCH_INCLUDE=/Users/sethford/Downloads/libtorch/include && export LIBTORCH_LIB=/Users/sethford/Downloads/libtorch/lib && export DYLD_LIBRARY_PATH=/Users/sethford/Downloads/libtorch/lib:$DYLD_LIBRARY_PATH && export CXXFLAGS="-I/Users/sethford/Downloads/libtorch/include -I/Users/sethford/Downloads/libtorch/include/torch/csrc/api/include -I/Users/sethford/Downloads/libtorch/include/torch/csrc -std=c++17" && export LIBRARY_PATH=/Users/sethford/Downloads/libtorch/lib:$LIBRARY_PATH && export MACOSX_DEPLOYMENT_TARGET=13.0 && cargo clean && cargo build

## Usage

### Voice Assistant

```bash
# Basic usage (simple TTS)
cargo run --bin voice_assistant -- --csm-path models/csm-1b

# With neural TTS (higher quality)
cargo run --bin voice_assistant -- --csm-path models/csm-1b --use-neural-tts

# With Whisper STT
cargo run --bin voice_assistant -- --csm-path models/csm-1b --use-whisper
```

### Neural TTS CLI

```bash
# Interactive mode
cargo run --bin neural_tts_cli -- --model-dir models/csm-1b/neural_tts

# Batch mode
cargo run --bin neural_tts_cli -- --model-dir models/csm-1b/neural_tts --interactive false --text "Hello, world!"
```

### WebSocket Server

```bash
cargo run --bin csm_server -- --models-dir models
```

## Model Preparation

### Setting up Neural TTS

1. Create model directory structure:
```bash
cargo run --release --bin setup_neural_tts -- --output-dir models/csm-1b/neural_tts
```

2. Train models (if you have data):
```bash
cargo run --release --bin train_neural_tts -- --data-dir datasets/audio --output-dir models/csm-1b/neural_tts
```

3. For testing/development, you can initialize with random weights:
```bash
cargo run --release --bin setup_neural_tts -- --output-dir models/csm-1b/neural_tts --init-defaults true
```

## License

[MIT License](LICENSE)

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Sesame Research: Crossing the Uncanny Valley of Voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
- [PyTorch](https://pytorch.org/)
- [tch-rs: Rust bindings for PyTorch](https://github.com/LaurentMazare/tch-rs)
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- [safetensors](https://github.com/huggingface/safetensors)
- **[Moshi Project (Kyutai Labs)](https://github.com/kyutai-labs/moshi)** - Source of the Mimi Neural Audio Codec used for vocoding. 
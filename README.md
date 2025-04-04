# CSM (Conversational Speech Model)

A high-performance Rust implementation of a conversational speech model based on the LLaMA architecture. This project enables real-time audio processing and generation for natural, context-aware speech interactions.

## Overview

CSM is a speech synthesis and conversation system that aims to create lifelike voice interactions through:

- Real-time streaming audio synthesis
- Contextual conversation understanding
- Multi-codebook audio tokenization (RVQ)
- Emotional expressivity and natural prosody
- High-performance Rust implementation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sethdford/csm.git
cd csm

# Install dependencies (see Prerequisites section)
# Configure PyTorch/libtorch

# Build the project
cargo build --release

# Start the WebSocket server
cargo run --bin csm_server -- -w models/llama-1B.pth -f llama-1B

# Open the demo in your browser
open http://localhost:3000
```

## Features

- **Two-Stage Architecture** - Backbone for text/semantics and Decoder for acoustic tokens
- **WebSocket API** - Real-time bidirectional communication for streaming audio
- **Multi-codebook Tokenization** - Rich speech representation using RVQ
- **LLaMA-based Transformer** - Advanced architecture for high-quality synthesis
- **Conversation Context Management** - Maintains conversation history for coherent interactions
- **Emotional Expressivity** - Fine-tuned for natural speech with appropriate emotions
- **Rust Performance** - Memory safety with high performance and thread-safety

## Prerequisites

- Rust 1.70 or later
- libtorch (PyTorch C++ API) - Version compatibility is crucial (see [Troubleshooting](#troubleshooting))
- Pre-trained model weights
- Modern web browser for the demo interface

Detailed installation instructions are available in our [Installation Guide](docs/installation_guide.md).

## Usage

### Command Line Arguments

```bash
# Process audio file
cargo run --bin csm -- \
  -w models/llama-1B.pth \
  -f llama-1B \
  -i audio/input.wav \
  -o output.wav \
  -t 0.8 -p 0.9 -k 50

# Start WebSocket server
cargo run --bin csm_server -- \
  -w models/llama-1B.pth \
  -f llama-1B \
  -p 8080
```

### API

The system provides a WebSocket API for real-time communication. See our [WebSocket API Documentation](docs/api/websocket_api.md) for detailed information about connection establishment, message formats, and protocol flow.

## Documentation

We maintain comprehensive documentation to help developers understand and extend CSM:

- [Architecture Overview](docs/architecture/system_architecture.md) - System components and data flow
- [WebSocket API](docs/api/websocket_api.md) - Complete API reference
- [Documentation Standards](docs/documentation_standards.md) - Our documentation guidelines
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to CSM

## Project Structure

```
src/
├── audio/         # Audio processing module
├── models/        # Model implementations including backbone & decoder
├── rvq/           # Residual Vector Quantization implementation
├── server/        # WebSocket server implementation
├── tokenization/  # Text and audio tokenization
├── utils/         # Utility functions and helpers
├── context.rs     # Conversation context management
├── errors.rs      # Error types and handling
├── vocoder.rs     # Audio waveform generation
├── websocket.rs   # WebSocket communication
├── lib.rs         # Library exports
└── main.rs        # Main program entry points
```

## Troubleshooting

### PyTorch Compatibility

The project requires specific versions of PyTorch and the tch-rs crate:

- For tch-rs 0.13.0: Use PyTorch 2.0.0
- For tch-rs 0.14.0: Use PyTorch 2.1.0
- For tch-rs 0.15.0: Use PyTorch 2.2.0

If you encounter build errors related to PyTorch, see our detailed [Troubleshooting Guide](docs/troubleshooting.md).

### Apple Silicon (M1/M2/M3) Compatibility

For Apple Silicon users, specialized setup may be required:

```bash
# For Apple Silicon (ARM64)
curl -O https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
export LIBTORCH=$(pwd)/libtorch
LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and organization
- Pull request process
- Documentation requirements
- Testing standards

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Sesame](https://www.sesame.com/) for their groundbreaking research in natural voice companions
- LLaMA model architecture by Meta AI
- PyTorch and tch-rs teams
- Rust community for the amazing ecosystem
- Contributors and maintainers of all dependencies 
# CSM Installation Guide

This guide provides detailed instructions for setting up the CSM project on various platforms. Follow these steps to ensure a smooth installation process.

## Prerequisites

Before installing CSM, ensure you have:

1. **Rust Toolchain** - Version 1.70 or later
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   # Or update existing installation:
   rustup update
   ```

2. **Build Tools**
   - **macOS**: Xcode Command Line Tools
     ```bash
     xcode-select --install
     ```
   - **Linux**: 
     ```bash
     sudo apt-get update
     sudo apt-get install build-essential
     ```
   - **Windows**: Microsoft Visual C++ Build Tools

3. **Git**
   ```bash
   # macOS
   brew install git
   
   # Linux
   sudo apt-get install git
   
   # Windows: download from git-scm.com
   ```

## Setting Up PyTorch/libtorch

CSM relies on the PyTorch C++ API (libtorch). Setting it up correctly is crucial for the project to work.

### Version Compatibility

Ensure you use compatible versions of libtorch and the tch-rs crate:
- tch-rs 0.13.0 → PyTorch 2.0.0
- tch-rs 0.14.0 → PyTorch 2.1.0
- tch-rs 0.15.0 → PyTorch 2.2.0

### macOS

```bash
# Download libtorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip

# Set environment variables
export LIBTORCH=$(pwd)/libtorch
echo 'export LIBTORCH='$(pwd)'/libtorch' >> ~/.zshrc  # or ~/.bashrc
```

#### Apple Silicon (M1/M2/M3) Specific Setup

For Apple Silicon Macs, additional steps are needed:

```bash
# Download ARM-compatible version
curl -O https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Set environment variables
export LIBTORCH=$(pwd)/libtorch
echo 'export LIBTORCH='$(pwd)'/libtorch' >> ~/.zshrc

# Use version bypass when building
LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build
```

### Linux

```bash
# Download libtorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# For CUDA support (if GPU available):
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip

# Extract the archive
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Set environment variables
export LIBTORCH=$(pwd)/libtorch
echo 'export LIBTORCH='$(pwd)'/libtorch' >> ~/.bashrc
```

### Windows

1. Download libtorch from the [PyTorch website](https://pytorch.org/get-started/locally/)
2. Extract to a location like `C:\libtorch`
3. Set environment variables:
   ```powershell
   $env:LIBTORCH = "C:\libtorch"
   ```
4. For permanent setup, add to system environment variables

### Alternative: Using PyTorch from Python

If you already have PyTorch installed in a Python environment, you can use it directly:

```bash
# Install PyTorch if not already installed
pip install torch

# Build using the Python installation
LIBTORCH_USE_PYTORCH=1 cargo build
```

## Downloading Model Weights

CSM requires pre-trained model weights to function correctly:

```bash
# Create models directory
mkdir -p models/mimi

# Download weights (replace with actual download location)
# This is an example - ensure you have the correct URL
wget -O models/mimi/model.safetensors https://example.com/path/to/model.safetensors
```

## Building the Project

Once all dependencies are installed, build the project:

```bash
# Clone the repository
git clone https://github.com/sethdford/csm.git
cd csm

# Build release version
cargo build --release
```

## Verifying Installation

Verify your installation with:

```bash
# Run a simple test
cargo run --bin csm -- --help

# If you have model weights, test audio processing
cargo run --bin csm -- -w models/llama-1B.pth -f llama-1B -i test_audio.wav -o output.wav
```

## Troubleshooting

### Common Issues

1. **Missing libtorch**
   - Error: `library not found for -ltorch`
   - Solution: Ensure LIBTORCH environment variable is set correctly

2. **Version mismatch**
   - Error: `undefined symbols` or `version mismatch`
   - Solution: Check tch-rs version in Cargo.toml matches your PyTorch version

3. **Compilation fails on Apple Silicon**
   - Error: Architecture-related errors
   - Solution: Use LIBTORCH_BYPASS_VERSION_CHECK=1 and ARM-compatible libtorch

4. **Missing model weights**
   - Error: `Failed to load model: weights file not found`
   - Solution: Ensure model weights are downloaded to the correct location

### Getting Help

If you encounter issues not addressed in this guide:

1. Check the [GitHub issues](https://github.com/sethdford/csm/issues) for similar problems
2. Consult the [PyTorch C++ documentation](https://pytorch.org/cppdocs/)
3. Ask for help in the project's discussion forums

## Next Steps

After successful installation:

1. Check out the [WebSocket API Documentation](api/websocket_api.md)
2. Read the [System Architecture](architecture/system_architecture.md) documentation
3. Try the demo application at `http://localhost:3000` after starting the server 
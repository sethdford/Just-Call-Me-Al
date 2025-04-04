# CSM Troubleshooting Guide

This guide addresses common issues that developers might encounter when working with the CSM project and provides solutions for resolving them.

## PyTorch/libtorch Compatibility Issues

### Version Mismatch Errors

**Problem**: You encounter errors like "undefined symbols" or "version mismatch" when building the project.

**Solution**:
1. Check which version of PyTorch/libtorch you have installed:
   ```bash
   # If using Python PyTorch
   python -c "import torch; print(torch.__version__)"
   ```

2. Ensure the `tch` crate version in `Cargo.toml` matches your PyTorch version:
   - PyTorch 2.0.x → tch = "0.13.0"
   - PyTorch 2.1.x → tch = "0.14.0"
   - PyTorch 2.2.x → tch = "0.15.0"

3. Update `Cargo.toml` if necessary:
   ```toml
   [dependencies]
   tch = "0.14.0"  # Adjust version number as needed
   ```

4. Clean and rebuild:
   ```bash
   cargo clean
   cargo build
   ```

### Apple Silicon (M1/M2/M3) Issues

**Problem**: Building fails on Apple Silicon (arm64 architecture) Macs with architecture mismatch errors.

**Solution**:
1. Download the ARM-compatible version of libtorch:
   ```bash
   curl -O https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
   ```

2. Set the environment variable:
   ```bash
   export LIBTORCH=$(pwd)/libtorch
   ```

3. Build with version bypass:
   ```bash
   LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build
   ```

4. If issues persist, try building with Rosetta 2:
   ```bash
   arch -x86_64 cargo build
   ```

### Library Not Found Errors

**Problem**: You see errors like "library not found for -ltorch" or similar.

**Solution**:
1. Verify that the LIBTORCH environment variable is set correctly:
   ```bash
   echo $LIBTORCH
   ```

2. Check that the library files exist:
   ```bash
   ls $LIBTORCH/lib
   ```

3. Ensure the dynamic library path includes libtorch:
   ```bash
   # macOS
   export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
   
   # Linux
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
   ```

## Model Weight Issues

### Model Loading Fails

**Problem**: Errors occur when trying to load model weights.

**Solution**:
1. Verify that the model files exist in the expected location:
   ```bash
   ls -la models/mimi/model.safetensors
   ls -la models/llama-1B.pth  # Or whatever model you're using
   ```

2. Check file permissions:
   ```bash
   chmod 644 models/mimi/model.safetensors
   ```

3. If using paths with spaces or special characters, use quotes:
   ```bash
   cargo run --bin csm -- -w "path/with spaces/model.pth" -f llama-1B
   ```

## WebSocket Server Issues

### Connection Refused

**Problem**: Unable to connect to the WebSocket server.

**Solution**:
1. Verify the server is running:
   ```bash
   ps aux | grep csm_server
   ```

2. Check that the port is not blocked by a firewall:
   ```bash
   # Linux
   sudo ufw status
   
   # macOS
   sudo lsof -i :8080
   ```

3. Try a different port by using the `-p` option when starting the server.

### Audio Streaming Issues

**Problem**: Audio is choppy, delayed, or not playing at all.

**Solution**:
1. Increase buffer sizes in the client application.
2. Check network latency between client and server.
3. Verify audio device settings and permissions in the browser.
4. Try a different browser to rule out browser-specific issues.

## Build Errors

### Compiler Errors

**Problem**: Rust compiler errors when building.

**Solution**:
1. Update your Rust toolchain:
   ```bash
   rustup update
   ```

2. Clean and rebuild:
   ```bash
   cargo clean
   cargo build
   ```

3. Check for missing dependencies:
   ```bash
   # Linux
   sudo apt-get install build-essential libssl-dev pkg-config
   
   # macOS
   brew install openssl pkg-config
   ```

### Dependency Resolution Failures

**Problem**: Cargo fails to resolve dependencies.

**Solution**:
1. Update the package index:
   ```bash
   cargo update
   ```

2. If there are conflicting versions, try specifying exact versions in Cargo.toml.

3. Check your internet connection and Cargo configuration.

## Audio Processing Issues

### Vocoder Problems

**Problem**: Poor audio quality or vocoder errors.

**Solution**:
1. Verify that the vocoder model is loaded correctly.
2. Check audio sample rate compatibility (vocoder may expect a specific rate).
3. Adjust input audio format if necessary:
   ```bash
   # Using ffmpeg to convert to the right format
   ffmpeg -i input.wav -ar 24000 -ac 1 -c:a pcm_s16le output.wav
   ```

## Performance Issues

### High CPU/Memory Usage

**Problem**: The application uses excessive CPU or memory resources.

**Solution**:
1. Configure smaller model variants if available.
2. Reduce batch size or buffer sizes.
3. Use GPU acceleration if available:
   ```bash
   # Set CUDA device
   export CUDA_VISIBLE_DEVICES=0
   ```

4. Profile the application to identify bottlenecks:
   ```bash
   # Linux
   perf record -g -- cargo run --release --bin csm_server
   perf report
   ```

## Getting Additional Help

If you encounter issues not covered in this guide:

1. **Check the GitHub Issues**: Search for existing issues that might describe your problem.

2. **Developer Documentation**: The project's documentation may have been updated with new solutions.

3. **Debug Mode**: Run with debug logging enabled:
   ```bash
   RUST_LOG=debug cargo run --bin csm_server
   ```

4. **Open a New Issue**: If you can't find a solution, open a new issue with:
   - A clear description of the problem
   - Steps to reproduce
   - System information (OS, Rust version, PyTorch version)
   - Relevant logs or error messages 
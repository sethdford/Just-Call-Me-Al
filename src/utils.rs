use tch::Device;
use anyhow::{Result, anyhow};

pub mod tensor;

pub fn get_device(device_str: Option<String>) -> Result<Device> {
    match device_str.as_deref() {
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda") | Some("gpu") if tch::utils::has_cuda() => Ok(Device::cuda_if_available()),
        Some("mps") if tch::utils::has_mps() => Ok(Device::Mps),
        None => { // Auto-detect: CUDA -> MPS -> CPU
            if tch::utils::has_cuda() {
                Ok(Device::cuda_if_available())
            } else if tch::utils::has_mps() {
                Ok(Device::Mps)
            } else {
                Ok(Device::Cpu)
            }
        }
        Some(other) => {
            // Handle cases where CUDA/MPS is requested but not available
            if (other == "cuda" || other == "gpu") && !tch::utils::has_cuda() {
                Err(anyhow!("CUDA device requested ('{}') but not available.", other))
            } else if other == "mps" && !tch::utils::has_mps() {
                Err(anyhow!("MPS device requested ('{}') but not available.", other))
            } else {
                Err(anyhow!("Unsupported or unavailable device specified: {}", other))
            }
        }
    }
} 
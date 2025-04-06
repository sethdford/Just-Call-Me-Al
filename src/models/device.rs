// Device abstraction for all models
use thiserror::Error;
use tracing::warn;

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(usize), // CUDA with ordinal/index
    Mps, // Metal Performance Shaders for Apple Silicon
    Vulkan, // For future support
}

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Device not available: {0}")]
    NotAvailable(String),
    #[error("Device error: {0}")]
    Other(String),
}

impl Device {
    /// Helper function to get CUDA device if available, or fall back to CPU
    pub fn cuda_if_available() -> Self {
        if tch::utils::has_cuda() {
            Self::Cuda(0)
        } else {
            warn!("CUDA requested but not available, falling back to CPU");
            Self::Cpu
        }
    }

    /// Helper function to get MPS device if available, or fall back to CPU
    pub fn mps_if_available() -> Self {
        if tch::utils::has_mps() {
            Self::Mps
        } else {
            warn!("MPS requested but not available, falling back to CPU");
            Self::Cpu
        }
    }

    /// Helper function to detect best available device
    pub fn auto_detect() -> Self {
        if tch::utils::has_cuda() {
            Self::Cuda(0)
        } else if tch::utils::has_mps() {
            Self::Mps
        } else {
            Self::Cpu
        }
    }

    /// Convert string to Device
    pub fn from_str(device_str: &str) -> Result<Self, DeviceError> {
        match device_str.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => {
                if tch::utils::has_cuda() {
                    Ok(Self::Cuda(0))
                } else {
                    Err(DeviceError::NotAvailable("CUDA".to_string()))
                }
            },
            "mps" | "metal" => {
                if tch::utils::has_mps() {
                    Ok(Self::Mps)
                } else {
                    Err(DeviceError::NotAvailable("MPS".to_string()))
                }
            },
            "vulkan" => {
                // Currently always return error as not implemented
                Err(DeviceError::NotAvailable("Vulkan".to_string()))
            },
            other => Err(DeviceError::Other(format!("Unknown device: {}", other))),
        }
    }

    /// Check if device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Check if device is CUDA
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Check if device is MPS
    pub fn is_mps(&self) -> bool {
        matches!(self, Self::Mps)
    }
} 
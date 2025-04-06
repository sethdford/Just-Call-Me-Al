//! This module provides compatibility for Moshi and its dependencies

#[cfg(feature = "enable_moshi")]
mod candle_compat {
    // Import types from half for compatibility
    use half::{bf16, f16};

    // Register a function to ensure this module is linked
    pub fn init_compat() {
        // This is a no-op function that forces the module to be included in the binary
    }

    // Helper functions for random number generation with half types
    // These can be used instead of direct rand crate functionality
    pub fn random_bf16(min: f32, max: f32) -> bf16 {
        let random_val = min + (max - min) * rand::random::<f32>();
        bf16::from_f32(random_val)
    }

    pub fn random_f16(min: f32, max: f32) -> f16 {
        let random_val = min + (max - min) * rand::random::<f32>();
        f16::from_f32(random_val)
    }
}

#[cfg(feature = "enable_moshi")]
pub use candle_compat::*;

#[cfg(not(feature = "enable_moshi"))]
pub mod candle_compat {
    // Empty implementation when Moshi is not enabled
    pub fn init_compat() {}
}

#[cfg(not(feature = "enable_moshi"))]
pub use candle_compat::*; 
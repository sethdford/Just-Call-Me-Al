//! Compatibility module for half and rand integration
//!
//! This module re-exports types from the moshi crate to provide
//! compatibility between half and rand.

// Remove the incorrect attempt to import from moshi
// pub use moshi::utils::{BF16Wrapper, F16Wrapper, UniformHalfExt};

// Restore the local rand_half module containing the type definitions
// Remove the conditional compilation attribute
// #[cfg(not(feature = "enable_moshi"))]
pub mod rand_half {
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};
    use half::{bf16, f16};

    pub type BF16Wrapper = bf16;
    pub type F16Wrapper = f16;

    pub trait UniformHalfExt {
        fn sample_f16<R: Rng + ?Sized>(&self, rng: &mut R) -> f16;
        fn sample_bf16<R: Rng + ?Sized>(&self, rng: &mut R) -> bf16;
    }

    impl<T: rand::distributions::uniform::SampleUniform> UniformHalfExt for Uniform<T>
    where
        f32: From<T>,
    {
        fn sample_f16<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
            f16::from_f32(f32::from(self.sample(rng)))
        }

        fn sample_bf16<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> half::bf16 {
            half::bf16::from_f32(f32::from(self.sample(rng)))
        }
    }
} 
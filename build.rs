// Simple build script to ensure bindgen is available for sentencepiece-sys
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // No additional steps needed, just having this file will ensure
    // that our build-dependencies (including bindgen) are properly available
    // to dependent build scripts.
} 
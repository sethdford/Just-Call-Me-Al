# Documentation Standards for CSM Project

## Overview
This document outlines the documentation standards for the Conversational Speech Model (CSM) project. Following these standards ensures consistent, high-quality documentation across the codebase, making it easier for new contributors to understand the project and for existing team members to maintain the code.

## Code Documentation

### Rust Docstrings
- All public items (functions, structs, enums, traits, modules) must have docstrings with format `///`.
- Documentation should include:
  - A brief one-line summary of what the item does
  - A more detailed explanation if needed
  - Examples where appropriate
  - Parameter descriptions
  - Return value descriptions
  - Error scenarios and how they're handled
  - Safety notes for unsafe code
  - Complexity and performance characteristics for critical operations

Example:
```rust
/// Converts audio data to RVQ tokens.
///
/// This function takes raw audio samples and converts them to RVQ tokens
/// using the configured codebooks. It handles streaming inputs efficiently.
///
/// # Examples
///
/// ```
/// let audio_data = vec![0.1, 0.2, 0.3];
/// let tokens = convert_audio_to_tokens(&audio_data, &config)?;
/// ```
///
/// # Parameters
///
/// * `audio_data` - The raw audio samples in the range [-1.0, 1.0]
/// * `config` - Configuration for tokenization including codebook selection
///
/// # Returns
///
/// A vector of RVQ tokens organized by codebook level.
///
/// # Errors
///
/// Returns an error if the audio data is invalid or if tokenization fails.
///
/// # Performance
///
/// Time complexity: O(n log m) where n is the number of samples and m is codebook size.
pub fn convert_audio_to_tokens(audio_data: &[f32], config: &TokenizerConfig) -> Result<Vec<Vec<u32>>> {
    // Implementation
}
```

### Private Code Documentation
- Private functions and complex logic should be documented with standard comments (`//`).
- Complex algorithms or logic should include explanatory comments inline.
- Avoid obvious comments that duplicate what the code clearly states.

### Module Documentation
- Each module should have doc comments at the top of the file explaining:
  - The purpose of the module
  - Key components/types in the module
  - How this module fits into the larger system
  - Any important patterns or conventions used

## API Documentation

### RESTful/WebSocket API
- Use OpenAPI/Swagger for REST API documentation.
- For WebSocket APIs, document:
  - Connection establishment
  - Message formats (with examples)
  - Expected responses
  - Error codes
  - Timeout behaviors
  - Authentication if applicable

## Architecture Documentation

### Component Diagrams
- Maintain up-to-date component diagrams in the `docs/architecture/` directory.
- Use standard formats (e.g., PlantUML, Mermaid) for diagrams to enable version control.

### Flow Charts
- Include flow charts for complex processes like the audio processing pipeline or synthesis workflow.
- Keep diagrams synchronized with code changes.

### Design Decisions
- Document significant design decisions in ADRs (Architecture Decision Records).
- Include:
  - Context/problem statement
  - Decision
  - Status
  - Consequences
  - Alternatives considered

## User Documentation

### README.md
- Keep the main README.md up-to-date with:
  - Project overview
  - Setup instructions
  - Basic usage examples
  - Link to more detailed documentation
  - Current limitations or known issues

### User Guides
- Provide comprehensive user guides for:
  - Installation and setup
  - Configuration options
  - API usage
  - Troubleshooting
  - Performance tuning

## Documentation Process

### Updates with Code Changes
- Update documentation whenever the code is changed.
- Include documentation updates in the same pull request as code changes.
- Document tech debt or known limitations for future work.

### Documentation Review
- Include documentation review as part of the code review process.
- Check for clarity, completeness, and accuracy.

### Versioning
- Version documentation to match software releases.
- Maintain a changelog for significant documentation changes.

## Definition of Done for Documentation

A task is not complete until:
1. All new public API items have docstrings
2. Module-level documentation is updated to reflect changes
3. Architecture diagrams are updated (if applicable)
4. User guides are updated (if applicable)
5. The README.md is updated (if significant changes)
6. Examples demonstrate the new functionality
7. Changelogs are updated to reflect new or changed features 
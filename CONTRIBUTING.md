# Contributing to the CSM Project

Thank you for your interest in contributing to the Conversational Speech Model (CSM) project! This document provides guidelines to help you contribute effectively.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and collaborative environment.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment following the instructions in README.md
4. Create a new branch for your changes

## Pull Request Process

1. Ensure your code follows the project's style guidelines
2. Add or update tests as necessary
3. Update documentation for any changes
4. Submit a pull request to the main repository

## Documentation Requirements

**Documentation is a critical component of the CSM project and is part of our definition of done.** All contributions must include appropriate documentation updates.

### Required Documentation Updates

For each pull request, ensure you have:

1. **Code Documentation**
   - Added docstrings to all new public functions, structs, and modules
   - Updated existing docstrings for modified code
   - Added inline comments for complex logic

2. **API Documentation**
   - Updated API documentation if the external interface has changed
   - Added examples for new API endpoints or methods

3. **Architecture Documentation**
   - Updated architecture diagrams if the system structure has changed
   - Added or updated flow charts for new or modified processes

4. **User Documentation**
   - Updated the README.md if necessary
   - Updated user guides for new features or changed behavior

5. **Task Documentation**
   - Ensure the task's documentation requirements in the "definition of done" are met
   - Update tasks.json if status or details have changed

### Documentation Style

Follow the guidelines in [docs/documentation_standards.md](docs/documentation_standards.md) for consistent documentation style.

## Development Workflow

1. **Pick a Task**: Select a task from the task list in tasks.json or the issue tracker
2. **Understand Requirements**: Make sure you understand the requirements and the definition of done
3. **Design**: For significant changes, create a design document in the docs/design directory
4. **Implement**: Write code with tests and documentation
5. **Test**: Ensure all tests pass and the feature works as expected
6. **Document**: Update documentation as described above
7. **Review**: Request code review

## Testing

- Write unit tests for all new code
- Write integration tests for complex features
- All tests must pass before a PR can be merged
- Document test coverage gaps

## Style Guidelines

- Run `rustfmt` on all code before committing
- Use `clippy` to catch common mistakes
- Follow the Rust API Guidelines for public APIs

## Questions or Help

If you need help with any part of the contribution process, please:
- Open an issue asking for help
- Reach out to the project maintainers
- Check the existing documentation for guidance

Thank you for contributing to the CSM project! 
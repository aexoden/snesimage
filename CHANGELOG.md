# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-17

### Changed

- Migrate error handling from `Box<dyn Error>` to `anyhow` for improved error
  messages
- Update Rust edition from 2018 to 2024
- License project under MIT OR Apache-2.0 (previously MIT)
- Update SDL2 dependency from 0.35 to 0.38
- Update ssimulacra2 dependency from 0.4 to 0.5
- Update image dependency from 0.24 to 0.25
- Update palette, rand, and cached to latest compatible versions
- Configure clippy linting and fix all lint issues

### Added

- GitHub Actions CI configuration
- cargo-deny configuration for dependency auditing
- Devbox development environment configuration
- Renovate configuration for automated dependency updates
- Pre-commit hooks

## [0.1.0] - 2023-04-08

Initial release. Provides a basic tool for optimizing images for SNES display
using k-means clustering for initial palette generation and iterative palette
optimization, with an SDL2-based GUI for tile assignment and optimization
control.

[0.1.1]: https://github.com/aexoden/snesimage/tree/v0.1.1
[0.1.0]: https://github.com/aexoden/snesimage/tree/v0.1.0

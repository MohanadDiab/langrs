# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-XX

### Added
- Modern pipeline architecture with dependency injection
- `LangRS` class for orchestrating the complete workflow
- `LangRSPipelineBuilder` for fluent pipeline construction
- `ModelFactory` for creating model instances
- Abstract base classes for models, tiling, outlier detection, and visualization
- Model registry system for easy extension
- Comprehensive configuration system (`LangRSConfig`)
- Image loading module with `ImageLoader` and `ImageData`
- Tiling strategies with `TilingStrategy` ABC and `SlidingWindowTiler`
- Refactored outlier detection with multiple methods
- Visualization abstraction with Matplotlib implementation
- Output management with `OutputManager`
- Geospatial export utilities
- Post-processing utilities (NMS)
- Comprehensive test suite (180+ tests)
- Integration tests
- Migration guide
- Usage examples
- Contributing guidelines

### Changed
- **BREAKING**: Renamed `LangRSPipeline` to `LangRS` (main class)
- **BREAKING**: Removed `create_pipeline()` function - use `LangRS()` directly
- **BREAKING**: Removed legacy `LangRS` implementation from `core.py`
- Removed dependency on `samgeo` (segment-geospatial)
- Direct implementation of GroundingDINO using `groundingdino-py`
- Direct implementation of SAM using `segment-anything-py`
- Model weight downloads now use `huggingface-hub` directly
- Geospatial utilities now use `rasterio` directly
- Complete refactoring following SOLID principles
- Improved type hints throughout
- Better error handling with custom exception hierarchy
- Modular architecture for easy extension
- Performance optimizations with `torch.no_grad()` for inference

### Fixed
- Fixed tuple serialization in YAML configuration
- Fixed import issues
- Fixed matplotlib backend for headless environments
- Fixed parameter handling in pipeline methods

### Removed
- Dependency on `segment-geospatial` package
- Legacy `LangRS` class implementation
- `create_pipeline()` convenience function (replaced by direct `LangRS()` instantiation)

### Security
- No known security issues

---

## [Previous Versions]

See git history for previous changelog entries.

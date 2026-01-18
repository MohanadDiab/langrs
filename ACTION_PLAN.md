# LangRS Package Refactoring Action Plan

## Executive Summary

This document outlines a comprehensive refactoring plan for the `langrs` package to improve code quality, extensibility, maintainability, and adherence to modern Python best practices. The refactoring will remove dependency on `samgeo` while maintaining access to Hugging Face model weights, implement SOLID principles, and create an extensible architecture for detection and segmentation models.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Architecture](#target-architecture)
3. [Detailed Refactoring Plan](#detailed-refactoring-plan)
4. [Implementation Phases](#implementation-phases)
5. [Code Quality Improvements](#code-quality-improvements)
6. [Migration Strategy](#migration-strategy)

---

## Current State Analysis

### Issues Identified

#### 1. **Tight Coupling to samgeo**
- **Problem**: `LangRS` inherits from `LangSAM` (samgeo), creating a hard dependency
- **Impact**: Cannot remove samgeo dependency, difficult to swap models, hard to test
- **Location**: `langrs/core.py:13`

#### 2. **No Abstraction Layers**
- **Problem**: Direct model access (`self.sam`, `self.predict_dino()`) without interfaces
- **Impact**: Cannot easily add new detection/segmentation models
- **Location**: Throughout `langrs/core.py`

#### 3. **Mixed Responsibilities**
- **Problem**: Single class handles image loading, detection, segmentation, outlier detection, visualization, geospatial conversion
- **Impact**: Violates Single Responsibility Principle, hard to maintain and test
- **Location**: `langrs/core.py` (500+ lines)

#### 4. **No Dependency Injection**
- **Problem**: Models and dependencies are instantiated internally
- **Impact**: Hard to test, cannot swap implementations, violates Dependency Inversion Principle
- **Location**: `langrs/core.py:34`

#### 5. **No Interfaces/Abstract Base Classes**
- **Problem**: No contracts for detection/segmentation models
- **Impact**: Cannot ensure consistent API, difficult to extend
- **Location**: Missing entirely

#### 6. **Hardcoded Configuration**
- **Problem**: File paths, thresholds, and parameters hardcoded in methods
- **Impact**: Inflexible, hard to configure
- **Location**: Throughout codebase

#### 7. **Poor Separation of Concerns**
- **Problem**: Visualization, I/O, processing logic all mixed together
- **Impact**: Hard to reuse components, test, and maintain
- **Location**: Multiple files

#### 8. **No Type Hints**
- **Problem**: Missing type annotations
- **Impact**: Poor IDE support, harder to understand code, no static type checking
- **Location**: All files

#### 9. **Inconsistent Error Handling**
- **Problem**: Generic `RuntimeError` with try-except blocks
- **Impact**: Hard to debug, poor error messages
- **Location**: Throughout codebase

#### 10. **No Configuration Management**
- **Problem**: No centralized configuration system
- **Impact**: Hard to manage defaults, environment-specific settings
- **Location**: Missing

---

## Target Architecture

### Proposed Structure

```
langrs/
├── __init__.py                 # Public API exports
├── core/                       # Core business logic
│   ├── __init__.py
│   ├── pipeline.py            # Main pipeline orchestrator
│   └── config.py              # Configuration management
├── models/                     # Model abstractions
│   ├── __init__.py
│   ├── base.py                # Abstract base classes
│   ├── detection/             # Detection model implementations
│   │   ├── __init__.py
│   │   ├── base.py            # DetectionModel ABC
│   │   ├── grounding_dino.py  # GroundingDINO implementation
│   │   └── registry.py        # Model registry
│   └── segmentation/          # Segmentation model implementations
│       ├── __init__.py
│       ├── base.py            # SegmentationModel ABC
│       ├── sam.py             # SAM implementation
│       └── registry.py        # Model registry
├── processing/                # Processing modules
│   ├── __init__.py
│   ├── image_loader.py        # Image loading and metadata
│   ├── tiling.py              # Sliding window/tiling logic
│   ├── outlier_detection.py   # Outlier detection (refactored
│   └── postprocessing.py      # NMS, filtering, etc.
├── visualization/             # Visualization module
│   ├── __init__.py
│   ├── base.py                # Visualization ABC
│   └── matplotlib_viz.py      # Matplotlib implementation
├── geospatial/                # Geospatial utilities
│   ├── __init__.py
│   ├── converter.py           # Coordinate conversion
│   └── exporter.py            # Shapefile export
├── io/                        # I/O operations
│   ├── __init__.py
│   ├── image_io.py            # Image I/O
│   └── output_manager.py      # Output file management
└── utils/                     # Utility functions
    ├── __init__.py
    ├── types.py               # Type definitions
    └── exceptions.py           # Custom exceptions
```

### Design Principles

1. **SOLID Principles**
   - **S**ingle Responsibility: Each class has one reason to change
   - **O**pen/Closed: Open for extension, closed for modification
   - **L**iskov Substitution: Subtypes must be substitutable
   - **I**nterface Segregation: Many specific interfaces vs one general
   - **D**ependency Inversion: Depend on abstractions, not concretions

2. **Dependency Injection**
   - Models injected via constructor
   - Configuration injected via config objects
   - Easy to mock for testing

3. **Interface-Based Design**
   - Abstract base classes for all model types
   - Protocol-based interfaces where appropriate
   - Clear contracts for all components

---

## Detailed Refactoring Plan

### Phase 1: Foundation - Interfaces and Base Classes

#### 1.1 Create Abstract Base Classes for Models

**File**: `langrs/models/base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch

class DetectionModel(ABC):
    """Abstract base class for object detection models."""
    
    @abstractmethod
    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect objects in an image based on text prompt.
        
        Args:
            image: Input image
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max)
        """
        pass
    
    @abstractmethod
    def load_weights(self, model_path: Optional[str] = None) -> None:
        """Load model weights from path or Hugging Face."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device the model is on."""
        pass


class SegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    
    @abstractmethod
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate segmentation masks for given bounding boxes.
        
        Args:
            image: Input image
            boxes: Bounding boxes tensor of shape (N, 4)
            
        Returns:
            Masks tensor of shape (N, H, W)
        """
        pass
    
    @abstractmethod
    def load_weights(self, model_path: Optional[str] = None) -> None:
        """Load model weights from path or Hugging Face."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device the model is on."""
        pass
```

**Action Items**:
- [ ] Create `langrs/models/base.py` with ABCs
- [ ] Define `DetectionModel` interface
- [ ] Define `SegmentationModel` interface
- [ ] Add type hints and docstrings
- [ ] Add unit tests for interface contracts

#### 1.2 Create Model Registry

**File**: `langrs/models/registry.py`

```python
from typing import Dict, Type, Optional
from .base import DetectionModel, SegmentationModel

class ModelRegistry:
    """Registry for model implementations."""
    
    _detection_models: Dict[str, Type[DetectionModel]] = {}
    _segmentation_models: Dict[str, Type[SegmentationModel]] = {}
    
    @classmethod
    def register_detection(cls, name: str):
        """Decorator to register a detection model."""
        def decorator(model_class: Type[DetectionModel]):
            cls._detection_models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_segmentation(cls, name: str):
        """Decorator to register a segmentation model."""
        def decorator(model_class: Type[SegmentationModel]):
            cls._segmentation_models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_detection_model(cls, name: str) -> Optional[Type[DetectionModel]]:
        """Get a registered detection model class."""
        return cls._detection_models.get(name)
    
    @classmethod
    def get_segmentation_model(cls, name: str) -> Optional[Type[SegmentationModel]]:
        """Get a registered segmentation model class."""
        return cls._segmentation_models.get(name)
    
    @classmethod
    def list_detection_models(cls) -> List[str]:
        """List all registered detection models."""
        return list(cls._detection_models.keys())
    
    @classmethod
    def list_segmentation_models(cls) -> List[str]:
        """List all registered segmentation models."""
        return list(cls._segmentation_models.keys())
```

**Action Items**:
- [ ] Create `langrs/models/registry.py`
- [ ] Implement registry pattern
- [ ] Add decorators for model registration
- [ ] Add unit tests

### Phase 2: Remove samgeo Dependency

#### 2.1 Implement GroundingDINO Directly

**File**: `langrs/models/detection/grounding_dino.py`

**Strategy**: 
- Use `groundingdino-py` directly (already in requirements)
- Download weights from Hugging Face using `huggingface_hub`
- Implement `DetectionModel` interface

**Key Changes**:
- Remove dependency on `samgeo.text_sam.LangSAM`
- Use `groundingdino-py` API directly
- Download model weights from Hugging Face (e.g., `IDEA-Research/grounding-dino-base`)

**Action Items**:
- [ ] Research `groundingdino-py` API
- [ ] Create `GroundingDINODetector` class implementing `DetectionModel`
- [ ] Implement Hugging Face weight downloading
- [ ] Add configuration for model variants
- [ ] Write unit tests
- [ ] Update documentation

#### 2.2 Implement SAM Directly

**File**: `langrs/models/segmentation/sam.py`

**Strategy**:
- Use `segment-anything-py` directly (already in requirements)
- Download weights from Hugging Face or Facebook's official URLs
- Implement `SegmentationModel` interface

**Key Changes**:
- Remove dependency on `samgeo` SAM wrapper
- Use `segment_anything` library directly
- Download SAM weights (e.g., `sam_vit_h_4b8939.pth`)

**Action Items**:
- [ ] Research `segment-anything-py` API
- [ ] Create `SAMSegmenter` class implementing `SegmentationModel`
- [ ] Implement weight downloading (Hugging Face or direct download)
- [ ] Add support for different SAM variants (vit_h, vit_l, vit_b)
- [ ] Write unit tests
- [ ] Update documentation

#### 2.3 Replace samgeo.common.get_crs

**File**: `langrs/geospatial/converter.py`

**Current**: Uses `from samgeo.common import get_crs`

**Solution**: Implement CRS extraction directly using `rasterio`

```python
import rasterio

def get_crs(image_path: str) -> Optional[str]:
    """Extract CRS from GeoTIFF using rasterio."""
    try:
        with rasterio.open(image_path) as src:
            if src.crs:
                return src.crs.to_string()
    except Exception:
        pass
    return None
```

**Action Items**:
- [ ] Remove `samgeo.common` import from `langrs/common.py`
- [ ] Implement `get_crs` in `langrs/geospatial/converter.py`
- [ ] Update all references
- [ ] Test with various GeoTIFF files

### Phase 3: Refactor Core Components

#### 3.1 Separate Image Loading

**File**: `langrs/processing/image_loader.py`

**Current**: `load_image()` in `langrs/common.py`

**Refactored**:
```python
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from PIL import Image
import rasterio

@dataclass
class ImageData:
    """Container for loaded image data."""
    image_path: Optional[str]
    pil_image: Image.Image
    np_image: np.ndarray
    source_crs: Optional[str]
    transform: Optional[rasterio.transform.Affine]
    is_georeferenced: bool
    
    @property
    def is_georeferenced(self) -> bool:
        return self.source_crs is not None


class ImageLoader:
    """Handles image loading from various sources."""
    
    def load(self, source: Union[str, np.ndarray, Image.Image]) -> ImageData:
        """Load image from file path, numpy array, or PIL Image."""
        # Implementation
        pass
```

**Action Items**:
- [ ] Create `ImageLoader` class
- [ ] Create `ImageData` dataclass
- [ ] Move `load_image()` logic to `ImageLoader.load()`
- [ ] Add type hints
- [ ] Add error handling with custom exceptions
- [ ] Write unit tests

#### 3.2 Extract Tiling Logic

**File**: `langrs/processing/tiling.py`

**Current**: `_slice_image_with_overlap()`, `_split_into_windows()` in `core.py`

**Refactored**:
```python
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import numpy as np

@dataclass
class Tile:
    """Represents a single tile/window."""
    image: Union[Image.Image, np.ndarray]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    offset_x: int = 0
    offset_y: int = 0


class TilingStrategy(ABC):
    """Abstract base class for tiling strategies."""
    
    @abstractmethod
    def create_tiles(
        self,
        image: Union[Image.Image, np.ndarray],
        window_size: int,
        overlap: int,
    ) -> List[Tile]:
        """Create tiles from image."""
        pass


class SlidingWindowTiler(TilingStrategy):
    """Sliding window tiling implementation."""
    
    def create_tiles(
        self,
        image: Union[Image.Image, np.ndarray],
        window_size: int,
        overlap: int,
    ) -> List[Tile]:
        """Create overlapping tiles using sliding window."""
        # Implementation
        pass
```

**Action Items**:
- [ ] Create `TilingStrategy` ABC
- [ ] Create `SlidingWindowTiler` implementation
- [ ] Extract tiling logic from `core.py`
- [ ] Create `Tile` dataclass
- [ ] Add type hints
- [ ] Write unit tests

#### 3.3 Refactor Outlier Detection

**File**: `langrs/processing/outlier_detection.py`

**Current**: Functions in `langrs/outlier_detection.py`

**Refactored**:
```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class OutlierDetector(ABC):
    """Abstract base class for outlier detection methods."""
    
    @abstractmethod
    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """
        Detect outliers in bounding box areas.
        
        Returns:
            Tuple of (predictions, filtered_boxes)
            predictions: -1 for outliers, 1 for inliers
        """
        pass


class ZScoreOutlierDetector(OutlierDetector):
    """Z-score based outlier detection."""
    # Implementation
    pass


class IQROutlierDetector(OutlierDetector):
    """IQR based outlier detection."""
    # Implementation
    pass

# ... other detectors
```

**Action Items**:
- [ ] Create `OutlierDetector` ABC
- [ ] Refactor each method into a class
- [ ] Remove plotting logic (move to visualization module)
- [ ] Add configuration via dataclasses
- [ ] Add type hints
- [ ] Write unit tests

#### 3.4 Extract Visualization

**File**: `langrs/visualization/matplotlib_viz.py`

**Current**: Plotting code scattered in `core.py` and `outlier_detection.py`

**Refactored**:
```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

class Visualizer(ABC):
    """Abstract base class for visualization."""
    
    @abstractmethod
    def plot_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[float, float, float, float]],
        output_path: str,
    ) -> None:
        """Plot bounding boxes on image."""
        pass
    
    @abstractmethod
    def plot_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        output_path: str,
    ) -> None:
        """Plot segmentation masks on image."""
        pass


class MatplotlibVisualizer(Visualizer):
    """Matplotlib-based visualization implementation."""
    # Implementation
    pass
```

**Action Items**:
- [ ] Create `Visualizer` ABC
- [ ] Create `MatplotlibVisualizer` implementation
- [ ] Extract all plotting code from `core.py`
- [ ] Extract plotting from `outlier_detection.py`
- [ ] Add configuration for plot styles
- [ ] Write unit tests

#### 3.5 Extract Geospatial Operations

**File**: `langrs/geospatial/`

**Current**: Functions in `langrs/common.py`

**Refactored**:
- `langrs/geospatial/converter.py`: Coordinate conversion
- `langrs/geospatial/exporter.py`: Shapefile export

**Action Items**:
- [ ] Create `geospatial/` module
- [ ] Move coordinate conversion functions
- [ ] Create `GeoDataFrameExporter` class
- [ ] Add type hints
- [ ] Write unit tests

#### 3.6 Create Output Manager

**File**: `langrs/io/output_manager.py`

**Current**: Hardcoded paths in `core.py`

**Refactored**:
```python
from pathlib import Path
from datetime import datetime
from typing import Optional

class OutputManager:
    """Manages output file paths and directories."""
    
    def __init__(
        self,
        base_output_path: str,
        create_timestamped: bool = True,
    ):
        self.base_path = Path(base_output_path)
        if create_timestamped:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.base_path / timestamp
        else:
            self.output_path = self.base_path
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, filename: str) -> Path:
        """Get full path for output file."""
        return self.output_path / filename
```

**Action Items**:
- [ ] Create `OutputManager` class
- [ ] Replace hardcoded paths in `core.py`
- [ ] Add configuration options
- [ ] Write unit tests

### Phase 4: Create Main Pipeline

#### 4.1 Pipeline Orchestrator

**File**: `langrs/core/pipeline.py`

**Refactored Main Class**:
```python
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image

from ..models.base import DetectionModel, SegmentationModel
from ..processing.image_loader import ImageLoader, ImageData
from ..processing.tiling import TilingStrategy
from ..processing.outlier_detection import OutlierDetector
from ..visualization.base import Visualizer
from ..geospatial.exporter import GeoDataFrameExporter
from ..io.output_manager import OutputManager
from ..core.config import LangRSConfig

class LangRSPipeline:
    """
    Main pipeline for remote sensing image segmentation.
    
    Uses dependency injection for all components.
    """
    
    def __init__(
        self,
        detection_model: DetectionModel,
        segmentation_model: SegmentationModel,
        image_loader: ImageLoader,
        tiling_strategy: TilingStrategy,
        outlier_detectors: Dict[str, OutlierDetector],
        visualizer: Visualizer,
        output_manager: OutputManager,
        config: LangRSConfig,
    ):
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self.image_loader = image_loader
        self.tiling_strategy = tiling_strategy
        self.outlier_detectors = outlier_detectors
        self.visualizer = visualizer
        self.output_manager = output_manager
        self.config = config
        
        self.image_data: Optional[ImageData] = None
        self.bounding_boxes: List[Tuple[float, float, float, float]] = []
        self.filtered_boxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self.masks: Optional[np.ndarray] = None
    
    def load_image(self, image_source) -> None:
        """Load and prepare image."""
        self.image_data = self.image_loader.load(image_source)
    
    def detect_objects(
        self,
        text_prompt: str,
        window_size: int = 500,
        overlap: int = 200,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> List[Tuple[float, float, float, float]]:
        """Detect objects using sliding window approach."""
        # Implementation using tiling and detection model
        pass
    
    def filter_outliers(
        self,
        method: Optional[str] = None,
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """Apply outlier detection methods."""
        # Implementation
        pass
    
    def segment(
        self,
        boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> np.ndarray:
        """Generate segmentation masks."""
        # Implementation
        pass
    
    def run_full_pipeline(
        self,
        image_source,
        text_prompt: str,
        **kwargs
    ) -> np.ndarray:
        """Run complete pipeline: load -> detect -> filter -> segment."""
        # Implementation
        pass
```

**Action Items**:
- [ ] Create `LangRSPipeline` class
- [ ] Implement dependency injection
- [ ] Implement pipeline methods
- [ ] Add comprehensive type hints
- [ ] Add error handling
- [ ] Write integration tests

#### 4.2 Configuration Management

**File**: `langrs/core/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    box_threshold: float = 0.3
    text_threshold: float = 0.3
    window_size: int = 500
    overlap: int = 200


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""
    multimask_output: bool = False
    window_size: int = 500
    overlap: int = 200


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection."""
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    svm_nu: float = 0.1
    isolation_forest_contamination: float = 0.25
    lof_contamination: float = 0.25
    lof_n_neighbors: int = 20
    robust_cov_contamination: float = 0.25


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    figsize: tuple = (10, 10)
    box_color: str = 'r'
    box_linewidth: int = 1
    mask_alpha: float = 0.4
    mask_colormap: str = 'viridis'


@dataclass
class LangRSConfig:
    """Main configuration for LangRS pipeline."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    outlier_detection: OutlierDetectionConfig = field(default_factory=OutlierDetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LangRSConfig':
        """Create config from dictionary."""
        # Implementation
        pass
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LangRSConfig':
        """Load config from YAML file."""
        # Implementation
        pass
```

**Action Items**:
- [ ] Create configuration dataclasses
- [ ] Add YAML/JSON loading support
- [ ] Add validation
- [ ] Write unit tests

### Phase 5: Factory and Builder Patterns

#### 5.1 Model Factory

**File**: `langrs/models/factory.py`

```python
from typing import Optional
from .base import DetectionModel, SegmentationModel
from .registry import ModelRegistry
from .detection.grounding_dino import GroundingDINODetector
from .segmentation.sam import SAMSegmenter

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_detection_model(
        model_name: str = "grounding_dino",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> DetectionModel:
        """Create a detection model instance."""
        model_class = ModelRegistry.get_detection_model(model_name)
        if model_class is None:
            raise ValueError(f"Unknown detection model: {model_name}")
        return model_class(model_path=model_path, device=device, **kwargs)
    
    @staticmethod
    def create_segmentation_model(
        model_name: str = "sam",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> SegmentationModel:
        """Create a segmentation model instance."""
        model_class = ModelRegistry.get_segmentation_model(model_name)
        if model_class is None:
            raise ValueError(f"Unknown segmentation model: {model_name}")
        return model_class(model_path=model_path, device=device, **kwargs)
```

**Action Items**:
- [ ] Create `ModelFactory` class
- [ ] Implement factory methods
- [ ] Add error handling
- [ ] Write unit tests

#### 5.2 Pipeline Builder

**File**: `langrs/core/builder.py`

```python
from typing import Optional, Dict
from .pipeline import LangRSPipeline
from .config import LangRSConfig
from ..models.factory import ModelFactory
# ... other imports

class LangRSPipelineBuilder:
    """Builder for creating LangRS pipeline instances."""
    
    def __init__(self):
        self.config: Optional[LangRSConfig] = None
        self.detection_model_name: str = "grounding_dino"
        self.segmentation_model_name: str = "sam"
        # ... other defaults
    
    def with_config(self, config: LangRSConfig) -> 'LangRSPipelineBuilder':
        """Set configuration."""
        self.config = config
        return self
    
    def with_detection_model(self, model_name: str) -> 'LangRSPipelineBuilder':
        """Set detection model."""
        self.detection_model_name = model_name
        return self
    
    def with_segmentation_model(self, model_name: str) -> 'LangRSPipelineBuilder':
        """Set segmentation model."""
        self.segmentation_model_name = model_name
        return self
    
    def build(self) -> LangRSPipeline:
        """Build the pipeline."""
        if self.config is None:
            self.config = LangRSConfig()
        
        # Create models
        detection_model = ModelFactory.create_detection_model(
            self.detection_model_name
        )
        segmentation_model = ModelFactory.create_segmentation_model(
            self.segmentation_model_name
        )
        
        # Create other components
        # ...
        
        return LangRSPipeline(
            detection_model=detection_model,
            segmentation_model=segmentation_model,
            # ... other components
            config=self.config,
        )
```

**Action Items**:
- [ ] Create `LangRSPipelineBuilder` class
- [ ] Implement fluent interface
- [ ] Add default component creation
- [ ] Write unit tests

### Phase 6: Public API Simplification

#### 6.1 Simplified Public API

**File**: `langrs/__init__.py`

```python
"""
LangRS - Remote Sensing Image Segmentation Package
"""

from .core.pipeline import LangRSPipeline
from .core.builder import LangRSPipelineBuilder
from .core.config import LangRSConfig
from .models.registry import ModelRegistry
from .processing.postprocessing import apply_nms, apply_nms_areas

# Convenience function for quick usage
def create_pipeline(
    detection_model: str = "grounding_dino",
    segmentation_model: str = "sam",
    output_path: str = "output",
    config: Optional[LangRSConfig] = None,
) -> LangRSPipeline:
    """
    Create a LangRS pipeline with default settings.
    
    Args:
        detection_model: Name of detection model to use
        segmentation_model: Name of segmentation model to use
        output_path: Base output directory
        config: Optional configuration object
        
    Returns:
        Configured LangRSPipeline instance
    """
    builder = LangRSPipelineBuilder()
    if config:
        builder.with_config(config)
    builder.with_detection_model(detection_model)
    builder.with_segmentation_model(segmentation_model)
    return builder.build()

__all__ = [
    "LangRSPipeline",
    "LangRSPipelineBuilder",
    "LangRSConfig",
    "ModelRegistry",
    "create_pipeline",
    "apply_nms",
    "apply_nms_areas",
]
```

**Action Items**:
- [ ] Update `__init__.py` with new API
- [ ] Create convenience function
- [ ] Update documentation
- [ ] Maintain backward compatibility layer (if needed)

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- Create base classes and interfaces
- Set up model registry
- Create configuration system
- Set up testing framework

### Phase 2: Remove samgeo (Week 3-4)
- Implement GroundingDINO directly
- Implement SAM directly
- Replace samgeo.common utilities
- Test model loading and inference

### Phase 3: Refactor Components (Week 5-7)
- Extract image loading
- Extract tiling logic
- Refactor outlier detection
- Extract visualization
- Extract geospatial operations
- Create output manager

### Phase 4: Pipeline & Integration (Week 8-9)
- Create main pipeline class
- Implement dependency injection
- Create factory and builder
- Integration testing

### Phase 5: Testing & Documentation (Week 10-11)
- Comprehensive unit tests
- Integration tests
- Update documentation
- Migration guide
- Examples

### Phase 6: Polish & Release (Week 12)
- Code review
- Performance optimization
- Final testing
- Release preparation

---

## Code Quality Improvements

### 1. Type Hints
- **Current**: No type hints
- **Target**: Complete type coverage
- **Tools**: `mypy` for static type checking

### 2. Documentation
- **Current**: Basic docstrings
- **Target**: Comprehensive docstrings with examples
- **Format**: Google-style or NumPy-style

### 3. Testing
- **Current**: No tests visible
- **Target**: 
  - Unit tests for all components (>80% coverage)
  - Integration tests for pipeline
  - Property-based tests where appropriate
- **Framework**: `pytest`

### 4. Error Handling
- **Current**: Generic RuntimeError
- **Target**: Custom exception hierarchy
- **File**: `langrs/utils/exceptions.py`

```python
class LangRSError(Exception):
    """Base exception for LangRS."""
    pass

class ModelLoadError(LangRSError):
    """Error loading model weights."""
    pass

class ImageLoadError(LangRSError):
    """Error loading image."""
    pass

class DetectionError(LangRSError):
    """Error during object detection."""
    pass

class SegmentationError(LangRSError):
    """Error during segmentation."""
    pass
```

### 5. Logging
- **Current**: No logging
- **Target**: Structured logging with appropriate levels
- **Library**: `logging` module

### 6. Code Formatting
- **Tool**: `black` for formatting
- **Tool**: `isort` for import sorting
- **Tool**: `flake8` or `ruff` for linting

### 7. Pre-commit Hooks
- **File**: `.pre-commit-config.yaml`
- **Hooks**: black, isort, mypy, flake8, pytest

---

## Migration Strategy

### Backward Compatibility

Create a compatibility layer for existing users:

**File**: `langrs/compat.py`**

```python
"""
Backward compatibility layer for existing LangRS API.
"""

from typing import Union
from .core.pipeline import LangRSPipeline
from .core.builder import LangRSPipelineBuilder

class LangRS(LangRSPipeline):
    """
    Legacy LangRS class for backward compatibility.
    
    This class maintains the old API while using the new architecture internally.
    """
    
    def __init__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        prompt: str,
        output_path: str,
    ):
        # Create pipeline using builder
        builder = LangRSPipelineBuilder()
        # ... configure builder
        pipeline = builder.build()
        
        # Initialize parent
        super().__init__(**pipeline.__dict__)
        
        # Load image
        self.load_image(image)
        self.prompt = prompt
        
        # Set output path
        # ...
    
    # Map old methods to new pipeline methods
    def generate_boxes(self, **kwargs):
        return self.detect_objects(text_prompt=self.prompt, **kwargs)
    
    # ... other method mappings
```

**Action Items**:
- [ ] Create compatibility layer
- [ ] Map old API to new API
- [ ] Add deprecation warnings
- [ ] Update migration guide

### Migration Guide

Create `MIGRATION_GUIDE.md` with:
- Step-by-step migration instructions
- API changes
- Configuration changes
- Examples of old vs new code

---

## Additional Improvements

### 1. Add Support for More Models

**Detection Models**:
- YOLO-World
- GLIP
- OWL-ViT

**Segmentation Models**:
- SAM2
- FastSAM
- MobileSAM

**Action Items**:
- [ ] Create implementations for each model
- [ ] Register in model registry
- [ ] Add tests
- [ ] Update documentation

### 2. Performance Optimizations

- **Quantization**: Add model quantization support
- **Batch Processing**: Support batch inference
- **GPU Memory Management**: Better memory handling
- **Caching**: Cache model weights and intermediate results

### 3. Configuration Files

Support YAML/JSON configuration files:

```yaml
detection:
  model: "grounding_dino"
  box_threshold: 0.3
  text_threshold: 0.3
  window_size: 500
  overlap: 200

segmentation:
  model: "sam"
  variant: "vit_h"
  multimask_output: false

outlier_detection:
  methods: ["zscore", "iqr", "lof"]
  zscore_threshold: 3.0

output:
  base_path: "output"
  create_timestamped: true
  save_geospatial: true
```

### 4. CLI Interface

Add command-line interface:

```bash
langrs process \
  --image path/to/image.tif \
  --prompt "roof" \
  --output output/ \
  --config config.yaml
```

### 5. Progress Tracking

Add progress bars for long-running operations:

```python
from tqdm import tqdm

# In pipeline methods
for tile in tqdm(tiles, desc="Processing tiles"):
    # Process tile
    pass
```

---

## Testing Strategy

### Unit Tests

- **Location**: `tests/unit/`
- **Coverage Target**: >80%
- **Focus**: Individual components in isolation

### Integration Tests

- **Location**: `tests/integration/`
- **Focus**: End-to-end pipeline execution
- **Data**: Use small test images

### Property-Based Tests

- **Library**: `hypothesis`
- **Focus**: Tiling, coordinate conversion, NMS

### Performance Tests

- **Location**: `tests/performance/`
- **Focus**: Memory usage, inference speed
- **Benchmarks**: Compare with old implementation

---

## Documentation Updates

### 1. API Documentation

- Generate with `sphinx` or `mkdocs`
- Include all public APIs
- Add usage examples

### 2. Architecture Documentation

- Document design decisions
- Include architecture diagrams
- Explain extension points

### 3. Developer Guide

- How to add new models
- How to add new outlier detectors
- Contribution guidelines

### 4. User Guide

- Installation instructions
- Quick start guide
- Advanced usage examples
- Troubleshooting

---

## Checklist Summary

### Foundation
- [ ] Create ABCs for models
- [ ] Create model registry
- [ ] Create configuration system
- [ ] Set up testing framework

### Remove samgeo
- [ ] Implement GroundingDINO directly
- [ ] Implement SAM directly
- [ ] Replace samgeo utilities
- [ ] Test model loading

### Refactoring
- [ ] Extract image loading
- [ ] Extract tiling logic
- [ ] Refactor outlier detection
- [ ] Extract visualization
- [ ] Extract geospatial operations
- [ ] Create output manager

### Pipeline
- [ ] Create main pipeline class
- [ ] Implement dependency injection
- [ ] Create factory
- [ ] Create builder
- [ ] Integration tests

### Quality
- [ ] Add type hints
- [ ] Add comprehensive tests
- [ ] Add logging
- [ ] Add error handling
- [ ] Code formatting
- [ ] Pre-commit hooks

### Documentation
- [ ] API documentation
- [ ] Architecture docs
- [ ] Migration guide
- [ ] User guide
- [ ] Developer guide

### Release
- [ ] Backward compatibility layer
- [ ] Update setup.py
- [ ] Update requirements.txt
- [ ] Version bump
- [ ] Release notes

---

## Conclusion

This refactoring plan transforms `langrs` from a tightly-coupled, hard-to-extend package into a modern, well-architected library following SOLID principles and Python best practices. The new architecture will:

1. **Remove samgeo dependency** while maintaining access to Hugging Face models
2. **Enable easy extension** of detection and segmentation models
3. **Improve testability** through dependency injection and interfaces
4. **Enhance maintainability** through clear separation of concerns
5. **Follow modern Python practices** with type hints, ABCs, and proper structure

The phased approach allows for incremental implementation while maintaining functionality throughout the refactoring process.

# LangRS 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohanadDiab/langrs/blob/main/examples/langrs.ipynb)
[![PyPI version](https://badge.fury.io/py/langrs.svg)](https://pypi.python.org/pypi/langrs)

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadDiab/langrs/main/assets/langrs_logo.png" alt="LangRS Logo" width="300"/>
</p>

**A modern, extensible Python package for zero-shot segmentation of aerial images using GroundingDINO and Segment Anything Model (SAM)**

## Introduction

LangRS is a Python package for remote sensing image segmentation that combines advanced techniques like bounding box detection, semantic segmentation, and outlier rejection to deliver precise and reliable segmentation of geospatial images. Built with modern Python best practices, SOLID principles, and a modular architecture for easy extension.

## How it works 

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadDiab/langrs/main/assets/pres.gif" alt="Performance Comparison" width="600"/>
</p>

## 📊 Package Performance vs Ground Truth

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadDiab/langrs/main/assets/10.png" alt="Performance Comparison" width="600"/>
</p>

## 🔄 Direct Comparison with SAMGEO Package

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadDiab/langrs/main/assets/11.png" alt="Comparison with Older Package" width="600"/>
</p>

## Features

- **Bounding Box Detection:** Locate objects in remote sensing images with a sliding window approach.
- **Outlier Detection:** Apply various statistical and machine learning methods to filter out anomalies in the detected objects based on the area of the detected bounding boxes.
- **Non-Max Suppression:** Applies NMS to the input bounding boxes, can reduce accuracy slightly, but greatly increases inference speed and lowers memory usage.
- **Area Calculation:** Compute and rank bounding boxes by their areas.
- **Image Segmentation:** Detect and extract objects based on text prompts using GroundingDINO, Rex-Omni, or SAM.
- **Modern Architecture:** Built with SOLID principles, dependency injection, and abstract base classes for easy extension.
- **Geospatial Support:** Automatic CRS extraction and shapefile export for bounding boxes and masks.

## Installation

### Install LangRS with pip

```bash
pip install langrs
```

## Usage

### Quick Start

Here is the simplest way to use LangRS:

```python
from langrs import LangRS

# Create LangRS with default settings
langrs = LangRS(output_path="output")

# Run the complete pipeline
masks = langrs.run_full_pipeline(
    image_source="path_to_your_tif_file",
    text_prompt="roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25,
)
```

### Step-by-Step Usage

For more control over the pipeline:

```python
from langrs import LangRS

# Create LangRS
langrs = LangRS(output_path="output")

# Load image
langrs.load_image("path_to_your_tif_file")

# Detect objects
boxes = langrs.detect_objects(
    text_prompt="roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25,
)

# Apply outlier rejection
# This will return a dict with the following keys:
# ['zscore', 'iqr', 'svm', 'svm_sgd', 'robust_covariance', 'lof', 'isolation_forest']
# The value of each key represents the bounding boxes from the previous step with the 
# outlier rejection method of the key's name applied to them
bboxes_filtered = langrs.filter_outliers()

# Retrieve certain bounding boxes 
bboxes_zscore = bboxes_filtered['zscore']

# Generate segmentation masks for the filtered bounding boxes
masks = langrs.segment(boxes=bboxes_zscore)
```

### Advanced Usage with Custom Configuration

```python
from langrs import LangRS, LangRSConfig

# Create custom configuration
config = LangRSConfig()
config.detection.box_threshold = 0.25
config.detection.text_threshold = 0.25
config.detection.window_size = 600
config.detection.overlap = 300
config.outlier_detection.zscore_threshold = 2.5

# Create LangRS with custom settings
langrs = LangRS(
    output_path="output",
    device="cpu",  # or "cuda" for GPU
    config=config,
)

# Use LangRS
masks = langrs.run_full_pipeline("path_to_your_tif_file", "roof")
```

### Using Different Detection Models

LangRS supports multiple detection models. You can use **GroundingDINO** (default) or **Rex-Omni**:

```python
from langrs import LangRS, ModelFactory

# Option 1: Use Rex-Omni via LangRS
langrs = LangRS(
    output_path="output",
    detection_model="rex_omni",  # Use Rex-Omni instead of GroundingDINO
    device="cpu",
)

# Option 2: Create custom Rex-Omni detector
from langrs import RexOmniDetector

detector = RexOmniDetector(
    model_variant="default",  # or "awq" for quantized version
    backend="transformers",    # or "vllm" for faster inference
    device="cpu",
)
detector.load_weights()  # Downloads from Hugging Face

# Use with LangRS
langrs = LangRS(
    output_path="output",
    _detection_model_instance=detector,
)
```

**Note on Rex-Omni:**
- Rex-Omni is a 3B-parameter MLLM that performs detection via next-token prediction
- Requires additional dependencies: `transformers`, `qwen-vl-utils`, `accelerate`, `huggingface_hub`
- Does not use `box_threshold` and `text_threshold` (no confidence scores)
- Detection quality is controlled by generation parameters (temperature, top_p, top_k)
- Supports both full model and AWQ quantized versions

### Input Parameters

#### `LangRS()` Initialization:
- `output_path`: Directory to save output files
- `detection_model`: Name of detection model ("grounding_dino" or "rex_omni", default: "grounding_dino")
- `segmentation_model`: Name of segmentation model (default: "sam")
- `device`: Device to use ('cpu' or 'cuda', default: auto-detect)
- `config`: Optional LangRSConfig object

#### `detect_objects()`:
- `text_prompt`: Text description of objects to detect
- `window_size` (int): Size of each chunk for processing. Default is `500`.
- `overlap` (int): Overlap size between chunks. Default is `200`.
- `box_threshold` (float): Confidence threshold for box detection. Default is `0.5`.
- `text_threshold` (float): Confidence threshold for text detection. Default is `0.5`.

#### `filter_outliers()`:
- `method` (optional): Specific method to apply. If None, applies all methods.
- Returns a dictionary with keys: `['zscore', 'iqr', 'svm', 'svm_sgd', 'robust_covariance', 'lof', 'isolation_forest']`

#### `segment()`:
- `boxes` (optional): List of bounding boxes. If None, uses detected boxes.
- `window_size` (int): Window size for tiling. Default from config.
- `overlap` (int): Overlap between windows. Default from config.

## Output

When the code runs, it generates the following outputs:
1. **Original Image with Bounding Boxes:** Shows the detected bounding boxes.
2. **Filtered Bounding Boxes:** Bounding boxes after applying outlier rejection.
3. **Segmentation Masks:** Overlays segmentation masks on the original image.
4. **Area Plot:** A scatter plot of bounding box areas to visualize distributions.
5. **Geospatial Files:** Shapefiles for bounding boxes and masks (if GeoTIFF input).

The results are saved in the specified `output` directory, organized with a timestamp to separate runs.

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Simple usage
- `advanced_usage.py` - Advanced features
- `step_by_step.py` - Step-by-step execution
- `custom_models.py` - Custom model selection
- `configuration_example.py` - Configuration management

## Citation

```bibtex
@article{DIAB2025100105,
title = {Optimizing zero-shot text-based segmentation of remote sensing imagery using SAM and Grounding DINO},
journal = {Artificial Intelligence in Geosciences},
volume = {6},
number = {1},
pages = {100105},
year = {2025},
issn = {2666-5441},
doi = {https://doi.org/10.1016/j.aiig.2025.100105},
url = {https://www.sciencedirect.com/science/article/pii/S2666544125000012},
author = {Mohanad Diab and Polychronis Kolokoussis and Maria Antonia Brovelli},
keywords = {Foundation models, Multi-modal models, Vision language models, Semantic segmentation, Segment anything model, Earth observation, Remote sensing},
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Development setup
- Code style
- Testing
- Pull request process

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For any questions or issues, please open an issue on GitHub or contact the project maintainers.

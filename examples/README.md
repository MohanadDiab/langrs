# Examples

This directory contains example scripts demonstrating how to use LangRS.

## Before Running Examples

**Important:** You must install the LangRS package before running examples:

```bash
# From the repository root directory
pip install -e .
```

If you get `ModuleNotFoundError: No module named 'langrs'`, it means the package is not installed.

## Running Examples

Run examples from the repository root directory:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
python examples/step_by_step.py
python examples/custom_models.py
python examples/configuration_example.py
```

## Example Files

- **basic_usage.py** - Simplest usage with default settings
- **advanced_usage.py** - Advanced features and step-by-step execution
- **step_by_step.py** - Manual pipeline control with NMS
- **custom_models.py** - Using different models and custom configurations
- **configuration_example.py** - Configuration management
- **grounding_dino_example.py** - Direct GroundingDINO usage
- **rex_omni_example.py** - Direct Rex-Omni usage
- **langrs.ipynb** - Jupyter notebook with interactive examples

## Data Requirements

Most examples expect image files in the `data/` directory:
- `data/image.tif` - Main test image (GeoTIFF)
- Other test images as needed

Make sure to update the image paths in the examples to match your data location.

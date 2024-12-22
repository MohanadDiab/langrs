# LangRS: Language-driven Remote Sensing Image Analysis

LangRS is a Python package for language-driven remote sensing image analysis. It combines natural language processing with computer vision techniques to analyze and segment remote sensing imagery based on textual descriptions.

## Features

- Image segmentation using the Segment Anything Model (SAM)
- Object detection using DINO
- Outlier detection for refining bounding boxes
- Evaluation metrics for segmentation and object detection
- Visualization tools for analysis results

## Installation

You can install LangRS using pip:

```
pip install langrs
```

Or you can clone the repository and install it locally:

```
git clone https://github.com/yourusername/langrs.git
cd langrs
pip install -e .
```

## Usage

Here's a basic example of how to use LangRS:

```python
from langrs import LangRS

# Initialize LangRS with a configuration file
lang_rs = LangRS('config.json')

# Process the image
lang_rs.process()
```

For more detailed examples, please refer to the `examples` directory.

## Configuration

LangRS uses a JSON configuration file. Here's an example:

```json
{
    "image_input": "path/to/your/image.tif",
    "text_input": "white cars",
    "tile_size": 1000,
    "overlap": 300,
    "tiling": false,
    "evaluation": true,
    "outlier_methods": ["isolation_forest"],
    "output_dir": "output",
    "ground_truth_bb": "path/to/ground_truth_bboxes.json",
    "ground_truth_mask": "path/to/ground_truth_mask.json"
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
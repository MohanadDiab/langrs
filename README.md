# LangRS Segmentation

LangRS Segmentation is a Python package for remote sensing image segmentation. It combines advanced techniques like bounding box detection, semantic segmentation, and outlier rejection to deliver precise and reliable analysis of geospatial images.

## Features

- **Image Segmentation:** Detect and extract objects based on text prompts using LangSAM.
- **Bounding Box Detection:** Locate objects in remote sensing images with a sliding window approach.
- **Outlier Detection:** Apply various statistical and machine learning methods to filter out anomalies in the detected objects.
- **Area Calculation:** Compute and rank bounding boxes by their areas.
- **Modular Design:** Easily extend functionality for custom workflows.

---

## Installation

### Prerequisites

Ensure you have Python 3.10 or higher installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Install LangRS Segmentation

Clone the repository and install the package locally:

```bash
git clone https://github.com/MohanadDiab/langrs.git
cd langrs
pip install .
```

---

## Usage

Here is an example of how to use the `LangRS` class for remote sensing image segmentation:

```python
from langrs.core import LangRS

def main():
    # Specify a text prompt to identify objects in the image
    text_input = "roof"

    # Path to the input remote sensing image
    image_input = "data/roi_kala.tif"

    # Initialize LangRS with the input image, text prompt, and output directory
    langrs = LangRS(image_input, text_input, "output")

    # Detect bounding boxes using the sliding window approach
    langrs.predict_dino(window_size=600, overlap=300, box_threshold=0.25, text_threshold=0.25)

    # Apply outlier rejection to filter anomalous bounding boxes
    langrs.outlier_rejection()

    # Generate segmentation masks for the filtered bounding boxes
    langrs.predict_sam(rejection_method="zscore")

if __name__ == "__main__":
    main()
```

### Input Parameters for LangRS Methods

#### `LangRS` Initialization:
- `image`: Path to the input image.
- `prompt`: Text prompt for object detection.
- `output_path`: Directory to save output files.

#### `predict_dino`:
- `window_size` (int): Size of each chunk for processing. Default is `500`.
- `overlap` (int): Overlap size between chunks. Default is `200`.
- `box_threshold` (float): Confidence threshold for box detection. Default is `0.5`.
- `text_threshold` (float): Confidence threshold for text detection. Default is `0.5`.

#### `outlier_rejection`:
Applies multiple outlier detection methods (e.g., Z-Score, IQR, SVM, LOF) to filter bounding boxes.

#### `predict_sam`:
- `rejection_method` (str): The method used for filtering outliers. Options include `zscore`, `iqr`, `svm`, `lof`, etc.

---

## Output

When the code runs, it generates the following outputs:
1. **Original Image with Bounding Boxes:** Shows the detected bounding boxes.
2. **Filtered Bounding Boxes:** Bounding boxes after applying outlier rejection.
3. **Segmentation Masks:** Overlays segmentation masks on the original image.
4. **Area Plot:** A scatter plot of bounding box areas to visualize distributions.

The results are saved in the specified `output` directory, organized with a timestamp to separate runs.

---

## Contributing

We welcome contributions! If you'd like to add features or fix bugs:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support

For any questions or issues, please open an issue on GitHub or contact the project maintainers.


[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/segment-geospatial/blob/main/docs/examples/satellite.ipynb)

[![image](https://img.shields.io/pypi/v/segment-geospatial.svg)](https://colab.research.google.com/github/MohanadDiab/langrs/examples/langrs.ipynb)


<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadDiab/langrs/main/assets/langrs_logo.png" alt="LangRS Logo" width="300"/>
</p>


# LangRS 

LangRS  is a Python package for remote sensing image segmentation, it is built on top of the [Segment-Geospatial](https://github.com/opengeos/segment-geospatial) package. It combines advanced techniques like bounding box detection, semantic segmentation, and outlier rejection to deliver precise and reliable segmentation of geospatial images.

## Citations
TBD

## Features

- **Bounding Box Detection:** Locate objects in remote sensing images with a sliding window approach.
- **Outlier Detection:** Apply various statistical and machine learning methods to filter out anomalies in the detected objects based on the area of the detected bounding boxes.
- **Area Calculation:** Compute and rank bounding boxes by their areas.
- **Image Segmentation:** Detect and extract objects based on text prompts using LangSAM.
---

## Installation


### Install LangRS with pip


```bash
pip install langrs
```

### Direct install


Ensure you have Python 3.10 or higher installed. Install the required dependencies.

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
    text_input = "object"

    # Path to the input remote sensing image
    image_input = "path_to_your_tif_file"

    # Initialize LangRS with the input image, text prompt, and output directory
    langrs = LangRS(image_input, text_input, "output_folder")

    # Detect bounding boxes using the sliding window approach with example parameters
    bounding_boxes = langrs.predict_dino(window_size=600, overlap=300, box_threshold=0.25, text_threshold=0.25)

    # Apply outlier rejection to filter anomalous bounding boxes
    # This will return a dict with the follwing keys:
    # ['zscore', 'iqr', 'svm', 'svm_sgd', 'robust_covariance', 'lof', 'isolation_forest']
    # The value of each key represent the boudning boxes from the previous step with the 
    # outlier rejection method of the key's name applied to them
    bboxes_filtered = langrs.outlier_rejection()

    # Generate segmentation masks for the filtered bounding boxes of the provided key
    masks = langrs.predict_sam(rejection_method="zscore")
    # Or
    masks = langrs.predict_sam(rejection_method="iqr")
    

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


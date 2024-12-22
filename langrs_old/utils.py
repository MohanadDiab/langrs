import os
import json
import numpy as np
from PIL import Image
import rasterio
from typing import List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Load an image as a numpy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image as a numpy array.

    Raises:
        FileNotFoundError: If the image file is not found.
        ValueError: If the image cannot be opened or processed.
    """
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                return src.read().transpose(1, 2, 0)
        else:
            return np.array(Image.open(image_path))
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Unable to open image file: {image_path}")

def array_to_pil_image(array: np.ndarray) -> Image.Image:
    """
    Convert a numpy array to a PIL Image.

    Args:
        array (np.ndarray): The input image as a numpy array.

    Returns:
        Image.Image: The image as a PIL Image object.

    Raises:
        ValueError: If the input array cannot be converted to an image.
    """
    try:
        if array.ndim == 3:
            return Image.fromarray(array)
        elif array.ndim == 2:
            return Image.fromarray(array, mode='L')
        else:
            raise ValueError("Input array must have 2 or 3 dimensions")
    except Exception as e:
        logging.error(f"Error converting array to PIL Image: {str(e)}")
        raise ValueError("Unable to convert array to PIL Image")

def sort_bboxes_by_area(bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """
    Sort bounding boxes by area in descending order.

    Args:
        bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes in format (x1, y1, x2, y2).

    Returns:
        List[Tuple[float, float, float, float]]: Sorted list of bounding boxes.
    """
    return sorted(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)

def save_json(data: Union[List, dict], file_path: str):
    """
    Save data to a JSON file.

    Args:
        data (Union[List, dict]): Data to be saved.
        file_path (str): Path to save the JSON file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data saved to {file_path}")
    except IOError as e:
        logging.error(f"Error saving JSON file: {str(e)}")
        raise

def load_json(file_path: str) -> Union[List, dict]:
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Union[List, dict]: Loaded data.

    Raises:
        FileNotFoundError: If the JSON file is not found.
        json.JSONDecodeError: If there's an error decoding the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {str(e)}")
        raise

def ensure_dir(directory: str):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File extension (lowercase, without the dot).
    """
    return os.path.splitext(file_path)[1].lower()[1:]
import os
import logging
from typing import Tuple, List, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torch


from .lang_sam_processor import LangSAMProcessor
from .utils import *

class ImageProcessor:
    """
    A class for processing images, including tiling and object detection/segmentation.
    """

    def __init__(self, config):
        """
        Initialize the ImageProcessor with configuration settings and LangSAM processor.

        Args:
            config: Configuration object containing processing parameters.
        """
        self.config = config
        self.lang_sam_processor = LangSAMProcessor()

    def process_image(self, image_path: str) -> Tuple[Image.Image, List[Tuple[float, float, float, float]], np.ndarray]:
        """
        Process an image, including tiling if necessary, and run object detection and segmentation.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[Image.Image, List[Tuple[float, float, float, float]], np.ndarray]: 
                Processed image, list of detected bounding boxes, and segmentation mask overlay.

        Raises:
            FileNotFoundError: If the input image file is not found.
            ValueError: If there's an error processing the image.
        """
        try:
            if self.config.get("tiling"):
                # Implement tiling logic here
                pass
            
            image_array = load_image_as_array(image_path)
            pil_image = array_to_pil_image(image_array)
            
            # Run detection and segmentation on chunks
            bounding_boxes, masks = self.run_detection_on_chunks(pil_image)
            
            # Create mask overlay
            mask_overlay = self.lang_sam_processor.create_mask_overlay(masks, np.array(pil_image).shape)
            
            return pil_image, bounding_boxes, mask_overlay
        except FileNotFoundError:
            logging.error(f"Input image file not found: {image_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Unable to process image: {str(e)}")

    def run_detection_on_chunks(self, image: Image.Image) -> Tuple[List[Tuple[float, float, float, float]], torch.Tensor]:
        """
        Run object detection and segmentation on image chunks and combine the results.

        Args:
            image (Image.Image): The input image.

        Returns:
            Tuple[List[Tuple[float, float, float, float]], torch.Tensor]: 
                List of detected bounding boxes and segmentation masks.

        Raises:
            ValueError: If there's an error during the detection process.
        """
        try:
            chunks = slice_image_with_overlap(image, 
                                              chunk_size=self.config.get("tile_size", 1000), 
                                              overlap=self.config.get("overlap", 300))
            all_bounding_boxes = []
            all_masks = []
            
            for chunk, offset_x, offset_y in chunks:
                chunk_boxes, chunk_masks = self.lang_sam_processor.process_image(
                    chunk,
                    text_prompt=self.config.get("text_input", ""),
                    box_threshold=self.config.get("box_threshold", 0.3),
                    text_threshold=self.config.get("text_threshold", 0.75)
                )
                localized_boxes = localize_bounding_boxes(chunk_boxes, offset_x, offset_y)
                all_bounding_boxes.extend(localized_boxes)
                all_masks.append(chunk_masks)
            
            # Combine masks from all chunks (this is a simplification, you might need a more sophisticated method)
            combined_masks = torch.cat(all_masks, dim=0)
            
            return all_bounding_boxes, combined_masks
        except Exception as e:
            logging.error(f"Error running detection on chunks: {str(e)}")
            raise ValueError(f"Detection process failed: {str(e)}")

# ... (rest of the file remains the same)
import logging
from typing import List, Tuple, Union

import torch
import numpy as np
from PIL import Image
from samgeo.text_sam import LangSAM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LangSAMProcessor:
    """
    A class for processing images using the LangSAM model for object detection and segmentation.
    """

    def __init__(self):
        """
        Initialize the LangSAMProcessor with the LangSAM model.
        """
        try:
            self.lang_sam = LangSAM()
            logging.info("LangSAM model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing LangSAM model: {str(e)}")
            raise

    def predict_dino(self, image: Union[np.ndarray, Image.Image], text_prompt: str, 
                     box_threshold: float = 0.3, text_threshold: float = 0.75) -> List[Tuple[float, float, float, float]]:
        """
        Perform object detection using the DINO model.

        Args:
            image (Union[np.ndarray, Image.Image]): The input image.
            text_prompt (str): The text prompt for object detection.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text prompts.

        Returns:
            List[Tuple[float, float, float, float]]: List of detected bounding boxes (x1, y1, x2, y2).

        Raises:
            ValueError: If there's an error during the detection process.
        """
        try:
            results = self.lang_sam.predict_dino(image=image, text_prompt=text_prompt,
                                                 box_threshold=box_threshold, text_threshold=text_threshold)
            return results[0]  # results[0] contains the bounding boxes
        except Exception as e:
            logging.error(f"Error in DINO prediction: {str(e)}")
            raise ValueError(f"DINO prediction failed: {str(e)}")

    def predict_sam(self, image: Union[np.ndarray, Image.Image], 
                    boxes: List[Tuple[float, float, float, float]]) -> torch.Tensor:
        """
        Perform segmentation using the SAM model.

        Args:
            image (Union[np.ndarray, Image.Image]): The input image.
            boxes (List[Tuple[float, float, float, float]]): List of bounding boxes to segment.

        Returns:
            torch.Tensor: Segmentation masks for the input boxes.

        Raises:
            ValueError: If there's an error during the segmentation process.
        """
        try:
            masks = self.lang_sam.predict_sam(image=image, boxes=torch.tensor(boxes))
            return masks
        except Exception as e:
            logging.error(f"Error in SAM prediction: {str(e)}")
            raise ValueError(f"SAM prediction failed: {str(e)}")

    def process_image(self, image: Union[np.ndarray, Image.Image], text_prompt: str, 
                      box_threshold: float = 0.3, text_threshold: float = 0.75) -> Tuple[List[Tuple[float, float, float, float]], torch.Tensor]:
        """
        Process an image through both DINO and SAM models.

        Args:
            image (Union[np.ndarray, Image.Image]): The input image.
            text_prompt (str): The text prompt for object detection.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text prompts.

        Returns:
            Tuple[List[Tuple[float, float, float, float]], torch.Tensor]: 
                Detected bounding boxes and corresponding segmentation masks.

        Raises:
            ValueError: If there's an error during the processing.
        """
        try:
            bounding_boxes = self.predict_dino(image, text_prompt, box_threshold, text_threshold)
            masks = self.predict_sam(image, bounding_boxes)
            return bounding_boxes, masks
        except Exception as e:
            logging.error(f"Error in image processing: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}")

    @staticmethod
    def create_mask_overlay(masks: torch.Tensor, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create a mask overlay from the predicted masks.

        Args:
            masks (torch.Tensor): Predicted segmentation masks.
            image_shape (Tuple[int, int, int]): Shape of the original image (height, width, channels).

        Returns:
            np.ndarray: Binary mask overlay.

        Raises:
            ValueError: If the mask or image shape is invalid.
        """
        if masks.dim() != 3:
            raise ValueError("Expected masks to be a 3D tensor (num_masks, height, width)")
        if len(image_shape) != 3:
            raise ValueError("Expected image_shape to be a 3-tuple (height, width, channels)")

        mask_overlay = np.zeros(image_shape[:2], dtype=np.uint8)
        for i, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy().astype(np.uint8)
            mask_overlay += ((mask > 0) * (i + 1)).astype(np.uint8)
        
        binary_overlay = (mask_overlay > 0) * 255  # Binary mask in [0, 255]
        logging.info(f"Created mask overlay with shape {binary_overlay.shape}")
        return binary_overlay
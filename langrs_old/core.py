import logging
from typing import List, Tuple
import json
import numpy as np
import os

from .config import LangRSConfig
from .image_processing import ImageProcessor
from .outlier_detection import apply_outlier_detection
from .visualization import (
    visualize_raster, visualize_bounding_boxes, visualize_sam_results,
    plot_confusion_matrix, plot_precision_recall_curve
)
from .evaluation import evaluate_detection, evaluate_segmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LangRS:
    def __init__(self, config_path: str):
        """
        Initialize the LangRS object with the given configuration file.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        self.config = LangRSConfig(config_path)
        self.image_processor = ImageProcessor(self.config)
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):
        """
        Main processing method that orchestrates the entire workflow.
        """
        try:
            # Process the image
            image, bboxes, mask_overlay = self.image_processor.process_image(self.config.get("image_input"))

            # Visualize input
            visualize_raster(image, save_path=os.path.join(self.output_dir, "input_image.png"))

            # Apply outlier detection to refine bounding boxes
            refined_bboxes = self._apply_outlier_detection(bboxes)

            # Visualize results
            self._visualize_results(image, refined_bboxes, mask_overlay)

            # Evaluate results if ground truth is available
            if self.config.get("evaluation"):
                self._evaluate_results(refined_bboxes, mask_overlay)

            logging.info("LangRS processing completed successfully")
        except Exception as e:
            logging.error(f"Error during LangRS processing: {str(e)}")
            raise

    def _apply_outlier_detection(self, bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Apply outlier detection to refine bounding boxes.

        Args:
            bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes.

        Returns:
            List[Tuple[float, float, float, float]]: Refined list of bounding boxes.
        """
        outlier_methods = self.config.get("outlier_methods", ["isolation_forest"])
        refined_bboxes = bboxes

        for method in outlier_methods:
            refined_bboxes = apply_outlier_detection(refined_bboxes, method=method)

        return refined_bboxes

    def _visualize_results(self, image, bboxes: List[Tuple[float, float, float, float]], mask_overlay):
        """
        Visualize the detection and segmentation results.

        Args:
            image: The input image.
            bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes.
            mask_overlay: The segmentation mask overlay.
        """
        visualize_bounding_boxes(image, bboxes, save_path=os.path.join(self.output_dir, "bounding_boxes.png"))
        visualize_sam_results(image, mask_overlay, save_path=os.path.join(self.output_dir, "segmentation_results.png"))

    def _evaluate_results(self, bboxes: List[Tuple[float, float, float, float]], mask_overlay: np.ndarray):
        """
        Evaluate the detection and segmentation results against ground truth.

        Args:
            bboxes (List[Tuple[float, float, float, float]]): List of predicted bounding boxes.
            mask_overlay (np.ndarray): Predicted segmentation mask.
        """
        # Load ground truth data
        with open(self.config.get("ground_truth_bb")) as f:
            gt_bboxes = json.load(f)
        with open(self.config.get("ground_truth_mask")) as f:
            gt_mask = np.array(json.load(f))

        # Evaluate object detection
        detection_metrics = evaluate_detection(bboxes, gt_bboxes)

        # Evaluate segmentation
        segmentation_metrics = evaluate_segmentation(mask_overlay, gt_mask)

        # Log or save the evaluation results
        logging.info(f"Detection Metrics: {detection_metrics}")
        logging.info(f"Segmentation Metrics: {segmentation_metrics}")

        # Optionally, save the metrics to a file
        with open(os.path.join(self.output_dir, "evaluation_results.json"), 'w') as f:
            json.dump({
                "detection": detection_metrics,
                "segmentation": segmentation_metrics
            }, f, indent=4)

        # Visualize evaluation results
        if 'confusion_matrix' in detection_metrics:
            plot_confusion_matrix(
                detection_metrics['confusion_matrix'],
                ['background', 'object'],
                save_path=os.path.join(self.output_dir, "confusion_matrix.png")
            )
        
        if 'precision' in detection_metrics and 'recall' in detection_metrics:
            plot_precision_recall_curve(
                detection_metrics['precision'],
                detection_metrics['recall'],
                save_path=os.path.join(self.output_dir, "precision_recall_curve.png")
            )
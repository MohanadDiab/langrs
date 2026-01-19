"""Main pipeline orchestrator for LangRS."""

from typing import Optional, List, Tuple, Dict, Union
import numpy as np
from PIL import Image
import torch

from ..models.base import DetectionModel, SegmentationModel
from ..processing.image_loader import ImageLoader, ImageData
from ..processing.tiling import TilingStrategy, Tile
from ..processing.outlier_detection import OutlierDetector
from ..visualization.base import Visualizer
from ..geospatial.exporter import (
    convert_bounding_boxes_to_geospatial,
    convert_masks_to_geospatial,
)
from ..io.output_manager import OutputManager
from ..core.config import LangRSConfig
from ..utils.exceptions import DetectionError, SegmentationError
from ..utils.types import BoundingBox


class LangRS:
    """
    Main class for remote sensing image segmentation.
    
    Uses dependency injection for all components, following SOLID principles.
    This allows for easy testing, swapping implementations, and extending functionality.
    
    Can be instantiated directly with simple parameters or with full dependency injection.
    """

    def __init__(
        self,
        output_path: str = "output",
        detection_model: Optional[str] = None,
        segmentation_model: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[LangRSConfig] = None,
        # Full dependency injection parameters (for advanced usage)
        _detection_model_instance: Optional[DetectionModel] = None,
        _segmentation_model_instance: Optional[SegmentationModel] = None,
        _image_loader: Optional[ImageLoader] = None,
        _tiling_strategy: Optional[TilingStrategy] = None,
        _outlier_detectors: Optional[Dict[str, OutlierDetector]] = None,
        _visualizer: Optional[Visualizer] = None,
        _output_manager: Optional[OutputManager] = None,
    ):
        """
        Initialize LangRS.
        
        Simple usage:
            langrs = LangRS(output_path="output")
        
        With custom models:
            langrs = LangRS(
                output_path="output",
                detection_model="grounding_dino",
                segmentation_model="sam",
                device="cpu"
            )
        
        Advanced usage (full dependency injection):
            langrs = LangRS(
                _detection_model_instance=detection_model,
                _segmentation_model_instance=segmentation_model,
                ...
            )
        
        Args:
            output_path: Base output directory
            detection_model: Name of detection model (default: "grounding_dino")
            segmentation_model: Name of segmentation model (default: "sam")
            device: Device to use ('cpu' or 'cuda', default: auto-detect)
            config: Optional configuration object
            _detection_model_instance: (Advanced) Detection model instance
            _segmentation_model_instance: (Advanced) Segmentation model instance
            _image_loader: (Advanced) Image loader instance
            _tiling_strategy: (Advanced) Tiling strategy instance
            _outlier_detectors: (Advanced) Dictionary of outlier detector instances
            _visualizer: (Advanced) Visualizer instance
            _output_manager: (Advanced) Output manager instance
        """
        # If full dependency injection is provided, use it
        if _detection_model_instance is not None:
            self.detection_model = _detection_model_instance
            self.segmentation_model = _segmentation_model_instance
            self.image_loader = _image_loader
            self.tiling_strategy = _tiling_strategy
            self.outlier_detectors = _outlier_detectors
            self.visualizer = _visualizer
            self.output_manager = _output_manager
            self.config = config or LangRSConfig()
        else:
            # Use builder for simple initialization
            from .builder import LangRSBuilder
            
            builder = LangRSBuilder()
            if config:
                builder.with_config(config)
            if device:
                builder.with_device(device)
            if detection_model:
                builder.with_detection_model(detection_model)
            if segmentation_model:
                builder.with_segmentation_model(segmentation_model)
            builder.with_output_path(output_path)
            
            built = builder.build()
            # Copy all attributes
            self.detection_model = built.detection_model
            self.segmentation_model = built.segmentation_model
            self.image_loader = built.image_loader
            self.tiling_strategy = built.tiling_strategy
            self.outlier_detectors = built.outlier_detectors
            self.visualizer = built.visualizer
            self.output_manager = built.output_manager
            self.config = built.config

        # Pipeline state
        self.image_data: Optional[ImageData] = None
        self.bounding_boxes: List[BoundingBox] = []
        self.filtered_boxes: Dict[str, List[BoundingBox]] = {}
        self.masks: Optional[np.ndarray] = None
        self.areas: Optional[np.ndarray] = None

    def load_image(self, image_source) -> None:
        """
        Load and prepare image.
        
        Args:
            image_source: Image source (file path, numpy array, or PIL Image)
        """
        self.image_data = self.image_loader.load(image_source)

    def detect_objects(
        self,
        text_prompt: str,
        window_size: Optional[int] = None,
        overlap: Optional[int] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[BoundingBox]:
        """
        Detect objects using sliding window approach.
        
        Args:
            text_prompt: Text description of objects to detect
            window_size: Size of each tile (uses config default if None)
            overlap: Overlap between tiles (uses config default if None)
            box_threshold: Confidence threshold for boxes (uses config default if None)
            text_threshold: Confidence threshold for text (uses config default if None)
            
        Returns:
            List of detected bounding boxes
            
        Raises:
            DetectionError: If detection fails
        """
        if self.image_data is None:
            raise DetectionError("Image not loaded. Call load_image() first.")

        if not self.detection_model.is_loaded:
            self.detection_model.load_weights()

        # Use config defaults if not provided (explicitly check for None)
        window_size = window_size if window_size is not None else self.config.detection.window_size
        overlap = overlap if overlap is not None else self.config.detection.overlap
        box_threshold = box_threshold if box_threshold is not None else self.config.detection.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.config.detection.text_threshold

        try:
            # Create tiles
            tiles = self.tiling_strategy.create_tiles(
                self.image_data.pil_image, window_size, overlap
            )

            # Detect objects in each tile
            all_boxes = []
            for tile in tiles:
                # Detect in tile
                tile_boxes = self.detection_model.detect(
                    tile.image,
                    text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                # Convert tile coordinates to image coordinates
                for box in tile_boxes:
                    x_min, y_min, x_max, y_max = box
                    global_box = (
                        x_min + tile.offset_x,
                        y_min + tile.offset_y,
                        x_max + tile.offset_x,
                        y_max + tile.offset_y,
                    )
                    all_boxes.append(global_box)

            self.bounding_boxes = all_boxes

            # Calculate areas
            self._calculate_areas()

            # Visualize bounding boxes
            output_path = self.output_manager.get_path_str("results_dino.jpg")
            self.visualizer.plot_boxes(
                self.image_data.pil_image,
                self.bounding_boxes,
                output_path,
                box_color=self.config.visualization.box_color,
                linewidth=self.config.visualization.box_linewidth,
            )

            # Save geospatial data if available
            if self.image_data.is_georeferenced and self.image_data.image_path:
                gdf_boxes = convert_bounding_boxes_to_geospatial(
                    self.bounding_boxes, image_path=self.image_data.image_path
                )
                gdf_boxes.to_file(
                    self.output_manager.get_path_str("bounding_boxes.shp")
                )

            return self.bounding_boxes

        except Exception as e:
            raise DetectionError(f"Object detection failed: {e}") from e

    def filter_outliers(
        self,
        method: Optional[str] = None,
    ) -> Dict[str, List[BoundingBox]]:
        """
        Apply outlier detection methods to filter bounding boxes.
        
        Args:
            method: Optional specific method name. If None, applies all methods.
            
        Returns:
            Dictionary mapping method names to filtered bounding boxes
        """
        if len(self.bounding_boxes) == 0:
            raise ValueError("No bounding boxes to filter. Run detect_objects() first.")

        if self.areas is None:
            self._calculate_areas()

        filtered_results = {}

        if method is not None:
            # Apply single method
            if method not in self.outlier_detectors:
                raise ValueError(f"Unknown outlier detection method: {method}")
            detector = self.outlier_detectors[method]
            predictions, filtered_boxes = detector.detect(self.areas, self.bounding_boxes)
            filtered_results[method] = filtered_boxes

            # Visualize
            output_path = self.output_manager.get_path_str(f"results_{method}.jpg")
            self.visualizer.plot_scatter(
                self.areas,
                outliers=predictions,
                title=f"{method.upper()} Outlier Detection",
                output_path=output_path,
                ylabel="Area (px sq.)",
            )

            # Visualize filtered boxes
            filtered_output_path = self.output_manager.get_path_str(
                f"results_{method}_filtered.jpg"
            )
            self.visualizer.plot_boxes(
                self.image_data.pil_image,
                filtered_boxes,
                filtered_output_path,
                box_color=self.config.visualization.box_color,
                linewidth=self.config.visualization.box_linewidth,
            )
        else:
            # Apply all methods
            for method_name, detector in self.outlier_detectors.items():
                predictions, filtered_boxes = detector.detect(
                    self.areas, self.bounding_boxes
                )
                filtered_results[method_name] = filtered_boxes

                # Visualize
                output_path = self.output_manager.get_path_str(
                    f"results_{method_name}.jpg"
                )
                self.visualizer.plot_scatter(
                    self.areas,
                    outliers=predictions,
                    title=f"{method_name.upper()} Outlier Detection",
                    output_path=output_path,
                    ylabel="Area (px sq.)",
                )

                # Visualize filtered boxes
                filtered_output_path = self.output_manager.get_path_str(
                    f"results_{method_name}_filtered.jpg"
                )
                self.visualizer.plot_boxes(
                    self.image_data.pil_image,
                    filtered_boxes,
                    filtered_output_path,
                    box_color=self.config.visualization.box_color,
                    linewidth=self.config.visualization.box_linewidth,
                )

        self.filtered_boxes = filtered_results
        return filtered_results

    def segment(
        self,
        boxes: Optional[List[BoundingBox]] = None,
        window_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate segmentation masks for bounding boxes.
        
        Args:
            boxes: Optional list of bounding boxes. If None, uses detected boxes.
            window_size: Window size for tiling (uses config default if None)
            overlap: Overlap between windows (uses config default if None)
            
        Returns:
            Segmentation mask as numpy array
            
        Raises:
            SegmentationError: If segmentation fails
        """
        if self.image_data is None:
            raise SegmentationError("Image not loaded. Call load_image() first.")

        if not self.segmentation_model.is_loaded:
            self.segmentation_model.load_weights()

        if boxes is None:
            if len(self.bounding_boxes) == 0:
                raise SegmentationError(
                    "No bounding boxes available. Run detect_objects() first."
                )
            boxes = self.bounding_boxes

        # Use config defaults if not provided (explicitly check for None)
        window_size = window_size if window_size is not None else self.config.segmentation.window_size
        overlap = overlap if overlap is not None else self.config.segmentation.overlap

        try:
            # Convert boxes to tensor
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

            # Create tiles for segmentation
            tiles = self.tiling_strategy.create_tiles(
                self.image_data.np_image, window_size, overlap
            )

            # Process each tile
            all_masks_info = []
            for tile in tiles:
                # Find boxes that intersect with this tile
                tile_boxes = []
                tile_box_indices = []
                for idx, box in enumerate(boxes):
                    bx_min, by_min, bx_max, by_max = box
                    if (
                        bx_min < tile.x_max
                        and bx_max > tile.x_min
                        and by_min < tile.y_max
                        and by_max > tile.y_min
                    ):
                        # Adjust box coordinates relative to tile
                        adj_box = [
                            max(0, bx_min - tile.offset_x),
                            max(0, by_min - tile.offset_y),
                            min(tile.x_max - tile.offset_x, bx_max - tile.offset_x),
                            min(tile.y_max - tile.offset_y, by_max - tile.offset_y),
                        ]
                        tile_boxes.append(adj_box)
                        tile_box_indices.append(idx)

                if tile_boxes:
                    tile_boxes_tensor = torch.tensor(tile_boxes, dtype=torch.float32)
                    tile_image = (
                        tile.image
                        if isinstance(tile.image, np.ndarray)
                        else np.array(tile.image)
                    )
                    masks = self.segmentation_model.segment(tile_image, tile_boxes_tensor)

                    # Store mask info with global coordinates
                    for i, mask in enumerate(masks):
                        all_masks_info.append(
                            (mask, tile.x_min, tile.y_min, tile.x_max, tile.y_max)
                        )

            if not all_masks_info:
                raise SegmentationError("No masks were generated.")

            # Merge masks
            self.masks = self._merge_masks(
                all_masks_info, self.image_data.np_image.shape[:2]
            )

            # Visualize masks
            output_path = self.output_manager.get_path_str("results_sam.jpg")
            self.visualizer.plot_masks(
                self.image_data.np_image,
                self.masks,
                output_path,
                alpha=self.config.visualization.mask_alpha,
                colormap=self.config.visualization.mask_colormap,
            )

            # Save geospatial data if available
            if self.image_data.is_georeferenced and self.image_data.image_path:
                if boxes != self.bounding_boxes:
                    # Save filtered boxes if different
                    gdf_boxes = convert_bounding_boxes_to_geospatial(
                        boxes, image_path=self.image_data.image_path
                    )
                    gdf_boxes.to_file(
                        self.output_manager.get_path_str("bounding_boxes_filtered.shp")
                    )

                gdf_masks = convert_masks_to_geospatial(
                    self.masks, image_path=self.image_data.image_path
                )
                gdf_masks.to_file(self.output_manager.get_path_str("masks.shp"))

            return self.masks

        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {e}") from e

    def run_full_pipeline(
        self,
        image_source,
        text_prompt: str,
        **kwargs
    ) -> np.ndarray:
        """
        Run complete pipeline: load -> detect -> filter -> segment.
        
        Args:
            image_source: Image source (file path, numpy array, or PIL Image)
            text_prompt: Text description of objects to detect
            **kwargs: Additional arguments passed to detect_objects()
            
        Returns:
            Segmentation mask as numpy array
        """
        self.load_image(image_source)
        self.detect_objects(text_prompt, **kwargs)
        self.filter_outliers()
        return self.segment()

    def _calculate_areas(self) -> None:
        """Calculate areas of bounding boxes."""
        if len(self.bounding_boxes) == 0:
            self.areas = np.array([])
            return

        areas = [
            (x_max - x_min) * (y_max - y_min)
            for x_min, y_min, x_max, y_max in self.bounding_boxes
        ]
        self.areas = np.array(areas).reshape(-1, 1)

        # Visualize area distribution
        output_path = self.output_manager.get_path_str("results_areas.jpg")
        self.visualizer.plot_scatter(
            self.areas,
            title="Bounding Box Areas",
            output_path=output_path,
            ylabel="Area (px sq.)",
        )

    def _merge_masks(
        self, masks_info: List[Tuple[torch.Tensor, int, int, int, int]], full_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Merge masks from tiles into full-size mask.
        
        Args:
            masks_info: List of (mask, x_min, y_min, x_max, y_max) tuples
            full_size: Full image size (height, width)
            
        Returns:
            Merged mask as numpy array
        """
        merged_mask = np.zeros(full_size, dtype=np.uint8)

        for mask, x_min, y_min, x_max, y_max in masks_info:
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            mask = mask.squeeze()

            # Convert to binary mask
            mask_binary = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            if mask.ndim == 2:
                mask_binary[mask > 0] = 255
            elif mask.ndim == 3:
                # Handle multiple masks - take union
                mask_binary[np.any(mask > 0, axis=0)] = 255

            # Merge into full mask
            merged_mask[y_min:y_max, x_min:x_max] = np.maximum(
                merged_mask[y_min:y_max, x_min:x_max], mask_binary
            )

        return merged_mask

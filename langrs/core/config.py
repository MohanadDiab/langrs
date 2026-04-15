"""Configuration management for LangRS pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import json
from pathlib import Path

from ..utils.exceptions import ConfigurationError


@dataclass
class DetectionConfig:
    """Configuration for object detection."""

    box_threshold: float = 0.3
    text_threshold: float = 0.3
    window_size: int = 500
    overlap: int = 200
    # Rex-Omni (MLLM) tiling controls:
    # - "auto": prefer full-image when it likely fits, else tile
    # - "always": always tile
    # - "never": never tile (single full-image call)
    tiling_mode: str = "auto"
    # Hard cap to avoid exploding number of tiles (MLLM calls).
    max_tiles: int = 64
    # Heuristic: if original image area <= this, run as single full-image call in auto mode.
    # (Rex-Omni will internally resize via smart_resize anyway.)
    full_image_max_area: int = 2_000_000
    # Optional NMS on aggregated detections (may help speed downstream steps).
    apply_nms: bool = False
    nms_iou_threshold: float = 0.5

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ConfigurationError(
                f"box_threshold must be between 0.0 and 1.0, got {self.box_threshold}"
            )
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ConfigurationError(
                f"text_threshold must be between 0.0 and 1.0, got {self.text_threshold}"
            )
        if self.window_size <= 0:
            raise ConfigurationError(
                f"window_size must be positive, got {self.window_size}"
            )
        if self.overlap < 0:
            raise ConfigurationError(
                f"overlap must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.window_size:
            raise ConfigurationError(
                f"overlap ({self.overlap}) must be less than window_size ({self.window_size})"
            )
        if self.tiling_mode not in {"auto", "always", "never"}:
            raise ConfigurationError(
                f"tiling_mode must be one of 'auto', 'always', 'never', got {self.tiling_mode!r}"
            )
        if self.max_tiles <= 0:
            raise ConfigurationError(
                f"max_tiles must be positive, got {self.max_tiles}"
            )
        if self.full_image_max_area <= 0:
            raise ConfigurationError(
                f"full_image_max_area must be positive, got {self.full_image_max_area}"
            )
        if not 0.0 < self.nms_iou_threshold <= 1.0:
            raise ConfigurationError(
                f"nms_iou_threshold must be between 0.0 and 1.0, got {self.nms_iou_threshold}"
            )


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""

    multimask_output: bool = False
    window_size: int = 500
    overlap: int = 200
    # Behavior for run_full_pipeline(): whether to use filtered boxes for segmentation.
    use_filtered_boxes: bool = True
    # Which outlier method to use when use_filtered_boxes=True.
    # - None: pick a deterministic first method
    # - "union": union boxes from all methods
    # - otherwise: use that method key (e.g. "zscore")
    filtered_boxes_method: Optional[str] = None

    def __post_init__(self):
        """Validate configuration values."""
        if self.window_size <= 0:
            raise ConfigurationError(
                f"window_size must be positive, got {self.window_size}"
            )
        if self.overlap < 0:
            raise ConfigurationError(
                f"overlap must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.window_size:
            raise ConfigurationError(
                f"overlap ({self.overlap}) must be less than window_size ({self.window_size})"
            )
        if self.filtered_boxes_method is not None and not isinstance(self.filtered_boxes_method, str):
            raise ConfigurationError(
                f"filtered_boxes_method must be a string or None, got {type(self.filtered_boxes_method).__name__}"
            )


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection methods."""

    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    svm_nu: float = 0.1
    isolation_forest_contamination: float = 0.25
    lof_contamination: float = 0.25
    lof_n_neighbors: int = 20
    robust_cov_contamination: float = 0.25

    def __post_init__(self):
        """Validate configuration values."""
        if self.zscore_threshold <= 0:
            raise ConfigurationError(
                f"zscore_threshold must be positive, got {self.zscore_threshold}"
            )
        if self.iqr_multiplier <= 0:
            raise ConfigurationError(
                f"iqr_multiplier must be positive, got {self.iqr_multiplier}"
            )
        if not 0.0 < self.svm_nu <= 1.0:
            raise ConfigurationError(
                f"svm_nu must be between 0.0 and 1.0, got {self.svm_nu}"
            )
        if not 0.0 < self.isolation_forest_contamination <= 0.5:
            raise ConfigurationError(
                f"isolation_forest_contamination must be between 0.0 and 0.5, "
                f"got {self.isolation_forest_contamination}"
            )
        if not 0.0 < self.lof_contamination <= 0.5:
            raise ConfigurationError(
                f"lof_contamination must be between 0.0 and 0.5, "
                f"got {self.lof_contamination}"
            )
        if self.lof_n_neighbors <= 0:
            raise ConfigurationError(
                f"lof_n_neighbors must be positive, got {self.lof_n_neighbors}"
            )
        if not 0.0 < self.robust_cov_contamination <= 0.5:
            raise ConfigurationError(
                f"robust_cov_contamination must be between 0.0 and 0.5, "
                f"got {self.robust_cov_contamination}"
            )


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: tuple = (10, 10)
    box_color: str = "r"
    box_linewidth: int = 1
    mask_alpha: float = 0.4
    mask_colormap: str = "viridis"

    def __post_init__(self):
        """Validate configuration values."""
        if len(self.figsize) != 2:
            raise ConfigurationError(
                f"figsize must be a tuple of length 2, got {self.figsize}"
            )
        if any(s <= 0 for s in self.figsize):
            raise ConfigurationError(
                f"figsize values must be positive, got {self.figsize}"
            )
        if self.box_linewidth <= 0:
            raise ConfigurationError(
                f"box_linewidth must be positive, got {self.box_linewidth}"
            )
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ConfigurationError(
                f"mask_alpha must be between 0.0 and 1.0, got {self.mask_alpha}"
            )


@dataclass
class LangRSConfig:
    """
    Main configuration for LangRS pipeline.
    
    This configuration object holds all sub-configurations and can be
    loaded from dictionaries or YAML/JSON files.
    """

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    outlier_detection: OutlierDetectionConfig = field(
        default_factory=OutlierDetectionConfig
    )
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LangRSConfig":
        """
        Create config from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values. Can contain
                        nested dictionaries for sub-configs.
                        
        Returns:
            LangRSConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid
            
        Example:
            config = LangRSConfig.from_dict({
                "detection": {
                    "box_threshold": 0.25,
                    "text_threshold": 0.25
                },
                "segmentation": {
                    "multimask_output": False
                }
            })
        """
        try:
            detection_dict = config_dict.get("detection", {})
            segmentation_dict = config_dict.get("segmentation", {})
            outlier_dict = config_dict.get("outlier_detection", {})
            viz_dict = config_dict.get("visualization", {})
            
            # Convert figsize list to tuple if present
            if "figsize" in viz_dict and isinstance(viz_dict["figsize"], list):
                viz_dict["figsize"] = tuple(viz_dict["figsize"])

            return cls(
                detection=DetectionConfig(**detection_dict),
                segmentation=SegmentationConfig(**segmentation_dict),
                outlier_detection=OutlierDetectionConfig(**outlier_dict),
                visualization=VisualizationConfig(**viz_dict),
            )
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LangRSConfig":
        """
        Load config from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            LangRSConfig instance
            
        Raises:
            ConfigurationError: If file cannot be read or is invalid
            FileNotFoundError: If file does not exist
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
            if config_dict is None:
                config_dict = {}
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading YAML config: {e}") from e

    @classmethod
    def from_json(cls, json_path: str) -> "LangRSConfig":
        """
        Load config from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            LangRSConfig instance
            
        Raises:
            ConfigurationError: If file cannot be read or is invalid
            FileNotFoundError: If file does not exist
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        try:
            with open(path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading JSON config: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "detection": {
                "box_threshold": self.detection.box_threshold,
                "text_threshold": self.detection.text_threshold,
                "window_size": self.detection.window_size,
                "overlap": self.detection.overlap,
                "tiling_mode": self.detection.tiling_mode,
                "max_tiles": self.detection.max_tiles,
                "full_image_max_area": self.detection.full_image_max_area,
                "apply_nms": self.detection.apply_nms,
                "nms_iou_threshold": self.detection.nms_iou_threshold,
            },
            "segmentation": {
                "multimask_output": self.segmentation.multimask_output,
                "window_size": self.segmentation.window_size,
                "overlap": self.segmentation.overlap,
                "use_filtered_boxes": self.segmentation.use_filtered_boxes,
                "filtered_boxes_method": self.segmentation.filtered_boxes_method,
            },
            "outlier_detection": {
                "zscore_threshold": self.outlier_detection.zscore_threshold,
                "iqr_multiplier": self.outlier_detection.iqr_multiplier,
                "svm_nu": self.outlier_detection.svm_nu,
                "isolation_forest_contamination": self.outlier_detection.isolation_forest_contamination,
                "lof_contamination": self.outlier_detection.lof_contamination,
                "lof_n_neighbors": self.outlier_detection.lof_n_neighbors,
                "robust_cov_contamination": self.outlier_detection.robust_cov_contamination,
            },
            "visualization": {
                "figsize": list(self.visualization.figsize),  # Convert tuple to list for YAML compatibility
                "box_color": self.visualization.box_color,
                "box_linewidth": self.visualization.box_linewidth,
                "mask_alpha": self.visualization.mask_alpha,
                "mask_colormap": self.visualization.mask_colormap,
            },
        }

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save config to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
            
        Raises:
            ConfigurationError: If file cannot be written
        """
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except Exception as e:
            raise ConfigurationError(f"Error saving YAML config: {e}") from e

    def to_json(self, json_path: str) -> None:
        """
        Save config to JSON file.
        
        Args:
            json_path: Path to save JSON file
            
        Raises:
            ConfigurationError: If file cannot be written
        """
        try:
            with open(json_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving JSON config: {e}") from e

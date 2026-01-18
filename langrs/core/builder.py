"""Builder for creating LangRS pipeline instances."""

from typing import Optional, Dict
import torch

from .pipeline import LangRS
from .config import LangRSConfig
from ..models.factory import ModelFactory
from ..processing.image_loader import ImageLoader
from ..processing.tiling import SlidingWindowTiler
from ..processing.outlier_detection import (
    ZScoreOutlierDetector,
    IQROutlierDetector,
    RobustCovarianceOutlierDetector,
    SVMOutlierDetector,
    IsolationForestOutlierDetector,
    LOFOutlierDetector,
)
from ..visualization.matplotlib_viz import MatplotlibVisualizer
from ..io.output_manager import OutputManager
from ..utils.exceptions import ModelLoadError


class LangRSPipelineBuilder:
    """
    Builder for creating LangRS pipeline instances.
    
    Provides a fluent interface for configuring and building pipelines.
    """

    def __init__(self):
        """Initialize builder with defaults."""
        self.config: Optional[LangRSConfig] = None
        self.detection_model_name: str = "grounding_dino"
        self.segmentation_model_name: str = "sam"
        self.detection_model_path: Optional[str] = None
        self.segmentation_model_path: Optional[str] = None
        self.device: Optional[str] = None
        self.output_path: str = "output"
        self.create_timestamped: bool = True

    def with_config(self, config: LangRSConfig) -> "LangRSPipelineBuilder":
        """
        Set configuration.
        
        Args:
            config: LangRSConfig instance
            
        Returns:
            Self for method chaining
        """
        self.config = config
        return self

    def with_detection_model(
        self, model_name: str, model_path: Optional[str] = None
    ) -> "LangRSPipelineBuilder":
        """
        Set detection model.
        
        Args:
            model_name: Name of detection model
            model_path: Optional path to model checkpoint
            
        Returns:
            Self for method chaining
        """
        self.detection_model_name = model_name
        self.detection_model_path = model_path
        return self

    def with_segmentation_model(
        self, model_name: str, model_path: Optional[str] = None
    ) -> "LangRSPipelineBuilder":
        """
        Set segmentation model.
        
        Args:
            model_name: Name of segmentation model
            model_path: Optional path to model checkpoint
            
        Returns:
            Self for method chaining
        """
        self.segmentation_model_name = model_name
        self.segmentation_model_path = model_path
        return self

    def with_device(self, device: str) -> "LangRSPipelineBuilder":
        """
        Set device for models.
        
        Args:
            device: Device string ('cpu' or 'cuda')
            
        Returns:
            Self for method chaining
        """
        self.device = device
        return self

    def with_output_path(
        self, output_path: str, create_timestamped: bool = True
    ) -> "LangRSPipelineBuilder":
        """
        Set output path.
        
        Args:
            output_path: Base output directory
            create_timestamped: Whether to create timestamped subdirectory
            
        Returns:
            Self for method chaining
        """
        self.output_path = output_path
        self.create_timestamped = create_timestamped
        return self

    def build(self) -> LangRS:
        """
        Build the pipeline with configured components.
        
        Returns:
            Configured LangRS instance
            
        Raises:
            ModelLoadError: If model creation fails
        """
        # Use default config if not provided
        if self.config is None:
            self.config = LangRSConfig()

        # Determine device
        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # Create models
        try:
            detection_model = ModelFactory.create_detection_model(
                model_name=self.detection_model_name,
                model_path=self.detection_model_path,
                device=device,
            )

            segmentation_model = ModelFactory.create_segmentation_model(
                model_name=self.segmentation_model_name,
                model_path=self.segmentation_model_path,
                device=device,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to create models: {e}") from e

        # Create other components
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        visualizer = MatplotlibVisualizer(
            figsize=self.config.visualization.figsize, dpi=100
        )
        output_manager = OutputManager(
            self.output_path, create_timestamped=self.create_timestamped
        )

        # Create outlier detectors from config
        outlier_detectors = {
            "zscore": ZScoreOutlierDetector(
                threshold=self.config.outlier_detection.zscore_threshold
            ),
            "iqr": IQROutlierDetector(
                multiplier=self.config.outlier_detection.iqr_multiplier
            ),
            "robust_covariance": RobustCovarianceOutlierDetector(
                contamination=self.config.outlier_detection.robust_cov_contamination
            ),
            "svm": SVMOutlierDetector(
                nu=self.config.outlier_detection.svm_nu, kernel="rbf", gamma=0.1
            ),
            "svm_sgd": SVMOutlierDetector(
                nu=self.config.outlier_detection.svm_nu, kernel="linear"
            ),
            "isolation_forest": IsolationForestOutlierDetector(
                contamination=self.config.outlier_detection.isolation_forest_contamination
            ),
            "lof": LOFOutlierDetector(
                contamination=self.config.outlier_detection.lof_contamination,
                n_neighbors=self.config.outlier_detection.lof_n_neighbors,
            ),
        }

        return LangRS(
            detection_model=detection_model,
            segmentation_model=segmentation_model,
            image_loader=image_loader,
            tiling_strategy=tiling_strategy,
            outlier_detectors=outlier_detectors,
            visualizer=visualizer,
            output_manager=output_manager,
            config=self.config,
        )

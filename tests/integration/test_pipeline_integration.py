"""Integration tests for complete pipeline workflows."""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from langrs import LangRS, LangRSPipelineBuilder, LangRSConfig
from langrs.models.base import DetectionModel, SegmentationModel
from langrs.processing.image_loader import ImageLoader
from langrs.processing.tiling import SlidingWindowTiler
from langrs.processing.outlier_detection import ZScoreOutlierDetector, IQROutlierDetector
from langrs.visualization.matplotlib_viz import MatplotlibVisualizer
from langrs.io.output_manager import OutputManager
import torch


class MockDetectionModel(DetectionModel):
    """Mock detection model for integration testing."""

    def __init__(self):
        self._device = torch.device("cpu")
        self._loaded = False

    def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
        # Return mock boxes
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]
        return [
            (w * 0.1, h * 0.1, w * 0.3, h * 0.3),
            (w * 0.5, h * 0.5, w * 0.7, h * 0.7),
        ]

    def load_weights(self, model_path=None):
        self._loaded = True

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self):
        return self._loaded


class MockSegmentationModel(SegmentationModel):
    """Mock segmentation model for integration testing."""

    def __init__(self):
        self._device = torch.device("cpu")
        self._loaded = False

    def segment(self, image, boxes):
        n = boxes.shape[0]
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.size[1], image.size[0]
        return torch.zeros((n, h, w), dtype=torch.float32)

    def load_weights(self, model_path=None):
        self._loaded = True

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self):
        return self._loaded


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new("RGB", (200, 200), color="blue")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_full_pipeline_workflow(self, sample_image, temp_output_dir):
        """Test complete pipeline workflow."""
        # Use mock models for testing
        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {
            "zscore": ZScoreOutlierDetector(),
            "iqr": IQROutlierDetector(),
        }
        visualizer = MatplotlibVisualizer()
        output_manager = OutputManager(temp_output_dir, create_timestamped=False)
        config = LangRSConfig()

        pipeline = LangRS(
            _detection_model_instance=detection_model,
            _segmentation_model_instance=segmentation_model,
            _image_loader=image_loader,
            _tiling_strategy=tiling_strategy,
            _outlier_detectors=outlier_detectors,
            _visualizer=visualizer,
            _output_manager=output_manager,
            config=config,
        )

        # Run full pipeline
        masks = pipeline.run_full_pipeline(
            sample_image, "test object", window_size=150, overlap=50
        )

        # Verify results
        assert masks is not None
        assert isinstance(masks, np.ndarray)
        assert pipeline.image_data is not None
        assert len(pipeline.bounding_boxes) > 0
        assert len(pipeline.filtered_boxes) > 0

        # Verify outputs were created
        output_files = list(Path(temp_output_dir).glob("*.jpg"))
        assert len(output_files) > 0

    def test_pipeline_step_by_step(self, sample_image, temp_output_dir):
        """Test pipeline step by step."""
        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {"zscore": ZScoreOutlierDetector()}
        visualizer = MatplotlibVisualizer()
        output_manager = OutputManager(temp_output_dir, create_timestamped=False)
        config = LangRSConfig()

        pipeline = LangRS(
            _detection_model_instance=detection_model,
            _segmentation_model_instance=segmentation_model,
            _image_loader=image_loader,
            _tiling_strategy=tiling_strategy,
            _outlier_detectors=outlier_detectors,
            _visualizer=visualizer,
            _output_manager=output_manager,
            config=config,
        )

        # Step 1: Load image
        pipeline.load_image(sample_image)
        assert pipeline.image_data is not None

        # Step 2: Detect objects
        boxes = pipeline.detect_objects("test", window_size=150, overlap=50)
        assert len(boxes) > 0

        # Step 3: Filter outliers
        filtered = pipeline.filter_outliers(method="zscore")
        assert "zscore" in filtered

        # Step 4: Segment
        masks = pipeline.segment(boxes=filtered["zscore"])
        assert masks is not None

    def test_pipeline_with_custom_config(self, sample_image, temp_output_dir):
        """Test pipeline with custom configuration."""
        config = LangRSConfig()
        config.detection.box_threshold = 0.25
        config.detection.text_threshold = 0.25
        config.detection.window_size = 150
        config.detection.overlap = 50

        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {"zscore": ZScoreOutlierDetector()}
        visualizer = MatplotlibVisualizer()
        output_manager = OutputManager(temp_output_dir, create_timestamped=False)

        pipeline = LangRS(
            _detection_model_instance=detection_model,
            _segmentation_model_instance=segmentation_model,
            _image_loader=image_loader,
            _tiling_strategy=tiling_strategy,
            _outlier_detectors=outlier_detectors,
            _visualizer=visualizer,
            _output_manager=output_manager,
            config=config,
        )

        boxes = pipeline.run_full_pipeline(sample_image, "test", window_size=150, overlap=50)
        assert boxes is not None

    def test_pipeline_with_numpy_image(self, temp_output_dir):
        """Test pipeline with numpy array image."""
        np_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {"zscore": ZScoreOutlierDetector()}
        visualizer = MatplotlibVisualizer()
        output_manager = OutputManager(temp_output_dir, create_timestamped=False)
        config = LangRSConfig()

        pipeline = LangRS(
            _detection_model_instance=detection_model,
            _segmentation_model_instance=segmentation_model,
            _image_loader=image_loader,
            _tiling_strategy=tiling_strategy,
            _outlier_detectors=outlier_detectors,
            _visualizer=visualizer,
            _output_manager=output_manager,
            config=config,
        )

        masks = pipeline.run_full_pipeline(np_image, "test", window_size=150, overlap=50)
        assert masks is not None

    def test_pipeline_error_handling(self, temp_output_dir):
        """Test pipeline error handling."""
        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {"zscore": ZScoreOutlierDetector()}
        visualizer = MatplotlibVisualizer()
        output_manager = OutputManager(temp_output_dir, create_timestamped=False)
        config = LangRSConfig()

        pipeline = LangRS(
            _detection_model_instance=detection_model,
            _segmentation_model_instance=segmentation_model,
            _image_loader=image_loader,
            _tiling_strategy=tiling_strategy,
            _outlier_detectors=outlier_detectors,
            _visualizer=visualizer,
            _output_manager=output_manager,
            config=config,
        )

        # Test error when image not loaded
        from langrs.utils.exceptions import DetectionError
        with pytest.raises(DetectionError):
            pipeline.detect_objects("test")

        # Test error when no boxes
        from langrs.utils.exceptions import SegmentationError
        sample_image = Image.new("RGB", (200, 200))
        pipeline.load_image(sample_image)
        with pytest.raises(SegmentationError):
            pipeline.segment()

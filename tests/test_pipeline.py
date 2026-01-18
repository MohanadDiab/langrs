"""Tests for LangRS pipeline."""

import pytest
import numpy as np
from PIL import Image
import torch
import tempfile
from pathlib import Path

from langrs.core.pipeline import LangRS
from langrs.core.config import LangRSConfig
from langrs.models.base import DetectionModel, SegmentationModel
from langrs.processing.image_loader import ImageLoader
from langrs.processing.tiling import SlidingWindowTiler
from langrs.processing.outlier_detection import ZScoreOutlierDetector
from langrs.visualization.matplotlib_viz import MatplotlibVisualizer
from langrs.io.output_manager import OutputManager
from langrs.utils.exceptions import DetectionError, SegmentationError


class MockDetectionModel(DetectionModel):
    """Mock detection model for testing."""

    def __init__(self):
        self._device = torch.device("cpu")
        self._loaded = False

    def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
        # Return some mock boxes
        return [(10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 90.0, 90.0)]

    def load_weights(self, model_path=None):
        self._loaded = True

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self):
        return self._loaded


class MockSegmentationModel(SegmentationModel):
    """Mock segmentation model for testing."""

    def __init__(self):
        self._device = torch.device("cpu")
        self._loaded = False

    def segment(self, image, boxes):
        n = boxes.shape[0]
        h, w = 100, 100
        return torch.zeros((n, h, w), dtype=torch.float32)

    def load_weights(self, model_path=None):
        self._loaded = True

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self):
        return self._loaded


class TestLangRS:
    """Test LangRS class."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        detection_model = MockDetectionModel()
        segmentation_model = MockSegmentationModel()
        image_loader = ImageLoader()
        tiling_strategy = SlidingWindowTiler()
        outlier_detectors = {"zscore": ZScoreOutlierDetector()}
        visualizer = MatplotlibVisualizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_manager = OutputManager(tmpdir, create_timestamped=False)
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
            yield pipeline

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.image_data is None
        assert len(pipeline.bounding_boxes) == 0
        assert pipeline.masks is None

    def test_load_image(self, pipeline, sample_image):
        """Test loading image."""
        pipeline.load_image(sample_image)
        assert pipeline.image_data is not None
        assert isinstance(pipeline.image_data.pil_image, Image.Image)

    def test_detect_objects_without_image(self, pipeline):
        """Test detect_objects raises error if image not loaded."""
        with pytest.raises(DetectionError, match="Image not loaded"):
            pipeline.detect_objects("test prompt")

    def test_detect_objects(self, pipeline, sample_image):
        """Test object detection."""
        pipeline.load_image(sample_image)
        boxes = pipeline.detect_objects("test prompt", window_size=100, overlap=0)
        
        assert len(boxes) > 0
        assert len(pipeline.bounding_boxes) > 0
        assert pipeline.areas is not None

    def test_filter_outliers_without_boxes(self, pipeline):
        """Test filter_outliers raises error if no boxes."""
        with pytest.raises(ValueError, match="No bounding boxes"):
            pipeline.filter_outliers()

    def test_filter_outliers_single_method(self, pipeline, sample_image):
        """Test filtering outliers with single method."""
        pipeline.load_image(sample_image)
        pipeline.detect_objects("test prompt", window_size=100, overlap=0)
        
        filtered = pipeline.filter_outliers(method="zscore")
        assert "zscore" in filtered
        assert len(filtered["zscore"]) <= len(pipeline.bounding_boxes)

    def test_filter_outliers_all_methods(self, pipeline, sample_image):
        """Test filtering outliers with all methods."""
        from langrs.processing.outlier_detection import (
            IQROutlierDetector,
            RobustCovarianceOutlierDetector,
        )
        
        pipeline.outlier_detectors = {
            "zscore": ZScoreOutlierDetector(),
            "iqr": IQROutlierDetector(),
            "robust": RobustCovarianceOutlierDetector(),
        }
        
        pipeline.load_image(sample_image)
        pipeline.detect_objects("test prompt", window_size=100, overlap=0)
        
        filtered = pipeline.filter_outliers()
        assert "zscore" in filtered
        assert "iqr" in filtered
        assert "robust" in filtered

    def test_segment_without_image(self, pipeline):
        """Test segment raises error if image not loaded."""
        with pytest.raises(SegmentationError, match="Image not loaded"):
            pipeline.segment()

    def test_segment_without_boxes(self, pipeline, sample_image):
        """Test segment raises error if no boxes."""
        pipeline.load_image(sample_image)
        with pytest.raises(SegmentationError, match="No bounding boxes"):
            pipeline.segment()

    def test_segment(self, pipeline, sample_image):
        """Test segmentation."""
        pipeline.load_image(sample_image)
        pipeline.detect_objects("test prompt", window_size=100, overlap=0)
        
        masks = pipeline.segment()
        assert masks is not None
        assert isinstance(masks, np.ndarray)

    def test_segment_with_custom_boxes(self, pipeline, sample_image):
        """Test segmentation with custom boxes."""
        pipeline.load_image(sample_image)
        custom_boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
        
        masks = pipeline.segment(boxes=custom_boxes)
        assert masks is not None

    def test_run_full_pipeline(self, pipeline, sample_image):
        """Test running full pipeline."""
        masks = pipeline.run_full_pipeline(sample_image, "test prompt", window_size=100, overlap=0)
        
        assert masks is not None
        assert pipeline.image_data is not None
        assert len(pipeline.bounding_boxes) > 0
        assert len(pipeline.filtered_boxes) > 0

    def test_calculate_areas(self, pipeline, sample_image):
        """Test area calculation."""
        pipeline.load_image(sample_image)
        pipeline.bounding_boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
        
        pipeline._calculate_areas()
        assert pipeline.areas is not None
        assert len(pipeline.areas) == 2

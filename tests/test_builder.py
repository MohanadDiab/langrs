"""Tests for pipeline builder."""

import pytest
import tempfile

from langrs.core.builder import LangRSBuilder
from langrs.core.config import LangRSConfig
from langrs.utils.exceptions import ModelLoadError


class TestLangRSBuilder:
    """Test LangRSBuilder class."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = LangRSBuilder()
        assert builder.detection_model_name == "grounding_dino"
        assert builder.segmentation_model_name == "sam"
        assert builder.output_path == "output"

    def test_with_config(self):
        """Test setting configuration."""
        builder = LangRSBuilder()
        config = LangRSConfig()
        config.detection.box_threshold = 0.5
        
        builder.with_config(config)
        assert builder.config is config
        assert builder.config.detection.box_threshold == 0.5

    def test_with_detection_model(self):
        """Test setting detection model."""
        builder = LangRSBuilder()
        builder.with_detection_model("grounding_dino", model_path="path.pth")
        
        assert builder.detection_model_name == "grounding_dino"
        assert builder.detection_model_path == "path.pth"

    def test_with_segmentation_model(self):
        """Test setting segmentation model."""
        builder = LangRSBuilder()
        builder.with_segmentation_model("sam", model_path="path.pth")
        
        assert builder.segmentation_model_name == "sam"
        assert builder.segmentation_model_path == "path.pth"

    def test_with_device(self):
        """Test setting device."""
        builder = LangRSBuilder()
        builder.with_device("cpu")
        
        assert builder.device == "cpu"

    def test_with_output_path(self):
        """Test setting output path."""
        builder = LangRSBuilder()
        builder.with_output_path("custom_output", create_timestamped=False)
        
        assert builder.output_path == "custom_output"
        assert builder.create_timestamped is False

    def test_fluent_interface(self):
        """Test fluent interface (method chaining)."""
        builder = (
            LangRSBuilder()
            .with_device("cpu")
            .with_output_path("test_output")
            .with_detection_model("grounding_dino")
        )
        
        assert builder.device == "cpu"
        assert builder.output_path == "test_output"
        assert builder.detection_model_name == "grounding_dino"

    def test_build_creates_pipeline(self):
        """Test building pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LangRSBuilder().with_output_path(tmpdir, create_timestamped=False)
            
            try:
                pipeline = builder.build()
                assert pipeline is not None
                assert pipeline.detection_model is not None
                assert pipeline.segmentation_model is not None
                assert pipeline.config is not None
            except (ImportError, ModelLoadError):
                pytest.skip("Models not available")

    def test_build_uses_default_config(self):
        """Test that build uses default config if not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LangRSBuilder().with_output_path(tmpdir, create_timestamped=False)
            
            try:
                pipeline = builder.build()
                assert pipeline.config is not None
                assert isinstance(pipeline.config, LangRSConfig)
            except (ImportError, ModelLoadError):
                pytest.skip("Models not available")

    def test_build_creates_all_components(self):
        """Test that build creates all required components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LangRSBuilder().with_output_path(tmpdir, create_timestamped=False)
            
            try:
                pipeline = builder.build()
                assert pipeline.image_loader is not None
                assert pipeline.tiling_strategy is not None
                assert len(pipeline.outlier_detectors) > 0
                assert pipeline.visualizer is not None
                assert pipeline.output_manager is not None
            except (ImportError, ModelLoadError):
                pytest.skip("Models not available")

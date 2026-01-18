"""Tests for configuration system."""

import pytest
import tempfile
import json
from pathlib import Path

from langrs.core.config import (
    LangRSConfig,
    DetectionConfig,
    SegmentationConfig,
    OutlierDetectionConfig,
    VisualizationConfig,
)
from langrs.utils.exceptions import ConfigurationError


class TestDetectionConfig:
    """Test DetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DetectionConfig()
        assert config.box_threshold == 0.3
        assert config.text_threshold == 0.3
        assert config.window_size == 500
        assert config.overlap == 200

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DetectionConfig(
            box_threshold=0.5, text_threshold=0.7, window_size=1000, overlap=300
        )
        assert config.box_threshold == 0.5
        assert config.text_threshold == 0.7
        assert config.window_size == 1000
        assert config.overlap == 300

    def test_box_threshold_validation_low(self):
        """Test box_threshold validation (too low)."""
        with pytest.raises(ConfigurationError, match="box_threshold must be between"):
            DetectionConfig(box_threshold=-0.1)

    def test_box_threshold_validation_high(self):
        """Test box_threshold validation (too high)."""
        with pytest.raises(ConfigurationError, match="box_threshold must be between"):
            DetectionConfig(box_threshold=1.5)

    def test_text_threshold_validation_low(self):
        """Test text_threshold validation (too low)."""
        with pytest.raises(ConfigurationError, match="text_threshold must be between"):
            DetectionConfig(text_threshold=-0.1)

    def test_text_threshold_validation_high(self):
        """Test text_threshold validation (too high)."""
        with pytest.raises(ConfigurationError, match="text_threshold must be between"):
            DetectionConfig(text_threshold=1.5)

    def test_window_size_validation(self):
        """Test window_size validation."""
        with pytest.raises(ConfigurationError, match="window_size must be positive"):
            DetectionConfig(window_size=0)

        with pytest.raises(ConfigurationError, match="window_size must be positive"):
            DetectionConfig(window_size=-10)

    def test_overlap_validation_negative(self):
        """Test overlap validation (negative)."""
        with pytest.raises(ConfigurationError, match="overlap must be non-negative"):
            DetectionConfig(overlap=-10)

    def test_overlap_validation_too_large(self):
        """Test overlap validation (too large)."""
        with pytest.raises(ConfigurationError, match="overlap.*must be less than window_size"):
            DetectionConfig(window_size=500, overlap=500)

        with pytest.raises(ConfigurationError, match="overlap.*must be less than window_size"):
            DetectionConfig(window_size=500, overlap=600)

    def test_valid_boundary_values(self):
        """Test valid boundary values."""
        # Minimum valid values
        config = DetectionConfig(box_threshold=0.0, text_threshold=0.0, window_size=1, overlap=0)
        assert config.box_threshold == 0.0
        assert config.text_threshold == 0.0
        assert config.window_size == 1
        assert config.overlap == 0

        # Maximum valid values
        config = DetectionConfig(box_threshold=1.0, text_threshold=1.0, window_size=1000, overlap=999)
        assert config.box_threshold == 1.0
        assert config.text_threshold == 1.0
        assert config.window_size == 1000
        assert config.overlap == 999


class TestSegmentationConfig:
    """Test SegmentationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SegmentationConfig()
        assert config.multimask_output is False
        assert config.window_size == 500
        assert config.overlap == 200

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SegmentationConfig(multimask_output=True, window_size=1000, overlap=300)
        assert config.multimask_output is True
        assert config.window_size == 1000
        assert config.overlap == 300

    def test_window_size_validation(self):
        """Test window_size validation."""
        with pytest.raises(ConfigurationError, match="window_size must be positive"):
            SegmentationConfig(window_size=0)

        with pytest.raises(ConfigurationError, match="window_size must be positive"):
            SegmentationConfig(window_size=-10)

    def test_overlap_validation_negative(self):
        """Test overlap validation (negative)."""
        with pytest.raises(ConfigurationError, match="overlap must be non-negative"):
            SegmentationConfig(overlap=-10)

    def test_overlap_validation_too_large(self):
        """Test overlap validation (too large)."""
        with pytest.raises(ConfigurationError, match="overlap.*must be less than window_size"):
            SegmentationConfig(window_size=500, overlap=500)

        with pytest.raises(ConfigurationError, match="overlap.*must be less than window_size"):
            SegmentationConfig(window_size=500, overlap=600)


class TestOutlierDetectionConfig:
    """Test OutlierDetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OutlierDetectionConfig()
        assert config.zscore_threshold == 3.0
        assert config.iqr_multiplier == 1.5
        assert config.svm_nu == 0.1
        assert config.isolation_forest_contamination == 0.25
        assert config.lof_contamination == 0.25
        assert config.lof_n_neighbors == 20
        assert config.robust_cov_contamination == 0.25

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OutlierDetectionConfig(
            zscore_threshold=2.5,
            iqr_multiplier=2.0,
            svm_nu=0.2,
            isolation_forest_contamination=0.3,
            lof_contamination=0.3,
            lof_n_neighbors=15,
            robust_cov_contamination=0.3,
        )
        assert config.zscore_threshold == 2.5
        assert config.iqr_multiplier == 2.0
        assert config.svm_nu == 0.2
        assert config.isolation_forest_contamination == 0.3
        assert config.lof_contamination == 0.3
        assert config.lof_n_neighbors == 15
        assert config.robust_cov_contamination == 0.3

    def test_zscore_threshold_validation(self):
        """Test zscore_threshold validation."""
        with pytest.raises(ConfigurationError, match="zscore_threshold must be positive"):
            OutlierDetectionConfig(zscore_threshold=0)

        with pytest.raises(ConfigurationError, match="zscore_threshold must be positive"):
            OutlierDetectionConfig(zscore_threshold=-1.0)

    def test_iqr_multiplier_validation(self):
        """Test iqr_multiplier validation."""
        with pytest.raises(ConfigurationError, match="iqr_multiplier must be positive"):
            OutlierDetectionConfig(iqr_multiplier=0)

        with pytest.raises(ConfigurationError, match="iqr_multiplier must be positive"):
            OutlierDetectionConfig(iqr_multiplier=-1.0)

    def test_svm_nu_validation_low(self):
        """Test svm_nu validation (too low)."""
        with pytest.raises(ConfigurationError, match="svm_nu must be between"):
            OutlierDetectionConfig(svm_nu=0.0)

    def test_svm_nu_validation_high(self):
        """Test svm_nu validation (too high)."""
        with pytest.raises(ConfigurationError, match="svm_nu must be between"):
            OutlierDetectionConfig(svm_nu=1.5)

    def test_isolation_forest_contamination_validation_low(self):
        """Test isolation_forest_contamination validation (too low)."""
        with pytest.raises(ConfigurationError, match="isolation_forest_contamination must be between"):
            OutlierDetectionConfig(isolation_forest_contamination=0.0)

    def test_isolation_forest_contamination_validation_high(self):
        """Test isolation_forest_contamination validation (too high)."""
        with pytest.raises(ConfigurationError, match="isolation_forest_contamination must be between"):
            OutlierDetectionConfig(isolation_forest_contamination=0.6)

    def test_lof_contamination_validation_low(self):
        """Test lof_contamination validation (too low)."""
        with pytest.raises(ConfigurationError, match="lof_contamination must be between"):
            OutlierDetectionConfig(lof_contamination=0.0)

    def test_lof_contamination_validation_high(self):
        """Test lof_contamination validation (too high)."""
        with pytest.raises(ConfigurationError, match="lof_contamination must be between"):
            OutlierDetectionConfig(lof_contamination=0.6)

    def test_lof_n_neighbors_validation(self):
        """Test lof_n_neighbors validation."""
        with pytest.raises(ConfigurationError, match="lof_n_neighbors must be positive"):
            OutlierDetectionConfig(lof_n_neighbors=0)

        with pytest.raises(ConfigurationError, match="lof_n_neighbors must be positive"):
            OutlierDetectionConfig(lof_n_neighbors=-1)

    def test_robust_cov_contamination_validation_low(self):
        """Test robust_cov_contamination validation (too low)."""
        with pytest.raises(ConfigurationError, match="robust_cov_contamination must be between"):
            OutlierDetectionConfig(robust_cov_contamination=0.0)

    def test_robust_cov_contamination_validation_high(self):
        """Test robust_cov_contamination validation (too high)."""
        with pytest.raises(ConfigurationError, match="robust_cov_contamination must be between"):
            OutlierDetectionConfig(robust_cov_contamination=0.6)


class TestVisualizationConfig:
    """Test VisualizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VisualizationConfig()
        assert config.figsize == (10, 10)
        assert config.box_color == "r"
        assert config.box_linewidth == 1
        assert config.mask_alpha == 0.4
        assert config.mask_colormap == "viridis"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VisualizationConfig(
            figsize=(15, 15),
            box_color="blue",
            box_linewidth=2,
            mask_alpha=0.6,
            mask_colormap="plasma",
        )
        assert config.figsize == (15, 15)
        assert config.box_color == "blue"
        assert config.box_linewidth == 2
        assert config.mask_alpha == 0.6
        assert config.mask_colormap == "plasma"

    def test_figsize_validation_wrong_length(self):
        """Test figsize validation (wrong length)."""
        with pytest.raises(ConfigurationError, match="figsize must be a tuple of length 2"):
            VisualizationConfig(figsize=(10,))

        with pytest.raises(ConfigurationError, match="figsize must be a tuple of length 2"):
            VisualizationConfig(figsize=(10, 10, 10))

    def test_figsize_validation_negative(self):
        """Test figsize validation (negative values)."""
        with pytest.raises(ConfigurationError, match="figsize values must be positive"):
            VisualizationConfig(figsize=(-10, 10))

        with pytest.raises(ConfigurationError, match="figsize values must be positive"):
            VisualizationConfig(figsize=(10, -10))

        with pytest.raises(ConfigurationError, match="figsize values must be positive"):
            VisualizationConfig(figsize=(0, 10))

    def test_box_linewidth_validation(self):
        """Test box_linewidth validation."""
        with pytest.raises(ConfigurationError, match="box_linewidth must be positive"):
            VisualizationConfig(box_linewidth=0)

        with pytest.raises(ConfigurationError, match="box_linewidth must be positive"):
            VisualizationConfig(box_linewidth=-1)

    def test_mask_alpha_validation_low(self):
        """Test mask_alpha validation (too low)."""
        with pytest.raises(ConfigurationError, match="mask_alpha must be between"):
            VisualizationConfig(mask_alpha=-0.1)

    def test_mask_alpha_validation_high(self):
        """Test mask_alpha validation (too high)."""
        with pytest.raises(ConfigurationError, match="mask_alpha must be between"):
            VisualizationConfig(mask_alpha=1.5)

    def test_valid_boundary_values(self):
        """Test valid boundary values."""
        config = VisualizationConfig(mask_alpha=0.0, box_linewidth=1)
        assert config.mask_alpha == 0.0

        config = VisualizationConfig(mask_alpha=1.0, box_linewidth=1)
        assert config.mask_alpha == 1.0


class TestLangRSConfig:
    """Test LangRSConfig main configuration class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = LangRSConfig()
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.segmentation, SegmentationConfig)
        assert isinstance(config.outlier_detection, OutlierDetectionConfig)
        assert isinstance(config.visualization, VisualizationConfig)

    def test_custom_config(self):
        """Test custom configuration creation."""
        detection = DetectionConfig(box_threshold=0.5)
        segmentation = SegmentationConfig(multimask_output=True)
        outlier = OutlierDetectionConfig(zscore_threshold=2.5)
        viz = VisualizationConfig(figsize=(15, 15))

        config = LangRSConfig(
            detection=detection,
            segmentation=segmentation,
            outlier_detection=outlier,
            visualization=viz,
        )

        assert config.detection.box_threshold == 0.5
        assert config.segmentation.multimask_output is True
        assert config.outlier_detection.zscore_threshold == 2.5
        assert config.visualization.figsize == (15, 15)

    def test_from_dict_empty(self):
        """Test creating config from empty dictionary."""
        config = LangRSConfig.from_dict({})
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.segmentation, SegmentationConfig)
        assert isinstance(config.outlier_detection, OutlierDetectionConfig)
        assert isinstance(config.visualization, VisualizationConfig)

    def test_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        config_dict = {
            "detection": {"box_threshold": 0.5},
            "segmentation": {"multimask_output": True},
        }
        config = LangRSConfig.from_dict(config_dict)
        assert config.detection.box_threshold == 0.5
        assert config.segmentation.multimask_output is True
        # Other values should be defaults
        assert config.detection.text_threshold == 0.3

    def test_from_dict_full(self):
        """Test creating config from full dictionary."""
        config_dict = {
            "detection": {
                "box_threshold": 0.5,
                "text_threshold": 0.7,
                "window_size": 1000,
                "overlap": 300,
            },
            "segmentation": {
                "multimask_output": True,
                "window_size": 800,
                "overlap": 200,
            },
            "outlier_detection": {
                "zscore_threshold": 2.5,
                "iqr_multiplier": 2.0,
            },
            "visualization": {
                "figsize": (15, 15),
                "box_color": "blue",
            },
        }
        config = LangRSConfig.from_dict(config_dict)
        assert config.detection.box_threshold == 0.5
        assert config.detection.text_threshold == 0.7
        assert config.segmentation.multimask_output is True
        assert config.outlier_detection.zscore_threshold == 2.5
        assert config.visualization.figsize == (15, 15)

    def test_from_dict_invalid(self):
        """Test creating config from invalid dictionary."""
        with pytest.raises(ConfigurationError):
            LangRSConfig.from_dict({"detection": {"box_threshold": -1}})

        with pytest.raises(ConfigurationError):
            LangRSConfig.from_dict({"detection": {"invalid_key": "value"}})

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
detection:
  box_threshold: 0.5
  text_threshold: 0.7
segmentation:
  multimask_output: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = LangRSConfig.from_yaml(yaml_path)
            assert config.detection.box_threshold == 0.5
            assert config.detection.text_threshold == 0.7
            assert config.segmentation.multimask_output is True
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_nonexistent_file(self):
        """Test loading config from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            LangRSConfig.from_yaml("nonexistent.yaml")

    def test_from_yaml_invalid_yaml(self):
        """Test loading config from invalid YAML file."""
        yaml_content = "invalid: yaml: content: ["
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                LangRSConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_empty_file(self):
        """Test loading config from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            config = LangRSConfig.from_yaml(yaml_path)
            # Should use defaults
            assert isinstance(config.detection, DetectionConfig)
        finally:
            Path(yaml_path).unlink()

    def test_from_json(self):
        """Test loading config from JSON file."""
        json_content = {
            "detection": {"box_threshold": 0.5, "text_threshold": 0.7},
            "segmentation": {"multimask_output": True},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            json_path = f.name

        try:
            config = LangRSConfig.from_json(json_path)
            assert config.detection.box_threshold == 0.5
            assert config.detection.text_threshold == 0.7
            assert config.segmentation.multimask_output is True
        finally:
            Path(json_path).unlink()

    def test_from_json_nonexistent_file(self):
        """Test loading config from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            LangRSConfig.from_json("nonexistent.json")

    def test_from_json_invalid_json(self):
        """Test loading config from invalid JSON file."""
        json_content = '{"invalid": json}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            json_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Invalid JSON"):
                LangRSConfig.from_json(json_path)
        finally:
            Path(json_path).unlink()

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = LangRSConfig()
        config_dict = config.to_dict()

        assert "detection" in config_dict
        assert "segmentation" in config_dict
        assert "outlier_detection" in config_dict
        assert "visualization" in config_dict

        assert config_dict["detection"]["box_threshold"] == 0.3
        assert config_dict["segmentation"]["multimask_output"] is False

    def test_to_yaml(self):
        """Test saving config to YAML file."""
        config = LangRSConfig()
        config.detection.box_threshold = 0.5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            config.to_yaml(yaml_path)
            # Verify it can be loaded back
            loaded_config = LangRSConfig.from_yaml(yaml_path)
            assert loaded_config.detection.box_threshold == 0.5
        finally:
            Path(yaml_path).unlink()

    def test_to_json(self):
        """Test saving config to JSON file."""
        config = LangRSConfig()
        config.detection.box_threshold = 0.5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            config.to_json(json_path)
            # Verify it can be loaded back
            loaded_config = LangRSConfig.from_json(json_path)
            assert loaded_config.detection.box_threshold == 0.5
        finally:
            Path(json_path).unlink()

    def test_round_trip_yaml(self):
        """Test round-trip: save to YAML and load back."""
        original = LangRSConfig()
        original.detection.box_threshold = 0.5
        original.segmentation.multimask_output = True
        original.outlier_detection.zscore_threshold = 2.5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            original.to_yaml(yaml_path)
            loaded = LangRSConfig.from_yaml(yaml_path)

            assert loaded.detection.box_threshold == original.detection.box_threshold
            assert loaded.segmentation.multimask_output == original.segmentation.multimask_output
            assert loaded.outlier_detection.zscore_threshold == original.outlier_detection.zscore_threshold
        finally:
            Path(yaml_path).unlink()

    def test_round_trip_json(self):
        """Test round-trip: save to JSON and load back."""
        original = LangRSConfig()
        original.detection.box_threshold = 0.5
        original.segmentation.multimask_output = True
        original.outlier_detection.zscore_threshold = 2.5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            original.to_json(json_path)
            loaded = LangRSConfig.from_json(json_path)

            assert loaded.detection.box_threshold == original.detection.box_threshold
            assert loaded.segmentation.multimask_output == original.segmentation.multimask_output
            assert loaded.outlier_detection.zscore_threshold == original.outlier_detection.zscore_threshold
        finally:
            Path(json_path).unlink()

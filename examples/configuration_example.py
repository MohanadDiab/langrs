"""
Configuration management example.

This example shows how to use configuration files
and programmatic configuration.
"""

from langrs import LangRSPipelineBuilder, LangRSConfig

# Option 1: Programmatic configuration
config = LangRSConfig()

# Detection settings
config.detection.box_threshold = 0.25
config.detection.text_threshold = 0.25
config.detection.window_size = 600
config.detection.overlap = 300

# Segmentation settings
config.segmentation.window_size = 600
config.segmentation.overlap = 300

# Outlier detection settings
config.outlier_detection.zscore_threshold = 2.5
config.outlier_detection.iqr_multiplier = 1.5
config.outlier_detection.lof_contamination = 0.25

# Visualization settings
config.visualization.figsize = (15, 15)
config.visualization.box_color = "blue"
config.visualization.mask_alpha = 0.5

# Build pipeline with config
pipeline = (
    LangRSPipelineBuilder()
    .with_config(config)
    .with_output_path("output")
    .build()
)

# Option 2: Load from YAML file
# config = LangRSConfig.from_yaml("config.yaml")
# pipeline = LangRSPipelineBuilder().with_config(config).build()

# Option 3: Load from JSON file
# config = LangRSConfig.from_json("config.json")
# pipeline = LangRSPipelineBuilder().with_config(config).build()

# Option 4: Save config to file
# config.to_yaml("my_config.yaml")
# config.to_json("my_config.json")

# Use pipeline
masks = pipeline.run_full_pipeline("data/image.tif", "roof")

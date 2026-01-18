"""
Advanced usage example for LangRS package.

This example demonstrates advanced features like custom configuration,
step-by-step pipeline execution, and custom model selection.
"""

from langrs import LangRS, LangRSPipelineBuilder, LangRSConfig

# Create custom configuration
config = LangRSConfig()
config.detection.box_threshold = 0.25
config.detection.text_threshold = 0.25
config.detection.window_size = 600
config.detection.overlap = 300
config.outlier_detection.zscore_threshold = 2.5
config.visualization.figsize = (15, 15)

# Create LangRS with custom settings
langrs = LangRS(
    output_path="output",
    detection_model="grounding_dino",
    segmentation_model="sam",
    device="cpu",  # or "cuda" for GPU
    config=config,
)

# Step 1: Load image
langrs.load_image("data/image.tif")

# Step 2: Detect objects
boxes = langrs.detect_objects("roof")
print(f"Detected {len(boxes)} bounding boxes")

# Step 3: Filter outliers
filtered = langrs.filter_outliers()
print(f"Available filtering methods: {list(filtered.keys())}")

# Use z-score filtered boxes
zscore_boxes = filtered["zscore"]
print(f"After z-score filtering: {len(zscore_boxes)} boxes")

# Step 4: Generate segmentation masks
masks = langrs.segment(boxes=zscore_boxes)
print(f"Generated masks with shape: {masks.shape}")

# Access pipeline state
print(f"Image loaded: {langrs.image_data is not None}")
print(f"Total boxes detected: {len(langrs.bounding_boxes)}")
print(f"Output directory: {langrs.output_manager.output_dir}")

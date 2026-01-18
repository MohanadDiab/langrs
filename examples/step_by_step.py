"""
Step-by-step pipeline example.

This example shows how to run the pipeline step by step
with full control over each stage.
"""

from langrs import LangRS
from langrs import apply_nms

# Create LangRS
langrs = LangRS(output_path="output")

# Load image
langrs.load_image("data/image.tif")
print("Image loaded")

# Detect objects with custom parameters
boxes = langrs.detect_objects(
    text_prompt="roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25,
)
print(f"Detected {len(boxes)} bounding boxes")

# Apply outlier detection - single method
filtered = langrs.filter_outliers(method="zscore")
zscore_boxes = filtered["zscore"]
print(f"After z-score filtering: {len(zscore_boxes)} boxes")

# Apply NMS to remove overlapping boxes
boxes_nms = apply_nms(zscore_boxes, iou_threshold=0.5)
print(f"After NMS: {len(boxes_nms)} boxes")

# Generate segmentation masks
masks = langrs.segment(boxes=boxes_nms.tolist())
print(f"Generated masks: {masks.shape}")

# Access results
print(f"\nLangRS State:")
print(f"  - Image data: {langrs.image_data is not None}")
print(f"  - Bounding boxes: {len(langrs.bounding_boxes)}")
print(f"  - Filtered boxes: {len(langrs.filtered_boxes)} methods")
print(f"  - Masks: {langrs.masks is not None}")
print(f"  - Output directory: {langrs.output_manager.output_dir}")

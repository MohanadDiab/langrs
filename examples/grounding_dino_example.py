"""
GroundingDINO detection example.

This example demonstrates how to use GroundingDINO for object detection
in remote sensing images using the LangRS package.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing langrs without installation
# This allows running the example directly without: pip install -e .
example_dir = Path(__file__).parent
project_root = example_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PIL import Image
from langrs import GroundingDINODetector, ModelFactory, OutputManager, MatplotlibVisualizer

# Example 1: Basic usage with default settings
print("=" * 60)
print("Example 1: Basic GroundingDINO Detection")
print("=" * 60)

# Create detector
detector = GroundingDINODetector(device="cpu")

# Load model weights (downloads from Hugging Face if not cached)
print("Loading model weights...")
detector.load_weights()
print("Model loaded successfully!")

# Load an image
image_path = "data/test.JPG"  # Update with your image path
try:
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.size}")
except FileNotFoundError:
    print(f"Image not found at {image_path}, creating a test image...")
    image = Image.new("RGB", (1024, 1024), color="blue")
    print("Created test image")

# Run detection
print("\nRunning detection with prompt: 'building, roof, house'...")
boxes = detector.detect(
    image=image,
    text_prompt="building, roof, house",
    box_threshold=0.25,
    text_threshold=0.25,
)

print(f"\nDetection results:")
print(f"  - Found {len(boxes)} bounding boxes")
if boxes:
    print(f"  - First box: {boxes[0]}")
    print(f"  - Box format: (x_min, y_min, x_max, y_max)")
    for i, box in enumerate(boxes[:5]):  # Show first 5 boxes
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        print(f"    Box {i+1}: ({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}) "
              f"[W: {width:.1f}, H: {height:.1f}]")
    
    # Save visualization
    output_path = output_manager.get_path_str("example1_detections.jpg")
    visualizer.plot_boxes(image, boxes, output_path, box_color="red", linewidth=2)
    print(f"\n  ✓ Saved visualization to: {output_path}")
else:
    print("  - No objects detected")

# Example 2: Using ModelFactory
print("\n" + "=" * 60)
print("Example 2: Using ModelFactory")
print("=" * 60)

detector2 = ModelFactory.create_detection_model(
    model_name="grounding_dino",
    device="cpu",
    model_variant="swint_ogc",  # Use SwinT variant (smaller, faster)
)

detector2.load_weights()
print("Model loaded via ModelFactory")

# Run detection with different thresholds
boxes2 = detector2.detect(
    image=image,
    text_prompt="object",
    box_threshold=0.3,  # Higher threshold = fewer but more confident detections
    text_threshold=0.3,
)

print(f"\nDetection results:")
print(f"  - Found {len(boxes2)} boxes with higher threshold")
if boxes2:
    # Save visualization
    output_path = output_manager.get_path_str("example2_detections.jpg")
    visualizer.plot_boxes(image, boxes2, output_path, box_color="blue", linewidth=2)
    print(f"  ✓ Saved visualization to: {output_path}")

# Example 3: Custom model path and variant
print("\n" + "=" * 60)
print("Example 3: Custom Configuration")
print("=" * 60)

# Use a different model variant
detector3 = GroundingDINODetector(
    device="cpu",
    model_variant="swinb_cogcoor",  # SwinB variant (larger, more accurate)
    cache_dir=None,  # Use default cache, or specify custom path
)

print("Available model variants:")
for variant in GroundingDINODetector.MODEL_VARIANTS.keys():
    print(f"  - {variant}")

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
print(f"\nAll output files saved to: {output_manager.output_dir}")
print("\nTips:")
print("  - Lower thresholds (0.2-0.25) = more detections, may include false positives")
print("  - Higher thresholds (0.3-0.4) = fewer but more confident detections")
print("  - Use 'swint_ogc' for faster inference, 'swinb_cogcoor' for better accuracy")
print("  - Multiple objects: use comma-separated prompts like 'building, roof, house'")

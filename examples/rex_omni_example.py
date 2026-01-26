"""
Rex-Omni detection example.

This example demonstrates how to use Rex-Omni for object detection
in remote sensing images using the LangRS package.

Rex-Omni is a 3B-parameter multimodal language model that performs
object detection as a next-token prediction problem.
"""

import os
from pathlib import Path
from PIL import Image
from langrs import RexOmniDetector, ModelFactory

# Set up cache directory outside home directory (to avoid "No space left on device" errors)
# Use environment variable if set, otherwise use a location with more space
cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE_DIR")
if not cache_dir:
    # Default to a location with more space (adjust path as needed)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # Create directory if it doesn't exist
    Path(cache_dir).parent.mkdir(parents=True, exist_ok=True)

print(f"Using cache directory: {cache_dir}")

# Determine device - use CPU if CUDA has compatibility issues (e.g., B200 GPU)
# Can override with DEVICE environment variable
device = os.environ.get("DEVICE", "cpu")  # Default to CPU for compatibility
print(f"Using device: {device}")

# Example 1: Basic usage with default settings
print("=" * 60)
print("Example 1: Basic Rex-Omni Detection")
print("=" * 60)

# Create detector
detector = RexOmniDetector(device=device, cache_dir=cache_dir)

# Load model weights (downloads from Hugging Face if not cached)
print("Loading model weights (this may take a few minutes on first run)...")
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
    box_threshold=0.3,  # Note: Rex-Omni doesn't use thresholds, but parameter is kept for API compatibility
    text_threshold=0.3,
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
else:
    print("  - No objects detected")

# Example 2: Using ModelFactory
print("\n" + "=" * 60)
print("Example 2: Using ModelFactory")
print("=" * 60)

detector2 = ModelFactory.create_detection_model(
    model_name="rex_omni",
    device=device,
    cache_dir=cache_dir,
    backend="transformers",  # Use transformers backend (default)
)

detector2.load_weights()
print("Model loaded via ModelFactory")

# Run detection with different prompt
boxes2 = detector2.detect(
    image=image,
    text_prompt="object",
    box_threshold=0.3,
    text_threshold=0.3,
)

print(f"Found {len(boxes2)} boxes with prompt 'object'")

# Example 3: Custom configuration with generation parameters
print("\n" + "=" * 60)
print("Example 3: Custom Configuration")
print("=" * 60)

# Create detector with custom generation parameters
# These control detection quality (Rex-Omni doesn't use confidence thresholds)
detector3 = RexOmniDetector(
    device=device,
    cache_dir=cache_dir,
    backend="transformers",
    temperature=0.0,  # Lower = more deterministic (default: 0.0)
    top_p=0.05,  # Lower = more focused sampling (default: 0.05)
    top_k=1,  # Lower = more conservative (default: 1)
    max_tokens=2048,  # Maximum tokens to generate (default: 2048)
)

print("Available model variants:")
for variant in RexOmniDetector.MODEL_VARIANTS.keys():
    print(f"  - {variant}")

print("\nGeneration parameters:")
print(f"  - Temperature: {detector3.temperature} (lower = more deterministic)")
print(f"  - Top-p: {detector3.top_p} (lower = more focused)")
print(f"  - Top-k: {detector3.top_k} (lower = more conservative)")
print(f"  - Max tokens: {detector3.max_tokens}")

# Example 4: Handling FlashAttention fallback
print("\n" + "=" * 60)
print("Example 4: Attention Implementation")
print("=" * 60)

# Rex-Omni automatically falls back to SDPA if FlashAttention is not available
# You can explicitly specify the attention implementation:
try:
    detector4 = RexOmniDetector(
        device=device,
        cache_dir=cache_dir,
        attn_implementation="sdpa",  # Explicitly use SDPA (Scaled Dot Product Attention)
    )
    print("Detector created with SDPA attention")
    print("Note: FlashAttention requires special installation and may not work on all systems")
except Exception as e:
    print(f"Note: {e}")

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
print("\nKey differences from GroundingDINO:")
print("  - Rex-Omni doesn't use confidence thresholds (box_threshold/text_threshold are ignored)")
print("  - Detection quality is controlled by generation parameters (temperature, top_p, top_k)")
print("  - Lower temperature (0.0) = more deterministic, consistent results")
print("  - Lower top_p (0.05) = more focused, precise detections")
print("  - Lower top_k (1) = more conservative, fewer false positives")
print("  - Multiple objects: use comma-separated prompts like 'building, roof, house'")
print("\nTips:")
print("  - For best results, use specific, descriptive prompts")
print("  - Rex-Omni works well with natural language descriptions")
print("  - If FlashAttention errors occur, the model automatically falls back to SDPA")
print("  - For GPU compatibility issues (e.g., B200), use device='cpu'")

"""Quick test script to verify Rex-Omni integration works with example image."""

from PIL import Image
from langrs import RexOmniDetector, ModelFactory

# Test 1: Direct instantiation
print("=" * 60)
print("Test 1: Direct RexOmniDetector instantiation")
print("=" * 60)

try:
    detector = RexOmniDetector(device="cpu")
    print("[OK] Detector created successfully")
    print(f"  - Device: {detector.device}")
    print(f"  - Backend: {detector.backend}")
    print(f"  - Model variant: {detector.model_variant}")
    print(f"  - Is loaded: {detector.is_loaded}")
except Exception as e:
    print(f"[ERROR] Failed to create detector: {e}")
    exit(1)

# Test 2: Model Factory
print("\n" + "=" * 60)
print("Test 2: ModelFactory creation")
print("=" * 60)

try:
    detector2 = ModelFactory.create_detection_model(
        model_name="rex_omni",
        device="cpu",
    )
    print("[OK] Detector created via ModelFactory")
    assert isinstance(detector2, RexOmniDetector)
    print("[OK] Detector is correct type")
except Exception as e:
    print(f"[ERROR] Failed to create via ModelFactory: {e}")
    exit(1)

# Test 3: Load model (this will download from HF)
print("\n" + "=" * 60)
print("Test 3: Loading model from Hugging Face")
print("=" * 60)
print("This may take a few minutes to download the model...")

try:
    print("  Downloading model from Hugging Face (this may take several minutes)...")
    detector.load_weights()
    print("[OK] Model loaded successfully")
    print(f"  - Is loaded: {detector.is_loaded}")
    print(f"  - Device: {detector.device}")
except Exception as e:
    error_msg = str(e)
    if "flash_attn" in error_msg.lower() or "flash_attention" in error_msg.lower():
        print(f"[WARNING] FlashAttention not available, retrying with sdpa...")
        # Retry with explicit sdpa
        detector2 = RexOmniDetector(device="cpu", attn_implementation="sdpa")
        try:
            detector2.load_weights()
            detector = detector2  # Use the working detector
            print("[OK] Model loaded successfully with sdpa attention")
            print(f"  - Is loaded: {detector.is_loaded}")
            print(f"  - Device: {detector.device}")
        except Exception as e2:
            print(f"[ERROR] Failed to load model even with sdpa: {e2}")
            print("  Note: This requires internet connection and may take time")
            exit(1)
    else:
        print(f"[ERROR] Failed to load model: {e}")
        print("  Note: This requires internet connection and may take time")
        exit(1)

# Test 4: Detection on example image
print("\n" + "=" * 60)
print("Test 4: Detection on example image")
print("=" * 60)

try:
    # Load test image
    image_path = "data/test.JPG"
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"[OK] Loaded image: {image_path}")
        print(f"  - Size: {image.size}")
    except FileNotFoundError:
        # Create a simple test image if file doesn't exist
        print(f"  Image not found at {image_path}, creating test image...")
        image = Image.new("RGB", (512, 512), color="red")
        print(f"  - Created test image: {image.size}")
    
    # Run detection
    print("\n  Running detection with prompt: 'object'...")
    boxes = detector.detect(
        image=image,
        text_prompt="object",
        box_threshold=0.3,
        text_threshold=0.3,
    )
    
    print(f"[OK] Detection completed")
    print(f"  - Found {len(boxes)} bounding boxes")
    if boxes:
        print(f"  - First box: {boxes[0]}")
        print(f"  - Box format: (x_min, y_min, x_max, y_max)")
    else:
        print("  - No objects detected (this is normal for test images)")
    
except Exception as e:
    print(f"[ERROR] Detection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed! Rex-Omni integration is working correctly.")
print("=" * 60)

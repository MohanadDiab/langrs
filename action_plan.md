# Rex-Omni Integration Action Plan

## Overview
This document outlines the step-by-step plan to integrate Rex-Omni as a detection model into LangRS, following the same pattern used for GroundingDINO integration.

## Understanding Rex-Omni

### What is Rex-Omni?
- **Type**: 3B-parameter Multimodal Large Language Model (MLLM)
- **Architecture**: Based on Qwen2.5-VL
- **Approach**: Performs object detection as a next-token prediction problem
- **Model Sources**:
  - Full model: `IDEA-Research/Rex-Omni` (Hugging Face)
  - Quantized (AWQ): `IDEA-Research/Rex-Omni-AWQ` (Hugging Face)

### Key Dependencies
From `requirements.txt`:
- `transformers==4.51.3`
- `qwen_vl_utils==0.0.14`
- `accelerate==1.10.1`
- `flash-attn==2.7.4.post1` (optional but recommended)
- `torch>=2.0.0`
- `vllm==0.9.1` (optional, for faster inference)
- Standard: `numpy`, `Pillow`, `matplotlib`

### API Structure
**Main Class**: `RexOmniWrapper` (from `rex_omni/wrapper.py`)

**Key Methods**:
- `__init__(model_path, backend="transformers", max_tokens=4096, temperature=0.0, ...)`
- `inference(images, task="detection", categories=[...]) -> List[Dict]`

**Output Format**:
```python
{
    "success": True,
    "extracted_predictions": {
        "category1": [
            {"type": "box", "coords": [x0, y0, x1, y1]},
            ...
        ],
        "category2": [...]
    },
    "raw_output": "...",
    "image_size": (w, h),
    ...
}
```

**Supported Tasks**: `detection`, `pointing`, `visual_prompting`, `keypoint`, `ocr_box`, `ocr_polygon`, `gui_grounding`, `gui_pointing`

### Key Differences from GroundingDINO
1. **Backend**: Uses `transformers` or `vllm` instead of `groundingdino-py`
2. **Model Architecture**: Qwen2.5-VL based (not DINO-based)
3. **Output Parsing**: Requires parsing special tokens (`<|object_ref_start|>`, `<|box_start|>`, etc.)
4. **Coordinate System**: Uses normalized bins [0-999] that need conversion to absolute coordinates
5. **Multiple Tasks**: Supports more than just detection

## Integration Steps

### Phase 1: Copy and Adapt Core Code

#### Step 1.1: Copy Essential Files
Copy the following files from `Rex-Omni/rex_omni/` to `langrs/models/detection/`:
- `parser.py` - Output parsing utilities (convert bins to coordinates)
- `tasks.py` - Task definitions (we only need detection task)
- `utils.py` - Utility functions (we may need some parsing helpers)

**Files to adapt**:
- `wrapper.py` → Create `rex_omni.py` (adapted for DetectionModel interface)

#### Step 1.2: Create Rex-Omni Detection Model Class
Create `langrs/models/detection/rex_omni.py`:

**Key Requirements**:
1. Inherit from `DetectionModel` (from `langrs.models.base`)
2. Implement required methods:
   - `detect(image, text_prompt, box_threshold, text_threshold) -> List[BoundingBox]`
   - `load_weights(model_path) -> None`
   - `device` property
   - `is_loaded` property

**Adaptation Strategy**:
- Wrap `RexOmniWrapper` internally
- Convert `text_prompt` to `categories` list (split by comma or use as single category)
- Convert output format from `{category: [{type: "box", coords: [...]}]}` to `List[BoundingBox]`
- Handle model loading from Hugging Face (similar to GroundingDINO)
- Support both `transformers` and `vllm` backends (make `transformers` default)

**Key Implementation Details**:
```python
class RexOmniDetector(DetectionModel):
    def __init__(self, model_path=None, device=None, backend="transformers", ...):
        # Initialize RexOmniWrapper internally
        # Handle device mapping
        # Set up model loading
        
    def detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.3):
        # Convert text_prompt to categories
        # Call rex.inference(task="detection", categories=[...])
        # Extract boxes from extracted_predictions
        # Convert to List[BoundingBox] format
        # Apply thresholds if needed (Rex-Omni doesn't use thresholds the same way)
        
    def load_weights(self, model_path=None):
        # Download from Hugging Face if model_path is None
        # Use huggingface_hub.hf_hub_download similar to GroundingDINO
        # Initialize RexOmniWrapper with model_path
```

### Phase 2: Handle Dependencies

#### Step 2.1: Add Optional Dependencies
Add to `requirements.txt` (as optional dependencies):
```txt
# Rex-Omni dependencies (optional)
transformers>=4.51.0
qwen-vl-utils>=0.0.14
accelerate>=1.10.0
# flash-attn is optional and may require special installation
```

**Note**: We should handle ImportError gracefully (like GroundingDINO does) if dependencies are missing.

#### Step 2.2: Handle Optional Backend
- `vllm` should be completely optional (only for advanced users)
- Default to `transformers` backend
- Add try/except for vllm imports

### Phase 3: Model Registration

#### Step 3.1: Register Model
In `langrs/models/__init__.py`:
```python
try:
    from .detection.rex_omni import RexOmniDetector
    ModelRegistry.register_detection("rex_omni")(RexOmniDetector)
except ImportError:
    pass
```

#### Step 3.2: Export Model
In `langrs/models/detection/__init__.py`:
```python
from .grounding_dino import GroundingDINODetector
from .rex_omni import RexOmniDetector

__all__ = ["GroundingDINODetector", "RexOmniDetector"]
```

#### Step 3.3: Add to Main Package (Optional)
In `langrs/__init__.py`:
```python
try:
    from .models.detection.rex_omni import RexOmniDetector
except ImportError:
    RexOmniDetector = None

# Add to __all__
```

### Phase 4: Handle Model Download

#### Step 4.1: Model Variants
Similar to GroundingDINO, support model variants:
```python
MODEL_VARIANTS = {
    "default": {
        "repo_id": "IDEA-Research/Rex-Omni",
    },
    "awq": {
        "repo_id": "IDEA-Research/Rex-Omni-AWQ",
        "quantization": "awq",
    },
}
```

#### Step 4.2: Download Logic
- Use `huggingface_hub.hf_hub_download` for model files
- Rex-Omni models are typically full model directories (not single checkpoint files)
- May need to download entire model directory or use `snapshot_download`

### Phase 5: Coordinate Conversion

#### Step 5.1: Understand Coordinate System
- Rex-Omni outputs coordinates in normalized bins [0-999]
- Parser converts to absolute coordinates: `x = (x_bin / 999.0) * width`
- Output format: `{category: [{"type": "box", "coords": [x0, y0, x1, y1]}]}`

#### Step 5.2: Convert to LangRS Format
- Extract all boxes from all categories
- Convert to `List[BoundingBox]` where `BoundingBox = Tuple[float, float, float, float]`
- Format: `(x_min, y_min, x_max, y_max)`

### Phase 6: Threshold Handling

#### Step 6.1: Understand Thresholds
- Rex-Omni doesn't use `box_threshold` and `text_threshold` the same way as GroundingDINO
- Rex-Omni uses generation parameters: `temperature`, `top_p`, `top_k`
- We may need to map thresholds to generation parameters or ignore them

**Options**:
1. Ignore thresholds (Rex-Omni doesn't output confidence scores)
2. Map thresholds to generation parameters (e.g., lower temperature = stricter)
3. Post-process to filter boxes (but no confidence scores available)

**Recommendation**: Document that thresholds are not used for Rex-Omni, or use them to adjust generation parameters.

### Phase 7: Testing

#### Step 7.1: Create Test File
Create `tests/test_rex_omni.py`:
- Test model loading (with and without Hugging Face)
- Test detection with various prompts
- Test coordinate conversion
- Test error handling (missing dependencies, invalid inputs)
- Test device handling (CPU/GPU)

#### Step 7.2: Integration Tests
- Test with LangRS pipeline
- Test with ModelFactory
- Test with different backends (transformers only, vllm optional)

### Phase 8: Documentation

#### Step 8.1: Update CONTRIBUTING.md
Add example of Rex-Omni integration as reference.

#### Step 8.2: Update README.md
- Mention Rex-Omni as alternative detection model
- Show usage example

#### Step 8.3: Add Docstrings
- Comprehensive docstrings for RexOmniDetector
- Document differences from GroundingDINO
- Document optional dependencies

### Phase 9: Cleanup

#### Step 9.1: Remove Rex-Omni Repository
After successful integration:
- Delete `Rex-Omni/` directory
- Verify all necessary code is in `langrs/models/detection/`

#### Step 9.2: Update .gitignore
Ensure any temporary files or model caches are ignored.

## Implementation Details

### Key Code Locations

**Files to Create**:
1. `langrs/models/detection/rex_omni.py` - Main detector class
2. `langrs/models/detection/parser.py` - Copy and adapt from Rex-Omni
3. `langrs/models/detection/tasks.py` - Copy and adapt (minimal, only detection)
4. `tests/test_rex_omni.py` - Test suite

**Files to Modify**:
1. `langrs/models/__init__.py` - Register model
2. `langrs/models/detection/__init__.py` - Export model
3. `langrs/__init__.py` - Optional: add to main exports
4. `requirements.txt` - Add optional dependencies
5. `setup.py` - Add optional dependencies if needed

### Critical Implementation Notes

1. **Import Handling**: Use try/except for all Rex-Omni specific imports:
   ```python
   try:
       from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
       from qwen_vl_utils import process_vision_info, smart_resize
       REX_OMNI_AVAILABLE = True
   except ImportError:
       REX_OMNI_AVAILABLE = False
   ```

2. **Model Loading**: 
   - Use `huggingface_hub.snapshot_download` for full model directory
   - Or use model path directly if already downloaded
   - Handle both local paths and Hugging Face repo IDs

3. **Backend Selection**:
   - Default to `transformers` (more compatible)
   - Make `vllm` optional (requires special installation)
   - Document backend differences

4. **Text Prompt Handling**:
   - GroundingDINO: single text prompt
   - Rex-Omni: list of categories
   - Strategy: Split text_prompt by comma, or use as single category

5. **Output Conversion**:
   ```python
   # Rex-Omni output format
   predictions = {
       "roof": [{"type": "box", "coords": [x0, y0, x1, y1]}, ...],
       "building": [...]
   }
   
   # Convert to LangRS format
   boxes = []
   for category, detections in predictions.items():
       for det in detections:
           if det["type"] == "box":
               boxes.append(tuple(det["coords"]))  # (x0, y0, x1, y1)
   ```

6. **Error Handling**:
   - ModelLoadError for loading failures
   - DetectionError for inference failures
   - ImportError for missing dependencies (with helpful message)

## Testing Checklist

- [ ] Model loads from Hugging Face
- [ ] Model loads from local path
- [ ] Detection works with single category
- [ ] Detection works with multiple categories (comma-separated prompt)
- [ ] Coordinates are correctly converted
- [ ] Works with CPU
- [ ] Works with CUDA (if available)
- [ ] Handles missing dependencies gracefully
- [ ] Integrates with LangRS pipeline
- [ ] Works with ModelFactory
- [ ] Error messages are helpful

## Potential Challenges

1. **Dependency Conflicts**: 
   - `transformers` version may conflict with other packages
   - `flash-attn` requires special installation
   - Solution: Make dependencies optional, document clearly

2. **Model Size**:
   - Rex-Omni is a 3B model (larger than GroundingDINO)
   - May require more memory
   - Solution: Document memory requirements, support AWQ quantized version

3. **Inference Speed**:
   - LLM-based inference may be slower than GroundingDINO
   - Solution: Support vllm backend for faster inference (optional)

4. **Threshold Interpretation**:
   - No confidence scores in output
   - Solution: Document that thresholds don't apply, or use generation parameters

5. **Coordinate System**:
   - Need to ensure correct conversion from bins to absolute coordinates
   - Solution: Test thoroughly, compare with GroundingDINO outputs

## Success Criteria

1. ✅ Rex-Omni can be used as a drop-in replacement for GroundingDINO
2. ✅ All tests pass
3. ✅ Documentation is complete
4. ✅ No dependency on external Rex-Omni repository
5. ✅ Works with existing LangRS pipeline
6. ✅ Handles errors gracefully
7. ✅ Supports both CPU and GPU

## Next Steps

1. Start with Phase 1: Copy and adapt core code
2. Test basic functionality before proceeding
3. Iterate on implementation based on testing results
4. Complete all phases systematically
5. Final cleanup and documentation

---

**Note**: This integration follows the same pattern as GroundingDINO, ensuring consistency in the codebase and making it easier for users to switch between detection models.

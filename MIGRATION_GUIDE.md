# Migration Guide: Old API to New API

This guide helps you migrate from the old `LangRS` API to the new `LangRSPipeline` API.

---

## Overview of Changes

### Old API (Still Supported)
```python
from langrs import LangRS

langrs = LangRS(image="image.tif", prompt="roof", output_path="output")
boxes = langrs.generate_boxes(window_size=600, overlap=300)
filtered = langrs.outlier_rejection()
masks = langrs.generate_masks(boxes=filtered['zscore'])
```

### New API (Recommended)
```python
from langrs import create_pipeline

pipeline = create_pipeline(output_path="output")
pipeline.load_image("image.tif")
boxes = pipeline.detect_objects("roof", window_size=600, overlap=300)
filtered = pipeline.filter_outliers()
masks = pipeline.segment(boxes=filtered['zscore'])
```

---

## Step-by-Step Migration

### Step 1: Import Changes

**Old:**
```python
from langrs import LangRS
```

**New:**
```python
from langrs import create_pipeline
# OR
from langrs import LangRSPipelineBuilder
```

### Step 2: Initialization

**Old:**
```python
langrs = LangRS(
    image="path/to/image.tif",
    prompt="roof",
    output_path="output"
)
```

**New (Simple):**
```python
pipeline = create_pipeline(output_path="output")
pipeline.load_image("path/to/image.tif")
```

**New (Advanced):**
```python
from langrs import LangRSPipelineBuilder, LangRSConfig

config = LangRSConfig()
config.detection.box_threshold = 0.25

pipeline = (
    LangRSPipelineBuilder()
    .with_config(config)
    .with_output_path("output")
    .build()
)
pipeline.load_image("path/to/image.tif")
```

### Step 3: Object Detection

**Old:**
```python
boxes = langrs.generate_boxes(
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)
```

**New:**
```python
boxes = pipeline.detect_objects(
    text_prompt="roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)
```

**Key Changes:**
- Method renamed: `generate_boxes()` → `detect_objects()`
- Prompt is now a parameter: `text_prompt="roof"` (not in constructor)
- Returns same format: list of bounding boxes

### Step 4: Outlier Rejection

**Old:**
```python
filtered = langrs.outlier_rejection()
# Returns dict with all methods
zscore_boxes = filtered['zscore']
```

**New:**
```python
# Apply all methods
filtered = pipeline.filter_outliers()
zscore_boxes = filtered['zscore']

# OR apply single method
zscore_boxes = pipeline.filter_outliers(method='zscore')
```

**Key Changes:**
- Method renamed: `outlier_rejection()` → `filter_outliers()`
- Can specify single method: `method='zscore'`
- Returns same format: dict mapping method names to filtered boxes

### Step 5: Segmentation

**Old:**
```python
masks = langrs.generate_masks(boxes=zscore_boxes)
```

**New:**
```python
masks = pipeline.segment(boxes=zscore_boxes)
```

**Key Changes:**
- Method renamed: `generate_masks()` → `segment()`
- Parameter name unchanged: `boxes=`
- Returns same format: numpy array mask

### Step 6: Full Pipeline

**Old:**
```python
langrs = LangRS(image="image.tif", prompt="roof", output_path="output")
masks = langrs.predict(
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)
```

**New:**
```python
pipeline = create_pipeline(output_path="output")
masks = pipeline.run_full_pipeline(
    "image.tif",
    "roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)
```

---

## Complete Migration Example

### Old Code
```python
from langrs import LangRS
from langrs.common import apply_nms

# Initialize
langrs = LangRS(
    image="data/image.tif",
    prompt="roof",
    output_path="output"
)

# Detect boxes
boxes = langrs.generate_boxes(
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)

# Filter outliers
filtered = langrs.outlier_rejection()
zscore_boxes = filtered['zscore']

# Apply NMS
boxes_nms = apply_nms(zscore_boxes, iou_threshold=0.5)

# Generate masks
masks = langrs.generate_masks(boxes=boxes_nms)
```

### New Code
```python
from langrs import create_pipeline
from langrs import apply_nms

# Initialize
pipeline = create_pipeline(output_path="output")
pipeline.load_image("data/image.tif")

# Detect boxes
boxes = pipeline.detect_objects(
    "roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)

# Filter outliers
filtered = pipeline.filter_outliers()
zscore_boxes = filtered['zscore']

# Apply NMS (same as before)
boxes_nms = apply_nms(zscore_boxes, iou_threshold=0.5)

# Generate masks
masks = pipeline.segment(boxes=boxes_nms)
```

### New Code (One-liner)
```python
from langrs import create_pipeline

pipeline = create_pipeline(output_path="output")
masks = pipeline.run_full_pipeline(
    "data/image.tif",
    "roof",
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25
)
```

---

## Advanced Features

### Custom Configuration

**Old:**
```python
# No direct config support - had to modify after initialization
```

**New:**
```python
from langrs import LangRSPipelineBuilder, LangRSConfig

config = LangRSConfig()
config.detection.box_threshold = 0.25
config.detection.window_size = 600
config.outlier_detection.zscore_threshold = 2.5

pipeline = (
    LangRSPipelineBuilder()
    .with_config(config)
    .with_output_path("output")
    .build()
)
```

### Custom Models

**Old:**
```python
# Not supported - tied to samgeo models
```

**New:**
```python
from langrs import LangRSPipelineBuilder, ModelFactory

# Use different models
detection_model = ModelFactory.create_detection_model("grounding_dino")
segmentation_model = ModelFactory.create_segmentation_model("sam")

# Or use builder
pipeline = (
    LangRSPipelineBuilder()
    .with_detection_model("grounding_dino")
    .with_segmentation_model("sam")
    .with_output_path("output")
    .build()
)
```

### Device Selection

**Old:**
```python
# Not directly supported
```

**New:**
```python
pipeline = (
    LangRSPipelineBuilder()
    .with_device("cpu")  # or "cuda"
    .with_output_path("output")
    .build()
)
```

---

## Backward Compatibility

The old `LangRS` class is still available for backward compatibility:

```python
from langrs import LangRS  # Still works!

langrs = LangRS(image="image.tif", prompt="roof", output_path="output")
boxes = langrs.generate_boxes()
```

However, it's recommended to migrate to the new API for:
- Better extensibility
- More control
- Future features
- Performance improvements

---

## Common Issues and Solutions

### Issue: "Image not loaded" error

**Solution:**
```python
# Old: Image loaded in constructor
langrs = LangRS(image="image.tif", ...)

# New: Must load explicitly
pipeline = create_pipeline(...)
pipeline.load_image("image.tif")
```

### Issue: Method names changed

**Solution:**
- `generate_boxes()` → `detect_objects()`
- `outlier_rejection()` → `filter_outliers()`
- `generate_masks()` → `segment()`
- `predict()` → `run_full_pipeline()`

### Issue: Prompt parameter location

**Old:**
```python
langrs = LangRS(..., prompt="roof", ...)
langrs.generate_boxes()  # Uses prompt from constructor
```

**New:**
```python
pipeline = create_pipeline(...)
pipeline.detect_objects("roof")  # Prompt as parameter
```

---

## Benefits of New API

1. **Flexibility**: Can swap components easily
2. **Testability**: All components can be mocked
3. **Extensibility**: Easy to add new models/methods
4. **Configuration**: Centralized config management
5. **Type Safety**: Full type hints
6. **Documentation**: Better docstrings

---

## Need Help?

If you encounter issues during migration:
1. Check this guide
2. Review the examples in `examples/` directory
3. Check the API documentation
4. Open an issue on GitHub

---

## Timeline

- **Current Version**: Old API still supported
- **Next Version**: Old API will show deprecation warnings
- **Future Version**: Old API may be removed (TBD)

We recommend migrating as soon as possible to take advantage of new features and improvements.

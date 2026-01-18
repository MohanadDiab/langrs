"""
Custom model selection example.

This example shows how to use different models or
create custom model instances.
"""

from langrs import (
    LangRS,
    LangRSPipelineBuilder,
    ModelFactory,
    ImageLoader,
    SlidingWindowTiler,
    ZScoreOutlierDetector,
    MatplotlibVisualizer,
    OutputManager,
    LangRSConfig,
)

# Option 1: Simple usage (recommended)
langrs = LangRS(
    output_path="output",
    detection_model="grounding_dino",
    segmentation_model="sam",
    device="cpu",
)

# Option 2: Use builder for more control
langrs = (
    LangRSPipelineBuilder()
    .with_detection_model("grounding_dino")
    .with_segmentation_model("sam")
    .with_device("cpu")
    .with_output_path("output")
    .build()
)

# Option 3: Manual construction with custom models (full control)
detection_model = ModelFactory.create_detection_model(
    model_name="grounding_dino",
    device="cpu",
    # model_path="path/to/custom/checkpoint.pth",  # Optional
)

segmentation_model = ModelFactory.create_segmentation_model(
    model_name="sam",
    model_variant="vit_b",  # Use smaller model
    device="cpu",
)

config = LangRSConfig()
image_loader = ImageLoader()
tiling_strategy = SlidingWindowTiler()
outlier_detectors = {
    "zscore": ZScoreOutlierDetector(threshold=2.5),
    "iqr": ZScoreOutlierDetector(),  # Can add more
}
visualizer = MatplotlibVisualizer(figsize=(15, 15))
output_manager = OutputManager("output", create_timestamped=True)

custom_langrs = LangRS(
    _detection_model_instance=detection_model,
    _segmentation_model_instance=segmentation_model,
    _image_loader=image_loader,
    _tiling_strategy=tiling_strategy,
    _outlier_detectors=outlier_detectors,
    _visualizer=visualizer,
    _output_manager=output_manager,
    config=config,
)

# Use the LangRS instance
masks = custom_langrs.run_full_pipeline("data/image.tif", "roof")

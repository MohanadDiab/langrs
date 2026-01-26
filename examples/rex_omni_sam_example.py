"""
Rex-Omni + SAM segmentation example.

This example demonstrates using Rex-Omni for object detection
and SAM for segmentation on the roi_kala.tif image.

Rex-Omni is a 3B-parameter multimodal language model that performs
object detection as a next-token prediction problem.
SAM (Segment Anything Model) is used for generating segmentation masks.
"""

from langrs import LangRS

# Create LangRS with Rex-Omni for detection and SAM for segmentation
# Output will be saved to timestamped directory: output/YYYYMMDD_HHMMSS/
langrs = LangRS(
    output_path="output",
    detection_model="rex_omni",  # Use Rex-Omni for detection
    segmentation_model="sam",    # Use SAM for segmentation (default)
    device="cpu",  # Change to "cuda" if you have GPU support
)

print("=" * 60)
print("Rex-Omni + SAM Segmentation Pipeline")
print("=" * 60)
print(f"Image: data/roi_kala.tif")
print(f"Detection model: Rex-Omni")
print(f"Segmentation model: SAM")
print(f"Output directory: {langrs.output_manager.output_dir}")
print("=" * 60)

# Run the complete pipeline: load -> detect -> filter -> segment
print("\nRunning full pipeline...")
print("This may take several minutes depending on your hardware.")
print("Rex-Omni will download model weights on first run (~3GB).\n")

masks = langrs.run_full_pipeline(
    image_source="data/roi_kala.tif",
    text_prompt="building, roof, house",  # Text prompt for detection
    window_size=600,
    overlap=300,
    box_threshold=0.25,  # Note: Rex-Omni doesn't use thresholds, but kept for API compatibility
    text_threshold=0.25,
)

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"Segmentation masks shape: {masks.shape}")
print(f"Results saved to: {langrs.output_manager.output_dir}")
print("\nGenerated files:")
print("  - results_dino.jpg: Detection bounding boxes")
print("  - results_areas.jpg: Bounding box area distribution")
print("  - results_*.jpg: Outlier detection plots (for each method)")
print("  - results_*_filtered.jpg: Filtered bounding boxes")
print("  - results_sam.jpg: Segmentation masks overlay")
if hasattr(langrs, 'image_data') and langrs.image_data and langrs.image_data.is_georeferenced:
    print("  - bounding_boxes.shp: Georeferenced bounding boxes")
    print("  - masks.shp: Georeferenced segmentation masks")
print("=" * 60)

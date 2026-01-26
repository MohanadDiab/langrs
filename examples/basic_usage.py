"""
Basic usage example for LangRS package.

This example demonstrates the simplest way to use LangRS
for remote sensing image segmentation.
"""

from langrs import LangRS

# Create LangRS with default settings
langrs = LangRS(output_path="output")

# Run the complete pipeline
masks = langrs.run_full_pipeline(
    image_source="data/image.tif",  # Path to your image
    text_prompt="roof",  # What to detect/segment
    window_size=600,
    overlap=300,
    box_threshold=0.25,
    text_threshold=0.25,
)

print(f"Segmentation complete! Masks shape: {masks.shape}")
print(f"Results saved to: {langrs.output_manager.output_dir}")

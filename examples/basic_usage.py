import json
from langrs_old import LangRS

# Create a configuration dictionary
config = {
    "image_input": "path/to/your/image.tif",
    "text_input": "white cars",
    "tile_size": 1000,
    "overlap": 300,
    "tiling": False,
    "evaluation": False,
    "outlier_methods": ["isolation_forest"],
    "output_dir": "output"
}

# Save the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

# Initialize LangRS with the configuration file
lang_rs = LangRS("config.json")

# Process the image
lang_rs.process()

print("Processing complete. Results saved in the 'output' directory.")
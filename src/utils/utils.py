from rasterio.enums import Resampling
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
import numpy as np


def load_tif_image_as_pil(image: str):
    # Load the image a numpy array for RGB bands only
    with rasterio.open(image) as src:
        # Read the RGB bands (1, 2, 3)
        rgb_image = np.array(src.read([1, 2, 3]))

    pil_image = Image.fromarray(np.transpose(rgb_image, (1, 2, 0)))
    
    return pil_image


def visualize_raster(image_input):
  with rasterio.open(image_input) as src:
    # Read the image
    tiff = src.read()

    # Plot the image
    plt.figure(figsize=(10, 10))
    show(tiff, transform=src.transform)


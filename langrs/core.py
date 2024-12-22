import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import json
import datetime
from samgeo import tms_to_geotiff, split_raster
from samgeo.text_sam import LangSAM
from samgeo import SamGeo
import rasterio
from rasterio.plot import show
import leafmap
from rasterio.enums import Resampling
from PIL import Image

class LangRS():
    def __init__(self, image, prompt, output_path):
        self.image_path = image
        self.prompt = prompt
        # make output path dynamic with timestamps
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(output_path, timestamp)
        os.makedirs(self.output_path, exist_ok=True)
        self.output_path_image = os.path.join(self.output_path, 'original_image.jpg')
        self.output_path_image_boxes = os.path.join(self.output_path, 'results_dino.jpg')
        self.output_path_image_masks = os.path.join(self.output_path, 'results_sam.jpg')
        self.output_path_image_areas = os.path.join(self.output_path, 'results_areas.jpg')
        self.sam = LangSAM()

        # Load the image a numpy array for RGB bands only
        with rasterio.open(self.image_path) as src:
            # Read the RGB bands (1, 2, 3)
            rgb_image = np.array(src.read([1, 2, 3]))

        self.pil_image = Image.fromarray(np.transpose(rgb_image, (1, 2, 0)))

    def predict(self,):
        pass

    def predict_dino(self, window_size, overlap, text_prompt = None):
        if text_prompt is None:
            text_prompt = self.prompt

        self.bounding_boxes = self._run_detection_on_chunks(chunk_size=window_size,
                                                       image=self.pil_image,
                                                       overlap=overlap,
                                                       text_prompt=text_prompt)
        

        # Plotting the bounding boxes directly on the array converted to an image
        fig, ax = plt.subplots(figsize=(10, 10))
        # Directly display the array as an image
        ax.imshow(self.pil_image, cmap='gray')

        # Plotting the bounding boxes on the directly converted image
        for bbox in self.bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        plt.savefig(self.output_path_image_boxes)
    
        return self.bounding_boxes
        

    def predict_sam(self,):
        pass

    def outlier_rejection(self,):
        pass

    def evaluate(self,):
        pass

    def _area_calculator(self, bounding_boxes=None):
        if bounding_boxes == None:
            bounding_boxes = self.bounding_boxes

        self.areas =  [(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in bounding_boxes]
        self.bboxes_with_areas = sorted(zip(bounding_boxes, self.areas), key=lambda x: x[1])
        self.sorted_bboxes = [bbox for bbox, area in self.bboxes_with_areas]
        self.sorted_areas =  [area for bbox, area in self.bboxes_with_areas]

        self.bboxes = self.sorted_bboxes
        self.data = np.array(self.bboxes)
        self.data = self.data.reshape(-1, 1)
        # Visualize and save the dataset
        plt.scatter(range(len(self.data)), self.data)
        plt.xlabel('Index')
        plt.ylabel('Area (px sq.)')
        plt.grid(True)
        plt.savefig(self.output_path_image_areas)
        plt.close()


    def _run_detection_on_chunks(self, image, chunk_size=1000, overlap=300, box_threshold=0.3, text_threshold=0.75, text_prompt=""):
        # Slice image into chunks with overlap
        chunks = self._slice_image_with_overlap(image, chunk_size, overlap)
        all_bounding_boxes = []
        # Loop through chunks and run detection
        for chunk, offset_x, offset_y in chunks:
            results = self.sam.predict_dino(image=chunk, box_threshold=box_threshold, text_threshold=text_threshold, text_prompt=text_prompt)
            # Localize bounding boxes to the original image coordinates
            localized_boxes = self._localize_bounding_boxes(results[0], offset_x, offset_y)
            all_bounding_boxes.extend(localized_boxes)
        return all_bounding_boxes

    def _slice_image_with_overlap(self, image, chunk_size=300, overlap=100):
        # Get image dimensions
        width, height = image.size
        # Create list of image chunks
        chunks = []
        for i in range(0, height, chunk_size - overlap):
            for j in range(0, width, chunk_size - overlap):
                left = j
                upper = i
                right = min(j + chunk_size, width)
                lower = min(i + chunk_size, height)
                chunk = image.crop((left, upper, right, lower))
                chunks.append((chunk, left, upper))
        return chunks
    
    def _localize_bounding_boxes(self, bounding_boxes, offset_x, offset_y):
        localized_boxes = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            localized_boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y))
        return localized_boxes

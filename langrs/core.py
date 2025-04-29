import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime
from samgeo.text_sam import LangSAM
from samgeo.common import *
from PIL import Image
from .outlier_detection import *
from .common import *
import torch

class LangRS(LangSAM):
    """
    A class for performing remote sensing image segmentation, bounding box detection,
    outlier rejection, and area calculations using LangSAM.
    """

    def __init__(self, image, prompt, output_path):
        """
        Initialize the LangRS class with the input image, text prompt, and output path.

        Args:
            image (str | np.ndarray | PIL.Image.Image): Input image, provided as a file path (GeoTIFF, PNG, JPG), 
                                                        a NumPy array (H, W, C), or a PIL Image object.
            prompt (str): Text prompt to guide the segmentation process.
            output_path (str): Directory path to save the results.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
            RuntimeError: If initialization fails due to an unexpected error.
        """

        super().__init__()
        
        try:
            os.makedirs(output_path, exist_ok=True)
            self.prompt = prompt

            # Create a dynamic output path with a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(output_path, timestamp)
            os.makedirs(self.output_path, exist_ok=True)

            # Define output file paths
            self.output_path_image = os.path.join(self.output_path, 'original_image.jpg')
            self.output_path_image_boxes = os.path.join(self.output_path, 'results_dino.jpg')
            self.output_path_image_masks = os.path.join(self.output_path, 'results_sam.jpg')
            self.output_path_image_areas = os.path.join(self.output_path, 'results_areas.jpg')

            # Load image and extract metadata
            self.image_path, self.pil_image, self.np_image, self.source_crs = load_image(image)

            # isgeo is a flag to check if the input image is georeferenced
            self.isgeo = True if self.source_crs is not None else False
 

        except Exception as e:
            raise RuntimeError(f"Error initializing LangRS: {e}")

    def predict(self, boxes=None, window_size=500, overlap=200, box_threshold=0.3, text_threshold=0.3, text_prompt=None):
        """
        Run the full prediction pipeline, including box generation, outlier rejection, 
        and mask generation.

        Args:
            boxes (list[torch.Tensor]): List of bounding boxes to run inference on.

        Returns:
            np.ndarray: Segmentation mask overlay.
        """

        self.generate_boxes(window_size=window_size, overlap=overlap, box_threshold=box_threshold, text_threshold=text_threshold, text_prompt=text_prompt)
        self.outlier_rejection()
        return self.generate_masks(boxes=boxes)

    def generate_boxes(self, window_size=500, overlap=200, box_threshold=0.5, text_threshold=0.5, text_prompt=None):
        """
        Detect bounding boxes using a sliding window approach with LangSAM.

        Args:
            window_size (int, optional): Size of the window for detection. Default is 500.
            overlap (int, optional): Overlap between windows to improve detection continuity. Default is 200.
            box_threshold (float, optional): Confidence threshold for detecting boxes. Default is 0.5.
            text_threshold (float, optional): Confidence threshold for text-based detection. Default is 0.5.
            text_prompt (str, optional): Custom text prompt for object detection. If None, uses the class prompt.

        Returns:
            list: Detected bounding boxes in the format [(x_min, y_min, x_max, y_max), ...].

        Raises:
            RuntimeError: If an error occurs during box generation.
        """

        try:
            if text_prompt is None:
                text_prompt = self.prompt

            self.bounding_boxes = self._run_hyperinference(
                image=self.pil_image,
                text_prompt=text_prompt,
                chunk_size=window_size,
                overlap=overlap,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            # Plot bounding boxes
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.pil_image)

            for bbox in self.bounding_boxes:
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            ax.axis('off')
            plt.savefig(self.output_path_image_boxes, bbox_inches='tight', pad_inches=0)
            plt.close()

            self._area_calculator()

            self.outlier_rejection()

            if self.isgeo:
                gdf_boxes = convert_bounding_boxes_to_geospatial(
                            bounding_boxes=self.bounding_boxes,
                            image_path=self.image_path,
                            )
                
                gdf_boxes.to_file(os.path.join(self.output_path, 'bounding_boxes.shp'))

            return self.bounding_boxes

        except Exception as e:
            raise RuntimeError(f"Error in generate_boxes: {e}")

    #TODO: Mohanad: Update readme.md for the new changes of the window_size and overlap
    #TODO: Mohanad: no_grad + quantization for both models
    def generate_masks(self, boxes: list = None, window_size=500, overlap=200):
        """
        Generate segmentation masks for detected objects using the SAM model in a tiled approach.

        Args:
            boxes (list[torch.Tensor], optional): List of bounding boxes to run inference on.
            window_size (int, optional): Window size for tiling. Default is 500.
            overlap (int, optional): Overlap between tiles. Default is 200.

        Returns:
            np.ndarray: Overlay mask representing segmented areas.
        """

        output_path = self.output_path_image_masks

        if boxes is not None:
            self.prediction_boxes = boxes
        else:
            self.prediction_boxes = self.bounding_boxes

        try:
            self.boxes_tensor = torch.tensor(np.array(self.prediction_boxes))
            
            # Prepare for windowed prediction
            windows = self._split_into_windows(self.np_image, window_size, overlap)
            masks_info = []

            for (x_min, y_min, x_max, y_max) in windows:
                tile_np = self.np_image[y_min:y_max, x_min:x_max]
                tile_pil = Image.fromarray(tile_np[:, :, :3])

                # Select boxes that fall inside this window
                local_boxes = []
                for box in self.prediction_boxes:
                    bx_min, by_min, bx_max, by_max = box
                    if (bx_min < x_max and bx_max > x_min) and (by_min < y_max and by_max > y_min):
                        # Adjust box coordinates relative to tile
                        adj_box = [
                            max(0, bx_min - x_min),
                            max(0, by_min - y_min),
                            min(x_max - x_min, bx_max - x_min),
                            min(y_max - y_min, by_max - y_min),
                        ]
                        local_boxes.append(adj_box)

                if local_boxes:
                    local_boxes_tensor = torch.tensor(local_boxes)
                    masks = self.predict_sam_tile(tile_pil, local_boxes_tensor)

                    for i in range(len(masks)):
                        masks_info.append((masks[i], (x_min, y_min, x_max, y_max)))

            if not masks_info:
                raise RuntimeError("No masks were generated.")

            self.mask_overlay = self._merge_masks_from_windows(
                masks_info, (self.np_image.shape[0], self.np_image.shape[1])
            )

            fig, ax = plt.subplots()
            ax.imshow(self.np_image)
            ax.imshow(self.mask_overlay, cmap="viridis", alpha=0.4)
            ax.axis('off')  
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if self.isgeo:
                gdf_boxes = convert_bounding_boxes_to_geospatial(
                    bounding_boxes=self.prediction_boxes,
                    image_path=self.image_path,
                )
                gdf_boxes.to_file(os.path.join(self.output_path, 'bounding_boxes_filtered.shp'))

                gdf_masks = convert_masks_to_geospatial(
                    masks=self.mask_overlay,
                    image_path=self.image_path,
                )
                gdf_masks.to_file(os.path.join(self.output_path, 'masks.shp'))

            return self.mask_overlay

        except Exception as e:
            raise RuntimeError(f"Error in generate_masks: {e}")

    def outlier_rejection(self, method=None, filter=False):
        """
        Perform outlier detection on detected bounding boxes using multiple statistical and ML methods.

        Returns:
            dict: Dictionary mapping each rejection method name to the corresponding filtered list of bounding boxes.

        Raises:
            RuntimeError: If an error occurs during outlier rejection.
        """

        try:
            self.output_path_image_zscore = os.path.join(self.output_path, 'results_zscore.jpg')
            self.output_path_image_iqr = os.path.join(self.output_path, 'results_iqr.jpg')
            self.output_path_image_lof = os.path.join(self.output_path, 'results_lof.jpg')
            self.output_path_image_iso = os.path.join(self.output_path, 'results_iso.jpg')
            self.output_path_image_svm = os.path.join(self.output_path, 'results_svm.jpg')
            self.output_path_image_svm_sgd = os.path.join(self.output_path, 'results_svm_sgd.jpg')
            self.output_path_image_rob = os.path.join(self.output_path, 'results_rob.jpg')

            self.y_pred_zscore = z_score_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_zscore)
            self.y_pred_iqr = iqr_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_iqr)
            self.y_pred_svm = svm_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_svm)
            self.y_pred_svm_sgd = svm_sgd_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_svm_sgd)
            self.y_pred_rob = rob_cov(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_rob)
            self.y_pred_lof = lof_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_lof)
            self.y_pred_iso = isolation_forest_outliers(self.data, self.pil_image, self.bboxes, output_dir=self.output_path_image_iso)

            self.rejection_methods = {
                "zscore": self.y_pred_zscore,
                "iqr": self.y_pred_iqr,
                "svm": self.y_pred_svm,
                "svm_sgd": self.y_pred_svm_sgd,
                "robust_covariance": self.y_pred_rob,
                "lof": self.y_pred_lof,
                "isolation_forest": self.y_pred_iso
            }

            if filter:
                for method, boxes in self.rejection_methods.copy().items():
                    self.rejection_methods[method] = apply_nms(boxes)

            if method is None:
              return self.rejection_methods
            else:
                return self.rejection_methods[method]
            
        except Exception as e:
            raise RuntimeError(f"Error in outlier_rejection: {e}")

    def _area_calculator(self, bounding_boxes=None):
        """
        Calculate areas of bounding boxes and sort them by size.

        Args:
            bounding_boxes (list, optional): List of bounding boxes to process. 
                                            Defaults to the detected boxes if None.

        Raises:
            RuntimeError: If area calculation fails due to an unexpected error.
        """

        try:
            if bounding_boxes is None:
                bounding_boxes = self.bounding_boxes

            self.areas = [(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in bounding_boxes]
            self.bboxes_with_areas = sorted(zip(bounding_boxes, self.areas), key=lambda x: x[1])
            self.sorted_bboxes = [bbox for bbox, area in self.bboxes_with_areas]
            self.sorted_areas = [area for bbox, area in self.bboxes_with_areas]

            self.bboxes = self.sorted_bboxes
            self.data = np.array(self.sorted_areas).reshape(-1, 1)

            plt.figure()
            plt.scatter(range(len(self.data)), self.data)
            plt.xlabel('Index')
            plt.ylabel('Area (px sq.)')
            plt.grid(True)
            plt.savefig(self.output_path_image_areas)
            plt.close()

        except Exception as e:
            raise RuntimeError(f"Error in _area_calculator: {e}")

    def _run_hyperinference(self, image, chunk_size=1000, overlap=300, box_threshold=0.3, text_threshold=0.75, text_prompt=""):
        """
        Run object detection on the image using a sliding window approach with overlap.

        Args:
            image (PIL.Image.Image): Input image for object detection.
            chunk_size (int, optional): Size of the window for each chunk. Default is 1000.
            overlap (int, optional): Overlap between adjacent chunks. Default is 300.
            box_threshold (float, optional): Confidence threshold for boxes. Default is 0.3.
            text_threshold (float, optional): Confidence threshold for text detection. Default is 0.75.
            text_prompt (str, optional): Text prompt for object detection. Default is empty string.

        Returns:
            list: List of detected bounding boxes localized to the original image coordinates.

        Raises:
            RuntimeError: If the detection process fails.
        """

        try:
            chunks = self._slice_image_with_overlap(image, chunk_size, overlap)
            all_bounding_boxes = []

            for chunk, offset_x, offset_y in chunks:
                results = self.predict_dino(
                    image=chunk,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    text_prompt=text_prompt
                )
                localized_boxes = self._localize_bounding_boxes(results[0], offset_x, offset_y)
                all_bounding_boxes.extend(localized_boxes)

            return all_bounding_boxes

        except Exception as e:
            raise RuntimeError(f"Error in hyperinference: {e}")

    def _slice_image_with_overlap(self, image, chunk_size=300, overlap=100):
        """
        Slice an image into overlapping chunks for processing.

        Args:
            image (PIL.Image.Image): Input image to slice.
            chunk_size (int, optional): Size of each chunk. Default is 300.
            overlap (int, optional): Overlap size between adjacent chunks. Default is 100.

        Returns:
            list: List of tuples, where each tuple contains a cropped image chunk and its 
                (left, upper) offset in the original image.

        Raises:
            RuntimeError: If slicing fails due to an unexpected error.
        """

        try:
            width, height = image.size
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

        except Exception as e:
            raise RuntimeError(f"Error in _slice_image_with_overlap: {e}")

    def _localize_bounding_boxes(self, bounding_boxes, offset_x, offset_y):
        """
        Convert bounding boxes from chunk coordinates to original image coordinates.

        Args:
            bounding_boxes (list): List of bounding boxes detected in a chunk.
            offset_x (int): Horizontal offset of the chunk in the original image.
            offset_y (int): Vertical offset of the chunk in the original image.

        Returns:
            list: List of bounding boxes adjusted to the full image coordinates.

        Raises:
            RuntimeError: If localization fails due to an unexpected error.
        """

        try:
            localized_boxes = []

            for box in bounding_boxes:
                x1, y1, x2, y2 = box
                localized_boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y))

            return localized_boxes

        except Exception as e:
            raise RuntimeError(f"Error in _localize_bounding_boxes: {e}")

    def _split_into_windows(self, image_np, window_size, overlap):
        """
        Splits the image into overlapping windows.

        Args:
            image_np (np.array): Full image as numpy array.
            window_size (int): Size of each window (square assumed).
            overlap (int): Number of pixels overlap between windows.

        Returns:
            List of tuples: (x_min, y_min, x_max, y_max) for each window.
        """
        h, w, _ = image_np.shape
        windows = []
        step = window_size - overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                x_min = x
                y_min = y
                x_max = min(x + window_size, w)
                y_max = min(y + window_size, h)
                windows.append((x_min, y_min, x_max, y_max))

        return windows
    
    def _merge_masks_from_windows(self, masks_info, full_size):
        """
        Merge the masks from windows into a full-size mask.

        Args:
            masks_info (list): List of tuples (mask, (x_min, y_min, x_max, y_max)).
            full_size (tuple): Full image size (height, width).

        Returns:
            np.ndarray: Full merged mask.
        """
        merged_mask = np.zeros(full_size, dtype=np.uint8)

        for idx, (mask, (x_min, y_min, x_max, y_max)) in enumerate(masks_info):
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            mask = mask.squeeze()
            mask_resized = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            mask_resized[mask > 0] = 255  # Binary mask
            
            merged_mask[y_min:y_max, x_min:x_max] = np.maximum(
                merged_mask[y_min:y_max, x_min:x_max], mask_resized
            )

        return merged_mask

    def predict_sam_tile(self, image_tile, boxes):
        """
        Run the SAM model prediction on a tile.

        Args:
            image_tile (PIL.Image): Image tile.
            boxes (torch.Tensor): Bounding boxes relative to the tile.

        Returns:
            Masks tensor.
        """
        image_array = np.asarray(image_tile)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes, image_array.shape[:2]
        )
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

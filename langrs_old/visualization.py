import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_raster(image_input: Union[str, np.ndarray], save_path: str = None):
    """
    Visualize a raster image.

    Args:
        image_input (Union[str, np.ndarray]): Path to the image file or numpy array of the image.
        save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
    """
    try:
        if isinstance(image_input, str):
            image = plt.imread(image_input)
        else:
            image = image_input

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"Raster visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error visualizing raster: {str(e)}")
        raise

def visualize_bounding_boxes(image: Union[np.ndarray, Image.Image], 
                             bboxes: List[Tuple[float, float, float, float]], 
                             save_path: str = None):
    """
    Visualize bounding boxes on an image.

    Args:
        image (Union[np.ndarray, Image.Image]): The input image.
        bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes in format (x1, y1, x2, y2).
        save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if isinstance(image, np.ndarray):
            ax.imshow(image)
        elif isinstance(image, Image.Image):
            ax.imshow(np.array(image))
        else:
            raise ValueError("Image must be a numpy array or PIL Image")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"Bounding box visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error visualizing bounding boxes: {str(e)}")
        raise

def visualize_sam_results(image: Union[np.ndarray, Image.Image], 
                          mask_overlay: np.ndarray, 
                          save_path: str = None):
    """
    Visualize SAM segmentation results.

    Args:
        image (Union[np.ndarray, Image.Image]): The input image.
        mask_overlay (np.ndarray): The segmentation mask overlay.
        save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if isinstance(image, np.ndarray):
            ax.imshow(image)
        elif isinstance(image, Image.Image):
            ax.imshow(np.array(image))
        else:
            raise ValueError("Image must be a numpy array or PIL Image")

        ax.imshow(mask_overlay, cmap="viridis", alpha=0.4)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"SAM results visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error visualizing SAM results: {str(e)}")
        raise

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str = None):
    """
    Plot confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        classes (List[str]): List of class names.
        save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, save_path: str = None):
    """
    Plot precision-recall curve.

    Args:
        precision (np.ndarray): Array of precision values.
        recall (np.ndarray): Array of recall values.
        save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Precision-Recall curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting precision-recall curve: {str(e)}")
        raise
import numpy as np
import torch
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) of two batches of bounding boxes.

    Args:
        boxes1 (torch.Tensor): First set of bounding boxes, shape (N, 4)
        boxes2 (torch.Tensor): Second set of bounding boxes, shape (M, 4)

    Returns:
        torch.Tensor: IoU values, shape (N, M)
    """
    x1_inter = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1_inter = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2_inter = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2_inter = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = boxes1_area[:, None] + boxes2_area - inter_area

    iou = inter_area / union_area

    return iou

def analyze_iou_matrix(iou_matrix: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Analyze the IoU matrix to calculate various metrics.

    Args:
        iou_matrix (torch.Tensor): IoU matrix of shape (N, M)
        threshold (float): IoU threshold for considering a match

    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    matches = iou_matrix >= threshold

    tp = matches.sum().item()
    fp = matches.size(0) - tp
    fn = matches.size(1) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def calculate_segmentation_metrics(ground_truth: np.ndarray, segmentation: np.ndarray) -> Dict[str, float]:
    """
    Calculate segmentation metrics comparing ground truth to predicted segmentation.

    Args:
        ground_truth (np.ndarray): Ground truth binary mask
        segmentation (np.ndarray): Predicted binary mask

    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    if ground_truth.shape != segmentation.shape:
        raise ValueError("Ground truth and segmentation must have the same shape")

    TP = np.sum((ground_truth == 1) & (segmentation == 1))
    TN = np.sum((ground_truth == 0) & (segmentation == 0))
    FP = np.sum((ground_truth == 0) & (segmentation == 1))
    FN = np.sum((ground_truth == 1) & (segmentation == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def evaluate_detection(predicted_boxes: List[Tuple[float, float, float, float]], 
                       ground_truth_boxes: List[Tuple[float, float, float, float]], 
                       iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate object detection results against ground truth.

    Args:
        predicted_boxes (List[Tuple[float, float, float, float]]): Predicted bounding boxes
        ground_truth_boxes (List[Tuple[float, float, float, float]]): Ground truth bounding boxes
        iou_threshold (float): IoU threshold for considering a match

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    pred_boxes = torch.tensor(predicted_boxes)
    gt_boxes = torch.tensor(ground_truth_boxes)

    iou_matrix = calculate_iou_batch(pred_boxes, gt_boxes)
    metrics = analyze_iou_matrix(iou_matrix, threshold=iou_threshold)

    logging.info(f"Detection Evaluation Metrics: {metrics}")
    return metrics

def evaluate_segmentation(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]:
    """
    Evaluate segmentation results against ground truth.

    Args:
        predicted_mask (np.ndarray): Predicted segmentation mask
        ground_truth_mask (np.ndarray): Ground truth segmentation mask

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    metrics = calculate_segmentation_metrics(ground_truth_mask, predicted_mask)
    logging.info(f"Segmentation Evaluation Metrics: {metrics}")
    return metrics
"""Post-processing utilities for bounding boxes."""

from typing import List
import torch

try:
    from torchvision.ops import nms
    TORCHVISION_AVAILABLE = True
except (ImportError, AttributeError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False
    nms = None

from ..utils.types import BoundingBox


def apply_nms(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: List of bounding boxes as (x_min, y_min, x_max, y_max)
        iou_threshold: Intersection over Union threshold for NMS
        
    Returns:
        Filtered bounding boxes as torch.Tensor
        
    Raises:
        ImportError: If torchvision is not available
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for NMS. Install it with: pip install torchvision"
        )
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.ones(len(boxes))  # Use uniform scores
    indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    return boxes_tensor[indices]


def apply_nms_areas(
    boxes: List[BoundingBox], iou_threshold: float = 0.5, inverse_area: bool = False
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes based on their areas.
    
    Smaller boxes will have higher scores if inverse_area is True.
    
    Args:
        boxes: List of bounding boxes as (x_min, y_min, x_max, y_max)
        iou_threshold: Intersection over Union threshold for NMS
        inverse_area: If True, smaller boxes will have higher scores
        
    Returns:
        Filtered bounding boxes as torch.Tensor
        
    Raises:
        ImportError: If torchvision is not available
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for NMS. Install it with: pip install torchvision"
        )
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

    # Compute area for each box: (x2 - x1) * (y2 - y1)
    areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
        boxes_tensor[:, 3] - boxes_tensor[:, 1]
    )

    if inverse_area:
        # Invert areas to use as scores (smaller boxes have higher scores) + regularization
        scores = 1.0 / areas + 1e-6
    else:
        # Use areas directly as scores (larger boxes have higher scores)
        scores = areas

    indices = nms(boxes_tensor, scores, iou_threshold)

    return boxes_tensor[indices]

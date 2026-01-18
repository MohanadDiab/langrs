"""Type definitions and aliases for LangRS."""

from typing import List, Tuple, Union
import numpy as np
from PIL import Image
import torch

# Type aliases
BoundingBox = Tuple[float, float, float, float]
BoundingBoxList = List[BoundingBox]
ImageInput = Union[str, np.ndarray, Image.Image]
MaskArray = np.ndarray
Device = Union[str, torch.device]

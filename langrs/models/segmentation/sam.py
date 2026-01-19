"""SAM (Segment Anything Model) segmentation implementation."""

from typing import Optional, Union
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..base import SegmentationModel
from ...utils.exceptions import ModelLoadError, SegmentationError


class SAMSegmenter(SegmentationModel):
    """
    SAM (Segment Anything Model) segmentation implementation.
    
    This implementation uses segment-anything-py directly and downloads
    model weights from Hugging Face or official sources.
    """

    # Model variants and their checkpoint info
    MODEL_VARIANTS = {
        "vit_h": {
            "checkpoint": "sam_vit_h_4b8939.pth",
            "repo_id": "facebook/sam-vit-huge",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        },
        "vit_l": {
            "checkpoint": "sam_vit_l_0b3195.pth",
            "repo_id": "facebook/sam-vit-large",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        },
        "vit_b": {
            "checkpoint": "sam_vit_b_01ec64.pth",
            "repo_id": "facebook/sam-vit-base",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        },
    }

    DEFAULT_VARIANT = "vit_h"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_variant: str = DEFAULT_VARIANT,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            model_path: Optional path to model checkpoint. If None, downloads from HF.
            device: Device to run model on ('cpu', 'cuda', or torch.device). Defaults to auto-detect.
            model_variant: Model variant to use ('vit_h', 'vit_l', 'vit_b'). Defaults to 'vit_h'.
            cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/huggingface.
            
        Raises:
            ImportError: If segment-anything-py is not installed
            ModelLoadError: If model loading fails
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything-py is not installed. "
                "Install it with: pip install segment-anything-py"
            )

        self.model_variant = model_variant
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        self._model = None
        self._predictor = None
        self._device = None
        self._loaded = False
        self._current_image = None

        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

        # Load model if path provided
        if model_path:
            self.load_weights(model_path)

    def load_weights(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights from path or Hugging Face.
        
        Args:
            model_path: Optional path to model checkpoint. If None, downloads from HF.
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            if model_path is None:
                # Download from Hugging Face
                if not HF_AVAILABLE:
                    raise ModelLoadError(
                        "huggingface_hub is not installed. "
                        "Install it with: pip install huggingface_hub"
                    )

                variant_info = self.MODEL_VARIANTS.get(
                    self.model_variant, self.MODEL_VARIANTS[self.DEFAULT_VARIANT]
                )

                # Try direct download from official URL (more reliable)
                import urllib.request
                checkpoint_path = os.path.join(
                    self.cache_dir, variant_info["checkpoint"]
                )
                
                # Check if file exists and is valid
                file_valid = False
                if os.path.exists(checkpoint_path):
                    try:
                        # Try to load a small part to verify it's not corrupted
                        import torch
                        with open(checkpoint_path, 'rb') as f:
                            # Check file size (should be > 100MB for SAM models)
                            file_size = os.path.getsize(checkpoint_path)
                            if file_size > 100 * 1024 * 1024:  # > 100MB
                                # Try to read the file header
                                header = f.read(1024)
                                if len(header) == 1024:
                                    file_valid = True
                    except Exception:
                        file_valid = False
                
                if not file_valid:
                    # Download from official URL
                    print(f"Downloading SAM model {variant_info['checkpoint']} from official source...")
                    print(f"This may take a few minutes (file is ~2.4GB for vit_h)...")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    try:
                        urllib.request.urlretrieve(variant_info["url"], checkpoint_path)
                        print(f"Download complete: {checkpoint_path}")
                    except Exception as e:
                        # Try Hugging Face as fallback
                        try:
                            checkpoint_path = hf_hub_download(
                                repo_id=variant_info["repo_id"],
                                filename=variant_info["checkpoint"],
                                cache_dir=self.cache_dir,
                                force_download=True,  # Force re-download if corrupted
                            )
                        except Exception:
                            raise ModelLoadError(
                                f"Failed to download SAM model from both official URL and Hugging Face: {e}"
                            )
            else:
                checkpoint_path = model_path
                if not os.path.exists(checkpoint_path):
                    raise ModelLoadError(f"Model checkpoint not found: {checkpoint_path}")

            # Load model
            self._model = sam_model_registry[self.model_variant](checkpoint=checkpoint_path)
            self._model.to(device=self._device)
            self._predictor = SamPredictor(self._model)
            self._model.eval()
            self._loaded = True

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM model: {e}") from e

    def set_image(self, image: Union[Image.Image, np.ndarray]) -> None:
        """
        Set the image for segmentation (required before segmenting).
        
        Args:
            image: Input image as PIL Image or numpy array
        """
        if not self._loaded:
            raise SegmentationError("Model not loaded. Call load_weights() first.")

        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image.convert("RGB"))
            else:
                image_np = image
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)

            self._predictor.set_image(image_np)
            self._current_image = image_np

        except Exception as e:
            raise SegmentationError(f"Failed to set image: {e}") from e

    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate segmentation masks for given bounding boxes.
        
        Args:
            image: Input image as PIL Image or numpy array
            boxes: Bounding boxes tensor of shape (N, 4) where each box
                  is (x_min, y_min, x_max, y_max)
                  
        Returns:
            Masks tensor of shape (N, H, W) where each mask is binary
            (0 or 1) indicating object pixels
            
        Raises:
            SegmentationError: If segmentation fails or model not loaded
        """
        if not self._loaded:
            raise SegmentationError("Model not loaded. Call load_weights() first.")

        try:
            # Set image if not already set or if different
            if self._current_image is None:
                self.set_image(image)
            else:
                # Check if image changed
                if isinstance(image, Image.Image):
                    current_np = np.array(image.convert("RGB"))
                else:
                    current_np = image
                if not np.array_equal(self._current_image, current_np):
                    self.set_image(image)

            # Convert boxes to format expected by SAM
            # SAM expects boxes in (x1, y1, x2, y2) format normalized to [0, 1]
            h, w = self._current_image.shape[:2]
            
            # Normalize boxes to [0, 1]
            sam_boxes_normalized = torch.zeros_like(boxes)
            sam_boxes_normalized[:, 0] = boxes[:, 0] / w  # x_min
            sam_boxes_normalized[:, 1] = boxes[:, 1] / h  # y_min
            sam_boxes_normalized[:, 2] = boxes[:, 2] / w  # x_max
            sam_boxes_normalized[:, 3] = boxes[:, 3] / h  # y_max

            # Run prediction with no_grad for inference performance
            sam_boxes_normalized = sam_boxes_normalized.to(self._device)
            with torch.no_grad():
                masks, scores, logits = self._predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=sam_boxes_normalized,
                    multimask_output=False,
                )

            # Convert to binary masks (0 or 1)
            masks = (masks > 0.0).float()

            return masks.cpu()

        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {e}") from e

    @property
    def device(self) -> torch.device:
        """Get the device the model is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._loaded

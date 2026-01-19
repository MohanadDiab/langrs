"""GroundingDINO detection model implementation."""

from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    import groundingdino.datasets.transforms as T
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..base import DetectionModel
from ...utils.exceptions import ModelLoadError, DetectionError
from ...utils.types import BoundingBox


class GroundingDINODetector(DetectionModel):
    """
    GroundingDINO object detection model.
    
    This implementation uses groundingdino-py directly and downloads
    model weights from Hugging Face.
    """

    # Model variants and their Hugging Face paths
    MODEL_VARIANTS = {
        "swinb_cogcoor": {
            "config": "GroundingDINO_SwinB.cfg.py",
            "checkpoint": "groundingdino_swinb_cogcoor.pth",
            "repo_id": "ShilongLiu/GroundingDINO",
        },
        "swint_ogc": {
            "config": "GroundingDINO_SwinT_OGC.cfg.py",
            "checkpoint": "groundingdino_swint_ogc.pth",
            "repo_id": "ShilongLiu/GroundingDINO",
        },
    }

    DEFAULT_VARIANT = "swint_ogc"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_variant: str = DEFAULT_VARIANT,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize GroundingDINO detector.
        
        Args:
            model_path: Optional path to model checkpoint. If None, downloads from HF.
            device: Device to run model on ('cpu', 'cuda', or torch.device). Defaults to auto-detect.
            model_variant: Model variant to use. Defaults to 'swint_ogc'.
            cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/huggingface.
            
        Raises:
            ImportError: If groundingdino-py is not installed
            ModelLoadError: If model loading fails
        """
        if not GROUNDINGDINO_AVAILABLE:
            raise ImportError(
                "groundingdino-py is not installed. "
                "Install it with: pip install groundingdino-py"
            )

        self.model_variant = model_variant
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        self._model = None
        self._device = None
        self._loaded = False

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

                # Download config and checkpoint
                config_path = hf_hub_download(
                    repo_id=variant_info["repo_id"],
                    filename=variant_info["config"],
                    cache_dir=self.cache_dir,
                )
                checkpoint_path = hf_hub_download(
                    repo_id=variant_info["repo_id"],
                    filename=variant_info["checkpoint"],
                    cache_dir=self.cache_dir,
                )

                # Load model
                self._model = load_model(config_path, checkpoint_path, device=str(self._device))
            else:
                # Load from local path
                if not os.path.exists(model_path):
                    raise ModelLoadError(f"Model checkpoint not found: {model_path}")

                # Try to find config file
                config_path = model_path.replace(".pth", ".py")
                if not os.path.exists(config_path):
                    # Use default config from HF
                    variant_info = self.MODEL_VARIANTS.get(
                        self.model_variant, self.MODEL_VARIANTS[self.DEFAULT_VARIANT]
                    )
                    config_path = hf_hub_download(
                        repo_id=variant_info["repo_id"],
                        filename=variant_info["config"],
                        cache_dir=self.cache_dir,
                    )

                self._model = load_model(config_path, model_path, device=str(self._device))

            self._model.eval()
            self._loaded = True

        except Exception as e:
            raise ModelLoadError(f"Failed to load GroundingDINO model: {e}") from e

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> List[BoundingBox]:
        """
        Detect objects in an image based on text prompt.
        
        Args:
            image: Input image as PIL Image or numpy array
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes (0.0-1.0)
            text_threshold: Confidence threshold for text matching (0.0-1.0)
            
        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max) tuples
            
        Raises:
            DetectionError: If detection fails or model not loaded
        """
        if not self._loaded:
            raise DetectionError("Model not loaded. Call load_weights() first.")

        try:
            # Convert image to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Convert PIL to numpy for groundingdino
            image_np = np.array(pil_image.convert("RGB"))

            # load_image expects a file path, but we have a numpy array
            # So we'll use the image directly and apply transforms manually
            image_source = pil_image.convert("RGB")
            # Apply the same transforms that load_image would apply
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image, _ = transform(image_source, None)
            # Ensure image tensor is on the correct device
            image = image.to(self._device)

            # Run prediction with no_grad for inference performance
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self._model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=str(self._device),  # Explicitly pass device to avoid CUDA default
                )

            # Convert boxes to (x_min, y_min, x_max, y_max) format
            bounding_boxes = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x_center, y_center, width, height = box
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    bounding_boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))

            return bounding_boxes

        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e

    @property
    def device(self) -> torch.device:
        """Get the device the model is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._loaded

"""Rex-Omni detection model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import torch

from ..base import DetectionModel
from ...rex_omni import RexOmniWrapper
from ...utils.exceptions import DetectionError, ModelLoadError
from ...utils.types import BoundingBox


def _prompt_to_categories(text_prompt: str) -> List[str]:
    return [p.strip() for p in text_prompt.split(",") if p.strip()]


@dataclass
class RexOmniConfig:
    model_path: str = "IDEA-Research/Rex-Omni"
    backend: str = "transformers"
    # Common generation defaults (match upstream-ish).
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.05
    top_k: int = 1
    repetition_penalty: float = 1.05
    # Transformers init defaults / fallbacks.
    attn_implementation: str = "flash_attention_2"
    device_map: str = "auto"


class RexOmniDetector(DetectionModel):
    """DetectionModel wrapper around RexOmniWrapper.inference(task='detection')."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        backend: str = "transformers",
        **kwargs,
    ):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._loaded = False
        self._rex: Optional[RexOmniWrapper] = None

        self._config = RexOmniConfig(
            model_path=model_path or RexOmniConfig.model_path,
            backend=backend,
            max_tokens=kwargs.pop("max_tokens", RexOmniConfig.max_tokens),
            temperature=kwargs.pop("temperature", RexOmniConfig.temperature),
            top_p=kwargs.pop("top_p", RexOmniConfig.top_p),
            top_k=kwargs.pop("top_k", RexOmniConfig.top_k),
            repetition_penalty=kwargs.pop(
                "repetition_penalty", RexOmniConfig.repetition_penalty
            ),
            attn_implementation=kwargs.pop(
                "attn_implementation", RexOmniConfig.attn_implementation
            ),
            device_map=kwargs.pop("device_map", RexOmniConfig.device_map),
        )
        self._kwargs = kwargs
        # Rex-Omni weights are loaded lazily via load_weights(), which is invoked
        # by the LangRS pipeline on first use.

    def load_weights(self, model_path: Optional[str] = None) -> None:
        try:
            if model_path is not None:
                self._config.model_path = model_path

            init_kwargs = dict(self._kwargs)
            if self._config.backend == "transformers":
                init_kwargs.setdefault("attn_implementation", self._config.attn_implementation)
                init_kwargs.setdefault("device_map", self._config.device_map)

            self._rex = RexOmniWrapper(
                model_path=self._config.model_path,
                backend=self._config.backend,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                top_k=self._config.top_k,
                repetition_penalty=self._config.repetition_penalty,
                **init_kwargs,
            )
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to load Rex-Omni model: {e}") from e

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> List[BoundingBox]:
        if not self._loaded or self._rex is None:
            raise DetectionError("Model not loaded. Call load_weights() first.")

        # Rex-Omni does not use DINO-style thresholds in the same way; keep signature
        # for interface compatibility.
        _ = (box_threshold, text_threshold)

        try:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            categories = _prompt_to_categories(text_prompt)
            if not categories:
                raise DetectionError(
                    "Empty categories after comma-splitting text_prompt. "
                    "Provide a comma-separated list (e.g. 'building, road')."
                )

            results = self._rex.inference(images=pil_image, task="detection", categories=categories)
            if not results or not results[0].get("success", False):
                raise DetectionError("Rex-Omni inference failed.")

            extracted = results[0].get("extracted_predictions") or {}
            boxes: List[BoundingBox] = []
            for _category, annotations in extracted.items():
                for ann in annotations:
                    if ann.get("type") != "box":
                        continue
                    coords = ann.get("coords") or []
                    if len(coords) != 4:
                        continue
                    x0, y0, x1, y1 = coords
                    boxes.append((float(x0), float(y0), float(x1), float(y1)))

            # If Rex provides confidence but parser doesn't expose it, we keep all boxes.
            return boxes
        except DetectionError:
            raise
        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e

    @property
    def device(self) -> torch.device:
        if self._rex is not None and hasattr(self._rex, "model"):
            try:
                return self._rex.model.device  # transformers
            except Exception:
                pass
        return self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


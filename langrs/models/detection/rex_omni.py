"""Rex-Omni detection model implementation."""

from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os
import time
import json

# Try to import Rex-Omni dependencies
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info, smart_resize
    REX_OMNI_AVAILABLE = True
except ImportError:
    REX_OMNI_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..base import DetectionModel
from ...utils.exceptions import ModelLoadError, DetectionError
from ...utils.types import BoundingBox, Device
from .parser import parse_prediction
from .tasks import TaskType, get_task_config


class RexOmniDetector(DetectionModel):
    """
    Rex-Omni object detection model.
    
    This implementation integrates Rex-Omni directly and downloads
    model weights from Hugging Face. Rex-Omni is a 3B-parameter MLLM
    that performs object detection as a next-token prediction problem.
    """

    # Model variants and their Hugging Face paths
    MODEL_VARIANTS = {
        "default": {
            "repo_id": "IDEA-Research/Rex-Omni",
            "quantization": None,
        },
        "awq": {
            "repo_id": "IDEA-Research/Rex-Omni-AWQ",
            "quantization": "awq",
        },
    }

    DEFAULT_VARIANT = "default"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_variant: str = DEFAULT_VARIANT,
        backend: str = "transformers",
        cache_dir: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.05,
        top_k: int = 1,
        repetition_penalty: float = 1.05,
        **kwargs
    ):
        """
        Initialize Rex-Omni detector.
        
        Args:
            model_path: Optional path to model directory or HF repo ID. 
                       If None, downloads from Hugging Face.
            device: Device to run model on ('cpu', 'cuda', or torch.device). 
                   Defaults to auto-detect.
            model_variant: Model variant to use ('default' or 'awq'). 
                          Defaults to 'default'.
            backend: Backend to use ('transformers' or 'vllm'). 
                    Defaults to 'transformers'. 'vllm' requires vllm package.
            cache_dir: Directory to cache downloaded models. 
                      Defaults to ~/.cache/huggingface.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repetition (>1.0 discourages repetition).
            **kwargs: Additional arguments passed to model initialization.
            
        Raises:
            ImportError: If required dependencies are not installed
            ModelLoadError: If model loading fails
        """
        if not REX_OMNI_AVAILABLE:
            raise ImportError(
                "Rex-Omni dependencies are not installed. "
                "Install them with: pip install transformers qwen-vl-utils accelerate"
            )

        if backend == "vllm" and not VLLM_AVAILABLE:
            raise ImportError(
                "vllm backend requested but vllm is not installed. "
                "Install it with: pip install vllm, or use backend='transformers'"
            )

        self.model_variant = model_variant
        self.backend = backend.lower()
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        self._model = None
        self._processor = None
        self._sampling_params = None
        self._device = None
        self._loaded = False
        self._model_type = None  # Will be set when model is initialized

        # Store generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.system_prompt = "You are a helpful assistant"
        self.min_pixels = 16 * 28 * 28
        self.max_pixels = 2560 * 28 * 28
        self.stop = ["<|im_end|>"]

        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

        # Store kwargs for model initialization
        self._model_kwargs = kwargs

        # Load model if path provided
        if model_path:
            self.load_weights(model_path)

    def load_weights(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights from path or Hugging Face.
        
        Args:
            model_path: Optional path to model directory or HF repo ID. 
                       If None, downloads from Hugging Face.
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            if model_path is None:
                # Download from Hugging Face using default variant
                if not HF_AVAILABLE:
                    raise ModelLoadError(
                        "huggingface_hub is not installed. "
                        "Install it with: pip install huggingface_hub"
                    )

                variant_info = self.MODEL_VARIANTS.get(
                    self.model_variant, self.MODEL_VARIANTS[self.DEFAULT_VARIANT]
                )

                # Download model directory
                model_path = snapshot_download(
                    repo_id=variant_info["repo_id"],
                    cache_dir=self.cache_dir,
                )
            else:
                # Check if model_path is a local path or Hugging Face repo ID
                if os.path.exists(model_path):
                    # It's a valid local path
                    pass
                elif "/" in model_path and not os.path.isabs(model_path):
                    # Likely a Hugging Face repo ID (contains '/' but not absolute path)
                    if not HF_AVAILABLE:
                        raise ModelLoadError(
                            "huggingface_hub is not installed. "
                            "Install it with: pip install huggingface_hub"
                        )
                    
                    # Download from Hugging Face
                    model_path = snapshot_download(
                        repo_id=model_path,
                        cache_dir=self.cache_dir,
                    )
                else:
                    # Local path that doesn't exist
                    raise ModelLoadError(f"Model path not found: {model_path}")

            # Initialize model based on backend
            self._initialize_model(model_path)

            self._loaded = True

        except Exception as e:
            raise ModelLoadError(f"Failed to load Rex-Omni model: {e}") from e

    def _initialize_model(self, model_path: str) -> None:
        """Initialize model and processor based on backend type"""
        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm is not installed")

            # Initialize VLLM model
            self._model = LLM(
                model=model_path,
                tokenizer=model_path,
                tokenizer_mode=self._model_kwargs.get("tokenizer_mode", "slow"),
                limit_mm_per_prompt=self._model_kwargs.get(
                    "limit_mm_per_prompt", {"image": 10, "video": 10}
                ),
                max_model_len=self._model_kwargs.get("max_model_len", 4096),
                gpu_memory_utilization=self._model_kwargs.get("gpu_memory_utilization", 0.8),
                tensor_parallel_size=self._model_kwargs.get("tensor_parallel_size", 1),
                trust_remote_code=self._model_kwargs.get("trust_remote_code", True),
                quantization=self.MODEL_VARIANTS.get(self.model_variant, {}).get("quantization"),
                **{
                    k: v
                    for k, v in self._model_kwargs.items()
                    if k not in [
                        "tokenizer_mode",
                        "limit_mm_per_prompt",
                        "max_model_len",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Set padding side to left for batch inference
            self._processor.tokenizer.padding_side = "left"

            # Set up sampling parameters
            self._sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                temperature=self.temperature,
                skip_special_tokens=False,
                stop=self.stop,
            )

            self._model_type = "vllm"

        elif self.backend == "transformers":
            # Determine attention implementation (try flash_attention_2, fallback to sdpa)
            attn_impl = self._model_kwargs.get("attn_implementation", None)
            if attn_impl is None:
                # Try flash_attention_2, fallback to sdpa if not available
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"  # Use PyTorch's SDPA (Scaled Dot Product Attention)
            
            # Initialize transformers model with error handling for flash_attention
            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=self._model_kwargs.get("torch_dtype", torch.bfloat16),
                    attn_implementation=attn_impl,
                    device_map=self._model_kwargs.get("device_map", "auto"),
                    trust_remote_code=self._model_kwargs.get("trust_remote_code", True),
                    **{
                        k: v
                        for k, v in self._model_kwargs.items()
                        if k not in [
                            "torch_dtype",
                            "attn_implementation",
                            "device_map",
                            "trust_remote_code",
                        ]
                    },
                )
            except (ImportError, ValueError) as e:
                # If flash_attention_2 fails, try with sdpa
                if "flash_attn" in str(e).lower() or "flash_attention" in str(e).lower():
                    if attn_impl == "flash_attention_2":
                        # Retry with sdpa
                        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            model_path,
                            torch_dtype=self._model_kwargs.get("torch_dtype", torch.bfloat16),
                            attn_implementation="sdpa",
                            device_map=self._model_kwargs.get("device_map", "auto"),
                            trust_remote_code=self._model_kwargs.get("trust_remote_code", True),
                            **{
                                k: v
                                for k, v in self._model_kwargs.items()
                                if k not in [
                                    "torch_dtype",
                                    "attn_implementation",
                                    "device_map",
                                    "trust_remote_code",
                                ]
                            },
                        )
                    else:
                        raise
                else:
                    raise

            # Initialize processor
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                use_fast=False,
            )

            # Set padding side to left for batch inference
            self._processor.tokenizer.padding_side = "left"

            self._model_type = "transformers"

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Choose 'transformers' or 'vllm'."
            )

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
            text_prompt: Text description of objects to detect (comma-separated for multiple)
            box_threshold: Not used for Rex-Omni. Rex-Omni doesn't output confidence scores,
                          so this parameter is ignored. Detection quality is controlled by
                          generation parameters (temperature, top_p, top_k) set during initialization.
            text_threshold: Not used for Rex-Omni. Rex-Omni doesn't output confidence scores,
                          so this parameter is ignored. Detection quality is controlled by
                          generation parameters (temperature, top_p, top_k) set during initialization.
            
        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max) tuples
            
        Note:
            Unlike GroundingDINO, Rex-Omni doesn't provide confidence scores for detections.
            To control detection quality, adjust the generation parameters when initializing
            the detector:
            - Lower temperature (default 0.0) = more deterministic
            - Lower top_p (default 0.05) = more focused sampling
            - Lower top_k (default 1) = more conservative generation
            
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

            # Convert text_prompt to categories list
            # Split by comma and strip whitespace
            categories = [cat.strip() for cat in text_prompt.split(",") if cat.strip()]
            if not categories:
                categories = [text_prompt.strip()] if text_prompt.strip() else ["objects"]

            # Run inference
            results = self._inference(
                image=pil_image,
                task=TaskType.DETECTION,
                categories=categories,
            )

            # Extract bounding boxes from results
            bounding_boxes = []
            if results and "extracted_predictions" in results:
                predictions = results["extracted_predictions"]
                for category, detections in predictions.items():
                    for detection in detections:
                        if detection.get("type") == "box" and "coords" in detection:
                            coords = detection["coords"]
                            if len(coords) == 4:
                                # Ensure coordinates are in correct order
                                x0, y0, x1, y1 = coords
                                # Make sure x0 < x1 and y0 < y1
                                x_min, x_max = min(x0, x1), max(x0, x1)
                                y_min, y_max = min(y0, y1), max(y0, y1)
                                bounding_boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))

            return bounding_boxes

        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e

    def _inference(
        self,
        image: Image.Image,
        task: TaskType,
        categories: List[str],
    ) -> dict:
        """Perform inference on a single image"""
        # Get image dimensions
        w, h = image.size

        # Generate prompt based on task
        prompt = self._generate_prompt(
            task=task,
            categories=categories,
            image_width=w,
            image_height=h,
        )

        # Calculate resized dimensions
        resized_height, resized_width = smart_resize(
            h,
            w,
            28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Prepare messages
        if self._model_type == "transformers":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        else:  # vllm
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

        # Generate response
        if self._model_type == "vllm":
            raw_output, _ = self._generate_vllm(messages)
        else:
            raw_output, _ = self._generate_transformers(messages)

        # Parse predictions
        extracted_predictions = parse_prediction(
            text=raw_output,
            w=w,
            h=h,
            task_type=task.value,
        )

        return {
            "success": True,
            "image_size": (w, h),
            "resized_size": (resized_width, resized_height),
            "task": task.value,
            "prompt": prompt,
            "raw_output": raw_output,
            "extracted_predictions": extracted_predictions,
        }

    def _generate_prompt(
        self,
        task: TaskType,
        categories: List[str],
        image_width: int,
        image_height: int,
    ) -> str:
        """Generate prompt based on task configuration"""
        task_config = get_task_config(task)

        if task_config.requires_categories and not categories:
            raise ValueError(f"Categories are required for {task.value} task")

        if categories:
            categories_str = ", ".join(categories)
            return task_config.prompt_template.format(categories=categories_str)
        else:
            return task_config.prompt_template.format(categories="objects")

    def _generate_vllm(self, messages: List[dict]) -> tuple:
        """Generate using VLLM model"""
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {"image": image_inputs}
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        # Generate
        outputs = self._model.generate(
            [llm_inputs], sampling_params=self._sampling_params
        )

        generated_text = outputs[0].outputs[0].text

        return generated_text, {}

    def _generate_transformers(self, messages: List[dict]) -> tuple:
        """Generate using Transformers model"""
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[messages[1]["content"][0]["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._processor.tokenizer.eos_token_id,
        }

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text, {}

    @property
    def device(self) -> torch.device:
        """Get the device the model is running on."""
        if self._loaded and self._model_type == "transformers":
            # Get device from model
            if hasattr(self._model, "device"):
                return self._model.device
            # Try to get from first parameter
            if hasattr(self._model, "parameters"):
                return next(self._model.parameters()).device
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._loaded

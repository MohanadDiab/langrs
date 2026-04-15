#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main wrapper class for Rex-Omni (vendored for LangRS).

Original project:
  https://github.com/IDEA-Research/Rex-Omni
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info, smart_resize
except ImportError:  # pragma: no cover
    process_vision_info = None
    smart_resize = None

from .parser import convert_boxes_to_normalized_bins, parse_prediction
from .tasks import TASK_CONFIGS, TaskType, get_keypoint_config, get_task_config

logger = logging.getLogger(__name__)


class RexOmniWrapper:
    """
    High-level wrapper for Rex-Omni
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        system_prompt: str = "You are a helpful assistant",
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 2560 * 28 * 28,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 1,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = False,
        stop: Optional[List[str]] = None,
        quantization: str = None,
        **kwargs,
    ):
        """
        Initialize the wrapper

        Args:
            model_path: Path to the model directory
            backend: Backend type ("transformers" or "vllm")
            system_prompt: System prompt for the model
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            skip_special_tokens: Whether to skip special tokens in output
            stop: Stop sequences for generation
            quantization: Quantization type
            **kwargs: Additional arguments for model initialization
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.system_prompt = system_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Store generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.stop = stop or ["<|im_end|>"]
        self.quantization = quantization

        # Initialize model and processor
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize model and processor based on backend type"""
        logger.info("Initializing %s backend...", self.backend)

        if self.backend == "vllm":
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            # Initialize VLLM model
            self.model = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                tokenizer_mode=kwargs.get("tokenizer_mode", "slow"),
                limit_mm_per_prompt=kwargs.get(
                    "limit_mm_per_prompt", {"image": 10, "video": 10}
                ),
                max_model_len=kwargs.get("max_model_len", 4096),
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                quantization=self.quantization,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
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
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Set padding side to left for batch inference with Flash Attention
            self.processor.tokenizer.padding_side = "left"

            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                temperature=self.temperature,
                skip_special_tokens=self.skip_special_tokens,
                stop=self.stop,
            )

            self.model_type = "vllm"

        elif self.backend == "transformers":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            # Initialize transformers model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                attn_implementation=kwargs.get(
                    "attn_implementation", "flash_attention_2"
                ),
                device_map=kwargs.get("device_map", "auto"),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "torch_dtype",
                        "attn_implementation",
                        "device_map",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                use_fast=False,
            )

            # Set padding side to left for batch inference with Flash Attention
            self.processor.tokenizer.padding_side = "left"

            self.model_type = "transformers"

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Choose 'transformers' or 'vllm'."
            )

    def inference(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task: Union[str, TaskType, List[Union[str, TaskType]]],
        categories: Optional[Union[str, List[str], List[List[str]]]] = None,
        keypoint_type: Optional[Union[str, List[str]]] = None,
        visual_prompt_boxes: Optional[
            Union[List[List[float]], List[List[List[float]]]]
        ] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference on images for various vision tasks.

        See upstream Rex-Omni README for detailed examples.
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]

        batch_size = len(images)

        # Normalize inputs to batch format
        tasks, categories_list, keypoint_types, visual_prompt_boxes_list = (
            self._normalize_batch_inputs(
                task, categories, keypoint_type, visual_prompt_boxes, batch_size
            )
        )

        # Perform batch inference
        return self._inference_batch(
            images=images,
            tasks=tasks,
            categories_list=categories_list,
            keypoint_types=keypoint_types,
            visual_prompt_boxes_list=visual_prompt_boxes_list,
            **kwargs,
        )

    def _normalize_batch_inputs(
        self,
        task: Union[str, TaskType, List[Union[str, TaskType]]],
        categories: Optional[Union[str, List[str], List[List[str]]]],
        keypoint_type: Optional[Union[str, List[str]]],
        visual_prompt_boxes: Optional[
            Union[List[List[float]], List[List[List[float]]]]
        ],
        batch_size: int,
    ) -> Tuple[
        List[TaskType],
        List[Optional[List[str]]],
        List[Optional[str]],
        List[Optional[List[List[float]]]],
    ]:
        """Normalize all inputs to batch format"""

        if isinstance(task, (str, TaskType)):
            if isinstance(task, str):
                task = TaskType(task.lower())
            tasks = [task] * batch_size
        else:
            tasks = []
            for t in task:
                if isinstance(t, str):
                    tasks.append(TaskType(t.lower()))
                else:
                    tasks.append(t)

            if len(tasks) != batch_size:
                raise ValueError(
                    f"Number of tasks ({len(tasks)}) must match number of images ({batch_size})"
                )

        if categories is None:
            categories_list = [None] * batch_size
        elif isinstance(categories, str):
            categories_list = [[categories]] * batch_size
        elif isinstance(categories, list):
            if len(categories) == 0:
                categories_list = [None] * batch_size
            elif isinstance(categories[0], str):
                categories_list = [categories] * batch_size
            else:
                categories_list = categories
                if len(categories_list) != batch_size:
                    raise ValueError(
                        f"Number of category lists ({len(categories_list)}) must match number of images ({batch_size})"
                    )
        else:
            categories_list = [None] * batch_size

        if keypoint_type is None:
            keypoint_types = [None] * batch_size
        elif isinstance(keypoint_type, str):
            keypoint_types = [keypoint_type] * batch_size
        else:
            keypoint_types = keypoint_type
            if len(keypoint_types) != batch_size:
                raise ValueError(
                    f"Number of keypoint types ({len(keypoint_types)}) must match number of images ({batch_size})"
                )

        if visual_prompt_boxes is None:
            visual_prompt_boxes_list = [None] * batch_size
        elif isinstance(visual_prompt_boxes, list):
            if len(visual_prompt_boxes) == 0:
                visual_prompt_boxes_list = [None] * batch_size
            elif isinstance(visual_prompt_boxes[0], (int, float)):
                visual_prompt_boxes_list = [[visual_prompt_boxes]] * batch_size
            elif isinstance(visual_prompt_boxes[0], list):
                if len(visual_prompt_boxes[0]) == 4 and isinstance(
                    visual_prompt_boxes[0][0], (int, float)
                ):
                    visual_prompt_boxes_list = [visual_prompt_boxes] * batch_size
                else:
                    visual_prompt_boxes_list = visual_prompt_boxes
                    if len(visual_prompt_boxes_list) != batch_size:
                        raise ValueError(
                            f"Number of visual prompt box lists ({len(visual_prompt_boxes_list)}) must match number of images ({batch_size})"
                        )
            else:
                visual_prompt_boxes_list = [None] * batch_size
        else:
            visual_prompt_boxes_list = [None] * batch_size

        return tasks, categories_list, keypoint_types, visual_prompt_boxes_list

    def _inference_batch(
        self,
        images: List[Image.Image],
        tasks: List[TaskType],
        categories_list: List[Optional[List[str]]],
        keypoint_types: List[Optional[str]],
        visual_prompt_boxes_list: List[Optional[List[List[float]]]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Perform true batch inference"""

        start_time = time.time()
        batch_size = len(images)

        batch_messages = []
        batch_prompts = []
        batch_image_sizes = []

        for i in range(batch_size):
            image = images[i]
            task = tasks[i]
            categories = categories_list[i]
            keypoint_type = keypoint_types[i]
            visual_prompt_boxes = visual_prompt_boxes_list[i]

            w, h = image.size
            batch_image_sizes.append((w, h))

            prompt = self._generate_prompt(
                task=task,
                categories=categories,
                keypoint_type=keypoint_type,
                visual_prompt_boxes=visual_prompt_boxes,
                image_width=w,
                image_height=h,
            )
            batch_prompts.append(prompt)

            if smart_resize is None:
                raise ImportError(
                    "qwen_vl_utils is required for Rex-Omni. "
                    "Install with: pip install qwen-vl-utils"
                )

            resized_height, resized_width = smart_resize(
                h,
                w,
                28,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            if self.model_type == "transformers":
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
            else:
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

            batch_messages.append(messages)

        if self.model_type == "vllm":
            batch_outputs, batch_generation_info = self._generate_vllm_batch(
                batch_messages
            )
        else:
            batch_outputs, batch_generation_info = self._generate_transformers_batch(
                batch_messages, images
            )

        results = []
        total_time = time.time() - start_time

        for i in range(batch_size):
            raw_output = batch_outputs[i]
            generation_info = batch_generation_info[i]
            w, h = batch_image_sizes[i]
            task = tasks[i]
            prompt = batch_prompts[i]

            extracted_predictions = parse_prediction(
                text=raw_output,
                w=w,
                h=h,
                task_type=task.value,
            )

            if smart_resize is None:
                raise ImportError(
                    "qwen_vl_utils is required for Rex-Omni. "
                    "Install with: pip install qwen-vl-utils"
                )

            resized_height, resized_width = smart_resize(
                h,
                w,
                28,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            result = {
                "success": True,
                "image_size": (w, h),
                "resized_size": (resized_width, resized_height),
                "task": task.value,
                "prompt": prompt,
                "raw_output": raw_output,
                "extracted_predictions": extracted_predictions,
                "inference_time": total_time,
                **generation_info,
            }
            results.append(result)

        return results

    def _generate_prompt(
        self,
        task: TaskType,
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        image_width: int = None,
        image_height: int = None,
    ) -> str:
        """Generate prompt based on task configuration"""

        task_config = get_task_config(task)

        if task == TaskType.VISUAL_PROMPTING:
            if visual_prompt_boxes is None:
                raise ValueError(
                    "Visual prompt boxes are required for visual prompting task"
                )

            word_mapped_boxes = convert_boxes_to_normalized_bins(
                visual_prompt_boxes, image_width, image_height
            )
            visual_prompt_dict = {"object_1": word_mapped_boxes}
            visual_prompt_json = json.dumps(visual_prompt_dict)

            return task_config.prompt_template.format(visual_prompt=visual_prompt_json)

        if task == TaskType.KEYPOINT:
            if categories is None:
                raise ValueError("Categories are required for keypoint task")
            if keypoint_type is None:
                raise ValueError("Keypoint type is required for keypoint task")

            keypoints_list = get_keypoint_config(keypoint_type)
            if keypoints_list is None:
                raise ValueError(f"Unknown keypoint type: {keypoint_type}")

            keypoints_str = ", ".join(keypoints_list)
            categories_str = (
                ", ".join(categories) if isinstance(categories, list) else categories
            )

            return task_config.prompt_template.format(
                categories=categories_str, keypoints=keypoints_str
            )

        if task_config.requires_categories and categories is None:
            raise ValueError(f"Categories are required for {task.value} task")

        if categories is not None:
            categories_str = ", ".join(categories) if isinstance(categories, list) else categories
            return task_config.prompt_template.format(categories=categories_str)

        return task_config.prompt_template.format(categories="objects")

    def _generate_vllm(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using VLLM model"""

        if process_vision_info is None:
            raise ImportError(
                "qwen_vl_utils is required for Rex-Omni. "
                "Install with: pip install qwen-vl-utils"
            )
        image_inputs, _video_inputs = process_vision_info(messages)
        mm_data = {"image": image_inputs}
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        generation_start = time.time()
        outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
        generation_time = time.time() - generation_start

        generated_text = outputs[0].outputs[0].text

        output_tokens = outputs[0].outputs[0].token_ids
        num_output_tokens = len(output_tokens) if output_tokens else 0

        prompt_token_ids = outputs[0].prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0

        tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

        return generated_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def _generate_vllm_batch(
        self, batch_messages: List[List[Dict]]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate using VLLM model for batch processing"""

        batch_inputs = []
        for messages in batch_messages:
            if process_vision_info is None:
                raise ImportError(
                    "qwen_vl_utils is required for Rex-Omni. "
                    "Install with: pip install qwen-vl-utils"
                )
            image_inputs, _video_inputs = process_vision_info(messages)
            mm_data = {"image": image_inputs}
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            batch_inputs.append(llm_inputs)

        generation_start = time.time()
        outputs = self.model.generate(batch_inputs, sampling_params=self.sampling_params)
        generation_time = time.time() - generation_start

        batch_outputs = []
        batch_generation_info = []

        for output in outputs:
            generated_text = output.outputs[0].text
            batch_outputs.append(generated_text)

            output_tokens = output.outputs[0].token_ids
            num_output_tokens = len(output_tokens) if output_tokens else 0

            prompt_token_ids = output.prompt_token_ids
            num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0

            tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

            batch_generation_info.append(
                {
                    "num_output_tokens": num_output_tokens,
                    "num_prompt_tokens": num_prompt_tokens,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                }
            )

        return batch_outputs, batch_generation_info

    def _generate_transformers(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using Transformers model"""

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        generation_start = time.time()
        inputs = self.processor(
            text=[text],
            images=[messages[1]["content"][0]["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - generation_start

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]

        num_output_tokens = len(generated_ids_trimmed[0])
        num_prompt_tokens = len(inputs.input_ids[0])
        tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

        return output_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def _generate_transformers_batch(
        self, batch_messages: List[List[Dict]], batch_images: List[Image.Image]
    ) -> Tuple[List[str], List[Dict]]:
        """Generate using Transformers model for batch processing"""

        batch_texts = []
        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        generation_start = time.time()
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - generation_start

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        batch_generation_info = []
        for i, output_ids in enumerate(generated_ids_trimmed):
            num_output_tokens = len(output_ids)
            num_prompt_tokens = len(inputs.input_ids[i])
            tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0
            batch_generation_info.append(
                {
                    "num_output_tokens": num_output_tokens,
                    "num_prompt_tokens": num_prompt_tokens,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                }
            )

        return batch_outputs, batch_generation_info

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks"""
        return [task.value for task in TaskType]

    def get_task_info(self, task: Union[str, TaskType]) -> Dict[str, Any]:
        """Get information about a specific task"""
        if isinstance(task, str):
            task = TaskType(task.lower())

        config = get_task_config(task)
        return {
            "name": config.name,
            "description": config.description,
            "output_format": config.output_format,
            "requires_categories": config.requires_categories,
            "requires_visual_prompt": config.requires_visual_prompt,
            "requires_keypoint_type": config.requires_keypoint_type,
            "prompt_template": config.prompt_template,
        }



#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output parsing utilities for Rex Omni (vendored for LangRS).

Original project:
  https://github.com/IDEA-Research/Rex-Omni
"""

import json
import re
from typing import Dict, List


def parse_prediction(text: str, w: int, h: int, task_type: str = "detection") -> Dict[str, List]:
    """
    Parse model output text to extract category-wise predictions.
    """
    if task_type == "keypoint":
        return parse_keypoint_prediction(text, w, h)
    return parse_standard_prediction(text, w, h)


def parse_standard_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse standard prediction output for detection, pointing, etc.
    """
    result: Dict[str, List] = {}

    text = text.split("<|im_end|>")[0]
    if not text.endswith("<|box_end|>"):
        text = text + "<|box_end|>"

    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    for category, coords_text in matches:
        category = category.strip()

        coord_pattern = r"<(\d+)>"
        annotations = []
        coord_strings = coords_text.split(",")

        for coord_str in coord_strings:
            coord_nums = re.findall(coord_pattern, coord_str.strip())

            if len(coord_nums) == 2:
                try:
                    x_bin = int(coord_nums[0])
                    y_bin = int(coord_nums[1])
                    x = (x_bin / 999.0) * w
                    y = (y_bin / 999.0) * h
                    annotations.append({"type": "point", "coords": [x, y]})
                except (ValueError, IndexError):
                    continue

            elif len(coord_nums) == 4:
                try:
                    x0_bin = int(coord_nums[0])
                    y0_bin = int(coord_nums[1])
                    x1_bin = int(coord_nums[2])
                    y1_bin = int(coord_nums[3])

                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h
                    annotations.append({"type": "box", "coords": [x0, y0, x1, y1]})
                except (ValueError, IndexError):
                    continue

            elif len(coord_nums) > 4 and len(coord_nums) % 2 == 0:
                try:
                    polygon_coords = []
                    for i in range(0, len(coord_nums), 2):
                        x_bin = int(coord_nums[i])
                        y_bin = int(coord_nums[i + 1])
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h
                        polygon_coords.append([x, y])
                    annotations.append({"type": "polygon", "coords": polygon_coords})
                except (ValueError, IndexError):
                    continue

        if category not in result:
            result[category] = []
        result[category].extend(annotations)

    return result


def parse_keypoint_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse keypoint task JSON output to extract bbox and keypoints.

    Kept for upstream parity; LangRS detection adapter does not rely on this.
    """
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return {}
        json_str = text[start_idx : end_idx + 1]
    else:
        json_str = json_matches[0]

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    result: Dict[str, List] = {}
    coord_pattern = r"<(\d+)>"

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        coord_matches = re.findall(coord_pattern, bbox) if isinstance(bbox, str) else []
        if len(coord_matches) != 4:
            continue

        x0_bin, y0_bin, x1_bin, y1_bin = [int(match) for match in coord_matches]
        converted_bbox = [
            (x0_bin / 999.0) * w,
            (y0_bin / 999.0) * h,
            (x1_bin / 999.0) * w,
            (y1_bin / 999.0) * h,
        ]

        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible" or kp_coords is None:
                converted_keypoints[kp_name] = "unvisible"
                continue
            kp_matches = re.findall(coord_pattern, kp_coords) if isinstance(kp_coords, str) else []
            if len(kp_matches) != 2:
                converted_keypoints[kp_name] = "unvisible"
                continue
            x_bin, y_bin = [int(match) for match in kp_matches]
            converted_keypoints[kp_name] = [(x_bin / 999.0) * w, (y_bin / 999.0) * h]

        category = "keypoint_instance"
        if instance_id:
            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        result.setdefault(category, []).append(
            {
                "type": "keypoint",
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "instance_id": instance_id,
            }
        )

    return result


def convert_boxes_to_normalized_bins(boxes: List[List[float]], ori_width: int, ori_height: int) -> List[str]:
    """Convert boxes from absolute coordinates to normalized bins (0-999) and map to words."""
    word_mapped_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box

        x0_norm = max(0.0, min(1.0, x0 / ori_width))
        x1_norm = max(0.0, min(1.0, x1 / ori_width))
        y0_norm = max(0.0, min(1.0, y0 / ori_height))
        y1_norm = max(0.0, min(1.0, y1 / ori_height))

        x0_bin = max(0, min(999, int(x0_norm * 999)))
        y0_bin = max(0, min(999, int(y0_norm * 999)))
        x1_bin = max(0, min(999, int(x1_norm * 999)))
        y1_bin = max(0, min(999, int(y1_norm * 999)))

        word_mapped_box = "".join([f"<{x0_bin}>", f"<{y0_bin}>", f"<{x1_bin}>", f"<{y1_bin}>"])
        word_mapped_boxes.append(word_mapped_box)

    return word_mapped_boxes


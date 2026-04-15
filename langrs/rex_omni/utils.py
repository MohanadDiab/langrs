#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for Rex Omni visualization (vendored for LangRS).

Original project:
  https://github.com/IDEA-Research/Rex-Omni
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ColorGenerator:
    """Generate consistent colors for visualization"""

    def __init__(self, color_type: str = "text"):
        self.color_type = color_type

        if color_type == "same":
            self.color = tuple((np.random.randint(0, 127, size=3) + 128).tolist())
        elif color_type == "text":
            np.random.seed(3396)
            self.num_colors = 300
            self.colors = np.random.randint(0, 127, size=(self.num_colors, 3)) + 128
        else:
            raise ValueError(f"Unknown color type: {color_type}")

    def get_color(self, text: str) -> Tuple[int, int, int]:
        """Get color for given text"""
        if self.color_type == "same":
            return self.color

        if self.color_type == "text":
            text_hash = hash(text)
            index = text_hash % self.num_colors
            color = tuple(self.colors[index])
            return color

        raise ValueError(f"Unknown color type: {self.color_type}")


def RexOmniVisualize(
    image: Image.Image,
    predictions: Dict[str, List[Dict]],
    font_size: int = 15,
    draw_width: int = 6,
    show_labels: bool = True,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    font_path: Optional[str] = None,
) -> Image.Image:
    """
    Visualize predictions on image.
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    font = _load_font(font_size, font_path)
    color_generator = ColorGenerator("text")

    for category, annotations in predictions.items():
        if custom_colors and category in custom_colors:
            color = custom_colors[category]
        else:
            color = color_generator.get_color(category)

        for annotation in annotations:
            annotation_type = annotation.get("type", "box")
            coords = annotation.get("coords", [])

            if annotation_type == "box" and len(coords) == 4:
                _draw_box(draw, coords, color, draw_width, category, font, show_labels)
            elif annotation_type == "point" and len(coords) == 2:
                _draw_point(draw, coords, color, draw_width, category, font, show_labels)
            elif annotation_type == "polygon" and len(coords) >= 3:
                _draw_polygon(
                    draw,
                    vis_image,
                    coords,
                    color,
                    draw_width,
                    category,
                    font,
                    show_labels,
                )
            elif annotation_type == "keypoint":
                _draw_keypoint(draw, annotation, color, draw_width, font, show_labels)

    return vis_image


def _load_font(font_size: int, font_path: Optional[str] = None) -> ImageFont.ImageFont:
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf",
    ]

    font = None
    for font_path_ in font_paths:
        try:
            font = ImageFont.truetype(font_path_, font_size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    return font


def _draw_box(
    draw: ImageDraw.ImageDraw,
    coords: List[float],
    color: Tuple[int, int, int],
    draw_width: int,
    label: str,
    font: ImageFont.ImageFont,
    show_labels: bool,
):
    x0, y0, x1, y1 = [int(c) for c in coords]
    if x0 >= x1 or y0 >= y1:
        return

    draw.rectangle([x0, y0, x1, y1], outline=color, width=draw_width)

    if show_labels and label:
        bbox = draw.textbbox((x0, y0), label, font)
        box_h = bbox[3] - bbox[1]

        y0_text = y0 - box_h - (draw_width * 2)
        y1_text = y0 + draw_width

        if y0_text < 0:
            y0_text = 0
            y1_text = y0 + 2 * draw_width + box_h

        draw.rectangle([x0, y0_text, bbox[2] + draw_width * 2, y1_text], fill=color)
        draw.text((x0 + draw_width, y0_text), label, fill="black", font=font)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    coords: List[float],
    color: Tuple[int, int, int],
    draw_width: int,
    label: str,
    font: ImageFont.ImageFont,
    show_labels: bool,
):
    x, y = [int(c) for c in coords]

    radius = max(8, draw_width)
    border_width = 3

    draw.ellipse(
        [
            x - radius - border_width,
            y - radius - border_width,
            x + radius + border_width,
            y + radius + border_width,
        ],
        fill="white",
        outline="white",
    )

    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=color)

    if show_labels and label:
        label_x, label_y = x + 15, y - 15
        bbox = draw.textbbox((label_x, label_y), label, font)
        box_h = bbox[3] - bbox[1]
        box_w = bbox[2] - bbox[0]
        padding = 4

        draw.rectangle(
            [
                label_x - padding,
                label_y - box_h - padding,
                label_x + box_w + padding,
                label_y + padding,
            ],
            fill=color,
        )
        draw.text((label_x, label_y - box_h), label, fill="white", font=font)


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    coords: List[List[float]],
    color: Tuple[int, int, int],
    draw_width: int,
    label: str,
    font: ImageFont.ImageFont,
    show_labels: bool,
):
    flat_coords: List[int] = []
    for point in coords:
        flat_coords.extend([int(point[0]), int(point[1])])

    draw.polygon(flat_coords, outline=color, width=draw_width)

    if show_labels and label and coords:
        label_x, label_y = int(coords[0][0]), int(coords[0][1]) - 10
        bbox = draw.textbbox((label_x, label_y), label, font)
        box_h = bbox[3] - bbox[1]
        box_w = bbox[2] - bbox[0]

        draw.rectangle([label_x - 4, label_y - box_h - 4, label_x + box_w + 4, label_y], fill=color)
        draw.text((label_x, label_y - box_h - 2), label, fill="black", font=font)


def _draw_keypoint(
    draw: ImageDraw.ImageDraw,
    annotation: Dict[str, Any],
    color: Tuple[int, int, int],
    draw_width: int,
    font: ImageFont.ImageFont,
    show_labels: bool,
):
    bbox = annotation.get("bbox", [])
    keypoints = annotation.get("keypoints", {})
    instance_id = annotation.get("instance_id", "")

    if len(bbox) == 4:
        _draw_box(draw, bbox, color, draw_width, instance_id, font, show_labels)

    skeleton_connections = [
        ("nose", "left eye"),
        ("nose", "right eye"),
        ("left eye", "left ear"),
        ("right eye", "right ear"),
        ("left shoulder", "right shoulder"),
        ("left shoulder", "left elbow"),
        ("right shoulder", "right elbow"),
        ("left elbow", "left wrist"),
        ("right elbow", "right wrist"),
        ("left shoulder", "left hip"),
        ("right shoulder", "right hip"),
        ("left hip", "right hip"),
        ("left hip", "left knee"),
        ("right hip", "right knee"),
        ("left knee", "left ankle"),
        ("right knee", "right ankle"),
    ]

    hand_skeleton_connections = [
        ("wrist", "thumb root"),
        ("thumb root", "thumb's third knuckle"),
        ("thumb's third knuckle", "thumb's second knuckle"),
        ("thumb's second knuckle", "thumb's first knuckle"),
        ("wrist", "forefinger's root"),
        ("forefinger's root", "forefinger's third knuckle"),
        ("forefinger's third knuckle", "forefinger's second knuckle"),
        ("forefinger's second knuckle", "forefinger's first knuckle"),
        ("wrist", "middle finger's root"),
        ("middle finger's root", "middle finger's third knuckle"),
        ("middle finger's third knuckle", "middle finger's second knuckle"),
        ("middle finger's second knuckle", "middle finger's first knuckle"),
        ("wrist", "ring finger's root"),
        ("ring finger's root", "ring finger's third knuckle"),
        ("ring finger's third knuckle", "ring finger's second knuckle"),
        ("ring finger's second knuckle", "ring finger's first knuckle"),
        ("wrist", "pinky finger's root"),
        ("pinky finger's root", "pinky finger's third knuckle"),
        ("pinky finger's third knuckle", "pinky finger's second knuckle"),
        ("pinky finger's second knuckle", "pinky finger's first knuckle"),
    ]

    animal_skeleton_connections = [
        ("left eye", "right eye"),
        ("left eye", "nose"),
        ("right eye", "nose"),
        ("nose", "neck"),
        ("neck", "left shoulder"),
        ("neck", "right shoulder"),
        ("left shoulder", "left elbow"),
        ("right shoulder", "right elbow"),
        ("left elbow", "left front paw"),
        ("right elbow", "right front paw"),
        ("neck", "left hip"),
        ("neck", "right hip"),
        ("left hip", "left knee"),
        ("right hip", "right knee"),
        ("left knee", "left back paw"),
        ("right knee", "right back paw"),
        ("neck", "root of tail"),
    ]

    if "wrist" in keypoints:
        connections = hand_skeleton_connections
    elif "left shoulder" in keypoints and "left hip" in keypoints:
        connections = skeleton_connections
    else:
        connections = animal_skeleton_connections

    if len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        bbox_area = (x1 - x0) * (y1 - y0)
        dynamic_radius = max(2, min(5, int((bbox_area / 10000) ** 0.5 * 4)))
    else:
        dynamic_radius = max(4, draw_width // 2)

    skeleton_color = (173, 216, 230)
    for kp1_name, kp2_name in connections:
        if (
            kp1_name in keypoints
            and kp2_name in keypoints
            and keypoints[kp1_name] != "unvisible"
            and keypoints[kp2_name] != "unvisible"
        ):
            kp1_coords = keypoints[kp1_name]
            kp2_coords = keypoints[kp2_name]
            if (
                isinstance(kp1_coords, list)
                and len(kp1_coords) == 2
                and isinstance(kp2_coords, list)
                and len(kp2_coords) == 2
            ):
                x1p, y1p = int(kp1_coords[0]), int(kp1_coords[1])
                x2p, y2p = int(kp2_coords[0]), int(kp2_coords[1])
                draw.line([(x1p, y1p), (x2p, y2p)], fill=skeleton_color, width=4)

    for kp_coords in keypoints.values():
        if kp_coords != "unvisible" and isinstance(kp_coords, list) and len(kp_coords) == 2:
            x, y = int(kp_coords[0]), int(kp_coords[1])
            kp_color = (51, 153, 255)
            draw.ellipse(
                [x - dynamic_radius, y - dynamic_radius, x + dynamic_radius, y + dynamic_radius],
                fill=kp_color,
                outline="white",
                width=3,
            )


def format_predictions_for_display(predictions: Dict[str, List[Dict]]) -> str:
    """Format predictions for text display"""
    if not predictions:
        return "No predictions found."

    lines = []
    for category, annotations in predictions.items():
        lines.append(f"\n{category.upper()}:")
        for i, annotation in enumerate(annotations):
            ann_type = annotation.get("type", "unknown")
            coords = annotation.get("coords", [])

            if ann_type == "box" and len(coords) == 4:
                x0, y0, x1, y1 = coords
                lines.append(f"  Box {i+1}: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
            elif ann_type == "point" and len(coords) == 2:
                x, y = coords
                lines.append(f"  Point {i+1}: ({x:.1f}, {y:.1f})")
            elif ann_type == "polygon":
                lines.append(f"  Polygon {i+1}: {len(coords)} points")
            elif ann_type == "keypoint":
                bbox = annotation.get("bbox", [])
                keypoints = annotation.get("keypoints", {})
                visible_kps = sum(1 for kp in keypoints.values() if kp != "unvisible")
                lines.append(f"  Instance {i+1}: {visible_kps}/{len(keypoints)} keypoints visible")

    return "\n".join(lines)


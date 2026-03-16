"""
Утилиты конвертации изображений: numpy <-> base64.
"""

import base64
import cv2
import numpy as np


def numpy_to_base64(img: np.ndarray, fmt: str = "png") -> str:
    """Кодирует BGR numpy array в base64-строку."""
    success, buf = cv2.imencode(f".{fmt}", img)
    if not success:
        raise ValueError(f"Не удалось закодировать изображение в формат {fmt}")
    return base64.b64encode(buf).decode("utf-8")


def serialize_results_images(
    results: dict, fmt: str = "png"
) -> dict[str, str | dict[str, str]]:
    """
    Конвертирует все numpy-изображения из результатов пайплайна в base64.

    Returns:
        {"annotated": "base64...", "layers": {"keypoints": "base64...", ...}}
    """
    images = {}

    annotated = results.get("annotated_image")
    if annotated is not None:
        images["annotated"] = numpy_to_base64(annotated, fmt)

    layer_images = results.get("layer_images", {})
    if layer_images:
        images["layers"] = {}
        for layer_name, layer_img in layer_images.items():
            if layer_img is not None and isinstance(layer_img, np.ndarray):
                images["layers"][layer_name] = numpy_to_base64(layer_img, fmt)

    heatmap_overlay = results.get("heatmap_overlay")
    if heatmap_overlay is not None:
        images.setdefault("layers", {})["gradcam"] = numpy_to_base64(
            heatmap_overlay, fmt
        )

    return images

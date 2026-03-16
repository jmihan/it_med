"""
Компоненты отображения медицинских изображений.
Поддерживает сравнение оригинал/разметка и послойную композицию.
"""

import cv2
import numpy as np
import streamlit as st
from typing import Dict, List, Optional


def _to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Конвертация BGR (OpenCV) → RGB (Streamlit)."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def render_comparison(original: np.ndarray, annotated: np.ndarray):
    """
    Отображение оригинала и размеченного снимка в двух колонках.
    Используется в режиме врача.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходный снимок")
        st.image(_to_rgb(original), use_container_width=True)
    with col2:
        st.subheader("Разметка ИИ")
        st.image(_to_rgb(annotated), use_container_width=True)


def render_layered(
    original: np.ndarray,
    layer_images: Dict[str, np.ndarray],
    active_layers: Dict[str, bool],
    layer_order: List[str],
):
    """
    Послойная композиция изображения для студенческого режима.
    Последовательно накладывает включённые слои на оригинал.

    Args:
        original: Исходное BGR изображение
        layer_images: {layer_key: BGR image с отрисованным слоем}
        active_layers: {layer_key: bool} — какие слои включены
        layer_order: Порядок наложения слоёв (снизу вверх)
    """
    # Собираем изображение с включёнными слоями
    # Каждый layer_image — это отдельный слой, нарисованный поверх оригинала.
    # Для композиции: берём пиксели, отличающиеся от оригинала в каждом слое.
    img = original.copy()

    for layer_key in layer_order:
        if not active_layers.get(layer_key, False):
            continue
        layer_img = layer_images.get(layer_key)
        if layer_img is None:
            continue

        # Маска: где слой отличается от оригинала (там нарисована графика)
        diff = cv2.absdiff(layer_img, original)
        mask = np.any(diff > 5, axis=2)  # Порог для шума

        # Наложение только изменённых пикселей
        img[mask] = layer_img[mask]

    st.image(_to_rgb(img), use_container_width=True)


def render_single(image: np.ndarray, caption: Optional[str] = None):
    """Отображение одного изображения."""
    st.image(_to_rgb(image), caption=caption, use_container_width=True)

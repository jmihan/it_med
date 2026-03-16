"""
Единый модуль загрузки медицинских изображений.

Поддерживает: DICOM (.dcm), PNG, JPG/JPEG.
Всегда возвращает BGR numpy array (формат OpenCV).
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Загрузка изображения из файла.

    Args:
        path: Путь к файлу (DICOM / PNG / JPG / JPEG)

    Returns:
        BGR numpy array (H, W, 3)
    """
    if path.lower().endswith(".dcm") or not path.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(float)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) * (255.0 / (img_max - img_min))
            img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception:
            pass  # Пробуем через OpenCV

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return _normalize_channels(img)


def load_from_upload(uploaded_file) -> np.ndarray:
    """
    Загрузка из Streamlit UploadedFile.

    Args:
        uploaded_file: объект st.file_uploader

    Returns:
        BGR numpy array (H, W, 3)
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".dcm"):
        try:
            import pydicom
            import io
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            img = ds.pixel_array.astype(float)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) * (255.0 / (img_max - img_min))
            img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception:
            pass

    # PNG / JPG / JPEG
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {uploaded_file.name}")
    return _normalize_channels(img)


def load_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """
    Загрузка изображения из байтов (для REST API).

    Args:
        data: Содержимое файла в байтах
        filename: Имя файла (для определения формата)

    Returns:
        BGR numpy array (H, W, 3)
    """
    if filename.lower().endswith(".dcm"):
        try:
            import pydicom
            import io
            ds = pydicom.dcmread(io.BytesIO(data))
            img = ds.pixel_array.astype(float)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) * (255.0 / (img_max - img_min))
            img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception:
            pass

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {filename}")
    return _normalize_channels(img)


def _normalize_channels(img: np.ndarray) -> np.ndarray:
    """Приведение к 3-канальному BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

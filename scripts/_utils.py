"""
Общие утилиты для скриптов проекта.

Содержит константы и функции, дублировавшиеся в нескольких скриптах.
"""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Директории с обработанными изображениями
IMAGE_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "train" / "normal",
    PROJECT_ROOT / "data" / "processed" / "train" / "pathology",
    PROJECT_ROOT / "data" / "processed" / "test" / "images",
]

KP_DIR = PROJECT_ROOT / "data" / "keypoints"

# 8 ключевых точек для анализа тазобедренного сустава
KEYPOINT_NAMES = [
    "L_TRC", "R_TRC", "L_ACE", "R_ACE",
    "L_FHC", "R_FHC", "L_FMM", "R_FMM",
]

# Цвета BGR для каждой точки
KP_COLORS = [
    (0,   0,   230),   # L_TRC  — красный
    (0,   200, 0),     # R_TRC  — зелёный
    (0,   215, 255),   # L_ACE  — жёлтый
    (220, 80,  0),     # R_ACE  — синий
    (180, 0,   180),   # L_FHC  — пурпурный
    (0,   180, 180),   # R_FHC  — циан
    (255, 140, 0),     # L_FMM  — оранжевый
    (100, 100, 255),   # R_FMM  — розовый
]

RADIUS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_image(fname: str, search_dirs: list[Path] | None = None) -> Path | None:
    """Ищет файл изображения в нескольких директориях."""
    dirs = search_dirs or IMAGE_DIRS
    for d in dirs:
        p = d / fname
        if p.exists():
            return p
    # Рекурсивный поиск как запасной вариант
    for d in dirs:
        for p in d.rglob(fname):
            return p
    return None


def draw_keypoints(img: np.ndarray, keypoints: list[dict]) -> np.ndarray:
    """Рисует ключевые точки на изображении."""
    out = img.copy()
    for kp in keypoints:
        x, y, v = int(kp["x"]), int(kp["y"]), kp["visible"]
        if v == 0:
            continue
        idx = kp["idx"]
        color = KP_COLORS[idx % len(KP_COLORS)]
        cv2.circle(out, (x, y), RADIUS, color, -1)
        cv2.circle(out, (x, y), RADIUS + 2, (255, 255, 255), 1)
        cv2.putText(out, str(idx + 1), (x + 10, y - 5),
                    FONT, 0.55, color, 2, cv2.LINE_AA)
    return out


def draw_roi(img: np.ndarray, roi: dict | None) -> np.ndarray:
    """Рисует ROI-прямоугольник на изображении."""
    if roi is None:
        return img
    out = img.copy()
    x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return out


def scan_image_dir(base_dir: Path) -> dict[str, tuple[Path, str]]:
    """
    Сканирует директорию с изображениями и возвращает {filename: (abs_path, split)}.

    Ожидаемая структура:
        base_dir/
            train/normal/
            train/pathology/
            test/images/

    split = "normal" | "pathology" | "test"
    """
    result = {}
    suffixes = {".png", ".jpg", ".jpeg"}

    for split_name, subdir in [("normal", "train/normal"),
                                ("pathology", "train/pathology"),
                                ("test", "test/images")]:
        d = base_dir / subdir
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in suffixes:
                result[p.name] = (p, split_name)

    return result

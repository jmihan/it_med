"""
Обрезка изображений по ROI (область интереса — таз).

Сканирует data/processed/ напрямую (без зависимости от valid_images.json).
Для каждого изображения ищет ROI в порядке приоритета:
  1. Ground-truth rect из VIA-аннотаций
  2. Предсказания из predicted_roi.json
  3. Детекция YOLO ROI-моделью на лету (если --detect)
  4. Копирование полного изображения (если --fallback-full)

Результат сохраняется с сохранением структуры:
  output_dir/train/normal/
  output_dir/train/pathology/
  output_dir/test/images/

Использование:
  python scripts/crop_by_roi.py
  python scripts/crop_by_roi.py --padding 30 --fallback-full
  python scripts/crop_by_roi.py --detect --roi-weights weights/hip_roi_v1.pt
  python scripts/crop_by_roi.py --input-dir data/processed --output-dir data/processed_cropped
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._utils import KP_DIR, scan_image_dir


# ── Загрузка ground-truth rect из VIA JSON ──────────────────────────────────

def load_ground_truth_rects() -> dict[str, tuple[int, int, int, int]]:
    """Возвращает {filename: (x, y, w, h)} из ручной разметки."""
    rects = {}
    for ann_file in ["annotations_norm.json", "annotations_patolog.json",
                     "annotations_test.json"]:
        p = KP_DIR / ann_file
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for record in raw.values():
            fname = record["filename"]
            for region in record.get("regions", []):
                sa = region.get("shape_attributes", {})
                if sa.get("name") == "rect":
                    rects[fname] = (int(sa["x"]), int(sa["y"]),
                                    int(sa["width"]), int(sa["height"]))
                    break
    return rects


# ── Загрузка predicted_roi.json ──────────────────────────────────────────────

def load_predicted_rects() -> dict[str, tuple[int, int, int, int]]:
    """Возвращает {filename: (x, y, w, h)} из предсказаний."""
    p = KP_DIR / "predicted_roi.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for fname, data in raw.items():
        roi = data.get("roi")
        if roi:
            result[fname] = (int(roi["x"]), int(roi["y"]),
                             int(roi["w"]), int(roi["h"]))
    return result


# ── Детекция ROI моделью YOLO ────────────────────────────────────────────────

def detect_roi_yolo(model, img_path: Path) -> tuple[int, int, int, int] | None:
    """Запускает YOLO-детектор и возвращает (x, y, w, h) или None."""
    results = model.predict(source=str(img_path), conf=0.25, verbose=False)
    res = results[0]
    if res.boxes is not None and len(res.boxes) > 0:
        best = res.boxes[res.boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    return None


# ── Обрезка ──────────────────────────────────────────────────────────────────

def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int,
               padding: int) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return img[y1:y2, x1:x2]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Обрезка изображений по ROI")
    parser.add_argument("--input-dir", default="data/processed",
                        help="Директория с исходными изображениями")
    parser.add_argument("--output-dir", default="data/processed_cropped",
                        help="Директория для сохранения обрезанных изображений")
    parser.add_argument("--padding", type=int, default=20,
                        help="Доп. отступ вокруг ROI в пикселях")
    parser.add_argument("--fallback-full", action="store_true",
                        help="Копировать полное изображение если ROI не найден")
    parser.add_argument("--detect", action="store_true",
                        help="Запускать YOLO-детектор для изображений без аннотации")
    parser.add_argument("--roi-weights", default="weights/hip_roi_v1.pt",
                        help="Путь к весам YOLO ROI-детектора (для --detect)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Сканируем директорию с изображениями
    images = scan_image_dir(input_dir)
    if not images:
        print(f"Ошибка: нет изображений в {input_dir}")
        return

    # Загружаем ROI из разных источников
    gt_rects = load_ground_truth_rects()
    pred_rects = load_predicted_rects()

    # Ground-truth имеет приоритет над предсказаниями
    all_rects = {**pred_rects, **gt_rects}

    # Опциональная модель для детекции на лету
    yolo_model = None
    if args.detect:
        roi_weights = Path(args.roi_weights)
        if roi_weights.exists():
            from ultralytics import YOLO
            yolo_model = YOLO(str(roi_weights))
            print(f"YOLO ROI-детектор загружен: {roi_weights}")
        else:
            print(f"[WARN] Веса ROI-детектора не найдены: {roi_weights}")

    print(f"Всего изображений:   {len(images)}")
    print(f"  Ground truth ROI:  {len(gt_rects)}")
    print(f"  Predicted ROI:     {len(pred_rects)}")
    print(f"  Покрыто rect:      {len(all_rects)}")

    # Создаём выходные директории
    for d in ["train/normal", "train/pathology", "test/images"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    ok = detected = fallback = no_rect = error = 0

    for fname, (src_path, split) in images.items():
        # Определяем выходной путь
        if split == "normal":
            dst = output_dir / "train" / "normal" / fname
        elif split == "pathology":
            dst = output_dir / "train" / "pathology" / fname
        else:
            dst = output_dir / "test" / "images" / fname

        # Ищем ROI
        rect = all_rects.get(fname)

        # Пробуем детекцию на лету
        if rect is None and yolo_model is not None:
            rect = detect_roi_yolo(yolo_model, src_path)
            if rect is not None:
                detected += 1

        if rect is not None:
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"  [ERROR] не удалось открыть: {src_path}")
                error += 1
                continue
            x, y, w, h = rect
            cropped = crop_image(img, x, y, w, h, args.padding)
            cv2.imwrite(str(dst), cropped)
            ok += 1
        elif args.fallback_full:
            shutil.copy2(src_path, dst)
            fallback += 1
        else:
            print(f"  [WARN] нет ROI для: {fname}")
            no_rect += 1

    print(f"\nГотово:")
    print(f"  Обрезано:          {ok}")
    if detected:
        print(f"  Детектировано:     {detected}")
    if fallback:
        print(f"  Скопировано целиком: {fallback}")
    if no_rect:
        print(f"  Без ROI (пропущено): {no_rect}")
    if error:
        print(f"  Ошибок:            {error}")
    print(f"Результат: {output_dir}")


if __name__ == "__main__":
    main()

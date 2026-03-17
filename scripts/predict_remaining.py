"""
Разметка изображений, не вошедших в обучающую выборку аннотаций,
а также всех тестовых изображений с помощью обученных моделей.

Определяет "оставшиеся" изображения как:
  valid_images.json (все 160 корректных) МИНУС аннотированные 45+45 (train)
  + все 24 тестовых изображения (предсказываются заново)

Запускает:
  1. YOLO Pose -> predicted_keypoints.json
  2. YOLO Detection (ROI) -> predicted_roi.json
  3. Визуализацию -> data/keypoints/predicted_viz/ (очищается перед записью)

Использование:
  python scripts/predict_remaining.py
  python scripts/predict_remaining.py --pose-weights runs/keypoints/hip_pose_v1/weights/best.pt
                                      --roi-weights  runs/roi/hip_roi_v1/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._utils import (
    KP_DIR, KEYPOINT_NAMES, PROJECT_ROOT as ROOT,
    draw_keypoints, draw_roi, scan_image_dir,
)

PROCESSED    = ROOT / "data" / "processed"

DEFAULT_POSE_WEIGHTS = ROOT / "runs" / "keypoints" / "hip_pose_v1" / "weights" / "best.pt"
DEFAULT_ROI_WEIGHTS  = ROOT / "runs" / "roi" / "hip_roi_v1" / "weights" / "best.pt"

OUT_KP_JSON  = KP_DIR / "predicted_keypoints.json"
OUT_ROI_JSON = KP_DIR / "predicted_roi.json"
VIZ_DIR      = KP_DIR / "predicted_viz"


# ── Вспомогательные функции ──────────────────────────────────────────────────

def load_annotated_names() -> set[str]:
    """Имена изображений, вошедших в обучающую выборку аннотаций."""
    names = set()
    for ann_file in ["annotations_norm.json", "annotations_patolog.json"]:
        p = KP_DIR / ann_file
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for record in raw.values():
            names.add(record["filename"])
    return names


def load_test_names() -> set[str]:
    test_file = KP_DIR / "annotations_test.json"
    if not test_file.exists():
        return set()
    with open(test_file, encoding="utf-8") as f:
        raw = json.load(f)
    return {record["filename"] for record in raw.values()}


# ── Предсказания ──────────────────────────────────────────────────────────────

def predict_keypoints(model: YOLO, image_paths: list[Path],
                      conf: float, imgsz: int) -> dict:
    results_dict = {}
    no_detect = []

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(source=str(img_path), conf=conf,
                                imgsz=imgsz, verbose=False)
        res = results[0]
        keypoints_data = []

        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            kps_xy   = res.keypoints.xy[0].cpu().numpy()
            kps_conf = res.keypoints.conf
            if kps_conf is not None:
                kps_conf = kps_conf[0].cpu().numpy()
            else:
                kps_conf = np.ones(len(kps_xy))

            for k, ((x, y), v) in enumerate(zip(kps_xy, kps_conf)):
                keypoints_data.append({
                    "idx":     k,
                    "name":    KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else str(k),
                    "x":       float(x),
                    "y":       float(y),
                    "visible": int(v > 0.3),
                    "conf":    float(v),
                })
        else:
            no_detect.append(img_path.name)

        results_dict[img_path.name] = {
            "file_path": str(img_path.relative_to(ROOT)),
            "keypoints": keypoints_data,
        }

        if i % 10 == 0 or i == len(image_paths):
            print(f"  pose {i}/{len(image_paths)}")

    if no_detect:
        print(f"  [WARN] Нет детекции точек ({len(no_detect)}): {no_detect}")

    return results_dict


def predict_roi(model: YOLO, image_paths: list[Path],
                conf: float, imgsz: int) -> dict:
    results_dict = {}
    no_detect = []

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(source=str(img_path), conf=conf,
                                imgsz=imgsz, verbose=False)
        res = results[0]
        roi_data = None

        if res.boxes is not None and len(res.boxes) > 0:
            # Берём bbox с наибольшей уверенностью
            best = res.boxes[res.boxes.conf.argmax()]
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
            roi_data = {
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "conf": float(best.conf[0]),
            }
        else:
            no_detect.append(img_path.name)

        results_dict[img_path.name] = {
            "file_path": str(img_path.relative_to(ROOT)),
            "roi": roi_data,
        }

        if i % 10 == 0 or i == len(image_paths):
            print(f"  roi  {i}/{len(image_paths)}")

    if no_detect:
        print(f"  [WARN] Нет детекции ROI ({len(no_detect)}): {no_detect}")

    return results_dict


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-weights", default=str(DEFAULT_POSE_WEIGHTS))
    parser.add_argument("--roi-weights",  default=str(DEFAULT_ROI_WEIGHTS))
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--imgsz",  type=int,   default=800)
    args = parser.parse_args()

    # Определяем какие изображения предсказывать:
    # train-оставшиеся = все train-картинки МИНУС аннотированные
    # test = все тестовые (предсказываются заново)
    all_images      = scan_image_dir(PROCESSED)
    annotated_names = load_annotated_names()

    remaining_paths = []
    test_paths = []
    for fname, (fpath, split) in all_images.items():
        if split == "test":
            test_paths.append(fpath)
        elif fname not in annotated_names:
            remaining_paths.append(fpath)

    print(f"Оставшиеся train-изображения: {len(remaining_paths)}")
    print(f"Тестовые изображения:          {len(test_paths)}")
    predict_paths = remaining_paths + test_paths
    print(f"Всего для предсказания:        {len(predict_paths)}")

    # ── YOLO Pose ────────────────────────────────────────────────────────────
    pose_weights = Path(args.pose_weights)
    if not pose_weights.exists():
        print(f"[ERROR] Веса YOLO Pose не найдены: {pose_weights}")
        print("Сначала обучите модель: python scripts/train_keypoints.py")
        return

    print(f"\nYOLO Pose: {pose_weights}")
    pose_model = YOLO(str(pose_weights))
    kp_results = predict_keypoints(pose_model, predict_paths, args.conf, args.imgsz)

    OUT_KP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_KP_JSON, "w", encoding="utf-8") as f:
        json.dump(kp_results, f, ensure_ascii=False, indent=2)
    detected = sum(1 for v in kp_results.values() if v["keypoints"])
    print(f"Ключевые точки: {detected}/{len(kp_results)} с детекцией -> {OUT_KP_JSON}")

    # ── YOLO ROI ─────────────────────────────────────────────────────────────
    roi_weights = Path(args.roi_weights)
    if not roi_weights.exists():
        print(f"[ERROR] Веса ROI-детектора не найдены: {roi_weights}")
        print("Сначала обучите модель: python scripts/train_roi.py")
        return

    print(f"\nYOLO ROI: {roi_weights}")
    roi_model  = YOLO(str(roi_weights))
    roi_results = predict_roi(roi_model, predict_paths, args.conf, args.imgsz)

    with open(OUT_ROI_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_results, f, ensure_ascii=False, indent=2)
    detected_roi = sum(1 for v in roi_results.values() if v["roi"])
    print(f"ROI: {detected_roi}/{len(roi_results)} с детекцией -> {OUT_ROI_JSON}")

    # ── Визуализация ──────────────────────────────────────────────────────────
    if VIZ_DIR.exists():
        for f in VIZ_DIR.iterdir():
            f.unlink()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nВизуализация -> {VIZ_DIR}")
    for img_path in predict_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        fname = img_path.name
        kp_data  = kp_results.get(fname, {}).get("keypoints", [])
        roi_data = roi_results.get(fname, {}).get("roi", None)

        vis = draw_keypoints(img, kp_data)
        vis = draw_roi(vis, roi_data)

        cv2.imwrite(str(VIZ_DIR / fname), vis)

    print(f"Готово: {len(list(VIZ_DIR.glob('*.png')))} визуализаций")


if __name__ == "__main__":
    main()

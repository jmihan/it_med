"""
Инференс YOLO Pose + YOLO ROI на изображениях.

Объединяет функционал predict_keypoints.py и predict_remaining.py.

Режимы:
  --mode all       — предсказание на всех изображениях в --images-dir
  --mode remaining — предсказание только на изображениях, не вошедших в аннотации
                     (нужно для дополнения уже аннотированных данных)

Использование:
  python scripts/predict.py --mode all
  python scripts/predict.py --mode remaining --annotation-dir data/annotations
  python scripts/predict.py --mode all --pose-weights weights/hip_keypoints_v1.pt --draw
  python scripts/predict.py --mode all --no-roi
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
    KEYPOINT_NAMES, PROJECT_ROOT as ROOT,
    draw_keypoints, draw_roi, scan_image_dir,
)

DEFAULT_POSE_WEIGHTS = ROOT / "weights" / "hip_keypoints_v1.pt"
DEFAULT_ROI_WEIGHTS  = ROOT / "weights" / "hip_roi_v1.pt"
DEFAULT_IMAGES_DIR   = ROOT / "data" / "processed"
DEFAULT_ANN_DIR      = ROOT / "data" / "annotations"
DEFAULT_OUT_KP_JSON  = ROOT / "data" / "keypoints" / "predicted_keypoints.json"
DEFAULT_OUT_ROI_JSON = ROOT / "data" / "keypoints" / "predicted_roi.json"
DEFAULT_VIZ_DIR      = ROOT / "data" / "keypoints" / "predicted_viz"


# ── Построение списка изображений ──────────────────────────────────────────

def load_annotated_names(annotation_dir: Path) -> set:
    """Имена изображений, вошедших в обучающую выборку аннотаций."""
    names = set()
    for ann_file in ["annotations_norm.json", "annotations_patolog.json"]:
        p = annotation_dir / ann_file
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for record in raw.values():
            names.add(record["filename"])
    return names


def build_predict_paths(mode: str, images_dir: Path, annotation_dir: Path) -> list:
    """Возвращает список Path для предсказания в зависимости от режима."""
    if mode == "all":
        paths = sorted(
            p for p in images_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        print(f"Режим 'all': найдено {len(paths)} изображений в {images_dir}")
        return paths

    # mode == "remaining"
    all_images = scan_image_dir(images_dir)
    annotated_names = load_annotated_names(annotation_dir)

    remaining_paths = []
    test_paths = []
    for fname, (fpath, split) in all_images.items():
        if split == "test":
            test_paths.append(fpath)
        elif fname not in annotated_names:
            remaining_paths.append(fpath)

    print(f"Режим 'remaining': оставшиеся train={len(remaining_paths)}, test={len(test_paths)}")
    return remaining_paths + test_paths


# ── Инференс YOLO Pose ─────────────────────────────────────────────────────

def predict_keypoints(model: YOLO, image_paths: list, conf: float, imgsz: int) -> dict:
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
        print(f"  [WARN] Нет детекции точек ({len(no_detect)}): {no_detect[:5]}{'...' if len(no_detect) > 5 else ''}")

    return results_dict


# ── Инференс YOLO ROI ──────────────────────────────────────────────────────

def predict_roi(model: YOLO, image_paths: list, conf: float, imgsz: int) -> dict:
    results_dict = {}
    no_detect = []

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(source=str(img_path), conf=conf,
                                imgsz=imgsz, verbose=False)
        res = results[0]
        roi_data = None

        if res.boxes is not None and len(res.boxes) > 0:
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
        print(f"  [WARN] Нет детекции ROI ({len(no_detect)}): {no_detect[:5]}{'...' if len(no_detect) > 5 else ''}")

    return results_dict


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Инференс YOLO Pose + YOLO ROI на изображениях"
    )
    parser.add_argument("--mode", choices=["all", "remaining"], default="remaining",
                        help="'all' — все изображения в --images-dir, "
                             "'remaining' — только не аннотированные (default: remaining)")
    parser.add_argument("--pose-weights", default=str(DEFAULT_POSE_WEIGHTS),
                        help="Путь к весам YOLO Pose (default: weights/hip_keypoints_v1.pt)")
    parser.add_argument("--roi-weights",  default=str(DEFAULT_ROI_WEIGHTS),
                        help="Путь к весам YOLO ROI (default: weights/hip_roi_v1.pt)")
    parser.add_argument("--images-dir",   default=str(DEFAULT_IMAGES_DIR),
                        help="Директория с изображениями для режима 'all' (default: data/processed)")
    parser.add_argument("--annotation-dir", default=str(DEFAULT_ANN_DIR),
                        help="Директория с VIA JSON аннотациями для режима 'remaining' "
                             "(default: data/annotations/)")
    parser.add_argument("--out-kp-json",  default=str(DEFAULT_OUT_KP_JSON),
                        help="Выходной JSON с ключевыми точками")
    parser.add_argument("--out-roi-json", default=str(DEFAULT_OUT_ROI_JSON),
                        help="Выходной JSON с ROI-боксами")
    parser.add_argument("--viz-dir",      default=str(DEFAULT_VIZ_DIR),
                        help="Директория для PNG-визуализаций (используется с --draw)")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Порог уверенности (default: 0.25)")
    parser.add_argument("--imgsz",  type=int,   default=800,
                        help="Размер входного изображения (default: 800)")
    parser.add_argument("--draw",   action="store_true",
                        help="Сохранить PNG-визуализации с точками и ROI")
    parser.add_argument("--no-roi", action="store_true",
                        help="Пропустить инференс YOLO ROI")
    args = parser.parse_args()

    predict_paths = build_predict_paths(
        args.mode,
        Path(args.images_dir),
        Path(args.annotation_dir),
    )

    if not predict_paths:
        print("[WARN] Нет изображений для предсказания.")
        return

    print(f"Всего для предсказания: {len(predict_paths)}")

    # ── YOLO Pose ──────────────────────────────────────────────────────────
    pose_weights = Path(args.pose_weights)
    if not pose_weights.exists():
        print(f"[ERROR] Веса YOLO Pose не найдены: {pose_weights}")
        print("Обучите модель: python scripts/train_keypoints.py")
        return

    print(f"\nYOLO Pose: {pose_weights}")
    pose_model = YOLO(str(pose_weights))
    kp_results = predict_keypoints(pose_model, predict_paths, args.conf, args.imgsz)

    out_kp = Path(args.out_kp_json)
    out_kp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_kp, "w", encoding="utf-8") as f:
        json.dump(kp_results, f, ensure_ascii=False, indent=2)
    detected = sum(1 for v in kp_results.values() if v["keypoints"])
    print(f"Ключевые точки: {detected}/{len(kp_results)} с детекцией -> {out_kp}")

    roi_results = {}

    # ── YOLO ROI ───────────────────────────────────────────────────────────
    if not args.no_roi:
        roi_weights = Path(args.roi_weights)
        if not roi_weights.exists():
            print(f"[WARN] Веса YOLO ROI не найдены: {roi_weights} — пропускаю ROI")
        else:
            print(f"\nYOLO ROI: {roi_weights}")
            roi_model = YOLO(str(roi_weights))
            roi_results = predict_roi(roi_model, predict_paths, args.conf, args.imgsz)

            out_roi = Path(args.out_roi_json)
            with open(out_roi, "w", encoding="utf-8") as f:
                json.dump(roi_results, f, ensure_ascii=False, indent=2)
            detected_roi = sum(1 for v in roi_results.values() if v["roi"])
            print(f"ROI: {detected_roi}/{len(roi_results)} с детекцией -> {out_roi}")

    # ── Визуализация ───────────────────────────────────────────────────────
    if args.draw:
        viz_dir = Path(args.viz_dir)
        if viz_dir.exists():
            for f in viz_dir.iterdir():
                f.unlink()
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nВизуализация -> {viz_dir}")
        for img_path in predict_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            fname = img_path.name
            kp_data  = kp_results.get(fname, {}).get("keypoints", [])
            roi_data = roi_results.get(fname, {}).get("roi") if roi_results else None
            vis = draw_keypoints(img, kp_data)
            vis = draw_roi(vis, roi_data)
            cv2.imwrite(str(viz_dir / fname), vis)

        saved = len(list(viz_dir.glob("*.png"))) + len(list(viz_dir.glob("*.jpg")))
        print(f"Готово: {saved} визуализаций")


if __name__ == "__main__":
    main()

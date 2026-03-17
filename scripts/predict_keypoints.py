"""
Инференс YOLO Pose на всех изображениях из data/processed.
Сохраняет координаты точек в JSON (VIA-совместимый формат)
и опционально рисует визуализацию.

Использование:
    python scripts/predict_keypoints.py
    python scripts/predict_keypoints.py --draw         # сохранить PNG с точками
    python scripts/predict_keypoints.py --conf 0.3    # порог уверенности
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

from scripts._utils import KEYPOINT_NAMES, PROJECT_ROOT as ROOT, draw_keypoints

WEIGHTS     = ROOT / "runs" / "keypoints" / "hip_pose_v1" / "weights" / "best.pt"
IMAGES_DIR  = ROOT / "data" / "processed"
OUT_JSON    = ROOT / "data" / "keypoints" / "predicted_keypoints.json"
OUT_VIZ_DIR = ROOT / "data" / "keypoints" / "predicted_viz"


def predict_all(weights: Path, images_dir: Path,
                conf: float, draw: bool, imgsz: int) -> dict:
    model = YOLO(str(weights))

    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    print(f"Найдено изображений: {len(image_paths)}")

    if draw:
        OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    results_dict = {}   # filename -> список точек
    no_detect = []

    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )
        res = results[0]

        keypoints_data = []

        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            # Берём предсказание с наибольшей уверенностью (первый bbox)
            kps_xy  = res.keypoints.xy[0].cpu().numpy()   # (8, 2)
            kps_conf = res.keypoints.conf                  # может быть None
            if kps_conf is not None:
                kps_conf = kps_conf[0].cpu().numpy()       # (8,)
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
            "file_path":  str(img_path.relative_to(ROOT)),
            "keypoints":  keypoints_data,
        }

        # Визуализация
        if draw and keypoints_data:
            img = cv2.imread(str(img_path))
            vis = draw_keypoints(img, keypoints_data)
            out_path = OUT_VIZ_DIR / img_path.name
            cv2.imwrite(str(out_path), vis)

        if i % 20 == 0 or i == len(image_paths):
            print(f"  {i}/{len(image_paths)} обработано")

    if no_detect:
        print(f"\n[WARN] Нет детекции ({len(no_detect)} шт.):")
        for n in no_detect:
            print(f"  {n}")

    return results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default=str(WEIGHTS))
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--out",        default=str(OUT_JSON))
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--draw",       action="store_true",
                        help="Сохранить PNG с нанесёнными точками")
    args = parser.parse_args()

    print(f"Модель:  {args.weights}")
    print(f"Данные:  {args.images_dir}")
    print(f"conf:    {args.conf}")

    results = predict_all(
        weights=Path(args.weights),
        images_dir=Path(args.images_dir),
        conf=args.conf,
        draw=args.draw,
        imgsz=args.imgsz,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    detected = sum(1 for v in results.values() if v["keypoints"])
    print(f"\nГотово: {detected}/{len(results)} с ключевыми точками")
    print(f"JSON:   {out_path}")
    if args.draw:
        print(f"Viz:    {OUT_VIZ_DIR}")


if __name__ == "__main__":
    main()

"""
Конвертер аннотаций VIA (VGG Image Annotator) -> YOLO Pose.

Читает три файла разметки (норма, патология, тест), извлекает только
point-регионы (игнорирует rect), конвертирует в YOLO Pose формат
и делит на train/val.

Входные файлы (в data/keypoints/):
  annotations_norm.json    — 45 изображений нормы
  annotations_patolog.json — 45 изображений патологии
  annotations_test.json    — 24 тестовых изображения

Выходной формат YOLO Pose .txt:
  <class_id> <cx> <cy> <w> <h> <x0> <y0> <v0> ... <x9> <y9> <v9>
  Все координаты нормализованы в [0, 1].

Порядок точек (10 штук):
  0: L_TRC  1: R_TRC  2: L_ACE  3: R_ACE
  4: L_FHC  5: R_FHC  6: L_FMM  7: R_FMM
  8: L_FMP  9: R_FMP

Использование:
  python scripts/convert_annotations.py
  python scripts/convert_annotations.py --val-split 0.2 --seed 42
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._utils import KP_DIR, KEYPOINT_NAMES, find_image

DEFAULT_NORM    = KP_DIR / "annotations_norm.json"
DEFAULT_PATOLOG = KP_DIR / "annotations_patolog.json"
DEFAULT_TEST    = KP_DIR / "annotations_test.json"
DEFAULT_OUT     = KP_DIR

NUM_KEYPOINTS  = len(KEYPOINT_NAMES)
CLASS_ID       = 0
BBOX_PADDING   = 0.15


# ── Чтение VIA JSON (только point) ───────────────────────────────────────────
def load_via_json(path: Path) -> dict[str, list[tuple[int, int]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for record in raw.values():
        fname = record["filename"]
        pts = []
        for region in record.get("regions", []):
            sa = region.get("shape_attributes", {})
            if sa.get("name") == "point":
                pts.append((int(sa["cx"]), int(sa["cy"])))
        if pts:
            result[fname] = pts
    return result


# ── Чтение VIA CSV (только point) ────────────────────────────────────────────
def load_via_csv(path: Path) -> dict[str, list[tuple[int, int]]]:
    rows: dict[str, dict[int, tuple]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            sa = json.loads(row["region_shape_attributes"])
            if sa.get("name") != "point":
                continue
            rid = int(row["region_id"])
            rows.setdefault(fname, {})[rid] = (int(sa["cx"]), int(sa["cy"]))
    return {
        fname: [pts[i] for i in sorted(pts)]
        for fname, pts in rows.items()
    }


def load_via(path: Path) -> dict[str, list[tuple[int, int]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_via_json(path)
    if suffix == ".csv":
        return load_via_csv(path)
    raise ValueError(f"Неизвестный формат '{suffix}'. Ожидается .json или .csv")



# ── bbox из точек ─────────────────────────────────────────────────────────────
def bbox_from_points(pts: list[tuple[int, int]],
                     img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    bw = x1 - x0
    bh = y1 - y0
    pad_x = bw * BBOX_PADDING
    pad_y = bh * BBOX_PADDING

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img_w, x1 + pad_x)
    y1 = min(img_h, y1 + pad_y)

    cx = ((x0 + x1) / 2) / img_w
    cy = ((y0 + y1) / 2) / img_h
    nw = (x1 - x0) / img_w
    nh = (y1 - y0) / img_h
    return cx, cy, nw, nh


# ── Строка YOLO Pose ──────────────────────────────────────────────────────────
def to_yolo_line(pts: list[tuple[int, int]], img_w: int, img_h: int) -> str:
    cx, cy, nw, nh = bbox_from_points(pts, img_w, img_h)
    parts = [f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    for i in range(NUM_KEYPOINTS):
        if i < len(pts):
            kx = max(0.0, min(1.0, pts[i][0] / img_w))
            ky = max(0.0, min(1.0, pts[i][1] / img_h))
            parts.append(f"{kx:.6f} {ky:.6f} 2")
        else:
            parts.append("0.000000 0.000000 0")
    return " ".join(parts)


# ── Конвертация и сплит ───────────────────────────────────────────────────────
def convert_and_split(
    annotations: list[tuple[str, list]],
    output_dir: Path,
    val_split: float,
    seed: int,
) -> dict:
    random.seed(seed)
    items = annotations[:]
    random.shuffle(items)

    n_val  = max(1, round(len(items) * val_split))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    stats = {}
    for split, data in splits.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok = skip = 0
        for fname, pts in data:
            src = find_image(fname)
            if src is None:
                print(f"  [WARN] не найдено: {fname}")
                skip += 1
                continue

            with Image.open(src) as im:
                w, h = im.size

            dst_img = img_out / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            line = to_yolo_line(pts, w, h)
            lbl_path = lbl_out / (src.stem + ".txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(line + "\n")

            ok += 1

        stats[split] = (ok, skip)

    return stats


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Конвертер VIA JSON/CSV -> YOLO Pose"
    )
    parser.add_argument("--norm",       default=str(DEFAULT_NORM))
    parser.add_argument("--patolog",    default=str(DEFAULT_PATOLOG))
    parser.add_argument("--test",       default=str(DEFAULT_TEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    all_data: dict[str, list] = {}
    for label, path_str in [("norm", args.norm),
                             ("patolog", args.patolog),
                             ("test", args.test)]:
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] файл не найден: {p}")
            continue
        data = load_via(p)
        print(f"{label}: {p.name}  ->  {len(data)} изображений, "
              f"{sum(len(v) for v in data.values())} точек")
        # Проверка количества точек
        for fname, pts in data.items():
            if len(pts) != NUM_KEYPOINTS:
                print(f"  [WARN] {fname}: {len(pts)} точек (ожидалось {NUM_KEYPOINTS})")
        all_data.update(data)

    print(f"\nВсего: {len(all_data)} изображений")

    stats = convert_and_split(list(all_data.items()), output_dir,
                               args.val_split, args.seed)

    print("\nРезультат:")
    for split, (ok, skip) in stats.items():
        print(f"  {split:5s}: {ok} сконвертировано, {skip} пропущено")


if __name__ == "__main__":
    main()

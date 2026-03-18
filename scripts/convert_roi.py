"""
Конвертер аннотаций VIA (rect-регионы) -> YOLO Detection датасет.

Читает rect-разметку из трёх файлов (norm, patolog, test),
конвертирует в YOLO Detection формат для обучения детектора ROI.

Входные файлы (по умолчанию в data/annotations/):
  annotations_norm.json    — изображения нормы
  annotations_patolog.json — изображения патологии
  annotations_test.json    — тестовые изображения

Выходной формат YOLO Detection .txt:
  <class_id> <cx> <cy> <w> <h>   (нормализованные [0,1])

Использование:
  python scripts/convert_roi.py
  python scripts/convert_roi.py --annotation-dir data/annotations --val-split 0.2
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

from scripts._utils import find_image, PROJECT_ROOT as ROOT

DEFAULT_ANN_DIR = ROOT / "data" / "annotations"
DEFAULT_OUT     = ROOT / "data" / "roi"

CLASS_ID = 0   # единственный класс — roi


# ── Чтение VIA JSON (только rect) ────────────────────────────────────────────
def load_via_json_rect(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for record in raw.values():
        fname = record["filename"]
        for region in record.get("regions", []):
            sa = region.get("shape_attributes", {})
            if sa.get("name") == "rect":
                result[fname] = (int(sa["x"]), int(sa["y"]),
                                 int(sa["width"]), int(sa["height"]))
                break
    return result


# ── Чтение VIA CSV (только rect) ─────────────────────────────────────────────
def load_via_csv_rect(path: Path) -> dict[str, tuple[int, int, int, int]]:
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if fname in result:
                continue
            sa = json.loads(row["region_shape_attributes"])
            if sa.get("name") == "rect":
                result[fname] = (int(sa["x"]), int(sa["y"]),
                                 int(sa["width"]), int(sa["height"]))
    return result


def load_via_rect(path: Path) -> dict[str, tuple[int, int, int, int]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_via_json_rect(path)
    if suffix == ".csv":
        return load_via_csv_rect(path)
    raise ValueError(f"Неизвестный формат '{suffix}'. Ожидается .json или .csv")



# ── Конвертация и сплит ───────────────────────────────────────────────────────
def convert_and_split(
    annotations: list[tuple[str, tuple]],
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
        for fname, (rx, ry, rw, rh) in data:
            src = find_image(fname)
            if src is None:
                print(f"  [WARN] не найдено: {fname}")
                skip += 1
                continue

            with Image.open(src) as im:
                img_w, img_h = im.size

            cx = max(0.0, min(1.0, (rx + rw / 2) / img_w))
            cy = max(0.0, min(1.0, (ry + rh / 2) / img_h))
            nw = max(0.0, min(1.0, rw / img_w))
            nh = max(0.0, min(1.0, rh / img_h))

            dst_img = img_out / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            lbl_path = lbl_out / (src.stem + ".txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            ok += 1

        stats[split] = (ok, skip)

    return stats


# ── dataset.yaml ──────────────────────────────────────────────────────────────
def write_dataset_yaml(output_dir: Path):
    yaml_path = output_dir / "dataset.yaml"
    content = (
        f"path: {output_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"\nnc: 1\n"
        f"names: ['roi']\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"dataset.yaml -> {yaml_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Конвертер VIA rect-аннотаций -> YOLO Detection датасет"
    )
    parser.add_argument("--annotation-dir", default=str(DEFAULT_ANN_DIR),
                        help="Директория с VIA JSON файлами (default: data/annotations/)")
    parser.add_argument("--norm",       default=None,
                        help="Переопределить путь к annotations_norm.json")
    parser.add_argument("--patolog",    default=None,
                        help="Переопределить путь к annotations_patolog.json")
    parser.add_argument("--test",       default=None,
                        help="Переопределить путь к annotations_test.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    norm_path    = Path(args.norm)    if args.norm    else ann_dir / "annotations_norm.json"
    patolog_path = Path(args.patolog) if args.patolog else ann_dir / "annotations_patolog.json"
    test_path    = Path(args.test)    if args.test    else ann_dir / "annotations_test.json"

    output_dir = Path(args.output_dir)

    all_rects: dict[str, tuple] = {}
    for label, p in [("norm", norm_path),
                     ("patolog", patolog_path),
                     ("test", test_path)]:
        if not p.exists():
            print(f"[WARN] файл не найден: {p}")
            continue
        rects = load_via_rect(p)
        print(f"{label}: {p.name}  ->  {len(rects)} прямоугольников")
        all_rects.update(rects)

    print(f"\nВсего: {len(all_rects)} изображений с rect-разметкой")

    stats = convert_and_split(list(all_rects.items()), output_dir,
                               args.val_split, args.seed)

    print("\nРезультат:")
    for split, (ok, skip) in stats.items():
        print(f"  {split:5s}: {ok} сконвертировано, {skip} пропущено")

    write_dataset_yaml(output_dir)


if __name__ == "__main__":
    main()

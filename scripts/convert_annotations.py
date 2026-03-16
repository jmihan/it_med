"""
Конвертер аннотаций VIA (VGG Image Annotator) → YOLO Pose.

Поддерживаемые входные форматы:
  - VIA JSON  (File → Export Annotations → JSON)
  - VIA CSV   (File → Export Annotations → CSV)

Выходной формат YOLO Pose .txt (один файл на изображение):
  <class_id> <cx> <cy> <w> <h> <x0> <y0> <v0> ... <x7> <y7> <v7>
  Все координаты нормализованы в [0, 1].

Структура выходной директории:
  output_dir/
    train/images/   train/labels/
    val/images/     val/labels/

Порядок точек (8 штук, region_id 0..7 из VIA):
  0: L_TRC   1: R_TRC   2: L_ACE   3: R_ACE
  4: L_FHC   5: R_FHC   6: L_FMM   7: R_FMM

Использование:
  python scripts/convert_annotations.py
  python scripts/convert_annotations.py --val-split 0.25 --seed 0
  python scripts/convert_annotations.py \\
      --norm   data/keypoints/via_export_csv\\ (norm1).csv \\
      --patolog data/keypoints/via_export_csv\\ (patolog1).csv
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


# ── Конфигурация по умолчанию ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KP_DIR = ROOT / "data" / "keypoints"

DEFAULT_NORM    = KP_DIR / "via_export_json (norm1).json"
DEFAULT_PATOLOG = KP_DIR / "via_export_json (patolog1).json"
DEFAULT_IMAGES  = ROOT / "data" / "processed" / "train"
DEFAULT_OUT     = KP_DIR

KEYPOINT_NAMES = ["L_TRC", "R_TRC", "L_ACE", "R_ACE",
                  "L_FHC", "R_FHC", "L_FMM", "R_FMM"]
NUM_KEYPOINTS  = len(KEYPOINT_NAMES)
CLASS_ID       = 0          # один класс — pelvis
BBOX_PADDING   = 0.15       # отступ bbox вокруг точек


# ── Чтение VIA JSON ──────────────────────────────────────────────────────────
def load_via_json(path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    VIA JSON: {filename+size: {filename, regions:[{shape_attributes:{name,cx,cy}}]}}
    Возвращает {filename: [(cx, cy), ...]}
    """
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


# ── Чтение VIA CSV ───────────────────────────────────────────────────────────
def load_via_csv(path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    VIA CSV: filename, ..., region_id, region_shape_attributes (JSON-строка)
    Возвращает {filename: [(cx, cy), ...]} — точки в порядке region_id.
    """
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


# ── Универсальный загрузчик ──────────────────────────────────────────────────
def load_via(path: Path) -> dict[str, list[tuple[int, int]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_via_json(path)
    if suffix == ".csv":
        return load_via_csv(path)
    raise ValueError(f"Неизвестный формат '{path.suffix}'. Ожидается .json или .csv")


# ── Поиск изображения ────────────────────────────────────────────────────────
def find_image(fname: str, images_dir: Path) -> Path | None:
    direct = images_dir / fname
    if direct.exists():
        return direct
    for p in images_dir.rglob(fname):
        return p
    return None


# ── Размер изображения ───────────────────────────────────────────────────────
def image_size(path: Path) -> tuple[int, int]:
    """Возвращает (width, height) без загрузки всего файла."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size   # (w, h)


# ── bbox из точек ────────────────────────────────────────────────────────────
def bbox_from_points(pts: list[tuple[int, int]],
                     img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Возвращает (cx, cy, w, h) нормализованные [0,1] с отступом."""
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


# ── Строка YOLO Pose ─────────────────────────────────────────────────────────
def to_yolo_line(pts: list[tuple[int, int]], img_w: int, img_h: int) -> str:
    cx, cy, nw, nh = bbox_from_points(pts, img_w, img_h)
    parts = [f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]

    for i in range(NUM_KEYPOINTS):
        if i < len(pts):
            kx = max(0.0, min(1.0, pts[i][0] / img_w))
            ky = max(0.0, min(1.0, pts[i][1] / img_h))
            parts.append(f"{kx:.6f} {ky:.6f} 2")   # v=2: видимая
        else:
            parts.append("0.000000 0.000000 0")

    return " ".join(parts)


# ── Конвертация и сплит ──────────────────────────────────────────────────────
def convert_and_split(
    annotations: list[tuple[str, list]],   # [(filename, [(cx,cy)...]), ...]
    images_dir: Path,
    output_dir: Path,
    val_split: float,
    seed: int,
):
    random.seed(seed)
    items = annotations[:]
    random.shuffle(items)

    n_val   = max(1, round(len(items) * val_split))
    splits  = {"val": items[:n_val], "train": items[n_val:]}

    stats = {}
    for split, data in splits.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok = skip = 0
        for fname, pts in data:
            src = find_image(fname, images_dir)
            if src is None:
                print(f"  [WARN] не найдено: {fname}")
                skip += 1
                continue

            w, h = image_size(src)

            # Копируем изображение
            dst_img = img_out / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            # Записываем label
            line = to_yolo_line(pts, w, h)
            lbl_path = lbl_out / (src.stem + ".txt")
            with open(lbl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            ok += 1

        stats[split] = (ok, skip)

    return stats


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Конвертер VIA JSON/CSV → YOLO Pose"
    )
    parser.add_argument("--norm",    default=str(DEFAULT_NORM),
                        help="Файл аннотаций нормы (.json или .csv)")
    parser.add_argument("--patolog", default=str(DEFAULT_PATOLOG),
                        help="Файл аннотаций патологии (.json или .csv)")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES),
                        help="Корневая папка с изображениями")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                        help="Выходная директория YOLO датасета")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    norm_path    = Path(args.norm)
    patolog_path = Path(args.patolog)
    images_dir   = Path(args.images_dir)
    output_dir   = Path(args.output_dir)

    # Загрузка
    print(f"Норма:     {norm_path.name}")
    norm = load_via(norm_path)
    print(f"  {len(norm)} изображений, {sum(len(v) for v in norm.values())} точек")

    print(f"Патология: {patolog_path.name}")
    pat = load_via(patolog_path)
    print(f"  {len(pat)} изображений, {sum(len(v) for v in pat.values())} точек")

    # Проверка количества точек
    all_data = list(norm.items()) + list(pat.items())
    for fname, pts in all_data:
        if len(pts) != NUM_KEYPOINTS:
            print(f"  [WARN] {fname}: {len(pts)} точек (ожидалось {NUM_KEYPOINTS})")

    # Конвертация
    print(f"\nКонвертация -> YOLO Pose (val={args.val_split*100:.0f}%, seed={args.seed}) ...")
    stats = convert_and_split(all_data, images_dir, output_dir, args.val_split, args.seed)

    print(f"\nРезультат:")
    for split, (ok, skip) in stats.items():
        print(f"  {split:5s}: {ok} сконвертировано, {skip} пропущено")

    print(f"\nСтруктура:")
    for split in ("train", "val"):
        print(f"  {output_dir / split / 'images'}")
        print(f"  {output_dir / split / 'labels'}")


if __name__ == "__main__":
    main()

"""
Читает разметку ключевых точек из VIA JSON или CSV,
делит на train/val, визуализирует.

Форматы VIA:
  JSON: {filename+size: {filename, size, regions: [{shape_attributes:{name,cx,cy}}]}}
  CSV:  filename, file_size, ..., region_shape_attributes (JSON-строка с cx, cy)

Использование:
    python scripts/split_and_visualize_keypoints.py
    python scripts/split_and_visualize_keypoints.py --val-ratio 0.25 --seed 0
    python scripts/split_and_visualize_keypoints.py --visualize-only
    python scripts/split_and_visualize_keypoints.py --no-show   # сохранить PNG
"""

import json
import csv
import random
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._utils import KP_DIR, find_image

# Входные файлы (VIA JSON или CSV — определяется автоматически)
NORM_FILE    = KP_DIR / "annotations_norm.json"
PATOLOG_FILE = KP_DIR / "annotations_patolog.json"

# Цвета для 8 ключевых точек
KP_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
]
# Подписи точек (по порядку разметки в VIA)
KP_LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8",
]


# ── Чтение VIA JSON ──────────────────────────────────────────────────────────
def load_via_json(path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Возвращает {filename: [(cx, cy), ...]} — список точек для каждого изображения.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    result = {}
    for record in raw.values():
        fname = record["filename"]
        points = []
        for region in record.get("regions", []):
            sa = region.get("shape_attributes", {})
            if sa.get("name") == "point":
                points.append((int(sa["cx"]), int(sa["cy"])))
        result[fname] = points
    return result


# ── Чтение VIA CSV ───────────────────────────────────────────────────────────
def load_via_csv(path: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Возвращает {filename: [(cx, cy), ...]} — список точек для каждого изображения.
    """
    result: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            sa = json.loads(row["region_shape_attributes"])
            if sa.get("name") == "point":
                result.setdefault(fname, []).append(
                    (int(sa["cx"]), int(sa["cy"]))
                )
    return result


# ── Универсальный загрузчик ──────────────────────────────────────────────────
def load_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_via_json(path)
    elif suffix == ".csv":
        return load_via_csv(path)
    else:
        raise ValueError(f"Неизвестный формат: {path.suffix}. Ожидается .json или .csv")


# ── Сплит ────────────────────────────────────────────────────────────────────
def split_annotations(
    data: dict[str, list],
    val_ratio: float,
    seed: int,
) -> tuple[dict, dict]:
    """Делит словарь {filename: points} на train и val."""
    filenames = list(data.keys())
    random.seed(seed)
    random.shuffle(filenames)

    n_val = max(1, round(len(filenames) * val_ratio))
    val_files = set(filenames[:n_val])

    train = {k: v for k, v in data.items() if k not in val_files}
    val   = {k: v for k, v in data.items() if k in val_files}
    return train, val


# ── Сохранение JSON (в формате VIA JSON для совместимости) ───────────────────
def save_via_json(data: dict[str, list[tuple[int, int]]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for fname, points in data.items():
        key = fname  # без размера — ОК для нашего загрузчика
        out[key] = {
            "filename": fname,
            "regions": [
                {
                    "shape_attributes": {"name": "point", "cx": cx, "cy": cy},
                    "region_attributes": {},
                }
                for cx, cy in points
            ],
            "file_attributes": {},
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено: {path}  ({len(data)} изобр.)")



# ── Визуализация ─────────────────────────────────────────────────────────────
def visualize(
    data: dict[str, list[tuple[int, int]]],
    title: str,
    cols: int = 4,
) -> plt.Figure | None:
    items = list(data.items())
    if not items:
        print(f"  [{title}] Нет данных.")
        return None

    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (fname, points) in enumerate(items):
        ax = axes[i]
        img_path = find_image(fname)

        if img_path is None:
            ax.text(
                0.5, 0.5, f"Файл не найден:\n{fname}",
                ha="center", va="center", fontsize=7, color="red",
                transform=ax.transAxes, wrap=True,
            )
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
            ax.imshow(img)

            for k, (cx, cy) in enumerate(points):
                color = KP_COLORS[k % len(KP_COLORS)]
                label = KP_LABELS[k] if k < len(KP_LABELS) else str(k + 1)
                ax.plot(cx, cy, "o", color=color, markersize=7,
                        markeredgecolor="white", markeredgewidth=0.8)
                ax.annotate(
                    label, (cx, cy),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=color, fontweight="bold",
                )

        short = fname if len(fname) <= 32 else "…" + fname[-30:]
        ax.set_title(f"{short}\n{len(points)} точек", fontsize=6.5)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Доля val (default: 0.2 → 4 из 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-only", action="store_true",
                        help="Читать уже готовые train/val JSON, не делать сплит")
    parser.add_argument("--no-show", action="store_true",
                        help="Не открывать окна, сохранить PNG в data/keypoints/viz/")
    args = parser.parse_args()

    train_norm_path   = KP_DIR / "train" / "norm.json"
    train_pat_path    = KP_DIR / "train" / "patolog.json"
    val_norm_path     = KP_DIR / "val"   / "norm.json"
    val_pat_path      = KP_DIR / "val"   / "patolog.json"

    if args.visualize_only:
        print("Чтение уже разделённых файлов ...")
        train_norm = load_annotations(train_norm_path)
        train_pat  = load_annotations(train_pat_path)
        val_norm   = load_annotations(val_norm_path)
        val_pat    = load_annotations(val_pat_path)
    else:
        # Загрузка
        print(f"Норма:    {NORM_FILE.name}")
        norm = load_annotations(NORM_FILE)
        print(f"  {len(norm)} изображений, {sum(len(v) for v in norm.values())} точек")

        print(f"Патология: {PATOLOG_FILE.name}")
        pat = load_annotations(PATOLOG_FILE)
        print(f"  {len(pat)} изображений, {sum(len(v) for v in pat.values())} точек")

        # Сплит
        print(f"\nРазделение (val={args.val_ratio*100:.0f}%, seed={args.seed}) ...")
        train_norm, val_norm = split_annotations(norm, args.val_ratio, args.seed)
        train_pat,  val_pat  = split_annotations(pat,  args.val_ratio, args.seed)

        print(f"  Норма:    train={len(train_norm)}, val={len(val_norm)}")
        print(f"  Патология: train={len(train_pat)}, val={len(val_pat)}")

        # Сохранение
        print("\nСохранение ...")
        save_via_json(train_norm, train_norm_path)
        save_via_json(val_norm,   val_norm_path)
        save_via_json(train_pat,  train_pat_path)
        save_via_json(val_pat,    val_pat_path)

    # Визуализация
    print("\nВизуализация ...")
    figs = [
        (visualize(train_norm, "TRAIN — Норма"),     "train_norm"),
        (visualize(val_norm,   "VAL — Норма"),       "val_norm"),
        (visualize(train_pat,  "TRAIN — Патология"), "train_patolog"),
        (visualize(val_pat,    "VAL — Патология"),   "val_patolog"),
    ]

    if args.no_show:
        viz_dir = KP_DIR / "viz"
        viz_dir.mkdir(exist_ok=True)
        for fig, name in figs:
            if fig is not None:
                out = viz_dir / f"{name}.png"
                fig.savefig(out, dpi=100, bbox_inches="tight")
                print(f"  Сохранено: {out}")
        plt.close("all")
    else:
        plt.show()

    print("Готово.")


if __name__ == "__main__":
    main()

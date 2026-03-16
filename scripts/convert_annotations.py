"""
Конвертер аннотаций из CVAT/COCO Keypoints в формат YOLO Pose.

Поддерживаемые входные форматы:
  - COCO Keypoints JSON (экспорт из CVAT / Label Studio)
  - CVAT for Images XML

Выходной формат:
  YOLO Pose .txt (один файл на изображение):
    <class_id> <cx> <cy> <w> <h> <x0> <y0> <v0> ... <x7> <y7> <v7>
  Все координаты нормализованы в [0, 1].

Использование:
  python scripts/convert_annotations.py \
      --input data/keypoints/annotations.json \
      --format coco \
      --images-dir data/processed/train \
      --output-dir data/keypoints \
      --val-split 0.2
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# Порядок ключевых точек для YOLO Pose
KEYPOINT_NAMES = [
    "L_TRC", "R_TRC", "L_ACE", "R_ACE",
    "L_FHC", "R_FHC", "L_FMM", "R_FMM",
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)


def parse_coco_keypoints(json_path: str) -> list:
    """
    Парсинг COCO Keypoints JSON.

    Ожидаемая структура:
    {
      "images": [{"id": 1, "file_name": "img.png", "width": W, "height": H}, ...],
      "annotations": [{
        "image_id": 1,
        "category_id": 1,
        "bbox": [x, y, w, h],           # абсолютные координаты
        "keypoints": [x0, y0, v0, x1, y1, v1, ...],  # 8 * 3 = 24 значения
      }, ...],
      "categories": [...]
    }

    Возвращает список dict: {
      "image_file": str, "img_w": int, "img_h": int,
      "bbox": (x, y, w, h), "keypoints": [(x, y, v), ...]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_map = {img["id"]: img for img in data["images"]}
    results = []

    for ann in data["annotations"]:
        img_info = images_map[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]

        # Парсинг bbox COCO: [x_min, y_min, width, height]
        bbox = ann.get("bbox")

        # Парсинг keypoints: [x0, y0, v0, x1, y1, v1, ...]
        raw_kpts = ann["keypoints"]
        keypoints = []
        for i in range(0, len(raw_kpts), 3):
            kx, ky, kv = raw_kpts[i], raw_kpts[i + 1], raw_kpts[i + 2]
            keypoints.append((kx, ky, kv))

        results.append({
            "image_file": img_info["file_name"],
            "img_w": img_w,
            "img_h": img_h,
            "bbox": tuple(bbox) if bbox else None,
            "keypoints": keypoints,
        })

    return results


def parse_cvat_xml(xml_path: str) -> list:
    """
    Парсинг CVAT for Images XML.

    Структура:
    <annotations>
      <image id="0" name="img.png" width="W" height="H">
        <skeleton label="pelvis">
          <points label="L_TRC" points="x,y" occluded="0"/>
          <points label="R_TRC" points="x,y" occluded="0"/>
          ...
        </skeleton>
      </image>
    </annotations>
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []

    for image_elem in root.findall("image"):
        img_file = image_elem.get("name")
        img_w = int(image_elem.get("width"))
        img_h = int(image_elem.get("height"))

        for skeleton in image_elem.findall(".//skeleton"):
            # Собираем точки по имени
            points_by_name = {}
            for pt in skeleton.findall("points"):
                label = pt.get("label")
                coords = pt.get("points").split(",")
                occluded = int(pt.get("occluded", "0"))
                x, y = float(coords[0]), float(coords[1])
                # visibility: 0=не размечена, 1=occluded, 2=видимая
                visibility = 1 if occluded else 2
                points_by_name[label] = (x, y, visibility)

            # Выстраиваем в правильном порядке
            keypoints = []
            for name in KEYPOINT_NAMES:
                if name in points_by_name:
                    keypoints.append(points_by_name[name])
                else:
                    keypoints.append((0.0, 0.0, 0))

            results.append({
                "image_file": img_file,
                "img_w": img_w,
                "img_h": img_h,
                "bbox": None,  # Вычислим из точек
                "keypoints": keypoints,
            })

    return results


def compute_bbox_from_keypoints(keypoints: list, img_w: int, img_h: int,
                                 padding_ratio: float = 0.2) -> tuple:
    """
    Вычисляет bounding box из видимых ключевых точек с отступом.
    Возвращает (x_min, y_min, width, height) в абсолютных координатах.
    """
    visible_pts = [(x, y) for x, y, v in keypoints if v > 0]
    if not visible_pts:
        return (0, 0, img_w, img_h)

    xs = [p[0] for p in visible_pts]
    ys = [p[1] for p in visible_pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min

    # Добавляем отступ
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_w, x_max + pad_x)
    y_max = min(img_h, y_max + pad_y)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def to_yolo_pose_line(annotation: dict) -> str:
    """
    Конвертирует аннотацию в строку YOLO Pose формата:
    <class_id> <cx> <cy> <w> <h> <x0> <y0> <v0> ... <x7> <y7> <v7>
    Все координаты нормализованы в [0, 1].
    """
    img_w = annotation["img_w"]
    img_h = annotation["img_h"]

    # Bbox
    bbox = annotation["bbox"]
    if bbox is None:
        bbox = compute_bbox_from_keypoints(annotation["keypoints"], img_w, img_h)

    bx, by, bw, bh = bbox
    # Конвертация в центр + размер, нормализация
    cx = (bx + bw / 2) / img_w
    cy = (by + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h

    # Ограничиваем значения
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]

    # Ключевые точки
    for kx, ky, kv in annotation["keypoints"]:
        if kv > 0:
            nx = kx / img_w
            ny = ky / img_h
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
        else:
            nx, ny = 0.0, 0.0
        parts.append(f"{nx:.6f} {ny:.6f} {int(kv)}")

    return " ".join(parts)


def find_image_path(image_file: str, images_dir: str) -> str | None:
    """Ищет файл изображения в images_dir (в том числе в подпапках)."""
    # Прямой путь
    direct = os.path.join(images_dir, image_file)
    if os.path.exists(direct):
        return direct

    # Поиск по имени файла в подпапках
    basename = os.path.basename(image_file)
    for root, _, files in os.walk(images_dir):
        if basename in files:
            return os.path.join(root, basename)

    return None


def convert_and_split(annotations: list, images_dir: str, output_dir: str,
                      val_split: float = 0.2, seed: int = 42):
    """
    Конвертирует аннотации и раскладывает в train/val структуру YOLO.
    """
    random.seed(seed)
    random.shuffle(annotations)

    val_count = int(len(annotations) * val_split)
    val_set = annotations[:val_count]
    train_set = annotations[val_count:]

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        count = 0
        for ann in split_data:
            # Найти исходное изображение
            src_img = find_image_path(ann["image_file"], images_dir)
            if src_img is None:
                print(f"[WARN] Изображение не найдено: {ann['image_file']}, пропуск")
                continue

            # Имя файла без расширения
            stem = Path(ann["image_file"]).stem
            ext = Path(src_img).suffix

            # Копируем изображение
            dst_img = os.path.join(img_dir, f"{stem}{ext}")
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            # Создаём label файл
            label_line = to_yolo_pose_line(ann)
            label_path = os.path.join(lbl_dir, f"{stem}.txt")

            # Если файл уже существует — дописываем (несколько объектов на снимке)
            with open(label_path, "a", encoding="utf-8") as f:
                f.write(label_line + "\n")

            count += 1

        print(f"[{split_name}] Сконвертировано {count} аннотаций")

    print(f"\nГотово! Структура:")
    print(f"  {output_dir}/train/images/")
    print(f"  {output_dir}/train/labels/")
    print(f"  {output_dir}/val/images/")
    print(f"  {output_dir}/val/labels/")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертер аннотаций CVAT/COCO → YOLO Pose"
    )
    parser.add_argument(
        "--input", required=True,
        help="Путь к файлу аннотаций (JSON для COCO, XML для CVAT)"
    )
    parser.add_argument(
        "--format", choices=["coco", "cvat"], required=True,
        help="Формат входного файла: coco (COCO Keypoints JSON) или cvat (CVAT XML)"
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Директория с исходными изображениями"
    )
    parser.add_argument(
        "--output-dir", default="data/keypoints",
        help="Директория для выходных данных (default: data/keypoints)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Доля валидационной выборки (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed для воспроизводимости разбиения (default: 42)"
    )

    args = parser.parse_args()

    # Парсинг аннотаций
    print(f"Чтение аннотаций из {args.input} (формат: {args.format})...")
    if args.format == "coco":
        annotations = parse_coco_keypoints(args.input)
    else:
        annotations = parse_cvat_xml(args.input)

    print(f"Найдено {len(annotations)} аннотаций")

    if not annotations:
        print("Нет аннотаций для конвертации!")
        return

    # Проверка количества точек
    for ann in annotations:
        if len(ann["keypoints"]) != NUM_KEYPOINTS:
            print(f"[WARN] {ann['image_file']}: ожидалось {NUM_KEYPOINTS} точек, "
                  f"получено {len(ann['keypoints'])}")

    # Конвертация и разбиение
    convert_and_split(
        annotations=annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

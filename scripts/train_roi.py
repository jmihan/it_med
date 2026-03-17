"""
Обучение YOLO Detection для детекции области интереса (ROI) на рентгенах таза.

Использование:
  python scripts/train_roi.py [--epochs 200] [--batch 8] [--device 0]
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_dataset():
    dataset_yaml  = PROJECT_ROOT / "data" / "roi" / "dataset.yaml"
    train_images  = PROJECT_ROOT / "data" / "roi" / "train" / "images"
    train_labels  = PROJECT_ROOT / "data" / "roi" / "train" / "labels"

    if not dataset_yaml.exists():
        print(f"[ERROR] Не найден конфиг датасета: {dataset_yaml}")
        print("Сначала запустите: python scripts/convert_roi.py")
        return False

    if not train_images.exists() or not any(train_images.iterdir()):
        print(f"[ERROR] Нет изображений: {train_images}")
        return False

    n_images = len(list(train_images.glob("*")))
    n_labels = len(list(train_labels.glob("*.txt")))
    print(f"Датасет ROI: {n_images} изображений, {n_labels} меток")
    if n_images != n_labels:
        print(f"[WARN] Количество изображений ({n_images}) != меток ({n_labels})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Обучение YOLO Detection для ROI"
    )
    parser.add_argument("--model",    default="yolo11n.pt")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--patience", type=int,   default=50)
    parser.add_argument("--name",     default="hip_roi_v1")
    parser.add_argument("--device",   default=None)
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"

    if not check_dataset():
        sys.exit(1)

    from ultralytics import YOLO

    if args.resume:
        last_weights = PROJECT_ROOT / "runs" / "roi" / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            print(f"[ERROR] Чекпоинт не найден: {last_weights}")
            sys.exit(1)
        model = YOLO(str(last_weights))
        print(f"Возобновление обучения с {last_weights}")
    else:
        model = YOLO(args.model)
        print(f"Загружена модель: {args.model}")

    dataset_yaml = str(PROJECT_ROOT / "data" / "roi" / "dataset.yaml")

    print(f"\nНачало обучения ROI: {args.epochs} эпох, batch={args.batch}, imgsz={args.imgsz}")
    print(f"Устройство: {args.device}")
    print("=" * 60)

    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,

        # Аугментации (без флипов — анатомическая ориентация важна)
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        degrees=10.0,
        translate=0.05,
        scale=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,

        project=str(PROJECT_ROOT / "runs" / "roi"),
        name=args.name,
        save=True,
        save_period=50,
        workers=4,
        exist_ok=True,
        verbose=True,
    )

    best_weights = PROJECT_ROOT / "runs" / "roi" / args.name / "weights" / "best.pt"
    target       = PROJECT_ROOT / "weights" / "hip_roi_v1.pt"
    if best_weights.exists():
        shutil.copy2(str(best_weights), str(target))
        print(f"\nЛучшие веса скопированы в: {target}")

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()

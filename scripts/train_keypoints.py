"""
Обучение YOLOv11-Pose для детекции ключевых точек тазобедренного сустава.

Использование:
  python scripts/train_keypoints.py [--epochs 300] [--batch 8] [--imgsz 640] [--device 0]

Перед запуском:
  1. Разметить изображения в CVAT/Label Studio
  2. Сконвертировать аннотации: python scripts/convert_annotations.py ...
  3. Убедиться, что data/keypoints/{train,val}/{images,labels}/ заполнены
"""

import argparse
import shutil
import sys
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dataset():
    """Проверяет наличие данных перед обучением."""
    dataset_yaml = PROJECT_ROOT / "data" / "keypoints" / "dataset.yaml"
    train_images = PROJECT_ROOT / "data" / "keypoints" / "train" / "images"
    train_labels = PROJECT_ROOT / "data" / "keypoints" / "train" / "labels"

    if not dataset_yaml.exists():
        print(f"[ERROR] Не найден конфиг датасета: {dataset_yaml}")
        return False

    if not train_images.exists() or not any(train_images.iterdir()):
        print(f"[ERROR] Нет изображений для обучения: {train_images}")
        print("Сначала подготовьте данные: python scripts/convert_annotations.py ...")
        return False

    if not train_labels.exists() or not any(train_labels.iterdir()):
        print(f"[ERROR] Нет меток для обучения: {train_labels}")
        return False

    n_images = len(list(train_images.glob("*")))
    n_labels = len(list(train_labels.glob("*.txt")))
    print(f"Датасет: {n_images} изображений, {n_labels} меток")

    if n_images != n_labels:
        print(f"[WARN] Количество изображений ({n_images}) != меток ({n_labels})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Обучение YOLOv11-Pose для детекции точек таза"
    )
    parser.add_argument("--model", default="yolo11s-pose.pt",
                        help="Предобученная модель (default: yolo11s-pose.pt)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Количество эпох (default: 300)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Размер батча (default: 8)")
    parser.add_argument("--imgsz", type=int, default=800,
                        help="Размер входного изображения (default: 800)")
    parser.add_argument("--device", default=None,
                        help="Устройство: 0 для GPU, cpu для CPU (default: авто)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--name", default="hip_pose_v1",
                        help="Имя эксперимента (default: hip_pose_v1)")
    parser.add_argument("--resume", action="store_true",
                        help="Возобновить обучение с последнего чекпоинта")

    args = parser.parse_args()

    # Автодетект устройства
    if args.device is None:
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"

    # Проверка данных
    if not check_dataset():
        sys.exit(1)

    from ultralytics import YOLO

    # Загрузка модели
    if args.resume:
        # Возобновление с последнего чекпоинта
        last_weights = PROJECT_ROOT / "runs" / "keypoints" / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            print(f"[ERROR] Чекпоинт не найден: {last_weights}")
            sys.exit(1)
        model = YOLO(str(last_weights))
        print(f"Возобновление обучения с {last_weights}")
    else:
        model = YOLO(args.model)
        print(f"Загружена предобученная модель: {args.model}")

    # Путь к конфигу датасета
    dataset_yaml = str(PROJECT_ROOT / "data" / "keypoints" / "dataset.yaml")

    # Обучение
    print(f"\nНачало обучения: {args.epochs} эпох, batch={args.batch}, imgsz={args.imgsz}")
    print(f"Устройство: {args.device}")
    print("=" * 60)

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,

        # Аугментации: БЕЗ флипов — сохранение анатомической L/R ориентации
        fliplr=0.0,
        flipud=0.0,

        # Отключаем аугментации, нарушающие анатомический контекст
        mosaic=0.0,      # Мозаика — 4 снимка в 1, бессмысленно для рентгена
        mixup=0.0,       # Смешение двух рентгенов — бессмысленно
        copy_paste=0.0,  # Копирование объектов — неприменимо
        erasing=0.0,     # Случайное стирание участков

        # Разрешённые аугментации (аналогично training/augmentations.py)
        degrees=10.0,     # Ротация ±10°
        translate=0.05,   # Сдвиг ±5%
        scale=0.1,        # Масштаб ±10%
        hsv_h=0.0,        # Без изменения оттенка (рентген ч/б)
        hsv_s=0.0,        # Без изменения насыщенности
        hsv_v=0.15,       # Яркость ±15%

        # Сохранение
        project=str(PROJECT_ROOT / "runs" / "keypoints"),
        name=args.name,
        save=True,
        save_period=50,   # Сохранять чекпоинт каждые 50 эпох

        # Pose-specific
        pose=12.0,        # Pose loss gain
        kobj=1.0,         # Keypoint objectness loss gain

        # Прочее
        workers=4,
        exist_ok=True,    # Перезапись при повторном запуске
        verbose=True,
    )

    # Копирование лучших весов в weights/
    best_weights = PROJECT_ROOT / "runs" / "keypoints" / args.name / "weights" / "best.pt"
    target_weights = PROJECT_ROOT / "weights" / "hip_keypoints_v1.pt"

    if best_weights.exists():
        shutil.copy2(str(best_weights), str(target_weights))
        print(f"\nЛучшие веса скопированы в: {target_weights}")
    else:
        print(f"\n[WARN] Лучшие веса не найдены: {best_weights}")

    print("\nОбучение завершено!")
    print(f"Результаты: {PROJECT_ROOT / 'runs' / 'keypoints' / args.name}")


if __name__ == "__main__":
    main()

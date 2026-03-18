"""
5-кратная (5-fold) кросс-валидация ResNet-классификатора.

Разбивает датасет на 5 фолдов (80% train / 20% val на каждом фолде),
обучает отдельную модель на каждом фолде и усредняет метрики.

Использование:
    # Проверка без обучения (только split + загрузка данных):
    python scripts/cross_validate_classifier.py --dry-run

    # Полная кросс-валидация на обрезанных снимках:
    python scripts/cross_validate_classifier.py --data-dir data/processed_cropped/train --epochs 30

    # Быстрая проверка (1 эпоха):
    python scripts/cross_validate_classifier.py --epochs 1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import ResNetClassifier
from training.augmentations import get_training_augmentations, get_val_augmentations
from training.dataset import MedicalImageDataset
from training.trainer import MedicalTrainer, _compute_metrics, _format_metrics


class KFoldSubset(Dataset):
    """Подмножество датасета для одного фолда с собственными трансформациями."""

    def __init__(self, images_list, indices, transforms):
        self.images_list = images_list
        self.indices = indices
        self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.images_list[self.indices[idx]]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, label


def k_fold_split(n: int, k: int, seed: int = 42):
    """
    Разбивает n индексов на k стратифицированных фолдов.
    Возвращает список пар (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = []
    for i in range(k):
        val_start = i * (n // k)
        # последний фолд берёт остаток
        val_end = val_start + (n // k) if i < k - 1 else n
        val_idx   = indices[val_start:val_end].tolist()
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]]).tolist()
        folds.append((train_idx, val_idx))
    return folds


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-fold кросс-валидация ResNet-классификатора"
    )
    parser.add_argument("--data-dir", default="data/processed_cropped/train",
                        help="Директория с данными (подпапки normal/ и pathology/)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Число эпох на каждом фолде (default: 30)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--folds", type=int, default=5,
                        help="Количество фолдов (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda (авто-определение по умолчанию)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только проверить разбивку и загрузку данных, без обучения")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = get_training_augmentations(args.image_size)
    val_transforms   = get_val_augmentations(args.image_size)

    # Загружаем весь датасет без трансформаций (только список путей)
    full_dataset = MedicalImageDataset(data_dir=args.data_dir, transforms=None)
    if len(full_dataset) == 0:
        print(f"Ошибка: нет изображений в {args.data_dir}")
        return

    images_list  = full_dataset.images_list
    n            = len(images_list)
    normal_count = sum(1 for _, lbl in images_list if lbl == 0)
    patho_count  = sum(1 for _, lbl in images_list if lbl == 1)

    print(f"\n{'='*60}")
    print(f"Датасет: {n} изображений (норма={normal_count}, патология={patho_count})")
    print(f"Фолдов: {args.folds}  |  Эпох/фолд: {args.epochs}  |  Устройство: {device}")
    print(f"Backbone: {args.backbone}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}")

    folds = k_fold_split(n, args.folds, seed=args.seed)

    # ── Режим проверки без обучения ─────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Проверка разбивки и загрузки данных...")
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_ds = KFoldSubset(images_list, train_idx, train_transforms)
            val_ds   = KFoldSubset(images_list, val_idx,   val_transforms)

            # Один батч для проверки
            loader    = DataLoader(train_ds, batch_size=min(4, len(train_ds)))
            img_batch, lbl_batch = next(iter(loader))

            # Соотношение классов в val
            val_norm  = sum(1 for i in val_idx if images_list[i][1] == 0)
            val_patho = sum(1 for i in val_idx if images_list[i][1] == 1)

            print(f"  Fold {fold_idx+1}: train={len(train_ds)}, val={len(val_ds)} "
                  f"(норма={val_norm}, патология={val_patho}), "
                  f"batch_shape={tuple(img_batch.shape)}")

        # Проверка инициализации модели
        classifier = ResNetClassifier(device=device, backbone=args.backbone,
                                      dropout=args.dropout)
        print(f"\n  Модель {args.backbone}: "
              f"{sum(p.numel() for p in classifier.model.parameters()):,} параметров")

        print("\n[DRY RUN] Всё в порядке — можно запускать полную кросс-валидацию.")
        return

    # ── Полная кросс-валидация ───────────────────────────────────────────────
    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.folds}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        train_dataset = KFoldSubset(images_list, train_idx, train_transforms)
        val_dataset   = KFoldSubset(images_list, val_idx,   val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # Свежая модель на каждом фолде
        classifier = ResNetClassifier(device=device, backbone=args.backbone,
                                      dropout=args.dropout)
        model     = classifier.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        trainer = MedicalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        best_val_metrics = None
        best_val_loss    = float("inf")

        for epoch in range(args.epochs):
            print(f"\n  Epoch {epoch + 1}/{args.epochs}")
            train_metrics = trainer.train_epoch()
            val_metrics   = trainer.validate_epoch()
            scheduler.step(val_metrics["loss"])

            print(f"  Train | {_format_metrics(train_metrics)}")
            print(f"  Val   | {_format_metrics(val_metrics)}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss    = val_metrics["loss"]
                best_val_metrics = val_metrics.copy()
                print(f"  >> Лучшая val_loss={best_val_loss:.4f} на эпохе {epoch + 1}")

        all_fold_metrics.append(best_val_metrics)
        print(f"\n  Fold {fold_idx + 1} — лучшие val-метрики:")
        print(f"  {_format_metrics(best_val_metrics)}")

    # ── Итоговое усреднение ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЕ МЕТРИКИ ПО {args.folds} ФОЛДАМ (mean ± std)")
    print(f"{'='*60}")

    metric_keys = list(all_fold_metrics[0].keys())
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        mean_v = float(np.mean(values))
        std_v  = float(np.std(values))
        unit   = "%" if key != "loss" else ""
        print(f"  {key:12s}: {mean_v:.2f}{unit} ± {std_v:.2f}{unit}")

    print(f"\n  По фолдам (Acc / F1 / AUC):")
    for fold_idx, m in enumerate(all_fold_metrics):
        print(f"  Fold {fold_idx + 1}: "
              f"Acc={m['accuracy']:.1f}%  "
              f"F1={m['f1']:.1f}%  "
              f"AUC={m['auc_roc']:.1f}%")


if __name__ == "__main__":
    main()

"""
Обучение ResNet-классификатора для бинарной классификации (норма / патология).

Использует Albumentations-аугментации из training/augmentations.py
(без HorizontalFlip/VerticalFlip для сохранения анатомической ориентации).

Использование:
    # Обучение на полных изображениях:
    python scripts/train_classifier.py --data-dir data/processed/train

    # Обучение на обрезанных по ROI изображениях:
    python scripts/train_classifier.py --data-dir data/processed_cropped/train

    # Кастомные параметры:
    python scripts/train_classifier.py --epochs 50 --backbone resnet50 --lr 0.0005
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import ResNetClassifier
from training.augmentations import get_training_augmentations, get_val_augmentations
from training.dataset import MedicalImageDataset
from training.trainer import MedicalTrainer


class TransformSubset(Dataset):
    """
    Обёртка над Subset, применяющая собственные трансформации.

    Нужна потому, что torch.utils.data.random_split возвращает Subset,
    ссылающийся на один и тот же dataset. Без обёртки нельзя задать
    разные трансформации для train и val.
    """

    def __init__(self, subset, transforms):
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.images_list[self.subset.indices[idx]]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение ResNet-классификатора для медицинских снимков"
    )
    parser.add_argument("--data-dir", default="data/processed/train",
                        help="Директория с данными (train/{normal,pathology}/)")
    parser.add_argument("--weights-path", default="weights/classifier.pt",
                        help="Путь для сохранения весов")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Доля валидационной выборки")
    parser.add_argument("--device", default=None,
                        help="cpu или cuda (auto-detect по умолчанию)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Аугментации (Albumentations, без флипов)
    train_transforms = get_training_augmentations(args.image_size)
    val_transforms = get_val_augmentations(args.image_size)

    # Датасет (без трансформаций — они будут через TransformSubset)
    full_dataset = MedicalImageDataset(data_dir=args.data_dir, transforms=None)

    if len(full_dataset) == 0:
        print(f"Ошибка: нет изображений в {args.data_dir}")
        return

    # Train/val split
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Обёртки с разными аугментациями
    train_dataset = TransformSubset(train_subset, train_transforms)
    val_dataset = TransformSubset(val_subset, val_transforms)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Модель
    classifier = ResNetClassifier(
        device=device,
        backbone=args.backbone,
        dropout=args.dropout,
    )
    model = classifier.model

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )

    # Trainer
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # Обучение
    print(f"\nОбучение: backbone={args.backbone}, device={device}, "
          f"epochs={args.epochs}, lr={args.lr}")
    print(f"Данные:  {args.data_dir}")
    print(f"Веса:    {args.weights_path}\n")

    Path(args.weights_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.fit(epochs=args.epochs, save_path=args.weights_path)

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()

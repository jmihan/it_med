import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Добавляем корень проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import CNNClassifier
from training.dataset import MedicalImageDataset
from training.trainer import MedicalTrainer

def main():
    # Параметры
    data_dir = "data/processed/train"
    weights_path = "weights/classifier.pt"
    batch_size = 16
    epochs = 10
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Трансформации
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Датасет
    full_dataset = MedicalImageDataset(data_dir=data_dir, transforms=train_transforms)
    
    if len(full_dataset) == 0:
        print(f"Error: No images found in {data_dir}")
        return

    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Применяем val_transforms к валидационному сету (через обертку, если нужно, 
    # но для простоты оставим так или переопределим transforms в датасете)
    val_dataset.dataset.transforms = val_transforms 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Модель
    classifier = CNNClassifier(device=device, backbone="resnet50", dropout=0.5)
    model = classifier.model

    # Loss и Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # Запуск обучения
    print(f"Starting training on {device}...")
    trainer.fit(epochs=epochs, save_path=weights_path)
    print("Training completed!")

if __name__ == "__main__":
    main()

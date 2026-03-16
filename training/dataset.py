import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class MedicalImageDataset(Dataset):
    """
    Загрузчик датасета для бинарной классификации медицинских изображений.
    Ожидает структуру:
    data_dir/
        normal/
        pathology/
    """
    
    def __init__(self, data_dir: str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.images_list: List[Tuple[str, int]] = []
        
        # Классы: normal = 0, pathology = 1
        self.classes = {"normal": 0, "pathology": 1}
        
        self._scan_directory()

    def _scan_directory(self):
        """
        Сканирует директорию и собирает пути к изображениям и их метки.
        """
        for class_name, class_idx in self.classes.items():
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found.")
                continue
                
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    self.images_list.append((img_path, class_idx))
        
        print(f"Found {len(self.images_list)} images in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Загружает изображение, применяет трансформации и возвращает тензор и метку.
        """
        img_path, label = self.images_list[idx]
        
        # Чтение изображения
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
            
        # Конвертация BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            # Если используются Albumentations
            if hasattr(self.transforms, 'to_dict'):
                augmented = self.transforms(image=image)
                image = augmented['image']
            else:
                # Если используются torchvision transforms
                image = self.transforms(image)
        else:
            # Базовая предобработка, если трансформации не заданы
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            
        return image, label
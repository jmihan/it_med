import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from typing import Dict, Union, Literal
from .base_model import BaseMLModel

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

class ResNetClassifier(BaseMLModel):
    """
    Классификатор для медицинских изображений на базе ResNet.
    Поддерживает backbone: resnet18, resnet50.
    """
    def __init__(
        self,
        weights_path: str = None,
        device: str = "cpu",
        backbone: Literal["resnet18", "resnet50"] = "resnet18",
        dropout: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if backbone not in _BACKBONES:
            raise ValueError(f"backbone должен быть одним из: {list(_BACKBONES)}")

        model_fn, pretrained_weights = _BACKBONES[backbone]
        self.model = model_fn(weights=pretrained_weights)

        # Заменяем классификатор: добавляем Dropout для регуляризации
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, 2),
        )

        self.model = self.model.to(self.device)

        if weights_path:
            self.weights_path = weights_path
            self._load_model()
        else:
            self.weights_path = None

    def _load_model(self):
        """Загрузка весов модели."""
        if self.weights_path:
            state_dict = torch.load(
                self.weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Предобработка изображения: ресайз, нормализация, HWC -> CHW."""
        img = cv2.resize(image, (224, 224))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        # Стандартная нормализация ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray) -> Dict[str, Union[int, float, str]]:
        """Предсказание класса и вероятностей обоих классов."""
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        class_id = int(torch.argmax(probs).item())
        prob_normal = float(probs[0].item())
        prob_pathology = float(probs[1].item())

        return {
            "class_id": class_id,
            "class_name": "Normal" if class_id == 0 else "Pathology",
            "confidence": max(prob_normal, prob_pathology),
            "prob_normal": prob_normal,
            "prob_pathology": prob_pathology,
        }

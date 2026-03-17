from abc import ABC, abstractmethod
import numpy as np


class BaseMLModel(ABC):
    """
    Базовый класс для нейросетей.
    Изолирует фреймворк (PyTorch/ONNX) от остального кода.
    """
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        """Загрузка модели и весов."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для инференса."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> any:
        """Полный цикл: предобработка -> инференс -> постобработка."""
        pass

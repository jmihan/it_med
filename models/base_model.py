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
        """TODO: Загрузка весов (torch.load или onnxruntime.InferenceSession)"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """TODO: Ресайз, нормализация, конвертация HWC -> CHW"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> any:
        """TODO: Полный цикл: preprocess -> forward pass -> postprocess"""
        pass
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class BaseMedicalPlugin(ABC):
    """
    Базовый класс для всех медицинских плагинов.
    Любой новый модуль (легкие, кости, УЗИ) должен наследоваться от него.
    """

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else {}
        self._load_models()

    @abstractmethod
    def _load_models(self):
        """Инициализация локальных моделей (PyTorch/ONNX)."""
        pass

    @abstractmethod
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Главный метод анализа.

        1. Preprocessing
        2. Model Inference (predict)
        3. Postprocessing (metrics calculation)
        4. Return standardized dictionary
        """
        pass

    @classmethod
    def get_ui_metadata(cls) -> Dict[str, Any]:
        """
        Метаданные для динамического построения UI.
        Переопределите в подклассе для настройки интерфейса.

        Возвращает dict с полями:
            display_name: str — Название в сайдбаре
            description: str — Описание модуля
            icon: str — Иконка (emoji)
            supported_formats: list — Форматы файлов
            stub: bool — True если модуль ещё не реализован
            has_classification: bool — Есть ли нейросетевой классификатор
            metric_definitions: list — [{key, label, unit, normal_range?, type?}]
            explanation_steps: list — [{title, description?, layer?}]
            visualization_layers: list — [{key, label, default}]
        """
        return {
            "display_name": cls.__name__,
            "description": "",
            "icon": "🔬",
            "supported_formats": ["png", "jpg", "jpeg", "dcm"],
            "stub": False,
            "has_classification": False,
            "metric_definitions": [],
            "explanation_steps": [],
            "visualization_layers": [],
        }

    def get_visualization_layers(self, image: np.ndarray, results: Dict) -> Dict[str, np.ndarray]:
        """
        Генерация отдельных визуализационных слоёв для студенческого режима.

        Args:
            image: Исходное изображение BGR
            results: Результат analyze()

        Returns:
            {layer_key: BGR numpy array} — изображение с отрисованным слоем
        """
        return {}

    def generate_explanation(self, results: Dict) -> List[Dict]:
        """
        Генерация пошагового объяснения для студенческого режима.

        Переопределите в подклассе для кастомных объяснений.

        Returns:
            Список шагов [{title, description, ...}]
        """
        return []

    def _load_config(self, path: str) -> dict:
        """Загрузка YAML-конфигурации плагина."""
        if path is None:
            return {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

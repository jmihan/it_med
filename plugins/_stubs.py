"""
Заглушки плагинов для демонстрации расширяемости платформы.

Показывают в UI дополнительные модули анализа, которые ещё не реализованы.
При выборе в интерфейсе отображается сообщение «Модуль в разработке».
"""

import numpy as np
from core.base_plugin import BaseMedicalPlugin


class LungAnalysisStub(BaseMedicalPlugin):
    """Заглушка: анализ рентгенограмм лёгких."""

    @classmethod
    def get_ui_metadata(cls):
        return {
            "display_name": "Анализ лёгких",
            "description": "Классификация патологий лёгких по рентгенограммам грудной клетки (пневмония, туберкулёз, опухоли)",
            "icon": "🫁",
            "supported_formats": ["png", "jpg", "jpeg", "dcm"],
            "stub": True,
            "has_classification": False,
            "metric_definitions": [],
            "explanation_steps": [],
            "visualization_layers": [],
        }

    def _load_models(self):
        pass

    def analyze(self, image: np.ndarray):
        raise NotImplementedError("Модуль анализа лёгких находится в разработке")


class ThyroidUltrasoundStub(BaseMedicalPlugin):
    """Заглушка: анализ УЗИ щитовидной железы."""

    @classmethod
    def get_ui_metadata(cls):
        return {
            "display_name": "УЗИ щитовидной железы",
            "description": "Детекция и классификация узловых образований щитовидной железы по данным ультразвуковой диагностики",
            "icon": "🔬",
            "supported_formats": ["png", "jpg", "jpeg"],
            "stub": True,
            "has_classification": False,
            "metric_definitions": [],
            "explanation_steps": [],
            "visualization_layers": [],
        }

    def _load_models(self):
        pass

    def analyze(self, image: np.ndarray):
        raise NotImplementedError("Модуль анализа УЗИ щитовидной железы находится в разработке")

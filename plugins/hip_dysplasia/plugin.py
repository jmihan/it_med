from core.base_plugin import BaseMedicalPlugin
from .metrics import calculate_all_metrics
from models.keypoint_detector import KeypointDetector
from models.classifier import ResNetClassifier
from visualization.drawing import ImageAnnotator

import numpy as np
from typing import Dict, Any


class HipDysplasiaPlugin(BaseMedicalPlugin):
    """
    Плагин для анализа дисплазии тазобедренного сустава.

    Основной метод диагностики — геометрический:
      1. Детекция 8 ключевых точек (YOLO Pose)
      2. Расчёт ацетабулярных углов (Хильгенрейнер)
      3. Определение патологии по пороговым значениям

    Классификатор ResNet используется опционально как второе мнение.
    """

    @classmethod
    def get_ui_metadata(cls) -> Dict[str, Any]:
        return {
            "display_name": "Дисплазия ТБС",
            "description": "Анализ рентгенограмм тазобедренного сустава: детекция ключевых точек, расчёт углов Хильгенрейнера, определение патологии",
            "icon": "🦴",
            "supported_formats": ["png", "jpg", "jpeg", "dcm"],
            "stub": False,
            "has_classification": True,
            "metric_definitions": [
                {
                    "key": "hilgenreiner_angle_left",
                    "label": "Угол Хильгенрейнера (лев.)",
                    "unit": "°",
                    "normal_range": [0, 30],
                },
                {
                    "key": "hilgenreiner_angle_right",
                    "label": "Угол Хильгенрейнера (прав.)",
                    "unit": "°",
                    "normal_range": [0, 30],
                },
                {
                    "key": "perkin_violation_left",
                    "label": "Нарушение линии Перкина (лев.)",
                    "unit": "",
                    "type": "bool",
                },
                {
                    "key": "perkin_violation_right",
                    "label": "Нарушение линии Перкина (прав.)",
                    "unit": "",
                    "type": "bool",
                },
            ],
            "explanation_steps": [
                {
                    "title": "Шаг 1: Поиск ключевых анатомических ориентиров",
                    "description": "Модель-детектор (YOLO Pose) находит 8 ключевых точек на снимке: Y-хрящи (TRC), края крыши вертлужной впадины (ACE), центры головок бедра (FHC), метафизы бедренных костей (FMM).",
                    "layer": "keypoints",
                },
                {
                    "title": "Шаг 2: Построение осей и расчёт углов",
                    "description": "Через Y-хрящи проводится линия Хильгенрейнера (базовая горизонтальная ось). От неё измеряются ацетабулярные углы — углы наклона крыши вертлужной впадины. Норма: до 30°.",
                    "layer": "angles",
                },
                {
                    "title": "Шаг 3: Зоны внимания нейросети (Grad-CAM)",
                    "description": "Тепловая карта показывает области снимка, оказавшие наибольшее влияние на решение классификатора ResNet.",
                    "layer": "gradcam",
                },
            ],
            "visualization_layers": [
                {"key": "keypoints", "label": "Ключевые точки", "default": True},
                {"key": "hilgenreiner", "label": "Линия Хильгенрейнера", "default": True},
                {"key": "acetabular_angles", "label": "Ацетабулярные углы", "default": True},
                {"key": "perkin_lines", "label": "Линии Перкина", "default": False},
                {"key": "gradcam", "label": "Тепловая карта (GradCAM)", "default": False},
            ],
        }

    def _load_models(self):
        """Загрузка моделей: детектор точек + классификатор (опционально)."""
        device = self.config.get('model', {}).get('device', 'cpu')
        if device == "gpu":
            device = "cuda"

        weights_path = self.config.get('model', {}).get('weights_path', 'weights/hip_keypoints_v1.pt')
        conf_threshold = self.config.get('model', {}).get('conf_threshold', 0.5)

        self.keypoint_detector = KeypointDetector(
            weights_path=weights_path,
            device=device,
            conf_threshold=conf_threshold,
        )

        # Классификатор (опционально, как второе мнение)
        classifier_weights = self.config.get('classifier_weights')
        self.classifier = None
        if classifier_weights:
            try:
                self.classifier = ResNetClassifier(
                    weights_path=classifier_weights,
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] Классификатор не загружен: {e}")

    def analyze(self, image):
        """
        Полный цикл анализа изображения.

        Args:
            image: numpy array (HxWxC, BGR)

        Returns:
            dict с полями:
              - pathology_detected: bool
              - keypoints: dict {имя: (x, y, conf)}
              - bbox: [x1, y1, x2, y2]
              - metrics: dict с углами и результатами диагностики
              - classification: dict от ResNet (или None)
              - method: str — "geometric"
        """
        # 1. Детекция ключевых точек (YOLO Pose)
        kp_result = self.keypoint_detector.predict(image)
        keypoints = kp_result["keypoints"]

        # 2. Расчёт всех метрик
        threshold = self.config.get('clinical_thresholds', {}).get(
            'hilgenreiner_angle_max_normal', 30.0
        )
        metrics = calculate_all_metrics(keypoints, threshold=threshold)

        # 3. Определение патологии
        if metrics.get("valid"):
            pathology_detected = metrics["pathology"]["any_pathology"]
        else:
            pathology_detected = False

        # 4. Классификация нейросетью (второе мнение)
        classification_result = None
        if self.classifier is not None:
            try:
                classification_result = self.classifier.predict(image)
            except Exception:
                pass

        # 5. Формирование результата
        return {
            "pathology_detected": pathology_detected,
            "keypoints": keypoints,
            "bbox": kp_result.get("bbox"),
            "detection_conf": kp_result.get("detection_conf", 0.0),
            "metrics": metrics,
            "classification": classification_result,
            "method": "geometric",
        }

    def generate_explanation(self, results: Dict):
        """Генерация пошагового объяснения для студенческого режима."""
        from .xai import generate_explanation
        return generate_explanation(results)

    def get_visualization_layers(self, image: np.ndarray, results: Dict) -> Dict[str, np.ndarray]:
        """Генерация отдельных визуализационных слоёв."""
        keypoints = results.get("keypoints", {})
        metrics = results.get("metrics", {})
        layers = {}

        layers["keypoints"] = ImageAnnotator.draw_keypoints(image, keypoints)
        layers["hilgenreiner"] = ImageAnnotator.draw_hilgenreiner_line(image, keypoints)

        if metrics.get("valid"):
            layers["acetabular_angles"] = ImageAnnotator.draw_acetabular_angles(
                image, keypoints, metrics
            )
        else:
            layers["acetabular_angles"] = image.copy()

        layers["perkin_lines"] = ImageAnnotator.draw_perkin_lines(image, keypoints)

        # GradCAM генерируется в pipeline (нужен доступ к explainer)
        # layers["gradcam"] заполняется позже

        return layers

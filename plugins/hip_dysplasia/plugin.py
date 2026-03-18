from pathlib import Path

from core.base_plugin import BaseMedicalPlugin
from .metrics import calculate_all_metrics
from models.keypoint_detector import KeypointDetector
from models.classifier import ResNetClassifier
from visualization.drawing import ImageAnnotator

import json
import numpy as np
from typing import Dict, Any

# Корень проекта (app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
                    "normal_range": [0, 25],
                },
                {
                    "key": "hilgenreiner_angle_right",
                    "label": "Угол Хильгенрейнера (прав.)",
                    "unit": "°",
                    "normal_range": [0, 25],
                },
                {
                    "key": "perkin_violation_left",
                    "label": "Линия Перкина (лев.)",
                    "unit": "",
                    "type": "bool",
                },
                {
                    "key": "perkin_violation_right",
                    "label": "Линия Перкина (прав.)",
                    "unit": "",
                    "type": "bool",
                },
                {
                    "key": "trc_distance_mm",
                    "label": "Расстояние между Y-хрящами",
                    "unit": "мм",
                },
                {
                    "key": "h_distance_left",
                    "label": "Высота h (лев.)",
                    "unit": "мм",
                    "normal_range": [8, 12],
                },
                {
                    "key": "h_distance_right",
                    "label": "Высота h (прав.)",
                    "unit": "мм",
                    "normal_range": [8, 12],
                },
                {
                    "key": "d_distance_left",
                    "label": "Дистанция d (лев.)",
                    "unit": "мм",
                    "normal_range": [10, 15],
                },
                {
                    "key": "d_distance_right",
                    "label": "Дистанция d (прав.)",
                    "unit": "мм",
                    "normal_range": [10, 15],
                },
            ],
            "explanation_steps": [
                {
                    "title": "Шаг 1: Поиск ключевых анатомических ориентиров",
                    "description": "Модель-детектор (YOLO Pose) находит 10 ключевых точек на снимке: Y-хрящи (TRC), края крыши вертлужной впадины (ACE), центры головок бедра (FHC), метафизы бедренных костей (FMM), верхние края метафизов (FMP).",
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
                {"key": "trc_distance", "label": "Расстояние TRC", "default": False},
                {"key": "h_d_distances", "label": "Расстояния h и d", "default": False},
                {"key": "gradcam", "label": "Тепловая карта (GradCAM)", "default": False},
            ],
        }

    def _resolve_weights_path(self, path_str: str) -> str:
        """Резолвит путь к весам относительно корня проекта."""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        resolved = PROJECT_ROOT / p
        return str(resolved)

    def _load_models(self):
        """Загрузка моделей: детектор точек + классификатор (опционально)."""
        device = self.config.get('model', {}).get('device', 'cpu')
        if device == "gpu":
            device = "cuda"
        self.device = device

        weights_path = self._resolve_weights_path(
            self.config.get('model', {}).get('weights_path', 'weights/hip_keypoints_v1.pt')
        )
        conf_threshold = self.config.get('model', {}).get('conf_threshold', 0.5)

        try:
            self.keypoint_detector = KeypointDetector(
                weights_path=weights_path,
                device=device,
                conf_threshold=conf_threshold,
            )
            print(f"[INFO] Детектор точек загружен: {weights_path}")
        except Exception as e:
            print(f"[WARN] Детектор точек не загружен: {e} (путь: {weights_path})")
            self.keypoint_detector = None

        # ROI-детектор (для обрезки перед классификатором)
        roi_weights = self._resolve_weights_path(
            self.config.get('roi_weights', 'weights/hip_roi_v1.pt')
        )
        self.roi_detector = None
        try:
            from ultralytics import YOLO
            self.roi_detector = YOLO(roi_weights)
            print(f"[INFO] ROI-детектор загружен: {roi_weights}")
        except Exception as e:
            print(f"[WARN] ROI-детектор не загружен: {e} (путь: {roi_weights})")

        # Классификатор (опционально, как второе мнение)
        classifier_weights = self.config.get('classifier_weights')
        classifier_backbone = self.config.get('classifier_backbone', 'resnet18')
        self.classifier = None
        if classifier_weights:
            classifier_path = self._resolve_weights_path(classifier_weights)
            try:
                self.classifier = ResNetClassifier(
                    weights_path=classifier_path,
                    device=device,
                    backbone=classifier_backbone,
                )
                print(f"[INFO] Классификатор загружен: {classifier_path} (backbone={classifier_backbone})")
            except Exception as e:
                print(f"[WARN] Классификатор не загружен: {e} (путь: {classifier_path})")

        # Масштаб пикселя (мм/пиксель) для расчёта расстояний
        self.pixel_spacing_mm = None
        self._scale_metadata = {}
        scale_path = PROJECT_ROOT / "data" / "processed" / "scale_metadata.json"
        if scale_path.exists():
            try:
                with open(scale_path, "r", encoding="utf-8") as f:
                    scale_data = json.load(f)
                self._scale_metadata = scale_data.get("images", {})
                spacing = scale_data.get("summary", {}).get("target_spacing_mm", [])
                if spacing:
                    self.pixel_spacing_mm = float(spacing[0])
                    print(f"[INFO] Масштаб пикселя: {self.pixel_spacing_mm} мм/px "
                          f"({len(self._scale_metadata)} изображений в метаданных)")
            except Exception as e:
                print(f"[WARN] Не удалось загрузить scale_metadata: {e}")

    def _crop_to_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Детектирует таз YOLO ROI-моделью и возвращает обрезанный фрагмент.
        Если детекция не удалась — возвращает исходное изображение.
        """
        padding = self.config.get('roi_padding', 20)
        try:
            results = self.roi_detector.predict(source=image, conf=0.25, verbose=False, device=self.device)
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                best = res.boxes[res.boxes.conf.argmax()]
                x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img_h, img_w = image.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_w, x2 + padding)
                y2 = min(img_h, y2 + padding)
                return image[y1:y2, x1:x2]
        except Exception as e:
            print(f"[WARN] ROI обрезка не удалась: {e}")
        return image

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
              - method: str — стратегия принятия решения
        """
        decision_mode = self.config.get('decision_mode', 'resnet_primary')

        # 1. Детекция ключевых точек (YOLO Pose)
        if self.keypoint_detector is not None:
            kp_result = self.keypoint_detector.predict(image)
        else:
            kp_result = {"keypoints": {}, "bbox": None, "detection_conf": 0.0}
        keypoints = kp_result["keypoints"]

        # 2. Классификация нейросетью (ResNet)
        # Classifier работает на обрезанном по ROI изображении — именно на таких данных обучался
        classification_result = None
        if self.classifier is not None:
            try:
                image_for_cls = self._crop_to_roi(image) if self.roi_detector is not None else image
                classification_result = self.classifier.predict(image_for_cls)
            except Exception:
                pass

        # 3. Расчёт геометрических метрик
        thresholds = self.config.get('clinical_thresholds', {})
        threshold = thresholds.get('hilgenreiner_angle_max_normal', 25.0)
        h_normal = tuple(thresholds.get('h_normal_mm', [8.0, 12.0]))
        d_normal = tuple(thresholds.get('d_normal_mm', [10.0, 15.0]))
        sublux_tol = thresholds.get('perkin_subluxation_tolerance_px', 10.0)

        metrics = calculate_all_metrics(
            keypoints, threshold=threshold,
            pixel_spacing_mm=self.pixel_spacing_mm,
            h_normal_range=h_normal,
            d_normal_range=d_normal,
            subluxation_tolerance_px=sublux_tol,
        )

        # 4. Определение патологии
        if decision_mode == "resnet_primary" and classification_result is not None:
            # ResNet — основной метод, геометрия — для отображения
            pathology_detected = classification_result.get("class_id", 0) == 1
            method = "resnet_primary"
        elif metrics.get("valid"):
            # Fallback на геометрию если ResNet недоступен
            pathology_detected = metrics["pathology"]["any_pathology"]
            method = "geometric"
        else:
            pathology_detected = False
            method = "insufficient_data"

        # 5. Формирование результата
        return {
            "pathology_detected": pathology_detected,
            "keypoints": keypoints,
            "bbox": kp_result.get("bbox"),
            "detection_conf": kp_result.get("detection_conf", 0.0),
            "metrics": metrics,
            "classification": classification_result,
            "method": method,
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
        layers["trc_distance"] = ImageAnnotator.draw_trc_distance(image, keypoints, metrics)
        layers["h_d_distances"] = ImageAnnotator.draw_h_d_distances(image, keypoints, metrics)

        # GradCAM генерируется в pipeline (нужен доступ к explainer)
        # layers["gradcam"] заполняется позже

        return layers

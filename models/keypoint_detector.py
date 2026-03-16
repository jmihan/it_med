from typing import Dict, List, Tuple
import numpy as np
from .base_model import BaseMLModel


class KeypointDetector(BaseMLModel):
    """
    Детектор анатомических точек на основе YOLO Pose (ultralytics).

    Обнаруживает 8 ключевых точек тазобедренного сустава:
      0: L_TRC — Левый Y-образный хрящ
      1: R_TRC — Правый Y-образный хрящ
      2: L_ACE — Левый край крыши вертлужной впадины
      3: R_ACE — Правый край крыши вертлужной впадины
      4: L_FHC — Центр левой головки бедра
      5: R_FHC — Центр правой головки бедра
      6: L_FMM — Середина левого метафиза бедра
      7: R_FMM — Середина правого метафиза бедра
    """

    KEYPOINT_NAMES = [
        "L_TRC", "R_TRC", "L_ACE", "R_ACE",
        "L_FHC", "R_FHC", "L_FMM", "R_FMM",
    ]

    def __init__(self, weights_path: str, device: str = "cpu",
                 conf_threshold: float = 0.5):
        self.device = device
        self.conf_threshold = conf_threshold
        super().__init__(weights_path)

    def _load_model(self):
        """Загрузка модели YOLO Pose через ultralytics."""
        from ultralytics import YOLO
        model = YOLO(self.weights_path)
        return model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """YOLO выполняет preprocessing самостоятельно."""
        return image

    def predict(self, image: np.ndarray) -> Dict:
        """
        Детекция ключевых точек на изображении.

        Args:
            image: Изображение в формате BGR (numpy array HxWxC)

        Returns:
            dict с полями:
              - keypoints: dict {имя_точки: (x, y, confidence)}
                  x, y — абсолютные координаты в пикселях
                  confidence — уверенность детекции точки [0, 1]
              - bbox: [x1, y1, x2, y2] — bounding box объекта
              - detection_conf: float — уверенность детекции объекта
        """
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        result = results[0]

        # Если ничего не обнаружено
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return {
                "keypoints": {},
                "bbox": None,
                "detection_conf": 0.0,
            }

        # Берём детекцию с наибольшей уверенностью
        best_idx = result.boxes.conf.argmax().item()
        kpts = result.keypoints.data[best_idx].cpu().numpy()  # (8, 3): x, y, conf
        bbox = result.boxes.xyxy[best_idx].cpu().numpy()       # (4,): x1, y1, x2, y2
        det_conf = result.boxes.conf[best_idx].item()

        # Формируем именованный словарь точек
        keypoints = {}
        for i, name in enumerate(self.KEYPOINT_NAMES):
            if i < len(kpts):
                x, y, conf = float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2])
                keypoints[name] = (x, y, conf)
            else:
                keypoints[name] = (0.0, 0.0, 0.0)

        return {
            "keypoints": keypoints,
            "bbox": bbox.tolist(),
            "detection_conf": det_conf,
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Детекция на батче изображений."""
        return [self.predict(img) for img in images]

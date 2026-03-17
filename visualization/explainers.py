import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import Literal, Optional


# Поддерживаемые методы XAI
XAIMethod = Literal["gradcam", "gradcam++", "eigencam"]

_METHOD_MAP = {
    "gradcam":   GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam":  EigenCAM,
}


def _get_target_layer(model: torch.nn.Module):
    """
    Возвращает последний сверточный слой ResNet.
    Именно он содержит пространственно богатые признаки перед пулингом.
    """
    return model.layer4[-1]


class ModelExplainer:
    """
    Визуализация влияния пикселей на решение классификатора (XAI).

    Поддерживаемые методы:
    - gradcam   — классический Grad-CAM (быстрый, стандарт)
    - gradcam++ — улучшенный Grad-CAM++ (лучше при нескольких объектах на снимке)
    - eigencam  — EigenCAM (не требует градиентов, подходит для BatchNorm-моделей)

    Пример использования:
        explainer = ModelExplainer(classifier.model, device="cuda")
        overlay = explainer.explain(image_bgr, class_id=1)
        cv2.imshow("GradCAM", overlay)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        method: XAIMethod = "gradcam",
    ):
        self.model = model
        self.device = torch.device(device)
        self.method = method
        self.target_layer = _get_target_layer(model)

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """BGR np.ndarray (H, W, 3) → нормализованный тензор (1, 3, 224, 224)."""
        img = cv2.resize(image_bgr, (224, 224))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def get_heatmap(
        self,
        image_bgr: np.ndarray,
        class_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Возвращает тепловую карту float32 размером (224, 224), значения [0, 1].

        Args:
            image_bgr: исходное изображение в формате BGR (как читает cv2.imread).
            class_id:  0 = Normal, 1 = Pathology.
                       None — берётся предсказанный класс.
        """
        tensor = self._preprocess(image_bgr)

        # Если class_id не задан — предсказываем
        if class_id is None:
            with torch.no_grad():
                logits = self.model(tensor)
            class_id = int(torch.argmax(logits, dim=1).item())

        cam_cls = _METHOD_MAP[self.method]
        targets = [ClassifierOutputTarget(class_id)]

        with cam_cls(model=self.model, target_layers=[self.target_layer]) as cam:
            heatmap = cam(input_tensor=tensor, targets=targets)[0]  # (224, 224)

        return heatmap.astype(np.float32)

    def overlay_on_image(
        self,
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Накладывает тепловую карту на исходное изображение.

        Args:
            image_bgr: исходное изображение BGR.
            heatmap:   float32 (224, 224), значения [0, 1].
            alpha:     прозрачность наложения (0 = только снимок, 1 = только карта).
            colormap:  цветовая схема cv2 (COLORMAP_JET, COLORMAP_INFERNO, ...).

        Returns:
            BGR изображение с наложенной тепловой картой, размером как оригинал.
        """
        h, w = image_bgr.shape[:2]

        # Тепловая карта → цветное изображение
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
        colored = cv2.applyColorMap(heatmap_resized, colormap)

        # Исходник → RGB нормализованный для наложения
        if len(image_bgr.shape) == 2:
            base = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        else:
            base = image_bgr.copy()

        # Нормализуем base в [0, 255] если нужно
        if base.dtype != np.uint8:
            base = np.clip(base, 0, 255).astype(np.uint8)

        overlay = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
        return overlay

    def explain(
        self,
        image_bgr: np.ndarray,
        class_id: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Полный pipeline: изображение → overlay с тепловой картой.

        Это основной метод для использования из UI и скриптов.

        Returns:
            BGR изображение с наложенным GradCAM, размером как оригинал.
        """
        heatmap = self.get_heatmap(image_bgr, class_id=class_id)
        return self.overlay_on_image(image_bgr, heatmap, alpha=alpha, colormap=colormap)

    def explain_both_classes(
        self,
        image_bgr: np.ndarray,
        alpha: float = 0.5,
    ) -> dict:
        """
        Строит тепловые карты для обоих классов одновременно.

        Полезно при анализе: «что модель видит как норму» vs «что видит как патологию».

        Returns:
            {
                "normal":    np.ndarray (overlay BGR),
                "pathology": np.ndarray (overlay BGR),
                "heatmap_normal":    np.ndarray float32 (224, 224),
                "heatmap_pathology": np.ndarray float32 (224, 224),
            }
        """
        heatmap_normal    = self.get_heatmap(image_bgr, class_id=0)
        heatmap_pathology = self.get_heatmap(image_bgr, class_id=1)

        return {
            "normal":            self.overlay_on_image(image_bgr, heatmap_normal,    alpha=alpha),
            "pathology":         self.overlay_on_image(image_bgr, heatmap_pathology, alpha=alpha),
            "heatmap_normal":    heatmap_normal,
            "heatmap_pathology": heatmap_pathology,
        }

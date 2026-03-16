import numpy as np
from .base_model import BaseMLModel

class ImageSegmentator(BaseMLModel):
    """
    Модель для выделения зон (сегментация). Например: легкие, опухоли, очаги воспаления.
    """

    def _load_model(self):
        """TODO: Загрузить веса сегментационной модели (U-Net, DeepLab, YOLO-seg)"""
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """TODO: Подготовка изображения (аналогично классификатору)"""
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Возвращает бинарную или многоклассовую маску размером с ИСХОДНОЕ изображение.
        
        TODO:
        1. Прогон через модель -> получение сырой маски (например, 512x512)
        2. argmax() для многоклассовой или threshold (>0.5) для бинарной
        3. ВАЖНО: Resize маски обратно в размер `image.shape[:2]` (Nearest Neighbor)
        4. Вернуть маску (numpy array)
        """
        # Заглушка: пустая маска
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
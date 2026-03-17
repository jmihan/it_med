class MedicalAnalysisError(Exception):
    """Базовый класс для ошибок платформы."""
    pass


class ModelInferenceError(MedicalAnalysisError):
    """Ошибка при инференсе модели (загрузка весов, forward pass)."""
    pass


class MissingKeypointsError(MedicalAnalysisError):
    """Модель не нашла достаточно ключевых точек для расчёта метрик."""
    pass


class PluginNotFoundError(MedicalAnalysisError):
    """Запрошенный плагин не зарегистрирован."""
    pass


class ImageLoadError(MedicalAnalysisError):
    """Ошибка загрузки или декодирования изображения."""
    pass

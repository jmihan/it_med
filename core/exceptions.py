class MedicalAnalysisError(Exception):
    """Базовый класс для ошибок платформы"""
    pass

class ModelInferenceError(MedicalAnalysisError):
    """TODO: Вызывать, если нейросеть упала с ошибкой или вернула мусор"""
    pass

class MissingKeypointsError(MedicalAnalysisError):
    """TODO: Вызывать, если модель не нашла нужные для расчета углов кости/точки"""
    pass
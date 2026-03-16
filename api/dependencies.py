"""
Общие зависимости FastAPI: singleton пайплайна анализа.
"""

from core.pipeline import AnalysisPipeline
from core.registry import register_all_plugins

_pipeline: AnalysisPipeline | None = None


def get_pipeline() -> AnalysisPipeline:
    """Получить (или создать) единственный экземпляр AnalysisPipeline."""
    global _pipeline
    if _pipeline is None:
        register_all_plugins()
        _pipeline = AnalysisPipeline()
    return _pipeline

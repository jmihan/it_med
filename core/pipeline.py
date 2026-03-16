"""
Оркестратор анализа: проводит изображение через нужный плагин,
обогащает результат визуализациями и XAI.
"""

import os
import numpy as np
from typing import Dict, Any, List, Tuple, Generator

from core.registry import PluginRegistry
from visualizating.drawing import ImageAnnotator


class AnalysisPipeline:
    """
    Оркестратор, который проводит изображение через нужный плагин
    и обогащает результат визуализациями для UI.
    """

    def __init__(self):
        self._plugin_cache = {}  # {(name, config_path): plugin_instance}
        self._explainer_cache = {}  # {plugin_name: ModelExplainer}

    def _get_plugin(self, plugin_name: str, config_path: str = None):
        """Получить или создать экземпляр плагина (с кэшированием)."""
        cache_key = (plugin_name, config_path)
        if cache_key not in self._plugin_cache:
            self._plugin_cache[cache_key] = PluginRegistry.get_plugin(plugin_name, config_path)
        return self._plugin_cache[cache_key]

    def _get_config_path(self, plugin_name: str) -> str:
        """Определить путь к конфигу плагина."""
        base = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base, "plugins", plugin_name, "config.yaml")
        if os.path.exists(config_path):
            return config_path
        return None

    def _get_explainer(self, plugin):
        """Получить ModelExplainer для классификатора плагина (если есть)."""
        classifier = getattr(plugin, 'classifier', None)
        if classifier is None:
            return None

        plugin_id = id(plugin)
        if plugin_id not in self._explainer_cache:
            try:
                from visualizating.explainers import ModelExplainer
                self._explainer_cache[plugin_id] = ModelExplainer(
                    model=classifier.model,
                    device=str(next(classifier.model.parameters()).device),
                    method="gradcam",
                )
            except Exception:
                self._explainer_cache[plugin_id] = None
        return self._explainer_cache[plugin_id]

    def run(self, image: np.ndarray, plugin_name: str, mode: str = 'doctor') -> Dict[str, Any]:
        """
        Полный цикл анализа изображения.

        Args:
            image: BGR numpy array
            plugin_name: Имя плагина из реестра
            mode: 'doctor' или 'student'

        Returns:
            Обогащённый словарь с результатами и визуализациями
        """
        # 1. Получить плагин
        config_path = self._get_config_path(plugin_name)
        plugin = self._get_plugin(plugin_name, config_path)

        # 2. Запустить анализ
        results = plugin.analyze(image)

        # 3. Метаданные плагина для UI
        metadata = plugin.get_ui_metadata()
        results["plugin_metadata"] = metadata
        results["original_image"] = image

        # 4. Полная аннотированная картинка
        keypoints = results.get("keypoints", {})
        metrics = results.get("metrics", {})

        results["annotated_image"] = ImageAnnotator.draw_full_analysis(
            image, keypoints, metrics
        )

        # 5. Послойные визуализации
        results["layer_images"] = plugin.get_visualization_layers(image, results)

        # 6. GradCAM (для студенческого режима или по запросу)
        results["heatmap"] = None
        results["heatmap_overlay"] = None

        if mode == 'student' or True:  # Всегда генерируем, если доступен
            explainer = self._get_explainer(plugin)
            if explainer is not None:
                try:
                    heatmap = explainer.get_heatmap(image, class_id=1)  # Патология
                    results["heatmap"] = heatmap
                    results["heatmap_overlay"] = ImageAnnotator.overlay_heatmap(image, heatmap)
                    # Добавляем в слои
                    results["layer_images"]["gradcam"] = results["heatmap_overlay"]
                except Exception:
                    pass

        # 7. Текстовые объяснения
        try:
            from plugins.hip_dysplasia.xai import generate_explanation
            results["explanation_steps"] = generate_explanation(results)
        except Exception:
            results["explanation_steps"] = []

        return results

    def run_batch(
        self,
        image_paths: List[str],
        plugin_name: str,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Пакетная обработка изображений.

        Args:
            image_paths: Список путей к изображениям
            plugin_name: Имя плагина

        Yields:
            (image_id, results_dict) для каждого изображения
        """
        from core.image_io import load_image

        for path in image_paths:
            # Извлечение ID из имени файла
            filename = os.path.basename(path)
            image_id = os.path.splitext(filename)[0]
            # Берём первый сегмент до '_' как ID (формат хакатона)
            if '_' in image_id:
                image_id = image_id.split('_')[0]

            try:
                image = load_image(path)
                results = self.run(image, plugin_name, mode='doctor')
                results["image_id"] = image_id
                results["source_path"] = path
                yield image_id, results
            except Exception as e:
                yield image_id, {
                    "error": str(e),
                    "image_id": image_id,
                    "source_path": path,
                    "pathology_detected": False,
                }

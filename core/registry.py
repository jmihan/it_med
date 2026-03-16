"""
Реестр всех доступных медицинских плагинов.
Позволяет добавлять новые модули анализа без изменения ядра.
"""

from typing import Dict, Type, Any
from core.base_plugin import BaseMedicalPlugin


class PluginRegistry:
    """
    Реестр плагинов. Плагины регистрируются по строковому имени,
    UI получает метаданные через get_all_metadata() для динамического построения интерфейса.
    """
    _plugins: Dict[str, Type[BaseMedicalPlugin]] = {}

    @classmethod
    def register(cls, name: str, plugin_class: Type[BaseMedicalPlugin]):
        """Зарегистрировать класс плагина под строковым именем."""
        cls._plugins[name] = plugin_class

    @classmethod
    def get_plugin(cls, name: str, config_path: str = None) -> BaseMedicalPlugin:
        """Инстанцировать и вернуть объект плагина."""
        if name not in cls._plugins:
            raise ValueError(f"Плагин '{name}' не найден! Доступные: {cls.list_available()}")
        return cls._plugins[name](config_path)

    @classmethod
    def list_available(cls) -> list:
        """Список имён зарегистрированных плагинов."""
        return list(cls._plugins.keys())

    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Метаданные всех плагинов для построения UI.
        Возвращает {plugin_name: metadata_dict}.
        """
        result = {}
        for name, plugin_class in cls._plugins.items():
            result[name] = plugin_class.get_ui_metadata()
        return result

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Метаданные конкретного плагина."""
        if name not in cls._plugins:
            raise ValueError(f"Плагин '{name}' не найден!")
        return cls._plugins[name].get_ui_metadata()

    @classmethod
    def is_stub(cls, name: str) -> bool:
        """Проверка: является ли плагин заглушкой."""
        meta = cls.get_metadata(name)
        return meta.get("stub", False)


def register_all_plugins():
    """Регистрация всех доступных плагинов в реестре."""
    import os

    # Основной плагин: дисплазия ТБС
    from plugins.hip_dysplasia.plugin import HipDysplasiaPlugin
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins", "hip_dysplasia")
    PluginRegistry.register("hip_dysplasia", HipDysplasiaPlugin)

    # Заглушки для демонстрации расширяемости
    from plugins._stubs import LungAnalysisStub, ThyroidUltrasoundStub
    PluginRegistry.register("lung_analysis", LungAnalysisStub)
    PluginRegistry.register("thyroid_ultrasound", ThyroidUltrasoundStub)

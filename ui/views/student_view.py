"""
Образовательный дашборд (режим студента).
Интерактивный: чекбоксы слоёв, пошаговые объяснения в expanders.
Полностью data-driven — не содержит хардкода под конкретный плагин.
"""

import streamlit as st
from typing import Dict, Any

from ui.components import image_viewer
import ui.state as state


def render(results: Dict[str, Any]):
    """
    Отрисовка образовательного дашборда.

    Args:
        results: Обогащённый результат из AnalysisPipeline.run()
    """
    metadata = results.get("plugin_metadata", {})
    st.header(f"Образовательный режим — {metadata.get('display_name', 'Анализ')}")

    original = results.get("original_image")
    layer_images = results.get("layer_images", {})
    viz_layers = metadata.get("visualization_layers", [])

    # --- Панель управления слоями ---
    if viz_layers:
        st.subheader("Управление слоями визуализации")
        layer_cols = st.columns(min(len(viz_layers), 5))

        active_layers = state.get("active_layers")
        # Инициализация при первом запуске
        if not active_layers:
            active_layers = {l["key"]: l.get("default", False) for l in viz_layers}
            state.set("active_layers", active_layers)

        for i, layer_def in enumerate(viz_layers):
            col_idx = i % len(layer_cols)
            with layer_cols[col_idx]:
                key = layer_def["key"]
                checked = st.checkbox(
                    layer_def["label"],
                    value=active_layers.get(key, layer_def.get("default", False)),
                    key=f"layer_{key}",
                )
                active_layers[key] = checked

        state.set("active_layers", active_layers)

        # --- Интерактивное изображение с выбранными слоями ---
        if original is not None:
            layer_order = [l["key"] for l in viz_layers]
            image_viewer.render_layered(original, layer_images, active_layers, layer_order)

    elif original is not None:
        image_viewer.render_single(original, caption="Исходный снимок")

    st.divider()

    # --- Пошаговое объяснение ---
    explanation_steps = results.get("explanation_steps", [])
    step_defs = metadata.get("explanation_steps", [])

    if explanation_steps:
        st.subheader("Пошаговый анализ")

        for i, step in enumerate(explanation_steps):
            title = step.get("title", f"Шаг {i + 1}")
            text = step.get("text", "")
            expanded = (i == 0)  # Первый шаг развёрнут

            with st.expander(title, expanded=expanded):
                st.markdown(text)

                # Показать соответствующий слой, если есть
                if i < len(step_defs):
                    layer_key = step_defs[i].get("layer")
                    if layer_key and layer_key in layer_images:
                        image_viewer.render_single(
                            layer_images[layer_key],
                            caption=f"Визуализация: {title}"
                        )

    # --- Сравнение вердиктов ---
    st.divider()
    st.subheader("Сравнение вердиктов")
    col_geo, col_nn = st.columns(2)

    with col_geo:
        st.markdown("**Геометрический анализ**")
        geo_path = results.get("geometric_pathology")
        geo_conf = results.get("geometric_confidence")
        if geo_path is None:
            st.warning("Недостаточно ключевых точек")
        elif geo_path:
            conf_str = f" ({geo_conf:.0%})" if geo_conf is not None else ""
            st.error(f"⚠️ Патология{conf_str}")
        else:
            conf_str = f" ({geo_conf:.0%})" if geo_conf is not None else ""
            st.success(f"✅ Норма{conf_str}")

    with col_nn:
        st.markdown("**ResNet-классификатор**")
        nn_path = results.get("resnet_pathology")
        nn_conf = results.get("resnet_confidence")
        if nn_path is None:
            st.info("Классификатор не подключён")
        elif nn_path:
            conf_str = f" ({nn_conf:.0%})" if nn_conf is not None else ""
            st.error(f"⚠️ Патология{conf_str}")
        else:
            conf_str = f" ({nn_conf:.0%})" if nn_conf is not None else ""
            st.success(f"✅ Норма{conf_str}")

    # --- Итоговое заключение системы ---
    pathology = results.get("pathology_detected", False)
    st.divider()
    if pathology:
        st.error("⚠️ **Обнаружены признаки патологии.** Рекомендуется консультация специалиста.")
    else:
        st.success("✅ **Признаков патологии не обнаружено.** Показатели в пределах нормы.")

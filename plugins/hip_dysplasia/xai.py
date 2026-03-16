"""
Генератор текстовых объяснений для образовательного режима.
Формирует пошаговое описание на основе реальных метрик анализа.
"""

from typing import Dict, List


def generate_explanation(results: dict) -> List[Dict[str, str]]:
    """
    Генерация пошагового объяснения на основе результатов анализа.

    Args:
        results: Полный результат из pipeline (содержит metrics, keypoints, classification)

    Returns:
        Список словарей [{title, text}] — шаги объяснения
    """
    metrics = results.get("metrics", {})
    keypoints = results.get("keypoints", {})
    classification = results.get("classification")
    pathology_detected = results.get("pathology_detected", False)

    steps = []

    # --- Шаг 1: Детекция ключевых точек ---
    detected = []
    for name, (x, y, conf) in keypoints.items():
        if conf >= 0.3:
            detected.append(f"  - **{name}**: координаты ({int(x)}, {int(y)}), уверенность {conf:.0%}")

    kp_text = f"Модель-детектор (YOLO Pose) обнаружила **{len(detected)} из 8** ключевых точек:\n"
    kp_text += "\n".join(detected) if detected else "  Точки не найдены."
    kp_text += "\n\nТочки разделены на 4 группы: Y-хрящи (TRC, зелёные), края крыши впадины (ACE, красные), центры головок бедра (FHC, синие), метафизы (FMM, оранжевые)."

    steps.append({
        "title": "Шаг 1: Поиск ключевых анатомических ориентиров",
        "text": kp_text,
    })

    # --- Шаг 2: Расчёт углов ---
    if metrics.get("valid"):
        angle_l = metrics.get("hilgenreiner_angle_left", 0)
        angle_r = metrics.get("hilgenreiner_angle_right", 0)
        threshold = metrics.get("pathology", {}).get("threshold", 30)
        path_l = metrics.get("pathology", {}).get("left", {}).get("is_pathology", False)
        path_r = metrics.get("pathology", {}).get("right", {}).get("is_pathology", False)

        angle_text = "Через Y-хрящи (L\\_TRC и R\\_TRC) проведена **линия Хильгенрейнера** — базовая горизонтальная ось таза.\n\n"
        angle_text += "От неё измерены ацетабулярные углы — углы наклона крыши вертлужной впадины:\n"
        angle_text += f"  - **Левый**: {angle_l:.1f}° — {'⚠️ ПАТОЛОГИЯ' if path_l else '✅ Норма'}\n"
        angle_text += f"  - **Правый**: {angle_r:.1f}° — {'⚠️ ПАТОЛОГИЯ' if path_r else '✅ Норма'}\n\n"
        angle_text += f"Порог нормы: **{threshold:.0f}°**. Значение выше порога указывает на дисплазию."

        # Линии Перкина
        perkin_l = metrics.get("perkin_violation_left")
        perkin_r = metrics.get("perkin_violation_right")
        if perkin_l is not None or perkin_r is not None:
            angle_text += "\n\n**Линии Перкина** (вертикали через края крыши):\n"
            if perkin_l is not None:
                angle_text += f"  - Левая: {'⚠️ Нарушение' if perkin_l else '✅ Норма'}\n"
            if perkin_r is not None:
                angle_text += f"  - Правая: {'⚠️ Нарушение' if perkin_r else '✅ Норма'}\n"
    else:
        angle_text = "Недостаточно точек для расчёта углов. Необходимы как минимум Y-хрящи (L\\_TRC, R\\_TRC) и края крыши (L\\_ACE, R\\_ACE)."

    steps.append({
        "title": "Шаг 2: Построение осей и расчёт углов",
        "text": angle_text,
    })

    # --- Шаг 3: Классификация и GradCAM ---
    if classification:
        cls_name = classification.get("class_name", "?")
        conf = classification.get("confidence", 0)
        p_norm = classification.get("prob_normal", 0)
        p_path = classification.get("prob_pathology", 0)

        cam_text = f"Нейросеть-классификатор (ResNet) дала независимую оценку:\n"
        cam_text += f"  - **Предсказание**: {cls_name} (уверенность {conf:.0%})\n"
        cam_text += f"  - Вероятность нормы: {p_norm:.0%}\n"
        cam_text += f"  - Вероятность патологии: {p_path:.0%}\n\n"
        cam_text += "Тепловая карта (Grad-CAM) показывает области, на которые нейросеть обращала наибольшее внимание при классификации. Яркие области — зоны максимального влияния на решение."
    else:
        cam_text = "Классификатор ResNet не подключён. Тепловая карта Grad-CAM недоступна."

    steps.append({
        "title": "Шаг 3: Зоны внимания нейросети (Grad-CAM)",
        "text": cam_text,
    })

    # --- Итоговый вывод ---
    if metrics.get("valid"):
        if pathology_detected:
            conclusion = "⚠️ **Заключение**: Обнаружены признаки дисплазии тазобедренного сустава. "
            conclusion += "Ацетабулярный угол превышает пороговое значение, что указывает на недоразвитие крыши вертлужной впадины. "
            conclusion += "Рекомендуется консультация ортопеда."
        else:
            conclusion = "✅ **Заключение**: Признаков дисплазии не обнаружено. "
            conclusion += "Ацетабулярные углы в пределах нормы, анатомические ориентиры соответствуют нормальному развитию тазобедренного сустава."
    else:
        conclusion = "⚠️ **Заключение**: Невозможно сделать вывод — недостаточно данных для геометрического анализа."

    steps.append({
        "title": "Итоговое заключение",
        "text": conclusion,
    })

    return steps

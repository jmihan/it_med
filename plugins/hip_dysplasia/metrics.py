"""
Расчёт диагностических метрик дисплазии тазобедренного сустава.

Реализованные метрики:
  - Ацетабулярный индекс (угол Хильгенрейнера)
  - Квадрантный анализ по линии Перкина
  - Расстояния h и d по Хильгенрейнеру
  - Триада Путти
  - Диагностика патологии по пороговым значениям
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Угол между двумя 2D-векторами в градусах (0..180).

    Args:
        v1: Первый вектор (2,)
        v2: Второй вектор (2,)

    Returns:
        Угол в градусах
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _hilgenreiner_basis(keypoints: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Базис линии Хильгенрейнера: направляющий и перпендикулярный единичные векторы.

    Returns:
        (l_trc, r_trc, h_norm, perp_norm)
        h_norm  — единичный вектор вдоль линии Хильгенрейнера (L_TRC → R_TRC)
        perp_norm — единичный перпендикуляр (вниз от линии)
    """
    l_trc = np.array(keypoints["L_TRC"][:2])
    r_trc = np.array(keypoints["R_TRC"][:2])
    h_vec = r_trc - l_trc
    h_len = np.linalg.norm(h_vec) + 1e-8
    h_norm = h_vec / h_len
    # Перпендикуляр вниз (в экранных координатах y↓ = вниз)
    perp_norm = np.array([h_norm[1], -h_norm[0]])
    # Убедимся, что перпендикуляр смотрит вниз (y положительный)
    if perp_norm[1] < 0:
        perp_norm = -perp_norm
    return l_trc, r_trc, h_norm, perp_norm


def calculate_hilgenreiner_angle(keypoints: dict, side: str = "left") -> float:
    """
    Расчёт ацетабулярного индекса (угла Хильгенрейнера).

    Линия Хильгенрейнера проходит через оба Y-образных хряща (L_TRC → R_TRC).
    Ацетабулярный угол — угол между линией Хильгенрейнера и линией от
    Y-хряща до края крыши вертлужной впадины (TRC → ACE) на соответствующей стороне.

    Норма для 6 мес: 20–25°. Превышение на 5°+ указывает на дисплазию.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)} — обнаруженные ключевые точки
        side: "left" или "right"

    Returns:
        Угол в градусах
    """
    l_trc = np.array(keypoints["L_TRC"][:2])
    r_trc = np.array(keypoints["R_TRC"][:2])

    # Вектор линии Хильгенрейнера (базовая горизонтальная линия)
    hilgenreiner_vec = r_trc - l_trc

    if side == "left":
        ace = np.array(keypoints["L_ACE"][:2])
        ace_vec = ace - l_trc
    else:
        ace = np.array(keypoints["R_ACE"][:2])
        ace_vec = ace - r_trc
        # Инвертируем направление для правой стороны
        hilgenreiner_vec = l_trc - r_trc

    angle = calculate_angle_between_vectors(hilgenreiner_vec, ace_vec)
    # Ацетабулярный угол по определению острый (< 90°)
    if angle > 90:
        angle = 180 - angle
    return angle


def check_pathology(angle_left: float, angle_right: float,
                    threshold: float = 25.0) -> Dict:
    """
    Определение патологии по ацетабулярным индексам обеих сторон.

    Args:
        angle_left: Ацетабулярный угол левой стороны (градусы)
        angle_right: Ацетабулярный угол правой стороны (градусы)
        threshold: Пороговое значение (default: 25° для ~6 мес)

    Returns:
        dict с результатами диагностики:
          - left: {angle, is_pathology}
          - right: {angle, is_pathology}
          - any_pathology: bool — есть ли патология хотя бы с одной стороны
    """
    return {
        "left": {
            "angle": round(angle_left, 1),
            "is_pathology": angle_left > threshold,
        },
        "right": {
            "angle": round(angle_right, 1),
            "is_pathology": angle_right > threshold,
        },
        "any_pathology": angle_left > threshold or angle_right > threshold,
        "threshold": threshold,
    }


def calculate_perkin_line_violation(keypoints: dict, side: str = "left",
                                    subluxation_tolerance_px: float = 10.0) -> Dict:
    """
    Квадрантный анализ положения головки бедра относительно линий Перкина и Хильгенрейнера.

    Линия Перкина — вертикаль через край крыши вертлужной впадины (ACE),
    перпендикулярная линии Хильгенрейнера.

    Квадранты:
      - inner_lower (норма): головка медиальнее Перкина и ниже Хильгенрейнера
      - outer_lower (подвывих): головка латеральнее Перкина, ниже Хильгенрейнера
      - outer_upper (вывих): головка латеральнее Перкина и выше Хильгенрейнера
      - inner_upper: головка медиальнее Перкина, выше Хильгенрейнера

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        side: "left" или "right"
        subluxation_tolerance_px: пиксельная толерантность для подвывиха

    Returns:
        dict с результатом:
          - violation: bool — True если головка латеральнее линии Перкина
          - femoral_point: (x, y) — использованная точка головки/метафиза
          - perkin_x: float — x-координата линии Перкина
          - quadrant: str — квадрант (inner_lower | outer_lower | outer_upper | inner_upper)
          - diagnosis: str — norm | subluxation | dislocation
          - lateral_displacement_px: float — боковое смещение (положит. = латерально)
          - vertical_displacement_px: float — вертикальное смещение (положит. = вверх)
    """
    l_trc, r_trc, h_norm, perp_norm = _hilgenreiner_basis(keypoints)

    if side == "left":
        ace = np.array(keypoints["L_ACE"][:2])
        fhc_conf = keypoints.get("L_FHC", (0, 0, 0))[2]
        if fhc_conf > 0.3:
            femoral = np.array(keypoints["L_FHC"][:2])
        else:
            femoral = np.array(keypoints["L_FMM"][:2])
    else:
        ace = np.array(keypoints["R_ACE"][:2])
        fhc_conf = keypoints.get("R_FHC", (0, 0, 0))[2]
        if fhc_conf > 0.3:
            femoral = np.array(keypoints["R_FHC"][:2])
        else:
            femoral = np.array(keypoints["R_FMM"][:2])

    vec_to_femoral = femoral - ace

    # Проекция на направление Хильгенрейнера (боковое смещение)
    lateral_proj = float(np.dot(vec_to_femoral, h_norm))
    # Проекция на перпендикуляр (вертикальное смещение, + = вниз от линии)
    vertical_proj = float(np.dot(vec_to_femoral, perp_norm))

    # h_norm направлен L_TRC → R_TRC
    # Для ЛЕВОЙ стороны: латерально = против h_norm (отрицательная проекция)
    # Для ПРАВОЙ стороны: латерально = по h_norm (положительная проекция)
    if side == "left":
        lateral_displacement = -lateral_proj  # Положит. = латерально (от центра)
    else:
        lateral_displacement = lateral_proj

    # vertical_proj > 0 = ниже линии Хильгенрейнера (норма)
    # vertical_proj < 0 = выше линии (смещение вверх)
    vertical_displacement = -vertical_proj  # Положит. = вверх (патология)

    # Определение квадранта
    is_lateral = lateral_displacement > subluxation_tolerance_px
    is_near_perkin = abs(lateral_displacement) <= subluxation_tolerance_px
    is_above = vertical_displacement > 0  # выше линии Хильгенрейнера

    if is_lateral or is_near_perkin:
        if is_above:
            quadrant = "outer_upper"
        else:
            quadrant = "outer_lower"
    else:
        if is_above:
            quadrant = "inner_upper"
        else:
            quadrant = "inner_lower"

    # Диагноз по квадранту и положению относительно линии Перкина
    if is_lateral:
        if is_above:
            diagnosis = "dislocation"  # Вывих
        else:
            diagnosis = "subluxation"  # Подвывих (латеральнее, но ниже)
    elif is_near_perkin:
        diagnosis = "subluxation"  # Подвывих (линия проходит через кость)
    else:
        diagnosis = "norm"

    violation = is_lateral or is_near_perkin

    return {
        "violation": bool(violation),
        "femoral_point": femoral.tolist(),
        "perkin_x": float(ace[0]),
        "quadrant": quadrant,
        "diagnosis": diagnosis,
        "lateral_displacement_px": round(lateral_displacement, 1),
        "vertical_displacement_px": round(vertical_displacement, 1),
    }


def calculate_trc_distance(keypoints: dict, pixel_spacing_mm: float = None) -> Dict:
    """
    Расчёт расстояния между Y-хрящами (L_TRC и R_TRC).

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        pixel_spacing_mm: Размер пикселя в мм (если None, возвращает только в пикселях)

    Returns:
        dict с расстоянием в пикселях и миллиметрах
    """
    l_trc = np.array(keypoints["L_TRC"][:2])
    r_trc = np.array(keypoints["R_TRC"][:2])

    distance_px = float(np.linalg.norm(r_trc - l_trc))

    result = {"distance_px": round(distance_px, 1)}
    if pixel_spacing_mm is not None:
        distance_mm = distance_px * pixel_spacing_mm
        result["distance_mm"] = round(distance_mm, 1)

    return result


def _get_femoral_reference(keypoints: dict, side: str) -> Optional[np.ndarray]:
    """
    Получить опорную точку проксимального отдела бедра для расчёта h/d.

    Приоритет: FMP (верхний край метафиза) → FHC (центр головки) → FMM (середина метафиза).
    FMP — наиболее точная точка для расчёта h (самая высокая точка шейки бедра).

    Returns:
        np.ndarray (x, y) или None если ни одна точка недоступна
    """
    prefix = "L" if side == "left" else "R"

    for suffix in ["FMP", "FHC", "FMM"]:
        key = f"{prefix}_{suffix}"
        if key in keypoints and keypoints[key][2] > 0.3:
            return np.array(keypoints[key][:2])
    return None


def calculate_h_distance(keypoints: dict, side: str = "left",
                         pixel_spacing_mm: float = None,
                         normal_range_mm: Tuple[float, float] = (8.0, 12.0)) -> Dict:
    """
    Расчёт высоты h — расстояние от линии Хильгенрейнера до верхнего края
    проксимального отдела бедра.

    Используется FHC (центр головки бедра) как ближайшая к верхнему краю точка,
    с fallback на FMM (середина метафиза).

    Норма: 8–12 мм. Уменьшение говорит о смещении вверх.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        side: "left" или "right"
        pixel_spacing_mm: Размер пикселя в мм
        normal_range_mm: Диапазон нормы в мм

    Returns:
        dict с расстоянием и оценкой нормы
    """
    ref_point = _get_femoral_reference(keypoints, side)
    if ref_point is None:
        return {"h_px": None, "h_mm": None, "is_normal": None, "error": "Точки FHC/FMM не найдены"}

    l_trc, r_trc, h_norm, perp_norm = _hilgenreiner_basis(keypoints)
    trc = l_trc if side == "left" else r_trc

    # h = перпендикулярное расстояние от опорной точки до линии Хильгенрейнера
    vec = ref_point - trc
    h_px = abs(float(np.dot(vec, perp_norm)))

    result = {"h_px": round(h_px, 1), "h_mm": None, "is_normal": None}

    if pixel_spacing_mm is not None:
        h_mm = h_px * pixel_spacing_mm
        result["h_mm"] = round(h_mm, 1)
        result["is_normal"] = normal_range_mm[0] <= h_mm <= normal_range_mm[1]

    return result


def calculate_d_distance(keypoints: dict, side: str = "left",
                         pixel_spacing_mm: float = None,
                         normal_range_mm: Tuple[float, float] = (10.0, 15.0)) -> Dict:
    """
    Расчёт дистанции d — расстояние от дна вертлужной впадины (TRC) до
    проекции верхнего края бедра на линию Хильгенрейнера.

    Норма: 10–15 мм. Увеличение указывает на латерализацию.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        side: "left" или "right"
        pixel_spacing_mm: Размер пикселя в мм
        normal_range_mm: Диапазон нормы в мм

    Returns:
        dict с расстоянием и оценкой нормы
    """
    ref_point = _get_femoral_reference(keypoints, side)
    if ref_point is None:
        return {"d_px": None, "d_mm": None, "is_normal": None, "error": "Точки FHC/FMM не найдены"}

    l_trc, r_trc, h_norm, perp_norm = _hilgenreiner_basis(keypoints)
    trc = l_trc if side == "left" else r_trc

    # d = проекция вектора (ref_point - TRC) на направление линии Хильгенрейнера
    vec = ref_point - trc
    d_px = abs(float(np.dot(vec, h_norm)))

    result = {"d_px": round(d_px, 1), "d_mm": None, "is_normal": None}

    if pixel_spacing_mm is not None:
        d_mm = d_px * pixel_spacing_mm
        result["d_mm"] = round(d_mm, 1)
        result["is_normal"] = normal_range_mm[0] <= d_mm <= normal_range_mm[1]

    return result


def check_putti_triad(keypoints: dict, metrics: dict) -> Dict:
    """
    Проверка триады Путти — трёх рентгенологических признаков дисплазии.

    1. Избыточная скошенность крыши вертлужной впадины (ацетабулярный угол)
    2. Смещение проксимального конца бедренной кости (латерализация/проксимальное)
    3. Гипоплазия или отсутствие ядра окостенения головки бедра (FHC)

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        metrics: dict из calculate_all_metrics (содержит pathology, h/d, perkin)

    Returns:
        dict с результатом оценки триады
    """
    pathology = metrics.get("pathology", {})

    # Признак 1: скошенность крыши (ацетабулярный угол > порога)
    sign_1 = (pathology.get("left", {}).get("is_pathology", False) or
              pathology.get("right", {}).get("is_pathology", False))

    # Признак 2: смещение бедренной кости
    # Проверяем h (уменьшение), d (увеличение), Перкин (нарушение)
    sign_2 = False
    for side in ["left", "right"]:
        h = metrics.get(f"h_distance_{side}", {})
        d = metrics.get(f"d_distance_{side}", {})
        perkin = metrics.get(f"perkin_{side}", {})

        if h.get("is_normal") is False:  # h < нормы = смещение вверх
            sign_2 = True
        if d.get("is_normal") is False:  # d > нормы = латерализация
            sign_2 = True
        if perkin.get("violation", False):
            sign_2 = True

    # Признак 3: гипоплазия/отсутствие ядра окостенения (FHC)
    # Низкая уверенность в детекции FHC = потенциальная гипоплазия
    sign_3 = False
    for prefix in ["L", "R"]:
        fhc_conf = keypoints.get(f"{prefix}_FHC", (0, 0, 0))[2]
        if fhc_conf < 0.3:
            sign_3 = True

    signs_present = sum([sign_1, sign_2, sign_3])

    details_parts = []
    if sign_1:
        details_parts.append("скошенность крыши впадины")
    if sign_2:
        details_parts.append("смещение бедренной кости")
    if sign_3:
        details_parts.append("гипоплазия ядра окостенения")

    if signs_present == 0:
        details = "Признаки триады Путти не обнаружены."
    elif signs_present == 3:
        details = f"Полная триада Путти: {', '.join(details_parts)}."
    else:
        details = f"Обнаружено {signs_present}/3 признаков: {', '.join(details_parts)}."

    return {
        "sign_1_roof_slope": sign_1,
        "sign_2_displacement": sign_2,
        "sign_3_ossification": sign_3,
        "signs_present": signs_present,
        "triad_complete": signs_present == 3,
        "details": details,
    }


def calculate_all_metrics(keypoints: dict, threshold: float = 25.0,
                          pixel_spacing_mm: float = None,
                          h_normal_range: Tuple[float, float] = (8.0, 12.0),
                          d_normal_range: Tuple[float, float] = (10.0, 15.0),
                          subluxation_tolerance_px: float = 10.0) -> Dict:
    """
    Расчёт всех диагностических метрик.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        threshold: Пороговое значение ацетабулярного угла (25° для 6 мес)
        pixel_spacing_mm: Размер пикселя в мм
        h_normal_range: Диапазон нормы h в мм
        d_normal_range: Диапазон нормы d в мм
        subluxation_tolerance_px: Толерантность подвывиха (пиксели)

    Returns:
        dict со всеми метриками
    """
    # Проверяем наличие минимально необходимых точек
    required = ["L_TRC", "R_TRC", "L_ACE", "R_ACE"]
    for name in required:
        if name not in keypoints or keypoints[name][2] < 0.1:
            return {
                "error": f"Не обнаружена ключевая точка: {name}",
                "valid": False,
            }

    # Ацетабулярные углы
    angle_left = calculate_hilgenreiner_angle(keypoints, side="left")
    angle_right = calculate_hilgenreiner_angle(keypoints, side="right")
    pathology = check_pathology(angle_left, angle_right, threshold)

    result = {
        "valid": True,
        "hilgenreiner_angle_left": round(angle_left, 1),
        "hilgenreiner_angle_right": round(angle_right, 1),
        "pathology": pathology,
    }

    # Расстояние между Y-хрящами
    trc_dist = calculate_trc_distance(keypoints, pixel_spacing_mm)
    result["trc_distance_px"] = trc_dist["distance_px"]
    if "distance_mm" in trc_dist:
        result["trc_distance_mm"] = trc_dist["distance_mm"]

    # Расстояния h и d
    for side in ["left", "right"]:
        result[f"h_distance_{side}"] = calculate_h_distance(
            keypoints, side, pixel_spacing_mm, h_normal_range
        )
        result[f"d_distance_{side}"] = calculate_d_distance(
            keypoints, side, pixel_spacing_mm, d_normal_range
        )

    # Линия Перкина с квадрантным анализом
    for side in ["left", "right"]:
        prefix = "L" if side == "left" else "R"
        fhc_key = f"{prefix}_FHC"
        fmm_key = f"{prefix}_FMM"
        has_femoral = (
            (fhc_key in keypoints and keypoints[fhc_key][2] > 0.3) or
            (fmm_key in keypoints and keypoints[fmm_key][2] > 0.3)
        )
        if has_femoral:
            perkin = calculate_perkin_line_violation(
                keypoints, side=side,
                subluxation_tolerance_px=subluxation_tolerance_px
            )
            result[f"perkin_{side}"] = perkin
            # Backward compatibility
            result[f"perkin_violation_{side}"] = perkin["violation"]

    # Триада Путти
    result["putti_triad"] = check_putti_triad(keypoints, result)

    return result

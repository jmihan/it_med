import os
import json
import cv2
import numpy as np
import pydicom
import re
from pathlib import Path
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_preparation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def slugify(text: str) -> str:
    """
    Конвертирует строку в безопасное имя файла:
    - Заменяет кириллицу на латиницу (транслитерация) или просто удаляет спецсимволы.
    - Оставляет только латинские буквы, цифры и подчеркивания.
    """
    # Простая транслитерация для основных кириллических символов
    symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
               u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")
    tr = {ord(a): ord(b) for a, b in zip(*symbols)}
    text = text.translate(tr)

    # Оставляем только буквы, цифры и подчеркивания
    text = re.sub(r'[^\w\s-]', '_', text)
    text = re.sub(r'[\s-]+', '_', text).strip('_')
    return text


def extract_pixel_spacing(ds: pydicom.Dataset) -> tuple | None:
    """
    Извлекает физический размер пикселя (мм/пиксель) из DICOM.

    Приоритет атрибутов:
      1. PixelSpacing — расстояние между центрами пикселей в плоскости пациента
      2. ImagerPixelSpacing — расстояние на детекторе (без коррекции увеличения)

    Если доступен EstimatedRadiographicMagnificationFactor, корректируем
    ImagerPixelSpacing на увеличение.

    Returns:
        (row_spacing_mm, col_spacing_mm) или None если данных нет
    """
    # PixelSpacing — наиболее точный атрибут (уже скорректирован на геометрию)
    if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
        try:
            spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
            if spacing[0] > 0 and spacing[1] > 0:
                return (spacing[0], spacing[1])
        except (IndexError, ValueError, TypeError):
            pass

    # ImagerPixelSpacing — размер на детекторе
    if hasattr(ds, 'ImagerPixelSpacing') and ds.ImagerPixelSpacing:
        try:
            spacing = [float(ds.ImagerPixelSpacing[0]), float(ds.ImagerPixelSpacing[1])]
            if spacing[0] > 0 and spacing[1] > 0:
                # Корректируем на увеличение если известен фактор
                mag = 1.0
                if hasattr(ds, 'EstimatedRadiographicMagnificationFactor'):
                    try:
                        mag = float(ds.EstimatedRadiographicMagnificationFactor)
                        if mag <= 0:
                            mag = 1.0
                    except (ValueError, TypeError):
                        mag = 1.0
                # Реальный размер на пациенте = размер на детекторе / увеличение
                return (spacing[0] / mag, spacing[1] / mag)
        except (IndexError, ValueError, TypeError):
            pass

    return None


class DatasetPreparer:
    def __init__(self, src_root: str, dst_root: str):
        self.src_root = Path(src_root)
        self.dst_root = Path(dst_root)
        self.stats = {"train_normal": 0, "train_pathology": 0, "test": 0, "errors": 0}

        # Определение маппинга папок для train
        self.normal_folders = ["Норма", "Норма_отдельные снимки"]
        self.pathology_folders = ["Патология", "Патология_jpg", "Патология_отдельные файлы"]

        # Собранные pixel spacing из DICOM (мм/пиксель)
        self.dicom_spacings = []          # [(row_sp, col_sp), ...]
        self.target_spacing = None        # Целевой масштаб (мм/пиксель)
        self.scale_metadata = {}          # {filename: {spacing, scale_factor, ...}}

    def setup_directories(self):
        """Создает структуру папок для обработанного датасета."""
        (self.dst_root / "train" / "normal").mkdir(parents=True, exist_ok=True)
        (self.dst_root / "train" / "pathology").mkdir(parents=True, exist_ok=True)
        (self.dst_root / "test" / "images").mkdir(parents=True, exist_ok=True)
        logging.info(f"Структура директорий создана в {self.dst_root}")

    def is_dicom(self, file_path: Path) -> bool:
        """Проверяет, является ли файл DICOM (по расширению или заголовку)."""
        if file_path.suffix.lower() == '.dcm':
            return True
        if file_path.suffix == '':
            try:
                with open(file_path, 'rb') as f:
                    f.seek(128)
                    return f.read(4) == b"DICM"
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # Первый проход: сбор PixelSpacing из всех DICOM
    # ------------------------------------------------------------------
    def collect_pixel_spacings(self):
        """
        Сканирует все DICOM-файлы в data/train и data/test,
        извлекает PixelSpacing и вычисляет целевой масштаб (медиана).
        """
        logging.info("Первый проход: сбор PixelSpacing из DICOM...")
        spacings = []

        for subdir in ["train", "test"]:
            src = self.src_root / subdir
            if not src.exists():
                continue
            for file_path in sorted(src.rglob("*")):
                if file_path.is_dir():
                    continue
                if not self.is_dicom(file_path):
                    continue
                try:
                    ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    spacing = extract_pixel_spacing(ds)
                    if spacing is not None:
                        spacings.append(spacing)
                        logging.debug(f"  {file_path.name}: spacing={spacing}")
                except Exception as e:
                    logging.warning(f"  Не удалось прочитать метаданные {file_path.name}: {e}")

        self.dicom_spacings = spacings

        if spacings:
            # Медиана по каждой оси
            row_spacings = [s[0] for s in spacings]
            col_spacings = [s[1] for s in spacings]
            median_row = float(np.median(row_spacings))
            median_col = float(np.median(col_spacings))
            self.target_spacing = (median_row, median_col)

            logging.info(f"  Найдено {len(spacings)} DICOM с PixelSpacing")
            logging.info(f"  Диапазон row: {min(row_spacings):.4f} — {max(row_spacings):.4f} мм/пкс")
            logging.info(f"  Диапазон col: {min(col_spacings):.4f} — {max(col_spacings):.4f} мм/пкс")
            logging.info(f"  Целевой масштаб (медиана): {median_row:.4f} x {median_col:.4f} мм/пкс")
        else:
            logging.warning("  Ни один DICOM не содержит PixelSpacing — "
                            "масштабирование невозможно, изображения сохранятся как есть")
            self.target_spacing = None

    # ------------------------------------------------------------------
    # Чтение и масштабирование
    # ------------------------------------------------------------------
    def read_image(self, file_path: Path) -> np.ndarray:
        """Читает изображение (DICOM или JPG) и возвращает numpy array."""
        try:
            if self.is_dicom(file_path):
                ds = pydicom.dcmread(str(file_path))
                img = ds.pixel_array.astype(float)
            else:
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Не удалось прочитать изображение через OpenCV")
                img = img.astype(float)

            # Нормализация 0-255
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) * (255.0 / (img_max - img_min))
            else:
                img = np.zeros_like(img)

            return img.astype(np.uint8)
        except Exception as e:
            logging.error(f"Ошибка при чтении {file_path}: {e}")
            self.stats["errors"] += 1
            return None

    def rescale_image(self, image: np.ndarray, source_spacing: tuple,
                      unique_name: str) -> np.ndarray:
        """
        Масштабирует изображение к целевому физическому разрешению.

        Если source_spacing = (0.2, 0.2) мм/пкс и target = (0.15, 0.15) мм/пкс,
        то scale_factor = 0.2/0.15 ≈ 1.33 → изображение увеличивается,
        потому что каждый исходный пиксель покрывает больше мм, чем целевой.

        Args:
            image: Исходное изображение
            source_spacing: (row_mm, col_mm) — физический размер пикселя источника
            unique_name: Имя файла для записи метаданных

        Returns:
            Масштабированное изображение
        """
        if self.target_spacing is None:
            # Масштабирование невозможно, сохраняем метаданные и возвращаем как есть
            self.scale_metadata[unique_name] = {
                "source_spacing_mm": list(source_spacing) if source_spacing else None,
                "target_spacing_mm": None,
                "scale_factor": [1.0, 1.0],
                "source_type": "unknown",
                "original_size": [image.shape[1], image.shape[0]],
                "rescaled_size": [image.shape[1], image.shape[0]],
            }
            return image

        scale_row = source_spacing[0] / self.target_spacing[0]
        scale_col = source_spacing[1] / self.target_spacing[1]

        # Если масштаб практически 1:1, не ресайзим (экономия качества)
        if abs(scale_row - 1.0) < 0.02 and abs(scale_col - 1.0) < 0.02:
            self.scale_metadata[unique_name] = {
                "source_spacing_mm": list(source_spacing),
                "target_spacing_mm": list(self.target_spacing),
                "scale_factor": [round(scale_row, 4), round(scale_col, 4)],
                "source_type": "dicom",
                "original_size": [image.shape[1], image.shape[0]],
                "rescaled_size": [image.shape[1], image.shape[0]],
            }
            return image

        new_h = int(round(image.shape[0] * scale_row))
        new_w = int(round(image.shape[1] * scale_col))

        # Выбираем интерполяцию: INTER_AREA для уменьшения, INTER_LINEAR для увеличения
        if scale_row < 1.0 and scale_col < 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR

        rescaled = cv2.resize(image, (new_w, new_h), interpolation=interp)

        self.scale_metadata[unique_name] = {
            "source_spacing_mm": [round(source_spacing[0], 6), round(source_spacing[1], 6)],
            "target_spacing_mm": [round(self.target_spacing[0], 6), round(self.target_spacing[1], 6)],
            "scale_factor": [round(scale_row, 4), round(scale_col, 4)],
            "source_type": "dicom",
            "original_size": [image.shape[1], image.shape[0]],
            "rescaled_size": [new_w, new_h],
        }

        return rescaled

    def get_dicom_spacing(self, file_path: Path) -> tuple | None:
        """Извлекает PixelSpacing из конкретного DICOM-файла."""
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            return extract_pixel_spacing(ds)
        except Exception:
            return None

    def save_as_png(self, image: np.ndarray, dst_path: Path):
        """Сохраняет массив как PNG файл."""
        cv2.imwrite(str(dst_path), image)

    # ------------------------------------------------------------------
    # Обработка train / test
    # ------------------------------------------------------------------
    def _process_file(self, file_path: Path, dst_path: Path, unique_name: str):
        """
        Общая логика обработки одного файла: чтение → масштабирование → сохранение.
        """
        img = self.read_image(file_path)
        if img is None:
            return False

        if self.target_spacing is not None:
            if self.is_dicom(file_path):
                spacing = self.get_dicom_spacing(file_path)
                if spacing is None:
                    # DICOM без PixelSpacing — используем медиану (scale=1)
                    spacing = self.target_spacing
                    source_type = "dicom_no_spacing"
                else:
                    source_type = "dicom"
            else:
                # JPEG — используем медиану как предполагаемый масштаб (scale=1)
                spacing = self.target_spacing
                source_type = "jpeg_assumed"

            img = self.rescale_image(img, spacing, unique_name)
            self.scale_metadata[unique_name]["source_type"] = source_type
        else:
            # Нет данных о масштабе — сохраняем как есть
            self.scale_metadata[unique_name] = {
                "source_spacing_mm": None,
                "target_spacing_mm": None,
                "scale_factor": [1.0, 1.0],
                "source_type": "dicom" if self.is_dicom(file_path) else "jpeg",
                "original_size": [img.shape[1], img.shape[0]],
                "rescaled_size": [img.shape[1], img.shape[0]],
            }

        self.save_as_png(img, dst_path)
        return True

    def process_train(self):
        """Обработка тренировочных данных с разделением на классы. Сохраняет ВСЕ снимки."""
        train_src = self.src_root / "train"
        if not train_src.exists():
            logging.warning("Папка train не найдена в исходных данных")
            return

        logging.info("Начало обработки TRAIN...")

        # Рекурсивный обход всех файлов в train
        all_files = sorted(list(train_src.rglob("*"))) # Сортировка для детерминизма
        for file_path in tqdm(all_files, desc="Processing Train"):
            if file_path.is_dir():
                continue

            # Проверка типа файла
            if not (self.is_dicom(file_path) or file_path.suffix.lower() in ['.jpg', '.jpeg']):
                continue

            # Определение целевой папки на основе пути
            target_subdir = None
            path_str = str(file_path)

            if any(folder in path_str for folder in self.normal_folders):
                target_subdir = "train/normal"
                stat_key = "train_normal"
            elif any(folder in path_str for folder in self.pathology_folders):
                target_subdir = "train/pathology"
                stat_key = "train_pathology"

            if target_subdir:
                # Генерируем уникальное имя на основе пути, чтобы избежать перезаписи
                # Используем slugify для очистки имени от кириллицы и спецсимволов
                relative_path = file_path.relative_to(train_src)
                parent_slug = slugify(str(relative_path.parent))
                stem_slug = slugify(file_path.stem)
                unique_name = f"{parent_slug}_{stem_slug}.png"

                dst_path = self.dst_root / target_subdir / unique_name
                if self._process_file(file_path, dst_path, unique_name):
                    self.stats[stat_key] += 1

    def process_test(self):
        """Обработка тестовых данных."""
        test_src = self.src_root / "test"
        if not test_src.exists():
            logging.warning("Папка test не найдена в исходных данных")
            return

        logging.info("Начало обработки TEST...")

        all_files = sorted(list(test_src.rglob("*")))
        for file_path in tqdm(all_files, desc="Processing Test"):
            if file_path.is_dir():
                continue


            if not (self.is_dicom(file_path) or file_path.suffix.lower() in ['.jpg', '.jpeg']):
                continue

            relative_path = file_path.relative_to(test_src)
            parent_slug = slugify(str(relative_path.parent))
            stem_slug = slugify(file_path.stem)
            unique_name = f"{parent_slug}_{stem_slug}.png"

            dst_path = self.dst_root / "test" / "images" / unique_name
            if self._process_file(file_path, dst_path, unique_name):
                self.stats["test"] += 1

    def save_scale_metadata(self):
        """
        Сохраняет метаданные масштабирования в JSON.

        Файл содержит для каждого изображения:
          - source_spacing_mm: исходный физический размер пикселя [row, col]
          - target_spacing_mm: целевой размер пикселя [row, col]
          - scale_factor: коэффициент масштабирования [row, col]
          - source_type: "dicom" | "dicom_no_spacing" | "jpeg_assumed"
          - original_size: [width, height] до масштабирования
          - rescaled_size: [width, height] после масштабирования

        Также содержит общую информацию:
          - target_spacing_mm: целевой масштаб (медиана)
          - total_dicom_with_spacing: кол-во DICOM с PixelSpacing
          - spacing_stats: статистика spacing'ов
        """
        metadata_path = self.dst_root / "scale_metadata.json"

        summary = {
            "target_spacing_mm": list(self.target_spacing) if self.target_spacing else None,
            "total_dicom_with_spacing": len(self.dicom_spacings),
        }

        if self.dicom_spacings:
            row_spacings = [s[0] for s in self.dicom_spacings]
            col_spacings = [s[1] for s in self.dicom_spacings]
            summary["spacing_stats"] = {
                "row_min": round(min(row_spacings), 6),
                "row_max": round(max(row_spacings), 6),
                "row_median": round(float(np.median(row_spacings)), 6),
                "row_mean": round(float(np.mean(row_spacings)), 6),
                "col_min": round(min(col_spacings), 6),
                "col_max": round(max(col_spacings), 6),
                "col_median": round(float(np.median(col_spacings)), 6),
                "col_mean": round(float(np.mean(col_spacings)), 6),
            }

        output = {
            "summary": summary,
            "images": self.scale_metadata,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"Метаданные масштабирования сохранены в {metadata_path}")

    def run(self):
        """Запуск полного цикла подготовки."""
        self.setup_directories()

        # 1. Первый проход: сбор PixelSpacing и расчёт целевого масштаба
        self.collect_pixel_spacings()

        # 2. Второй проход: обработка с масштабированием
        self.process_train()
        self.process_test()

        # 3. Сохранение метаданных масштабирования
        self.save_scale_metadata()

        logging.info("Обработка завершена!")
        logging.info(f"Статистика: {self.stats}")


if __name__ == "__main__":
    # Пути относительно корня проекта
    SRC_DATA = "data"
    DST_DATA = "data/processed"

    preparer = DatasetPreparer(SRC_DATA, DST_DATA)
    preparer.run()

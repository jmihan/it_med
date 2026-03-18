# MedAI Platform — Интеллектуальная система диагностики по медицинским изображениям

Платформа для автоматической диагностики патологий по рентгенографическим снимкам. Первый реализованный модуль — **дисплазия тазобедренного сустава (ТБС)**. Система решает две задачи: клиническую (помощь врачу) и образовательную (визуализация процесса анализа для обучающихся).

---

## Оглавление

- [Задача](#задача)
- [Архитектура](#архитектура)
- [Модуль: дисплазия ТБС](#модуль-дисплазия-тбс)
- [Интерфейс пользователя](#интерфейс-пользователя)
- [REST API](#rest-api)
- [Установка и запуск](#установка-и-запуск)
- [Обучение моделей](#обучение-моделей)
- [Добавление нового модуля](#добавление-нового-модуля)
- [Структура проекта](#структура-проекта)

---

## Задача

Система предназначена для диагностики **дисплазии тазобедренного сустава** у новорождённых по рентгенограммам. Выполняется автоматически:

1. **Классификация** — есть патология или нет (двойной вердикт: геометрия + ResNet).
2. **Метрический анализ** — расчёт ацетабулярного индекса, угол Хильгенрейнера, линия Перкина, расстояния h/d, признак триады Путти.
3. **XAI-объяснение** — визуализация зон влияния (Grad-CAM) и пошаговое текстовое объяснение (режим студента).

**Клинические нормы:**
- Угол Хильгенрейнера < 30° — норма
- Угол Хильгенрейнера ≥ 30° — дисплазия
- Головка бедра медиальнее линии Перкина — норма

---

## Архитектура

Система построена как **плагинная платформа**: ядро не зависит от конкретного заболевания — каждый модуль подключается через стандартный интерфейс `BaseMedicalPlugin`.

```
main.py ──────────────► ui/app.py (Streamlit :8501)
                                │
                        core/pipeline.py
                        (AnalysisPipeline)
                                │
run_api.py ──────────► api/server.py      core/registry.py
(FastAPI :8000)                           (PluginRegistry)
                                                │
                              ┌─────────────────┤
                  plugins/hip_dysplasia/     plugins/_stubs.py
                  plugin.py                 (LungAnalysis,
                      │                     ThyroidUltrasound)
                  metrics.py   xai.py
                  (Хильгенрейнер, Перкин,
                   h/d, триада Путти)
                      │
              models/keypoint_detector.py   models/classifier.py
              (YOLO Pose — 8 точек)         (ResNet18/50)
```

**Ключевые принципы:**
- **Расширяемость** — новый тип анализа = новый класс-наследник `BaseMedicalPlugin`, без изменений в ядре.
- **Два интерфейса** — Streamlit UI для демо/использования и FastAPI для интеграции в другие системы.
- **Двойной вердикт** — геометрический (YOLO Pose + расчёт углов) и нейросетевой (ResNet); режим приоритета настраивается в `config.yaml`.

---

## Модуль: дисплазия ТБС

### Пайплайн анализа

```
Изображение (DICOM / PNG / JPG)
        │
        ▼
  YOLO Pose → 8 ключевых точек
  L_TRC, R_TRC  — Y-образные хрящи (линия Хильгенрейнера)
  L_ACE, R_ACE  — края крыш вертлужных впадин
  L_FHC, R_FHC  — центры головок бёдер
  L_FMM, R_FMM  — метафизы бедренных костей
        │
        ▼
  Геометрические метрики:
  • Угол Хильгенрейнера (лев. / прав.)
  • Нарушение линии Перкина (лев. / прав.)
  • Расстояния h и d (высота и латеральное смещение)
  • TRC-расстояние (между Y-хрящами)
  • Признак триады Путти
        │
        ▼
  Геометрический вердикт (угол ≥ 30° = патология)
        │
        ▼
  ResNet18/50 — второй вердикт + Grad-CAM тепловая карта
  (на ROI-кропе таза, если доступен YOLO ROI-детектор)
        │
        ▼
  Итоговый вердикт (конфигурируемая стратегия)
```

### Ключевые точки YOLO Pose

| Точка   | Описание                           |
|---------|------------------------------------|
| L_TRC   | Y-хрящ слева                       |
| R_TRC   | Y-хрящ справа                      |
| L_ACE   | Край крыши вертлужной впадины (л.) |
| R_ACE   | Край крыши вертлужной впадины (п.) |
| L_FHC   | Центр головки бедра слева          |
| R_FHC   | Центр головки бедра справа         |
| L_FMM   | Метафиз бедренной кости слева      |
| R_FMM   | Метафиз бедренной кости справа     |

### Классификатор (ResNet)

- Backbone: ResNet18 / ResNet50 (ImageNet pretrained, fine-tuned)
- Вход: 224×224 RGB, нормализация ImageNet
- Выход: вероятности Normal / Pathology + уверенность
- Аугментации: ShiftScaleRotate, ElasticTransform, CLAHE, RandomGamma, RandomBrightnessContrast, GaussianBlur, GaussNoise, Sharpen, CoarseDropout
- **Горизонтальный и вертикальный флип намеренно исключены** — левый и правый суставы имеют диагностическое значение
- Поддерживает ROI-кропы: `classifier_resnet50_cropped.pt` обучен на вырезанных тазах

### Веса моделей

| Файл                               | Описание                              |
|------------------------------------|---------------------------------------|
| `weights/classifier.pt`            | ResNet18/50 (полный снимок, 90 МБ)    |
| `weights/classifier_resnet50_cropped.pt` | ResNet50, обученный на ROI-кропах |
| `weights/hip_keypoints_v1.pt`      | YOLO Pose — 8 анатомических точек     |
| `weights/hip_roi_v1.pt`            | YOLO — детектор области таза (ROI)    |

---

## Интерфейс пользователя

### Режим врача (клинический)

- Загрузка изображения (PNG / JPG / DICOM)
- Автоматический анализ: двойная классификация + полный набор метрик
- Аннотированное изображение с ключевыми точками, линиями и углами
- Панели GeometricVerdict и ResNetVerdict с индикаторами нормы/патологии
- Уверенность классификатора, углы обеих сторон
- Пакетная обработка нескольких снимков

### Режим студента (образовательный)

Пошаговая визуализация процесса диагностики:

1. **Поиск ключевых точек** — YOLO Pose находит 8 ориентиров на снимке
2. **Построение линий и расчёт углов** — линия Хильгенрейнера, ацетабулярные углы
3. **Тепловая карта (Grad-CAM)** — зоны влияния на решение ResNet для обоих классов

Каждый шаг разворачивается отдельно с текстовым объяснением и аннотированным изображением.

### Экспорт

- Результаты анализа экспортируются в **ZIP-архив** (аннотированные изображения, Grad-CAM карты)

---

## REST API

FastAPI сервер предоставляет следующие эндпоинты:

| Метод | Путь                    | Описание                       |
|-------|-------------------------|--------------------------------|
| GET   | `/api/v1/health`        | Health check                   |
| GET   | `/api/v1/plugins`       | Список доступных плагинов      |
| POST  | `/api/v1/analyze`       | Анализ одного изображения      |
| POST  | `/api/v1/analyze/batch` | Пакетный анализ                |

Аутентификация: заголовок `X-API-Key`.

Документация Swagger доступна по адресу `http://localhost:8000/docs`.

---

## Установка и запуск

### Локально

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить Streamlit UI
python main.py
# → http://localhost:8501

# Запустить REST API
python run_api.py
# → http://localhost:8000
# → http://localhost:8000/docs
```

### Docker Compose

```bash
docker-compose up
# Nginx :80 → Streamlit (/) + API (/api/*)
```

### Подготовка данных

```bash
# Распаковать тренировочные и тестовые данные из .7z архивов
python scripts/prepare_dataset.py
# Результат: data/processed/train/{normal,pathology}/
#            data/processed/test/{normal,pathology}/

# Опционально: вырезать ROI (тазобедренный сустав) для обучения
python scripts/crop_by_roi.py \
    --input-dir data/processed \
    --output-dir data/processed_cropped \
    --fallback-full
# Результат: data/processed_cropped/train/{normal,pathology}/
```

---

## Обучение моделей

### Классификатор (ResNet)

```bash
# На полных снимках
python scripts/train_classifier.py \
    --data-dir data/processed/train \
    --epochs 30 --batch-size 16 --backbone resnet18

# На ROI-кропах (рекомендуется)
python scripts/train_classifier.py \
    --data-dir data/processed_cropped/train \
    --epochs 30 --backbone resnet50

# Выход: weights/classifier.pt
```

Метрики обучения: Accuracy, Precision, Recall, F1, Specificity, AUC-ROC. Сохраняется лучший чекпоинт.

### Детектор ключевых точек (YOLO Pose)

```bash
# Конвертировать аннотации VIA JSON/CSV → формат YOLO Pose
python scripts/convert_annotations.py

# Обучить YOLO Pose (8 анатомических точек)
python scripts/train_keypoints.py
# Выход: weights/hip_keypoints_v1.pt
```

### Кросс-валидация классификатора

```bash
# K-fold кросс-валидация для оценки обобщаемости
python scripts/kfold_cv.py --data-dir data/processed/train --k 5
```

### Визуализация объяснений (Grad-CAM)

```bash
python scripts/run_explainer.py \
    --image path/to/image.jpg \
    --weights weights/classifier.pt \
    --method gradcam      # или gradcam++, eigencam
    --both                # тепловые карты для обоих классов
```

---

## Добавление нового модуля

1. Скопировать `plugins/template_plugin/` как шаблон
2. Унаследоваться от `BaseMedicalPlugin` и реализовать два метода:
   - `_load_models()` — инициализация моделей
   - `analyze(image) -> dict` — возвращает стандартизированный словарь результатов
3. Переопределить `get_ui_metadata()` — имя, иконка, определения метрик, шаги объяснения, слои визуализации
4. Зарегистрировать плагин в `core/registry.py`
5. Добавить в селектор анализа в `ui/app.py`

```python
# Минимальный пример нового плагина
from core.base_plugin import BaseMedicalPlugin

class LungPlugin(BaseMedicalPlugin):
    def _load_models(self):
        self.model = load_my_model()

    def analyze(self, image):
        result = self.model.predict(image)
        return {
            "pathology_detected": result.is_pathology,
            "metrics": {"confidence": result.confidence},
        }
```

---

## Структура проекта

```
app/
├── main.py                     # Точка входа: запуск Streamlit UI
├── run_api.py                  # Точка входа: запуск FastAPI
├── requirements.txt
├── docker-compose.yaml
│
├── core/
│   ├── base_plugin.py          # Абстрактный BaseMedicalPlugin
│   ├── registry.py             # PluginRegistry — реестр плагинов
│   ├── pipeline.py             # AnalysisPipeline — оркестрация
│   ├── image_io.py             # Загрузка DICOM / PNG / JPG
│   └── exceptions.py           # PluginNotFoundError, ImageLoadError
│
├── plugins/
│   ├── hip_dysplasia/
│   │   ├── plugin.py           # HipDysplasiaPlugin
│   │   ├── metrics.py          # Хильгенрейнер, Перкин, h/d, триада Путти
│   │   ├── xai.py              # Текстовые объяснения для студента
│   │   └── config.yaml         # Конфигурация плагина
│   ├── template_plugin/        # Шаблон для нового модуля
│   └── _stubs.py               # Демо-плагины: лёгкие, щитовидная железа
│
├── models/
│   ├── base_model.py           # BaseMLModel
│   ├── classifier.py           # ResNetClassifier (ResNet18/50)
│   └── keypoint_detector.py    # KeypointDetector (YOLO Pose)
│
├── training/
│   ├── trainer.py              # MedicalTrainer (Acc/Prec/Rec/F1/AUC)
│   ├── dataset.py              # MedicalImageDataset
│   └── augmentations.py        # Albumentations pipeline (без флипов)
│
├── visualization/
│   ├── explainers.py           # ModelExplainer (GradCAM / GradCAM++ / EigenCAM)
│   └── drawing.py              # ImageAnnotator — отрисовка точек и линий
│
├── ui/
│   ├── app.py                  # Главное Streamlit-приложение
│   ├── state.py                # Управление session state
│   ├── components/
│   │   └── sidebar.py          # Выбор плагина и режима
│   ├── page_views/
│   │   ├── single_analysis.py  # Анализ одного снимка
│   │   └── batch_processing.py # Пакетная обработка
│   └── views/
│       ├── doctor_view.py      # Режим врача (клинический)
│       └── student_view.py     # Режим студента (образовательный)
│
├── api/
│   ├── server.py               # FastAPI app factory
│   ├── auth.py                 # X-API-Key аутентификация
│   ├── config.py               # Настройки из .env
│   ├── routes/                 # health, plugins, analyze
│   └── schemas/                # Pydantic модели запросов/ответов
│
├── scripts/
│   ├── prepare_dataset.py      # Распаковка и подготовка данных
│   ├── train_classifier.py     # Обучение ResNet
│   ├── train_keypoints.py      # Обучение YOLO Pose
│   ├── crop_by_roi.py          # ROI-кроппинг снимков
│   ├── convert_annotations.py  # VIA JSON/CSV → YOLO Pose формат
│   ├── run_explainer.py        # Генерация Grad-CAM объяснений
│   ├── kfold_cv.py             # K-fold кросс-валидация
│   └── _utils.py               # Общие утилиты: find_image, draw_keypoints
│
├── data/
│   ├── train/                  # Исходные данные обучения (DICOM + JPG)
│   │   ├── Норма/
│   │   ├── Патология/
│   │   ├── норма_отдельные снимки/
│   │   ├── патология_jpg/
│   │   └── патология_отдельные файлы/
│   ├── test/                   # Тестовая выборка (24 снимка)
│   ├── processed/              # Подготовленные данные
│   │   ├── train/{normal,pathology}/
│   │   └── test/{normal,pathology}/
│   ├── processed_cropped/      # ROI-кропы
│   │   ├── train/{normal,pathology}/
│   │   └── test/{normal,pathology}/
│   ├── keypoints/              # YOLO Pose разметка (114 изображений)
│   └── annotations/            # VIA JSON/CSV аннотации
│
├── weights/
│   ├── classifier.pt                   # ResNet классификатор (90 МБ)
│   ├── classifier_resnet50_cropped.pt  # ResNet50 на ROI-кропах
│   ├── hip_keypoints_v1.pt             # YOLO Pose детектор точек
│   └── hip_roi_v1.pt                   # YOLO детектор области таза
│
└── deploy/
    └── nginx.conf              # Nginx reverse proxy
```

---

## Технологический стек

| Компонент           | Технология                                      |
|---------------------|-------------------------------------------------|
| UI                  | Streamlit                                       |
| REST API            | FastAPI + Uvicorn                               |
| Классификатор       | PyTorch, ResNet18/50 (ImageNet pretrained)      |
| Детектор точек      | Ultralytics YOLO Pose (YOLO11)                  |
| Объяснимость (XAI)  | pytorch-grad-cam (GradCAM, GradCAM++, EigenCAM) |
| Аугментации         | Albumentations                                  |
| Медицинские форматы | pydicom (DICOM)                                 |
| Деплой              | Docker, Docker Compose, Nginx                   |

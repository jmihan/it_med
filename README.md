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

1. **Классификация** — есть патология или нет.
2. **Метрический анализ** — расчёт ацетабулярного индекса (угол Хильгенрейнера) и проверка линии Перкина.
3. **XAI-объяснение** — визуализация того, на основе чего сделан вывод (для врача и студента).

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
                                  plugins/hip_dysplasia/plugin.py
                                        │               │
                              models/               metrics.py
                              keypoint_detector.py  (Хильгенрейнер,
                              classifier.py          Перкин)
                              (YOLO Pose)
                              (ResNet18/50)
```

**Ключевые принципы:**
- **Расширяемость** — новый тип анализа = новый класс-наследник `BaseMedicalPlugin`, без изменений в ядре.
- **Два интерфейса** — Streamlit UI для демо/использования и FastAPI для интеграции в другие системы.
- **Два уровня диагностики** — геометрический (YOLO Pose + расчёт углов) как основной; ResNet-классификатор как второе мнение.

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
  Расчёт метрик
  • Угол Хильгенрейнера (лев. / прав.)
  • Нарушение линии Перкина (лев. / прав.)
        │
        ▼
  Диагностика: патология если угол ≥ 30°
        │
        ▼
  ResNet18 (опционально) — второе мнение + Grad-CAM тепловая карта
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
- Выход: вероятности Normal / Pathology
- Аугментации: ShiftScaleRotate, ElasticTransform, RandomBrightnessContrast, GaussNoise
- **Горизонтальный и вертикальный флип намеренно исключены** — левый и правый суставы имеют диагностическое значение

---

## Интерфейс пользователя

### Режим врача (клинический)

- Загрузка изображения (PNG / JPG / DICOM)
- Автоматический анализ: классификация + метрики
- Аннотированное изображение с ключевыми точками, линией Хильгенрейнера и углами
- Вывод ацетабулярных углов обеих сторон с индикатором нормы/патологии
- Уверенность классификатора и метод анализа

### Режим студента (образовательный)

Пошаговая визуализация процесса диагностики:

1. **Поиск ключевых точек** — YOLO Pose находит 8 ориентиров на снимке
2. **Построение линий и расчёт углов** — линия Хильгенрейнера, ацетабулярные углы
3. **Тепловая карта (Grad-CAM)** — зоны влияния на решение ResNet

Каждый шаг разворачивается отдельно с текстовым объяснением и аннотированным изображением.

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
# Клонировать репозиторий и установить зависимости
pip install -r requirements.txt

# Запустить Streamlit UI
python main.py
# → http://localhost:8501

# Запустить REST API
python run_api.py
# → http://localhost:8000
```

### Docker Compose

```bash
docker-compose up
# Nginx :80 → Streamlit (/) + API (/api/*)
```

### Подготовка данных

```bash
# Распаковать тренировочные и тестовые данные
python scripts/prepare_dataset.py
# Результат: data/processed/train/{normal,pathology}/
#            data/processed/test/{normal,pathology}/
```

---

## Обучение моделей

### Классификатор (ResNet)

```bash
python scripts/train_classifier.py
# Вход:  data/processed/train/{normal,pathology}/
# Выход: weights/classifier.pt
```

Метрики обучения: Accuracy, Precision, Recall, F1, Specificity, AUC-ROC. Сохраняется лучший чекпоинт.

### Детектор ключевых точек (YOLO Pose)

```bash
# Сначала конвертировать аннотации в формат YOLO
python scripts/convert_annotations.py

# Обучить YOLO Pose
python scripts/train_keypoints.py
# Выход: weights/hip_keypoints_v1.pt
```

### Визуализация объяснений (Grad-CAM)

```bash
python scripts/run_explainer.py \
    --image path/to/image.jpg \
    --weights weights/classifier.pt \
    --method gradcam   # или gradcam++, eigencam
    --both             # сгенерировать тепловые карты для обоих классов
```

### Ноутбуки

| Файл                               | Содержание                         |
|------------------------------------|------------------------------------|
| `notebooks/01_eda.ipynb`           | Разведочный анализ данных (EDA)    |
| `notebooks/02_train_baseline.ipynb`| Обучение базовой модели            |
| `notebooks/03_inference_test.ipynb`| Тестирование инференса             |

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
├── docker-compose.yml
│
├── core/
│   ├── base_plugin.py          # Абстрактный BaseMedicalPlugin
│   ├── registry.py             # PluginRegistry — реестр плагинов
│   └── pipeline.py             # AnalysisPipeline — оркестрация
│
├── plugins/
│   ├── hip_dysplasia/
│   │   ├── plugin.py           # HipDysplasiaPlugin
│   │   ├── metrics.py          # Угол Хильгенрейнера, линия Перкина
│   │   └── xai.py              # Текстовые объяснения для студента
│   └── template_plugin/        # Шаблон для нового модуля
│
├── models/
│   ├── base_model.py           # BaseMLModel
│   ├── classifier.py           # ResNetClassifier (ResNet18/50)
│   └── keypoint_detector.py    # KeypointDetector (YOLO Pose)
│
├── training/
│   ├── trainer.py              # MedicalTrainer
│   ├── dataset.py              # MedicalImageDataset
│   └── augmentations.py        # Albumentations pipeline
│
├── visualization/
│   ├── explainers.py           # ModelExplainer (GradCAM / GradCAM++ / EigenCAM)
│   └── drawing.py              # ImageAnnotator — отрисовка точек и линий
│
├── ui/
│   ├── app.py                  # Главное Streamlit-приложение
│   ├── doctor_view.py          # Режим врача
│   └── student_view.py         # Режим студента
│
├── api/
│   ├── server.py               # FastAPI app
│   ├── auth.py                 # API key аутентификация
│   ├── routes/                 # health, plugins, analyze
│   └── schemas/                # Pydantic модели запросов/ответов
│
├── scripts/
│   ├── prepare_dataset.py      # Распаковка и подготовка данных
│   ├── train_classifier.py     # Обучение ResNet
│   ├── train_keypoints.py      # Обучение YOLO Pose
│   ├── convert_annotations.py  # Конвертация аннотаций в YOLO формат
│   └── run_explainer.py        # Генерация Grad-CAM объяснений
│
├── notebooks/
│   ├── 01_eda.ipynb            # Разведочный анализ данных
│   ├── 02_train_baseline.ipynb
│   └── 03_inference_test.ipynb
│
├── data/
│   ├── train/                  # Исходные данные обучения (DICOM + JPG)
│   ├── test/                   # Тестовая выборка
│   └── processed/              # Подготовленные данные
│       ├── train/{normal,pathology}/
│       └── test/{normal,pathology}/
│
└── weights/
    ├── classifier.pt           # ResNet классификатор
    └── hip_keypoints_v1.pt     # YOLO Pose детектор точек
```

---

## Технологический стек

| Компонент           | Технология                                      |
|---------------------|-------------------------------------------------|
| UI                  | Streamlit                                       |
| REST API            | FastAPI + Uvicorn                               |
| Классификатор       | PyTorch, ResNet18/50 (ImageNet pretrained)      |
| Детектор точек      | Ultralytics YOLO Pose                           |
| Объяснимость (XAI)  | pytorch-grad-cam (GradCAM, GradCAM++, EigenCAM) |
| Аугментации         | Albumentations                                  |
| Медицинские форматы | pydicom (DICOM)                                 |
| Деплой              | Docker, Docker Compose, Nginx                   |

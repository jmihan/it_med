import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(image_size: int = 224) -> A.Compose:
    """
    Аугментации для обучения на медицинских снимках (рентген/КТ/МРТ).

    Намеренно исключены:
    - HorizontalFlip / VerticalFlip — левый и правый суставы/органы имеют значение.
    - Сильные цветовые искажения — важна диагностическая ценность снимка.
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # Небольшие геометрические искажения
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,  # cv2.BORDER_CONSTANT — чёрный фон вместо зеркала
            p=0.7,
        ),

        # Имитация деформаций тканей (актуально для УЗИ и МРТ)
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            alpha_affine=5,
            border_mode=0,
            p=0.3,
        ),

        # Яркость и контраст — компенсируем разное оборудование и настройки экспозиции
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6,
        ),

        # Лёгкое размытие для устойчивости к шуму
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Шум — имитация артефактов сенсора
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

        # Нормализация ImageNet (модель предобучена на ImageNet)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_augmentations(image_size: int = 224) -> A.Compose:
    """
    Трансформации для валидации и инференса — только ресайз и нормализация.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

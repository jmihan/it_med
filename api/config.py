"""
Конфигурация API-сервера.
Настройки загружаются из переменных окружения или .env файла.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Сервер
API_HOST = os.getenv("MEDAI_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("MEDAI_API_PORT", "8000"))

# Аутентификация
API_KEYS: list[str] = [
    k.strip()
    for k in os.getenv("MEDAI_API_KEYS", "dev-key-change-me").split(",")
    if k.strip()
]

# CORS — разрешённые источники (через запятую)
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("MEDAI_CORS_ORIGINS", "*").split(",")
    if o.strip()
]

# Лимиты
MAX_FILE_SIZE_MB = int(os.getenv("MEDAI_MAX_FILE_SIZE_MB", "50"))
MAX_BATCH_SIZE = int(os.getenv("MEDAI_MAX_BATCH_SIZE", "50"))

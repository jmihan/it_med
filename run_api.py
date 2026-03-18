"""
Точка входа для REST API сервера.

Запуск:
    python run_api.py
    # Сервер стартует на http://0.0.0.0:8000
    # Документация: http://localhost:8000/docs
"""

import uvicorn

from api.config import API_HOST, API_PORT
from api.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "run_api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )

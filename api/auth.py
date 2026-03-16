"""
Аутентификация по API-ключу (заголовок X-API-Key).
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from api.config import API_KEYS

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Dependency: проверяет наличие и валидность API-ключа."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует заголовок X-API-Key",
        )
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недействительный API-ключ",
        )
    return api_key

import secrets
import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import (
    API_KEY,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="APIKey",
    description="API key required for prediction requests.",
)


_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    return api_key


def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _rate_limit_store[client_ip]

        timestamps[:] = [
            timestamp
            for timestamp in timestamps
            if timestamp > window_start
        ]

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )

        timestamps.append(now)


def reset_rate_limit_store() -> None:
    with _rate_limit_lock:
        _rate_limit_store.clear()

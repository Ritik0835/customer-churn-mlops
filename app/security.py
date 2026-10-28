import secrets
import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import (
    API_KEY,
    MAX_TRACKED_CLIENTS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    TRUSTED_PROXY_COUNT,
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


def get_client_ip(request: Request) -> str:
    """Return the address used as the rate-limit key.

    X-Forwarded-For can be forged by any client, so it is only used when
    TRUSTED_PROXY_COUNT > 0. In that case the entry added by the last trusted
    proxy (counted from the right) is used, which a client cannot control.
    """
    peer = request.client.host if request.client else "unknown"

    if TRUSTED_PROXY_COUNT <= 0:
        return peer

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]

    if len(hops) >= TRUSTED_PROXY_COUNT:
        return hops[-TRUSTED_PROXY_COUNT]

    return peer


def _prune_expired(window_start: float) -> None:
    """Drop clients with no request inside the current window."""
    expired = [
        client_ip
        for client_ip, timestamps in _rate_limit_store.items()
        if not timestamps or timestamps[-1] <= window_start
    ]
    for client_ip in expired:
        del _rate_limit_store[client_ip]


def check_rate_limit(request: Request) -> None:
    """Rate limit per client IP.

    Runs before API-key verification so that wrong-key guessing is limited too.
    """
    client_ip = get_client_ip(request)

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        if len(_rate_limit_store) >= MAX_TRACKED_CLIENTS:
            _prune_expired(window_start)

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
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )
        timestamps.append(now)


def reset_rate_limit_store() -> None:
    with _rate_limit_lock:
        _rate_limit_store.clear()

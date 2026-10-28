import os


MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

API_KEY = os.getenv("API_KEY")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))

RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
)

# Number of trusted reverse proxies in front of the app. 0 (default) means the
# X-Forwarded-For header is ignored, because any client can forge it. Set it to
# the number of proxy hops your platform adds (usually 1) to use the address
# appended by the last trusted proxy as the client IP.
TRUSTED_PROXY_COUNT = int(os.getenv("TRUSTED_PROXY_COUNT", "0"))

# Upper bound on distinct client keys kept by the in-memory rate limiter.
MAX_TRACKED_CLIENTS = int(os.getenv("MAX_TRACKED_CLIENTS", "10000"))

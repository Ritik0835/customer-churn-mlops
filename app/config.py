import os


MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

API_KEY = os.getenv("API_KEY")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))

RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
)

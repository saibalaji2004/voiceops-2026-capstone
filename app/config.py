"""Configuration for the VoiceOps ZentroMart capstone."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_model_list(list_name: str, legacy_name: str, default: str) -> list[str]:
    """Read comma-separated model IDs, falling back to the legacy single ID."""
    raw = os.getenv(list_name, "").strip()
    if not raw:
        raw = os.getenv(legacy_name, default).strip()

    models: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        model = item.strip()
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    return models or [default]


# Provider keys. Missing keys disable that provider.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Multi-model fallback lists. The singular variables remain supported for
# compatibility with older .env files. The list order is the preference order.
GEMINI_MODELS = _read_model_list(
    "GEMINI_MODELS", "GEMINI_MODEL", "gemini-3.6-flash"
)
GROQ_MODELS = _read_model_list(
    "GROQ_MODELS", "GROQ_MODEL", "openai/gpt-oss-20b"
)
OPENROUTER_MODELS = _read_model_list(
    "OPENROUTER_MODELS", "OPENROUTER_MODEL", "openrouter/free"
)

# Cloudflare Workers AI requires both an Account ID and a Workers AI API token.
# CLOUDFLARE_API_KEY is accepted as a compatibility alias for API_TOKEN.
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN", os.getenv("CLOUDFLARE_API_KEY", "")
).strip()
CLOUDFLARE_MODELS = _read_model_list(
    "CLOUDFLARE_MODELS",
    "CLOUDFLARE_MODEL",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)

# Legacy model names remain available to any existing code that imports them.
GEMINI_MODEL = GEMINI_MODELS[0]
GROQ_MODEL = GROQ_MODELS[0]
OPENROUTER_MODEL = OPENROUTER_MODELS[0]
CLOUDFLARE_MODEL = CLOUDFLARE_MODELS[0]

# Required later for Gnani speech-to-text and text-to-speech.
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "").strip()

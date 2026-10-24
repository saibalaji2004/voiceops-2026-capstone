"""Equal-peer multi-provider LLM router with fallback."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

from app import config


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
STATE_FILE = OUTPUT_DIR / "provider_router_state.json"

_LOCK = threading.RLock()


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    model: str
    kind: str
    base_url: str = ""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    attempted_providers: tuple[str, ...]


class AllProvidersUnavailable(RuntimeError):
    """Raised when no configured provider can answer."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _provider_list() -> list[Provider]:
    """Return every configured provider/model endpoint in preference order."""
    providers: list[Provider] = []

    if config.GEMINI_API_KEY:
        for model in config.GEMINI_MODELS:
            providers.append(
                Provider(
                    name="gemini",
                    api_key=config.GEMINI_API_KEY,
                    model=model,
                    kind="gemini",
                )
            )

    if config.GROQ_API_KEY:
        for model in config.GROQ_MODELS:
            providers.append(
                Provider(
                    name="groq",
                    api_key=config.GROQ_API_KEY,
                    model=model,
                    kind="openai-compatible",
                    base_url="https://api.groq.com/openai/v1",
                )
            )

    if config.OPENROUTER_API_KEY:
        for model in config.OPENROUTER_MODELS:
            providers.append(
                Provider(
                    name="openrouter",
                    api_key=config.OPENROUTER_API_KEY,
                    model=model,
                    kind="openai-compatible",
                    base_url="https://openrouter.ai/api/v1",
                )
            )

    if config.CLOUDFLARE_API_TOKEN and config.CLOUDFLARE_ACCOUNT_ID:
        cloudflare_base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{config.CLOUDFLARE_ACCOUNT_ID}/ai/v1"
        )
        for model in config.CLOUDFLARE_MODELS:
            providers.append(
                Provider(
                    name="cloudflare",
                    api_key=config.CLOUDFLARE_API_TOKEN,
                    model=model,
                    kind="openai-compatible",
                    base_url=cloudflare_base_url,
                )
            )

    return providers


def _endpoint_key(provider: Provider) -> str:
    """Use a model-specific cooldown key so one model limit doesn't block peers."""
    return f"{provider.name}:{provider.model}"


def configured_providers() -> list[dict[str, str]]:
    """Return configured endpoints without revealing credentials."""
    return [
        {"provider": provider.name, "model": provider.model}
        for provider in _provider_list()
    ]


def _read_state() -> dict[str, Any]:
    """Read round-robin position and provider cooldowns."""

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            raise ValueError("Invalid provider state")

    except (OSError, json.JSONDecodeError, ValueError):
        state = {}

    if not isinstance(state.get("cooldowns"), dict):
        state["cooldowns"] = {}

    state.setdefault("next_index", 0)

    return state


def _write_state(state: dict[str, Any]) -> None:
    """Persist state atomically."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporary = STATE_FILE.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    temporary.replace(STATE_FILE)


def _reserve_start_index(provider_count: int) -> int:
    """Rotate the first provider used for each request."""

    with _LOCK:
        state = _read_state()

        start = int(state.get("next_index", 0)) % provider_count
        state["next_index"] = (start + 1) % provider_count

        _write_state(state)

        return start


def _cooldown_remaining(provider_name: str) -> tuple[float, str]:
    """Return remaining cooldown time and its reason."""

    with _LOCK:
        state = _read_state()
        entry = state["cooldowns"].get(provider_name)

        if not isinstance(entry, dict):
            return 0.0, ""

        try:
            until = datetime.fromisoformat(entry["until_utc"])

            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)

        except (ValueError, TypeError, KeyError):
            state["cooldowns"].pop(provider_name, None)
            _write_state(state)
            return 0.0, ""

        remaining = (until - _now()).total_seconds()

        if remaining <= 0:
            state["cooldowns"].pop(provider_name, None)
            _write_state(state)
            return 0.0, ""

        return remaining, str(entry.get("reason", "unavailable"))


def _parse_duration_string(value: str) -> float | None:
    """Parse durations such as 2m30s, 1h5m, or 1d2h."""

    value = value.strip().lower()

    if value.isdigit():
        return float(value)

    match = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)d)?\s*"
        r"(?:(\d+(?:\.\d+)?)h)?\s*"
        r"(?:(\d+(?:\.\d+)?)m)?\s*"
        r"(?:(\d+(?:\.\d+)?)s)?",
        value,
    )

    if not match or not any(part is not None for part in match.groups()):
        return None

    days, hours, minutes, seconds = (
        float(part or 0) for part in match.groups()
    )

    return (
        days * 86400
        + hours * 3600
        + minutes * 60
        + seconds
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract provider retry delays where available."""

    message = str(exc)

    match = re.search(
        r"(?:PLEASE\s+)?RETRY IN\s+"
        r"(\d+)\s*H"
        r"(?:\s*(\d+)\s*M)?"
        r"(?:\s*([\d.]+)\s*S)?",
        message,
        flags=re.IGNORECASE,
    )

    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = float(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}

    retry_after = (
        headers.get("retry-after")
        or headers.get("Retry-After")
    )

    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except (ValueError, TypeError):
            parsed = _parse_duration_string(str(retry_after))

            if parsed is not None:
                return max(1.0, parsed)

    return None


def _is_daily_quota_error(message: str) -> bool:
    """Recognize common daily-quota error messages."""

    upper = message.upper()

    daily_markers = (
        "GENERATE_CONTENT_FREE_TIER_REQUESTS",
        "GENERATEREQUESTSPERDAYPERPROJECTPERMODEL-FREETIER",
        "FREE_MODEL_DAILY_REQUESTS",
        "FREE-MODELS-PER-DAY",
        "FREE MODELS PER DAY",
        "DAILY QUOTA",
        "DAILY LIMIT",
        "REQUESTS PER DAY",
        "DAILY FREE ALLOCATION",
        "FREE DAILY ALLOCATION",
        "DAILY NEURONS",
        "NEURONS PER DAY",
        "EXCEEDED THE DAILY LIMIT",
    )

    if any(marker in upper for marker in daily_markers):
        return True

    return bool(
        "429" in upper
        and "RESOURCE_EXHAUSTED" in upper
        and "PLEASE RETRY IN" in upper
        and re.search(r"RETRY IN\s+\d+\s*H", upper)
    )


def _record_failure_cooldown(
    provider_name: str,
    exc: Exception,
) -> None:
    """Temporarily skip providers reporting rate limits or service failures."""

    message = str(exc)
    upper = message.upper()

    daily = _is_daily_quota_error(message)
    status = getattr(exc, "status_code", None)
    retry_after = _retry_after_seconds(exc)

    if daily:
        seconds = (
            retry_after + 60
            if retry_after
            else 24 * 3600
        )
        reason = "daily quota exhausted"

    elif (
        status == 429
        or "429" in upper
        or "RATE LIMIT" in upper
        or "RESOURCE_EXHAUSTED" in upper
    ):
        seconds = retry_after if retry_after else 60
        reason = "rate limited or quota exhausted"

    elif status in (500, 502, 503, 504) or any(
        marker in upper
        for marker in (
            "UNAVAILABLE",
            "TIMEOUT",
            "502",
            "503",
            "504",
        )
    ):
        seconds = retry_after if retry_after else 30
        reason = "temporary service error"

    else:
        # Authentication errors and invalid models need correction;
        # don't permanently disable the provider.
        return

    seconds = max(1.0, min(float(seconds), 7 * 24 * 3600))
    until = _now() + timedelta(seconds=seconds)

    with _LOCK:
        state = _read_state()

        state["cooldowns"][provider_name] = {
            "until_utc": until.isoformat(),
            "reason": reason,
        }

        _write_state(state)


def _gemini_call(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str,
) -> tuple[str, str]:
    """Call the Gemini API."""

    client = genai.Client(api_key=provider.api_key)

    contents = []

    for message in messages:
        role = message.get("role", "user")

        if role not in ("user", "assistant", "model"):
            continue

        gemini_role = (
            "model"
            if role in ("assistant", "model")
            else "user"
        )

        content = str(message.get("content", ""))

        contents.append(
            types.Content(
                role=gemini_role,
                parts=[
                    types.Part.from_text(text=content)
                ],
            )
        )

    try:
        response = client.models.generate_content(
            model=provider.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )

        text = (response.text or "").strip()

        actual_model = (
            getattr(response, "model_version", None)
            or provider.model
        )

        if not text:
            raise RuntimeError("Gemini returned an empty text response")

        return text, str(actual_model)

    finally:
        client.close()


def _openai_compatible_call(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str,
) -> tuple[str, str]:
    """Call Groq or OpenRouter through the OpenAI-compatible interface."""

    if provider.name == "openrouter":
        model_id = provider.model.strip().lower()

        # Enforce the free-only OpenRouter configuration.
        if (
            model_id != "openrouter/free"
            and not model_id.endswith(":free")
        ):
            raise ValueError(
                "Free-only policy: OPENROUTER_MODEL must be "
                "'openrouter/free' or a model ID ending in ':free'."
            )

    client = OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=45.0,
        max_retries=0,
    )

    api_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for message in messages:
        role = message.get("role", "user")

        if role not in ("user", "assistant"):
            continue

        api_messages.append(
            {
                "role": role,
                "content": str(message.get("content", "")),
            }
        )

    try:
        response = client.chat.completions.create(
            model=provider.model,
            messages=api_messages,
            temperature=0.2,
            max_tokens = 500,
        )

        text = (
            response.choices[0].message.content or ""
        ).strip()

        actual_model = (
            getattr(response, "model", None)
            or provider.model
        )

        if not text:
            raise RuntimeError(
                f"{provider.name} returned an empty response"
            )

        return text, str(actual_model)

    finally:
        client.close()


def _call_provider(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str,
) -> tuple[str, str]:
    if provider.kind == "gemini":
        return _gemini_call(provider, messages, system_prompt)

    return _openai_compatible_call(
        provider,
        messages,
        system_prompt,
    )


def generate_reply(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> LLMResponse:
    """Rotate equally among providers, falling back when one fails."""

    providers = _provider_list()

    if not providers:
        raise AllProvidersUnavailable(
            "No text LLM providers are configured. Set at least one of "
            "GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, or a valid "
            "CLOUDFLARE_API_TOKEN plus CLOUDFLARE_ACCOUNT_ID."
        )

    start = _reserve_start_index(len(providers))
    attempted = []
    errors = []

    for offset in range(len(providers)):
        provider = providers[
            (start + offset) % len(providers)
        ]

        endpoint_key = _endpoint_key(provider)

        # OpenRouter's free-model daily pool and Cloudflare's daily neuron
        # allocation are shared across models. Gemini free quotas can be
        # model-specific, so a daily quota on one Gemini model must not block
        # its other models. Ignore any stale provider-wide Gemini cooldown
        # left by an older router version.
        if provider.name in {"openrouter", "cloudflare"}:
            provider_remaining, provider_reason = _cooldown_remaining(provider.name)
            if provider_remaining > 0:
                errors.append(
                    f"{provider.name}: provider-wide cooldown ({provider_reason}, "
                    f"about {int(provider_remaining)} seconds remaining)"
                )
                continue

        remaining, reason = _cooldown_remaining(endpoint_key)
        if remaining > 0:
            errors.append(
                f"{provider.name} ({provider.model}): cooling down ({reason}, "
                f"about {int(remaining)} seconds remaining)"
            )
            continue

        attempt_label = f"{provider.name}[{provider.model}]"
        attempted.append(attempt_label)

        print(
            f"[LLM Router] Trying {provider.name} "
            f"({provider.model})...",
            flush=True,
        )

        try:
            text, actual_model = _call_provider(
                provider,
                messages,
                system_prompt,
            )

            return LLMResponse(
                text=text,
                provider=provider.name,
                model=actual_model,
                attempted_providers=tuple(attempted),
            )

        except Exception as exc:
            _record_failure_cooldown(endpoint_key, exc)

            # OpenRouter's free-model request cap and Cloudflare's daily
            # neuron allocation are shared across their configured models.
            # Record those limits at provider level to avoid retrying every
            # model against the same exhausted pool. Keep Gemini daily limits
            # model-specific: Gemini's errors can explicitly be per-model.
            error_text = str(exc)
            upper_error = error_text.upper()

            openrouter_shared_limit = (
                provider.name == "openrouter"
                and (
                    "FREE-MODELS-PER-DAY" in upper_error
                    or "FREE MODELS PER DAY" in upper_error
                )
            )
            cloudflare_shared_limit = (
                provider.name == "cloudflare"
                and _is_daily_quota_error(error_text)
            )

            if openrouter_shared_limit or cloudflare_shared_limit:
                _record_failure_cooldown(provider.name, exc)

            short_error = str(exc).replace("\n", " ")[:240]

            errors.append(
                f"{provider.name}: {type(exc).__name__}: "
                f"{short_error}"
            )

            print(
                f"[LLM Router] {provider.name} failed; "
                "trying another configured provider.",
                flush=True,
            )

    raise AllProvidersUnavailable(
        "All configured LLM providers are unavailable or cooling down. "
        + " | ".join(errors)
    )


def router_status() -> list[dict[str, str]]:
    """Report status for every configured model endpoint, without secrets."""
    results: list[dict[str, str]] = []

    for provider in _provider_list():
        seconds, reason = _cooldown_remaining(_endpoint_key(provider))
        if seconds > 0:
            status = (
                f"cooldown: {reason}, "
                f"{int(seconds)} seconds remaining"
            )
        else:
            status = "ready"

        results.append({
            "provider": provider.name,
            "model": provider.model,
            "status": status,
        })

    return results

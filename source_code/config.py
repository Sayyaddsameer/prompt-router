"""
config.py — Application settings read entirely from environment variables.
Zero hardcoded values: models, thresholds, file paths — all configurable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str                             # required; no default
    classifier_model: str = "gpt-4o-mini"           # cheap + fast for classification
    responder_model: str = "gpt-4o-mini"            # higher-quality for responses
    openai_base_url: str = "https://api.openai.com/v1"

    # ── Routing ───────────────────────────────────────────────────────────────
    # Below this confidence score the intent is treated as 'unclear'
    confidence_threshold: float = 0.7
    # Path to the JSON file holding expert system prompts (relative to CWD)
    prompts_file: str = "prompts.json"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = "route_log.jsonl"
    log_level: str = "INFO"

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = 8000
    host: str = "0.0.0.0"

    # ── Classifier prompt template (single source of truth) ───────────────────
    # Kept in settings so it can be overridden via env without code changes.
    classifier_system_prompt: str = (
        "Your task is to classify the user's intent. "
        "Choose exactly ONE label from this list: code, data, writing, career, unclear. "
        "Respond with ONLY a valid JSON object on a single line — no markdown, no explanation, no extra text. "
        "The object must have exactly two keys: "
        "'intent' (the chosen label as a string) and "
        "'confidence' (your certainty as a float between 0.0 and 1.0). "
        "Example: {\"intent\": \"code\", \"confidence\": 0.95}"
    )

    # Labels the classifier is allowed to return
    valid_intents: list[str] = ["code", "data", "writing", "career", "unclear"]

    # Fallback when JSON parsing fails
    fallback_intent: str = "unclear"
    fallback_confidence: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

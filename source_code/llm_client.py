"""
llm_client.py — Thin, reusable wrapper around the OpenAI chat completions API.

All model names, base URLs, and API keys come from Settings — nothing hardcoded.
Provides two helpers:
  - chat()  : generic chat call, returns the raw assistant message string.
  - json_chat(): like chat(), but validates the response is parseable JSON.
"""
from __future__ import annotations

import json
import logging
from typing import List

from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


def _build_client() -> AsyncOpenAI:
    """Construct an AsyncOpenAI client from settings."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


# Module-level lazy client (created once, reused for all calls)
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


async def chat(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Make a single chat completion call and return the assistant's text.

    :param model:         OpenAI model identifier (from settings).
    :param system_prompt: The system role message.
    :param user_message:  The user role message.
    :param temperature:   Sampling temperature (lower = more deterministic).
    :param max_tokens:    Maximum tokens in the response.
    :returns:             The assistant message content as a plain string.
    :raises:              openai.OpenAIError on API failures.
    """
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    logger.debug("LLM call → model=%s, temp=%.1f, tokens=%d", model, temperature, max_tokens)

    response = await get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    logger.debug("LLM response ← %d chars", len(content))
    return content


async def json_chat(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> dict:
    """
    Like chat(), but attempts to parse the response as JSON.
    Tries two strategies:
      1. Direct json.loads on the raw response.
      2. Extracts the first {...} substring and parses that.
    Returns an empty dict on any parse failure (caller must handle).

    :returns: Parsed dict or {} on failure.
    """
    raw = await chat(
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first {...} block (handles markdown code-fenced responses)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("json_chat: could not parse LLM response as JSON. raw=%r", raw[:200])
    return {}

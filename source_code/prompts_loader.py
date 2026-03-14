"""
prompts_loader.py — Loads expert system prompts from the JSON config file.

Prompts are NEVER hardcoded in business logic. This module provides a single
cached dictionary keyed by intent label, loaded from the path specified in
settings.prompts_file (relative to the working directory).
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

from config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_prompts() -> Dict[str, dict]:
    """
    Load and return the full prompts dictionary from the JSON config file.

    :returns: dict keyed by intent label, each value has:
              - label (str):        Human-readable persona name
              - emoji (str):        Display emoji
              - system_prompt (str): The actual system prompt text
    :raises FileNotFoundError: if the prompts file does not exist.
    :raises json.JSONDecodeError: if the file contains invalid JSON.
    """
    settings = get_settings()
    path = Path(settings.prompts_file)
    if not path.is_absolute():
        # Resolve relative to the working directory (set to /app in the container)
        path = Path.cwd() / path

    logger.info("Loading prompts from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, dict] = json.load(fh)

    logger.info("Loaded %d expert personas: %s", len(data), list(data.keys()))
    return data


def get_system_prompt(intent: str) -> str:
    """
    Return the system prompt text for the given intent label.
    Falls back to the 'unclear' prompt if the intent is not found.

    :param intent: Intent label string (e.g. 'code', 'data').
    :returns:      System prompt string.
    """
    prompts = load_prompts()
    entry = prompts.get(intent) or prompts.get("unclear", {})
    return entry.get("system_prompt", "You are a helpful assistant.")


def get_persona_label(intent: str) -> str:
    """Return the human-readable persona label for display purposes."""
    prompts = load_prompts()
    entry = prompts.get(intent, {})
    emoji = entry.get("emoji", "🤖")
    label = entry.get("label", intent.title())
    return f"{emoji} {label}"

"""
route_logger.py — Structured JSONL logging for every routing decision.

Each call to log_route() appends one JSON object (one line) to the
file specified in settings.log_file. The log is append-only and thread-safe
for single-process use (asyncio).

Required fields per log entry:
  - intent         (str)
  - confidence     (float)
  - user_message   (str)
  - final_response (str)

Additional enrichment fields:
  - timestamp      (ISO-8601 UTC)
  - override       (bool)
  - persona        (str)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import get_settings
from prompts_loader import get_persona_label

logger = logging.getLogger(__name__)


def log_route(
    user_message: str,
    intent_result: dict,
    final_response: str,
) -> None:
    """
    Append one JSON Lines entry to the route log file.

    :param user_message:   The original user input.
    :param intent_result:  The dict from classify_intent().
    :param final_response: The response string returned to the user.
    """
    settings = get_settings()
    log_path = Path(settings.log_file)

    intent = intent_result.get("intent", settings.fallback_intent)
    confidence = intent_result.get("confidence", settings.fallback_confidence)
    override = intent_result.get("override", False)

    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "override": override,
        "persona": get_persona_label(intent),
        "user_message": user_message,
        "final_response": final_response,
    }

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Logged route entry to %s", log_path)
    except OSError as exc:
        # Logging failure should never crash the main response flow
        logger.error("log_route: could not write to %s: %s", log_path, exc)

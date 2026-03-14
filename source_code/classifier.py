"""
classifier.py — Intent classification via LLM.

classify_intent(message) is the primary public function.

It implements the stretch goals:
  - Manual override: messages prefixed with @<intent> skip the LLM entirely.
  - Confidence threshold: low-confidence predictions are downgraded to 'unclear'.
  - Graceful fallback: any parse/API error → {"intent": "unclear", "confidence": 0.0}
"""
from __future__ import annotations

import logging
import re

from config import get_settings
from llm_client import json_chat

logger = logging.getLogger(__name__)

# Regex for the manual override prefix: @code, @data, @writing, @career, @unclear
_OVERRIDE_RE = re.compile(r"^@(\w+)\s*", re.IGNORECASE)


def _make_fallback() -> dict:
    """Return the safe fallback classification."""
    settings = get_settings()
    return {
        "intent": settings.fallback_intent,
        "confidence": settings.fallback_confidence,
        "override": False,
    }


def _check_manual_override(message: str) -> tuple[dict | None, str]:
    """
    If the message starts with @<intent>, return an override classification
    and the stripped message (without the prefix).

    Returns (classification_dict, cleaned_message) or (None, original_message).
    """
    settings = get_settings()
    match = _OVERRIDE_RE.match(message)
    if not match:
        return None, message

    raw_intent = match.group(1).lower()
    # Only honour overrides to valid intent labels
    if raw_intent in settings.valid_intents:
        stripped = message[match.end():]
        classification = {
            "intent": raw_intent,
            "confidence": 1.0,
            "override": True,
        }
        logger.info("Manual override detected → intent=%s", raw_intent)
        return classification, stripped

    return None, message


async def classify_intent(message: str) -> dict:
    """
    Classify the intent of *message* using an LLM call.

    Steps:
      1. Check for a manual @<intent> override prefix.
      2. If no override, call the LLM classifier with a focused prompt.
      3. Validate the returned dict; apply confidence threshold.
      4. On any failure, return the safe 'unclear' fallback.

    :param message: Raw user message string.
    :returns: dict with keys: intent (str), confidence (float), override (bool).
              The 'override' key signals whether routing was manually forced.
    """
    settings = get_settings()

    # ── Step 1: manual override ────────────────────────────────────────────
    override_result, cleaned_message = _check_manual_override(message)
    if override_result is not None:
        return {**override_result, "_cleaned_message": cleaned_message}

    # ── Step 2: LLM classification ────────────────────────────────────────
    try:
        raw = await json_chat(
            model=settings.classifier_model,
            system_prompt=settings.classifier_system_prompt,
            user_message=message,
            temperature=0.0,
            max_tokens=64,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("classify_intent: LLM call failed: %s", exc, exc_info=True)
        return _make_fallback()

    # ── Step 3: validate parsed response ──────────────────────────────────
    if not raw:
        logger.warning("classify_intent: empty/unparseable LLM response → fallback")
        return _make_fallback()

    intent = str(raw.get("intent", "")).lower().strip()
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # Reject unknown labels
    if intent not in settings.valid_intents:
        logger.warning("classify_intent: unknown intent '%s' → fallback", intent)
        return _make_fallback()

    # ── Step 4: confidence threshold ──────────────────────────────────────
    if confidence < settings.confidence_threshold and intent != "unclear":
        logger.info(
            "classify_intent: confidence %.2f below threshold %.2f for intent='%s' → unclear",
            confidence, settings.confidence_threshold, intent,
        )
        return {
            "intent": "unclear",
            "confidence": confidence,
            "override": False,
        }

    logger.info(
        "classify_intent: intent='%s', confidence=%.2f", intent, confidence
    )
    return {"intent": intent, "confidence": confidence, "override": False}

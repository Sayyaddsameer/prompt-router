"""
router.py — Intent-based routing to expert personas.

route_and_respond(message, intent_result) is the primary public function.

It:
  1. Resolves the intent label (handling the 'unclear' path).
  2. Looks up the matching system prompt from the prompts config.
  3. Makes a second LLM call with that specialist system prompt.
  4. Returns the final response string.
"""
from __future__ import annotations

import logging

from config import get_settings
from llm_client import chat
from prompts_loader import get_system_prompt, get_persona_label

logger = logging.getLogger(__name__)


async def route_and_respond(message: str, intent_result: dict) -> str:
    """
    Select the correct expert system prompt and generate a final response.

    If the intent is 'unclear' (either classified as such or low confidence),
    the 'unclear' persona is used to ask a targeted clarifying question rather
    than guessing or routing to a default expert.

    If a manual override cleaned message is present in intent_result
    (key '_cleaned_message'), the stripped message (without the @prefix) is
    used for the LLM call so the persona receives a clean user query.

    :param message:       The original user message.
    :param intent_result: The dict returned by classify_intent().
    :returns:             The final response string from the LLM.
    """
    settings = get_settings()
    intent = intent_result.get("intent", settings.fallback_intent)
    confidence = intent_result.get("confidence", settings.fallback_confidence)
    is_override = intent_result.get("override", False)

    # Use cleaned message if a manual @prefix was stripped by the classifier
    effective_message = intent_result.get("_cleaned_message", message) or message

    persona = get_persona_label(intent)
    system_prompt = get_system_prompt(intent)

    logger.info(
        "Routing → persona=%s, intent=%s, confidence=%.2f, override=%s",
        persona, intent, confidence, is_override,
    )

    try:
        response = await chat(
            model=settings.responder_model,
            system_prompt=system_prompt,
            user_message=effective_message,
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("route_and_respond: LLM call failed: %s", exc, exc_info=True)
        response = (
            "I'm sorry, I encountered an error generating a response. "
            "Please try again in a moment."
        )

    logger.info("Response generated (%d chars) for intent='%s'", len(response), intent)
    return response

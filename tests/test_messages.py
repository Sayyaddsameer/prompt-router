"""
tests/test_messages.py — Integration test running all 15 required messages.

Requires a real OPENAI_API_KEY in the environment.
Run: pytest tests/test_messages.py -v -s

Each test sends one of the 15 required messages through the full pipeline
and validates the output structure. No assertions on exact intent labels
since those depend on the LLM — we check the structural guarantees instead.
"""
from __future__ import annotations

import asyncio
import os

import pytest

# Skip entire module if no API key is available (e.g. CI without secrets)
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

# The required 15 test messages from the specification
TEST_MESSAGES = [
    "how do i sort a list of objects in python?",
    "explain this sql query for me",
    "This paragraph sounds awkward, can you help me fix it?",
    "I'm preparing for a job interview, any tips?",
    "what's the average of these numbers: 12, 45, 23, 67, 34",
    "Help me make this better.",
    "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.",
    "hey",
    "Can you write me a poem about clouds?",
    "Rewrite this sentence to be more professional.",
    "I'm not sure what to do with my career.",
    "what is a pivot table",
    "fxi thsi bug pls: for i in range(10) print(i)",
    "How do I structure a cover letter?",
    "My boss says my writing is too verbose.",
]

# Additional edge-case messages (stretch goal)
EDGE_CASE_MESSAGES = [
    "@code Fix this: print('hello'",          # manual override — should → code
    "@data what is the median of 1,2,3,4,5",  # manual override — should → data
    "a",                                       # single character
    "x" * 500,                                 # very long message
]

ALL_VALID_INTENTS = {"code", "data", "writing", "career", "unclear"}


async def _run_message(message: str) -> dict:
    from classifier import classify_intent
    from router import route_and_respond
    from route_logger import log_route

    intent_result = await classify_intent(message)
    response = await route_and_respond(message, intent_result)
    log_route(message, intent_result, response)
    return {"intent_result": intent_result, "response": response}


@pytest.mark.asyncio
@pytest.mark.parametrize("message", TEST_MESSAGES)
async def test_required_message(message: str):
    result = await _run_message(message)
    intent_result = result["intent_result"]
    response = result["response"]

    # Structural guarantees (core requirements):
    assert "intent" in intent_result, "intent key missing"
    assert "confidence" in intent_result, "confidence key missing"
    assert intent_result["intent"] in ALL_VALID_INTENTS, f"Unknown intent: {intent_result['intent']}"
    assert 0.0 <= intent_result["confidence"] <= 1.0, "Confidence out of range"
    assert isinstance(response, str) and len(response) > 0, "Empty response"


@pytest.mark.asyncio
@pytest.mark.parametrize("message", EDGE_CASE_MESSAGES)
async def test_edge_case_message(message: str):
    result = await _run_message(message)
    intent_result = result["intent_result"]
    response = result["response"]

    assert intent_result["intent"] in ALL_VALID_INTENTS
    assert isinstance(response, str) and len(response) > 0

    # Manual overrides should always have confidence=1.0
    if message.startswith("@"):
        assert intent_result.get("override") is True
        assert intent_result["confidence"] == 1.0

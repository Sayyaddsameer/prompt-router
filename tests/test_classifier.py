"""
tests/test_classifier.py — Unit tests for classify_intent().

These tests use mocking to avoid real OpenAI API calls.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# classify_intent — happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classifies_code():
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "code", "confidence": 0.95})):
        from classifier import classify_intent
        result = await classify_intent("how do i sort a list in python?")
    assert result["intent"] == "code"
    assert result["confidence"] == 0.95
    assert result["override"] is False


@pytest.mark.asyncio
async def test_classifies_data():
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "data", "confidence": 0.91})):
        from classifier import classify_intent
        result = await classify_intent("what is the mean of 12, 45, 23?")
    assert result["intent"] == "data"


@pytest.mark.asyncio
async def test_classifies_writing():
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "writing", "confidence": 0.88})):
        from classifier import classify_intent
        result = await classify_intent("my paragraph sounds awkward, help me")
    assert result["intent"] == "writing"


@pytest.mark.asyncio
async def test_classifies_career():
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "career", "confidence": 0.87})):
        from classifier import classify_intent
        result = await classify_intent("I'm preparing for a job interview")
    assert result["intent"] == "career"


# ---------------------------------------------------------------------------
# classify_intent — graceful error handling (Core Requirement 6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_defaults_to_unclear():
    """LLM returning invalid JSON must not crash; defaults to unclear/0.0."""
    with patch("classifier.json_chat", new=AsyncMock(return_value={})):
        from classifier import classify_intent
        result = await classify_intent("some message")
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_unknown_intent_label_defaults_to_unclear():
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "poetry", "confidence": 0.9})):
        from classifier import classify_intent
        result = await classify_intent("write me a sonnet")
    assert result["intent"] == "unclear"


@pytest.mark.asyncio
async def test_api_exception_defaults_to_unclear():
    with patch("classifier.json_chat", new=AsyncMock(side_effect=RuntimeError("network error"))):
        from classifier import classify_intent
        result = await classify_intent("any message")
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# classify_intent — confidence threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_low_confidence_downgraded_to_unclear():
    """Confidence below threshold (0.7 by default) → intent becomes unclear."""
    with patch("classifier.json_chat", new=AsyncMock(return_value={"intent": "code", "confidence": 0.5})):
        from classifier import classify_intent
        result = await classify_intent("hmm maybe fix this")
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.5   # original confidence preserved


# ---------------------------------------------------------------------------
# classify_intent — manual override (Stretch Goal)
# ---------------------------------------------------------------------------

def test_manual_override_code():
    from classifier import _check_manual_override
    result, cleaned = _check_manual_override("@code Fix this bug in my script")
    assert result is not None
    assert result["intent"] == "code"
    assert result["confidence"] == 1.0
    assert result["override"] is True
    assert cleaned == "Fix this bug in my script"


def test_manual_override_data():
    from classifier import _check_manual_override
    result, cleaned = _check_manual_override("@data what is the median?")
    assert result["intent"] == "data"
    assert cleaned == "what is the median?"


def test_manual_override_unknown_intent_ignored():
    """@unknownintent should NOT be treated as a valid override."""
    from classifier import _check_manual_override
    result, cleaned = _check_manual_override("@poetry write me a haiku")
    assert result is None
    assert cleaned == "@poetry write me a haiku"


def test_no_override_prefix():
    from classifier import _check_manual_override
    result, cleaned = _check_manual_override("how do i fix this bug?")
    assert result is None
    assert cleaned == "how do i fix this bug?"

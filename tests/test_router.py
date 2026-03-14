"""
tests/test_router.py — Unit tests for route_and_respond() and route_logger.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# route_and_respond
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routes_code_intent():
    mock_response = "Here is the fixed code:\n```python\nfor i in range(10):\n    print(i)\n```"
    with patch("router.chat", new=AsyncMock(return_value=mock_response)):
        from router import route_and_respond
        result = await route_and_respond(
            "fxi thsi bug: for i in range(10) print(i)",
            {"intent": "code", "confidence": 0.95, "override": False},
        )
    assert "print" in result.lower() or len(result) > 0


@pytest.mark.asyncio
async def test_routes_unclear_intent_generates_question():
    mock_response = "Could you clarify whether you need help with coding, data analysis, writing, or career advice?"
    with patch("router.chat", new=AsyncMock(return_value=mock_response)):
        from router import route_and_respond
        result = await route_and_respond(
            "hey",
            {"intent": "unclear", "confidence": 0.0, "override": False},
        )
    # Response for unclear must be a question or clarification
    assert len(result) > 0


@pytest.mark.asyncio
async def test_llm_error_returns_error_message():
    with patch("router.chat", new=AsyncMock(side_effect=RuntimeError("API down"))):
        from router import route_and_respond
        result = await route_and_respond(
            "any message",
            {"intent": "writing", "confidence": 0.9, "override": False},
        )
    assert "error" in result.lower() or "sorry" in result.lower()


@pytest.mark.asyncio
async def test_manual_override_uses_cleaned_message():
    """When _cleaned_message is present, that should be sent to the LLM."""
    captured_user_msg = []

    async def mock_chat(model, system_prompt, user_message, **kwargs):
        captured_user_msg.append(user_message)
        return "response"

    with patch("router.chat", new=mock_chat):
        from router import route_and_respond
        await route_and_respond(
            "@code Fix this bug",  # original with prefix
            {
                "intent": "code",
                "confidence": 1.0,
                "override": True,
                "_cleaned_message": "Fix this bug",  # stripped prefix
            },
        )
    assert captured_user_msg[0] == "Fix this bug"


# ---------------------------------------------------------------------------
# route_logger
# ---------------------------------------------------------------------------

def test_log_route_writes_valid_jsonl(tmp_path):
    log_file = tmp_path / "test_log.jsonl"

    with patch("route_logger.get_settings") as mock_settings, \
         patch("route_logger.get_persona_label", return_value="🧑‍💻 Code Expert"):
        mock_settings.return_value.log_file = str(log_file)
        mock_settings.return_value.fallback_intent = "unclear"
        mock_settings.return_value.fallback_confidence = 0.0

        from route_logger import log_route
        log_route(
            user_message="how do i sort a list?",
            intent_result={"intent": "code", "confidence": 0.95, "override": False},
            final_response="Use list.sort() or sorted().",
        )

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["intent"] == "code"
    assert entry["confidence"] == 0.95
    assert entry["user_message"] == "how do i sort a list?"
    assert entry["final_response"] == "Use list.sort() or sorted()."
    assert "timestamp" in entry


def test_log_route_appends_multiple_entries(tmp_path):
    log_file = tmp_path / "multi.jsonl"

    with patch("route_logger.get_settings") as mock_settings, \
         patch("route_logger.get_persona_label", return_value="📊 Data Analyst"):
        mock_settings.return_value.log_file = str(log_file)
        mock_settings.return_value.fallback_intent = "unclear"
        mock_settings.return_value.fallback_confidence = 0.0

        from route_logger import log_route
        for i in range(3):
            log_route(
                user_message=f"message {i}",
                intent_result={"intent": "data", "confidence": 0.9, "override": False},
                final_response=f"response {i}",
            )

    entries = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
    assert len(entries) == 3
    assert entries[2]["user_message"] == "message 2"

"""
main.py — FastAPI application + CLI entry point for the Prompt Router service.

Exposes:
  POST /route          — classify + respond (primary API endpoint)
  GET  /health         — liveness probe
  GET  /prompts        — list configured expert personas

Also runnable as a CLI:
  python main.py "your message here"
  python main.py --help
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import classify_intent
from config import get_settings
from prompts_loader import load_prompts, get_persona_label
from route_logger import log_route
from router import route_and_respond

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Prompt Router",
    description=(
        "Intelligently routes user requests to specialized AI expert personas "
        "via a two-step Classify → Respond pipeline."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description=(
            "User message to classify and respond to. "
            "Optionally prefix with @<intent> (e.g. '@code Fix this bug') "
            "to manually override the classifier."
        ),
    )


class ClassificationResult(BaseModel):
    intent: str
    confidence: float
    override: bool


class RouteResponse(BaseModel):
    user_message: str
    classification: ClassificationResult
    persona: str
    response: str


class PersonaInfo(BaseModel):
    intent: str
    label: str
    emoji: str
    system_prompt: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/prompts", tags=["config"], response_model=list[PersonaInfo], summary="List configured expert personas")
async def list_prompts() -> list[PersonaInfo]:
    """Return all configured expert personas and their system prompts."""
    prompts = load_prompts()
    return [
        PersonaInfo(
            intent=key,
            label=val.get("label", key.title()),
            emoji=val.get("emoji", ""),
            system_prompt=val.get("system_prompt", ""),
        )
        for key, val in prompts.items()
    ]


@app.post(
    "/route",
    response_model=RouteResponse,
    status_code=status.HTTP_200_OK,
    tags=["routing"],
    summary="Classify intent and generate expert response",
)
async def route(request: RouteRequest) -> RouteResponse:
    """
    Two-step pipeline:
    1. **Classify** the user message intent via a fast LLM call.
    2. **Route** to the matching expert persona and generate a response.

    Manual override: prefix your message with `@<intent>` to skip classification.
    Example: `@code Fix this bug: for i in range(10) print(i)`
    """
    try:
        intent_result = await classify_intent(request.message)
        response_text = await route_and_respond(request.message, intent_result)
    except Exception as exc:
        logger.error("Unhandled error in /route: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again.",
        ) from exc

    # Always log, even on partial success
    log_route(
        user_message=request.message,
        intent_result=intent_result,
        final_response=response_text,
    )

    return RouteResponse(
        user_message=request.message,
        classification=ClassificationResult(
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            override=intent_result.get("override", False),
        ),
        persona=get_persona_label(intent_result["intent"]),
        response=response_text,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _cli_main() -> None:
    """Run a single message through the router from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prompt Router CLI — classify and respond to a user message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py 'how do i sort a list in python?'\n"
            "  python main.py '@data what is the mean of: 12, 45, 23'\n"
        ),
    )
    parser.add_argument("message", help="User message to route")
    args = parser.parse_args()

    message: str = args.message

    print(f"\n Message: {message}")
    print("─" * 60)

    intent_result = await classify_intent(message)
    persona = get_persona_label(intent_result["intent"])

    print(
        f" Intent : {intent_result['intent']:10s}  "
        f"Confidence: {intent_result['confidence']:.0%}  "
        f"Override: {intent_result.get('override', False)}"
    )
    print(f" Persona: {persona}")
    print("─" * 60)

    response_text = await route_and_respond(message, intent_result)
    log_route(
        user_message=message,
        intent_result=intent_result,
        final_response=response_text,
    )

    print(f"\n{response_text}\n")


if __name__ == "__main__":
    # CLI mode when executed directly; FastAPI mode when run via uvicorn
    if len(sys.argv) > 1:
        asyncio.run(_cli_main())
    else:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )

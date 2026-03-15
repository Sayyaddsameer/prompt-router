# Prompt Router

A production-ready, intent-based AI routing service that classifies user messages and delegates them to specialized expert personas. Built with **Python 3.11 + FastAPI + OpenAI**.

## Project Demo

Full architecture walkthrough and live API demonstration.

**Watch the demo:**  
[![Watch the Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1Lt7qkdcVr9EOU9_CSVt1-SrSYI4BgT7s/view?usp=sharing)

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────┐
│  1. classify_intent()           │  Fast, cheap LLM call (gpt-4o-mini)
│     Returns { intent, confidence}│  Returns JSON: intent label + confidence score
└───────────────┬─────────────────┘
                │
     ┌──────────▼──────────┐
     │  Confidence check   │  Below threshold → treat as "unclear"
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  route_and_respond()│  Select expert system prompt from prompts.json
     │  Expert LLM call    │  Second LLM call with specialized persona
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  log_route()        │  Append to route_log.jsonl (JSONL format)
     └─────────────────────┘
```

### Supported Intents

| Intent | Persona | Trigger |
|--------|---------|---------|
| `code` | Code Expert | Programming, debugging, SQL, algorithms |
| `data` | Data Analyst | Statistics, data interpretation, pivot tables |
| `writing` | Writing Coach | Editing feedback, clarity, tone improvement |
| `career` | Career Advisor | Interviews, cover letters, professional advice |
| `unclear` | Clarification | Ambiguous, off-topic, or low-confidence inputs |

### Stretch Goals Implemented
- **Confidence threshold** — inputs below `CONFIDENCE_THRESHOLD` (default 0.7) are routed to `unclear`
- **Manual override** — prefix messages with `@<intent>` (e.g. `@code fix this bug`) to bypass classification
- **FastAPI web UI** — interactive docs at `/docs` with the detected intent and confidence shown in the response
- **CLI mode** — run a single message from the command line: `python main.py "your message"`

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# Clone the repo and enter the directory
git clone https://github.com/YOU/prompt-router.git
cd prompt-router

# Copy and fill in your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Create the log file (needed for the volume mount)
echo "" > route_log.jsonl

# Build and run
docker-compose up --build
```

The API is now available at **http://localhost:8000**.
Interactive docs: **http://localhost:8000/docs**

### Option 2: Local (no Docker)

```bash
cd prompt-router

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r source_code/requirements.txt

# Copy env file and fill in your key
cp .env.example .env

# Run the API server (from the source_code directory)
cd source_code
uvicorn main:app --reload --port 8000
```

### Option 3: CLI

```bash
cd source_code
python main.py "how do i sort a list in python?"
python main.py "@data what is the median of 10, 20, 30?"
```

---

## Environment Variables

All configuration is loaded from environment variables. See `.env.example` for documentation.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `CLASSIFIER_MODEL` | `gpt-4o-mini` | Model for intent classification |
| `RESPONDER_MODEL` | `gpt-4o-mini` | Model for expert response generation |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL (override for Azure/proxies) |
| `CONFIDENCE_THRESHOLD` | `0.7` | Min confidence to trust the intent label |
| `PROMPTS_FILE` | `prompts.json` | Path to expert prompts config file |
| `LOG_FILE` | `route_log.jsonl` | Path to the JSONL route log |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `8000` | HTTP server port |
| `HOST` | `0.0.0.0` | HTTP server host |

---

## API Reference

### `POST /route`

**Classify and respond to a user message.**

```http
POST /route
Content-Type: application/json

{
  "message": "how do i sort a list of objects in python?"
}
```

Response (`200 OK`):
```json
{
  "user_message": "how do i sort a list of objects in python?",
  "classification": {
    "intent": "code",
    "confidence": 0.95,
    "override": false
  },
  "persona": "Code Expert",
  "response": "Use the `key` parameter of `sorted()` or `.sort()`..."
}
```

**Manual override** (bypass classifier):
```json
{ "message": "@code Fix this bug: for i in range(10) print(i)" }
```

### `GET /prompts`

Returns all configured expert personas and their system prompts.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Log Format (`route_log.jsonl`)

Each request appends one JSON object (one line):

```json
{
  "timestamp": "2026-03-14T05:48:01.123456+00:00",
  "intent": "code",
  "confidence": 0.95,
  "override": false,
  "persona": "Code Expert",
  "user_message": "how do i sort a list of objects in python?",
  "final_response": "Use list.sort()..."
}
```

---

## Running Tests

Unit tests use mocking — **no API key required**:

```bash
cd prompt-router
pip install pytest pytest-asyncio httpx
pytest tests/test_classifier.py tests/test_router.py -v
```

Integration tests (15 required messages + edge cases — requires `OPENAI_API_KEY`):

```bash
pytest tests/test_messages.py -v -s
```

---

## Project Structure

```
prompt-router/
├── .env.example                # Environment variable template
├── .gitignore / .gitattributes / .dockerignore
├── Dockerfile                  # Multistage Python 3.11 build
├── docker-compose.yml          # Single service, env from .env
├── pytest.ini                  # Test configuration
├── route_log.jsonl             # JSONL route log (appended to at runtime)
├── source_code/
│   ├── requirements.txt
│   ├── prompts.json            # Expert system prompts (configurable)
│   ├── config.py               # Settings via pydantic-settings (all from env)
│   ├── llm_client.py           # AsyncOpenAI wrapper with JSON parsing
│   ├── prompts_loader.py       # Loads prompts.json, cached singleton
│   ├── classifier.py           # classify_intent() — LLM + override + threshold
│   ├── router.py               # route_and_respond() — expert selection + LLM call
│   ├── route_logger.py         # log_route() — JSONL append
│   └── main.py                 # FastAPI app + CLI entry point
└── tests/
    ├── test_classifier.py      # Unit tests (mocked, no API key)
    ├── test_router.py          # Unit tests (mocked, no API key)
    └── test_messages.py        # Integration: all 15 required messages
```

---

## Design Decisions

**Two-step pipeline**: The classify + respond separation is intentional. The classifier uses `temperature=0.0` and a strict schema to maximize determinism. The responder uses `temperature=0.3` for richer, more natural answers.

**Confidence threshold**: Rather than blindly trusting the classifier, low-confidence labels are downgraded to `unclear`. This prevents confidently-wrong routing (e.g. a vague message being sent to the Code Expert).

**Dual-strategy JSON parsing**: `json_chat()` first tries a direct `json.loads()`, then extracts the first `{...}` block. This handles LLMs that occasionally wrap their JSON in markdown code fences.

**Prompts in JSON**: Expert prompts live in `prompts.json` — independent of business logic, easy to update without code changes, and auditable in version control.

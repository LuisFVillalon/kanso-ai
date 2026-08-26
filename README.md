# TaskMaster AI

FastAPI microservice powering the AI features of TaskMaster. Deployed on [Fly.io](https://fly.io) (region: `ams`).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health / welcome |
| `GET` | `/health` | Liveness check |
| `POST` | `/learning-resources` | Curated video/article/exercise links derived from a note |

---

## Services

### `learning_resources` — `/learning-resources`

Three-step pipeline for note-based resource discovery (`app/services/learning_resources/`):

1. **Search plan** (`planner.py`) — the LLM extracts topic, learner level, and three targeted queries (video, article, exercise).
2. **Web search** (`search.py`) — DuckDuckGo (`ddgs`) runs each query (video query is scoped to `site:youtube.com`), filtered against sponsored/paywall/course-listing keyword lists (`filters.py`) and a dead-link check, then scored for credibility (`credibility.py`).
3. **Ranking** (`ranking.py`) — deterministic, no LLM call: the highest-scoring candidate per resource type is picked, with a plain-language explanation of why.

**Request**
```json
{
  "title": "string",
  "headings": ["string"],
  "highlights": ["string"],
  "lists": ["string"],
  "styled_text": ["string"],
  "tables": ["string"],
  "plain_text": "string"
}
```
All fields are optional (default to empty).

**Response**
```json
{
  "topic": "string",
  "resources": [
    { "type": "video|article|exercise", "title", "url", "why", "platform", "activity_label" }
  ]
}
```

---

## Project structure

```
taskmaster-ai/
├── app/
│   ├── main.py                              # FastAPI app, CORS, router mount
│   ├── api/
│   │   └── routes/
│   │       └── learning_resources_router.py # /learning-resources route handler
│   └── services/
│       └── learning_resources/              # Note-to-resource pipeline
│           ├── __init__.py                  # Public API: StructuredNoteContent, get_learning_resources
│           ├── pipeline.py                  # Orchestrates the 3-step pipeline below
│           ├── note_content.py              # StructuredNoteContent
│           ├── ai_client.py                 # LLM client/provider setup (OpenAI or Gemini)
│           ├── planner.py                   # Step 1: note -> topic + search queries
│           ├── search.py                    # Step 2: web search, filtering, dead-link check
│           ├── filters.py                   # Sponsored/paywall/course-listing keyword filters
│           ├── credibility.py               # Step 3 input: scores + explains each candidate
│           ├── ranking.py                   # Step 3: picks + explains the best candidate per type
│           └── text_matching.py             # Shared keyword-matching helpers
├── Dockerfile
├── fly.toml
└── requirements.txt
```

---

## Environment variables

The LLM provider is picked by `AI_PROVIDER` (`openai` or `gemini`); if unset, it defaults to `openai` when `OPENAI_API_KEY` is present, otherwise `gemini`.

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_PROVIDER` | No | `openai` or `gemini` — selects which client/model below is used |
| `OPENAI_API_KEY` | If provider is `openai` | OpenAI API key |
| `OPENAI_MODEL` | No | Model ID (default `gpt-4.1-mini`) |
| `GEMINI_API_KEY` | If provider is `gemini` | Gemini API key (used via Google's OpenAI-compatible endpoint) |
| `GEMINI_MODEL` | No | Model ID (default `gemini-2.5-flash`) |

---

## Local development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# create a .env file with the variables from "Environment variables" above
uvicorn app.main:app --reload --port 8080
```

## Deployment

Deployed via Fly.io. Push to trigger a deploy:

```bash
fly deploy
```

The app runs on port `8080` (configurable via `$PORT`), auto-starts and auto-stops machines, and enforces HTTPS.

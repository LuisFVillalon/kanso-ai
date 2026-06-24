# TaskMaster AI

FastAPI microservice powering the AI features of TaskMaster. Deployed on [Fly.io](https://fly.io) (region: `ams`).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health / welcome |
| `GET` | `/health` | Liveness check |
| `POST` | `/plan-tasks` | Generate a parent task + scheduled subtasks via LLM |
| `POST` | `/task-debrief` | Today's action plan, horizon warning, and workload analysis |
| `POST` | `/ai-debrief` | Time & Tag Audit across tasks, habits, notes, and calendar settings |
| `POST` | `/daily-briefing` | Morning briefing synthesised from tasks and notes |
| `POST` | `/learning-resources` | Curated video/article/exercise links derived from a note |

All authenticated endpoints require an `Authorization: Bearer <jwt>` header. The `x-timezone` header (IANA string, e.g. `America/Los_Angeles`) is accepted on debrief endpoints.

---

## Services

### `createAITaskPlan` — `/plan-tasks`

Two-phase subtask generator.

**Phase 1 — deterministic scheduling**
- Splits total estimated hours into chunks using session-type presets (`bite_size`: 15–30 min, `deep_work`: 1–2 hr).
- Reserves ~15% of hours as a final review slot.
- Distributes slots across category-specific phases (see table below) before the LLM is called.
- Detects schedule overload against a 6 h/day cognitive-load cap; priority-1 tasks carry a 1.4× urgency weight.
- Collects plan-quality warnings (missing description, low hour estimate, tight deadline, capacity exceeded).

**Phase 2 — LLM content generation**
- Sends pre-computed slots + phase assignments to OpenAI.
- Model writes only titles and descriptions — dates, durations, and phases are never touched by the LLM.
- Falls back to category-aware hardcoded subtasks if the LLM returns invalid JSON.

**Supported categories and phases**

| Category | Phases |
|----------|--------|
| `homework` | Understand → Research → Draft → Complete → Review |
| `test` | Review Notes → Practice → Mock → Analyze → Final Review |
| `project` | Requirements → Implement → Iterate → Test/Debug → Polish/Submit |
| `interview` | Company Research → Behavioral → Technical → Mock Interview → Logistics |
| `skill` | Concepts → Guided Practice → Independent → Apply |

**Response**
```json
{
  "new_task": { /* created task object */ },
  "subtasks": [ { "title", "description", "due_date", "due_time", "estimated_time", ... } ],
  "overload_warning": "string | null",
  "plan_warnings": ["string"]
}
```

---

### `createTaskDebrief` — `/task-debrief`

Fetches tasks, habits, and the user's profile server-side, then calls OpenAI to produce:

- **overdue_tasks** — tasks past their due date
- **tasks_due_today** — tasks due today
- **task_recommendations** — 2–5 priority-ordered execution recommendations (composite urgency score: 55% priority + 45% due-time proximity)
- **remaining_habits** — habits not yet logged today
- **future_horizon_warning** — tasks due in 2–4 days + early-start recommendations
- **workload_analysis** — task count, total hours, break cadence, and feasibility verdict (Light / Moderate / Heavy / Overloaded)

---

### `getLearningResources` — `/learning-resources`

Three-step pipeline for note-based resource discovery:

1. **Search plan** — OpenAI extracts topic, learner level, and three targeted queries (video, article, exercise).
2. **Web search** — DuckDuckGo (`ddgs`) runs each query (video query is scoped to `site:youtube.com`).
3. **Credibility scoring + LLM ranking** — candidates are scored (trust signals, clickbait penalties) and the top 24 are ranked by OpenAI, which selects one resource per type.

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

### `ai-debrief` — `/ai-debrief`

Fetches tasks, habits, notes, and calendar settings in parallel, then calls OpenAI for a Time & Tag Audit covering:
- Smart habit scheduling
- Horizon scanning
- Tag volume audit

---

## Utility: `filterData`

Core scheduling logic shared by the plan-tasks service.

| Function | Purpose |
|----------|---------|
| `compute_task_weight` | Returns cognitive-load hours (priority-1 × 1.4×) |
| `build_active_workload_by_day` | Aggregates existing task load by date; undated tasks spread over 7 days |
| `detect_overload` | Reports utilization %, available slack, and whether a new task can fit |
| `split_into_chunks` | Splits hours into session-preset-sized chunks (capped at 8 subtasks) |
| `schedule_durations` | Maps durations to dates in `[start, buffer_end]` without exceeding daily cap |

**Constants**

| Constant | Value |
|----------|-------|
| `DAILY_CAP_HOURS` | 6.0 h |
| `MAX_SUBTASKS` | 8 |
| `URGENCY_WEIGHT_MULTIPLIER` | 1.4× |
| `UNDATED_TASK_SPREAD_DAYS` | 7 |

---

## Project structure

```
taskmaster-ai/
├── app/
│   ├── main.py                      # FastAPI app, CORS, router mount
│   ├── api/
│   │   └── routes/
│   │       └── tasks_router.py      # All route handlers
│   ├── services/
│   │   ├── createAITaskPlan.py      # Two-phase subtask generator
│   │   ├── createTaskDebrief.py     # Task debrief service
│   │   └── getLearningResources.py  # Note-to-resource pipeline
│   ├── schemas/
│   │   ├── task_schema.py           # Task, TaskCreate Pydantic models
│   │   └── tag_schema.py            # Tag Pydantic model
│   └── util/
│       └── filterData.py            # Workload / scheduling utilities
├── Dockerfile
├── fly.toml
└── requirements.txt
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | Yes | Model ID (e.g. `gpt-4.1-mini`) |
| `TASKMASTER_BACKEND_URL` | Yes | Base URL of the TaskMaster backend API |

---

## Local development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in env vars
uvicorn app.main:app --reload --port 8080
```

## Deployment

Deployed via Fly.io. Push to trigger a deploy:

```bash
fly deploy
```

The app runs on port `8080` (configurable via `$PORT`), auto-starts and auto-stops machines, and enforces HTTPS.

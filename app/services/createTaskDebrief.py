"""
Task Debrief — Today's Action Plan, Future Horizon Warning, and Workload Analysis.

Pipeline:
  tasks_router.py fetches /get-tasks, /get-habits, and /get-profile server-side
  and passes the raw data here.

  This module:
    1. Splits uncompleted tasks into two buckets:
         today_tasks    — due today or overdue (due_date ≤ today)
         upcoming_tasks — due in the next 1–7 days
    2. For today's tasks, computes a composite urgency score per task:
         score = 0.55 × (priority/5) + 0.45 × due_time_urgency
       Tasks are pre-sorted by score descending before being sent to the LLM.
    3. Derives workload metrics:
         total_today_hours, remaining_time_today_hours
    4. Filters habits not yet completed today (pending_today_habits).
    5. Enriches each task with: due_label (overdue / due today / due tomorrow / due in N days),
       parent_task_title, and parent_task_category so the LLM can describe context and urgency.
    6. Packages everything — including user_name and pending_habits_count — into a JSON payload and calls OpenAI.
    7. Returns { today_action_plan, future_horizon_warning, workload_analysis }.

Output schema:
  {
      "overdue_tasks":          list[str],  # one item per overdue task
      "tasks_due_today":        list[str],  # one item per task due today
      "task_recommendations":   list[str],  # priority-ordered execution recommendations
      "remaining_habits":       list[str],  # pending habits the user still needs to log
      "future_horizon_warning": list[str],  # upcoming deadline spike warnings
      "workload_analysis":      list[str],  # cognitive footprint and feasibility verdict
  }
"""

import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client     = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL")

_END_OF_DAY_HOUR = 23  # 11 PM local — usable day boundary for scheduling

_FALLBACK: dict = {
    "overdue_tasks":          ["Could not generate debrief — please try refreshing."],
    "tasks_due_today":        [],
    "task_recommendations":   [],
    "remaining_habits":       [],
    "future_horizon_warning": [],
    "workload_analysis":      [],
}

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_BODY = (
    "You are a precision task execution advisor running a Task Debrief. "
    "Your job is to turn raw task data into an actionable daily playbook — "
    "no fluff, no generalities, no invented details.\n\n"

    "The payload includes a `user_name` field. If it is non-empty, address the user by "
    "their first name at least once across all sections to make the debrief feel personal.\n\n"

    "Return ONLY valid JSON with exactly six keys. "
    "Each key maps to an ARRAY OF STRINGS. "
    "Each string is one complete, self-contained thought that will appear as a single bullet point in the UI. "
    "Related information MUST stay together in the same array item — never split a single idea across two items.\n\n"

    '"overdue_tasks": array of strings — one string per task where due_date < today. '
    "Clearly state it is overdue and reference the exact title. "
    "If parent_task_title is non-null, mention the parent task. "
    "If there are no overdue tasks, output exactly one string: 'No overdue tasks — you are all caught up!'\n\n"

    '"tasks_due_today": array of strings — one string per task where due_date == today. '
    "Reference the exact title and note it must be completed today. "
    "If parent_task_title is non-null, mention the parent task. "
    "If there are no tasks due today, output exactly one string: 'No tasks are due today.'\n\n"

    '"task_recommendations": array of 2–5 strings — recommend which tasks to execute first. '
    "Strictly prioritize by nearest due date/time and priority level (leverage the pre-sorted order and urgency scores). "
    "CRITICAL: each recommended task MUST be its own separate string — never combine multiple tasks into one string. "
    "For each, state: exact title, its due_label (e.g., 'overdue', 'due today', 'due tomorrow'), and if parent_task_title "
    "is non-null, also state the parent task's title and its parent_task_category (e.g., 'as part of [parent title]'). "
    "Do NOT include the numeric priority level in the text.\n\n"

    '"remaining_habits": array of strings — one string per habit in pending_today_habits. '
    "Treat each as a non-negotiable daily commitment. "
    "If pending_today_habits is empty, output exactly one string: 'All habits completed for today — great work!'\n\n"

    '"future_horizon_warning": array of strings in two parts.\n'
    "PART 1 — Tasks due in 2–4 days:\n"
    "  • Use ONLY tasks from `horizon_warning_tasks` in the payload (pre-filtered to due exactly 2–4 days from today).\n"
    "  • For each task output ONE string in this exact format:\n"
    "    '[exact title] — due in [days_until_due] days'\n"
    "    If the task has a non-null parent_task_title, append: ', part of [parent_task_title]'\n"
    "  • Each task is its own string. If horizon_warning_tasks is empty, output one string: 'No tasks are due in the next 2–4 days.'\n\n"
    "PART 2 — Early-start recommendations:\n"
    "  • Output one string as a section header: 'Recommendations — get an early start on:'\n"
    "  • Then add 1–3 recommendation strings drawn from upcoming_tasks (any in the payload), ranked by highest priority first, then earliest due date.\n"
    "  • Each recommendation is its own string. Format: 'Get a head start on [exact title] (due in [N] days) — [one-sentence reason].'\n"
    "  • The reason MUST incorporate the task's actual `priority` value from the payload (1 = lowest, 5 = highest). "
    "For high-priority tasks (4–5), lead with the urgency that priority creates. "
    "For mid-priority tasks (2–3), frame it around deadline proximity and effort. "
    "Never invent a priority level — always read it from the task object.\n\n"

    '"workload_analysis": array of exactly 3 strings. '
    "Express all time and volume metrics in natural, conversational language. Never echo raw field names.\n"
    "Include the following specific bullet points:\n"
    "  1. A bullet stating exactly how many tasks are on the schedule for today and the total estimated hours required (e.g., 'You have X tasks lined up today, totaling roughly Y hours of work.').\n"
    "  2. A break cadence recommendation derived directly from the total estimated hours. Use the following rules:\n"
    "     • Under 2 hours: no formal breaks needed — mention they can work straight through.\n"
    "     • 2–3 hours: suggest 1 short break of about 10 minutes at the halfway point.\n"
    "     • 3–5 hours: suggest 2–3 breaks of 10–15 minutes spaced evenly through the session.\n"
    "     • 5–7 hours: suggest a 15-minute break every 90 minutes (roughly 3–4 breaks total).\n"
    "     • Over 7 hours: suggest a 15-minute break every 60–90 minutes plus one longer 30-minute midday break.\n"
    "     State the number of breaks and their duration explicitly (e.g., 'Plan for 3 breaks of about 15 minutes each, roughly every 90 minutes.').\n"
    "  3. A feasibility verdict with two parts in a single string:\n"
    "     PART A — Load category: classify today using exactly one of these labels based on total hours: "
    "'Light day' (under 3 h), 'Moderate day' (3–5 h), 'Heavy day' (5–7 h), or 'Overloaded day' (over 7 h). "
    "State the label and one sentence interpreting what that means for the user.\n"
    "     PART B — Triage action (only when the verdict is 'Heavy day' or 'Overloaded day'): "
    "identify the single task the user should proactively push to tomorrow and name the triage reason using exactly one of: "
    "(a) 'lowest priority' — it has the lowest priority score among today's tasks; "
    "(b) 'largest time-sink' — it carries the highest estimated hours; "
    "(c) 'least urgent' — it has the latest due date or lowest urgency. "
    "Format: 'Consider pushing [exact task title] to tomorrow — it is the [triage reason] on today's list.' "
    "If the verdict is 'Light day' or 'Moderate day', skip PART B entirely.\n\n"

    "Global rules:\n"
    "• Every value is a JSON array of strings — NO prose paragraphs, no markdown, no numbering.\n"
    "• Each string is one complete thought; never split related information across two items.\n"
    "• Refer to every task by its exact title as given in the payload.\n"
    "• Never invent tasks, durations, or dates not present in the payload.\n"
    "• Warm but direct, executive tone — knowledgeable friend, not a calendar app.\n"
    "• Return raw JSON only — no code fences, no extra keys."
)


def _make_system_prompt(current_time: str, today_str: str) -> str:
    header = (
        f"TEMPORAL CONTEXT — treat these as absolute ground truth:\n"
        f"  Current date and time : {current_time}\n"
        f"  Today's date          : {today_str}\n\n"
    )
    return header + _SYSTEM_BODY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_list(value: object) -> list[str]:
    """Coerce an LLM field into a list of non-empty strings."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _resolve_tz(name: str | None) -> ZoneInfo:
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return ZoneInfo("UTC")


def _current_time_label(dt: datetime) -> str:
    h, m   = dt.hour, dt.minute
    period = "AM" if h < 12 else "PM"
    h12    = h % 12 or 12
    time_str = f"{h12}:{m:02d} {period}" if m else f"{h12} {period}"
    return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year} at {time_str}"


def _parse_due_hour(due_time: str | None) -> float | None:
    """Parse 'HH:MM' or 'HH:MM:SS' → decimal hour (e.g. 14:30 → 14.5)."""
    if not due_time:
        return None
    try:
        parts = str(due_time).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except (ValueError, IndexError, AttributeError):
        return None


def _due_time_urgency(due_hour: float | None, now_hour: float) -> float:
    """
    Score 0.0–1.0.  Higher = more time-critical.
    Tasks without a due_time get 0.30 (flexible / no hard deadline today).
    """
    if due_hour is None:
        return 0.30
    hours_until = due_hour - now_hour
    if hours_until <= 0:
        return 1.00   # already overdue or due right now
    if hours_until <= 2:
        return 0.90
    if hours_until <= 4:
        return 0.75
    if hours_until <= 8:
        return 0.50
    return 0.30


def _urgency_score(task: dict, now_hour: float) -> float:
    """Composite urgency: priority 55 %, due_time 45 %."""
    priority = max(1, min(5, int(task.get("priority") or 1)))
    dt_score = _due_time_urgency(_parse_due_hour(task.get("due_time")), now_hour)
    return round(
        (priority / 5) * 0.55 +
        dt_score       * 0.45,
        3,
    )


# ── Core synthesis ────────────────────────────────────────────────────────────

def _synthesize(tasks: list[dict], habits: list[dict], user_name: str, user_tz: ZoneInfo) -> tuple[dict, str]:
    """Pre-compute the full LLM payload."""
    now      = datetime.now(user_tz)
    today    = now.date().isoformat()
    now_hour = now.hour + now.minute / 60
    horizon  = (now.date() + timedelta(days=7)).isoformat()

    # ── Bucket tasks ─────────────────────────────────────────────────────────
    today_raw: list[dict] = []
    upcoming_raw: list[dict] = []

    for t in tasks:
        if t.get("completed"):
            continue
        due = str(t.get("due_date") or "")[:10]
        if not due:
            continue
        if due <= today:
            today_raw.append(t)
        elif due <= horizon:
            upcoming_raw.append(t)

    # ── Build id → task lookup for parent title resolution ───────────────────
    tasks_by_id: dict[int, dict] = {}
    for t in tasks:
        tid = t.get("id")
        if tid is not None:
            tasks_by_id[int(tid)] = t

    # ── Shape and sort today's tasks by urgency ───────────────────────────────
    def _shape(t: dict, score: float | None = None) -> dict:
        parent_id = t.get("parent_task_id")
        parent_title: str | None = None
        parent_category: str | None = None
        if parent_id is not None:
            parent = tasks_by_id.get(int(parent_id))
            if parent:
                parent_title    = parent.get("title")    or None
                parent_category = parent.get("category") or None

        due_str      = str(t.get("due_date", ""))[:10]
        tomorrow_str = (now.date() + timedelta(days=1)).isoformat()
        if not due_str:
            due_label: str | None = None
        elif due_str < today:
            due_label = "overdue"
        elif due_str == today:
            due_label = "due today"
        elif due_str == tomorrow_str:
            due_label = "due tomorrow"
        else:
            try:
                days = (datetime.strptime(due_str, "%Y-%m-%d").date() - now.date()).days
                due_label = f"due in {days} days"
            except ValueError:
                due_label = None

        shaped: dict = {
            "id":                   t.get("id"),
            "title":                t.get("title", ""),
            "due_date":             due_str,
            "due_label":            due_label,
            "due_time":             t.get("due_time"),
            "priority":             max(1, min(5, int(t.get("priority") or 1))),
            "estimated_time":       round(float(t.get("estimated_time") or 0), 2),
            "parent_task_title":    parent_title,
            "parent_task_category": parent_category,
        }
        if score is not None:
            shaped["urgency_score"] = score
        return shaped

    sorted_today = [
        _shape(t, _urgency_score(t, now_hour))
        for t in sorted(today_raw, key=lambda x: _urgency_score(x, now_hour), reverse=True)
    ]

    # Upcoming: sorted by due_date asc, then priority desc
    upcoming = sorted(
        [_shape(t) for t in upcoming_raw],
        key=lambda x: (x["due_date"], -x["priority"]),
    )[:10]

    # Horizon warning: tasks due in Today+2 to Today+4 (inclusive)
    horizon_start = (now.date() + timedelta(days=2)).isoformat()
    horizon_end   = (now.date() + timedelta(days=4)).isoformat()

    def _days_until(due_str: str) -> int | None:
        try:
            return (datetime.strptime(due_str, "%Y-%m-%d").date() - now.date()).days
        except ValueError:
            return None

    horizon_warning_tasks = []
    for t in sorted(
        [t for t in upcoming_raw if horizon_start <= str(t.get("due_date", ""))[:10] <= horizon_end],
        key=lambda x: (str(x.get("due_date", ""))[:10], -(max(1, min(5, int(x.get("priority") or 1))))),
    ):
        s = _shape(t)
        s["days_until_due"] = _days_until(s["due_date"])
        horizon_warning_tasks.append(s)

    # ── Pending habits (not yet completed today) ──────────────────────────────
    pending_today_habits = [
        {"id": h.get("id"), "title": h.get("title", "")}
        for h in habits
        if not h.get("logged_today", False)
    ]

    # ── Workload metrics ──────────────────────────────────────────────────────
    total_today_hours = round(sum(t["estimated_time"] for t in sorted_today), 1)
    remaining_day_hrs = max(0.0, _END_OF_DAY_HOUR - now_hour)
    remaining_time    = max(0.0, remaining_day_hrs - total_today_hours)

    current_time = _current_time_label(now)

    payload = {
        "user_name":                  user_name,
        "current_time":               current_time,
        "today":                      today,
        "now_hour":                   round(now_hour, 2),
        "total_today_hours":          total_today_hours,
        "remaining_time_today_hours": round(remaining_time, 1),
        "pending_habits_count":       len(pending_today_habits),
        "sorted_today_tasks":         sorted_today,
        "upcoming_tasks":             upcoming,
        "horizon_warning_tasks":      horizon_warning_tasks,
        "pending_today_habits":       pending_today_habits,
    }

    return payload, _make_system_prompt(current_time, today)


# ── Public entry point ────────────────────────────────────────────────────────

async def create_task_debrief(
    tasks:         list[dict],
    habits:        list[dict] = [],
    user_name:     str = "",
    timezone_name: str = "UTC",
) -> dict:
    """
    Synthesise the user's task state into a Task Debrief.

    Args:
        tasks:         Raw task dicts from /get-tasks.
        habits:        Raw habit dicts from /get-habits (includes logged_today flag).
        user_name:     Display name from the user's profile (for personalisation).
        timezone_name: IANA timezone string (e.g. "America/Los_Angeles").

    Returns:
        {
            "overdue_tasks":          list[str],
            "tasks_due_today":        list[str],
            "task_recommendations":   list[str],
            "remaining_habits":       list[str],
            "future_horizon_warning": list[str],
            "workload_analysis":      list[str],
        }
    """
    user_tz = _resolve_tz(timezone_name)
    payload, system_prompt = _synthesize(tasks, habits, user_name, user_tz)

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate the Task Debrief from this structured data:\n\n"
                    + json.dumps(payload, separators=(",", ":"), default=str)
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=1600,
    )

    raw = (response.choices[0].message.content or "").strip()

    try:
        result = json.loads(raw)
        return {
            "overdue_tasks":          _normalize_list(result.get("overdue_tasks")),
            "tasks_due_today":        _normalize_list(result.get("tasks_due_today")),
            "task_recommendations":   _normalize_list(result.get("task_recommendations")),
            "remaining_habits":       _normalize_list(result.get("remaining_habits")),
            "future_horizon_warning": _normalize_list(result.get("future_horizon_warning")),
            "workload_analysis":      _normalize_list(result.get("workload_analysis")),
        }
    except (json.JSONDecodeError, TypeError):
        return _FALLBACK.copy()

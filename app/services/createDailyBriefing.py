"""
Daily Briefing service — Chief of Staff edition.

Data flow:
  tasks_router.py fetches (in parallel):
    • /get-tasks              → all user tasks
    • /work-blocks            → confirmed + suggested blocks
    • /google-calendar/events → today's calendar events (empty if not connected)
    • /get-notes              → notes for contextual cross-referencing

  This module synthesises the raw data into three pre-computed sets:
    • scheduled_tasks   — tasks due in the next 48 h that have a confirmed/suggested block
    • at_risk_tasks     — tasks due in the next 48 h with NO block at all
    • timeline          — calendar events + confirmed blocks for TODAY, sorted chronologically
                          with gap_minutes_to_next for tight-transition detection

  All timestamps are converted to the user's local timezone BEFORE being labelled,
  so the LLM receives pre-localised strings and must never convert them.

  The system prompt is built dynamically per request so the LLM's temporal anchor
  (current date + time) lives at the highest-priority level — in the system turn —
  not buried in the data payload where it competes with raw ISO strings.

  Output: dict { "pulse": str, "timeline": str, "action_items": list[str] }
"""

import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client     = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL")

# ── Fallback ──────────────────────────────────────────────────────────────────
_FALLBACK: dict = {
    "pulse":        "Could not generate briefing — please try refreshing.",
    "timeline":     "",
    "action_items": [],
}

# ── Static body of the system prompt ─────────────────────────────────────────
# The temporal header (current date/time, today, tomorrow) is prepended
# dynamically in _make_system_prompt() so the LLM treats it as ground truth.

_SYSTEM_BODY = (
    "You are an elite Chief of Staff delivering a morning intelligence briefing. "
    "Your purpose is to make the user feel prepared and in control.\n\n"

    # ── Overdue rule (explicit, with examples) ────────────────────────────────
    "OVERDUE RULE: A task is overdue ONLY if its due_date is strictly less than "
    "the 'today' date provided above. Use the provided 'today' value as your "
    "sole reference — never use your own estimate of the current date. "
    "Example: if today is 2026-04-11, a task with due_date 2026-04-12 is NOT "
    "overdue — it is due tomorrow. Only tasks in the pre-computed 'overdue_tasks' "
    "array are overdue; do not reclassify any other task as overdue.\n\n"

    # ── Scheduling rule ───────────────────────────────────────────────────────
    "SCHEDULING RULE: Each task in 'scheduled_tasks' and 'at_risk_tasks' carries "
    "an 'is_scheduled' boolean and a 'scheduled_time' string. "
    "is_scheduled=true means the task ALREADY has a confirmed or pending work block. "
    "The 'scheduled_time' field shows exactly when that block is (e.g. 'Sun Apr 12, 8:00 AM'). "
    "NEVER create an action item for a task where is_scheduled is true, even if the "
    "block falls tomorrow or later in the week. "
    "Only tasks from 'at_risk_tasks' where is_scheduled is false belong in action_items.\n\n"

    # ── Completed tasks ───────────────────────────────────────────────────────
    "COMPLETED RULE: Tasks where 'completed' is true have already been finished. "
    "They must NEVER appear in overdue_tasks, at_risk_tasks, or action_items. "
    "The pre-computed arrays already exclude completed tasks, but if you see one "
    "in the raw data with completed=true, ignore it entirely.\n\n"

    # ── Output schema ─────────────────────────────────────────────────────────
    "Return ONLY valid JSON with exactly these three keys:\n\n"

    '"pulse": A 2–3 sentence paragraph. '
    "Lead with 'overdue_tasks' if any — name them urgently by title. "
    "Then reference 'confirmed_blocks', 'at_risk_count', and 'complexity_score' "
    "to set the day's emotional tone.\n\n"

    '"timeline": A 3–5 sentence paragraph. '
    "Narrate the day chronologically using every item in the 'timeline' array. "
    "Use ONLY the 'start_label' and 'end_label' values — they are already in the "
    "user's local timezone; never convert or adjust them. "
    "Connect adjacent items: \"Because you have [Event] at [start_label], your "
    "[Work Block] right after keeps [Deadline] on track.\" "
    "If gap_minutes_to_next ≤ 15 or a work block follows a ≥60-min event, suggest "
    "a brief break by name. "
    "If timeline is empty, note the open schedule and reference the at-risk tasks.\n\n"

    '"action_items": Array of short imperative strings (≤15 words each). '
    "Include ONLY tasks from 'at_risk_tasks' where is_scheduled is false. "
    "Pattern: \"Schedule a block for \'[Task Title]\' — due [due_label].\" "
    "If every at_risk task has is_scheduled=true, or the array is empty, return: "
    "[\"All critical tasks have blocks assigned — you\'re on track!\"]\n\n"

    "Global rules:\n"
    "• pulse and timeline are prose paragraphs — no bullets, no markdown.\n"
    "• Only reference explicitly provided data — never invent events or deadlines.\n"
    "• Warm, direct, executive tone — knowledgeable friend, not a calendar app.\n"
    "• Return raw JSON only — no code fences, no extra keys."
)


def _make_system_prompt(current_time: str, today_str: str, tomorrow_str: str) -> str:
    """
    Prepend a hard temporal anchor to the system prompt.

    Putting the current date/time in the SYSTEM turn (not just the data payload)
    makes it the highest-priority context for the LLM.  This prevents the model
    from falling back to its training-cutoff date when reasoning about overdue vs
    upcoming tasks.
    """
    header = (
        f"TEMPORAL CONTEXT — treat these as absolute ground truth:\n"
        f"  Current date and time : {current_time}\n"
        f"  Today's date          : {today_str}\n"
        f"  Tomorrow's date       : {tomorrow_str}\n\n"
    )
    return header + _SYSTEM_BODY


_USER_PROMPT = "Synthesise the briefing from this structured data:\n\n"

# ── Verb-hint rules ───────────────────────────────────────────────────────────
_VERB_RULES: list[tuple[list[str], str]] = [
    (["midterm", "final exam", "exam", "quiz", "test prep"], "Study for"),
    (["meeting", "sync", "standup", "stand-up", "call", "1:1", "interview"], "Attend"),
    (["report", "draft", "essay", "paper", "proposal", "write"], "Finalize"),
    (["gym", "workout", "work out", "run", "exercise", "training", "lift"], "Complete session for"),
    (["review", "read", "audit"], "Review"),
    (["deploy", "release", "ship", "launch"], "Deploy"),
    (["debug", "fix", "patch", "resolve", "bug"], "Fix"),
    (["presentation", "present", "slides", "demo"], "Prepare presentation for"),
    (["submit", "turn in", "upload", "hand in"], "Submit"),
    (["project", "build", "implement", "develop", "sprint"], "Work on"),
]


def _verb_hint(title: str) -> str:
    lower = title.lower()
    for keywords, verb in _VERB_RULES:
        if any(kw in lower for kw in keywords):
            return verb
    return "Complete"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


# ── Timezone helpers ──────────────────────────────────────────────────────────

def _resolve_tz(name: str | None) -> ZoneInfo:
    """Resolve an IANA timezone name; silently fall back to UTC on any error."""
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return ZoneInfo("UTC")


def _parse_aware(iso_str: str) -> datetime:
    """Parse ISO 8601 → timezone-aware datetime; treats naive strings as UTC."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_local(iso_str: str, tz: ZoneInfo) -> datetime:
    """Parse and convert to the user's local timezone."""
    return _parse_aware(iso_str).astimezone(tz)


def _local_date_key(iso_str: str, tz: ZoneInfo) -> str:
    """
    Return 'YYYY-MM-DD' in the user's local timezone.

    Critical: a work block stored as '2026-04-11T00:00:00+00:00' (midnight UTC)
    is '2026-04-10' for a Pacific Time user (UTC-7).  Using UTC [:10] slicing
    would assign it to the wrong day.
    """
    try:
        return _to_local(iso_str, tz).date().isoformat()
    except (ValueError, TypeError):
        return iso_str[:10]


def _format_12h(iso_str: str, tz: ZoneInfo) -> str:
    """
    Format an ISO 8601 timestamp as '9:00 AM' in the user's local timezone.

    Example: '2026-04-11T16:00:00+00:00' with tz=America/Los_Angeles → '9:00 AM'
    """
    try:
        dt = _to_local(iso_str, tz)
        h, m = dt.hour, dt.minute
        period = "AM" if h < 12 else "PM"
        h12    = h % 12 or 12
        return f"{h12}:{m:02d} {period}" if m else f"{h12} {period}"
    except (ValueError, TypeError):
        return iso_str


def _scheduled_time_label(wb: dict, tz: ZoneInfo) -> str | None:
    """
    Return a human-readable label like 'Sun Apr 12, 8:00 AM' for a work block.

    This is embedded directly on the task summary so the LLM sees the explicit
    connection between 'Create Frontend' and its scheduled slot without having to
    cross-reference a separate work_blocks array.
    """
    try:
        dt      = _to_local(wb["start_time"], tz)
        day_abbr   = dt.strftime("%a")   # "Sun"
        month_abbr = dt.strftime("%b")   # "Apr"
        h, m    = dt.hour, dt.minute
        period  = "AM" if h < 12 else "PM"
        h12     = h % 12 or 12
        time_str = f"{h12}:{m:02d} {period}" if m else f"{h12} {period}"
        return f"{day_abbr} {month_abbr} {dt.day}, {time_str}"  # "Sun Apr 12, 8:00 AM"
    except (ValueError, KeyError, TypeError):
        return None


def _current_time_label(dt: datetime) -> str:
    """Format a local datetime as 'Saturday, April 11, 2026 at 10:30 AM'."""
    h, m    = dt.hour, dt.minute
    period  = "AM" if h < 12 else "PM"
    h12     = h % 12 or 12
    time_str = f"{h12}:{m:02d} {period}" if m else f"{h12} {period}"
    return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year} at {time_str}"


def _due_label(task: dict, today_str: str, tomorrow_str: str) -> str:
    """
    Return a human-readable due label ('9:00 AM today', 'tomorrow', '2026-04-15').

    due_time is stored as "HH:MM" in the user's local time — no conversion needed.
    """
    due_date = str(task.get("due_date") or "")[:10]
    due_time = task.get("due_time") or ""

    if due_date == today_str:
        if due_time:
            try:
                h, m   = int(due_time[:2]), int(due_time[3:5])
                period = "AM" if h < 12 else "PM"
                h12    = h % 12 or 12
                label  = f"{h12}:{m:02d} {period}" if m else f"{h12} {period}"
                return f"{label} today"
            except (ValueError, IndexError):
                pass
        return "tonight"

    if due_date == tomorrow_str:
        return "tomorrow"

    return due_date


def _build_task_summary(
    t: dict,
    today_str: str,
    tomorrow_str: str,
    is_scheduled: bool,
    scheduled_time: str | None = None,
) -> dict:
    return {
        "id":             t.get("id"),
        "title":          t.get("title", ""),
        "verb_hint":      _verb_hint(t.get("title", "")),
        "completed":      t.get("completed", False),   # guard rail: LLM must ignore completed tasks
        "priority":       t.get("priority"),
        "due_date":       str(t.get("due_date", ""))[:10] if t.get("due_date") else None,
        "due_time":       t.get("due_time") or None,
        "due_label":      _due_label(t, today_str, tomorrow_str),
        "tags":           [tg.get("name", "") for tg in (t.get("tags") or [])],
        "category":       t.get("category"),
        "complexity":     t.get("complexity") or 0,
        # ── Explicit scheduling state ──────────────────────────────────────────
        # is_scheduled=True means a confirmed OR suggested work block already
        # exists for this task (within the 48-hour window).  The LLM must treat
        # this as "covered" and must NOT add an action item for it.
        "is_scheduled":   is_scheduled,
        # scheduled_time is the pre-formatted local time of the earliest block,
        # e.g. "Sun Apr 12, 8:00 AM".  Null when is_scheduled=False.
        "scheduled_time": scheduled_time,
    }


# ── Core synthesis ────────────────────────────────────────────────────────────

def _synthesize(
    tasks:           list[dict],
    work_blocks:     list[dict],
    calendar_events: list[dict],
    notes:           list[dict],
    user_tz:         ZoneInfo,
) -> tuple[dict, str]:
    """
    Pre-compute logical task sets and the day timeline.

    Returns (context_dict, system_prompt_str).

    The system prompt is returned here (not at module level) because it must
    embed the current local date/time as a hard temporal anchor.

    CRITICAL FIX — 48-hour block window:
      due_soon covers tasks due today AND tomorrow.  The block-scan window must
      match: a task due tomorrow with a block at 8 AM tomorrow has is_scheduled=True
      even though that block doesn't fall on today.  Narrowing the scan to
      today_str only caused confirmed blocks to be missed, producing redundant
      action items like "Schedule a block for Create Frontend" even when one existed.
    """
    # ── Local date/time anchor ────────────────────────────────────────────────
    now_local      = datetime.now(user_tz)
    today_local    = now_local.date()
    today_str      = today_local.isoformat()
    tomorrow_str   = (today_local + timedelta(days=1)).isoformat()
    current_time   = _current_time_label(now_local)

    # ── Active tasks (completed tasks are excluded from all analysis) ─────────
    active = [t for t in tasks if not t.get("completed")]

    # ── Task sets ─────────────────────────────────────────────────────────────

    overdue_tasks = [
        _build_task_summary(t, today_str, tomorrow_str, False)
        for t in active
        if (d := str(t.get("due_date") or "")[:10]) and d < today_str
    ]

    # Tasks due in the next 48 h (today + tomorrow)
    due_soon = [
        t for t in active
        if str(t.get("due_date") or "")[:10] in (today_str, tomorrow_str)
    ]

    # ── Work-block sets ───────────────────────────────────────────────────────
    #
    # FIX: scan the SAME 48-hour window used by due_soon.
    # A task due tomorrow (Apr 12) whose block starts Apr 12 at 08:00 must have
    # is_scheduled=True.  Filtering blocks to today_str only misses it entirely.
    #
    # scheduled_relevant = all non-dismissed blocks that land on today OR tomorrow.
    # confirmed_today    = confirmed blocks that land on today (timeline only).

    relevant_dates = {today_str, tomorrow_str}

    scheduled_relevant: list[dict] = [
        wb for wb in work_blocks
        if wb.get("status") in ("confirmed", "suggested")
        and _local_date_key(wb.get("start_time", ""), user_tz) in relevant_dates
    ]

    # Build task_id → earliest block mapping so we can pass scheduled_time to summaries
    scheduled_block_by_task: dict[int, dict] = {}
    for wb in scheduled_relevant:
        tid = wb.get("task_id")
        if tid is None:
            continue
        if tid not in scheduled_block_by_task:
            scheduled_block_by_task[tid] = wb
        else:
            # Keep the earliest block for this task
            try:
                if _parse_aware(wb["start_time"]) < _parse_aware(scheduled_block_by_task[tid]["start_time"]):
                    scheduled_block_by_task[tid] = wb
            except (ValueError, KeyError):
                pass

    scheduled_task_ids: set[int] = set(scheduled_block_by_task.keys())

    # Timeline: only confirmed blocks that fall on TODAY (tomorrow's blocks are future)
    confirmed_today: list[dict] = [
        wb for wb in scheduled_relevant
        if wb.get("status") == "confirmed"
        and _local_date_key(wb.get("start_time", ""), user_tz) == today_str
    ]

    # ── Task-ID → full task lookup (for work block timeline labels) ───────────
    task_by_id: dict[int, dict] = {
        t["id"]: t for t in tasks if t.get("id") is not None
    }

    scheduled_tasks = [
        _build_task_summary(
            t, today_str, tomorrow_str,
            is_scheduled=True,
            scheduled_time=_scheduled_time_label(scheduled_block_by_task[t["id"]], user_tz),
        )
        for t in due_soon
        if t.get("id") in scheduled_task_ids
    ]
    at_risk_tasks = [
        _build_task_summary(t, today_str, tomorrow_str, is_scheduled=False)
        for t in due_soon
        if t.get("id") not in scheduled_task_ids
    ]

    # ── Timeline ──────────────────────────────────────────────────────────────
    # Merge today's non-all-day calendar events with today's confirmed work blocks.
    # All time labels are pre-localised — LLM must use them verbatim.

    raw_items: list[tuple[datetime, dict]] = []

    for ev in calendar_events:
        if ev.get("is_all_day"):
            continue
        start_str = ev.get("start", "")
        if _local_date_key(start_str, user_tz) != today_str:
            continue
        try:
            s_dt = _parse_aware(start_str)
            e_dt = _parse_aware(ev["end"]) if ev.get("end") else s_dt
            raw_items.append((s_dt, {
                "type":             "calendar_event",
                "title":            ev.get("title", ""),
                "start_label":      _format_12h(start_str, user_tz),
                "end_label":        _format_12h(ev["end"], user_tz) if ev.get("end") else "",
                "duration_minutes": int((e_dt - s_dt).total_seconds() / 60),
                "_end_dt":          e_dt,
            }))
        except (ValueError, KeyError):
            pass

    for wb in confirmed_today:
        task_info  = task_by_id.get(wb.get("task_id"))
        task_title = task_info["title"] if task_info else "Work Block"
        task_tags  = [tg.get("name", "") for tg in (task_info.get("tags") or [])] if task_info else []
        try:
            s_dt = _parse_aware(wb["start_time"])
            e_dt = _parse_aware(wb["end_time"])
            raw_items.append((s_dt, {
                "type":             "work_block",
                "title":            task_title,
                "task_tags":        task_tags,
                "start_label":      _format_12h(wb["start_time"], user_tz),
                "end_label":        _format_12h(wb["end_time"], user_tz),
                "duration_minutes": int((e_dt - s_dt).total_seconds() / 60),
                "ai_reasoning":     wb.get("ai_reasoning", ""),
                "_end_dt":          e_dt,
            }))
        except (ValueError, KeyError):
            pass

    raw_items.sort(key=lambda x: x[0])

    timeline: list[dict] = []
    for i, (_, item) in enumerate(raw_items):
        if i + 1 < len(raw_items):
            gap = int((raw_items[i + 1][0] - item["_end_dt"]).total_seconds() / 60)
            item["gap_minutes_to_next"] = max(gap, 0)
        else:
            item["gap_minutes_to_next"] = None
        item.pop("_end_dt", None)
        timeline.append(item)

    # ── Relevant notes ────────────────────────────────────────────────────────

    priority_tags: set[str] = {
        tg.get("name", "")
        for t in active
        if str(t.get("due_date", ""))[:10] <= today_str
        for tg in (t.get("tags") or [])
    }
    tag_matched = [
        n for n in notes
        if priority_tags & {tg.get("name", "") for tg in (n.get("tags") or [])}
    ]
    note_pool = tag_matched if tag_matched else notes
    relevant_notes = [
        {
            "title":   n.get("title", ""),
            "snippet": _strip_html(n.get("content", ""))[:200],
            "tags":    [tg.get("name", "") for tg in (n.get("tags") or [])],
        }
        for n in note_pool
    ][:6]

    # ── Complexity & tag density ──────────────────────────────────────────────

    complexity_score = sum(
        t.get("complexity") or 0 for t in active
        if str(t.get("due_date", ""))[:10] == today_str
    )

    all_tags   = [tg for t in active for tg in [x.get("name", "") for x in (t.get("tags") or [])]]
    tag_counts = Counter(all_tags)
    top        = tag_counts.most_common(1)
    tag_density = (
        {"dominant_tag": top[0][0], "count": top[0][1], "total_active": len(active)}
        if top else None
    )

    context = {
        # ── Temporal anchors (redundant with system prompt — belt-and-suspenders) ──
        "current_time":     current_time,           # "Saturday, April 11, 2026 at 10:30 AM"
        "today":            today_str,              # "2026-04-11"
        "tomorrow":         tomorrow_str,           # "2026-04-12"
        "timezone":         str(user_tz),           # "America/Los_Angeles"
        # ── Scalar signals ────────────────────────────────────────────────────
        "confirmed_blocks": len(confirmed_today),
        "at_risk_count":    len(at_risk_tasks),
        "complexity_score": complexity_score,
        "has_calendar":     bool(calendar_events),
        # ── Task sets ─────────────────────────────────────────────────────────
        # Each entry carries is_scheduled + scheduled_time so the LLM can verify
        # the task-to-block connection without cross-referencing another array.
        "overdue_tasks":    overdue_tasks,
        "scheduled_tasks":  scheduled_tasks,
        "at_risk_tasks":    at_risk_tasks,
        # ── Narrative spine ───────────────────────────────────────────────────
        "timeline":         timeline,
        # ── Supporting context ────────────────────────────────────────────────
        "relevant_notes":   relevant_notes,
        "tag_density":      tag_density,
    }

    system_prompt = _make_system_prompt(current_time, today_str, tomorrow_str)
    return context, system_prompt


# ── Public entry point ────────────────────────────────────────────────────────

async def create_daily_briefing(
    tasks:           list[dict],
    work_blocks:     list[dict],
    calendar_events: list[dict],
    notes:           list[dict],
    timezone_name:   str = "UTC",
) -> dict:
    """
    Synthesise tasks, work blocks, calendar events, and notes into a structured
    briefing: { "pulse": str, "timeline": str, "action_items": list[str] }.

    timezone_name: IANA timezone string forwarded from the browser via the
                   X-Timezone header (e.g. "America/Los_Angeles").
    """
    user_tz = _resolve_tz(timezone_name)
    context, system_prompt = _synthesize(
        tasks, work_blocks, calendar_events, notes, user_tz
    )

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _USER_PROMPT + json.dumps(context, separators=(",", ":"))},
        ],
        response_format={"type": "json_object"},
        temperature=0.35,
        max_tokens=700,
    )

    raw = (response.choices[0].message.content or "").strip()

    try:
        result = json.loads(raw)
        return {
            "pulse":        str(result.get("pulse", "")),
            "timeline":     str(result.get("timeline", "")),
            "action_items": list(result.get("action_items", [])),
        }
    except (json.JSONDecodeError, TypeError):
        return _FALLBACK.copy()

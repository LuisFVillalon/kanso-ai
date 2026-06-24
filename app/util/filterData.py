from collections import defaultdict
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from math import ceil

# ── Workload constants ────────────────────────────────────────────────────────

# Maximum cognitive-load hours a user should accumulate on any single day.
DAILY_CAP_HOURS: float = 6.0

# Maximum number of subtasks ever generated for one parent task.
# Prevents an overwhelming plan for very large tasks.
MAX_SUBTASKS: int = 8

# Urgent tasks feel heavier than their estimated_time alone suggests.
URGENCY_WEIGHT_MULTIPLIER: float = 1.4

# How many days to spread the load of tasks that have no due_date.
UNDATED_TASK_SPREAD_DAYS: int = 7


# ── Task weight scoring ───────────────────────────────────────────────────────

def compute_task_weight(task) -> float:
    """
    Return a float representing the *cognitive load* of a task.

    Weight = estimated_time × urgency_multiplier

    Priority-1 tasks carry a 1.4× multiplier to reflect their urgency.
    """
    hours = float(task.estimated_time or 0.0)
    if hours <= 0:
        return 0.0
    # Priority 1 is the most important; treat it like the old "urgent" flag
    priority = getattr(task, "priority", None)
    urgency_mult = URGENCY_WEIGHT_MULTIPLIER if priority == 1 else 1.0
    return round(hours * urgency_mult, 4)


# ── Workload aggregation ──────────────────────────────────────────────────────

def build_active_workload_by_day(active_tasks: list) -> dict[str, float]:
    """
    Return a dict mapping ISO date strings to total cognitive-load hours.

    - Tasks WITH a due_date: full weight lands on that day.
    - Tasks WITHOUT a due_date: weight is spread evenly across the next
      UNDATED_TASK_SPREAD_DAYS days as a diffuse background load.
      Previously these were silently dropped, causing the scheduler to
      underestimate the user's real workload.
    """
    by_day: dict[str, float] = defaultdict(float)
    today = date.today()

    for task in active_tasks:
        if task.completed:
            continue
        if not task.estimated_time:
            continue

        weight = compute_task_weight(task)

        if task.due_date:
            key = (
                task.due_date.isoformat()
                if hasattr(task.due_date, "isoformat")
                else str(task.due_date)[:10]
            )
            by_day[key] += weight
        else:
            # Spread undated load evenly over the next N days
            daily_share = round(weight / UNDATED_TASK_SPREAD_DAYS, 4)
            for offset in range(UNDATED_TASK_SPREAD_DAYS):
                spread_key = (today + timedelta(days=offset)).isoformat()
                by_day[spread_key] += daily_share

    return dict(by_day)


# ── Overload detection ────────────────────────────────────────────────────────

def detect_overload(
    workload_by_day: dict[str, float],
    start_date: date,
    end_date: date,
    new_task_hours: float,
    daily_cap: float = DAILY_CAP_HOURS,
) -> dict:
    """
    Scan the scheduling window and return an overload summary.

    Return shape:
        {
          "overloaded_days": int,     # days already at or above cap
          "available_hours": float,   # total slack hours in the window
          "can_fit": bool,            # whether new_task_hours can be placed
          "utilization_pct": float,   # % of window capacity already consumed
        }
    """
    window_days = max(1, (end_date - start_date).days + 1)
    total_capacity = window_days * daily_cap
    total_used = sum(
        v for k, v in workload_by_day.items()
        if start_date.isoformat() <= k <= end_date.isoformat()
    )
    overloaded_days = sum(
        1 for k, v in workload_by_day.items()
        if start_date.isoformat() <= k <= end_date.isoformat() and v >= daily_cap
    )
    available_hours = max(0.0, total_capacity - total_used)
    return {
        "overloaded_days": overloaded_days,
        "available_hours": round(available_hours, 2),
        "can_fit": available_hours >= new_task_hours,
        "utilization_pct": round((total_used / total_capacity) * 100, 1) if total_capacity > 0 else 0.0,
    }


# ── Hour splitting ────────────────────────────────────────────────────────────

def split_into_chunks(
    total_hours: float,
    target_size: float = 2.0,
    min_size: float = 1.0,
    max_size: float = 3.0,
    increment: float = 0.5,
) -> List[float]:
    """
    Return a list of durations (hours) that sum exactly to total_hours,
    each between min_size and max_size in `increment` steps.

    Count is capped at MAX_SUBTASKS. If total_hours requires more chunks,
    remaining time is absorbed into the last chunk (it may slightly exceed
    max_size) rather than generating an overwhelming number of subtasks.
    """
    def round_inc(x):
        return round(round(x / increment) * increment, 10)

    if total_hours <= 0:
        return []

    n = max(1, int(ceil(total_hours / target_size)))
    n = min(n, MAX_SUBTASKS)

    while n * max_size < total_hours and n < MAX_SUBTASKS:
        n += 1

    base = min(total_hours / n, max_size)
    chunks = [round_inc(base) for _ in range(n)]
    diff = round_inc(total_hours - sum(chunks))

    step = increment
    attempts = 0
    while abs(diff) >= 1e-9 and attempts < 10000:
        attempts += 1
        if diff > 0:
            chunks.sort()
            for i in range(len(chunks)):
                if chunks[i] + step <= max_size + 1e-9:
                    chunks[i] = round_inc(chunks[i] + step)
                    diff = round_inc(diff - step)
                    break
            else:
                if n < MAX_SUBTASKS:
                    n += 1
                    chunks.append(round_inc(min(max_size, step)))
                    diff = round_inc(total_hours - sum(chunks))
                else:
                    # At MAX_SUBTASKS cap — absorb remainder into last chunk
                    chunks[-1] = round_inc(chunks[-1] + diff)
                    diff = 0.0
        else:
            chunks.sort(reverse=True)
            for i in range(len(chunks)):
                if chunks[i] - step >= min_size - 1e-9:
                    chunks[i] = round_inc(chunks[i] - step)
                    diff = round_inc(diff + step)
                    break
            else:
                if len(chunks) > 1:
                    chunks.sort()
                    a = chunks.pop(0)
                    b = chunks.pop(0)
                    merged = round_inc(a + b)
                    if merged <= max_size + 1e-9:
                        chunks.append(merged)
                    else:
                        chunks.extend([a, b])
                        break
                    diff = round_inc(total_hours - sum(chunks))
                else:
                    break

    # Final float-error correction
    total = round(sum(chunks), 10)
    if abs(total - round(total_hours, 10)) > 1e-6:
        chunks[-1] = round(chunks[-1] + (total_hours - total), 10)

    return chunks


# ── Scheduling ────────────────────────────────────────────────────────────────

def schedule_durations(
    durations: List[float],
    start_date: date,
    buffer_end: date,
    active_workload_by_day: Dict[str, float],
    daily_cap: float = DAILY_CAP_HOURS,
    due_time_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Map each duration to a date in [start_date, buffer_end] without
    exceeding daily_cap (accounting for active_workload_by_day).

    Returns scheduling dicts only — no titles or descriptions:
        [{"date": date, "duration": float, "due_time": str, "overloaded": bool}, ...]

    When a duration cannot fit anywhere in the window the entry is placed
    on buffer_end with "overloaded": True so the service layer can surface
    a warning to the user instead of silently exceeding the cap.
    """
    if not durations:
        return []

    n = len(durations)
    total_days = max(0, (buffer_end - start_date).days)

    if n == 1:
        target_day_offsets = [0]
    else:
        step = total_days / max(1, n - 1)
        target_day_offsets = [int(round(i * step)) for i in range(n)]

    workloads = {k: float(v) for k, v in active_workload_by_day.items()}
    scheduled = []

    for i, dur in enumerate(durations):
        preferred_date = start_date + timedelta(days=target_day_offsets[i])
        placed = False

        # Forward scan from preferred_date to buffer_end
        scan_end = max(buffer_end, preferred_date)
        delta = 0
        while (preferred_date + timedelta(days=delta)) <= scan_end:
            candidate = preferred_date + timedelta(days=delta)
            key = candidate.isoformat()
            if workloads.get(key, 0.0) + dur <= daily_cap + 1e-9:
                workloads[key] = workloads.get(key, 0.0) + dur
                scheduled.append({"date": candidate, "duration": dur, "due_time": due_time_str, "overloaded": False})
                placed = True
                break
            delta += 1

        if not placed:
            # Full window scan from the beginning
            candidate = start_date
            while candidate <= buffer_end:
                key = candidate.isoformat()
                if workloads.get(key, 0.0) + dur <= daily_cap + 1e-9:
                    workloads[key] = workloads.get(key, 0.0) + dur
                    scheduled.append({"date": candidate, "duration": dur, "due_time": due_time_str, "overloaded": False})
                    placed = True
                    break
                candidate += timedelta(days=1)

        if not placed:
            # Overloaded — flag it rather than silently exceeding the cap
            key = buffer_end.isoformat()
            workloads[key] = workloads.get(key, 0.0) + dur
            scheduled.append({"date": buffer_end, "duration": dur, "due_time": due_time_str, "overloaded": True})

    return scheduled

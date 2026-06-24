"""
AI task plan generator — breaks a user-created task into scheduled subtasks.

Design decisions
─────────────────
Burnout prevention
    The scheduler enforces a 6-hour/day cognitive-load cap (DAILY_CAP_HOURS).
    Load is measured in weighted hours (priority-1 × 1.4), so a high-stakes
    deadline registers heavier than routine work. Subtasks that cannot fit the
    window are flagged with overloaded=True and surface a warning to the user.

Imperative tone + structured descriptions
    SYSTEM_MESSAGE mandates strong action verbs, bans filler phrases, and
    requires a "Purpose:" line in every description so users understand not
    just what to do but why each subtask matters.

Category phases
    Each category has a canonical phase list (understand → research → draft →
    complete → review, etc.). Subtask slots are distributed across phases
    deterministically before the LLM call, so the model fills pre-assigned
    roles rather than inventing a progression from scratch.

Two-phase approach
    1. Deterministic — hours are split, review is reserved, phases assigned,
       and dates scheduled without the LLM, so the schedule is stable.
    2. Generative — the LLM only writes titles and descriptions, filling
       pre-computed slots. This prevents hallucinated dates/durations.

Review reservation
    Roughly 10–20% of total hours (minimum one session-preset chunk) is
    carved off and appended as a final review/submit/logistics slot.
    Category-specific labels guide the LLM to write appropriate content.

Priority influence
    Priority-1 tasks receive less buffer compression (window extends closer to
    the deadline), urgency-weighted hours for overload detection, and stronger
    warnings when capacity is tight.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.schemas.task_schema import TaskCreate, Task
from app.util.filterData import (
    DAILY_CAP_HOURS,
    URGENCY_WEIGHT_MULTIPLIER,
    build_active_workload_by_day,
    detect_overload,
    split_into_chunks,
    schedule_durations,
)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL")

SYSTEM_MESSAGE = (
    "You are a precise JSON-only task planning assistant. Return ONLY valid JSON.\n"
    "Every title must start with a strong action verb (Draft, Build, Research, Implement, "
    "Solve, Write, Review, Test, Configure, Outline, Analyze, Complete, etc.).\n"
    "Every description must follow this exact format:\n"
    "  Line 1: one imperative sentence stating exactly what to do.\n"
    "  Lines 2+: if multiple actions are needed, use bullet points with '- item'.\n"
    "  Final section: 'Purpose: [one sentence explaining why this subtask matters "
    "for completing the main task].'\n"
    "Banned phrases: 'you should', 'consider', 'try to', 'make sure', 'don't forget', "
    "'comprehensive', 'thorough', 'detailed'.\n"
    "Never mention dates, times, durations, tag names, or JSON keys in any output field.\n"
    "When schedule utilization is >= 80%, target minimum viable deliverables."
)

# ── Session-type presets ──────────────────────────────────────────────────────

SESSION_PRESETS: dict[str, dict] = {
    "bite_size": {
        "target_size": 0.5,
        "min_size": 0.25,
        "max_size": 0.5,
        "increment": 0.25,
    },
    "deep_work": {
        "target_size": 1.5,
        "min_size": 1.0,
        "max_size": 2.0,
        "increment": 0.5,
    },
}
_DEFAULT_SESSION = "deep_work"

SESSION_SCOPE_HINT: dict[str, str] = {
    "bite_size": (
        "Each subtask represents 15-30 minutes of focused work. "
        "Keep scope tight — one concrete micro-deliverable per subtask."
    ),
    "deep_work": (
        "Each subtask represents 1-2 hours of focused, uninterrupted work. "
        "Scope each subtask to fill a meaningful work block."
    ),
}

SESSION_TYPE_LABEL: dict[str, str] = {
    "bite_size": "Bite-sized (15-30 min per subtask) — one concrete micro-deliverable per subtask",
    "deep_work": "Deep work (1-2 hr per subtask) — one meaningful deliverable per focus block",
}

# ── Category phases ───────────────────────────────────────────────────────────
# Each phase is a named stage in the canonical work progression.
# Subtask slots are distributed across these phases before the LLM call.

CATEGORY_PHASES: dict[str, list[dict]] = {
    "homework": [
        {"name": "understand", "label": "Understand the Problem"},
        {"name": "research",   "label": "Research and Gather Notes"},
        {"name": "draft",      "label": "Outline or Draft"},
        {"name": "complete",   "label": "Complete the Work"},
        {"name": "review",     "label": "Review, Polish, and Submit"},
    ],
    "test": [
        {"name": "review_notes", "label": "Review Notes and Readings"},
        {"name": "practice",     "label": "Practice Problems"},
        {"name": "mock",         "label": "Mock Test or Timed Quiz"},
        {"name": "analyze",      "label": "Analyze Mistakes"},
        {"name": "final_review", "label": "Final Review and Recall Pass"},
    ],
    "project": [
        {"name": "requirements",  "label": "Requirements and Design"},
        {"name": "implement",     "label": "Implementation"},
        {"name": "iterate",       "label": "Iteration and Refinement"},
        {"name": "test_debug",    "label": "Testing and Debugging"},
        {"name": "polish_submit", "label": "Documentation, Polish, and Submission"},
    ],
    "interview": [
        {"name": "company_research", "label": "Company and Role Research"},
        {"name": "behavioral",       "label": "Behavioral Story Preparation"},
        {"name": "technical",        "label": "Technical and Domain Review"},
        {"name": "mock_interview",   "label": "Mock Interview Practice"},
        {"name": "logistics",        "label": "Final Logistics and Prep"},
    ],
    "skill": [
        {"name": "concepts",        "label": "Concept Introduction and Setup"},
        {"name": "guided_practice", "label": "Guided Practice with Examples"},
        {"name": "independent",     "label": "Independent Practice"},
        {"name": "apply",           "label": "Real-World Application and Reflection"},
    ],
}

_DEFAULT_PHASES: list[dict] = [
    {"name": "plan",    "label": "Planning and Understanding"},
    {"name": "execute", "label": "Execution"},
    {"name": "review",  "label": "Review and Delivery"},
]

# The canonical final phase for each category (pinned to the last slot when review is reserved)
FINAL_REVIEW_PHASE: dict[str, dict] = {
    "homework":  {"name": "review",        "label": "Review, Polish, and Submit"},
    "test":      {"name": "final_review",  "label": "Final Review and Recall Pass"},
    "project":   {"name": "polish_submit", "label": "Documentation, Polish, and Submission"},
    "interview": {"name": "logistics",     "label": "Final Logistics and Prep"},
    "skill":     {"name": "apply",         "label": "Real-World Application and Reflection"},
}
_DEFAULT_FINAL_REVIEW = {"name": "review", "label": "Final Review and Delivery"}

# ── Category guidance (injected into prompt) ──────────────────────────────────

CATEGORY_GUIDANCE: dict[str, str] = {
    "homework": (
        "understand the problem -> research / gather notes -> outline or draft -> "
        "complete the work -> review and polish"
    ),
    "test": (
        "review lecture notes and readings -> practice problems (easy -> hard) -> "
        "mock test or timed quiz -> analyze mistakes -> final review"
    ),
    "project": (
        "requirements analysis -> design / architecture -> implementation -> "
        "testing and debugging -> documentation and polish"
    ),
    "interview": (
        "company and role research -> behavioral story prep -> "
        "technical / domain review -> mock interviews -> final logistics"
    ),
    "skill": (
        "concept introduction -> guided practice with worked examples -> "
        "independent practice -> real-world application and reflection"
    ),
}

_DEFAULT_GUIDANCE = "planning -> execution -> review / delivery"

# ── Category coverage requirements (injected into prompt) ─────────────────────

CATEGORY_COVERAGE: dict[str, str] = {
    "homework":  "understanding, work completion, and review/submission",
    "test":      "review, practice, mistake analysis, and final recall",
    "project":   "requirements/design, implementation, testing/debugging, and polish/submission",
    "interview": "company research, behavioral prep, technical/domain prep, mock practice, and logistics",
    "skill":     "concept learning, guided practice, independent practice, and real-world application",
}
_DEFAULT_COVERAGE = "understanding/planning, execution, and review/delivery"

# ── Minimum estimated hours by category (for quality warnings) ────────────────

CATEGORY_MIN_HOURS: dict[str, float] = {
    "homework":  1.0,
    "test":      2.0,
    "project":   3.0,
    "interview": 3.0,
    "skill":     1.5,
}

# ── Category-aware fallback subtasks ─────────────────────────────────────────
# Used when the LLM fails or returns invalid JSON.
# Keyed by category -> phase name -> {title, description}.

CATEGORY_FALLBACKS: dict[str, dict[str, dict]] = {
    "homework": {
        "understand": {
            "title": "Analyze the Assignment Requirements",
            "description": (
                "Read the full assignment prompt and identify every deliverable.\n"
                "- Highlight unclear requirements or open questions.\n"
                "- List any missing information that must be resolved before starting.\n\n"
                "Purpose: Establish a clear understanding of the problem before investing time, "
                "so work targets the right deliverables from the start."
            ),
        },
        "research": {
            "title": "Research and Collect Source Material",
            "description": (
                "Gather notes, textbook excerpts, lecture slides, or examples relevant to the assignment.\n"
                "- Organize sources by subtopic.\n"
                "- Note key facts, formulas, or examples you will use.\n\n"
                "Purpose: Build the raw material needed to complete the assignment "
                "without interrupting the drafting phase."
            ),
        },
        "draft": {
            "title": "Outline the Structure Before Drafting",
            "description": (
                "Create a short outline mapping required sections to main ideas.\n"
                "- Add the core argument or answer for each section.\n"
                "- Mark any gaps that require additional research.\n\n"
                "Purpose: A clear outline prevents restarts mid-draft and makes the completion phase faster."
            ),
        },
        "complete": {
            "title": "Complete the Assignment",
            "description": (
                "Execute the main body of work using the outline and gathered research.\n"
                "- Work section by section, targeting one deliverable per session.\n"
                "- Leave placeholder notes for anything that needs revision later.\n\n"
                "Purpose: Produce the primary artifact so the final review has real content to improve."
            ),
        },
        "review": {
            "title": "Review, Polish, and Submit",
            "description": (
                "Read the completed assignment against the original requirements.\n"
                "- Fix any errors, omissions, or formatting issues.\n"
                "- Verify all deliverables are present and correctly formatted.\n"
                "- Submit before the deadline.\n\n"
                "Purpose: Catch errors before submission and confirm every requirement is satisfied."
            ),
        },
    },
    "test": {
        "review_notes": {
            "title": "Review Course Notes and Key Readings",
            "description": (
                "Read through lecture notes, slides, and required readings for the exam topics.\n"
                "- Mark the highest-density or most-tested concepts.\n"
                "- Summarize each topic in one sentence to confirm understanding.\n\n"
                "Purpose: Build a complete map of testable material before shifting to active practice."
            ),
        },
        "practice": {
            "title": "Solve Practice Problems",
            "description": (
                "Work through practice problems in order of difficulty, starting with the easiest.\n"
                "- Attempt each problem without notes before checking answers.\n"
                "- Mark every problem you got wrong or guessed.\n\n"
                "Purpose: Practice under realistic conditions reveals gaps that passive review misses."
            ),
        },
        "mock": {
            "title": "Complete a Full Mock Test",
            "description": (
                "Take a timed mock exam covering the full scope of testable material.\n"
                "- Simulate real conditions: no notes, strict time limit.\n"
                "- Record your score and note which sections took the most time.\n\n"
                "Purpose: A mock test builds pacing skill and surfaces weak areas before the real exam."
            ),
        },
        "analyze": {
            "title": "Analyze Mistakes and Fill Gaps",
            "description": (
                "Review every incorrect or uncertain answer from practice and mock tests.\n"
                "- Identify the root cause of each error.\n"
                "- Re-study only the material behind each error.\n\n"
                "Purpose: Targeted gap-filling is more efficient than reviewing everything again."
            ),
        },
        "final_review": {
            "title": "Execute a Final Recall Pass",
            "description": (
                "Do a rapid pass through all key concepts without looking at notes.\n"
                "- Test recall on formulas, definitions, and major examples.\n"
                "- Review any lingering weak spots from the mistake analysis.\n\n"
                "Purpose: A final recall pass consolidates memory and builds exam-day confidence."
            ),
        },
    },
    "project": {
        "requirements": {
            "title": "Define Requirements and Architecture",
            "description": (
                "Identify the functional and non-functional requirements for the project.\n"
                "- Sketch the architecture or system design at a high level.\n"
                "- List all major components, dependencies, and interfaces.\n\n"
                "Purpose: A clear requirements baseline prevents rework and keeps implementation decisions consistent."
            ),
        },
        "implement": {
            "title": "Implement the Core Functionality",
            "description": (
                "Build the primary features as defined in the requirements.\n"
                "- Write code in small, testable increments.\n"
                "- Commit working code before moving to the next feature.\n\n"
                "Purpose: Incremental implementation reduces integration risk and keeps progress visible."
            ),
        },
        "iterate": {
            "title": "Refine and Iterate on the Implementation",
            "description": (
                "Review the current implementation against requirements and improve it.\n"
                "- Fix identified issues and add missing edge cases.\n"
                "- Refactor any brittle or unclear sections.\n\n"
                "Purpose: Iteration catches design problems early, before testing reveals them as bugs."
            ),
        },
        "test_debug": {
            "title": "Test and Debug the Implementation",
            "description": (
                "Run tests against all implemented features and fix any failures.\n"
                "- Write or run tests for each component.\n"
                "- Document any known issues or limitations.\n\n"
                "Purpose: Systematic testing ensures correctness and a stable submission."
            ),
        },
        "polish_submit": {
            "title": "Document, Polish, and Submit",
            "description": (
                "Write final documentation and clean up the codebase or deliverable.\n"
                "- Add required comments, README, or report sections.\n"
                "- Run a final check against submission requirements and submit.\n\n"
                "Purpose: A polished submission demonstrates quality and satisfies all requirements."
            ),
        },
    },
    "interview": {
        "company_research": {
            "title": "Research the Company and Role",
            "description": (
                "Gather information about the company's mission, products, team, and recent news.\n"
                "- Identify the specific skills and experience the role requires.\n"
                "- Prepare two or three tailored questions to ask the interviewer.\n\n"
                "Purpose: Demonstrating company knowledge signals genuine interest and improves fit answers."
            ),
        },
        "behavioral": {
            "title": "Prepare Behavioral Stories Using STAR Format",
            "description": (
                "Draft answers to common behavioral questions using STAR (Situation, Task, Action, Result).\n"
                "- Cover leadership, conflict, failure, and collaboration scenarios.\n"
                "- Time each answer to stay under two minutes.\n\n"
                "Purpose: Rehearsed stories prevent blanking under pressure and make answers concrete."
            ),
        },
        "technical": {
            "title": "Review Technical and Domain Material",
            "description": (
                "Study the technical topics most likely to appear based on the role and job description.\n"
                "- Solve relevant practice problems or review domain concepts.\n"
                "- Write brief notes on the most important topics to reinforce recall.\n\n"
                "Purpose: Technical preparation ensures confidence when the interviewer probes specific knowledge."
            ),
        },
        "mock_interview": {
            "title": "Run a Full Mock Interview",
            "description": (
                "Conduct a timed mock interview covering behavioral and technical questions.\n"
                "- Ask a peer or use an AI tool to simulate the interview.\n"
                "- Review it for pacing, filler words, and answer quality.\n\n"
                "Purpose: A mock interview exposes gaps and builds confident answers under pressure."
            ),
        },
        "logistics": {
            "title": "Confirm Logistics and Final Preparation",
            "description": (
                "Verify all interview logistics: time, link or location, contact name, and format.\n"
                "- Prepare your environment and materials.\n"
                "- Do a final review of key stories and technical topics.\n\n"
                "Purpose: Logistics failures on interview day are preventable — resolve them in advance."
            ),
        },
    },
    "skill": {
        "concepts": {
            "title": "Learn the Core Concepts",
            "description": (
                "Study the foundational concepts, terminology, and mental models for the skill.\n"
                "- Read documentation, tutorials, or textbook chapters.\n"
                "- Summarize each key concept in your own words.\n\n"
                "Purpose: A strong conceptual foundation makes practice more effective and prevents misunderstandings."
            ),
        },
        "guided_practice": {
            "title": "Complete Guided Practice Exercises",
            "description": (
                "Work through structured exercises or tutorials with worked examples.\n"
                "- Follow each step and verify the output at each stage.\n"
                "- Note any steps you had to re-read or repeat.\n\n"
                "Purpose: Guided practice builds correct habits before moving to independent work."
            ),
        },
        "independent": {
            "title": "Practice Independently Without Guidance",
            "description": (
                "Solve problems or complete exercises without referring to tutorials.\n"
                "- Increase the difficulty with each attempt.\n"
                "- Check your answers only after completing the full problem.\n\n"
                "Purpose: Independent practice reveals real gaps and builds confidence to apply the skill without support."
            ),
        },
        "apply": {
            "title": "Apply the Skill to a Real Problem",
            "description": (
                "Use the skill in a real-world context — a project, dataset, or personal challenge.\n"
                "- Choose a problem that requires the skill non-trivially.\n"
                "- Reflect on what was easy, what was hard, and what to study next.\n\n"
                "Purpose: Real application cements learning and produces a tangible result you can demonstrate."
            ),
        },
    },
}

_DEFAULT_FALLBACKS: list[dict] = [
    {
        "title": "Plan and Understand the Scope",
        "description": (
            "Review the task requirements and identify all deliverables.\n"
            "- Break the work into concrete steps.\n"
            "- Note any open questions that must be resolved before starting.\n\n"
            "Purpose: A clear plan prevents wasted effort and keeps execution on track."
        ),
    },
    {
        "title": "Execute the Primary Work",
        "description": (
            "Complete the main body of work defined in the planning step.\n"
            "- Work in focused increments, targeting one deliverable per session.\n"
            "- Mark progress and flag any blockers immediately.\n\n"
            "Purpose: Structured execution produces the primary output the final review depends on."
        ),
    },
    {
        "title": "Review and Deliver",
        "description": (
            "Check the completed work against the original requirements.\n"
            "- Fix any errors, gaps, or formatting issues.\n"
            "- Submit or hand off before the deadline.\n\n"
            "Purpose: A final review catches errors before delivery and confirms all requirements are satisfied."
        ),
    },
]


# ── Helper: phase assignment ──────────────────────────────────────────────────

def _spread_phases(n: int, phases: list[dict]) -> list[dict]:
    """Distribute n slots evenly across the available phase list."""
    if not phases or n <= 0:
        return [{"name": "execute", "label": "Execution"}] * max(n, 0)
    return [phases[int(i * len(phases) / n)] for i in range(n)]


def _assign_phases(n_slots: int, category: str | None, review_reserved: bool) -> list[dict]:
    """
    Return a list of phase dicts, one per slot.
    If review_reserved, the last slot is pinned to the category's final review phase.
    """
    phases = CATEGORY_PHASES.get(category or "", _DEFAULT_PHASES)
    if n_slots <= 0:
        return []
    if review_reserved and n_slots >= 2:
        final_phase = FINAL_REVIEW_PHASE.get(category or "", _DEFAULT_FINAL_REVIEW)
        main_phases = [p for p in phases if p["name"] != final_phase["name"]]
        return _spread_phases(n_slots - 1, main_phases) + [final_phase]
    return _spread_phases(n_slots, phases)


# ── Helper: review reservation ────────────────────────────────────────────────

def _reserve_review(total_hours: float, preset: dict) -> tuple[float, float]:
    """
    Return (main_hours, review_hours).
    Carves off 10-20% of total as a final review session (minimum one preset chunk).
    Returns (total_hours, 0.0) when total is too small to split meaningfully.
    """
    min_size  = preset["min_size"]
    increment = preset["increment"]

    if total_hours < min_size * 2:
        return total_hours, 0.0

    # Target ~15%, bounded to [min_size, min_size * 2], rounded to nearest increment
    raw = total_hours * 0.15
    rev = round(round(raw / increment) * increment, 10)
    rev = max(min_size, min(rev, min_size * 2))
    main = total_hours - rev

    if main < min_size:
        return total_hours, 0.0

    return main, rev


# ── Helper: priority buffer days ──────────────────────────────────────────────

def _priority_buffer_days(priority: Optional[int]) -> int:
    """Days to subtract from due_date when computing buffer_end."""
    if priority == 1:
        return 1   # high priority — less buffer compression, more scheduling room
    if priority is not None and priority >= 3:
        return 3   # low priority — more conservative
    return 2       # default


# ── Helper: domain constraints prompt section ─────────────────────────────────

def _build_domain_constraints(tags: list) -> str:
    if not tags:
        return (
            "No domain tags provided — infer domain context from the task title "
            "and description only."
        )
    tag_names = [t.name for t in tags]
    lines = [
        f"Active domain constraints from tags: {', '.join(tag_names)}",
        "These tags represent tools, technologies, courses, or domain areas. "
        "Actions and deliverables MUST reflect them:",
    ] + [f"  - '{n}': use {n}-specific actions, examples, and tools." for n in tag_names]
    return "\n".join(lines)


# ── Helper: workload context ──────────────────────────────────────────────────

def _build_workload_context(overload_info: dict) -> str:
    utilization     = overload_info["utilization_pct"]
    available       = overload_info["available_hours"]
    overloaded_days = overload_info["overloaded_days"]
    can_fit         = overload_info["can_fit"]

    lines = [
        f"Window utilization: {utilization}% of daily cap already consumed.",
        f"Available slack: {available}h remaining in the scheduling window.",
    ]
    if overloaded_days > 0:
        lines.append(
            f"Overloaded days: {overloaded_days} day(s) are already at or above "
            f"the {DAILY_CAP_HOURS}h daily cap."
        )
    if not can_fit:
        lines.append(
            "Capacity warning: new task hours exceed remaining slack — "
            "some subtasks will be force-placed beyond the daily cap."
        )
    elif utilization >= 80:
        lines.append(
            "Schedule is tight. Descriptions must target minimum viable "
            "deliverables and eliminate any non-essential scope."
        )
    else:
        lines.append("Schedule has comfortable headroom — standard pacing applies.")
    return "\n".join(f"  {line}" for line in lines)


# ── Helper: plan quality warnings ─────────────────────────────────────────────

def _collect_plan_warnings(
    new_task: TaskCreate,
    total_hours: float,
    overload_info: dict,
    start_date,
    buffer_end,
    due_date,
) -> list[str]:
    warnings: list[str] = []
    category = (new_task.category or "").lower()
    priority = new_task.priority

    if not (new_task.description or "").strip():
        warnings.append(
            "No task description was provided — generated subtasks may be generic. "
            "Add a description for more specific subtask content."
        )

    min_h = CATEGORY_MIN_HOURS.get(category, 0)
    if min_h > 0 and total_hours < min_h:
        warnings.append(
            f"The estimated time ({total_hours}h) seems low for a {category} task. "
            f"A typical {category} takes at least {min_h}h — consider revising the estimate."
        )

    if due_date:
        days_until_due = (due_date - start_date).days
        if days_until_due < 1:
            warnings.append(
                "The due date is today or in the past — scheduling is severely constrained."
            )
        elif buffer_end == due_date and days_until_due <= 2:
            warnings.append(
                "The due date leaves no review buffer. "
                "Aim to complete the main work at least one day before the deadline."
            )

    if not overload_info["can_fit"]:
        available = overload_info["available_hours"]
        if priority == 1:
            warnings.append(
                f"High-priority task cannot fit in the available window — "
                f"only {available}h of slack remain vs. {total_hours}h estimated. "
                "Reschedule other tasks or extend the deadline to avoid overload."
            )
        else:
            warnings.append(
                f"This task exceeds available schedule capacity — only {available}h of slack remain. "
                "Some subtasks will be placed past the daily cap."
            )
    elif priority == 1 and overload_info["available_hours"] < total_hours * 1.5:
        warnings.append(
            f"High-priority task has limited slack ({overload_info['available_hours']}h available "
            f"vs. {total_hours}h estimated). Protect this time from other commitments."
        )

    return warnings


# ── Helper: fallback subtask ──────────────────────────────────────────────────

def _build_fallback(phase: dict, category: str | None, index: int) -> dict:
    cat_fb   = CATEGORY_FALLBACKS.get(category or "", {})
    phase_fb = cat_fb.get(phase.get("name", ""), None)
    if phase_fb:
        return phase_fb
    return _DEFAULT_FALLBACKS[index % len(_DEFAULT_FALLBACKS)]


# ── Prompt template ───────────────────────────────────────────────────────────

BASE_INSTRUCTIONS = """
Generate EXACTLY {n} subtasks for the task below.
Overall progression: {guidance}

PHASE ASSIGNMENTS — each slot has a required phase; generate content that fits it:
{phase_assignments}

DOMAIN CONSTRAINTS:
{domain_constraints}

SCOPE: {scope_hint}

TASK CONTEXT (read-only — do NOT copy values into titles or descriptions):
- Total estimated hours: {total_hours}h across {n} subtask(s)
- Session type: {session_type_label}
- Due: {due_date_label}{due_time_label}

SCHEDULE CONTEXT (read-only — calibrate pacing only, do NOT mention in output):
{workload_context}

REQUIRED COVERAGE for this category: {category_coverage}

TONE — non-negotiable:
- Every title MUST start with a strong action verb.
- Banned: "you should", "consider", "try to", "make sure", "don't forget", "comprehensive", "thorough", "detailed".
- Each subtask must be distinct and move the work forward.
- If schedule utilization is >= 80%, target minimum viable deliverables.

DESCRIPTION FORMAT — required for every subtask:
  Line 1: one imperative sentence stating exactly what to do.
  Lines 2+: if multiple actions are needed, use bullet points ("- item").
  Final section: "Purpose: [one sentence why this subtask matters for the main task]."

  Example:
    "Create a short outline before writing.
    - Identify the required sections.
    - Add the main idea for each section.
    - Mark any missing information to research next.

    Purpose: Build a clear structure before drafting so the final work is easier to complete and review."

CONTENT:
- Each subtask must match its assigned phase above.
- Include one concrete, measurable deliverable per subtask.
- Subtasks must increase in difficulty or scope across the sequence.
- Stay domain-relevant to the tags and task description.
- Do NOT mention time, dates, tag names, or JSON keys in titles or descriptions.

Task title: {title}
Task description: {description}
""".strip()


# ── Main service function ─────────────────────────────────────────────────────

async def create_subtasks_with_llm(
    active_tasks: list[Task],
    new_task: TaskCreate,
    created_task: Task,
) -> dict:
    """
    Return:
        "subtasks":        list of subtask dicts ready for persistence
        "overload_warning": str | None   — non-None if schedule is tight
        "plan_warnings":    list[str]    — quality/planning notices
    """

    active_workload_by_day = build_active_workload_by_day(active_tasks)

    # ── STEP 1: Split hours + reserve review ────────────────────────────────
    total_hours  = float(new_task.estimated_time or 0.0)
    session_type = (new_task.session_type or _DEFAULT_SESSION).lower()
    preset       = SESSION_PRESETS.get(session_type, SESSION_PRESETS[_DEFAULT_SESSION])
    priority     = new_task.priority
    category     = (new_task.category or "").lower() or None

    main_hours, review_hours = _reserve_review(total_hours, preset)
    review_reserved = review_hours > 0

    if main_hours <= preset["min_size"]:
        main_durations = [main_hours] if main_hours > 0 else [preset["min_size"]]
    else:
        main_durations = split_into_chunks(main_hours, **preset)

    durations     = main_durations + ([review_hours] if review_reserved else [])
    subtask_count = len(durations)

    # ── STEP 2: Assign phases deterministically ──────────────────────────────
    phases = _assign_phases(subtask_count, category, review_reserved)

    # ── STEP 3: Build scheduling window (priority affects buffer compression) ─
    today       = datetime.now(timezone.utc).date()
    start_date  = today
    due_date    = new_task.due_date
    buffer_days = _priority_buffer_days(priority)

    if not due_date:
        buffer_end = start_date + timedelta(days=max(0, subtask_count - 1))
    else:
        buffer_end = due_date - timedelta(days=buffer_days)
        if buffer_end < start_date:
            buffer_end = due_date

    # ── STEP 4: Overload detection (urgency-weighted for priority-1) ─────────
    urgency_mult   = URGENCY_WEIGHT_MULTIPLIER if priority == 1 else 1.0
    weighted_hours = total_hours * urgency_mult

    overload_info = detect_overload(
        workload_by_day=active_workload_by_day,
        start_date=start_date,
        end_date=buffer_end,
        new_task_hours=weighted_hours,
        daily_cap=DAILY_CAP_HOURS,
    )

    overload_warning: str | None = None
    if not overload_info["can_fit"]:
        overload_warning = (
            f"Your schedule is {overload_info['utilization_pct']}% full between "
            f"{start_date} and {buffer_end}. Only {overload_info['available_hours']}h "
            f"of slack remain — this plan may push some subtasks past the daily cap."
        )

    # ── STEP 5: Collect plan quality warnings ───────────────────────────────
    plan_warnings = _collect_plan_warnings(
        new_task, total_hours, overload_info, start_date, buffer_end, due_date
    )

    # ── STEP 6: Build LLM prompt ────────────────────────────────────────────
    guidance           = CATEGORY_GUIDANCE.get(category or "", _DEFAULT_GUIDANCE)
    scope_hint         = SESSION_SCOPE_HINT.get(session_type, SESSION_SCOPE_HINT[_DEFAULT_SESSION])
    session_type_label = SESSION_TYPE_LABEL.get(session_type, session_type)
    due_date_label     = str(due_date) if due_date else "not set"
    due_time_label     = f" at {new_task.due_time}" if new_task.due_time else ""
    workload_context   = _build_workload_context(overload_info)
    domain_constraints = _build_domain_constraints(new_task.tags or [])
    category_coverage  = CATEGORY_COVERAGE.get(category or "", _DEFAULT_COVERAGE)

    phase_assignments = "\n".join(
        f"  Slot {i + 1} of {subtask_count}: Phase -- \"{p['label']}\""
        for i, p in enumerate(phases)
    )

    semantic_prompt = BASE_INSTRUCTIONS.format(
        n=subtask_count,
        guidance=guidance,
        phase_assignments=phase_assignments,
        domain_constraints=domain_constraints,
        scope_hint=scope_hint,
        total_hours=total_hours,
        session_type_label=session_type_label,
        due_date_label=due_date_label,
        due_time_label=due_time_label,
        workload_context=workload_context,
        category_coverage=category_coverage,
        title=new_task.title,
        description=new_task.description or "(no description provided)",
    ) + f"""

Return ONLY a valid JSON array with EXACTLY {subtask_count} objects.
Each object must have exactly two string keys:
{{"title": "...", "description": "..."}}
"""

    # ── STEP 7: LLM call ────────────────────────────────────────────────────
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": semantic_prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        llm_text   = (response.choices[0].message.content or "").strip()
        json_start = llm_text.find("[")
        json_end   = llm_text.rfind("]") + 1
        raw_items  = json.loads(llm_text[json_start:json_end])
        if not isinstance(raw_items, list):
            raw_items = []
    except Exception:
        raw_items = []

    # ── STEP 8: Validate + pad/truncate LLM output ──────────────────────────
    valid_items: list[dict] = []
    for item in raw_items:
        if (
            isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("description"), str)
            and item["title"].strip()
            and item["description"].strip()
        ):
            valid_items.append({
                "title": item["title"].strip(),
                "description": item["description"].strip(),
            })

    # Pad with category-aware fallbacks when LLM output is short or invalid
    while len(valid_items) < subtask_count:
        i     = len(valid_items)
        phase = phases[i] if i < len(phases) else _DEFAULT_FINAL_REVIEW
        valid_items.append(_build_fallback(phase, category, i))

    valid_items = valid_items[:subtask_count]

    # ── STEP 9: Schedule durations onto dates ────────────────────────────────
    scheduled = schedule_durations(
        durations,
        start_date,
        buffer_end,
        active_workload_by_day,
        daily_cap=DAILY_CAP_HOURS,
        due_time_str=str(new_task.due_time) if new_task.due_time else None,
    )

    if any(item.get("overloaded") for item in scheduled):
        overload_warning = overload_warning or (
            "One or more subtasks could not be placed within the daily cap "
            "and were scheduled on the last available day. Consider reducing "
            "scope or extending the due date."
        )

    # ── STEP 10: Merge semantic content + schedule ───────────────────────────
    final_subtasks = []
    for i, slot in enumerate(scheduled):
        sem = valid_items[i]
        final_subtasks.append({
            "parent_task_id": created_task["id"],
            "title": sem["title"],
            "description": sem["description"],
            "category": new_task.category,
            "session_type": session_type,
            "due_date": slot["date"],
            "due_time": slot["due_time"],
            "estimated_time": slot["duration"],
            "tags": new_task.tags if new_task.tags else [],
        })

    return {
        "subtasks": final_subtasks,
        "overload_warning": overload_warning,
        "plan_warnings": plan_warnings,
    }

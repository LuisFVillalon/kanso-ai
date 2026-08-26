"""Public entrypoint: orchestrates the note-to-resource pipeline.

1. extract_search_plan() — LLM turns the note into a topic + three queries.
2. collect_candidates()  — web search + filtering + credibility scoring.
3. rank_verified_resources() — deterministic pick of the best candidate per
   resource type, with a plain-language explanation.
"""

from typing import Any, Dict

from app.services.learning_resources.note_content import StructuredNoteContent
from app.services.learning_resources.planner import extract_search_plan
from app.services.learning_resources.ranking import rank_verified_resources
from app.services.learning_resources.search import collect_candidates

MIN_CONTENT_LENGTH = 20


async def get_learning_resources(note_content: StructuredNoteContent) -> Dict[str, Any]:
    if note_content.total_length() < MIN_CONTENT_LENGTH:
        return {
            "topic": "",
            "resources": [],
            "message": (
                "Add a bit more to your note — a title, heading, list, table, or "
                "highlighted/styled text — so we can figure out what to look for."
            ),
        }

    plan = await extract_search_plan(note_content)
    candidates = await collect_candidates(plan)

    if not candidates:
        return {
            "topic": plan.get("topic", ""),
            "resources": [],
        }

    ranked = rank_verified_resources(plan, candidates)

    return {
        "topic": ranked.get("topic") or plan.get("topic", ""),
        "resources": ranked.get("resources", []),
    }

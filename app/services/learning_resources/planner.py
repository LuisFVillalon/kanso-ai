import json
from typing import Any

from app.services.learning_resources.ai_client import MODEL_NAME, client, gemini_call_kwargs
from app.services.learning_resources.note_content import StructuredNoteContent

SEARCH_PLAN_SCHEMA = {
    "name": "learning_resource_search_plan",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "learner_level": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced", "unknown"],
            },
            "video_query": {"type": "string"},
            "article_query": {"type": "string"},
            "exercise_query": {"type": "string"},
        },
        "required": [
            "topic",
            "learner_level",
            "video_query",
            "article_query",
            "exercise_query",
        ],
    },
}


async def extract_search_plan(note_content: StructuredNoteContent) -> dict[str, Any]:
    system = (
        "You analyze user notes and create search queries for learning resources. "
        "Do not name fake resources. Create broad but useful queries that can work "
        "for any domain: school, hobbies, fitness, cooking, business, art, trades, "
        "languages, personal productivity, coding, history, music, and more. "
        "The notes are provided as labeled sections (Title, Headings, Highlights, "
        "Lists, Styled text, Tables, Notes) in descending order of importance — "
        "weigh earlier sections more heavily when they conflict with later ones."
    )

    user = f"""
Analyze these notes:

{note_content.to_prompt_sections()}

Create:
1. A core topic.
2. The likely learner level.
3. One search query for a helpful video.
4. One search query for a helpful article.
5. One search query for a helpful exercise, activity, worksheet, tutorial, practice, or interactive resource.

Avoid overly academic wording unless the notes are academic.
"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=400,
        response_format={
            "type": "json_schema",
            "json_schema": SEARCH_PLAN_SCHEMA,
        },
        **gemini_call_kwargs(),
    )

    return json.loads(response.choices[0].message.content or "{}")

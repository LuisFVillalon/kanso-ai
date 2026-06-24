import asyncio
import json
import os
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ddgs import DDGS

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# -----------------------------
# Basic cleanup
# -----------------------------

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


# -----------------------------
# Credibility scorer
# -----------------------------

LOW_QUALITY_DOMAINS = [
    "pinterest.",
    "quora.",
    "reddit.",
    "medium.com",
    "tumblr.",
    "fandom.",
    "answers.com",
    "wikihow.",
]

CLICKBAIT_TERMS = [
    "shocking",
    "secret",
    "you won't believe",
    "ultimate hack",
    "top 10",
    "top ten",
    "insane",
    "crazy",
]

HIGH_TRUST_DOMAIN_HINTS = [
    ".edu",
    ".gov",
    ".org",
    "museum",
    "library",
    "archive",
    "university",
    "college",
    "academy",
    "institute",
    "official",
]

HIGH_TRUST_TITLE_TERMS = [
    "official",
    "documentation",
    "guide",
    "lesson",
    "course",
    "tutorial",
    "introduction",
    "explained",
    "lecture",
    "practice",
    "exercise",
    "module",
]


def credibility_score(result: Dict[str, Any]) -> int:
    score = 0

    url = result.get("url", "").lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    domain = _domain(url)

    # Trust signals
    if any(hint in url for hint in HIGH_TRUST_DOMAIN_HINTS):
        score += 3

    if any(term in title for term in HIGH_TRUST_TITLE_TERMS):
        score += 2

    if any(term in snippet for term in HIGH_TRUST_TITLE_TERMS):
        score += 1

    # YouTube can be good for video, but not automatically authoritative
    if "youtube.com" in domain or "youtu.be" in domain:
        score += 1

    # Penalize low-quality/general forum/content farm sources
    if any(bad in url for bad in LOW_QUALITY_DOMAINS):
        score -= 3

    if any(term in title for term in CLICKBAIT_TERMS):
        score -= 3

    if len(snippet) < 40:
        score -= 1

    return score


# -----------------------------
# LLM schemas
# -----------------------------

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

RANKED_RESOURCES_SCHEMA = {
    "name": "ranked_learning_resources",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "resources": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["video", "article", "exercise"],
                        },
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "why": {"type": "string"},
                        "platform": {"type": "string"},
                        "activity_label": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "title",
                        "url",
                        "why",
                        "platform",
                        "activity_label",
                    ],
                },
            },
        },
        "required": ["topic", "resources"],
    },
}


# -----------------------------
# LLM: extract search plan
# -----------------------------

async def extract_search_plan(note_content: str) -> Dict[str, Any]:
    system = (
        "You analyze user notes and create search queries for learning resources. "
        "Do not name fake resources. Create broad but useful queries that can work "
        "for any domain: school, hobbies, fitness, cooking, business, art, trades, "
        "languages, personal productivity, coding, history, music, and more."
    )

    user = f"""
Analyze these notes:

{note_content}

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
    )

    return json.loads(response.choices[0].message.content or "{}")


# -----------------------------
# Web search using DuckDuckGo
# pip install ddgs
# -----------------------------

def _search_sync(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            })

    return results


async def search_web(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_search_sync, query, max_results)


def _is_youtube(url: str) -> bool:
    d = _domain(url)
    return "youtube.com" in d or "youtu.be" in d


async def collect_candidates(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    video_query = plan["video_query"] + " site:youtube.com"
    searches = [
        ("video", video_query),
        ("article", plan["article_query"]),
        ("exercise", plan["exercise_query"]),
    ]

    all_candidates = []

    for resource_type, query in searches:
        results = await search_web(query, max_results=8)

        for result in results:
            if resource_type == "video" and not _is_youtube(result.get("url", "")):
                continue
            result["type"] = resource_type
            result["credibility_score"] = credibility_score(result)
            result["domain"] = _domain(result.get("url", ""))
            all_candidates.append(result)

    # Remove duplicates by URL
    seen = set()
    unique = []

    for item in all_candidates:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)

    # Keep only reasonably credible candidates
    unique.sort(key=lambda x: x["credibility_score"], reverse=True)

    return unique[:24]


# -----------------------------
# LLM: rank verified candidates
# -----------------------------

async def rank_verified_resources(
    note_content: str,
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    compact_candidates = [
        {
            "type": c["type"],
            "title": c["title"],
            "url": c["url"],
            "snippet": c["snippet"],
            "domain": c["domain"],
            "credibility_score": c["credibility_score"],
        }
        for c in candidates
    ]

    system = (
        "You are a learning resource curator. "
        "You may only choose from the provided candidates. "
        "Never invent titles, platforms, or URLs. "
        "Prefer reputable, educational, official, expert, nonprofit, established publication, "
        "museum, university, government, or well-known instructional sources. "
        "Avoid clickbait, thin SEO pages, random blogs, forums, and low-quality summaries."
    )

    user = f"""
User notes:

{note_content}

Detected topic:
{plan.get("topic")}

Learner level:
{plan.get("learner_level")}

Candidates:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Choose exactly one video, one article, and one exercise if available.
If a category has no good candidate, omit that category.

The "why" field must be under 15 words and use coaching tone.
Use activity labels like Watch, Read, Practice, Try, Drill, Reflect, Build, Cook, Train, Sketch, or Review.
"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=700,
        response_format={
            "type": "json_schema",
            "json_schema": RANKED_RESOURCES_SCHEMA,
        },
    )

    return json.loads(response.choices[0].message.content or "{}")


# -----------------------------
# Public function
# -----------------------------

async def get_learning_resources(note_content: str) -> Dict[str, Any]:
    clean = _strip_html(note_content)[:1500]

    if len(clean) < 20:
        return {"topic": "", "resources": []}

    plan = await extract_search_plan(clean)
    candidates = await collect_candidates(plan)

    if not candidates:
        return {
            "topic": plan.get("topic", ""),
            "resources": [],
        }

    ranked = await rank_verified_resources(clean, plan, candidates)

    return {
        "topic": ranked.get("topic") or plan.get("topic", ""),
        "resources": ranked.get("resources", []),
    }

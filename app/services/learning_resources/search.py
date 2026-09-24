import asyncio
from typing import Any

import httpx
from ddgs import DDGS

from app.services.learning_resources.credibility import credibility_score
from app.services.learning_resources.filters import (
    is_course_or_tutorial,
    is_paywalled,
    is_sponsored,
    is_youtube,
)
from app.services.learning_resources.text_matching import domain_of

MAX_CANDIDATES_PER_QUERY = 5
MAX_RETURNED_CANDIDATES = 15

DEAD_LINK_TIMEOUT = 6.0
DEAD_LINK_USER_AGENT = "Mozilla/5.0 (compatible; KansoLinkChecker/1.0)"


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            })

    return results


async def search_web(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_search_sync, query, max_results)


async def _is_url_alive(client: httpx.AsyncClient, url: str) -> bool:
    """A HEAD request is enough to catch 404s/dead domains cheaply. Some sites
    reject HEAD (405) or misconfigure it, so those fall back to a real GET
    before being counted as dead."""
    try:
        response = await client.head(url, timeout=DEAD_LINK_TIMEOUT, follow_redirects=True)
        if response.status_code in (405, 403) or response.status_code >= 500:
            response = await client.get(url, timeout=DEAD_LINK_TIMEOUT, follow_redirects=True)
        return response.status_code < 400
    except httpx.HTTPError:
        return False


async def _filter_dead_links(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    async with httpx.AsyncClient(headers={"User-Agent": DEAD_LINK_USER_AGENT}) as client:
        alive_flags = await asyncio.gather(
            *(_is_url_alive(client, c["url"]) for c in candidates)
        )

    return [c for c, alive in zip(candidates, alive_flags, strict=True) if alive]


def _passes_type_filters(resource_type: str, result: dict[str, Any]) -> bool:
    if is_sponsored(result):
        return False
    if resource_type in ("article", "exercise") and is_paywalled(result):
        return False
    if resource_type == "article" and is_course_or_tutorial(result):
        return False
    if resource_type == "video" and not is_youtube(result.get("url", "")):
        return False
    return True


def _deduplicate_by_url(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []

    for item in results:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)

    return unique


async def collect_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    searches = [
        ("video", plan["video_query"] + " site:youtube.com"),
        ("article", plan["article_query"]),
        ("exercise", plan["exercise_query"]),
    ]

    all_candidates = []

    for resource_type, query in searches:
        results = await search_web(query, max_results=MAX_CANDIDATES_PER_QUERY)

        for result in results:
            if not _passes_type_filters(resource_type, result):
                continue
            result["type"] = resource_type
            result["credibility_score"] = credibility_score(result)
            result["domain"] = domain_of(result.get("url", ""))
            all_candidates.append(result)

    unique = _deduplicate_by_url(all_candidates)

    # Drop dead links / 404s before anything gets ranked or returned
    unique = await _filter_dead_links(unique)

    # Keep only the most credible candidates
    unique.sort(key=lambda x: x["credibility_score"], reverse=True)

    return unique[:MAX_RETURNED_CANDIDATES]

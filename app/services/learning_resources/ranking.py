"""Deterministic ranking of already-verified candidates — no LLM call. One
candidate per resource type is picked (highest credibility score) and given a
plain-language explanation of why it was chosen."""

from typing import Any, Dict, List

from app.services.learning_resources.credibility import credibility_reasons

ACTIVITY_LABELS = {
    "video": "Watch",
    "article": "Read",
    "exercise": "Practice",
}

KNOWN_PLATFORMS = {
    "youtube.com": "YouTube",
    "khanacademy.org": "Khan Academy",
    "coursera.org": "Coursera",
    "udemy.com": "Udemy",
    "edx.org": "edX",
}


def _platform_from_domain(domain: str) -> str:
    if domain in KNOWN_PLATFORMS:
        return KNOWN_PLATFORMS[domain]
    name = domain.split(".")[0]
    return name.capitalize() if name else domain


def _survival_reasons(candidate: Dict[str, Any]) -> List[str]:
    """Every ranked candidate already passed collect_candidates()'s filters —
    these state that plainly, so the explanation doesn't only list the
    (possibly empty) set of credibility signals that happened to fire."""
    reasons = ["link is not sponsored", "link is not a dead link"]

    if candidate["type"] in ("article", "exercise"):
        reasons.append("link is not paywalled")
    if candidate["type"] == "article":
        reasons.append("link is not a course or tutorial listing")

    return reasons


def _explain_choice(candidate: Dict[str, Any]) -> str:
    reasons = credibility_reasons(candidate) + _survival_reasons(candidate)
    return f"Picked {candidate['domain']}: {'; '.join(reasons)}."


def rank_verified_resources(
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    topic = plan.get("topic", "")

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)

    resources = []
    for resource_type in ("video", "article", "exercise"):
        type_candidates = by_type.get(resource_type)
        if not type_candidates:
            continue
        best = max(type_candidates, key=lambda c: c["credibility_score"])
        resources.append(
            {
                "type": resource_type,
                "title": best["title"],
                "url": best["url"],
                "why": _explain_choice(best),
                "platform": _platform_from_domain(best["domain"]),
                "activity_label": ACTIVITY_LABELS[resource_type],
            }
        )

    return {"topic": topic, "resources": resources}

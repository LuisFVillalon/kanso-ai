"""Small, reusable text-matching primitives shared by the credibility scorer
and the sponsored/paywall/course-listing filters, so each of those stays a
short list of terms instead of its own hand-rolled string-matching logic."""

from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def first_match(text: str, terms: Iterable[str]) -> Optional[str]:
    """Returns the first term found in text, or None."""
    return next((term for term in terms if term in text), None)


def result_fields(result: Dict[str, Any]) -> Dict[str, str]:
    """Lower-cased url/title/snippet/domain, computed once per lookup."""
    url = result.get("url", "").lower()
    return {
        "url": url,
        "title": result.get("title", "").lower(),
        "snippet": result.get("snippet", "").lower(),
        "domain": domain_of(url),
    }


def matches_any_field(
    result: Dict[str, Any],
    *,
    url_terms: Iterable[str] = (),
    domain_terms: Iterable[str] = (),
    title_terms: Iterable[str] = (),
    snippet_terms: Iterable[str] = (),
) -> bool:
    """True if any of the given term lists is found in the matching field of
    `result`. Search results are checked against several unrelated keyword
    lists (sponsored markers, paywall domains, course-listing terms, ...) in
    exactly this shape, so this is the one place that shape is implemented."""
    fields = result_fields(result)
    return any(
        first_match(fields[field_name], terms) is not None
        for field_name, terms in (
            ("url", url_terms),
            ("domain", domain_terms),
            ("title", title_terms),
            ("snippet", snippet_terms),
        )
    )

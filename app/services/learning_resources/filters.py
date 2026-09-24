"""Pass/fail checks applied to raw search results before they're scored and
ranked. Each filter is just a keyword list plus which fields to check it
against — see text_matching.matches_any_field for the shared matching logic."""

from typing import Any

from app.services.learning_resources.text_matching import domain_of, matches_any_field

# Ad/tracking redirects and sponsored-listing markers DDG can surface alongside
# organic results.
SPONSORED_URL_HINTS = [
    "duckduckgo.com/y.js",
    "bing.com/aclick",
    "googleadservices.com",
    "googlesyndication.com",
    "doubleclick.net",
    "/aclk?",
]

SPONSORED_TEXT_TERMS = [
    "sponsored",
    "advertisement",
]

# Outlets with a hard subscription/metered paywall on most articles. Sites like
# medium.com are deliberately excluded here since only some of their stories
# are member-only — those are caught by PAYWALL_TEXT_MARKERS instead.
PAYWALL_DOMAINS = [
    "nytimes.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "bloomberg.com",
    "washingtonpost.com",
    "newyorker.com",
    "businessinsider.com",
    "forbes.com",
    "thetimes.co.uk",
    "telegraph.co.uk",
    "hbr.org",
    "seekingalpha.com",
    "latimes.com",
]

PAYWALL_TEXT_MARKERS = [
    "member-only",
    "members only",
    "member only",
    "members-only",
    "subscriber exclusive",
    "subscribers only",
    "subscription required",
    "sign in to read",
    "sign up to read",
    "unlock this story",
    "paywall",
]

# Articles should be write-ups/explainers, not course listings, tutorials, or
# how-to guides — those belong to the "video"/"exercise" resource types.
COURSE_TUTORIAL_DOMAINS = [
    "coursera.org",
    "udemy.com",
    "edx.org",
    "khanacademy.org",
    "skillshare.com",
    "pluralsight.com",
    "codecademy.com",
    "udacity.com",
    "linkedin.com/learning",
    "masterclass.com",
]

COURSE_TUTORIAL_TERMS = [
    "course",
    "tutorial",
    "how to",
    "how-to",
    "lesson",
    "class",
    "bootcamp",
    "masterclass",
    "workshop",
    "certification",
    "curriculum",
    "training program",
    "step by step guide",
    "step-by-step guide",
    "walkthrough",
]


def is_youtube(url: str) -> bool:
    domain = domain_of(url)
    return "youtube.com" in domain or "youtu.be" in domain


def is_sponsored(result: dict[str, Any]) -> bool:
    return matches_any_field(
        result,
        url_terms=SPONSORED_URL_HINTS,
        title_terms=SPONSORED_TEXT_TERMS,
        snippet_terms=SPONSORED_TEXT_TERMS,
    )


def is_paywalled(result: dict[str, Any]) -> bool:
    return matches_any_field(
        result,
        domain_terms=PAYWALL_DOMAINS,
        title_terms=PAYWALL_TEXT_MARKERS,
        snippet_terms=PAYWALL_TEXT_MARKERS,
    )


def is_course_or_tutorial(result: dict[str, Any]) -> bool:
    return matches_any_field(
        result,
        domain_terms=COURSE_TUTORIAL_DOMAINS,
        title_terms=COURSE_TUTORIAL_TERMS,
        snippet_terms=COURSE_TUTORIAL_TERMS,
    )

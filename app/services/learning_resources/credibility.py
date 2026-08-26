"""Scores how trustworthy a search result looks, and can explain that score
in plain language.

Scoring and explaining used to be two separate functions that each hard-coded
the same signals (trust keywords, low-quality domains, clickbait phrases,
...). That meant adding or tuning a signal required touching both in lockstep,
and nothing would catch it if they drifted apart. Here each signal is defined
once, as data, and both credibility_score() and credibility_reasons() derive
from that single table.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.services.learning_resources.text_matching import first_match, result_fields

LOW_QUALITY_DOMAINS = [
    "pinterest.",
    "quora.",
    "wikipedia.",
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

MIN_SNIPPET_LENGTH = 40


@dataclass(frozen=True)
class CredibilitySignal:
    weight: int
    # Given the result's lower-cased fields, returns the matched value (e.g.
    # the term that hit) or None if the signal didn't fire.
    find: Callable[[Dict[str, str]], Optional[str]]
    # Given (matched value, fields), renders the human-readable reason.
    describe: Callable[[str, Dict[str, str]], str]


def _term_signal(
    field_name: str,
    terms: List[str],
    weight: int,
    describe: Callable[[str, Dict[str, str]], str],
) -> CredibilitySignal:
    return CredibilitySignal(
        weight=weight,
        find=lambda fields: first_match(fields[field_name], terms),
        describe=describe,
    )


def _short_snippet(fields: Dict[str, str]) -> Optional[str]:
    return "short" if len(fields["snippet"]) < MIN_SNIPPET_LENGTH else None


CREDIBILITY_SIGNALS: List[CredibilitySignal] = [
    _term_signal(
        "url", HIGH_TRUST_DOMAIN_HINTS, 3,
        lambda term, fields: f'domain "{fields["domain"]}" matched trust keyword "{term}"',
    ),
    _term_signal(
        "title", HIGH_TRUST_TITLE_TERMS, 2,
        lambda term, fields: f'title matched with keyword "{term}"',
    ),
    _term_signal(
        "snippet", HIGH_TRUST_TITLE_TERMS, 1,
        lambda term, fields: f'snippet matched with keyword "{term}"',
    ),
    _term_signal(
        "url", LOW_QUALITY_DOMAINS, -3,
        lambda term, fields: f'domain is on the low-quality source list ("{term}")',
    ),
    _term_signal(
        "title", CLICKBAIT_TERMS, -3,
        lambda term, fields: f'title matched a clickbait phrase ("{term}")',
    ),
    CredibilitySignal(
        weight=-1,
        find=_short_snippet,
        describe=lambda _, fields: f"snippet is very short (under {MIN_SNIPPET_LENGTH} characters)",
    ),
]


def _fired_signals(result: Dict[str, Any]) -> List[CredibilitySignal]:
    fields = result_fields(result)
    return [signal for signal in CREDIBILITY_SIGNALS if signal.find(fields) is not None]


def credibility_score(result: Dict[str, Any]) -> int:
    return sum(signal.weight for signal in _fired_signals(result))


def credibility_reasons(result: Dict[str, Any]) -> List[str]:
    fields = result_fields(result)
    return [signal.describe(signal.find(fields), fields) for signal in _fired_signals(result)]

from dataclasses import dataclass, field
from typing import List


@dataclass
class StructuredNoteContent:
    """The condensed, per-category note representation produced by the
    frontend extractor. Categories are already ordered by importance
    (title > headings > highlights > lists > styled_text > tables) and the
    combined content is already capped under ~1500 chars — this class just
    carries that structure through to the LLM prompts instead of collapsing
    it into a run-on string.
    """

    title: str = ""
    headings: List[str] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    lists: List[str] = field(default_factory=list)
    styled_text: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    plain_text: str = ""

    def total_length(self) -> int:
        return (
            len(self.title)
            + sum(len(x) for x in self.headings)
            + sum(len(x) for x in self.highlights)
            + sum(len(x) for x in self.lists)
            + sum(len(x) for x in self.styled_text)
            + sum(len(x) for x in self.tables)
            + len(self.plain_text)
        )

    def to_prompt_sections(self) -> str:
        """Renders the categories as labeled sections for an LLM prompt,
        instead of flattening everything into one run-on block of text."""
        sections: List[str] = []

        if self.title:
            sections.append(f"Title: {self.title}")
        if self.headings:
            sections.append("Headings:\n" + "\n".join(self.headings))
        if self.highlights:
            sections.append("Highlights:\n" + "\n".join(self.highlights))
        if self.lists:
            sections.append("Lists:\n" + "\n".join(self.lists))
        if self.styled_text:
            sections.append("Styled text:\n" + "\n".join(self.styled_text))
        if self.tables:
            sections.append("Tables:\n" + "\n\n".join(self.tables))
        if self.plain_text:
            sections.append("Notes:\n" + self.plain_text)

        return "\n\n".join(sections)

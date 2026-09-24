import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.core.auth import get_current_user_id
from app.core.rate_limit import RateLimiter
from app.services.learning_resources import StructuredNoteContent, get_learning_resources

log = logging.getLogger(__name__)
router = APIRouter()

# The frontend extractor condenses a note to ~1,500 characters before sending
# it; these bounds leave headroom while keeping each LLM prompt small.
MAX_TOTAL_CHARS = 4_000
_Items = Field(default_factory=list, max_length=50)

# Every request is an LLM call plus three web searches, so cap each user.
_per_minute = RateLimiter(max_calls=5, period=60)
_per_day = RateLimiter(max_calls=50, period=24 * 60 * 60)


# ── Learning Resources ────────────────────────────────────────────────────────

class LearningResourcesRequest(BaseModel):
    title: str = Field("", max_length=500)
    headings: list[str] = _Items
    highlights: list[str] = _Items
    lists: list[str] = _Items
    styled_text: list[str] = _Items
    tables: list[str] = _Items
    plain_text: str = Field("", max_length=MAX_TOTAL_CHARS)

    @model_validator(mode="after")
    def _cap_total_length(self):
        total = (
            len(self.title)
            + len(self.plain_text)
            + sum(len(s) for group in (self.headings, self.highlights, self.lists, self.styled_text, self.tables)
                  for s in group)
        )
        if total > MAX_TOTAL_CHARS:
            raise ValueError(f"Note content is too long ({total} > {MAX_TOTAL_CHARS} characters).")
        return self


@router.post("/learning-resources")
async def learning_resources(
    request: LearningResourcesRequest,
    user_id: str = Depends(get_current_user_id),
):
    _per_minute.check(user_id)
    _per_day.check(user_id)

    note_content = StructuredNoteContent(
        title=request.title,
        headings=request.headings,
        highlights=request.highlights,
        lists=request.lists,
        styled_text=request.styled_text,
        tables=request.tables,
        plain_text=request.plain_text,
    )
    try:
        return await get_learning_resources(note_content)
    except Exception:
        log.exception("learning-resources failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Couldn't find resources right now. Please try again in a moment.",
        ) from None

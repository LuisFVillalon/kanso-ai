from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.learning_resources import StructuredNoteContent, get_learning_resources

load_dotenv()
router = APIRouter()


# ── Learning Resources ────────────────────────────────────────────────────────

class LearningResourcesRequest(BaseModel):
    title: str = ""
    headings: list[str] = []
    highlights: list[str] = []
    lists: list[str] = []
    styled_text: list[str] = []
    tables: list[str] = []
    plain_text: str = ""


@router.post("/learning-resources")
async def learning_resources(request: LearningResourcesRequest):
    try:
        note_content = StructuredNoteContent(
            title=request.title,
            headings=request.headings,
            highlights=request.highlights,
            lists=request.lists,
            styled_text=request.styled_text,
            tables=request.tables,
            plain_text=request.plain_text,
        )
        result = await get_learning_resources(note_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource generation failed: {str(e)}")
    return result

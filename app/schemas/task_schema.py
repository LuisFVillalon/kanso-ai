from decimal import Decimal
from datetime import date, time, datetime
from pydantic import BaseModel, field_validator
from typing import List, Optional
from app.schemas.tag_schema import Tag

VALID_SESSION_TYPES = {"bite_size", "deep_work"}


class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    completed: bool = False
    priority: Optional[int] = None

    due_date: Optional[date] = None
    due_time: Optional[time] = None
    session_type: Optional[str] = None

    completed_date: Optional[datetime] = None

    estimated_time: Optional[float] = None
    parent_task_id: Optional[int] = None
    user_id: Optional[str] = None

    tags: List[Tag] = []

    @field_validator("due_date", "due_time", "session_type", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v

    @field_validator("session_type", mode="after")
    @classmethod
    def validate_session_type(cls, v):
        if v is not None and v not in VALID_SESSION_TYPES:
            raise ValueError(f"session_type must be one of {VALID_SESSION_TYPES}")
        return v

    @field_validator("estimated_time", mode="before")
    @classmethod
    def validate_estimated_time(cls, v):
        if v is None:
            return v
        if isinstance(v, Decimal):
            v = float(v)
        if v < 0:
            raise ValueError("Estimated time must be a non-negative number (hours)")
        return v

class TaskCreate(TaskBase):
    pass


class Task(TaskBase):
    id: int
    created_date: datetime


    model_config = {
        "from_attributes": True
    }
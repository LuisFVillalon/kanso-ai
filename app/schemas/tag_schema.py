from pydantic import BaseModel


class TagBase(BaseModel):
    name: str
    color: str | None = None


class TagCreate(TagBase):
    pass


class Tag(TagBase):
    id: int

    model_config = {"from_attributes": True}

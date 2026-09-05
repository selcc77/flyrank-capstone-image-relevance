from enum import Enum

from pydantic import BaseModel, Field


class ImageCategory(str, Enum):
    ANIMAL = "animal"
    NATURE = "nature"
    FOOD = "food"
    PEOPLE = "people"
    TECHNOLOGY = "technology"


class ImageMetadata(BaseModel):
    subject: str = Field(min_length=1)
    category: ImageCategory
    attributes: list[str]
    caption: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
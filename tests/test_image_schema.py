import pytest
from pydantic import ValidationError

from app.schemas.image import ImageCategory, ImageMetadata


def test_valid_image_metadata():
    metadata = ImageMetadata(
        subject="red fox",
        category=ImageCategory.ANIMAL,
        attributes=["orange fur", "forest"],
        caption="A red fox standing in a forest",
        confidence=0.98,
    )

    assert metadata.subject == "red fox"
    assert metadata.category == ImageCategory.ANIMAL
    assert metadata.confidence == 0.98


def test_confidence_cannot_be_above_one():
    with pytest.raises(ValidationError):
        ImageMetadata(
            subject="red fox",
            category=ImageCategory.ANIMAL,
            attributes=["orange fur"],
            caption="A red fox",
            confidence=1.5,
        )


def test_confidence_cannot_be_negative():
    with pytest.raises(ValidationError):
        ImageMetadata(
            subject="red fox",
            category=ImageCategory.ANIMAL,
            attributes=["orange fur"],
            caption="A red fox",
            confidence=-0.1,
        )


def test_invalid_category_is_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(
            subject="red fox",
            category="banana",
            attributes=["orange fur"],
            caption="A red fox",
            confidence=0.98,
        )


def test_empty_subject_is_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(
            subject="",
            category=ImageCategory.ANIMAL,
            attributes=["orange fur"],
            caption="A red fox",
            confidence=0.98,
        )
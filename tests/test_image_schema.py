import pytest
from pydantic import ValidationError

from app.schemas.image import ImageMetadata


def test_valid_image_metadata():
    metadata = ImageMetadata(
        subject="red fox",
        category="animal",
        attributes=["orange fur", "wild", "forest"],
        caption="A red fox standing in a forest",
        confidence=0.94,
    )

    assert metadata.subject == "red fox"
    assert metadata.confidence == 0.94


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        ImageMetadata(
            subject="red fox",
            category="animal",
            attributes=["orange fur"],
            caption="A red fox",
            confidence=1.5,
        )
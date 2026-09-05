from sqlalchemy.orm import Session

from app.models.image import Image
from app.schemas.image import ImageMetadata


def save_image_analysis(
    db: Session,
    filename: str,
    file_path: str,
    metadata: ImageMetadata,
) -> Image:
    image = Image(
        filename=filename,
        file_path=file_path,
        subject=metadata.subject,
        category=metadata.category.value,
        attributes=metadata.attributes,
        caption=metadata.caption,
        confidence=metadata.confidence,
    )

    db.add(image)
    db.commit()
    db.refresh(image)

    return image

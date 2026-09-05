from sqlalchemy import Float, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )

    subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    attributes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False
    )

    caption: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

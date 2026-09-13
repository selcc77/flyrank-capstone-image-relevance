from app.ai.gemini_client import GeminiClient
from app.core.database import SessionLocal
from app.models.image import Image
from app.core.matching import build_image_text


def main():
    db = SessionLocal()
    gemini = GeminiClient()

    try:
        images = (
            db.query(Image)
            .filter(Image.embedding.is_(None))
            .all()
        )

        print(f"Images needing embeddings: {len(images)}")

        for image in images:
            text = build_image_text(image)

            print(
                f"Generating embedding for image "
                f"{image.id}: {image.filename}"
            )

            image.embedding = gemini.create_embedding(text)

        db.commit()

        print("Image embeddings generated successfully.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
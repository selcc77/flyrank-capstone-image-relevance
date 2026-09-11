from app.ai.gemini_client import GeminiClient
from app.core.database import SessionLocal
from app.models.article import Article


def main():
    db = SessionLocal()
    gemini = GeminiClient()

    try:
        articles = (
            db.query(Article)
            .filter(Article.embedding.is_(None))
            .all()
        )

        print(f"Articles needing embeddings: {len(articles)}")

        for article in articles:
            text = f"{article.title} {article.content}"

            print(f"Generating embedding for article {article.id}: {article.title}")

            article.embedding = gemini.create_embedding(text)

        db.commit()

        print("Article embeddings generated successfully.")

    finally:
        db.close()


if __name__ == "__main__":
    main()

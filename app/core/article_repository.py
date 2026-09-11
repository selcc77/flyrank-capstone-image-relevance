from sqlalchemy.orm import Session

from app.ai.gemini_client import GeminiClient
from app.models.article import Article


def create_article(
    db: Session,
    title: str,
    content: str,
    gemini: GeminiClient,
) -> Article:
    article_text = f"{title} {content}"

    embedding = gemini.create_embedding(article_text)

    article = Article(
        title=title,
        content=content,
        embedding=embedding,
    )

    db.add(article)
    db.commit()
    db.refresh(article)

    return article


def get_all_articles(db: Session) -> list[Article]:
    return db.query(Article).all()
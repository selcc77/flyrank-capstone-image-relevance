from sqlalchemy.orm import Session

from app.models.article import Article


def create_article(
    db: Session,
    title: str,
    content: str,
) -> Article:
    article = Article(
        title=title,
        content=content,
    )

    db.add(article)
    db.commit()
    db.refresh(article)

    return article


def get_all_articles(db: Session) -> list[Article]:
    return db.query(Article).all()
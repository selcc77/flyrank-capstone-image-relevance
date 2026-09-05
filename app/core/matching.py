from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.article import Article
from app.models.image import Image


MATCH_THRESHOLD = 0.05


def build_image_text(image: Image) -> str:
    return " ".join(
        [
            image.subject,
            image.category,
            *image.attributes,
            image.caption,
        ]
    )


def build_article_text(article: Article) -> str:
    return f"{article.title} {article.content}"


def match_image_to_articles(
    image: Image,
    articles: list[Article],
) -> list[dict]:
    if not articles:
        return []

    image_text = build_image_text(image)
    article_texts = [build_article_text(article) for article in articles]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [image_text, *article_texts]
    )

    similarities = cosine_similarity(
        vectors[0:1],
        vectors[1:],
    )[0]

    results = []

    for article, score in zip(articles, similarities):
        score = round(float(score), 4)

        results.append(
            {
                "article_id": article.id,
                "title": article.title,
                "score": score,
                "is_relevant": score >= MATCH_THRESHOLD,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results
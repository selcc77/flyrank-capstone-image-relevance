from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.article import Article
from app.models.image import Image


TFIDF_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7
MATCH_THRESHOLD = 0.50


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

    article_texts = [
        build_article_text(article)
        for article in articles
    ]

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()

    tfidf_vectors = vectorizer.fit_transform(
        [image_text, *article_texts]
    )

    tfidf_scores = cosine_similarity(
        tfidf_vectors[0:1],
        tfidf_vectors[1:],
    )[0]

    # Use the image embedding already stored in the database.
    if image.embedding is None:
        return []

    article_embeddings = [
        article.embedding
        for article in articles
        if article.embedding is not None
    ]

    # Only compare against articles that have embeddings.
    articles_with_embeddings = [
        article
        for article in articles
        if article.embedding is not None
    ]

    if not articles_with_embeddings:
        return []

    article_texts = [
        build_article_text(article)
        for article in articles_with_embeddings
    ]

    vectorizer = TfidfVectorizer()

    tfidf_vectors = vectorizer.fit_transform(
        [image_text, *article_texts]
    )

    tfidf_scores = cosine_similarity(
        tfidf_vectors[0:1],
        tfidf_vectors[1:],
    )[0]

    semantic_scores = cosine_similarity(
        [image.embedding],
        article_embeddings,
    )[0]

    results = []

    for article, tfidf_score, semantic_score in zip(
        articles_with_embeddings,
        tfidf_scores,
        semantic_scores,
    ):
        final_score = (
            TFIDF_WEIGHT * float(tfidf_score)
            + SEMANTIC_WEIGHT * float(semantic_score)
        )

        if final_score < MATCH_THRESHOLD:
            continue

        results.append(
            {
                "article_id": article.id,
                "title": article.title,
                "tfidf_score": round(float(tfidf_score), 4),
                "semantic_score": round(float(semantic_score), 4),
                "score": round(final_score, 4),
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results
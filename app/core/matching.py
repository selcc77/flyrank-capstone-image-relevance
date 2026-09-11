from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ai.gemini_client import GeminiClient
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

    # Generate an embedding only for the image.
    # Article embeddings are already stored in the database.
    gemini = GeminiClient()
    image_embedding = gemini.create_embedding(image_text)

    article_embeddings = [
        article.embedding
        for article in articles
    ]

    semantic_scores = cosine_similarity(
        [image_embedding],
        article_embeddings,
    )[0]

    results = []

    for article, tfidf_score, semantic_score in zip(
        articles,
        tfidf_scores,
        semantic_scores,
    ):
        final_score = (
            TFIDF_WEIGHT * float(tfidf_score)
            + SEMANTIC_WEIGHT * float(semantic_score)
        )

        # Ignore articles that are not sufficiently relevant.
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
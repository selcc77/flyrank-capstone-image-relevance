from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.article import Article
from app.models.image import Image


TFIDF_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7

MATCH_THRESHOLD = 0.50
SEMANTIC_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.70


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


def is_subject_compatible(image: Image, article: Article) -> bool:
    """
    Check whether the main subject/category of the image
    is reasonably compatible with the article.

    This is a simple rule-based guard for now.
    """

    image_subject = image.subject.lower()
    image_category = image.category.lower()

    article_text = build_article_text(article).lower()

    subject_words = image_subject.split()

    subject_match = any(
        word in article_text
        for word in subject_words
        if len(word) > 2
    )

    category_keywords = {
        "animal": [
            "animal",
            "wildlife",
            "fox",
            "wolf",
            "dog",
            "cat",
            "bird",
            "bear",
        ],
        "nature": [
            "nature",
            "forest",
            "mountain",
            "landscape",
            "river",
            "snow",
        ],
        "food": [
            "food",
            "cook",
            "recipe",
            "pasta",
            "meal",
            "dish",
        ],
        "people": [
            "people",
            "person",
            "human",
            "portrait",
        ],
        "technology": [
            "technology",
            "software",
            "computer",
            "programming",
            "web",
            "application",
        ],
    }

    keywords = category_keywords.get(image_category, [])

    category_match = any(
        keyword in article_text
        for keyword in keywords
    )

    return subject_match or category_match


def match_image_to_articles(
    image: Image,
    articles: list[Article],
) -> list[dict]:
    if not articles:
        return []

    # Mismatch guard: reject images with low AI confidence.
    if image.confidence < CONFIDENCE_THRESHOLD:
        return []

    # An image must have an embedding before matching.
    if image.embedding is None:
        return []

    articles_with_embeddings = [
        article
        for article in articles
        if article.embedding is not None
    ]

    if not articles_with_embeddings:
        return []

    image_text = build_image_text(image)

    article_texts = [
        build_article_text(article)
        for article in articles_with_embeddings
    ]

    # Calculate TF-IDF similarity.
    vectorizer = TfidfVectorizer()

    tfidf_vectors = vectorizer.fit_transform(
        [image_text, *article_texts]
    )

    tfidf_scores = cosine_similarity(
        tfidf_vectors[0:1],
        tfidf_vectors[1:],
    )[0]

    # Calculate semantic similarity using stored embeddings.
    article_embeddings = [
        article.embedding
        for article in articles_with_embeddings
    ]

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
        tfidf_score = float(tfidf_score)
        semantic_score = float(semantic_score)

        final_score = (
            TFIDF_WEIGHT * tfidf_score
            + SEMANTIC_WEIGHT * semantic_score
        )

        # Mismatch guard:
        # 1. Overall similarity must be high enough.
        # 2. Semantic similarity must be high enough.
        # 3. Image subject/category must be compatible.
        if final_score < MATCH_THRESHOLD:
            continue

        if semantic_score < SEMANTIC_THRESHOLD:
            continue

        if not is_subject_compatible(image, article):
            continue

        results.append(
            {
                "article_id": article.id,
                "title": article.title,
                "tfidf_score": round(tfidf_score, 4),
                "semantic_score": round(semantic_score, 4),
                "score": round(final_score, 4),
                "decision": "accepted",
                "explanation": (
                    "Image confidence, semantic similarity, "
                    "and subject/category compatibility "
                    "all meet the matching requirements."
                ),
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results
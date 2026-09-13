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


def is_subject_compatible(
    image: Image,
    article: Article,
) -> bool:
    """
    Check whether the image subject/category is
    reasonably compatible with the article.
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

    keywords = category_keywords.get(
        image_category,
        [],
    )

    category_match = any(
        keyword in article_text
        for keyword in keywords
    )

    return subject_match or category_match


def calculate_similarity(
    image: Image,
    article: Article,
) -> dict:
    """
    Calculate TF-IDF, semantic, and final similarity
    between one image and one article.
    """

    image_text = build_image_text(image)
    article_text = build_article_text(article)

    vectorizer = TfidfVectorizer()

    tfidf_vectors = vectorizer.fit_transform(
        [image_text, article_text]
    )

    tfidf_score = float(
        cosine_similarity(
            tfidf_vectors[0:1],
            tfidf_vectors[1:2],
        )[0][0]
    )

    if image.embedding is None:
        semantic_score = 0.0
    elif article.embedding is None:
        semantic_score = 0.0
    else:
        semantic_score = float(
            cosine_similarity(
                [image.embedding],
                [article.embedding],
            )[0][0]
        )

    final_score = (
        TFIDF_WEIGHT * tfidf_score
        + SEMANTIC_WEIGHT * semantic_score
    )

    return {
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "score": final_score,
    }


def is_match_accepted(
    image: Image,
    article: Article,
    scores: dict,
) -> bool:
    """
    Apply the mismatch guard.
    """

    if image.confidence < CONFIDENCE_THRESHOLD:
        return False

    if scores["score"] < MATCH_THRESHOLD:
        return False

    if scores["semantic_score"] < SEMANTIC_THRESHOLD:
        return False

    if not is_subject_compatible(
        image,
        article,
    ):
        return False

    return True


def build_match_result(
    image: Image,
    article: Article,
) -> dict:
    """
    Build a complete match result including
    decision and human-readable explanation.
    """

    scores = calculate_similarity(
        image,
        article,
    )

    accepted = is_match_accepted(
        image,
        article,
        scores,
    )

    if accepted:
        decision = "accepted"
        explanation = (
            "Image confidence, semantic similarity, "
            "and subject/category compatibility "
            "all meet the matching requirements."
        )
    elif image.confidence < CONFIDENCE_THRESHOLD:
        decision = "rejected"
        explanation = (
            "The image analysis confidence is too low "
            "for a reliable recommendation."
        )
    elif scores["semantic_score"] < SEMANTIC_THRESHOLD:
        decision = "rejected"
        explanation = (
            "The semantic similarity between the image "
            "and article is too low."
        )
    elif scores["score"] < MATCH_THRESHOLD:
        decision = "rejected"
        explanation = (
            "The combined similarity score is below "
            "the matching threshold."
        )
    elif not is_subject_compatible(
        image,
        article,
    ):
        decision = "rejected"
        explanation = (
            "The image subject or category is not "
            "compatible with this article."
        )
    else:
        decision = "rejected"
        explanation = (
            "The image does not satisfy the "
            "matching requirements."
        )

    return {
        "article_id": article.id,
        "title": article.title,
        "tfidf_score": round(
            scores["tfidf_score"],
            4,
        ),
        "semantic_score": round(
            scores["semantic_score"],
            4,
        ),
        "score": round(
            scores["score"],
            4,
        ),
        "decision": decision,
        "explanation": explanation,
    }


def match_image_to_articles(
    image: Image,
    articles: list[Article],
) -> list[dict]:
    """
    Existing image -> article matching.
    """

    if not articles:
        return []

    if image.confidence < CONFIDENCE_THRESHOLD:
        return []

    if image.embedding is None:
        return []

    articles_with_embeddings = [
        article
        for article in articles
        if article.embedding is not None
    ]

    if not articles_with_embeddings:
        return []

    results = []

    for article in articles_with_embeddings:
        result = build_match_result(
            image,
            article,
        )

        if result["decision"] == "accepted":
            results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results


def match_article_to_images(
    article: Article,
    images: list[Image],
) -> list[dict]:
    """
    Match one article against all available images.

    This is the main post -> image matching flow
    required by the capstone.
    """

    if not images:
        return []

    if article.embedding is None:
        return []

    processed_images = [
        image
        for image in images
        if image.embedding is not None
    ]

    if not processed_images:
        return []

    results = []

    for image in processed_images:
        result = build_match_result(
            image,
            article,
        )

        if result["decision"] == "accepted":
            results.append(
                {
                    "image_id": image.id,
                    "filename": image.filename,
                    "subject": image.subject,
                    "category": image.category,
                    "tfidf_score": result["tfidf_score"],
                    "semantic_score": result["semantic_score"],
                    "score": result["score"],
                    "decision": result["decision"],
                    "explanation": result["explanation"],
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results
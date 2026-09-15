from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core import matching
from app.main import app


def test_build_image_text():
    image = SimpleNamespace(
        subject="red fox",
        category="animal",
        attributes=["red fur", "snow"],
        caption="A red fox standing in snow.",
    )

    result = matching.build_image_text(image)

    assert result == (
        "red fox animal red fur snow "
        "A red fox standing in snow."
    )


def test_build_article_text():
    article = SimpleNamespace(
        title="How Red Foxes Survive Winter",
        content="Red foxes live in snowy forests.",
    )

    result = matching.build_article_text(article)

    assert result == (
        "How Red Foxes Survive Winter "
        "Red foxes live in snowy forests."
    )


def test_matching_ranks_best_article_first():
    image = SimpleNamespace(
        id=1,
        subject="red fox",
        category="animal",
        attributes=["red fur", "snow"],
        caption="A red fox standing in snow.",
        confidence=0.98,
        embedding=[1.0, 0.0],
    )

    articles = [
        SimpleNamespace(
            id=1,
            title="How Red Foxes Survive Winter",
            content="Red foxes live in snowy forests.",
            embedding=[1.0, 0.0],
        ),
        SimpleNamespace(
            id=2,
            title="How to Cook Pasta",
            content="Pasta is prepared with tomato sauce.",
            embedding=[0.0, 1.0],
        ),
    ]

    results = matching.match_image_to_articles(
        image,
        articles,
    )

    assert len(results) == 1
    assert results[0]["article_id"] == 1
    assert results[0]["title"] == "How Red Foxes Survive Winter"
    assert results[0]["decision"] == "accepted"
    assert results[0]["score"] >= 0.50


def test_matches_image_not_found():
    client = TestClient(app)

    response = client.get("/images/999999/matches")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_suggestion_not_found():
    client = TestClient(app)

    response = client.get("/suggestions/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Suggestion not found"


def test_approve_suggestion():
    client = TestClient(app)

    response = client.post(
        "/suggestions/1/approve"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert data["review_status"] == "approved"
    assert data["message"] == "Suggestion approved"


def test_reject_suggestion():
    client = TestClient(app)

    response = client.post(
        "/suggestions/2/reject"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 2
    assert data["review_status"] == "rejected"
    assert data["message"] == "Suggestion rejected"
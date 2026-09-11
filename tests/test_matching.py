from types import SimpleNamespace

import app.core.matching as matching


def test_build_image_text():
    image = SimpleNamespace(
        subject="red fox",
        category="animal",
        attributes=["red fur", "snow"],
        caption="A red fox standing in snow.",
    )

    result = matching.build_image_text(image)

    assert "red fox" in result
    assert "animal" in result
    assert "snow" in result


def test_build_article_text():
    article = SimpleNamespace(
        title="How Red Foxes Survive Winter",
        content="Red foxes survive cold winters by growing thicker fur.",
    )

    result = matching.build_article_text(article)

    assert "How Red Foxes Survive Winter" in result
    assert "thicker fur" in result


def test_matching_ranks_best_article_first(monkeypatch):
    image = SimpleNamespace(
        id=1,
        subject="red fox",
        category="animal",
        attributes=["red fur", "snow"],
        caption="A red fox standing in snow.",
    )

    articles = [
        SimpleNamespace(
            id=1,
            title="How Red Foxes Survive Winter",
            content="Red foxes live in snowy forests.",
        ),
        SimpleNamespace(
            id=2,
            title="How to Cook Pasta",
            content="Pasta is prepared with tomato sauce.",
        ),
    ]

    class FakeGemini:
        def create_embedding(self, text):
            if "fox" in text.lower() or "snow" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]

    monkeypatch.setattr(
        matching,
        "GeminiClient",
        FakeGemini,
    )

    results = matching.match_image_to_articles(
        image,
        articles,
    )

    assert results[0]["article_id"] == 1
    assert results[0]["title"] == "How Red Foxes Survive Winter"
    assert results[0]["score"] > results[1]["score"]
from fastapi.testclient import TestClient

from app.main import app


def test_matches_image_not_found():
    client = TestClient(app)

    response = client.get("/images/999/matches")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}
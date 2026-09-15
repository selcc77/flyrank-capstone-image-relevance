from sqlalchemy.orm import Session

from app.models.suggestion import Suggestion


def create_suggestion(
    db: Session,
    post_id: int,
    image_id: int,
    score: float,
    decision: str,
    explanation: str,
) -> Suggestion:
    suggestion = Suggestion(
        post_id=post_id,
        image_id=image_id,
        score=score,
        decision=decision,
        explanation=explanation,
        review_status="pending",
    )

    db.add(suggestion)
    db.commit()
    db.refresh(suggestion)

    return suggestion


def get_suggestion(
    db: Session,
    suggestion_id: int,
) -> Suggestion | None:
    return (
        db.query(Suggestion)
        .filter(Suggestion.id == suggestion_id)
        .first()
    )


def get_suggestions_for_post(
    db: Session,
    post_id: int,
) -> list[Suggestion]:
    return (
        db.query(Suggestion)
        .filter(Suggestion.post_id == post_id)
        .order_by(Suggestion.score.desc())
        .all()
    )


def get_suggestion_for_post_and_image(
    db: Session,
    post_id: int,
    image_id: int,
) -> Suggestion | None:
    return (
        db.query(Suggestion)
        .filter(
            Suggestion.post_id == post_id,
            Suggestion.image_id == image_id,
        )
        .first()
    )


def update_review_status(
    db: Session,
    suggestion_id: int,
    review_status: str,
) -> Suggestion | None:
    suggestion = get_suggestion(db, suggestion_id)

    if suggestion is None:
        return None

    suggestion.review_status = review_status

    db.commit()
    db.refresh(suggestion)

    return suggestion
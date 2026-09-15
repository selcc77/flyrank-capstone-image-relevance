from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.ai.gemini_client import GeminiClient
from app.core.article_repository import create_article
from app.core.database import SessionLocal
from app.core.image_repository import save_image_analysis
from app.core.matching import (
    match_article_to_images,
    match_image_to_articles,
)
from app.core.suggestion_repository import (
    create_suggestion,
    get_suggestion,
    get_suggestion_for_post_and_image,
    update_review_status,
)
from app.models.article import Article
from app.models.image import Image
from app.schemas.article import ArticleCreate
from app.schemas.image import ImageMetadata


app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    version="0.1.0",
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "message": "AI Image Matching Engine is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/images/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        Path(file.filename or "image.jpg").suffix
        or ".jpg"
    )

    unique_filename = (
        f"{uuid4().hex}{suffix.lower()}"
    )

    file_path = upload_dir / unique_filename

    file_path.write_bytes(image_bytes)

    client = GeminiClient()

    result = client.analyze_image(
        str(file_path)
    )

    metadata = ImageMetadata.model_validate(result)

    saved_image = save_image_analysis(
        db=db,
        filename=file.filename or unique_filename,
        file_path=str(file_path),
        metadata=metadata,
    )

    return {
        "id": saved_image.id,
        "filename": saved_image.filename,
        "analysis": metadata.model_dump(),
    }


@app.post("/articles")
def create_article_endpoint(
    article: ArticleCreate,
    db: Session = Depends(get_db),
):
    gemini = GeminiClient()

    saved_article = create_article(
        db=db,
        title=article.title,
        content=article.content,
        gemini=gemini,
    )

    return {
        "id": saved_article.id,
        "title": saved_article.title,
        "content": saved_article.content,
    }


@app.post("/posts")
def create_post_endpoint(
    post: ArticleCreate,
    db: Session = Depends(get_db),
):
    gemini = GeminiClient()

    saved_post = create_article(
        db=db,
        title=post.title,
        content=post.content,
        gemini=gemini,
    )

    return {
        "id": saved_post.id,
        "title": saved_post.title,
        "content": saved_post.content,
    }


@app.get("/posts/{post_id}")
def get_post(
    post_id: int,
    db: Session = Depends(get_db),
):
    post = (
        db.query(Article)
        .filter(Article.id == post_id)
        .first()
    )

    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Post not found",
        )

    return {
        "id": post.id,
        "title": post.title,
        "content": post.content,
    }


@app.get("/posts/{post_id}/images")
def get_post_images(
    post_id: int,
    db: Session = Depends(get_db),
):
    post = (
        db.query(Article)
        .filter(Article.id == post_id)
        .first()
    )

    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Post not found",
        )

    images = db.query(Image).all()

    matches = match_article_to_images(
        article=post,
        images=images,
    )

    suggestions = []

    for match in matches:
        existing_suggestion = get_suggestion_for_post_and_image(
            db=db,
            post_id=post.id,
            image_id=match["image_id"],
        )

        if existing_suggestion is not None:
            suggestion = existing_suggestion
        else:
            suggestion = create_suggestion(
                db=db,
                post_id=post.id,
                image_id=match["image_id"],
                score=match["score"],
                decision=match["decision"],
                explanation=match["explanation"],
            )

        suggestions.append(
            {
                "suggestion_id": suggestion.id,
                "image_id": match["image_id"],
                "filename": match["filename"],
                "subject": match["subject"],
                "category": match["category"],
                "tfidf_score": match["tfidf_score"],
                "semantic_score": match["semantic_score"],
                "score": match["score"],
                "decision": match["decision"],
                "explanation": match["explanation"],
                "review_status": suggestion.review_status,
            }
        )

    return {
        "post_id": post.id,
        "post_title": post.title,
        "matches": suggestions,
    }


@app.get("/images/{image_id}/matches")
def get_image_matches(
    image_id: int,
    db: Session = Depends(get_db),
):
    image = (
        db.query(Image)
        .filter(Image.id == image_id)
        .first()
    )

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    articles = db.query(Article).all()

    matches = match_image_to_articles(
        image=image,
        articles=articles,
    )

    return {
        "image_id": image.id,
        "matches": matches,
    }


@app.get("/suggestions/{suggestion_id}")
def get_suggestion_endpoint(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    suggestion = get_suggestion(
        db=db,
        suggestion_id=suggestion_id,
    )

    if suggestion is None:
        raise HTTPException(
            status_code=404,
            detail="Suggestion not found",
        )

    return {
        "id": suggestion.id,
        "post_id": suggestion.post_id,
        "image_id": suggestion.image_id,
        "score": suggestion.score,
        "decision": suggestion.decision,
        "explanation": suggestion.explanation,
        "review_status": suggestion.review_status,
    }


@app.post("/suggestions/{suggestion_id}/approve")
def approve_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    suggestion = update_review_status(
        db=db,
        suggestion_id=suggestion_id,
        review_status="approved",
    )

    if suggestion is None:
        raise HTTPException(
            status_code=404,
            detail="Suggestion not found",
        )

    return {
        "id": suggestion.id,
        "post_id": suggestion.post_id,
        "image_id": suggestion.image_id,
        "score": suggestion.score,
        "decision": suggestion.decision,
        "explanation": suggestion.explanation,
        "review_status": suggestion.review_status,
        "message": "Suggestion approved",
    }


@app.post("/suggestions/{suggestion_id}/reject")
def reject_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    suggestion = update_review_status(
        db=db,
        suggestion_id=suggestion_id,
        review_status="rejected",
    )

    if suggestion is None:
        raise HTTPException(
            status_code=404,
            detail="Suggestion not found",
        )

    return {
        "id": suggestion.id,
        "post_id": suggestion.post_id,
        "image_id": suggestion.image_id,
        "score": suggestion.score,
        "decision": suggestion.decision,
        "explanation": suggestion.explanation,
        "review_status": suggestion.review_status,
        "message": "Suggestion rejected",
    }
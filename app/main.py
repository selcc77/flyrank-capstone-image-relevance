from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.ai.gemini_client import GeminiClient
from app.core.database import SessionLocal
from app.core.image_repository import save_image_analysis
from app.schemas.image import ImageMetadata
from app.core.article_repository import create_article
from app.schemas.article import ArticleCreate

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
    return {"message": "AI Image Matching Engine is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/images/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    unique_filename = f"{uuid4().hex}{suffix.lower()}"
    file_path = upload_dir / unique_filename

    file_path.write_bytes(image_bytes)

    client = GeminiClient()
    result = client.analyze_image(str(file_path))

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
    saved_article = create_article(
        db=db,
        title=article.title,
        content=article.content,
    )

    return {
        "id": saved_article.id,
        "title": saved_article.title,
        "content": saved_article.content,
    }
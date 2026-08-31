from fastapi import FastAPI

app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    version="0.1.0",
)


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
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.image import ImageMetadata


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def analyze_image(self, image_path: str) -> ImageMetadata:
        image = self.client.files.upload(file=image_path)

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                image,
                """
                Analyze this image.

                Identify the main subject and classify it.
                Describe important visual attributes.
                Write a concise caption.
                Provide your confidence from 0.0 to 1.0.

                Return only the requested structured data.
                """,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageMetadata,
            ),
        )

        return ImageMetadata.model_validate_json(response.text)
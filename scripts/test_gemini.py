from app.ai.gemini_client import GeminiClient


client = GeminiClient()

result = client.analyze_image("data/test/fox.jpg")

print(result.model_dump_json(indent=2))
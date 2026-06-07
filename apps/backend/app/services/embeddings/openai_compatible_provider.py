import os
import httpx
from app.services.embeddings.base import EmbeddingProvider
from app.core.config import settings

class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", settings.OPENAI_EMBEDDING_MODEL)
        self._dimension = 1536  # text-embedding-3-small default

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            # Return zero vectors as fallback
            return [[0.0] * self._dimension for _ in texts]
        
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
        except Exception:
            return [[0.0] * self._dimension for _ in texts]
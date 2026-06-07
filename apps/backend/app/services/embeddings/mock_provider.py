import hashlib
from app.services.embeddings.base import EmbeddingProvider
from app.core.config import settings

class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._dimension = settings.EMBEDDING_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Deterministic pseudo-random vectors based on text hash
        result = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vec = [(seed % 1000) / 1000.0 for _ in range(self._dimension)]
            # Normalize
            norm = sum(v*v for v in vec) ** 0.5
            vec = [v / norm for v in vec]
            result.append(vec)
        return result
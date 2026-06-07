import os
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.mock_provider import MockEmbeddingProvider
from app.services.embeddings.openai_compatible_provider import OpenAICompatibleEmbeddingProvider

def get_embedding_provider() -> EmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "mock").lower().strip()
    if provider_name == "openai":
        return OpenAICompatibleEmbeddingProvider()
    return MockEmbeddingProvider()
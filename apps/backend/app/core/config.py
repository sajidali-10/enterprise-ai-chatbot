import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Chatbot API"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://chatbot:changeme@postgres:5432/chatbot"
    )
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
    MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "chatbot-uploads")
    UPLOAD_MAX_SIZE_MB: int = int(os.getenv("UPLOAD_MAX_SIZE_MB", "10"))
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "3000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "mock")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Retrieval settings (Phase 5 - Hybrid Search & Reranking)
    RETRIEVAL_VECTOR_TOP_K: int = int(os.getenv("RETRIEVAL_VECTOR_TOP_K", "15"))
    RETRIEVAL_KEYWORD_TOP_K: int = int(os.getenv("RETRIEVAL_KEYWORD_TOP_K", "15"))
    RETRIEVAL_FINAL_TOP_K: int = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "5"))
    RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.1"))
    RETRIEVAL_RERANKER_TYPE: str = os.getenv("RETRIEVAL_RERANKER_TYPE", "noop")
    RETRIEVAL_RERANK_FINAL_K: int = int(os.getenv("RETRIEVAL_RERANK_FINAL_K", "20"))
    RETRIEVAL_VECTOR_WEIGHT: float = float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.7"))
    RETRIEVAL_KEYWORD_WEIGHT: float = float(os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.3"))
    RETRIEVAL_QUERY_REWRITER: str = os.getenv("RETRIEVAL_QUERY_REWRITER", "passthrough")
    # Admin/debug mode for showing retrieval details
    RETRIEVAL_SHOW_DEBUG: bool = os.getenv("RETRIEVAL_SHOW_DEBUG", "false").lower() in ("true", "1", "yes")

    # Security settings (Phase 6 - Authentication & Authorization)
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    DEV_AUTH_ENABLED: bool = os.getenv("DEV_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    DEV_USER_SECRET: str = os.getenv("DEV_USER_SECRET", "dev-secret-change-in-production")
    # SSO/OIDC placeholder settings (for future integration)
    OIDC_ENABLED: bool = os.getenv("OIDC_ENABLED", "false").lower() in ("true", "1", "yes")
    OIDC_ISSUER_URL: str = os.getenv("OIDC_ISSUER_URL", "")
    OIDC_CLIENT_ID: str = os.getenv("OIDC_CLIENT_ID", "")
    OIDC_CLIENT_SECRET: str = os.getenv("OIDC_CLIENT_SECRET", "")

settings = Settings()
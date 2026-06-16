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

    # Phase 10 - Answer Grounding & Hallucination Prevention
    RAG_MIN_RELEVANCE_SCORE: float = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.3"))

    # Security settings (Phase 6 - Authentication & Authorization)
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    DEV_AUTH_ENABLED: bool = os.getenv("DEV_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    DEV_USER_SECRET: str = os.getenv("DEV_USER_SECRET", "dev-secret-change-in-production")
    # SSO/OIDC placeholder settings (for future integration)
    OIDC_ENABLED: bool = os.getenv("OIDC_ENABLED", "false").lower() in ("true", "1", "yes")
    OIDC_ISSUER_URL: str = os.getenv("OIDC_ISSUER_URL", "")
    OIDC_CLIENT_ID: str = os.getenv("OIDC_CLIENT_ID", "")
    OIDC_CLIENT_SECRET: str = os.getenv("OIDC_CLIENT_SECRET", "")

    # Phase 12 - Production JWT Authentication & Authorization
    AUTH_MODE: str = os.getenv("AUTH_MODE", "dev")  # dev | local | oidc
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Phase 14 - CORS hardening
    CORS_ALLOWED_ORIGINS: str = os.getenv("CORS_ALLOWED_ORIGINS", "")

    # Phase 14 - Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    def get_cors_origins(self) -> list[str]:
        """Return the list of allowed CORS origins based on configuration."""
        if self.AUTH_MODE == "dev":
            # Dev mode: allow localhost + any configured origins
            origins = ["http://localhost:3000", "http://localhost:8000"]
            if self.CORS_ALLOWED_ORIGINS:
                origins.extend(
                    [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
                )
            return origins
        # Production/local mode: only configured origins (no wildcard)
        if self.CORS_ALLOWED_ORIGINS:
            return [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
        # Fallback: if no origins configured in production, use empty list
        # which effectively blocks cross-origin requests (safe default)
        return []

    # Bootstrap admin user (used only when AUTH_MODE=local and no admin exists)
    BOOTSTRAP_ADMIN_EMAIL: str = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "admin@example.com")
    BOOTSTRAP_ADMIN_USERNAME: str = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    BOOTSTRAP_ADMIN_PASSWORD: str = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")

settings = Settings()
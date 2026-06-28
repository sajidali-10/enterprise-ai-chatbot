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

    # Phase 21 - RAG Configuration Layer
    RAG_PIPELINE_PROVIDER: str = os.getenv("RAG_PIPELINE_PROVIDER", "custom")
    DOCUMENT_LOADER_PROVIDER: str = os.getenv("DOCUMENT_LOADER_PROVIDER", "custom")
    TEXT_SPLITTER_PROVIDER: str = os.getenv("TEXT_SPLITTER_PROVIDER", "custom")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    # Phase 23 — Embedding provider upgrade foundation
    EMBEDDING_NORMALIZE: bool = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in ("true", "1", "yes")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu | cuda | mps
    VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "qdrant")
    RETRIEVER_PROVIDER: str = os.getenv("RETRIEVER_PROVIDER", "custom")
    RERANKER_PROVIDER: str = os.getenv("RERANKER_PROVIDER", "none")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    RAG_CONTEXT_MAX_CHUNKS: int = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "6"))
    RAG_CONTEXT_MAX_CHARACTERS: int = int(os.getenv("RAG_CONTEXT_MAX_CHARACTERS", "12000"))
    RAG_CONVERSATION_CONTEXT_ENABLED: bool = os.getenv("RAG_CONVERSATION_CONTEXT_ENABLED", "true").lower() in ("true", "1", "yes")
    RAG_CONVERSATION_CONTEXT_MAX_MESSAGES: int = int(os.getenv("RAG_CONVERSATION_CONTEXT_MAX_MESSAGES", "6"))
    RAG_CONVERSATION_CONTEXT_MAX_CHARACTERS: int = int(os.getenv("RAG_CONVERSATION_CONTEXT_MAX_CHARACTERS", "3500"))

    # Phase 25 — Retriever & Reranker Upgrade Foundation
    # Retrieval strategy: vector (default) or hybrid (keyword + vector)
    RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "vector")  # vector | hybrid
    # Candidate K: number of results to fetch before final top_k truncation / reranking
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "15"))
    # Reranker settings (disabled by default — future phases will enable)
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "6"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "none")  # none | cohere | bge | cross_encoder
    # Hybrid search: disabled by default (future phases will enable)
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() in ("true", "1", "yes")

    # Phase 30E — Retrieval Strategy Abstraction & MMR Diversity
    # Active retrieval strategy: similarity | hybrid | mmr | hybrid_mmr
    # Generic, document-agnostic selection. No domain-specific keywords are hardcoded.
    RAG_RETRIEVAL_STRATEGY: str = os.getenv("RAG_RETRIEVAL_STRATEGY", "hybrid_mmr")
    # MMR lambda: trade-off between relevance (1.0) and diversity (0.0)
    RAG_MMR_LAMBDA: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))
    # Generic scoring weights (used by hybrid strategy on top of vector similarity)
    RAG_KEYWORD_WEIGHT: float = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3"))
    RAG_VECTOR_WEIGHT: float = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
    # Maximum chunks to forward to LLM after final selection
    RAG_CONTEXT_MAX_CHUNKS: int = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "6"))
    # Minimum score below which a chunk is filtered out before selection
    RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.1"))

    # Phase 30E Hotfix v2 — Evidence-Aware Grounding Thresholds
    # These thresholds separate "ranking score" from "answerability score".
    # A question with strong retrieval score but weak keyword overlap is NOT
    # automatically answerable. Conversely, a relevant short query should
    # not be blocked just because the score is moderate.
    #
    # STRONG: answer confidently with sources
    #   - top_score >= RAG_STRONG_EVIDENCE_THRESHOLD
    #   - AND at least RAG_MIN_SUPPORTING_CHUNKS chunks meet supporting criteria
    #   - AND question signals (phrase/number/acronym or strong keyword overlap) are in chunk content
    RAG_STRONG_EVIDENCE_THRESHOLD: float = float(os.getenv("RAG_STRONG_EVIDENCE_THRESHOLD", "0.65"))
    # MEDIUM: answer cautiously with a caveat ("Based on the retrieved sources...")
    #   - top_score >= RAG_MEDIUM_EVIDENCE_THRESHOLD
    #   - AND at least one chunk contains at least one meaningful question signal
    RAG_MEDIUM_EVIDENCE_THRESHOLD: float = float(os.getenv("RAG_MEDIUM_EVIDENCE_THRESHOLD", "0.35"))
    # Minimum keyword overlap (question keywords appearing in chunk content)
    # below which the evidence is considered weak even if scores are decent.
    RAG_MIN_KEYWORD_OVERLAP: float = float(os.getenv("RAG_MIN_KEYWORD_OVERLAP", "0.30"))
    # Minimum number of chunks that must contain question signals for the
    # answer to be considered supported (prevents single weak chunk answers).
    RAG_MIN_SUPPORTING_CHUNKS: int = int(os.getenv("RAG_MIN_SUPPORTING_CHUNKS", "1"))
    # Minimum keyword score (0..1) for a chunk to be considered to "support"
    # the question. Generic single-word overlap below this does not count.
    RAG_MIN_KEYWORD_SCORE: float = float(os.getenv("RAG_MIN_KEYWORD_SCORE", "0.20"))
    # Optional: minimum vector similarity required for strong evidence even
    # when keyword score is high (defense against adversarial lexical overlap).
    RAG_MIN_VECTOR_SCORE_FOR_STRONG: float = float(os.getenv("RAG_MIN_VECTOR_SCORE_FOR_STRONG", "0.45"))

    # Phase 26 — RAGAS Evaluation Foundation
    # Evaluator LLM provider for RAGAS metrics (faithfulness, answer_relevancy, context_precision)
    RAGAS_ENABLED: bool = os.getenv("RAGAS_ENABLED", "false").lower() in ("true", "1", "yes")
    RAGAS_EVALUATOR_PROVIDER: str = os.getenv("RAGAS_EVALUATOR_PROVIDER", "litellm")  # litellm | openrouter | openai
    RAGAS_EVALUATOR_MODEL: str = os.getenv("RAGAS_EVALUATOR_MODEL", "openrouter-gpt-oss")
    RAGAS_REPORT_DIR: str = os.getenv("RAGAS_REPORT_DIR", "/app/evals/ragas")
    RAGAS_MAX_CASES: int = int(os.getenv("RAGAS_MAX_CASES", "20"))
    RAGAS_SAVE_RESULTS: bool = os.getenv("RAGAS_SAVE_RESULTS", "true").lower() in ("true", "1", "yes")

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

    # Phase 27 — LangSmith Observability Foundation
    # Disabled by default — no external calls when LANGSMITH_TRACING=false
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "hiplink-ai-assistant")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    # LANGSMITH_API_KEY is read directly from env at runtime, NOT stored in settings
    LANGSMITH_LOG_FULL_PROMPT: bool = os.getenv("LANGSMITH_LOG_FULL_PROMPT", "false").lower() in ("true", "1", "yes")
    LANGSMITH_LOG_DOCUMENT_TEXT: bool = os.getenv("LANGSMITH_LOG_DOCUMENT_TEXT", "false").lower() in ("true", "1", "yes")
    LANGSMITH_LOG_USER_INPUT: bool = os.getenv("LANGSMITH_LOG_USER_INPUT", "true").lower() in ("true", "1", "yes")
    LANGSMITH_LOG_RETRIEVED_CONTEXT: bool = os.getenv("LANGSMITH_LOG_RETRIEVED_CONTEXT", "false").lower() in ("true", "1", "yes")
    # Phase 31A — additional privacy / sampling controls
    LANGSMITH_LOG_LLM_OUTPUT: bool = os.getenv("LANGSMITH_LOG_LLM_OUTPUT", "false").lower() in ("true", "1", "yes")
    LANGSMITH_REDACT_METADATA: bool = os.getenv("LANGSMITH_REDACT_METADATA", "true").lower() in ("true", "1", "yes")
    LANGSMITH_MAX_CONTEXT_CHARS: int = int(os.getenv("LANGSMITH_MAX_CONTEXT_CHARS", "400"))
    LANGSMITH_SAMPLE_RATE: float = float(os.getenv("LANGSMITH_SAMPLE_RATE", "1.0"))

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
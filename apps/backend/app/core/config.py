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

    # -------------------------------------------------------------------
    # Phase 34A — Enterprise OCR & Image Ingestion
    # -------------------------------------------------------------------
    # Master switch for the OCR subsystem. When false, image uploads
    # are accepted and stored but no OCR is performed — and the upload
    # endpoint will return a clear 4xx error if a *direct* image is
    # uploaded with no usable native text. Document processing continues
    # to work for text-only documents when OCR is disabled.
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes", "on")
    # Active OCR provider. Mirrors the LLM/embedding provider pattern.
    OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "tesseract")
    # Default Tesseract language code(s). Comma-separated for multi-lang.
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    # 0..100. Below this threshold an OCR row is flagged low_confidence
    # but still indexed (with a warning) so knowledge is not lost.
    OCR_MIN_CONFIDENCE: int = int(os.getenv("OCR_MIN_CONFIDENCE", "60"))
    # PDF OCR fallback: enable / disable OCR for pages whose native
    # text is below the per-page minimum.
    OCR_PDF_FALLBACK: bool = os.getenv("OCR_PDF_FALLBACK", "true").lower() in ("true", "1", "yes", "on")
    OCR_PDF_PAGE_TEXT_MIN_CHARS: int = int(os.getenv("OCR_PDF_PAGE_TEXT_MIN_CHARS", "40"))
    # DOCX embedded-image OCR.
    OCR_DOCX_IMAGES: bool = os.getenv("OCR_DOCX_IMAGES", "true").lower() in ("true", "1", "yes", "on")
    OCR_DOCX_IMAGE_MAX_COUNT: int = int(os.getenv("OCR_DOCX_IMAGE_MAX_COUNT", "50"))
    # Per-image safety caps.
    OCR_IMAGE_MAX_SIZE_MB: int = int(os.getenv("OCR_IMAGE_MAX_SIZE_MB", "15"))
    OCR_RENDER_DPI: int = int(os.getenv("OCR_RENDER_DPI", "200"))
    OCR_TIMEOUT_S: float = float(os.getenv("OCR_TIMEOUT_S", "120"))
    OCR_UPSCALING_ENABLED: bool = os.getenv("OCR_UPSCALING_ENABLED", "true").lower() in ("true", "1", "yes", "on")
    # TESSDATA_PREFIX (optional override).
    TESSDATA_PREFIX: str = os.getenv("TESSDATA_PREFIX", "")

    # -------------------------------------------------------------------
    # Phase 34A.1 — RAG Retrieval & DB Migration Hardening
    # -------------------------------------------------------------------
    # Maximum number of candidate chunks returned by the initial hybrid
    # retrieval (before final thresholding). Acts as a ceiling on the
    # input to MMR + exact-match reranking.
    RAG_MAX_RETRIEVED_CANDIDATES: int = int(os.getenv("RAG_MAX_RETRIEVED_CANDIDATES", "30"))
    # Maximum number of final sources presented to the LLM after
    # relevance thresholding.
    RAG_MAX_FINAL_SOURCES: int = int(os.getenv("RAG_MAX_FINAL_SOURCES", "6"))
    # Multiplicative boost applied per exact error-code hit during
    # reranking. 0.30 = up to +30% per matched code.
    RAG_EXACT_MATCH_BOOST: float = float(os.getenv("RAG_EXACT_MATCH_BOOST", "0.30"))
    # Multiplicative boost applied per exact technical-term hit.
    RAG_EXACT_TERM_BOOST: float = float(os.getenv("RAG_EXACT_TERM_BOOST", "0.15"))
    # Master switch for image-aware query routing. When false the
    # pipeline falls back to the existing broad-KB behaviour.
    RAG_IMAGE_AWARE_ROUTING_ENABLED: bool = os.getenv(
        "RAG_IMAGE_AWARE_ROUTING_ENABLED", "true"
    ).lower() in ("true", "1", "yes")
    # Minimum fused-retrieval score required for a chunk to be
    # forwarded to the LLM. Chunks below this are dropped.
    RAG_MIN_RELEVANCE_FLOOR: float = float(os.getenv("RAG_MIN_RELEVANCE_FLOOR", "0.10"))
    # Production-safe schema policy. When false (the default), the
    # backend startup does NOT call Base.metadata.create_all() — Alembic
    # is the sole owner of production schema. Tests opt in via the
    # DB_AUTO_CREATE_SCHEMA=true env var or by using a sqlite test DB
    # in conftest.py (which calls create_all itself).
    DB_AUTO_CREATE_SCHEMA: bool = os.getenv("DB_AUTO_CREATE_SCHEMA", "false").lower() in (
        "true", "1", "yes"
    )

    # Phase 31B — LangGraph Agentic RAG Pilot (optional, off by default)
    # Master switch. When false, the agentic pipeline is fully inert: it is
    # not imported at request time and the classic knowledge_base path is
    # unaffected. Set RAG_AGENTIC_ENABLED=true to opt in.
    RAG_AGENTIC_ENABLED: bool = os.getenv("RAG_AGENTIC_ENABLED", "false").lower() in ("true", "1", "yes")
    # Framework selector. Only "langgraph" is supported today; kept as a
    # config field so future frameworks (e.g. a custom planner) can be
    # added without a code change.
    RAG_AGENTIC_FRAMEWORK: str = os.getenv("RAG_AGENTIC_FRAMEWORK", "langgraph")
    # Whether the chat endpoint should route mode=knowledge_base requests
    # through the agentic pipeline when RAG_AGENTIC_ENABLED=true. When
    # false, mode=knowledge_base keeps using the classic pipeline even
    # though the agentic pipeline is loaded. This lets operators turn
    # the agentic path on only for explicit mode=agentic_knowledge_base
    # requests — a safer rollout than flipping the default behavior.
    RAG_AGENTIC_DEFAULT: bool = os.getenv("RAG_AGENTIC_DEFAULT", "false").lower() in ("true", "1", "yes")
    # Maximum number of retrieve/rewrite cycles the agentic graph may run
    # before finalizing. 1 = one retrieval pass + one rewrite + retry max.
    RAG_AGENTIC_MAX_RETRIES: int = int(os.getenv("RAG_AGENTIC_MAX_RETRIES", "1"))
    # Whether the rewrite_query_if_needed node may trigger a rewrite when
    # evidence is weak. When false, the graph goes straight from
    # evaluate_evidence to generate_answer.
    RAG_AGENTIC_REWRITE_ENABLED: bool = os.getenv("RAG_AGENTIC_REWRITE_ENABLED", "true").lower() in ("true", "1", "yes")
    # When true, the verify_citations node enforces citations on the
    # generated answer; a missing citation triggers the
    # answer_lacks_citations fallback (same behavior as classic RAG).
    RAG_AGENTIC_REQUIRE_CITATIONS: bool = os.getenv("RAG_AGENTIC_REQUIRE_CITATIONS", "true").lower() in ("true", "1", "yes")
    # When true, any unhandled exception inside the agentic graph falls
    # back to the classic pipeline instead of failing the request. This
    # is the safety net that keeps the agentic pilot from ever breaking
    # the chat endpoint.
    RAG_AGENTIC_FALLBACK_TO_CLASSIC: bool = os.getenv("RAG_AGENTIC_FALLBACK_TO_CLASSIC", "true").lower() in ("true", "1", "yes")

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

    # -------------------------------------------------------------------
    # Phase 34B — Automatic Vision Intelligence
    # -------------------------------------------------------------------
    # Vision settings are INDEPENDENT from the text LLM settings
    # (LLM_PROVIDER / LLM_MODEL / OPENAI_*) so operators can configure
    # a separate Vision-capable model (e.g. gpt-4o-mini-vision) without
    # changing the chat model.
    #
    # Master switch. When false the Image Intelligence Router degrades
    # to OCR-only for every image — the existing Phase 34A pipeline
    # runs untouched and ZERO Vision API calls are made.
    VISION_ENABLED: bool = os.getenv("VISION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on"
    )
    # Active Vision provider. Built-in: mock | openai-compatible.
    # mock is the safe default — deterministic, no network, no key.
    # openai-compatible speaks the OpenAI Chat Completions Vision
    # protocol (works against OpenAI, OpenRouter, Azure-OpenAI, local
    # vLLM gateways, etc.).
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "mock")
    # Provider-specific base URL (openai-compatible).
    VISION_BASE_URL: str = os.getenv("VISION_BASE_URL", "")
    # Model name for the Vision provider (independent from LLM_MODEL).
    VISION_MODEL: str = os.getenv("VISION_MODEL", "")
    # API key read directly from env at runtime, NEVER stored in
    # Settings — same pattern as LANGSMITH_API_KEY.
    # VISION_TIMEOUT_SECONDS — per-call timeout for the Vision provider.
    VISION_TIMEOUT_SECONDS: float = float(os.getenv("VISION_TIMEOUT_SECONDS", "30"))
    # VISION_MAX_RETRIES — number of times the provider layer may
    # retry transient failures (rate limit, timeout). 0 = no retry.
    VISION_MAX_RETRIES: int = int(os.getenv("VISION_MAX_RETRIES", "1"))
    # Router threshold: OCR confidence (0..100) below which Vision is
    # considered as a fallback to enrich low-confidence OCR.
    VISION_OCR_CONFIDENCE_THRESHOLD: int = int(
        os.getenv("VISION_OCR_CONFIDENCE_THRESHOLD", "55")
    )
    # Router threshold: OCR text length below which Vision is
    # considered as a fallback (diagrams, screenshots with little
    # textual content).
    VISION_MIN_OCR_TEXT_LENGTH: int = int(
        os.getenv("VISION_MIN_OCR_TEXT_LENGTH", "25")
    )
    # Router master switch. When false, every image is OCR-only
    # regardless of intent / thresholds (useful for debugging the
    # router itself in isolation).
    VISION_ROUTER_ENABLED: bool = os.getenv(
        "VISION_ROUTER_ENABLED", "true"
    ).lower() in ("true", "1", "yes", "on")
    # Maximum number of bytes the provider may receive in a single
    # image call. Guards against oversized image uploads.
    VISION_MAX_IMAGE_BYTES: int = int(
        os.getenv("VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
    )
    # Cache schema version — bump to invalidate ALL persisted Vision
    # results across the deployment. Persisted rows with a different
    # schema_version are treated as cache-misses and re-analysed.
    VISION_CACHE_SCHEMA_VERSION: int = int(
        os.getenv("VISION_CACHE_SCHEMA_VERSION", "1")
    )

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
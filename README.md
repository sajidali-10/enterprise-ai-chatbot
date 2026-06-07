# Enterprise AI Chatbot

A production-ready foundation for an enterprise AI chatbot application, built with FastAPI, Next.js, PostgreSQL, Qdrant, Redis, and MinIO — orchestrated via Docker Compose.

## Overview

This project establishes the foundational infrastructure for an enterprise-grade AI chatbot. The current Phase 1 deliverable adds a basic chat interface and backend API with an LLM provider abstraction, built on the Phase 0 foundation.

### Technology Stack

| Component    | Technology          | Purpose                                      |
|--------------|---------------------|----------------------------------------------|
| Backend      | FastAPI + Python    | REST API server with async support           |
| Frontend     | Next.js + TypeScript| React-based web interface                    |
| Database     | PostgreSQL 16       | Primary relational data store                |
| Vector Store | Qdrant              | Embedding storage for RAG (deferred Phase 1) |
| Cache        | Redis 7             | Caching and session management               |
| Object Store | MinIO               | S3-compatible storage for documents          |
| Orchestration| Docker Compose      | Container orchestration and service discovery|

## Prerequisites

- **Docker** (version 20.10 or later)
- **Docker Compose** (version 2.0 or later) or **Docker Compose Plugin**
- **GNU Make** (for convenience commands)

Verify prerequisites:

```bash
docker --version
docker compose version
make --version
```

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd enterprise-ai-chatbot
```

### 2. Configure environment

```bash
cp .env.example .env
```

Update `.env` values if needed (especially production secrets).

### 3. Start services

```bash
make up
```

This command will:
- Build Docker images for backend and frontend
- Start all services in detached mode
- Create and mount required volumes

### 4. Access the application

| Service       | URL                           |
|---------------|-------------------------------|
| Frontend      | http://localhost:3000         |
| Backend API   | http://localhost:8000         |
| API Docs      | http://localhost:8000/docs    |
| MinIO Console | http://localhost:9001         |

## Project Structure

```
enterprise-ai-chatbot/
├── apps/
│   ├── frontend/                 # Next.js application
│   │   ├── app/                  # App router pages
│   │   ├── public/               # Static assets
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── next.config.js
│   │   ├── tailwind.config.ts
│   │   └── postcss.config.js
│   └── backend/                  # FastAPI application
│       ├── app/
│       │   ├── main.py           # Application entry point
│       │   ├── api/              # API route handlers
│       │   ├── core/             # Core configuration
│       │   ├── db/               # Database utilities
│       │   ├── services/         # Business logic services
│       │   ├── rag/              # RAG utilities (deferred Phase 3)
│       │   ├── ingestion/        # Document ingestion (Phase 2)
│       │   └── security/         # Auth & security (deferred Phase 1)
│       ├── tests/                # Pytest test suite
│       ├── Dockerfile
│       └── requirements.txt
├── docs/                         # Project documentation
├── infra/                        # Infrastructure as code (future)
├── docker-compose.yml            # Service orchestration
├── Makefile                      # Development commands
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Makefile Commands Reference

| Command              | Description                                          |
|----------------------|------------------------------------------------------|
| `make up`            | Start all services (builds if needed, detached mode) |
| `make down`          | Stop all services                                    |
| `make logs`          | Follow logs from all services                        |
| `make logs-backend`  | Follow logs from backend only                        |
| `make logs-frontend` | Follow logs from frontend only                       |
| `make backend-test`  | Run pytest tests in backend container                |
| `make build`         | Build images without starting containers             |
| `make check`         | Validate docker-compose.yml configuration            |
| `make clean`         | Remove all containers, volumes, and orphans          |
| `make restart-backend` | Restart backend service                           |
| `make restart-frontend` | Restart frontend service                        |

## Phase 0 Scope

Phase 0 establishes a minimal working foundation with:

- Docker Compose orchestration for all services
- FastAPI backend with `/health` endpoint
- Next.js frontend with health status display
- PostgreSQL, Qdrant, Redis, and MinIO service definitions
- Makefile for common development tasks

### What's NOT included in Phase 0

The following features are deferred to subsequent phases:

- **Chat UI or chat logic** — No conversation interface
- **RAG pipeline** — Embedding generation, vector indexing, semantic search
- **Document upload** — File ingestion and processing
- **Authentication** — Login, JWT, RBAC, user management
- **LLM integration** — LangChain, LangGraph, LiteLLM, or local LLM code
- **Third-party integrations** — Zendesk, Slack, or external services

## Service Health Checks

Each service exposes a health check endpoint or CLI command for monitoring:

| Service    | Health Check Method              | Container Port |
|------------|----------------------------------|----------------|
| postgres   | `pg_isready` command             | 5432           |
| qdrant     | HTTP `GET /health`              | 6333           |
| redis      | `redis-cli ping`                | 6379           |
| minio      | `mc ready local`                | 9000, 9001     |
| backend    | `GET /health`                   | 8000           |

## Development

### Running backend tests

```bash
make backend-test
```

### Viewing backend logs

```bash
make logs-backend
```

### Checking service status

```bash
docker compose ps
```

### Validating configuration

```bash
make check
```

## Chat Usage

Navigate to `http://localhost:3000/chat` to access the chat interface.

The chat sends messages to the backend via `POST /api/chat` with a JSON body `{"message": "..."}`. By default, the mock provider responds with canned responses.

### Switching to OpenAI-Compatible Provider

1. Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...` in your `.env` file
2. Restart the backend:
   ```bash
   make restart-backend
   ```

### Running Backend Chat Tests

```bash
make backend-test
```

## Document Upload

Navigate to `http://localhost:3000/documents/upload` to upload documents.

Supported file types:
- PDF (`.pdf`)
- Plain text (`.txt`)
- Markdown (`.md`)
- Word (`.docx`)

Maximum file size: 10 MB (configurable via `UPLOAD_MAX_SIZE_MB`)

### Upload Flow

1. Select or drag-and-drop a file
2. Frontend validates type and size
3. Backend stores the file in MinIO
4. Backend extracts text via parser pipeline
5. Metadata and extracted text are saved in PostgreSQL

### View Uploaded Documents

Navigate to `http://localhost:3000/documents` to see a list of uploaded documents.

### Running Document Tests

```bash
make backend-test
```

## Document Indexing

The index endpoint processes a document's extracted text into chunks and stores them in both PostgreSQL and Qdrant for semantic search.

### Trigger Indexing

```bash
curl -X POST http://localhost:8000/api/documents/{document_id}/index
```

### How It Works

1. Fetch the document's latest version with extracted text
2. Split text into ~3000-char chunks with ~400-char overlap
3. Store chunks in PostgreSQL (`document_chunks` table) with metadata
4. Generate embeddings via the configured provider (default: mock)
5. Upsert vectors to Qdrant collection with chunk metadata payload

### Embedding Providers

- **Mock** (default): Deterministic pseudo-random vectors, no API key needed
- **OpenAI-compatible**: Set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | qdrant | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION` | documents | Collection name |
| `CHUNK_SIZE` | 3000 | Max characters per chunk |
| `CHUNK_OVERLAP` | 400 | Overlap characters between chunks |
| `EMBEDDING_PROVIDER` | mock | mock or openai |
| `EMBEDDING_DIMENSION` | 384 | Vector dimension (mock provider) |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |

## RAG Chat (Phase 4 & Phase 5)

The chat endpoint supports two modes:

### Normal Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### RAG Chat (with document retrieval)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is in the knowledge base?", "mode": "rag"}'
```

When `mode: "rag"` is specified, the system:
1. **Phase 5**: Optionally rewrites the query for better retrieval
2. Performs hybrid retrieval (vector + keyword search) by default
3. Optionally reranks results using configurable reranker
4. Builds a prompt with the retrieved context
5. Generates an answer with inline citations [1], [2], etc.
6. Returns citations below the answer

### Hybrid Retrieval (Phase 5)

Phase 5 introduces hybrid retrieval combining:
- **Vector search (Qdrant)**: Semantic similarity using embeddings
- **Keyword search (PostgreSQL)**: Full-text search with `ts_rank` scoring

Results are fused using weighted score normalization:
```
fused_score = (vector_weight × normalized_vector_score) + (keyword_weight × normalized_keyword_score)
```

Default weights: 70% vector, 30% keyword (configurable).

### Reranking (Phase 5)

After fusion, results can be reranked using:
- **noop** (default): No reranking, use fused scores
- **mock**: Simple heuristic reranking based on text overlap and title matches

Future: Cohere Rerank and BGE reranker integrations.

### Query Rewriting (Phase 5)

Optional query rewriting before retrieval:
- **passthrough** (default): No rewriting
- **mock**: Basic query expansion (adds trailing `?`)

### Debug Mode

Enable debug mode to see retrieval details:
```bash
curl -X POST "http://localhost:8000/api/chat?mode=rag&debug=true" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the return policy?"}'
```

Response includes `debug_info` with:
- `original_query` / `rewritten_query`
- `vector_results_count` / `keyword_results_count`
- `config` settings used
- Per-chunk scores and rankings

### Retrieval Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_VECTOR_TOP_K` | 15 | Vector search results to fetch |
| `RETRIEVAL_KEYWORD_TOP_K` | 15 | Keyword search results to fetch |
| `RETRIEVAL_FINAL_TOP_K` | 5 | Final results after fusion/reranking |
| `RETRIEVAL_MIN_SCORE` | 0.1 | Minimum score threshold |
| `RETRIEVAL_RERANKER_TYPE` | noop | Reranker: noop, mock, cohere, bge |
| `RETRIEVAL_RERANK_FINAL_K` | 20 | Number of results to rerank |
| `RETRIEVAL_VECTOR_WEIGHT` | 0.7 | Weight for vector scores in fusion |
| `RETRIEVAL_KEYWORD_WEIGHT` | 0.3 | Weight for keyword scores in fusion |
| `RETRIEVAL_QUERY_REWRITER` | passthrough | Rewriter: passthrough, mock |
| `RETRIEVAL_SHOW_DEBUG` | false | Always show debug info in responses |

### Citation Format
Citations are returned in the response:
```json
{
  "message": "According to [1]...",
  "citations": [
    {
      "index": 1,
      "source_file_name": "policy.pdf",
      "content_snippet": "The capital of France is Paris...",
      "relevance_score": 0.92
    }
  ]
}
```

If no relevant chunks are found (score below threshold), the system returns:
"I could not find enough information in the approved knowledge base to answer confidently."

## RAG Evaluation (Phase 5)

Run retrieval evaluation to measure quality:

```bash
python scripts/run_rag_eval.py
```

### Evaluation Dataset Format

Questions are in JSONL format at `evals/questions.jsonl`:
```json
{"id": "q1", "question": "What is the return policy?", "category": "policy", "expected_keywords": ["return", "refund", "days"]}
```

### Evaluation Output

The script outputs:
- Average chunks retrieved per query
- Top score and average top-5 scores
- Keyword coverage (% of expected keywords found)
- Category breakdown with per-category metrics

### Running Custom Evaluation

```bash
# Use custom questions file
python scripts/run_rag_eval.py --questions path/to/my_questions.jsonl

# Write results to JSON
python scripts/run_rag_eval.py --output results.json
```

## Phase 6: Authentication, RBAC, and Audit Logging

Phase 6 adds authentication, role-based access control (RBAC), document-level permissions, and audit logging for compliance and security monitoring.

### Architecture Overview

The security layer is designed to support local development (dev headers) and future enterprise SSO/OIDC integration without changing application code.

```
Request → CompositeAuthProvider
           ├── DevAuthProvider (X-Dev-User header, local dev only)
           ├── DatabaseAuthProvider (placeholder for SSO/JWT)
           └── AnonymousAuthProvider (fallback)
         ↓
    AuthContext (user_id, username, role)
         ↓
PermissionChecker (document-level filtering)
         ↓
    RAG Retrieval (filtered results only)
         ↓
AuditLogger (records all interactions)
```

### User Roles

| Role    | Access Level                                      |
|---------|---------------------------------------------------|
| ADMIN   | Full access to all documents (bypasses filtering) |
| USER    | Access only to explicitly permitted documents     |
| VIEWER  | Read-only access to explicitly permitted documents|

### Dev Authentication (Local Development)

Use the `X-Dev-User` header to authenticate without a database record:

```bash
# Regular user
curl -H "X-Dev-User: myname" http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the policy?"}'

# Admin user (bypasses all permission checks)
curl -H "X-Dev-User: admin_superuser" http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the policy?"}'

# Viewer (read-only)
curl -H "X-Dev-User: viewer_guest" http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the policy?"}'
```

Username prefix determines role:
- `admin_*` → ADMIN role
- `viewer_*` → VIEWER role
- anything else → USER role

**Important:** Dev users have `user_id=None` and cannot access any documents (they need explicit permissions in the database). Only admin dev users bypass this restriction.

### Document-Level Permissions

Permissions are stored in the `document_permissions` table:

| Field      | Type   | Description                          |
|------------|--------|--------------------------------------|
| user_id    | int    | Reference to users table             |
| document_id| int    | Reference to documents table         |
| can_read   | bool   | Allow reading this document          |
| can_write  | bool   | Allow modifying this document        |
| granted_by | int    | User ID who granted the permission   |

**Permission flow:**
1. User makes request with auth context
2. RAG retrieves chunks from all documents
3. `PermissionChecker.filter_accessible_documents()` removes unauthorized documents **before** answer generation
4. Only authorized chunks are sent to the LLM

Admin users bypass permission filtering entirely.

### Audit Logging

All RAG interactions are logged to the `audit_logs` table:

| Field                 | Type     | Description                              |
|-----------------------|----------|------------------------------------------|
| user_id               | int?     | NULL for anonymous/dev users             |
| username              | string   | Always populated                         |
| action                | enum     | CHAT_RAG, CHAT_MESSAGE, etc.             |
| question              | text     | User's question                          |
| retrieved_document_ids| text     | JSON array of accessed document IDs      |
| retrieved_chunk_ids   | text     | JSON array of accessed chunk IDs         |
| model_provider        | string   | e.g., "openai", "mock"                   |
| model_name            | string   | e.g., "gpt-4o-mini"                       |
| status                | string   | success, failure, error                  |
| duration_ms           | int      | Request duration in milliseconds         |
| request_ip            | string   | Client IP address                        |
| request_user_agent    | string   | Client user agent                        |
| created_at            | datetime | Indexed for time-series queries          |

### Database Migration

Run the security tables migration:

```bash
docker compose exec backend alembic upgrade head
```

This creates:
- `users` — user accounts with roles
- `document_permissions` — document-level access control
- `audit_logs` — compliance audit trail

### Future SSO/OIDC Integration

The `AuthProviderBase` interface allows dropping in OIDC/JWT validation without changing application code:

```python
class AuthProviderBase:
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        raise NotImplementedError
```

To add OIDC:
1. Implement `AuthProviderBase` with JWT validation
2. Add it to `CompositeAuthProvider` in `get_auth_provider()`
3. Remove or lower priority of `DevAuthProvider`

## License

Proprietary — All rights reserved.
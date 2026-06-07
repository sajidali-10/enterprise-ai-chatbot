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

## License

Proprietary — All rights reserved.
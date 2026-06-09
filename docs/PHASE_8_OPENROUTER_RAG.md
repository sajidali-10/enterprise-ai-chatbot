# Phase 8: Real LLM Usage and RAG Quality

This document describes how to enable real LLM providers and improve RAG answer quality.

## Enabling OpenRouter

OpenRouter provides unified access to 100+ LLMs from various providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.) through an OpenAI-compatible API.

### 1. Get an API Key

1. Sign up at [https://openrouter.ai](https://openrouter.ai)
2. Navigate to [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. Generate a new API key

### 2. Configure Environment

Add to your `.env` file:

```bash
# Enable OpenRouter as the LLM provider
LLM_PROVIDER=openrouter

# Your API key from OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx

# Model selection (default: google/gemini-2.0-flash-exp)
OPENROUTER_MODEL=google/gemini-2.0-flash-exp

# Optional: Your site for ranking improvements
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=Enterprise AI Chatbot
```

### 3. Recommended Low-Cost Models

| Model | Provider | Cost | Best For |
|-------|----------|------|----------|
| `google/gemini-2.0-flash-exp` | Google | ~$0.05/1M tokens | Fast, cheap, latest |
| `anthropic/claude-3-haiku` | Anthropic | ~$0.25/1M tokens | Good quality, fast |
| `meta-llama/llama-3-8b-instruct` | Meta | ~$0.20/1M tokens | Open source |
| `mistralai/mistral-7b-instruct` | Mistral | ~$0.25/1M tokens | Balanced |

For the latest pricing, see [OpenRouter Models](https://openrouter.ai/models).

### 4. Rebuild and Restart

```bash
docker compose up --build -d backend
```

### 5. Verify Configuration

Check the logs to confirm the provider is initialized:

```bash
docker compose logs backend | grep -i openrouter
```

Or test via the API:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "mode": "normal"}'
```

## Testing RAG with Real Model

### Prerequisites

1. Documents must be indexed in the vector database. If empty, upload some documents first via `/api/documents`.

### Test RAG Mode

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What topics are covered in the knowledge base?", "mode": "rag"}'
```

Expected response structure:

```json
{
  "message": "Based on the retrieved documents...",
  "role": "assistant",
  "citations": [
    {
      "index": 1,
      "source_file_name": "document.pdf",
      "content_snippet": "First 200 characters...",
      "relevance_score": 0.95
    }
  ]
}
```

### Debug Mode

For debugging retrieval, add `debug=true`:

```bash
curl "http://localhost:8000/api/chat?debug=true" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "Your question", "mode": "rag"}'
```

This returns additional metadata about retrieval scores and sources.

## Safe Fallback Behavior

The system handles errors gracefully:

| Scenario | Behavior |
|----------|----------|
| `OPENROUTER_API_KEY` not set | Returns clear error message with instructions |
| Invalid/expired API key | Returns error with link to get new key |
| Credit limit exceeded | Returns error with link to add credits |
| Rate limit exceeded | Returns error suggesting retry |
| Network failure | Returns error with details |

**Important**: Errors are NOT silently hidden. All error conditions return informative messages.

## RAG Prompt Quality

The RAG prompt template now includes:

1. **Context-Only Answers**: Model instructed to answer ONLY from retrieved context
2. **Citation Requirements**: Model must cite sources using [1], [2], etc.
3. **Honesty Guidelines**: If context is insufficient, model says so clearly
4. **No Fabrication**: Model explicitly told not to invent information
5. **Support Style**: Concise, helpful responses for support scenarios

## Running Tests

```bash
# Run all tests
docker compose exec backend python -m pytest tests/ -v

# Run only RAG integration tests
docker compose exec backend python -m pytest tests/test_rag_llm_integration.py -v

# Run with coverage
docker compose exec backend python -m pytest tests/ -v --cov=app --cov-report=html
```

## Architecture Notes

- **No LangGraph**: This phase does not add LangGraph or LangChain
- **No JWT Auth**: Authentication remains development-mode only
- **No Observability**: No telemetry added in this phase
- **Minimal Changes**: Architecture unchanged; only provider implementation and prompts improved
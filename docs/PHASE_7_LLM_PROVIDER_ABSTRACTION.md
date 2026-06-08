# Phase 7: LLM Provider Abstraction

**Date:** 2026-06-08  
**Status:** ✅ COMPLETE

---

## Goal

Replace direct Mock LLM dependency with a provider abstraction that supports multiple LLM backends through configuration.

---

## Architecture

### Provider Interface

```python
class LlmProvider(ABC):
    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        pass
```

### Provider Registry

| Provider | Env Var Value | Description | API Key Required |
|----------|--------------|-------------|------------------|
| Mock | `mock` | Returns fake responses | No |
| OpenAI | `openai` | Direct OpenAI API | Yes |
| OpenRouter | `openrouter` | Unified API to 100+ LLMs | Yes |
| Ollama | `ollama` | Local LLM inference | No (local) |
| LiteLLM | `litellm` | Unified interface to 100+ LLMs | Yes* |

\* Except for local providers like Ollama

### Selection Logic

```python
def get_llm_provider() -> LlmProvider:
    provider_name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    # Falls back to mock for unknown providers
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | Provider selection |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-exp` | OpenRouter model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model |
| `LITELLM_MODEL` | `gpt-4o-mini` | LiteLLM model |
| `LITELLM_API_KEY` | - | LiteLLM API key |

### Provider-Specific Configuration

#### OpenRouter
```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-2.0-flash-exp
OPENROUTER_SITE_URL=https://yoursite.com
OPENROUTER_SITE_NAME=YourSite
```

#### Ollama (Local)
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_KEEP_ALIVE=5m
```

#### LiteLLM
```bash
LLM_PROVIDER=litellm
LITELLM_MODEL=claude-3-haiku
LITELLM_API_KEY=sk-ant-...
```

---

## Files Changed

| File | Change |
|------|--------|
| `app/services/llm/__init__.py` | Provider registry, lazy imports |
| `app/services/llm/litellm_provider.py` | NEW — LiteLLM integration |
| `app/services/llm/openrouter_provider.py` | NEW — OpenRouter integration |
| `app/services/llm/ollama_provider.py` | NEW — Ollama integration |
| `tests/test_llm_provider.py` | NEW — 15 provider tests |

---

## Migration Guide

### From Mock to OpenAI

1. Set environment variables:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

2. Restart backend

3. Test:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Dev-User: admin_test" \
  -d '{"message": "Hello", "mode": "normal"}'
```

### From Mock to Ollama (Local)

1. Install Ollama: https://ollama.ai/

2. Pull a model:
```bash
ollama pull llama3.2
```

3. Start Ollama server (usually automatic on install)

4. Set environment variables:
```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
```

5. Restart backend

### From Mock to OpenRouter

1. Get API key: https://openrouter.ai/keys

2. Set environment variables:
```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

3. Optionally choose a model:
```bash
export OPENROUTER_MODEL=google/gemini-2.0-flash-exp
```

4. Restart backend

---

## Provider Comparison

| Provider | Cost | Latency | Privacy | Models |
|----------|------|---------|---------|--------|
| Mock | Free | Instant | 100% | N/A |
| OpenAI | Pay | Fast | No | GPT-4o, GPT-4o-mini, etc. |
| OpenRouter | Pay | Varies | No | 100+ models |
| Ollama | Free | Depends on hardware | 100% | Llama, Mistral, etc. |
| LiteLLM | Pay | Varies | No | 100+ models |

---

## Error Handling

All providers implement graceful fallback:

1. **No API key**: Returns mock response with explanation
2. **Network error**: Returns mock response with error message
3. **Invalid key**: Returns mock response with auth error
4. **Rate limit**: Returns mock response with retry hint

Example fallback response:
```
[OpenRouter Fallback] OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys. Using mock response instead.
```

---

## Phase 7 Status

| Component | Status |
|-----------|--------|
| Mock Provider | ✅ Pass (existing) |
| OpenAI Provider | ✅ Pass (existing) |
| OpenRouter Provider | ✅ Pass (new) |
| Ollama Provider | ✅ Pass (new) |
| LiteLLM Provider | ✅ Pass (new) |
| Provider Selection | ✅ Pass |
| Fallback Behavior | ✅ Pass |
| Tests | ✅ 15/15 pass |

---

## Next Phase (Recommended)

### Phase 7.1: RAG Answer Quality

Improve the mock provider's RAG answer quality by:
1. Better citation formatting
2. More contextual responses
3. Source attribution that matches retrieved chunks

### Phase 8: Real LLM Integration

When ready for production:
1. Connect real OpenAI or OpenRouter
2. Tune prompt templates for chosen model
3. Add rate limiting and cost tracking
4. Add model selection to frontend
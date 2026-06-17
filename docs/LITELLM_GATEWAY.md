# LiteLLM Gateway — Phase 15

> **What is this?**  
> This document describes how to deploy and use the LiteLLM Gateway as an optional proxy layer for LLM calls in this project.

---

## Overview

### Direct OpenRouter Mode (Default)

When `LLM_PROVIDER=openrouter`, the backend calls OpenRouter directly via its REST API:

```
Backend  →  OpenRouter API  →  OpenRouter  →  Upstream LLM
```

- **No extra service required** — just the backend and an OpenRouter API key.
- Configuration: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`.
- Simpler stack, same functionality.

### LiteLLM Gateway Mode

When `LLM_PROVIDER=litellm`, the backend calls a self-hosted LiteLLM Gateway, which routes to the same (or other) LLMs:

```
Backend  →  LiteLLM Gateway  →  OpenRouter (or any other provider)  →  Upstream LLM
```

- **Requires LiteLLM Gateway service** — runs in Docker.
- Centralized routing, failover, and cost controls.
- Configuration: `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY`, `LITELLM_MODEL`.
- Model names are defined in `infra/litellm/config.yaml`.

### When to Use Each Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| Development / single upstream provider | Direct OpenRouter (`LLM_PROVIDER=openrouter`) |
| Need failover, routing, or multi-provider | LiteLLM Gateway (`LLM_PROVIDER=litellm`) |
| Want to avoid per-service API key management | LiteLLM Gateway |
| Already have LiteLLM in your stack | LiteLLM Gateway |
| Minimal setup, fastest initial deployment | Direct OpenRouter |

---

## Quick Start

### 1. Copy Environment Variables

```bash
cp .env.example .env
```

### 2. Enable LiteLLM Gateway (Optional)

In `.env`, set:

```bash
# Enable LiteLLM Gateway mode in the backend
LLM_PROVIDER=litellm

# Enable the LiteLLM Gateway Docker service
LITELLM_ENABLED=true

# Set a strong master key (change from default!)
LITELLM_MASTER_KEY=your-strong-random-key-here

# Set your OpenRouter API key (used by the gateway to call upstream models)
OPENROUTER_API_KEY=sk_or_...

# Default model — must match a model defined in config.yaml
LITELLM_MODEL=openrouter-gpt-oss
```

> **Security:** Change `LITELLM_MASTER_KEY` to a strong random value before deploying. Never commit real keys to version control.

### 3. Start the Services

```bash
# Build all images (including the new litellm service)
docker compose build backend frontend litellm

# Start all services
docker compose up -d --force-recreate litellm backend frontend

# Verify LiteLLM Gateway is running
docker compose exec litellm curl http://localhost:4000/health

# Verify backend can reach the gateway
curl http://localhost:8000/api/health/provider
```

Expected `/api/health/provider` response:
```json
{
  "provider": "litellm",
  "model": "openrouter-gpt-oss",
  "base_url_host": "litellm",
  "gateway_mode": true,
  "litellm_enabled": true
}
```

---

## Environment Variables Reference

### Backend Provider Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `openrouter` \| `litellm` \| `openai` \| `ollama` \| `mock` | Selects the active LLM provider |
| `LITELLM_ENABLED` | `true` \| `false` | Controls whether the LiteLLM Gateway Docker service is used |

### Direct OpenRouter Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Your OpenRouter API key (from https://openrouter.ai/keys) |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-exp` | Model to use |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |

### LiteLLM Gateway Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm:4000` | LiteLLM Gateway internal URL |
| `LITELLM_MASTER_KEY` | `changeme_litellm_master_key` | **Change this!** Master key for gateway auth |
| `LITELLM_MODEL` | `openrouter-gpt-oss` | Model name as defined in `config.yaml` |
| `LITELLM_TIMEOUT` | `120` | Request timeout in seconds |
| `LITELLM_EXTERNAL_PORT` | `4000` | External host port (optional, only if you need host access) |

### Switching Between Modes

```bash
# Use direct OpenRouter (no LiteLLM Gateway needed)
LLM_PROVIDER=openrouter

# Use LiteLLM Gateway
LLM_PROVIDER=litellm
```

No code changes required — switching is purely environment-driven.

---

## LiteLLM Gateway Configuration

The gateway is configured via `infra/litellm/config.yaml`. This file is mounted read-only into the LiteLLM container.

### Model Definitions

Each model is mapped to a logical name and a real upstream model:

```yaml
model_list:
  - model_name: openrouter-gpt-oss
    litellm_params:
      model: openrouter/google/gemini-2.0-flash-exp
      api_key: env:OPENROUTER_API_KEY   # loaded from docker-compose env
```

### Available Models (Pre-configured)

| Logical Name | Upstream Model | Provider | Notes |
|---|---|---|---|
| `openrouter-gpt-oss` | `google/gemini-2.0-flash-exp` | OpenRouter | Recommended, fast, low cost |
| `openrouter-claude` | `anthropic/claude-3-haiku` | OpenRouter | Good quality |
| `openrouter-llama` | `meta-llama/llama-3-8b-instruct` | OpenRouter | Open-source |
| `openrouter-free` | `mistralai/mistral-7b-instruct` | OpenRouter | Balanced |
| `local-ollama` | `ollama_chat/llama3.2` | Ollama (local) | No API key needed |

### Adding Custom Models

1. Edit `infra/litellm/config.yaml`.
2. Add a new entry under `model_list`:

   ```yaml
   - model_name: my-custom-model
     litellm_params:
       model: openrouter/anthropic/claude-3-sonnet
       api_key: env:OPENROUTER_API_KEY
     model_info:
       mode: chat
   ```

3. Update `LITELLM_MODEL=my-custom-model` in `.env`.
4. Restart the litellm service: `docker compose restart litellm`.

---

## Security Guidance

### LiteLLM Master Key

- The `LITELLM_MASTER_KEY` is the gateway's authentication token.
- Clients (including the backend) must provide `Bearer <master_key>` in the `Authorization` header.
- **Never use the default value** (`changeme_litellm_master_key`) in production.
- Generate a strong random key:

  ```bash
  openssl rand -base64 32
  ```

- **Never commit real keys to version control.** Use `.env` (excluded by `.gitignore`).

### API Keys

- OpenRouter API keys are configured in `docker-compose.yml` under the `litellm` service's environment.
- Keys are referenced in `config.yaml` as `env:OPENROUTER_API_KEY` — they are never hardcoded.
- The backend does NOT need the OpenRouter API key when using `LLM_PROVIDER=litellm` — only the gateway's master key.

### Secrets in Logs and Responses

- The backend's `/api/health/provider` endpoint **never exposes secrets** — only the base URL hostname is returned.
- Provider error messages are sanitized — master keys and API keys never appear in logs or API responses.
- The `LiteLLMProvider` class never logs `Authorization` header values.

### Network Isolation

- The LiteLLM Gateway runs on the internal Docker network (`chatbot-network`).
- External access to port 4000 is disabled by default. Enable only if needed:

  ```bash
  LITELLM_EXTERNAL_PORT=4000 docker compose up -d
  ```

---

## Health Checks

### Backend Provider Info

```bash
curl http://localhost:8000/api/health/provider
```

Response:
```json
{
  "provider": "litellm",
  "model": "openrouter-gpt-oss",
  "base_url_host": "litellm",
  "gateway_mode": true,
  "litellm_enabled": true
}
```

### LiteLLM Gateway Health

```bash
curl http://localhost:4000/health
```

Or inside the Docker network:

```bash
docker compose exec litellm curl http://localhost:4000/health
```

### Backend Health

```bash
curl http://localhost:8000/health
```

---

## Switching Back to Direct OpenRouter

To disable the LiteLLM Gateway and use direct OpenRouter:

1. In `.env`:

   ```bash
   LLM_PROVIDER=openrouter
   LITELLM_ENABLED=false
   ```

2. Restart the backend (no need to stop the litellm container if you want to keep it running):

   ```bash
   docker compose restart backend
   ```

3. Verify:

   ```bash
   curl http://localhost:8000/api/health/provider
   ```

   Should show `"provider": "openrouter"` and `"gateway_mode": false`.

---

## Troubleshooting

### 401 or 403 — Authentication Failed

**Symptom:** `"[LiteLLM Gateway] authentication failed. Check LITELLM_MASTER_KEY."`

**Cause:** The `LITELLM_MASTER_KEY` in your `.env` does not match the one configured in `docker-compose.yml`.

**Fix:**
1. Check the master key in `.env`: `grep LITELLM_MASTER_KEY .env`
2. Check the master key in `docker-compose.yml`: `grep LITELLM_MASTER_KEY docker-compose.yml`
3. Ensure they match. Restart: `docker compose restart litellm backend`

---

### 404 — Model Not Found

**Symptom:** `"[LiteLLM Gateway] model '...' not found."`

**Cause:** The model name (`LITELLM_MODEL`) does not match any model defined in `infra/litellm/config.yaml`.

**Fix:**
1. List available models: look at `model_list` in `infra/litellm/config.yaml`.
2. Set `LITELLM_MODEL` to a valid model name (e.g., `openrouter-gpt-oss`).
3. Restart the backend: `docker compose restart backend`

---

### Connection Refused — Gateway Unreachable

**Symptom:** `"[LiteLLM Gateway] unreachable. Ensure the litellm service is running."`

**Cause:** The backend cannot reach `http://litellm:4000`.

**Fix:**
1. Verify the litellm container is running: `docker compose ps litellm`
2. Check container logs: `docker compose logs litellm`
3. If not running, start it: `docker compose up -d litellm`
4. Check health: `curl http://localhost:4000/health` (if external port is exposed)

---

### Rate Limited (429)

**Symptom:** `"[LiteLLM Gateway] rate limit exceeded."`

**Cause:** Either OpenRouter or the upstream provider rate-limited the request.

**Fix:** Wait and retry. Consider switching to a different model or reducing request frequency.

---

### Empty Response or Timeout

**Symptom:** Request times out or returns empty.

**Cause:** Model may be slow or unavailable.

**Fix:**
1. Check LiteLLM logs: `docker compose logs litellm`
2. Try a different model (e.g., switch from `openrouter-claude` to `openrouter-gpt-oss`)
3. Check `LITELLM_TIMEOUT` — increase if the model is slow

---

### Docker Compose Build Fails

**Symptom:** `docker compose build litellm` fails.

**Cause:** The LiteLLM image may have a changed tag or network issues.

**Fix:**
1. Update to the latest image: `docker compose pull litellm`
2. Check the [official LiteLLM Docker docs](https://docs.litellm.ai/docs/proxy/docker) for the correct image tag.
3. Verify the image: `docker images ghcr.io/berriai/litellm`

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Next.js)                                     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                      │
│                                                         │
│  LLM_PROVIDER=openrouter  →  OpenRouterProvider         │
│  LLM_PROVIDER=litellm     →  LiteLLMProvider            │
│                         (httpx → LiteLLM Gateway)       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (internal)
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌────────────────────┐    ┌──────────────────────────┐
│  LiteLLM Gateway   │    │  OpenRouter API          │
│  (litellm:4000)    │───▶│  (if LLM_PROVIDER=litellm│
│                    │    │   or =openrouter)        │
│  Uses config.yaml  │    └──────────────────────────┘
│  Auth via master   │
│  key               │
└────────────────────┘
```

When `LLM_PROVIDER=openrouter` (direct mode), the backend calls OpenRouter directly — the `litellm` container can be stopped or never started.

When `LLM_PROVIDER=litellm` (gateway mode), the backend calls the LiteLLM Gateway, which routes to OpenRouter (or another configured upstream).

---

## Further Reading

- [LiteLLM Documentation](https://docs.litellm.ai/docs)
- [LiteLLM Proxy Docker Setup](https://docs.litellm.ai/docs/proxy/docker)
- [OpenRouter Models](https://openrouter.ai/models)
- [LiteLLM Config YAML Reference](https://docs.litellm.ai/docs/proxy/configs)
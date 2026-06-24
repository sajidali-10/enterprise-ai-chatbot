FROM python:3.11-slim

WORKDIR /app

# Create non-root user for production hygiene
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create home directory and HuggingFace cache dir writable by appuser
RUN mkdir -p /home/appuser && chown appuser:appgroup /home/appuser
# RAGAS report directory (created here so it exists even before first eval run)
RUN mkdir -p /app/evals/ragas && chown appuser:appgroup /app/evals/ragas
ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_home/transformers

COPY requirements.txt .
# Phase 26: RAGAS 0.2.x imports langchain_community.chat_models.vertexai.ChatVertexAI
# which was removed in langchain-community>=0.4.0 (langchain 1.x). Provide a minimal stub
# so the import chain completes. We do not use VertexAI — this is only to unblock ragas.
RUN python -c "\
import os; \
p='/usr/local/lib/python3.11/site-packages/langchain_community/chat_models/vertexai.py'; \
d=os.path.dirname(p); \
os.makedirs(d, exist_ok=True); \
content = '''# Stub: ChatVertexAI/VertexAI removed from langchain-community>=0.4.0\n\
# ragas imports this only for type hints and LLM initialization paths.\n\
# We provide minimal shims so the ragas import chain completes.\n\
from typing import Any, Dict, Optional, Sequence\n\
\nclass _BaseVertexAI:\n    model_name: str = \"\"\n    temperature: float = 0.0\n    max_tokens: int = 256\n    streaming: bool = False\n    callbacks: Any = None\n    callback_manager: Any = None\n    def __init__(self, **kwargs: Any) -> None:\n        pass\n    def __call__(self, prompt: str, **kwargs: Any) -> str:\n        raise NotImplementedError(\"vertexai stub\")\n    async def __acall__(self, prompt: str, **kwargs: Any) -> str:\n        raise NotImplementedError(\"vertexai stub\")\n    def invoke(self, input: Any, **kwargs: Any) -> Any:\n        raise NotImplementedError(\"vertexai stub\")\n    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:\n        raise NotImplementedError(\"vertexai stub\")\n    @property\n    def _llm_type(self) -> str:\n        return \"vertexai\"\n\nclass ChatVertexAI(_BaseVertexAI):\n    model_name: str = \"chat-bison\"\n    def __init__(self, model_name: Optional[str] = None, **kwargs: Any) -> None:\n        super().__init__(**kwargs)\n        if model_name:\n            self.model_name = model_name\n\nclass VertexAI(_BaseVertexAI):\n    model_name: str = \"text-bison\"\n    def __init__(self, model_name: Optional[str] = None, **kwargs: Any) -> None:\n        super().__init__(**kwargs)\n        if model_name:\n            self.model_name = model_name\n'''; \
open(p,'w').write(content); \
print('vertexai stub created')"

# Pin torch to CPU-only from PyPI before requirements install.
# This ensures pip doesn't pull GPU wheels from the PyTorch index.
# torch 2.4.0 on PyPI is the CPU-only build; GPU wheels live only on
# download.pytorch.org which we intentionally avoid here.
RUN pip install --no-cache-dir torch==2.4.0
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY evaluations/ ./evaluations/
COPY scripts/ ./scripts/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
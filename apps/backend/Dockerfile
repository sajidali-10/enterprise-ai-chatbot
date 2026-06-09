FROM python:3.11-slim

WORKDIR /app

# Create non-root user for production hygiene
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create home directory and HuggingFace cache dir writable by appuser
RUN mkdir -p /home/appuser && chown appuser:appgroup /home/appuser
ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_home/transformers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/
COPY alembic/ ./alembic/
COPY alembic.ini .

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#!/bin/bash
# ==============================================================================
# Reset Knowledge Base Script
# ==============================================================================
# WARNING: This script deletes all documents, chunks, and vector data.
# Use only in development environments.
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}  Knowledge Base Reset Script${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo ""

# Check if we're in development mode
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Check confirmation
echo -e "${RED}WARNING: This will delete ALL documents and vectors!${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Get values with defaults
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-chatbot}"
POSTGRES_USER="${POSTGRES_USER:-chatbot}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-changeme}"

QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-documents}"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-chatbot-uploads}"

echo -e "${GREEN}[1/4] Clearing Qdrant collection...${NC}"
# Clear Qdrant collection
curl -s -X DELETE "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}" || true
echo "Qdrant collection deleted."

echo -e "${GREEN}[2/4] Clearing PostgreSQL tables...${NC}"
# Clear PostgreSQL tables using psql
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
TRUNCATE TABLE document_chunks CASCADE;
TRUNCATE TABLE document_versions CASCADE;
TRUNCATE TABLE documents CASCADE;
" 2>/dev/null || echo "Note: PostgreSQL tables may not exist yet or be empty."

echo -e "${GREEN}[3/4] Clearing MinIO files...${NC}"
# Install mc if not available and clear MinIO bucket
command -v mc >/dev/null 2>&1 || { echo "mc not found, skipping MinIO cleanup"; }
if command -v mc >/dev/null 2>&1; then
    mc alias set local "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" 2>/dev/null || true
    mc rm --recursive --force "local/$MINIO_BUCKET/" 2>/dev/null || echo "MinIO bucket cleared or empty."
fi

echo -e "${GREEN}[4/4] Done!${NC}"
echo ""
echo -e "${GREEN}Knowledge base has been reset.${NC}"
echo ""
echo "To re-index documents with real embeddings:"
echo "  1. Set EMBEDDING_PROVIDER=local in .env"
echo "  2. Upload documents via the API"
echo "  3. Or call POST /api/documents/reindex-all to reindex existing documents"
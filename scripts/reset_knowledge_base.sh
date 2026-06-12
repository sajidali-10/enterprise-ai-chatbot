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

# Track overall success
CLEANUP_FAILED=0

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}  Knowledge Base Reset Script${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo ""

# Safely load environment variables from .env
# - Ignores comments (lines starting with #)
# - Ignores blank lines
# - Supports quoted values
# - Does not execute arbitrary content
load_env_file() {
    local env_file="$1"
    local required_vars="${2:-}"
    
    if [ ! -f "$env_file" ]; then
        echo -e "${RED}Error: $env_file not found${NC}" >&2
        return 1
    fi
    
    local missing_vars=""
    
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip blank lines
        [ -z "$line" ] && continue
        
        # Skip comments (lines starting with #)
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Extract variable name (before =)
        local var_name
        var_name=$(echo "$line" | sed 's/=.*//' | tr -d '[:space:]')
        
        # Skip if no variable name
        [ -z "$var_name" ] && continue
        
        # Extract value (after =), remove surrounding quotes if present
        local value
        value=$(echo "$line" | sed 's/^[^=]*=//')
        
        # Remove surrounding quotes (single or double)
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/; s/^'"'"'\(.*\)'"'"'$/\1/')
        
        # Export the variable
        export "$var_name"="$value"
        
    done < "$env_file"
    
    # Check for required variables
    if [ -n "$required_vars" ]; then
        for var in $required_vars; do
            if [ -z "${!var}" ]; then
                missing_vars="$missing_vars $var"
            fi
        done
        if [ -n "$missing_vars" ]; then
            echo -e "${RED}Error: Required environment variables missing:$missing_vars${NC}" >&2
            return 1
        fi
    fi
    
    return 0
}

# Load only required database/service variables
ENV_FILE=".env"
REQUIRED_VARS="POSTGRES_HOST POSTGRES_PORT POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD QDRANT_HOST QDRANT_PORT QDRANT_COLLECTION MINIO_ENDPOINT MINIO_ROOT_USER MINIO_ROOT_PASSWORD MINIO_BUCKET"

if [ -f "$ENV_FILE" ]; then
    load_env_file "$ENV_FILE" "$REQUIRED_VARS" || exit 1
else
    echo -e "${YELLOW}Warning: $ENV_FILE not found, using defaults${NC}"
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

# For Qdrant from host, use localhost (port mapping)
QDRANT_HOST_FOR_API="${QDRANT_HOST}"
if [ "$QDRANT_HOST" = "qdrant" ]; then
    QDRANT_HOST_FOR_API="localhost"
fi

MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-chatbot-uploads}"

# ==============================================================================
# 1. Clear Qdrant collection
# ==============================================================================
echo -e "${GREEN}[1/5] Clearing Qdrant collection...${NC}"
QDRANT_RESPONSE=$(curl -s -X DELETE "http://${QDRANT_HOST_FOR_API}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}" 2>&1) || true
if echo "$QDRANT_RESPONSE" | grep -q "error"; then
    echo -e "${YELLOW}Warning: Qdrant collection may not exist: $QDRANT_RESPONSE${NC}"
else
    echo "Qdrant collection '${QDRANT_COLLECTION}' deleted."
fi

# ==============================================================================
# 2. Clear PostgreSQL tables using docker compose exec
# ==============================================================================
echo -e "${GREEN}[2/5] Clearing PostgreSQL tables...${NC}"

# Run psql inside the postgres container to ensure proper connectivity
PG_RESULT=$(docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
-- First check if tables exist
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name IN ('documents', 'document_versions', 'document_chunks');
" 2>&1) || true

if echo "$PG_RESULT" | grep -q "error\|fatal\|could not connect"; then
    echo -e "${RED}Error: Could not connect to PostgreSQL: $PG_RESULT${NC}"
    CLEANUP_FAILED=1
else
    # Check if tables exist
    TABLE_COUNT=$(echo "$PG_RESULT" | tail -n 1 | tr -d ' ')
    
    if [ "$TABLE_COUNT" = "0" ] || [ -z "$TABLE_COUNT" ]; then
        echo "PostgreSQL tables do not exist yet."
    else
        # Tables exist, truncate them
        TRUNCATE_RESULT=$(docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
        TRUNCATE TABLE document_chunks CASCADE;
        TRUNCATE TABLE document_versions CASCADE;
        TRUNCATE TABLE documents CASCADE;
        " 2>&1) || true
        
        if echo "$TRUNCATE_RESULT" | grep -q "error\|ERROR"; then
            echo -e "${RED}Error truncating tables: $TRUNCATE_RESULT${NC}"
            CLEANUP_FAILED=1
        else
            echo "PostgreSQL tables truncated successfully."
        fi
    fi
fi

# ==============================================================================
# 3. Clear MinIO files using docker compose exec minio mc
# ==============================================================================
echo -e "${GREEN}[3/5] Clearing MinIO files...${NC}"

# Configure mc alias inside the minio container
# Use http://localhost:9000 because we're executing inside the container network
MC_CONFIGURE=$(docker compose exec -T minio mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" 2>&1) || {
    echo -e "${YELLOW}Warning: Could not configure MinIO mc alias: $MC_CONFIGURE${NC}"
}

# List and remove all objects in the bucket
echo "Removing objects from MinIO bucket: $MINIO_BUCKET"
MC_LS_RESULT=$(docker compose exec -T minio mc ls "local/${MINIO_BUCKET}" 2>&1) || true

if echo "$MC_LS_RESULT" | grep -q "not found\|does not exist"; then
    echo "MinIO bucket '$MINIO_BUCKET' does not exist."
    
    # Recreate the bucket
    echo "Recreating MinIO bucket..."
    MC_MB_RESULT=$(docker compose exec -T minio mc mb "local/${MINIO_BUCKET}" 2>&1) || {
        echo -e "${RED}Error creating MinIO bucket: $MC_MB_RESULT${NC}"
        CLEANUP_FAILED=1
    }
    if [ $CLEANUP_FAILED -eq 0 ]; then
        echo "MinIO bucket '$MINIO_BUCKET' created."
    fi
else
    # Bucket exists, remove all objects
    MC_RM_RESULT=$(docker compose exec -T minio mc rm --recursive --force "local/${MINIO_BUCKET}/" 2>&1) || true
    
    if echo "$MC_RM_RESULT" | grep -q "error\|Error"; then
        echo -e "${RED}Error removing MinIO objects: $MC_RM_RESULT${NC}"
        CLEANUP_FAILED=1
    else
        echo "MinIO bucket '$MINIO_BUCKET' cleared."
        
        # Recreate the bucket to ensure it's fresh
        MC_RB_RESULT=$(docker compose exec -T minio mc rb "local/${MINIO_BUCKET}" 2>&1) || true
        MC_MB_RESULT=$(docker compose exec -T minio mc mb "local/${MINIO_BUCKET}" 2>&1) || {
            echo -e "${RED}Error recreating MinIO bucket: $MC_MB_RESULT${NC}"
            CLEANUP_FAILED=1
        }
        if [ $CLEANUP_FAILED -eq 0 ]; then
            echo "MinIO bucket '$MINIO_BUCKET' recreated."
        fi
    fi
fi

# ==============================================================================
# 4. Verify cleanup
# ==============================================================================
echo -e "${GREEN}[4/5] Verifying cleanup...${NC}"
echo ""

echo "Verification counts:"
echo "--------------------"

# Verify PostgreSQL
DOC_COUNT=$(docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM documents;" 2>/dev/null | tr -d ' ' || echo "N/A")
CHUNK_COUNT=$(docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM document_chunks;" 2>/dev/null | tr -d ' ' || echo "N/A")
VERSION_COUNT=$(docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM document_versions;" 2>/dev/null | tr -d ' ' || echo "N/A")

echo "  PostgreSQL:"
echo "    - documents: $DOC_COUNT"
echo "    - document_chunks: $CHUNK_COUNT"
echo "    - document_versions: $VERSION_COUNT"

# Verify Qdrant
QDRANT_CHECK=$(curl -s "http://${QDRANT_HOST_FOR_API}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}" 2>&1 || echo "{}")
if echo "$QDRANT_CHECK" | grep -q '"result":null\|does not exist'; then
    echo "  Qdrant: Collection '${QDRANT_COLLECTION}' does not exist (empty)"
else
    VECTOR_COUNT=$(curl -s "http://${QDRANT_HOST_FOR_API}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}" 2>&1 | grep -o '"points_count":[0-9]*' | cut -d: -f2 || echo "N/A")
    echo "  Qdrant: Collection '${QDRANT_COLLECTION}' - points: $VECTOR_COUNT"
fi

# Verify MinIO
MINIO_OBJECTS=$(docker compose exec -T minio mc ls "local/${MINIO_BUCKET}/" 2>&1 | wc -l || echo "0")
echo "  MinIO: Bucket '${MINIO_BUCKET}' - objects: $MINIO_OBJECTS"

echo ""

# ==============================================================================
# 5. Final status
# ==============================================================================
echo -e "${GREEN}[5/5] Final status${NC}"

# Check if all counts are zero or N/A
if [ "$DOC_COUNT" = "0" ] && [ "$CHUNK_COUNT" = "0" ] && [ "$VERSION_COUNT" = "0" ]; then
    echo -e "${GREEN}PostgreSQL: CLEANED${NC}"
else
    echo -e "${RED}PostgreSQL: FAILED (counts above are not zero)${NC}"
    CLEANUP_FAILED=1
fi

if echo "$QDRANT_CHECK" | grep -q '"result":null\|does not exist\|404\|not found\|"error"'; then
    echo -e "${GREEN}Qdrant: CLEANED${NC}"
else
    # Check if points_count is 0
    if echo "$QDRANT_CHECK" | grep -q '"points_count":0'; then
        echo -e "${GREEN}Qdrant: CLEANED (0 points)${NC}"
    else
        echo -e "${RED}Qdrant: FAILED (collection may still have data)${NC}"
        CLEANUP_FAILED=1
    fi
fi

if [ "$MINIO_OBJECTS" = "0" ] || [ "$MINIO_OBJECTS" = "1" ]; then
    # 1 because wc -l returns 1 for empty output
    echo -e "${GREEN}MinIO: CLEANED${NC}"
else
    echo -e "${RED}MinIO: FAILED (bucket may still have objects)${NC}"
    CLEANUP_FAILED=1
fi

echo ""

if [ $CLEANUP_FAILED -ne 0 ]; then
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}  Reset FAILED - See errors above${NC}"
    echo -e "${RED}======================================================================${NC}"
    exit 1
fi

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  Knowledge base has been reset successfully!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "To re-index documents with real embeddings:"
echo "  1. Set EMBEDDING_PROVIDER=local in .env"
echo "  2. Upload documents via the API"
echo "  3. Or call POST /api/documents/reindex-all to reindex existing documents"
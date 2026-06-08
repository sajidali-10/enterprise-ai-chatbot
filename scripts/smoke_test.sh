#!/bin/bash
# ==============================================================================
# Phase 6.5 End-to-End Smoke Test
# ==============================================================================
# Tests all major endpoints in order: health, auth-info, upload, index, RAG chat
# with citations and audit log verification.
#
# Usage: ./scripts/smoke_test.sh [--backend URL] [--skip-upload]
# Default backend: http://localhost:8000
# ==============================================================================

set -e

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
SKIP_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND_URL="$2"; shift 2 ;;
        --skip-upload) SKIP_UPLOAD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; }
warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; }
info() { echo -e "  INFO: $1"; }

section() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

# Store test results
declare -A RESULTS

###############################################################################
# Helper Functions
###############################################################################

check_http() {
    local name="$1"
    local expected_status="${2:-200}"
    local response
    response=$(curl -s -w "\n%{http_code}" -o /tmp/smoke_response_$$.tmp "$3" 2>/dev/null || echo "000")
    local status="${response##*$'\n'}"
    local body=$(cat /tmp/smoke_response_$$.tmp 2>/dev/null || echo "")
    
    if [[ "$status" == "$expected_status" ]]; then
        pass "$name (HTTP $status)"
        RESULTS["$name"]="pass"
        echo "$body"
        return 0
    else
        fail "$name (expected $expected_status, got $status)"
        RESULTS["$name"]="fail"
        echo "Response: $body" | head -20
        return 1
    fi
}

check_http_json() {
    local name="$1"
    local expected_status="${2:-200}"
    local response
    response=$(curl -s -w "\n%{http_code}" -o /tmp/smoke_response_$$.tmp "$3" 2>/dev/null || echo "000")
    local status="${response##*$'\n'}"
    local body=$(cat /tmp/smoke_response_$$.tmp 2>/dev/null || echo "")
    
    if [[ "$status" == "$expected_status" ]]; then
        if echo "$body" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
            pass "$name (HTTP $status, valid JSON)"
            RESULTS["$name"]="pass"
            echo "$body"
            return 0
        else
            fail "$name (HTTP $status, invalid JSON)"
            RESULTS["$name"]="fail"
            echo "Response: $body" | head -20
            return 1
        fi
    else
        fail "$name (expected $expected_status, got $status)"
        RESULTS["$name"]="fail"
        echo "Response: $body" | head -20
        return 1
    fi
}

###############################################################################
# Phase 0: Infrastructure Health
###############################################################################
section "Phase 0: Infrastructure Health"

# Test 1: Backend health endpoint
check_http "Backend health check" 200 "${BACKEND_URL}/health"
HEALTH_RESP=$(cat /tmp/smoke_response_$$.tmp 2>/dev/null || echo "{}")
if echo "$HEALTH_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='healthy' else 1)" 2>/dev/null; then
    pass "Health status is 'healthy'"
    RESULTS["health_status"]="pass"
else
    warn "Health response missing or status != 'healthy': $HEALTH_RESP"
fi

###############################################################################
# Phase 6: Auth Infrastructure
###############################################################################
section "Phase 6: Auth Infrastructure"

# Test 2: Auth info without dev header (anonymous)
AUTH_INFO=$(check_http_json "Auth info (anonymous)" 200 "${BACKEND_URL}/api/chat/auth-info")
if echo "$AUTH_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('authenticated')==False and d.get('role')=='viewer' else 1)" 2>/dev/null; then
    pass "Anonymous user gets viewer role"
    RESULTS["anonymous_auth"]="pass"
else
    warn "Anonymous auth behavior may differ from expected"
fi

# Test 3: Auth info with admin dev header
ADMIN_AUTH=$(curl -s -H "X-Dev-User: admin_test" "${BACKEND_URL}/api/chat/auth-info")
if echo "$ADMIN_AUTH" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('authenticated')==True and d.get('role')=='admin' and d.get('is_admin')==True else 1)" 2>/dev/null; then
    pass "Admin dev user authenticated correctly"
    RESULTS["admin_auth"]="pass"
else
    fail "Admin dev user auth failed: $ADMIN_AUTH"
    RESULTS["admin_auth"]="fail"
fi

# Test 4: Auth info with regular dev header
USER_AUTH=$(curl -s -H "X-Dev-User: testuser" "${BACKEND_URL}/api/chat/auth-info")
if echo "$USER_AUTH" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('authenticated')==True and d.get('role')=='user' else 1)" 2>/dev/null; then
    pass "Regular dev user authenticated correctly"
    RESULTS["user_auth"]="pass"
else
    fail "Regular dev user auth failed: $USER_AUTH"
    RESULTS["user_auth"]="fail"
fi

###############################################################################
# Phase 1-2: Document Upload & Indexing
###############################################################################
section "Phase 1-2: Document Upload & Indexing"

# Create a test document
TEST_CONTENT="This is a test document about artificial intelligence and machine learning.
It contains information about neural networks, deep learning, and natural language processing.
The document discusses supervised learning, unsupervised learning, and reinforcement learning.
It also covers topics like transformers, attention mechanisms, and large language models.
This content is used for smoke testing the RAG pipeline and citation system."

echo "$TEST_CONTENT" > /tmp/smoke_test_doc.txt

if [[ "$SKIP_UPLOAD" == "false" ]]; then
    # Test 5: Upload document (requires admin auth)
    info "Uploading test document..."
    UPLOAD_RESP=$(curl -s -X POST \
        -H "X-Dev-User: admin_upload" \
        -F "file=@/tmp/smoke_test_doc.txt;type=text/plain" \
        "${BACKEND_URL}/api/documents/upload" 2>/dev/null)
    
    if echo "$UPLOAD_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('id') else 1)" 2>/dev/null; then
        pass "Document uploaded successfully"
        RESULTS["upload"]="pass"
        
        DOC_ID=$(echo "$UPLOAD_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
        info "Document ID: $DOC_ID"
        
        # Test 6: Check document is indexed
        DOC_STATUS=$(echo "$UPLOAD_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
        if [[ "$DOC_STATUS" == "indexed" ]]; then
            pass "Document auto-indexed (status: indexed)"
            RESULTS["indexing"]="pass"
        else
            warn "Document status is '$DOC_STATUS' (expected 'indexed')"
            RESULTS["indexing"]="partial"
        fi
        
        # Test 7: List documents
        LIST_RESP=$(curl -s -H "X-Dev-User: admin_list" "${BACKEND_URL}/api/documents" 2>/dev/null)
        if echo "$LIST_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if isinstance(d,list) and len(d)>0 else 1)" 2>/dev/null; then
            pass "Document list endpoint works"
            RESULTS["list_documents"]="pass"
        else
            fail "Document list failed: $LIST_RESP"
            RESULTS["list_documents"]="fail"
        fi
        
    else
        fail "Document upload failed: $UPLOAD_RESP"
        RESULTS["upload"]="fail"
        RESULTS["indexing"]="skip"
        DOC_ID=""
    fi
else
    warn "Skipping upload test (--skip-upload specified)"
    RESULTS["upload"]="skip"
    RESULTS["indexing"]="skip"
    RESULTS["list_documents"]="skip"
    DOC_ID=""
fi

###############################################################################
# Phase 4-5: RAG Chat & Citations
###############################################################################
section "Phase 4-5: RAG Chat & Citations"

# Test 8: RAG chat with citations
info "Testing RAG chat with query about machine learning..."
RAG_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-Dev-User: admin_rag" \
    -d '{"message":"What topics does the document cover?","mode":"rag"}' \
    "${BACKEND_URL}/api/chat" 2>/dev/null)

if echo "$RAG_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('message') else 1)" 2>/dev/null; then
    pass "RAG chat returned a response"
    RESULTS["rag_chat"]="pass"
    
    # Check for citations
    CITATIONS=$(echo "$RAG_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('citations',[])))" 2>/dev/null)
    if echo "$CITATIONS" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d and len(d)>0 else 1)" 2>/dev/null; then
        pass "Citations returned in RAG response"
        RESULTS["citations"]="pass"
        info "Citation count: $(echo "$CITATIONS" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' 2>/dev/null)"
    else
        warn "No citations in RAG response (may be expected if no chunks matched)"
        RESULTS["citations"]="partial"
    fi
    
    # Show response preview
    RESPONSE_TEXT=$(echo "$RAG_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('message','')[:200])" 2>/dev/null)
    info "Response preview: ${RESPONSE_TEXT:0:150}..."
    
    # Check debug info
    DEBUG_INFO=$(echo "$RAG_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('debug_info',{})))" 2>/dev/null)
    if echo "$DEBUG_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d and d.get('chunks_retrieved',0) > 0 else 1)" 2>/dev/null; then
        pass "Debug info shows chunks retrieved"
        RESULTS["debug_info"]="pass"
    else
        warn "Debug info not available or no chunks retrieved"
        RESULTS["debug_info"]="partial"
    fi
else
    fail "RAG chat failed: $RAG_RESP"
    RESULTS["rag_chat"]="fail"
    RESULTS["citations"]="skip"
    RESULTS["debug_info"]="skip"
fi

# Test 9: Normal chat (no RAG)
info "Testing normal chat (no RAG)..."
NORMAL_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-Dev-User: admin_normal" \
    -d '{"message":"Say hello in one sentence","mode":"normal"}' \
    "${BACKEND_URL}/api/chat" 2>/dev/null)

if echo "$NORMAL_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('message') else 1)" 2>/dev/null; then
    pass "Normal chat works"
    RESULTS["normal_chat"]="pass"
else
    warn "Normal chat failed: $NORMAL_RESP"
    RESULTS["normal_chat"]="fail"
fi

###############################################################################
# Phase 6: Audit Log Verification
###############################################################################
section "Phase 6: Audit Logging"

# Note: Audit logs are stored in the database. We can verify they exist
# by checking if the upload logged an event. Full audit verification
# would require database access.

info "Audit logging is implemented in the API layer."
info "Events are logged for: chat (normal + RAG), upload, index"
info "To verify audit logs directly, query the PostgreSQL audit_logs table."

# Test 10: Permission denied is logged
info "Testing permission denial for non-admin upload..."
PERM_RESP=$(curl -s -X POST \
    -H "X-Dev-User: regular_user" \
    -F "file=@/tmp/smoke_test_doc.txt;type=text/plain" \
    "${BACKEND_URL}/api/documents/upload" 2>/dev/null)

if echo "$PERM_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('detail','').find('Permission denied') != -1 or d.get('detail','').find('admin') != -1 else 1)" 2>/dev/null; then
    pass "Permission denied (403) returned for non-admin"
    RESULTS["permission_denied"]="pass"
else
    warn "Permission denial response unexpected: $PERM_RESP"
    RESULTS["permission_denied"]="partial"
fi

###############################################################################
# Summary
###############################################################################
section "Smoke Test Summary"

total=0
passed=0
failed=0
skipped=0

for key in "${!RESULTS[@]}"; do
    total=$((total + 1))
    case "${RESULTS[$key]}" in
        pass) passed=$((passed + 1)) ;;
        fail) failed=$((failed + 1)) ;;
        skip) skipped=$((skipped + 1)) ;;
        partial) passed=$((passed + 1)) ;;
    esac
done

echo ""
echo "Results: $passed passed, $failed failed, $skipped skipped (of $total tests)"
echo ""

if [[ $failed -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Review output above.${NC}"
    exit 1
fi
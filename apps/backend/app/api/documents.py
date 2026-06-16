import uuid
import mimetypes
import hashlib
import io
import os
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request

from app.security.dependencies import require_permission
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.minio_client import get_minio_client, ensure_bucket_exists
from app.core.config import settings
from app.core.rate_limit import rate_limit
from app.models.document import Document, DocumentVersion, DocumentChunk
from app.ingestion.pipeline import process_document, get_parser
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.services.embeddings import get_embedding_provider
from app.services.vector.qdrant_service import ensure_collection, upsert_chunks

# Import security modules for Phase 6
try:
    from app.security.auth import AuthContext, authenticate_request
    from app.security.models import UserRole, AuditAction, User
    from app.security.audit import get_audit_logger
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None
    UserRole = None


def get_auth_context(request: Request) -> AuthContext:
    """
    Extract authentication context from request.
    
    Returns AuthContext with user info, role, and auth status.
    In development, supports X-Dev-User header.
    """
    if not HAS_SECURITY:
        return None
    return authenticate_request(request)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

logger = logging.getLogger(__name__)

# Allowed MIME types for document uploads
# PDF, DOCX, TXT, MD, CSV
ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
    "application/csv",
}

# Allowed file extensions (for additional security validation)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and unsafe characters.

    - Strips path components (takes basename only)
    - Removes control characters
    - Limits length
    - Returns a safe filename string
    """
    import re
    # Take basename only (prevents path traversal like ../../etc/passwd)
    safe = os.path.basename(filename)
    # Remove control characters and potentially dangerous chars
    safe = re.sub(r'[<>:"|?*\x00-\x1f]', '', safe)
    # Limit to 255 chars (common filesystem limit)
    safe = safe[:255]
    # Strip leading/trailing whitespace and dots
    safe = safe.strip(' .')
    if not safe:
        safe = "unnamed_file"
    return safe


def _validate_filename(filename: str) -> None:
    """
    Validate a filename for security issues.

    Raises HTTPException if the filename contains path traversal or is empty.
    """
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename is required")
    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")
    # Check extension
    ext = _ext_from_filename(filename).lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file extension: {ext}")

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    if request.client:
        return request.client.host
    return "unknown"


def _ext_from_filename(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""

@router.post("/upload")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    visibility: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_upload_documents")),
    _=Depends(rate_limit(max_requests=10, window=60)),
):
    """
    Upload a document, extract text, and auto-index for RAG.

    **Authentication Required:** Yes
    **Permission Required:** can_upload_documents

    Phase 13 visibility rules:
      - Admin upload defaults to visibility='global'. The optional form field
        'visibility' (private|shared|global) overrides the default.
      - Regular user upload defaults to visibility='private'. The 'visibility'
        form field is IGNORED for non-admins to enforce least-privilege.
      - Viewer upload is blocked at the can_upload_documents permission layer
        (viewer role does not have this permission; returns 403).

    The document's owner_user_id is set to the current authenticated user.

    Status flow:
    - "pending" - Initial state after upload
    - "extracting" - Text extraction in progress
    - "extracted" - Text extracted, ready for indexing
    - "indexing" - Chunking, embedding, and Qdrant upsert in progress
    - "indexed" - Fully indexed and ready for RAG retrieval
    - "failed" - Something went wrong (check error message)
    """
    max_size = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    content = file.file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Validate and sanitize filename
    _validate_filename(file.filename)
    safe_original_name = _sanitize_filename(file.filename)

    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    if mime_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    ext = _ext_from_filename(safe_original_name)
    storage_key = f"{uuid.uuid4().hex}{ext}"
    
    ensure_bucket_exists()
    client = get_minio_client()
    client.put_object(
        settings.MINIO_BUCKET,
        storage_key,
        data=io.BytesIO(content),
        length=len(content),
        content_type=mime_type,
    )
    
    # Create document with pending status
    # Phase 13: capture owner_user_id and apply role-aware visibility default.
    is_admin = bool(getattr(auth, "is_admin", lambda: False)())
    if is_admin:
        # Admin: default to 'global', allow override via form field.
        chosen_visibility = (visibility or "global").lower()
    else:
        # Regular user: ALWAYS private (ignore any client-supplied visibility).
        chosen_visibility = "private"
    if chosen_visibility not in ("private", "shared", "global"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid visibility '{chosen_visibility}'. Must be one of: private, shared, global.",
        )

    doc = Document(
        filename=storage_key,
        original_name=safe_original_name,
        mime_type=mime_type,
        size_bytes=len(content),
        status="pending",
        visibility=chosen_visibility,
        owner_user_id=auth.user_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    # Extract text
    doc.status = "extracting"
    db.commit()
    
    try:
        extracted_text = process_document(content, mime_type)
        extractor_type = type(get_parser(mime_type)).__name__ if get_parser(mime_type) else None
    except Exception as e:
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    
    # Check if extracted text is empty
    if not extracted_text or not extracted_text.strip():
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=400, detail="No extractable text found in document.")
    
    version = DocumentVersion(
        document_id=doc.id,
        version_number=1,
        storage_key=storage_key,
        extractor_type=extractor_type,
        extracted_text=extracted_text,
    )
    db.add(version)
    doc.status = "extracted"
    db.commit()
    
    # Auto-index the document
    doc.status = "indexing"
    db.commit()
    
    try:
        # Chunk the text
        chunker = RecursiveChunker()
        metadata = {
            "source_file_name": doc.original_name,
            "title": doc.original_name,
        }
        chunks = chunker.chunk(extracted_text, metadata)
        
        if not chunks:
            doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Validate each chunk has non-empty content before creating records/vectors
        valid_chunks = [c for c in chunks if c.content is not None and c.content.strip()]
        if len(valid_chunks) != len(chunks):
            invalid_count = len(chunks) - len(valid_chunks)
            logger.warning(
                f"Filtered {invalid_count} invalid chunks with empty/None content for document {doc.id}"
            )
        chunks = valid_chunks
        
        if not chunks:
            doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No chunks with valid content generated from document")
        
        # Store chunks in PostgreSQL
        chunk_ids = []
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=doc.id,
                document_version_id=version.id,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                section_heading=chunk.section_heading,
                page_number=chunk.page_number,
                source_file_name=chunk.source_file_name,
                content_hash=chunk.content_hash,
                content=chunk.content,
            )
            db.add(db_chunk)
            db.flush()
            chunk_ids.append(db_chunk.id)
        
        db.commit()
        
        # Get embeddings
        provider = get_embedding_provider()
        texts = [c.content for c in chunks]
        embeddings = provider.embed(texts)
        
        # Ensure Qdrant collection exists
        ensure_collection(provider.dimension)
        
        # Upsert to Qdrant
        chunks_with_embeddings = list(zip(texts, embeddings))
        metadata_payloads = []
        for i, chunk in enumerate(chunks):
            metadata_payloads.append({
                "chunk_id": chunk_ids[i],
                "document_id": doc.id,
                "document_version_id": version.id,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "source_file_name": chunk.source_file_name,
                "title": chunk.title,
                "section_heading": chunk.section_heading,
            })
        
        upsert_chunks(chunks_with_embeddings, metadata_payloads)
        
        doc.status = "indexed"
        db.commit()
        
        # Phase 6: Log successful upload with audit
        if HAS_SECURITY:
            try:
                audit_logger = get_audit_logger()
                from app.security.audit import AuditEvent
                event = AuditEvent(
                    action=AuditAction.DOCUMENT_UPLOAD,
                    username=auth.username,
                    user_id=auth.user_id,
                    status="success",
                    request_ip=get_client_ip(request),
                    request_user_agent=request.headers.get("user-agent", "")[:500],
                    details={
                        "document_id": doc.id,
                        "original_name": doc.original_name,
                        "chunks_created": len(chunks),
                    },
                )
                audit_logger.log(event)
            except Exception:
                pass  # Don't fail the request if audit logging fails
        
    except HTTPException:
        raise
    except Exception as e:
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    
    return {
        "id": doc.id,
        "original_name": doc.original_name,
        "mime_type": doc.mime_type,
        "size_bytes": doc.size_bytes,
        "status": doc.status,
        "visibility": doc.visibility,
        "owner_user_id": doc.owner_user_id,
        "storage_key": storage_key,
        "chunks_created": len(chunks) if 'chunks' in dir() else 0,
    }

@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_view_documents")),
):
    """
    List documents the caller is authorized to view.

    Visibility rules (Phase 13):
      - Admin: sees every document.
      - User: sees global + owned + user-shared + role-shared docs.
      - Viewer: sees global + user-shared + role-shared docs.

    Each payload item now includes `visibility`, `owner_user_id`, and
    `owner_username` so the frontend can render badges and the access panel.
    """
    # Resolve the set of document IDs the caller can access.
    from app.security.permissions import get_accessible_document_ids, can_manage_document
    accessible_ids = set(get_accessible_document_ids(auth, db=db))

    if not accessible_ids:
        return []

    # Pull only the authorized docs, with owner joined in a single query.
    rows = (
        db.query(Document, User.username)
        .outerjoin(User, Document.owner_user_id == User.id)
        .filter(Document.id.in_(accessible_ids))
        .order_by(Document.created_at.desc())
        .all()
    )

    return [
        {
            "id": d.id,
            "original_name": d.original_name,
            "mime_type": d.mime_type,
            "size_bytes": d.size_bytes,
            "status": d.status,
            "chunk_count": len(d.chunks) if d.chunks else 0,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "visibility": d.visibility,
            "owner_user_id": d.owner_user_id,
            "owner_username": owner_username,
            # Per-doc manage flag so the UI can hide Delete/Reindex for rows
            # the caller can't actually manage (even if can_delete_documents /
            # can_reindex_documents are true at the role level).
            "can_manage": can_manage_document(auth, d.id, db=db),
        }
        for d, owner_username in rows
    ]


@router.post("/{document_id}/index")
def index_document(
    request: Request,
    document_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_reindex_documents")),
):
    """
    Manually re-index a document that has extracted text but needs chunking/embedding.

    **Authentication Required:** Yes
    **Permission Required:** can_reindex_documents (overall) + Phase 13 per-doc manage.

    Phase 13: same per-doc manage check as delete_document. The caller must
    satisfy ``can_manage_document(auth, document_id)``.

    This is useful if:
    - A previous indexing attempt failed
    - The document was uploaded with extraction only (status="extracted")
    - You want to re-chunk with different settings
    """
    from app.security.permissions import can_manage_document, can_access_document

    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Phase 13: enforce per-doc manage permission (404 to avoid enumeration when caller can't view).
    if not can_manage_document(auth, document_id, db=db):
        if not can_access_document(auth, document_id, db=db):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to reindex this document",
        )

    # Get latest version
    latest_version = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == document_id
    ).order_by(DocumentVersion.version_number.desc()).first()
    
    if not latest_version or not latest_version.extracted_text:
        raise HTTPException(status_code=400, detail="No extracted text to index")
    
    doc.status = "indexing"
    db.commit()
    
    try:
        # Chunk the text
        chunker = RecursiveChunker()
        metadata = {
            "source_file_name": doc.original_name,
            "title": doc.original_name,
        }
        chunks = chunker.chunk(latest_version.extracted_text, metadata)
        
        if not chunks:
            doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No chunks generated")
        
        # Validate each chunk has non-empty content before creating records/vectors
        valid_chunks = [c for c in chunks if c.content is not None and c.content.strip()]
        if len(valid_chunks) != len(chunks):
            invalid_count = len(chunks) - len(valid_chunks)
            logger.warning(
                f"Filtered {invalid_count} invalid chunks with empty/None content for document {document_id}"
            )
        chunks = valid_chunks
        
        if not chunks:
            doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No chunks with valid content generated")
        
        # Store chunks in PostgreSQL
        chunk_ids = []
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=document_id,
                document_version_id=latest_version.id,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                section_heading=chunk.section_heading,
                page_number=chunk.page_number,
                source_file_name=chunk.source_file_name,
                content_hash=chunk.content_hash,
                content=chunk.content,
            )
            db.add(db_chunk)
            db.flush()
            chunk_ids.append(db_chunk.id)
        
        db.commit()
        
        # Get embeddings
        provider = get_embedding_provider()
        texts = [c.content for c in chunks]
        embeddings = provider.embed(texts)
        
        # Ensure Qdrant collection exists
        ensure_collection(provider.dimension)
        
        # Upsert to Qdrant
        chunks_with_embeddings = list(zip(texts, embeddings))
        metadata_payloads = []
        for i, chunk in enumerate(chunks):
            metadata_payloads.append({
                "chunk_id": chunk_ids[i],
                "document_id": document_id,
                "document_version_id": latest_version.id,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "source_file_name": chunk.source_file_name,
                "title": chunk.title,
                "section_heading": chunk.section_heading,
            })
        
        upsert_chunks(chunks_with_embeddings, metadata_payloads)
        
        doc.status = "indexed"
        db.commit()
        
        # Phase 6: Log successful indexing with audit
        if HAS_SECURITY:
            try:
                audit_logger = get_audit_logger()
                from app.security.audit import AuditEvent
                event = AuditEvent(
                    action=AuditAction.DOCUMENT_INDEX,
                    username=auth.username,
                    user_id=auth.user_id,
                    status="success",
                    request_ip=get_client_ip(request),
                    request_user_agent=request.headers.get("user-agent", "")[:500],
                    details={
                        "document_id": document_id,
                        "version_id": latest_version.id,
                        "chunks_created": len(chunks),
                    },
                )
                audit_logger.log(event)
            except Exception:
                pass  # Don't fail the request if audit logging fails
        
        return {
            "document_id": document_id,
            "version_id": latest_version.id,
            "chunks_created": len(chunks),
            "embedding_dimension": provider.dimension,
        }
    except HTTPException:
        raise
    except Exception as e:
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.delete("/{document_id}")
def delete_document(
    request: Request,
    document_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_delete_documents")),
):
    """
    Delete a document and all its associated data.

    Deletes:
    - Document metadata from PostgreSQL
    - All chunks from PostgreSQL
    - All vectors from Qdrant
    - File from MinIO

    **Authentication Required:** Yes
    **Permission Required:** can_delete_documents (overall) + Phase 13 per-doc manage.

    Phase 13: in addition to the role-level permission, the caller must satisfy
    ``can_manage_document(auth, document_id)``:
      - admin (any) → always allowed
      - owner (owner_user_id == auth.user_id) → allowed
      - user-share with access_level='manage' → allowed
      - role-share with access_level='manage' → allowed
      - otherwise → 403

    The document is hidden behind a 404 (not 403) when the caller can neither view
    nor manage it, to avoid enumeration of private document IDs.
    """
    from app.security.permissions import can_manage_document, can_access_document

    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Phase 13: enforce per-doc manage permission.
    if not can_manage_document(auth, document_id, db=db):
        # If the caller can't even view this doc, return 404 to avoid enumeration.
        if not can_access_document(auth, document_id, db=db):
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to delete this document",
        )

    # Get storage key before deleting
    storage_key = doc.filename

    # Delete from Qdrant
    try:
        from app.services.vector.qdrant_service import delete_vectors_by_document_id
        deleted_vectors = delete_vectors_by_document_id(document_id)
    except Exception:
        deleted_vectors = 0

    # Delete chunks from PostgreSQL
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

    # Delete versions from PostgreSQL
    db.query(DocumentVersion).filter(DocumentVersion.document_id == document_id).delete()

    # Delete document
    db.delete(doc)
    db.commit()

    # Delete from MinIO
    try:
        client = get_minio_client()
        client.remove_object(settings.MINIO_BUCKET, storage_key)
    except Exception:
        pass  # Don't fail if MinIO delete fails

    return {"deleted": True, "document_id": document_id, "vectors_deleted": deleted_vectors}


@router.post("/reindex-all")
def reindex_all_documents(
    request: Request,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_reindex_documents")),
):
    """
    Re-index all documents with the current embedding provider.

    This clears all Qdrant vectors and re-embeds all documents.
    Use this after changing the embedding provider.

    **Authentication Required:** Yes
    **Permission Required:** can_reindex_documents
    """
    # Get all indexed documents
    docs = db.query(Document).filter(Document.status == "indexed").all()

    if not docs:
        return {"reindexed": 0, "message": "No indexed documents found"}

    # Clear Qdrant collection
    try:
        from app.services.vector.qdrant_service import delete_collection, ensure_collection
        delete_collection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant: {str(e)}")

    # Get embedding provider
    provider = get_embedding_provider()
    ensure_collection(provider.dimension)

    reindexed_count = 0
    errors = []

    for doc in docs:
        try:
            # Get all chunks for this document
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).order_by(DocumentChunk.chunk_index).all()

            if not chunks:
                continue

            # Get embeddings
            texts = [c.content for c in chunks]
            embeddings = provider.embed(texts)

            # Prepare metadata
            metadata_payloads = []
            for i, chunk in enumerate(chunks):
                metadata_payloads.append({
                    "chunk_id": chunk.id,
                    "document_id": doc.id,
                    "document_version_id": chunk.document_version_id,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                    "source_file_name": chunk.source_file_name,
                    "title": chunk.title,
                    "section_heading": chunk.section_heading,
                })

            # Upsert to Qdrant
            chunks_with_embeddings = list(zip(texts, embeddings))
            upsert_chunks(chunks_with_embeddings, metadata_payloads)
            reindexed_count += 1

        except Exception as e:
            errors.append({"document_id": doc.id, "error": str(e)})

    return {
        "reindexed": reindexed_count,
        "total_documents": len(docs),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "mock"),
        "embedding_dimension": provider.dimension,
        "errors": errors if errors else None,
    }
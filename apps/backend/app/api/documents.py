import uuid
import mimetypes
import hashlib
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.minio_client import get_minio_client, ensure_bucket_exists
from app.core.config import settings
from app.models.document import Document, DocumentVersion, DocumentChunk
from app.ingestion.pipeline import process_document, get_parser
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.services.embeddings import get_embedding_provider
from app.services.vector.qdrant_service import ensure_collection, upsert_chunks

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

def _ext_from_filename(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""

@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    max_size = settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    content = file.file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    if mime_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    ext = _ext_from_filename(file.filename)
    storage_key = f"{uuid.uuid4().hex}{ext}"
    
    ensure_bucket_exists()
    client = get_minio_client()
    client.put_object(
        settings.MINIO_BUCKET,
        storage_key,
        data=content,
        length=len(content),
        content_type=mime_type,
    )
    
    doc = Document(
        filename=storage_key,
        original_name=file.filename,
        mime_type=mime_type,
        size_bytes=len(content),
        status="completed",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    try:
        extracted_text = process_document(content, mime_type)
        extractor_type = type(get_parser(mime_type)).__name__ if get_parser(mime_type) else None
    except Exception:
        extracted_text = None
        extractor_type = None
    
    version = DocumentVersion(
        document_id=doc.id,
        version_number=1,
        storage_key=storage_key,
        extractor_type=extractor_type,
        extracted_text=extracted_text,
    )
    db.add(version)
    db.commit()
    
    return {
        "id": doc.id,
        "original_name": doc.original_name,
        "mime_type": doc.mime_type,
        "size_bytes": doc.size_bytes,
        "status": doc.status,
        "storage_key": storage_key,
        "extracted_text": extracted_text,
    }

@router.get("")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "original_name": d.original_name,
            "mime_type": d.mime_type,
            "size_bytes": d.size_bytes,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.post("/{document_id}/index")
def index_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get latest version
    latest_version = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == document_id
    ).order_by(DocumentVersion.version_number.desc()).first()
    
    if not latest_version or not latest_version.extracted_text:
        raise HTTPException(status_code=400, detail="No extracted text to index")
    
    # Chunk the text
    chunker = RecursiveChunker()
    metadata = {
        "source_file_name": doc.original_name,
        "title": doc.original_name,
    }
    chunks = chunker.chunk(latest_version.extracted_text, metadata)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated")
    
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
    
    return {
        "document_id": document_id,
        "version_id": latest_version.id,
        "chunks_created": len(chunks),
        "embedding_dimension": provider.dimension,
    }
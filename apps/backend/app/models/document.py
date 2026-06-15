from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    status = Column(String, default="pending")
    # Phase 13: visibility (private | shared | global) and ownership.
    visibility = Column(String(20), nullable=False, server_default="global")
    owner_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    # Owner relationship is resolved lazily via string reference to avoid circular imports
    # (app.security.models.User is imported elsewhere; SQLAlchemy resolves at mapper-config time).
    owner = relationship("User", foreign_keys=[owner_user_id], lazy="joined")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, default=1)
    storage_key = Column(String, nullable=False)
    extractor_type = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    document = relationship("Document", back_populates="versions")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document_version_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    title = Column(String, nullable=True)
    section_heading = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    source_file_name = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="chunks")

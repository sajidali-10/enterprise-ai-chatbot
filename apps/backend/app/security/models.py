"""
Security Models

SQLAlchemy models for authentication, authorization, and audit logging.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, ForeignKey, 
    Enum as SQLEnum, Text, func
)
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from app.db.base import Base


class UserRole(str, PyEnum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class AuditAction(str, PyEnum):
    CHAT_MESSAGE = "chat_message"
    CHAT_RAG = "chat_rag"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_INDEX = "document_index"
    DOCUMENT_DELETE = "document_delete"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"


class User(Base):
    """
    User model for authentication.
    
    In production, this would integrate with an external IdP (SSO/OIDC).
    For local development, supports dev user header bypass.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)  # NULL for SSO users
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_external = Column(Boolean, default=False, nullable=False)  # True for SSO users
    external_id = Column(String(255), nullable=True)  # SSO subject claim
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    permissions = relationship("DocumentPermission", back_populates="user", foreign_keys="DocumentPermission.user_id", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class DocumentPermission(Base):
    """
    Document-level permissions.
    
    Controls which users can access which documents.
    Admin role bypasses all permission checks.
    """
    __tablename__ = "document_permissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    can_read = Column(Boolean, default=True, nullable=False)
    can_write = Column(Boolean, default=False, nullable=False)
    granted_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="permissions", foreign_keys=[user_id])
    document = relationship("Document")
    grantor = relationship("User", foreign_keys=[granted_by])

    def __repr__(self):
        return f"<DocumentPermission(user_id={self.user_id}, document_id={self.document_id}, can_read={self.can_read})>"


class AuditLog(Base):
    """
    Audit log for tracking user actions.
    
    Records all significant events for compliance and security monitoring.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # NULL for anonymous/dev
    username = Column(String(100), nullable=False)
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    
    # Request details
    request_ip = Column(String(45), nullable=True)  # IPv6 max length
    request_user_agent = Column(String(500), nullable=True)
    
    # Action-specific details stored as JSON
    details = Column(Text, nullable=True)  # JSON string
    
    # RAG-specific fields
    question = Column(Text, nullable=True)
    retrieved_document_ids = Column(Text, nullable=True)  # JSON array of IDs
    retrieved_chunk_ids = Column(Text, nullable=True)  # JSON array of IDs
    
    # Response status
    status = Column(String(20), nullable=False, index=True)  # success, failure, error
    error_message = Column(Text, nullable=True)
    
    # Model/provider info (for RAG)
    model_provider = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    
    # Timing
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', user='{self.username}', status='{self.status}')>"
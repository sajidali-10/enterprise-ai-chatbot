"""
Audit Logging

Provides audit logging for compliance and security monitoring.
Records user actions including RAG queries, document access, and system events.
"""

import json
import time
from typing import Optional, Any
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.security.models import AuditLog, AuditAction
from app.security.auth import AuthContext
from app.db.session import SessionLocal


@dataclass
class AuditEvent:
    """
    Audit event data structure.
    
    Use this to build audit events before logging them.
    """
    action: AuditAction
    username: str
    user_id: Optional[int] = None
    request_ip: Optional[str] = None
    request_user_agent: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    
    # RAG-specific fields
    question: Optional[str] = None
    retrieved_document_ids: Optional[list[int]] = None
    retrieved_chunk_ids: Optional[list[int]] = None
    
    # Response status
    status: str = "success"
    error_message: Optional[str] = None
    
    # Model info
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    
    # Timing
    duration_ms: Optional[int] = None


class AuditLogger:
    """
    Audit logger for recording user actions.
    
    Usage:
        audit = AuditLogger()
        audit.log_chat_rag(
            auth=auth_context,
            question="What is the policy?",
            retrieved_doc_ids=[1, 2, 3],
            retrieved_chunk_ids=[10, 11, 12, 15],
            model_provider="openai",
            model_name="gpt-4",
            duration_ms=150,
        )
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def _create_log_entry(self, event: AuditEvent) -> AuditLog:
        """Create an AuditLog entry from an AuditEvent."""
        return AuditLog(
            user_id=event.user_id,
            username=event.username,
            action=event.action,
            request_ip=event.request_ip,
            request_user_agent=event.request_user_agent,
            details=json.dumps(event.details) if event.details else None,
            question=event.question,
            retrieved_document_ids=json.dumps(event.retrieved_document_ids) if event.retrieved_document_ids else None,
            retrieved_chunk_ids=json.dumps(event.retrieved_chunk_ids) if event.retrieved_chunk_ids else None,
            status=event.status,
            error_message=event.error_message,
            model_provider=event.model_provider,
            model_name=event.model_name,
            duration_ms=event.duration_ms,
        )
    
    def log(self, event: AuditEvent) -> AuditLog:
        """
        Log an audit event to the database.
        
        Returns the created AuditLog entry.
        """
        log_entry = self._create_log_entry(event)
        self.db.add(log_entry)
        self.db.commit()
        self.db.refresh(log_entry)
        return log_entry
    
    def log_chat_rag(
        self,
        auth: AuthContext,
        question: str,
        retrieved_document_ids: list[int],
        retrieved_chunk_ids: list[int],
        model_provider: str,
        model_name: str,
        status: str = "success",
        error_message: Optional[str] = None,
        request_ip: Optional[str] = None,
        request_user_agent: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ):
        """Log a RAG chat interaction."""
        event = AuditEvent(
            action=AuditAction.CHAT_RAG,
            username=auth.username,
            user_id=auth.user_id,
            question=question,
            retrieved_document_ids=retrieved_document_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            model_provider=model_provider,
            model_name=model_name,
            status=status,
            error_message=error_message,
            request_ip=request_ip,
            request_user_agent=request_user_agent,
            duration_ms=duration_ms,
        )
        return self.log(event)
    
    def log_chat_message(
        self,
        auth: AuthContext,
        status: str = "success",
        error_message: Optional[str] = None,
        request_ip: Optional[str] = None,
        request_user_agent: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ):
        """Log a non-RAG chat message."""
        event = AuditEvent(
            action=AuditAction.CHAT_MESSAGE,
            username=auth.username,
            user_id=auth.user_id,
            status=status,
            error_message=error_message,
            request_ip=request_ip,
            request_user_agent=request_user_agent,
            duration_ms=duration_ms,
        )
        return self.log(event)


def get_audit_logger() -> AuditLogger:
    """Get an audit logger with a new database session."""
    return AuditLogger(SessionLocal())


@contextmanager
def timed_audit_log(audit_logger: AuditLogger, event: AuditEvent):
    """
    Context manager for timing an operation and logging duration.
    
    Usage:
        audit = get_audit_logger()
        event = AuditEvent(action=AuditAction.CHAT_RAG, username="user", ...)
        with timed_audit_log(audit, event) as timed_event:
            # Do work
            pass
        # Duration is automatically calculated and logged
    """
    start_time = time.time()
    try:
        yield event
    except Exception as e:
        event.status = "error"
        event.error_message = str(e)
        raise
    finally:
        event.duration_ms = int((time.time() - start_time) * 1000)


def log_rag_query(
    auth: AuthContext,
    question: str,
    retrieved_document_ids: list[int],
    retrieved_chunk_ids: list[int],
    model_provider: str,
    model_name: str,
    status: str = "success",
    error_message: Optional[str] = None,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
    duration_ms: Optional[int] = None,
):
    """
    Convenience function to log a RAG query.
    
    Creates its own database session.
    """
    logger = get_audit_logger()
    try:
        return logger.log_chat_rag(
            auth=auth,
            question=question,
            retrieved_document_ids=retrieved_document_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            model_provider=model_provider,
            model_name=model_name,
            status=status,
            error_message=error_message,
            request_ip=request_ip,
            request_user_agent=request_user_agent,
            duration_ms=duration_ms,
        )
    finally:
        logger.db.close()
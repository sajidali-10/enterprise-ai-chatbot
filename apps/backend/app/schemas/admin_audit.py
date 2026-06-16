"""
Schemas for admin audit log API.
"""

from typing import Optional, Any, List
from pydantic import BaseModel


class AuditLogEntry(BaseModel):
    id: int
    user_id: Optional[int] = None
    username: str
    action: str
    request_ip: Optional[str] = None
    request_user_agent: Optional[str] = None
    details: Optional[Any] = None
    question: Optional[str] = None
    retrieved_document_ids: Optional[List[int]] = None
    status: str
    error_message: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: Optional[str] = None

    class ConfigDict:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    entries: List[AuditLogEntry]
    total: int
    page: int
    page_size: int
    total_pages: int

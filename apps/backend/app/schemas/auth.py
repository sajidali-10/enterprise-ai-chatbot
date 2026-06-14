"""
Authentication API schemas.
"""

from typing import Optional
from pydantic import BaseModel

from app.security.models import UserRole


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserPermissions(BaseModel):
    can_use_general_chat: bool
    can_use_knowledge_base: bool
    can_use_debug: bool
    can_view_documents: bool
    can_upload_documents: bool
    can_reindex_documents: bool
    can_delete_documents: bool
    can_access_observability: bool
    can_access_evaluations: bool
    can_submit_feedback: bool
    can_manage_users: bool


class UserInfo(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    permissions: UserPermissions

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserInfo

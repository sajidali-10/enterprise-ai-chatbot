"""
Admin user management schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.security.models import UserRole


class SafeUser(BaseModel):
    """User object safe to return to clients (no password/hash)."""

    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class SafeUserWithPermissions(SafeUser):
    """User object including effective role-derived permissions."""

    permissions: dict


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)
    full_name: Optional[str] = Field(None, max_length=255)
    role: UserRole = UserRole.USER
    is_active: bool = True

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = (v or "").strip()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v


class UpdateUserRequest(BaseModel):
    """Partial update of user fields (password is updated via reset-password)."""

    email: Optional[str] = Field(None, max_length=255)
    full_name: Optional[str] = Field(None, max_length=255)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=1, max_length=255)


class MessageResponse(BaseModel):
    success: bool
    detail: str

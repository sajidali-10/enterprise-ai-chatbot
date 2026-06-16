"""
Password hashing utilities using bcrypt via passlib.
Includes password policy validation for production security.
"""

import re
from typing import Optional

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Minimum password length
MIN_PASSWORD_LENGTH = 12

# Common weak passwords to reject (case-insensitive)
COMMON_WEAK_PASSWORDS = {
    "password123!",
    "password123",
    "password1",
    "password",
    "changeme",
    "admin123",
    "admin123!",
    "admin",
    "qwerty",
    "qwerty123",
    "letmein",
    "letmein123",
    "welcome1",
    "welcome123",
    "12345678",
    "123456789",
    "1234567890",
    "abc123",
    "monkey",
    "dragon",
    "baseball",
    "football",
    "iloveyou",
    " sunshine",
    "princess",
    " trustno1",
    " master",
    "access",
    " shadow",
    " michael",
    " superman",
    " jordan23",
    "harley",
}


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def validate_password_policy(password: str) -> Optional[str]:
    """
    Validate a password against the production security policy.

    Requirements:
    - Minimum 12 characters
    - At least 3 of 4 character classes: uppercase, lowercase, number, symbol
    - Not in the list of common weak passwords

    Returns:
        None if password is valid, otherwise a safe error message string.
        Error messages are intentionally generic to avoid helping attackers.
    """
    if not password:
        return "Password is required."

    if len(password) < MIN_PASSWORD_LENGTH:
        return (
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )

    # Count character classes
    has_upper = bool(re.search(r"[A-Z]", password))
    has_lower = bool(re.search(r"[a-z]", password))
    has_digit = bool(re.search(r"[0-9]", password))
    has_symbol = bool(re.search(r"[^A-Za-z0-9]", password))

    classes_met = sum([has_upper, has_lower, has_digit, has_symbol])
    if classes_met < 3:
        return (
            "Password must include at least 3 of the following: "
            "uppercase letters, lowercase letters, numbers, and symbols."
        )

    # Check against common weak passwords (case-insensitive)
    password_lower = password.lower()
    for weak in COMMON_WEAK_PASSWORDS:
        if password_lower == weak.lower():
            return "Password is too common or easily guessed. Please choose a stronger password."

    return None

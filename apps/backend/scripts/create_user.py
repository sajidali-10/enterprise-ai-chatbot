#!/usr/bin/env python3
"""
CLI helper for creating local users.

Usage:
    docker compose exec backend python /app/scripts/create_user.py \\
        --username alice --email alice@example.com --password '<password>' \\
        --role user

Roles: admin | user | viewer
Password is read from --password or from the CREATE_USER_PASSWORD env var
to avoid leaking secrets into shell history.

If the username or email already exists, the script exits non-zero.
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local user (admin only CLI helper).")
    parser.add_argument("--username", required=True, help="Unique username (1-100 chars)")
    parser.add_argument("--email", required=True, help="User email")
    parser.add_argument(
        "--password",
        default=os.environ.get("CREATE_USER_PASSWORD", ""),
        help="User password (or set CREATE_USER_PASSWORD env var). Min length 8.",
    )
    parser.add_argument(
        "--role",
        default="user",
        choices=["admin", "user", "viewer"],
        help="User role (default: user)",
    )
    parser.add_argument("--full-name", default=None, help="Optional full name")
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Create the user as inactive (default: active)",
    )
    args = parser.parse_args()

    if not args.password or len(args.password) < 8:
        print("ERROR: password is required and must be at least 8 characters.", file=sys.stderr)
        return 2

    # Imports inside main() so --help works without DB connection
    from app.db.session import SessionLocal
    from app.security.models import User, UserRole
    from app.security.password import hash_password

    db = SessionLocal()
    try:
        username = args.username.strip()
        email = args.email.strip().lower()

        if db.query(User).filter(User.username == username).first():
            print(f"ERROR: username '{username}' already exists.", file=sys.stderr)
            return 1
        if db.query(User).filter(User.email == email).first():
            print(f"ERROR: email '{email}' already exists.", file=sys.stderr)
            return 1

        user = User(
            username=username,
            email=email,
            full_name=args.full_name,
            hashed_password=hash_password(args.password),
            role=UserRole(args.role),
            is_active=not args.inactive,
            is_external=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Do not print the password or hash
        print(f"OK: created user id={user.id} username={user.username} role={user.role.value} active={user.is_active}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        db.rollback()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())

"""phase33_rbac_hardening

Revision ID: 011
Revises: 010
Create Date: 2026-07-10

Phase 33: RBAC hardening for enterprise security.

Changes:
1. Add SYSADMIN to userrole enum (highest privilege, protected role)
2. Add new audit actions: USER_DELETED, USER_REACTIVATED, PROTECTED_USER_ACTION_DENIED
3. Add is_protected column to users table (protects sysadmin from modification)
4. Add soft-delete columns: is_deleted, deleted_at, deleted_by
5. Promote bootstrap admin to sysadmin with is_protected=true
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Add SYSADMIN to userrole enum if not exists
    has_sysadmin = conn.execute(text("""
        SELECT 1 FROM pg_enum
        WHERE enumtypid = 'userrole'::regtype AND enumlabel = 'sysadmin'
    """)).fetchone()
    if not has_sysadmin:
        op.execute("ALTER TYPE userrole ADD VALUE 'sysadmin'")
        # Commit the transaction so the new enum value is visible for subsequent operations
        conn.commit()

    # 2. Add new audit action enum values if not exist
    new_audit_actions = [
        "USER_DELETED",
        "USER_REACTIVATED",
        "PROTECTED_USER_ACTION_DENIED",
    ]
    for value in new_audit_actions:
        exists = conn.execute(text("""
            SELECT 1 FROM pg_enum
            WHERE enumtypid = 'auditaction'::regtype AND enumlabel = :val
        """), {"val": value}).fetchone()
        if not exists:
            op.execute(f"ALTER TYPE auditaction ADD VALUE '{value}'")

    # 3. Add is_protected column to users table if not exists
    has_is_protected = conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'is_protected'
    """)).fetchone()
    if not has_is_protected:
        op.add_column(
            "users",
            sa.Column("is_protected", sa.Boolean(), nullable=False, server_default=sa.false())
        )

    # 4. Add soft-delete columns if not exist
    has_is_deleted = conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'is_deleted'
    """)).fetchone()
    if not has_is_deleted:
        op.add_column(
            "users",
            sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.false(), index=True)
        )

    has_deleted_at = conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'deleted_at'
    """)).fetchone()
    if not has_deleted_at:
        op.add_column(
            "users",
            sa.Column("deleted_at", sa.DateTime(), nullable=True)
        )

    has_deleted_by = conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'deleted_by'
    """)).fetchone()
    if not has_deleted_by:
        op.add_column(
            "users",
            sa.Column("deleted_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True)
        )

    # 5. Promote bootstrap admin to sysadmin with is_protected=true
    # Get bootstrap admin username from settings (default 'admin')
    # This runs after all schema changes are committed
    conn.commit()
    
    # Check if bootstrap admin exists and needs promotion
    bootstrap_admin = conn.execute(text("""
        SELECT id, username, role, is_protected FROM users
        WHERE username = 'admin' AND is_deleted = false
    """)).fetchone()
    
    if bootstrap_admin:
        user_id, username, current_role, is_protected = bootstrap_admin
        needs_promotion = (str(current_role).lower() != 'sysadmin') or (not is_protected)
        
        if needs_promotion:
            # Promote to sysadmin and set is_protected=true
            conn.execute(text("""
                UPDATE users
                SET role = 'sysadmin'::userrole,
                    is_protected = true
                WHERE id = :user_id
            """), {"user_id": user_id})
            conn.commit()


def downgrade() -> None:
    conn = op.get_bind()

    # Drop index first if it exists
    has_index = conn.execute(text("""
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'users' AND indexname = 'ix_users_is_deleted'
    """)).fetchone()
    if has_index:
        op.drop_index("ix_users_is_deleted", "users")

    # Drop columns in reverse order of addition (only if they exist)
    for col in ["deleted_by", "deleted_at", "is_deleted", "is_protected"]:
        has_col = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = :col
        """), {"col": col}).fetchone()
        if has_col:
            op.drop_column("users", col)

    # Note: PostgreSQL does not support removing enum values.
    # The downgrade for enum values is a no-op.
    pass

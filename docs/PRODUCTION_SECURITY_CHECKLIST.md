# Production Security Checklist

This document provides a comprehensive security checklist for deploying the Enterprise AI Chatbot in production.

---

## Table of Contents

1. [Required Environment Variables](#required-environment-variables)
2. [Secret Generation & Rotation](#secret-generation--rotation)
3. [Admin Password Setup & Reset](#admin-password-setup--reset)
4. [Docker & Deployment Hardening](#docker--deployment-hardening)
5. [Network & Firewall Rules](#network--firewall-rules)
6. [Backup & Recovery](#backup--recovery)
7. [Password Policy](#password-policy)
8. [Security Headers](#security-headers)
9. [Rate Limiting](#rate-limiting)
10. [Audit Logging](#audit-logging)
11. [Operational Security](#operational-security)

---

## Required Environment Variables

### Critical (Application will fail to start if missing)

| Variable | Description | Example |
|----------|-------------|---------|
| `AUTH_MODE` | Authentication mode: `dev`, `local`, `oidc` | `local` |
| `JWT_SECRET_KEY` | Strong random secret for JWT signing (≥48 chars base64) | *(generate with openssl)* |
| `POSTGRES_HOST` | PostgreSQL hostname | `postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | PostgreSQL database name | `chatbot` |
| `POSTGRES_USER` | PostgreSQL username | `chatbot` |
| `POSTGRES_PASSWORD` | Strong random PostgreSQL password | *(generate)* |
| `MINIO_ROOT_USER` | MinIO admin username | *(change from default)* |
| `MINIO_ROOT_PASSWORD` | Strong random MinIO admin password | *(generate)* |

### Required for LLM Provider

| Variable | Required When | Description |
|----------|--------------|-------------|
| `OPENROUTER_API_KEY` | `LLM_PROVIDER=openrouter` | API key from https://openrouter.ai/keys |
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` | API key from https://platform.openai.com/ |

### Required for Bootstrap (first admin creation)

| Variable | Description | Security Note |
|----------|-------------|---------------|
| `BOOTSTRAP_ADMIN_EMAIL` | Admin email address | Must be valid email |
| `BOOTSTRAP_ADMIN_USERNAME` | Admin username | `admin` (recommended: change after login) |
| `BOOTSTRAP_ADMIN_PASSWORD` | Bootstrap admin password | **Must be changed on first login** |

### Required for Production Security Features

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ALLOWED_ORIGINS` | Comma-separated allowed frontend origins | *(empty = restrictive)* |
| `RATE_LIMIT_ENABLED` | Enable API rate limiting | `true` |
| `UPLOAD_MAX_SIZE_MB` | Maximum file upload size in MB | `10` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiry in minutes | `60` |

### Development Only (never use in production)

| Variable | Description |
|----------|-------------|
| `DEV_AUTH_ENABLED` | Enable dev header authentication |
| `DEV_USER_SECRET` | Secret for dev header bypass |

---

## Secret Generation & Rotation

### Generate a Secure JWT Secret

```bash
# Generate a 64-byte base64-encoded secret (recommended)
openssl rand -base64 64

# Example output:
# abc123...xyz789 (≈ 88 characters)
```

### Generate Strong Passwords

```bash
# PostgreSQL password
openssl rand -base64 24

# MinIO password
openssl rand -base64 24

# Bootstrap admin password (must meet password policy)
# See [Password Policy](#password-policy) section below
```

### Secret Rotation Procedure

**JWT Secret:**
1. Schedule maintenance window (all users will be logged out)
2. Generate new JWT secret
3. Update `.env` with new `JWT_SECRET_KEY`
4. Restart backend containers: `docker compose restart backend`
5. Notify users to re-login

**Database Password:**
1. Generate new password
2. Stop backend: `docker compose stop backend`
3. Update `.env` with new `POSTGRES_PASSWORD`
4. Change password in PostgreSQL (via psql or admin interface)
5. Restart: `docker compose up -d backend`

**MinIO Password:**
1. Log into MinIO console (port 9001)
2. Change root password via UI
3. Update `.env` with new `MINIO_ROOT_PASSWORD`
4. Restart backend containers

---

## Admin Password Setup & Reset

### First-Time Bootstrap Admin Setup

1. Set `BOOTSTRAP_ADMIN_PASSWORD` in `.env` to a strong temporary password
2. Start the application: `docker compose up -d`
3. Log in as admin with the bootstrap password
4. **Immediately change the password via Admin → Users → Reset Password**
5. The password reset increments `token_version`, invalidating the bootstrap session
6. Clear `BOOTSTRAP_ADMIN_PASSWORD` from `.env` and restart

### Reset Admin Password (locked out)

1. Access the backend container:
   ```bash
   docker compose exec backend /bin/bash
   ```
2. Run the password reset script (or use a one-off Python script):
   ```python
   from app.db.session import SessionLocal
   from app.security.models import User, UserRole
   from app.security.password import hash_password
   
   db = SessionLocal()
   user = db.query(User).filter(User.role == UserRole.ADMIN).first()
   user.hashed_password = hash_password("NewStrongP@ssw0rd!")
   user.token_version += 1  # Invalidate all existing sessions
   db.commit()
   ```
3. Log in with the new password

### Enforcing Strong Admin Passwords

Admin passwords must meet the following policy:
- **Minimum length:** 12 characters
- **Character classes:** At least 3 of:
  - Uppercase letter (A–Z)
  - Lowercase letter (a–z)
  - Number (0–9)
  - Special symbol (!@#$%^&* etc.)
- **Prohibited values:** Common weak passwords are rejected
  - `Password123!`
  - `ChangeMe`
  - `admin123`
  - `qwerty`
  - `Password1`
  - `12345678`
  - `letmein`
  - `welcome1`

---

## Docker & Deployment Hardening

### Non-Root Containers

The backend and frontend containers should run as non-root users:

**Backend Container:**
- App runs as `appuser` (UID 1000)
- No root access inside container
- Files owned by `appuser:appuser`

**Frontend Container:**
- App runs as `nextjs` user (UID 1001)
- Static files owned by `nextjs` user

### Minimize Image Attack Surface

```dockerfile
# Backend Dockerfile — use multi-stage build
FROM python:3.11-slim as builder
# ... install build dependencies ...

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 appuser
USER appuser
WORKDIR /home/appuser
COPY --from=builder --chown=appuser:appuser /install /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
```

### Image Scanning (recommended)

```bash
# Scan backend image
docker scan chatbot-backend:latest

# Or use Trivy
trivy image chatbot-backend:latest
```

---

## Network & Firewall Rules

### Recommended Port Exposure

| Port | Service | Production Exposure |
|------|---------|---------------------|
| 3000 | Next.js Frontend | **Public** (or via reverse proxy) |
| 8000 | FastAPI Backend | **Internal only** (frontend → backend) |
| 5432 | PostgreSQL | **Private subnet only** |
| 6379 | Redis | **Private subnet only** |
| 9000 | MinIO API | **Private subnet only** |
| 9001 | MinIO Console | **Private subnet only** (or VPN-restricted) |
| 6333 | Qdrant | **Private subnet only** |

### Security Group / Firewall Rules

```
# ALLOW inbound
TCP 80   from 0.0.0.0/0   → Reverse proxy (Nginx/Traefik)
TCP 443  from 0.0.0.0/0   → HTTPS reverse proxy

# DENY direct public access to backend and services
TCP 8000 from 0.0.0.0/0   → DENY
TCP 5432 from 0.0.0.0/0   → DENY
TCP 6379 from 0.0.0.0/0   → DENY
TCP 9000 from 0.0.0.0/0   → DENY
TCP 9001 from 0.0.0.0/0   → DENY
TCP 6333 from 0.0.0.0/0   → DENY
```

### Reverse Proxy Recommendations

Use a reverse proxy (Nginx, Traefik, or AWS ALB) in front of the frontend:
- Terminate TLS/SSL at the proxy
- Rate limit at the edge
- Block direct access to backend port 8000
- Add Web Application Firewall (WAF) rules

---

## Backup & Recovery

### PostgreSQL Backup

**Backup:**
```bash
# Full database dump
docker compose exec postgres pg_dump \
  -U $POSTGRES_USER \
  -d $POSTGRES_DB \
  --format=custom \
  > /backups/chatbot_$(date +%Y%m%d_%H%M%S).dump
```

**Restore:**
```bash
# Restore from backup
docker compose exec -T postgres pg_restore \
  -U $POSTGRES_USER \
  -d $POSTGRES_DB \
  --clean --if-exists \
  < /backups/chatbot_20240115_120000.dump
```

**Recommended frequency:** Daily automated backups, retain 14 days

### MinIO Documents Backup

**Backup:**
```bash
# Mirror MinIO bucket to local backup
docker run --rm --network chatbot-network \
  -v /backups/minio:/backup \
  minio/mc \
  mirror chatbot_minio/chatbot-uploads /backup
```

**Restore:**
```bash
# Restore from backup
docker run --rm --network chatbot-network \
  -v /backups/minio:/backup \
  -e MC_HOST_backup=http://minioadmin:minioadmin@minio:9000 \
  minio/mc \
  mirror /backup backup/chatbot-uploads
```

**Recommended frequency:** Daily incremental, weekly full sync

### Qdrant Vector Store Backup

Qdrant vectors can be rebuilt from PostgreSQL document data:
1. Backup Qdrant storage directory: `/var/lib/qdrant/storage` (or volume)
2. **Recovery strategy:** Rebuild from source documents
   ```bash
   # Trigger reindex of all documents via API
   curl -X POST http://localhost:8000/api/documents/reindex-all \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

**Recommended frequency:** Weekly snapshot, or rely on rebuild from source

### Environment & Secrets Backup

- `.env` file: Store in env-encrypted vault (e.g., AWS Secrets Manager, HashiCorp Vault, 1Password)
- Never commit secrets to Git
- Keep an offline encrypted backup of all secrets
- Document secret rotation dates

### Backup Validation Procedure

After any restore:
1. Verify PostgreSQL: `SELECT COUNT(*) FROM users;`
2. Verify MinIO: Check document count matches expected
3. Verify Qdrant: Run a RAG query and confirm results
4. Verify auth: Log in with existing user credentials
5. Run full test suite

---

## Password Policy

All user passwords (including admin, regular users, and bootstrap) must meet:

| Requirement | Minimum |
|-------------|---------|
| Length | 12 characters |
| Character classes | 3 of 4 (uppercase, lowercase, number, symbol) |
| Common passwords | Rejected (see prohibited list) |

**Password reset mandatory on:**
- First admin login (bootstrap password)
- Admin-initiated password reset
- Any security incident response

---

## Security Headers

The backend automatically adds these HTTP security headers to all responses:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME type sniffing |
| `X-Frame-Options` | `DENY` | Prevents clickjacking |
| `Referrer-Policy` | `no-referrer` | Limits referrer info leakage |
| `Permissions-Policy` | Restrictive defaults | Limits browser feature access |
| `Content-Security-Policy` | Baseline for Next.js | Prevents XSS injection |
| `Cache-Control` | `no-store` for auth APIs | Prevents sensitive data caching |

Verify headers are present:
```bash
curl -I http://localhost:8000/health
```

---

## Rate Limiting

Rate limits protect against brute-force and abuse:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `POST /api/auth/login` | 5 attempts | 60 seconds |
| `POST /api/documents/upload` | 10 uploads | 60 seconds |
| `POST /api/chat` | 30 requests | 60 seconds |
| `POST /api/admin/users*` | 20 requests | 60 seconds |

**Configuration:**
- Redis-backed by default (requires Redis service running)
- In-memory fallback if Redis unavailable
- Disable in tests: `RATE_LIMIT_ENABLED=false`

Verify rate limiting:
```bash
# Repeated login attempts should return 429
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username_or_email":"test","password":"wrong"}'
```

---

## Audit Logging

All security-relevant events are logged to the `audit_logs` table:

**Logged Events:**
- Login success / failure
- Logout
- Password reset
- User created / updated / deactivated / reactivated
- Role changed
- Document uploaded / deleted / reindexed
- Document access changed
- Admin API access denied

**Audit Fields:**
- `actor_user_id`, `actor_username`
- `action`, `target_type`, `target_id`
- `timestamp`, `request_ip`, `user_agent`
- `result` (success/failure), `reason`

**Never Logged:**
- Passwords or password hashes
- JWT tokens
- OpenRouter / OpenAI API keys
- Raw document contents
- Full RAG prompts

Admin audit log viewer available at: **Admin → Audit Logs**

---

## Operational Security

### Pre-Deployment Checklist

- [ ] `.env` created from `.env.example` with all secrets generated
- [ ] `JWT_SECRET_KEY` is strong (≥48 chars, random)
- [ ] `POSTGRES_PASSWORD` is strong and not a placeholder
- [ ] `MINIO_ROOT_PASSWORD` is strong and not a placeholder
- [ ] `BOOTSTRAP_ADMIN_PASSWORD` is strong (changed after first login)
- [ ] `CORS_ALLOWED_ORIGINS` is set to your domain(s)
- [ ] `RATE_LIMIT_ENABLED=true`
- [ ] Docker containers run as non-root
- [ ] Only ports 80/443 exposed publicly (reverse proxy)
- [ ] Database, Redis, MinIO, Qdrant on private subnet
- [ ] TLS/SSL certificates configured
- [ ] Backup schedule configured and tested
- [ ] Audit logging enabled
- [ ] `.env` backed up to secure vault

### Incident Response

**Suspected credential compromise:**
1. Change `JWT_SECRET_KEY` (logs out all users)
2. Reset admin password via database
3. Review audit logs for unauthorized access
4. Rotate all service passwords (PostgreSQL, MinIO)
5. Notify affected users

**Unauthorized admin access:**
1. Check audit logs for recent admin actions
2. Revert suspicious changes
3. Disable compromised user account
4. Force password reset for all admin accounts

### Monitoring Recommendations

- Monitor for 429 responses (rate limit triggered)
- Alert on failed login spikes
- Alert on admin API access denied events
- Monitor disk usage for uploads and Qdrant growth
- Set up log aggregation (ELK, Datadog, etc.)

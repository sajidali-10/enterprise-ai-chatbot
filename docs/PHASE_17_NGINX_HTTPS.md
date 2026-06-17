# Phase 17 — Nginx HTTPS Reverse Proxy

## Architecture

```
Browser
  └─ HTTPS 443 ──► Nginx (:443, reverse proxy)
                        ├─ /health       ──► backend:8000/health
                        ├─ /api/*        ──► backend:8000/api/*
                        └─ /*            ──► frontend:3000
                   HTTP 80  ──► Nginx (:80)
                        └─ 301 redirect  ──► HTTPS
```

**All backend/frontend services are internal to the Docker network.** Only ports 80 and 443 are exposed to the host.

## Files Added/Changed

| File | Change |
|------|--------|
| `infra/nginx/nginx.conf` | Main Nginx configuration |
| `infra/nginx/conf.d/chatbot.conf` | Routing and SSL configuration |
| `infra/nginx/ssl/README.md` | Certificate placement instructions |
| `infra/nginx/ssl/private.key` | **Do not commit** — your private key |
| `infra/nginx/ssl/fullchain.crt` | Your SSL certificate chain |
| `docker-compose.yml` | Added `nginx` service, removed public ports from `frontend`/`backend` |
| `.env` | Updated `CORS_ALLOWED_ORIGINS`, added `CHATBOT_DOMAIN`, `NEXT_PUBLIC_API_BASE_URL` |
| `.env.example` | Documented new variables |

## Certificate Placement

1. Copy your certificate files to:
   ```
   infra/nginx/ssl/fullchain.crt   # Your full certificate chain
   infra/nginx/ssl/private.key     # Your unencrypted private key
   ```

2. Set correct permissions:
   ```bash
   chmod 600 infra/nginx/ssl/private.key
   chmod 644 infra/nginx/ssl/fullchain.crt
   chown root:root infra/nginx/ssl/private.key infra/nginx/ssl/fullchain.crt
   ```

3. Replace the self-signed certificate (for local testing only):
   ```bash
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout infra/nginx/ssl/private.key \
     -out infra/nginx/ssl/fullchain.crt \
     -subj "/CN=your-domain.com"
   ```

## Environment Variables

Update `.env` before starting nginx:

```bash
# Required: your registered domain
CHATBOT_DOMAIN=chatbot.example.com

# Backend CORS — must match your domain
CORS_ALLOWED_ORIGINS=https://chatbot.example.com

# Frontend API URL — same-origin through nginx
NEXT_PUBLIC_API_BASE_URL=https://${CHATET_DOMAIN}
```

## Deployment Commands

```bash
# 1. Validate docker-compose configuration
docker compose config

# 2. Start all services (nginx routes traffic to frontend/backend)
docker compose up -d

# 3. Verify all services are running
docker compose ps

# 4. Check nginx logs
docker compose logs --tail=100 nginx

# 5. Test HTTP → HTTPS redirect
curl -I http://localhost

# 6. Test HTTPS
curl -Ik https://localhost
curl -Ik https://localhost/health

# 7. Test backend health through nginx
curl -k https://localhost/api/health/provider
```

## Nginx Routing

| Path | Destination | Notes |
|------|-------------|-------|
| `/health` | `backend:8000/health` | No API prefix needed |
| `/api/*` | `backend:8000/api/*` | Proxied as-is |
| `/*` | `frontend:3000` | All other traffic |

## AWS Security Group Recommendations

After validating the nginx setup, update your EC2 Security Group:

| Port | Source | Purpose |
|------|--------|---------|
| 22 | `YOUR_ADMIN_IP/32` | SSH (restrict to your IP) |
| 80 | `0.0.0.0/0` | HTTP → HTTPS redirect |
| 443 | `0.0.0.0/0` | HTTPS (Nginx) |
| ~~3000~~ | **Remove** | Frontend (now internal) |
| ~~8000~~ | **Remove** | Backend (now internal) |
| 5432 | **Remove** | PostgreSQL (internal only) |
| 6379 | **Remove** | Redis (internal only) |
| 6333 | **Remove** | Qdrant (internal only) |
| 9000, 9001 | **Remove** | MinIO (internal only) |

## Certificate Renewal

1. Stop nginx:
   ```bash
   docker compose stop nginx
   ```

2. Replace `infra/nginx/ssl/fullchain.crt` and/or `infra/nginx/ssl/private.key`

3. Restart nginx:
   ```bash
   docker compose up -d nginx
   ```

4. Verify:
   ```bash
   curl -Ik https://your-domain.com/health
   ```

## Rollback

If nginx fails and you need to temporarily restore direct access:

```bash
# Stop nginx
docker compose stop nginx

# Restore public ports on frontend/backend by editing docker-compose.yml
# and uncommenting the ports: lines for frontend and backend

# Restart frontend and backend directly
docker compose up -d --force-recreate frontend backend

# Restore previous NEXT_PUBLIC_API_URL in .env if needed
# e.g., NEXT_PUBLIC_API_URL=http://YOUR_IP:8000
```

## Troubleshooting

**502 Bad Gateway**
- Check that `frontend` and `backend` containers are running: `docker compose ps`
- Check nginx logs: `docker compose logs nginx`
- Verify container names match your docker-compose service names

**SSL certificate errors**
- Verify `fullchain.crt` contains the full chain (server cert + intermediates)
- Verify `private.key` is the correct unencrypted key for the certificate
- Check file permissions: key should be 600, cert should be 644

**Too many redirects**
- Check that `CORS_ALLOWED_ORIGINS` matches exactly (including https://)
- Ensure your browser isn't caching old redirects

**Frontend can't reach API**
- Verify `NEXT_PUBLIC_API_BASE_URL=https://your-domain.com` in `.env`
- The frontend should use same-origin `/api/*` calls, not direct backend IP

**CORS errors**
- Backend's `CORS_ALLOWED_ORIGINS` must include the frontend's origin
- Example: `CORS_ALLOWED_ORIGINS=https://chatbot.example.com`

## Security Notes

- Nginx runs as a separate container (not inside backend/frontend)
- TLS 1.2 and 1.3 only; weak ciphers disabled
- Security headers set: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`
- No private keys or certificates are committed to version control
- Backend/frontend ports are internal to Docker network
- `LITELLM_ENABLED=false` and `LLM_PROVIDER=openrouter` remain unchanged
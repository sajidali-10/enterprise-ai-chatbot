# SSL Certificates

Place your SSL certificate and private key files here before starting the nginx container.

## Required Files

| File | Description |
|------|-------------|
| `fullchain.crt` | Your full certificate chain (server cert + intermediate CA(s)) |
| `private.key` | Your unencrypted private key |

## Permissions

```bash
chmod 600 infra/nginx/ssl/private.key
chmod 644 infra/nginx/ssl/fullchain.crt
chown root:root infra/nginx/ssl/private.key infra/nginx/ssl/fullchain.crt
```

## Certificate Replacement

1. Stop the nginx container:
   ```bash
   docker compose stop nginx
   ```

2. Replace the certificate and/or key files in `infra/nginx/ssl/`

3. Restart nginx:
   ```bash
   docker compose up -d nginx
   ```

4. Verify:
   ```bash
   curl -Ik https://your-domain.com/health
   ```

## Generating a Self-Signed Certificate (for local testing only)

```bash
# Generate private key and self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infra/nginx/ssl/private.key \
  -out infra/nginx/ssl/fullchain.crt \
  -subj "/CN=localhost"
```

## Notes

- **Never commit private keys or real certificates to version control**
- The `.gitignore` in this directory blocks all `*.key`, `*.crt`, and `*.pem` files
- Use a trusted CA (Let's Encrypt, DigiCert, etc.) for production
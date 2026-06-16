"""
Security Headers Middleware

Adds HTTP security headers to all responses for production hardening.
Designed to protect against common web vulnerabilities without breaking
a Next.js frontend application.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects security headers into every response.

    Headers set:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - Referrer-Policy: no-referrer
    - Permissions-Policy: restrictive defaults
    - Content-Security-Policy: baseline for Next.js SPA
    - Cache-Control: no-store for auth endpoints
    """

    def __init__(self, app, **options):
        super().__init__(app, **options)
        self._base_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            ),
            # Baseline CSP suitable for a Next.js SPA.
            # Blocks inline scripts/styles and only allows resources from
            # same-origin and the API backend. Adjust as needed.
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "media-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            ),
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Add base security headers
        for header, value in self._base_headers.items():
            if header not in response.headers:
                response.headers[header] = value

        # Cache-Control for sensitive authenticated endpoints
        path = request.url.path
        if path.startswith("/api/auth") or path.startswith("/api/admin"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

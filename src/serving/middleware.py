"""Middleware for logging, timing, and error handling."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header and log request duration."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "[%s] %s %s — %d — %.2fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log incoming request details."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        logger.debug(
            "Incoming: %s %s from %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        return await call_next(request)

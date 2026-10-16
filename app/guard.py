"""The local access guard: Host and Origin checks on every request, a per-install token on /api/."""

from __future__ import annotations

import hmac
import secrets
from pathlib import Path

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

TOKEN_FILE = "access_token"
TOKEN_HEADER = "x-loci-token"
OPEN_API_PATH = "/api/health"


def load_token(data_dir: Path, env_token: str | None) -> tuple[str, str]:
    """(token, warning): the environment token, else the one in data_dir, else a new one written there.

    The warning is "" unless a new token could not be written; the token then lives in memory only.
    """
    if env_token:
        return env_token, ""
    path = data_dir / TOKEN_FILE
    try:
        stored = path.read_text(encoding="utf-8").strip()
    except OSError:
        stored = ""
    if stored:
        return stored, ""
    token = secrets.token_urlsafe(32)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(token, encoding="utf-8")
    except OSError as exc:
        warning = f"the access token could not be written to {path}: {exc}; open tabs stop working at the next restart"
        return token, warning
    return token, ""


class AccessGuard:
    """Pure ASGI middleware: Host (421), Origin (403), then the token on /api/ except /api/health (401)."""

    def __init__(self, app: ASGIApp, token: str, port: int) -> None:
        self.app = app
        self.token = token
        self.port = port
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{host}" for host in self.allowed_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in scope["headers"]}
        address = f"http://127.0.0.1:{self.port}"
        refusal = None
        if headers.get("host") not in self.allowed_hosts:
            refusal = JSONResponse({"error": f"this page is served at {address}", "fix": f"open {address}"}, 421)
        elif "origin" in headers and headers["origin"] not in self.allowed_origins:
            refusal = JSONResponse(
                {"error": "request from another origin", "fix": f"open the page from {address}"}, 403
            )
        else:
            path = scope["path"]
            if path.startswith("/api/") and path != OPEN_API_PATH:
                given = headers.get(TOKEN_HEADER, "").encode("utf-8")
                if not hmac.compare_digest(given, self.token.encode("utf-8")):
                    refusal = JSONResponse({"error": "no access token", "fix": "reload the page"}, 401)
        if refusal is not None:
            await refusal(scope, receive, send)
            return
        await self.app(scope, receive, send)

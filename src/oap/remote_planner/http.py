"""Loopback-only HTTP host for an SSH-forwarded planner service."""
from __future__ import annotations

import ipaddress
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .protocol import PlanningProtocolError, canonical_json
from .service import PlannerService


def make_http_server(
    service: PlannerService,
    *,
    host: str = "127.0.0.1",
    port: int,
    max_request_bytes: int = 8 << 20,
) -> ThreadingHTTPServer:
    """Build a loopback-only server; SSH supplies transport authentication."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("planner HTTP host must be a literal loopback IP") from exc
    if not address.is_loopback:
        raise ValueError(
            "planner HTTP service must bind loopback and be reached via SSH"
        )
    request_limit = int(max_request_bytes)
    if request_limit < 1:
        raise ValueError("max_request_bytes must be >= 1")

    class Handler(BaseHTTPRequestHandler):
        server_version = "OaPRemotePlanner/1"

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path != "/v1/plan":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if length < 1 or length > request_limit:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise PlanningProtocolError(
                        "planning request must be a JSON object"
                    )
                response = service.handle(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json_error(
                    HTTPStatus.BAD_REQUEST, "malformed_json"
                )
                return
            except PlanningProtocolError:
                # Identity, replay, and deadline failures are intentionally
                # indistinguishable to a remote caller and never include knots.
                self._json_error(
                    HTTPStatus.CONFLICT, "planning_request_rejected"
                )
                return
            encoded = canonical_json(response).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path == "/v1/info":
                payload = service.info()
            elif self.path == "/v1/health":
                payload = service.health()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            encoded = canonical_json(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Deployment logging is owned by the Slurm/service wrapper.
            del fmt, args

        def _json_error(self, status: HTTPStatus, code: str) -> None:
            encoded = canonical_json({"error": code}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return ThreadingHTTPServer((host, int(port)), Handler)

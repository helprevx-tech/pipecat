# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Versioned gateway identity, validation, and bounded WebSocket lifecycle."""

import asyncio
import json
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from websockets.protocol import State

from pipecat.utils.errors import ErrorCategory

PROTOCOL_VERSION = 1
MAX_MESSAGE_SIZE = 2 * 1024 * 1024
PERMANENT_CODES = frozenset(
    {
        "unauthorized",
        "authentication_failed",
        "invalid_token",
        "invocation_expired",
        "forbidden",
        "model_forbidden",
        "invalid_configuration",
        "invalid_request",
        "quota_exceeded",
        "budget_exceeded",
        "unsupported_model",
        "unsupported_setting",
    }
)
TRANSIENT_CODES = frozenset({"rate_limited", "provider_unavailable", "gateway_unavailable"})


class GatewayError(Exception):
    """Sanitized gateway failure; untrusted provider text is never logged."""

    def __init__(self, code: str, *, retryable: bool = False):
        """Create a sanitized error from a recognized protocol code."""
        self.code = (
            code
            if isinstance(code, str) and code in PERMANENT_CODES | TRANSIENT_CODES
            else "protocol_error"
        )
        self.permanent = self.code in PERMANENT_CODES or self.code == "protocol_error"
        self.retryable = retryable and not self.permanent
        super().__init__(f"AayiqAgent model gateway: {self.code}")

    @property
    def category(self):
        """Return the framework category without exposing provider response bodies."""
        if self.code in {
            "unauthorized",
            "authentication_failed",
            "invalid_token",
            "invocation_expired",
        }:
            return ErrorCategory.AUTHENTICATION
        if self.code in {"forbidden", "model_forbidden"}:
            return ErrorCategory.AUTHORIZATION
        if self.code in {"quota_exceeded", "budget_exceeded"}:
            return ErrorCategory.QUOTA
        if self.code == "rate_limited":
            return ErrorCategory.RATE_LIMIT
        if self.permanent:
            return ErrorCategory.INVALID_REQUEST
        return ErrorCategory.CONNECTIVITY

    @classmethod
    def from_message(cls, message: dict):
        """Parse only known error codes, ignoring untrusted diagnostic content."""
        return cls(
            message.get("code", "protocol_error"), retryable=message.get("retryable") is True
        )


def required_text(value: str, name: str) -> str:
    """Validate nonempty identifiers and prevent header/control-character injection."""
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be a nonempty string of at most 4096 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True)
class GatewayIdentity:
    """Operator endpoint and scoped invocation credential available before setup."""

    gateway_base_url: str
    invocation_token: str = field(repr=False)
    request_id: str
    allow_insecure: bool = False

    def __post_init__(self):
        required_text(self.invocation_token, "invocation_token")
        required_text(self.request_id, "request_id")
        raw = required_text(self.gateway_base_url, "gateway_base_url")
        parsed = urlsplit(raw)
        if parsed.scheme not in {"https", "wss", "http", "ws"} or not parsed.hostname:
            raise ValueError("gateway_base_url must be an absolute HTTP(S) or WS(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("gateway_base_url must not contain credentials, query, or fragment")
        if parsed.scheme in {"http", "ws"} and not self.allow_insecure:
            raise ValueError("Plaintext gateway requires explicit allow_insecure=True")
        if parsed.path.rstrip("/") not in {"", "/api/v1"}:
            raise ValueError("gateway_base_url path must be empty or /api/v1")
        # Access validates malformed/non-numeric ports without echoing credentials.
        try:
            _ = parsed.port
        except ValueError:
            raise ValueError("gateway_base_url has an invalid port") from None

    def url(self, route: str, *, websocket: bool = False) -> str:
        """Join a fixed protocol route without duplicating its version prefix."""
        parsed = urlsplit(self.gateway_base_url)
        secure = parsed.scheme in {"https", "wss"}
        scheme = ("wss" if secure else "ws") if websocket else ("https" if secure else "http")
        return urlunsplit((scheme, parsed.netloc, "/api/v1/" + route.lstrip("/"), "", ""))

    def headers(self) -> dict[str, str]:
        """Build handshake headers; credentials never enter the request URL."""
        return {
            "Authorization": f"Bearer {self.invocation_token}",
            "X-AayiqAgent-Request-Id": self.request_id,
            "X-AayiqAgent-Protocol-Version": str(PROTOCOL_VERSION),
        }

    def attribution(self) -> dict:
        """Identify one metered request without selecting its organization."""
        return {"request_id": self.request_id, "protocol_version": PROTOCOL_VERSION}


def decode_message(raw: str | bytes) -> dict:
    """Reject malformed or oversized protocol messages without echoing their contents."""
    if not isinstance(raw, str) or len(raw) > MAX_MESSAGE_SIZE:
        raise GatewayError("protocol_error")
    try:
        message = json.loads(raw)
    except (ValueError, TypeError):
        raise GatewayError("protocol_error") from None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise GatewayError("protocol_error")
    return message


def connection_error(exception: Exception) -> GatewayError:
    """Classify handshake status without exposing response bodies or authorization."""
    if isinstance(exception, GatewayError):
        return exception
    status = getattr(getattr(exception, "response", None), "status_code", None)
    if status in {401, 403}:
        return GatewayError("unauthorized")
    if status in {400, 404, 422}:
        return GatewayError("invalid_configuration")
    return GatewayError("gateway_unavailable", retryable=True)


def validate_pcm(audio: bytes):
    """Validate bounded mono PCM16 audio before sending it over the gateway."""
    if not isinstance(audio, bytes) or len(audio) > MAX_MESSAGE_SIZE or len(audio) % 2:
        raise ValueError("Audio must contain whole signed 16-bit mono PCM samples within 2 MiB")


class GatewayWebsocketMixin:
    """Shared transport lifecycle; subclasses supply config and event processing.

    The first protocol message must confirm readiness. No receiver races that
    handshake, and no audio or silent keepalive is accepted before it succeeds.
    """

    _ready_type = "ready"

    def _init_gateway(self, identity: GatewayIdentity, route: str, ready_timeout: float):
        if not 0 < ready_timeout <= 60:
            raise ValueError("ready_timeout must be greater than zero and at most 60 seconds")
        self._gateway = identity
        self._invocation_trace_id = identity.request_id
        self._gateway_route = route
        self._ready_timeout = ready_timeout
        self._gateway_ready = False
        self._gateway_failed = False
        self._receive_task = None
        self._protocol_keepalive_task = None

    def _gateway_url(self) -> str:
        return self._gateway.url(self._gateway_route, websocket=True)

    def _gateway_config(self) -> dict | None:
        raise NotImplementedError

    async def _connect_websocket(self):
        if self._gateway_failed or not self.is_usable:
            raise GatewayError("forbidden")
        if self._gateway_ready and self._websocket and self._websocket.state is State.OPEN:
            return
        self._gateway_ready = False
        self._gateway = replace(self._gateway, request_id=str(uuid4()))
        try:
            self._websocket = await self._websocket_connect(
                self._gateway_url(),
                additional_headers={
                    **self._gateway.headers(),
                    "X-AayiqAgent-Trace-Id": self._invocation_trace_id,
                },
                open_timeout=self._ready_timeout,
                max_size=MAX_MESSAGE_SIZE,
                max_queue=16,
            )
            config = self._gateway_config()
            if config is not None:
                await self._websocket.send(json.dumps(config))
            message = decode_message(
                await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=self._ready_timeout,
                )
            )
            if message["type"] == "error":
                raise GatewayError.from_message(message)
            if message["type"] != self._ready_type:
                raise GatewayError("protocol_error")
            await self._gateway_connected(message)
            self._gateway_ready = True
        except Exception as exception:
            error = connection_error(exception)
            await self._disconnect_websocket()
            # AIService.start catches exceptions; emit the classified frame first.
            await self._fail_gateway(error)
            raise error from None

    async def _gateway_connected(self, message: dict):
        pass

    async def _fail_gateway(self, error: GatewayError):
        if self._gateway_failed:
            return
        self._gateway_ready = False
        self._gateway_failed = True
        # This speech connection cannot resume partial output. The caller can
        # replace the service with a new admission; this instance is terminal.
        await self.push_error(error_msg=str(error), exception=error, force_treat_as_permanent=True)

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._receive_task is None or self._receive_task.done():
            self._receive_task = self.create_task(self._gateway_receive())
        if self._protocol_keepalive_task is None or self._protocol_keepalive_task.done():
            self._protocol_keepalive_task = self.create_task(self._gateway_keepalive())

    async def _gateway_receive(self):
        try:
            await self._receive_messages()
            if not self._disconnecting:
                await self._fail_gateway(GatewayError("gateway_unavailable", retryable=True))
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            await self._fail_gateway(connection_error(exception))
        finally:
            await self._disconnect_websocket()

    async def _gateway_keepalive(self):
        try:
            while self._gateway_ready:
                await asyncio.sleep(5)
                if self._gateway_ready:
                    await self._send_gateway_keepalive()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The receive task owns connection failures and their error frame.
            if self._websocket:
                await self._websocket.close()

    async def _send_gateway_keepalive(self):
        await self._gateway_send({"type": "keepalive"})

    async def _gateway_send(self, message: dict | bytes):
        if not self._gateway_ready or self._websocket is None or not self.is_usable:
            raise GatewayError("gateway_unavailable", retryable=True)
        if isinstance(message, dict):
            payload = json.dumps(message, ensure_ascii=False)
            if len(payload.encode("utf-8")) > 65536:
                raise GatewayError("invalid_request")
            await self._websocket.send(payload)
        else:
            # The gateway accepts PCM in bounded 64 KiB frames. Preserve every
            # sample without retrying an ambiguously submitted chunk.
            for offset in range(0, len(message), 65536):
                await self._websocket.send(message[offset : offset + 65536])

    async def _disconnect(self):
        await super()._disconnect()
        for name in ("_receive_task", "_protocol_keepalive_task"):
            task = getattr(self, name)
            if task is not None and task is not asyncio.current_task():
                await self.cancel_task(task, timeout=2.0)
            setattr(self, name, None)
        await self._disconnect_websocket()

    async def _disconnect_websocket(self):
        self._gateway_ready = False
        websocket, self._websocket = self._websocket, None
        if websocket:
            try:
                await asyncio.wait_for(websocket.close(), timeout=2.0)
            except Exception:
                pass

    def _classify_error(self, exception):
        if isinstance(exception, GatewayError):
            return exception.category
        return super()._classify_error(exception)

    def _get_websocket(self):
        if self._websocket is None:
            raise GatewayError("gateway_unavailable", retryable=True)
        return self._websocket

    def _is_keepalive_ready(self) -> bool:
        return self._gateway_ready and self._websocket is not None

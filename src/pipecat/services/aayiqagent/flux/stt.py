# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Gateway transport preserving Flux's native turn and Configure state machine."""

import time
from urllib.parse import quote

from pipecat.services.aayiqagent.protocol import (
    GatewayError,
    GatewayIdentity,
    GatewayWebsocketMixin,
    decode_message,
    required_text,
    validate_pcm,
)
from pipecat.services.deepgram.flux.stt_base import DeepgramFluxSTTBase, DeepgramFluxSTTSettings
from pipecat.services.websocket_service import WebsocketService


class AayiqAgentFluxSTTService(GatewayWebsocketMixin, DeepgramFluxSTTBase, WebsocketService):
    """Scoped Flux transport; constructor authorization is required before setup."""

    Settings = DeepgramFluxSTTSettings
    _ready_type = "Connected"

    def __init__(
        self,
        *,
        gateway_base_url: str,
        invocation_token: str,
        request_id: str,
        model: str,
        sample_rate: int | None = None,
        allow_insecure: bool = False,
        ready_timeout: float = 10.0,
        settings=None,
        should_interrupt: bool = False,
        watchdog_min_timeout: float = 0.5,
        mip_opt_out: bool | None = None,
        tag: list[str] | None = None,
        **kwargs,
    ):
        """Configure scoped Flux authorization before the pre-start setup connection."""
        identity = GatewayIdentity(gateway_base_url, invocation_token, request_id, allow_insecure)
        defaults = self.Settings(
            model=required_text(model, "model"),
            language=None,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
            language_hints=None,
            numerals=None,
            profanity_filter=None,
            redact=None,
        )
        if settings:
            self._validate_gateway_settings(settings, model)
            defaults.apply_update(settings)
        DeepgramFluxSTTBase.__init__(
            self,
            encoding="linear16",
            mip_opt_out=mip_opt_out,
            tag=tag,
            should_interrupt=should_interrupt,
            watchdog_min_timeout=watchdog_min_timeout,
            settings=defaults,
            sample_rate=sample_rate,
            **kwargs,
        )
        WebsocketService.__init__(self, reconnect_on_error=False)
        self._init_gateway(identity, "stt/flux", ready_timeout)

    @staticmethod
    def _validate_gateway_settings(settings, model):
        given = settings.given_fields()
        if settings.extra or given.get("language") is not None:
            raise ValueError("Use native Flux settings and language_hints, not language or extra")
        effective_model = given.get("model", model)
        required_text(effective_model, "model")
        if given.get("language_hints") and effective_model != "flux-general-multi":
            raise ValueError("language_hints requires the flux-general-multi model")

    async def _update_settings(self, delta):
        self._validate_gateway_settings(delta, self._settings.model)
        return await super()._update_settings(delta)

    @property
    def supports_ttfs(self) -> bool:
        """Declare that Flux owns turn boundaries rather than TTFS finalization."""
        return False

    def _gateway_url(self):
        # The base encodes repeated keyterms/tags; encode its model slot too.
        query = self._build_query_string()
        model = str(self._settings.model)
        query = query.replace("model=" + model, "model=" + quote(model, safe=""), 1)
        return super()._gateway_url() + "?" + query

    def _gateway_config(self):
        return None

    async def _gateway_connected(self, message):
        self._connection_established_event.clear()
        self._user_is_speaking = False
        await self._handle_message(message)
        await self._await_connection_established()

    async def _connect(self):
        # DeepgramFluxSTTBase's abstract hook is not cooperative with WebsocketService.
        await WebsocketService._connect(self)
        await GatewayWebsocketMixin._connect(self)
        if self._watchdog_task is None:
            self._watchdog_task = self.create_task(self._watchdog_task_handler())
        await self._call_event_handler("on_connected")

    async def _disconnect(self):
        await WebsocketService._disconnect(self)
        await GatewayWebsocketMixin._disconnect(self)

    async def _disconnect_websocket(self):
        if self._watchdog_task:
            await self.cancel_task(self._watchdog_task, timeout=2.0)
            self._watchdog_task = None
        self._connection_established_event.clear()
        self._reset_configure_state()
        self._last_stt_time = None
        if self._gateway_ready:
            try:
                await self._gateway_send({"type": "CloseStream"})
            except Exception:
                pass
        await self.stop_all_metrics()
        await super()._disconnect_websocket()

    async def _transport_send_audio(self, audio: bytes):
        validate_pcm(audio)
        await self._gateway_send(audio)

    async def _transport_send_json(self, message: dict):
        await self._gateway_send(message)

    def _transport_is_active(self):
        return self._gateway_ready and self._websocket is not None

    async def _send_gateway_keepalive(self):
        # Native Flux liveness uses WebSocket ping; watchdog silence remains metered.
        if self._websocket:
            await self._websocket.ping()

    async def run_stt(self, audio: bytes):
        """Submit metered PCM and update the inherited speech watchdog."""
        self._last_stt_time = time.monotonic()
        self._last_audio_chunk_duration = len(audio) / (self.sample_rate * 2)
        await self._transport_send_audio(audio)
        yield None

    async def _receive_messages(self):
        async for raw in self._get_websocket():
            message = decode_message(raw)
            if message["type"] == "error":
                raise GatewayError.from_message(message)
            await self._handle_message(message)

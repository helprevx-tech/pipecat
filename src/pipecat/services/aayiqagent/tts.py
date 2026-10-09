# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Context-scoped speech synthesis through the owned invocation gateway."""

import base64
import math
from dataclasses import dataclass, field

from pipecat.frames.frames import StartFrame, TTSAudioRawFrame, TTSStartedFrame
from pipecat.services.aayiqagent.protocol import (
    MAX_MESSAGE_SIZE,
    GatewayError,
    GatewayIdentity,
    GatewayWebsocketMixin,
    decode_message,
    required_text,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class AayiqAgentTTSSettings(TTSSettings):
    """Model, voice, and speed; language, pitch, and volume are unsupported."""

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AayiqAgentTTSService(GatewayWebsocketMixin, WebsocketTTSService):
    """PCM synthesis with absolute context-relative alignment and bounded context state."""

    Settings = AayiqAgentTTSSettings

    def __init__(
        self,
        *,
        gateway_base_url: str,
        invocation_token: str,
        request_id: str,
        model: str,
        voice: str,
        sample_rate: int | None = None,
        allow_insecure: bool = False,
        ready_timeout: float = 10.0,
        settings: AayiqAgentTTSSettings | None = None,
        **kwargs,
    ):
        """Configure authorized voice synthesis and optional operator-supported speed."""
        identity = GatewayIdentity(gateway_base_url, invocation_token, request_id, allow_insecure)
        defaults = self.Settings(
            model=required_text(model, "model"),
            voice=required_text(voice, "voice"),
            language=None,
            speed=None,
        )
        if settings:
            self._validate_settings(settings)
            defaults.apply_update(settings)
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=defaults,
            reconnect_on_error=False,
            **kwargs,
        )
        self._init_gateway(identity, "tts/stream", ready_timeout)
        # Only live contexts are retained. A terminal/cancelled ID cannot be revived
        # by a late audio event, even while its audio queue is still draining.
        self._remote_contexts: dict[str, bool] = {}  # value: close_context sent
        self._alignment_times: dict[str, float] = {}

    @staticmethod
    def _validate_settings(settings):
        given = settings.given_fields()
        if settings.extra or given.get("language") is not None:
            raise ValueError("Gateway TTS supports model, voice and speed only")
        for key in ("model", "voice"):
            if key in given:
                required_text(given[key], key)
        speed = given.get("speed")
        if speed is not None and (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or not 0.6 <= speed <= 1.5
        ):
            raise ValueError("speed must be between 0.6 and 1.5")

    def _voice_settings(self) -> dict:
        return {} if self._settings.speed is None else {"speed": self._settings.speed}

    def _gateway_config(self):
        return {
            "type": "config",
            "model": self._settings.model,
            "voice": self._settings.voice,
            "sample_rate": self.sample_rate,
            "settings": self._voice_settings(),
            **self._gateway.attribution(),
        }

    def can_generate_metrics(self) -> bool:
        """Enable framework synthesis latency and input-character metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Connect after the pipeline establishes its output sample rate."""
        await super().start(frame)
        await self._connect()

    async def _update_settings(self, delta):
        self._validate_settings(delta)
        # Settings are captured when a context opens, never changed mid-context.
        if self._remote_contexts:
            raise ValueError("Finish or cancel active TTS contexts before changing settings")
        changed = await super()._update_settings(delta)
        if "model" in changed and self._gateway_ready:
            await self._disconnect()
            await self._connect()
        return changed

    async def _receive_messages(self):
        async for raw in self._get_websocket():
            await self._handle_gateway_message(decode_message(raw))

    async def _handle_gateway_message(self, message: dict):
        kind = message["type"]
        if kind == "error":
            raise GatewayError.from_message(message)
        context_id = message.get("context_id")
        if not isinstance(context_id, str) or not context_id:
            raise GatewayError("protocol_error")
        if kind not in {"audio", "alignment", "final"}:
            raise GatewayError("protocol_error")
        if context_id not in self._remote_contexts:
            return
        if kind == "final":
            self._remote_contexts.pop(context_id, None)
            self._alignment_times.pop(context_id, None)
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
            return
        if not self.audio_context_available(context_id):
            return
        if kind == "audio":
            try:
                audio = base64.b64decode(message["audio"], validate=True)
                if len(audio) > MAX_MESSAGE_SIZE:
                    raise ValueError("Gateway audio chunk exceeds its limit")
                # Provider chunks may split a PCM sample. The inherited audio
                # context assembler preserves the remainder for the next chunk.
            except (KeyError, TypeError, ValueError):
                raise GatewayError("protocol_error") from None
            await self.stop_ttfb_metrics()
            await self.start_word_timestamps()
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio,
                    self.sample_rate,
                    1,
                    context_id=context_id,
                ),
            )
        elif kind == "alignment":
            data = message.get("data")
            words = data.get("words") if isinstance(data, dict) else None
            if not isinstance(words, list) or len(words) > 10000:
                raise GatewayError("protocol_error")
            times = []
            last = self._alignment_times.get(context_id, 0.0)
            for entry in words:
                if not isinstance(entry, dict):
                    raise GatewayError("protocol_error")
                word, start = entry.get("word"), entry.get("start")
                if (
                    not isinstance(word, str)
                    or isinstance(start, bool)
                    or not isinstance(start, (int, float))
                    or not math.isfinite(start)
                    or start / 1000 < last
                ):
                    raise GatewayError("protocol_error")
                last = start / 1000
                times.append((word, last))
            self._alignment_times[context_id] = last
            await self.add_word_timestamps(times, context_id)

    async def run_tts(self, text: str, context_id: str):
        """Open a context once and submit each text fragment exactly once."""
        required_text(context_id, "context_id")
        if not isinstance(text, str) or len(text) > 20000:
            raise ValueError("Synthesis text must be a string of at most 20000 characters")
        if not self._gateway_ready:
            raise GatewayError("gateway_unavailable", retryable=True)
        if context_id not in self._remote_contexts:
            if len(self._remote_contexts) >= 32:
                raise GatewayError("invalid_request")
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)
            await self._gateway_send(
                {
                    "type": "create_context",
                    "context_id": context_id,
                    "model": self._settings.model,
                    "voice": self._settings.voice,
                    "settings": self._voice_settings(),
                }
            )
            self._remote_contexts[context_id] = False
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)
        if self._remote_contexts[context_id]:
            raise GatewayError("invalid_request")
        await self._gateway_send({"type": "synthesize", "text": text, "context_id": context_id})
        await self.start_tts_usage_metrics(text)
        yield None

    async def _finish_context(self, context_id: str):
        if context_id in self._remote_contexts and not self._remote_contexts[context_id]:
            self._remote_contexts[context_id] = True
            await self._gateway_send({"type": "close_context", "context_id": context_id})

    async def on_turn_context_completed(self):
        """Close the remote context when the current utterance completes."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id:
            await self._finish_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel provider work and immediately suppress late context events."""
        if context_id in self._remote_contexts:
            self._remote_contexts.pop(context_id)
            self._alignment_times.pop(context_id, None)
            if self._gateway_ready:
                await self._gateway_send({"type": "cancel", "context_id": context_id})
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        # Idle timeout must also cancel any abandoned remote generation.
        """Release alignment state and cancel a context abandoned by its idle timeout."""
        if context_id in self._remote_contexts and self._gateway_ready:
            await self._gateway_send({"type": "cancel", "context_id": context_id})
        self._remote_contexts.pop(context_id, None)
        self._alignment_times.pop(context_id, None)
        await super().on_audio_context_completed(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Flush the Cartesia-backed context terminally; further text needs a new context."""
        context_id = context_id or self.get_active_audio_context_id()
        if context_id in self._remote_contexts and not self._remote_contexts[context_id]:
            self._remote_contexts[context_id] = True
            await self._gateway_send({"type": "flush", "context_id": context_id})

    async def _send_gateway_keepalive(self):
        for context_id, closing in list(self._remote_contexts.items()):
            if not closing:
                await self._gateway_send({"type": "keepalive", "context_id": context_id})

    async def _disconnect_websocket(self):
        await self._close_connection_contexts()
        self._remote_contexts.clear()
        self._alignment_times.clear()
        await self.stop_all_metrics()
        await super()._disconnect_websocket()

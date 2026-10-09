# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""PCM transcription through the authenticated AayiqAgent gateway."""

from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.aayiqagent.protocol import (
    GatewayError,
    GatewayIdentity,
    GatewayWebsocketMixin,
    decode_message,
    required_text,
    validate_pcm,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601


@dataclass
class AayiqAgentSTTSettings(STTSettings):
    """Model and language; changing either establishes a new authorized stream."""


class AayiqAgentSTTService(GatewayWebsocketMixin, WebsocketSTTService):
    """Mono signed PCM16 transcription with finalization and external turn proposals."""

    Settings = AayiqAgentSTTSettings

    def __init__(
        self,
        *,
        gateway_base_url: str,
        invocation_token: str,
        request_id: str,
        model: str,
        language: str = "multi",
        sample_rate: int | None = None,
        interim_results: bool = True,
        vad_events: bool = False,
        keyterms: list[str] | None = None,
        allow_insecure: bool = False,
        ready_timeout: float = 10.0,
        settings: AayiqAgentSTTSettings | None = None,
        **kwargs,
    ):
        """Configure authorized recognition, readiness timeout and connection-only options."""
        identity = GatewayIdentity(gateway_base_url, invocation_token, request_id, allow_insecure)
        defaults = self.Settings(model=required_text(model, "model"), language=language)
        if settings:
            self._validate_settings(settings)
            defaults.apply_update(settings)
        super().__init__(
            sample_rate=sample_rate, settings=defaults, reconnect_on_error=False, **kwargs
        )
        self._init_gateway(identity, "stt/stream", ready_timeout)
        self._interim_results = interim_results
        self._vad_events = vad_events
        self._keyterms = list(keyterms or [])
        if len(self._keyterms) > 100 or any(
            not isinstance(x, str) or not 0 < len(x) <= 100 for x in self._keyterms
        ):
            raise ValueError("keyterms must contain at most 100 strings of at most 100 characters")
        self._register_event_handler("on_speech_started")
        self._register_event_handler("on_speech_ended")

    @staticmethod
    def _validate_settings(settings):
        if settings.extra:
            raise ValueError("Only model and language runtime STT settings are supported")
        for key, value in settings.given_fields().items():
            required_text(value, key)

    def can_generate_metrics(self) -> bool:
        """Enable framework latency and audio usage metrics."""
        return True

    def service_metadata_frame(self):
        """Recommend external turn strategies when provider VAD is enabled."""
        frame = super().service_metadata_frame()
        if self._vad_events:
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    def _gateway_config(self):
        return {
            "type": "config",
            "model": self._settings.model,
            "language": self._settings.language,
            "sample_rate": self.sample_rate,
            "interim_results": self._interim_results,
            "vad_events": self._vad_events,
            "keyterms": self._keyterms,
            **self._gateway.attribution(),
        }

    async def start(self, frame: StartFrame):
        """Connect only after the pipeline sample rate is established."""
        await super().start(frame)
        await self._connect()

    async def _update_settings(self, delta):
        self._validate_settings(delta)
        changed = await super()._update_settings(delta)
        if changed and self._gateway_ready:
            await self._reconnect()
        return changed

    async def _receive_messages(self):
        async for raw in self._get_websocket():
            message = decode_message(raw)
            kind = message["type"]
            if kind == "error":
                raise GatewayError.from_message(message)
            if kind == "transcription":
                await self._handle_transcription(message)
            elif kind == "speech_started" and self._vad_events:
                await self.start_ttfb_metrics()
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
                await self._call_event_handler("on_speech_started")
            elif kind == "speech_ended" and self._vad_events:
                await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
                await self._call_event_handler("on_speech_ended")
            elif kind not in {"ready", "speech_started", "speech_ended"}:
                raise GatewayError("protocol_error")

    async def _handle_transcription(self, message: dict):
        text = message.get("text", "")
        if not isinstance(text, str) or not isinstance(message.get("is_final"), bool):
            raise GatewayError("protocol_error")
        final = message["is_final"]
        if final and message.get("from_finalize") is True:
            self.confirm_finalize()
        if not text:
            # Empty finalization acknowledges the flush, not the next utterance.
            if final and message.get("from_finalize") is True:
                self._finalize_pending = False
            return
        try:
            language = Language(message["language"]) if message.get("language") else None
        except ValueError:
            language = None
        frame_class = TranscriptionFrame if final else InterimTranscriptionFrame
        await self.push_frame(
            frame_class(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=language,
                result={"confidence": message.get("confidence")},
            )
        )

    async def process_frame(self, frame: Frame, direction):
        """Request an explicit finalization acknowledgment when local speech ends."""
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._gateway_ready:
            self.request_finalize()
            await self._gateway_send({"type": "finalize"})

    async def run_stt(self, audio: bytes):
        """Send PCM only after the gateway confirms recognition readiness."""
        validate_pcm(audio)
        await self._gateway_send(audio)
        yield None

    async def _disconnect_websocket(self):
        if self._gateway_ready and self._websocket:
            try:
                await self._gateway_send({"type": "end_of_stream"})
            except Exception:
                pass
        await super()._disconnect_websocket()

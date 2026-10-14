#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FastAPI WebSocket transport implementation for Pipecat.

This module provides WebSocket-based transport for real-time audio/video streaming
using FastAPI and WebSocket connections. Supports binary and text serialization
with configurable session timeouts and WAV header generation.

An optional output policy preserves opaque source ownership through serializer
filters, packet remainders and final sends. Policy-free callers retain their
existing buffering, wire data and disconnect behavior.
"""

import asyncio
import time
import typing
from collections.abc import Awaitable, Callable

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import (
    AUDIO_OUTPUT_DISCARDED,
    AudioOutputDiscarded,
    AudioOutputPolicy,
    BaseOutputTransport,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.enums import EndTaskReason
from pipecat.utils.security.allowed_origins import default_allowed_origins, is_origin_allowed

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use FastAPI websockets, you need to `uv add "pipecat-ai[websocket]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e


# Default time (in seconds) to wait for the WebSocket close handshake in
# ``FastAPIWebsocketClient.disconnect()`` before proceeding with shutdown.
_WS_CLOSE_TIMEOUT_DEFAULT = 0.5

# Unlike a captured None, this value means no output filter owns buffered audio.
_NO_AUDIO_OUTPUT_CONTEXT = object()


class FastAPIWebsocketParams(TransportParams):
    """Configuration parameters for FastAPI WebSocket transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for encoding/decoding messages.
        session_timeout: Session timeout in seconds, None for no timeout.
        fixed_audio_packet_size: Optional fixed-size packetization for raw PCM audio payloads.
            Useful when the remote WebSocket media endpoint requires strict audio framing.
        allowed_origins: List of allowed origins. Empty list allows all
            origins. When set, connections with a missing or disallowed Origin header
            are rejected. Defaults to ``PIPECAT_ALLOWED_ORIGINS`` env var
            (comma-separated).
        ws_close_timeout: Maximum time, in seconds, to wait for the WebSocket
            close handshake during disconnect. The close is initiated in a
            background task before we start waiting, so the close frame is sent
            to the peer in the common case; this only bounds how long we wait
            for the peer to acknowledge it before letting shutdown proceed.
            Prevents a dead or half-closed peer (e.g. a telephony call already
            torn down on the provider's side) from stalling pipeline shutdown on
            the ASGI server's close-handshake timeout. Increase it for
            high-latency peers that need longer to complete a graceful close.
    """

    add_wav_header: bool = False
    serializer: FrameSerializer | None = None
    session_timeout: int | None = None
    fixed_audio_packet_size: int | None = None
    allowed_origins: list[str] = Field(default_factory=default_allowed_origins)
    ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT


class FastAPIWebsocketCallbacks(BaseModel):
    """Callback functions for WebSocket events.

    Parameters:
        on_client_connected: Called when a client connects to the WebSocket.
        on_client_disconnected: Called when a client disconnects from the WebSocket.
        on_session_timeout: Called when a session timeout occurs.
    """

    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
    on_session_timeout: Callable[[WebSocket], Awaitable[None]]


class _WebSocketMessageIterator:
    """Async iterator for WebSocket messages that yields both binary and text."""

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes | str:
        message = await self._websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise StopAsyncIteration
        if "bytes" in message and message["bytes"] is not None:
            return message["bytes"]
        if "text" in message and message["text"] is not None:
            return message["text"]
        raise StopAsyncIteration


class FastAPIWebsocketClient:
    """WebSocket client wrapper for handling connections and message passing.

    Manages WebSocket state, message sending/receiving, and connection lifecycle
    with support for both binary and text message types.
    """

    def __init__(
        self,
        websocket: WebSocket,
        callbacks: FastAPIWebsocketCallbacks,
        ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT,
    ):
        """Initialize the WebSocket client.

        Args:
            websocket: The FastAPI WebSocket connection.
            callbacks: Event callback functions.
            ws_close_timeout: Maximum time, in seconds, to wait for the close
                handshake in ``disconnect()`` before proceeding.
        """
        self._websocket = websocket
        self._closing = False
        self._callbacks = callbacks
        self._leave_counter = 0
        self._transfer_in_progress = False
        self._ws_close_timeout = ws_close_timeout
        self._close_task: asyncio.Task | None = None

    async def setup(self, _: FrameProcessorSetup):
        """Set up the WebSocket client.

        Args:
            _: The start frame (unused).
        """
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        """Get an async iterator for receiving WebSocket messages.

        Returns:
            An async iterator yielding bytes or strings.
        """
        return _WebSocketMessageIterator(self._websocket)

    async def send(self, data: str | bytes, *, send_guard: Callable[[], bool] | None = None):
        """Send data through the WebSocket connection.

        Args:
            data: The data to send (string or bytes).
            send_guard: Optional pure source check immediately before the socket
                call. A deliberate stale drop returns AUDIO_OUTPUT_DISCARDED;
                guarded I/O failures return False. Ordinary callers retain the
                existing None result and exception logging. Cancellation always
                propagates; a started remote write is never retried here.
        """
        try:
            if send_guard is not None and send_guard() is not True:
                return AUDIO_OUTPUT_DISCARDED
            if self._can_send():
                if isinstance(data, bytes):
                    await self._websocket.send_bytes(data)
                else:
                    await self._websocket.send_text(data)
                if send_guard is not None:
                    return True
            elif send_guard is not None:
                return False
        except Exception as e:
            if send_guard is not None:
                logger.warning(
                    f"{self} guarded send failed: {e.__class__.__name__}, "
                    f"application_state: {self._websocket.application_state}"
                )
                return False
            logger.warning(
                f"{self} exception sending data: {e.__class__.__name__} ({e}), application_state: {self._websocket.application_state}"
            )

    async def disconnect(self):
        """Disconnect the WebSocket client.

        The close handshake is bounded by ``ws_close_timeout``. The close is
        initiated in a background task before we start waiting, so the close
        frame is sent to the peer in the common case; we then wait at most
        ``ws_close_timeout`` seconds for the peer to acknowledge it. If the peer
        never replies (e.g. a half-closed connection after the remote side
        already hung up), we stop waiting and let shutdown proceed instead of
        blocking on the ASGI server's close-handshake timeout. The close task is
        left running and its eventual result is logged, since the underlying
        ``close()`` may not respond to cancellation.
        """
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            # We initiated the close either way; the flag must be set even when
            # the close frame is skipped below, so the receive loop does not
            # report a client disconnect and the output transport stops writing.
            self._closing = True
            # The application side may have sent its close frame through another
            # path (e.g. a route handler closing the socket directly) while the
            # client state is still CONNECTED; sending a second close frame
            # raises. Same guard as _can_send().
            if self._websocket.application_state == WebSocketState.DISCONNECTED:
                logger.debug(f"{self} websocket close frame already sent; skipping close")
                return
            self._close_task = asyncio.create_task(self._websocket.close(), name="fastapi-ws-close")
            self._close_task.add_done_callback(self._on_close_done)
            done, _ = await asyncio.wait({self._close_task}, timeout=self._ws_close_timeout)
            if not done:
                logger.debug(
                    f"{self} WebSocket close exceeded {self._ws_close_timeout}s; "
                    "proceeding with shutdown"
                )

    def _on_close_done(self, task: asyncio.Task):
        """Log the outcome of the WebSocket close task.

        Runs whether the close completed within ``ws_close_timeout`` or
        afterwards, so an error raised by a slow close is still surfaced.
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception while closing the websocket: {e}")

    async def trigger_client_disconnected(self):
        """Trigger the client disconnected callback."""
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        """Trigger the client connected callback."""
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timeout(self):
        """Trigger the client timeout callback."""
        await self._callbacks.on_session_timeout(self._websocket)

    def _can_send(self):
        """Check if data can be sent through the WebSocket."""
        return (
            self.is_connected
            and not self.is_closing
            and self._websocket.application_state != WebSocketState.DISCONNECTED
        )

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is currently connected.

        Returns:
            True if the WebSocket is in connected state.
        """
        return self._websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_closing(self) -> bool:
        """Check if the WebSocket is currently closing.

        Returns:
            True if the WebSocket is in the process of closing.
        """
        return self._closing


class FastAPIWebsocketInputTransport(BaseInputTransport):
    """Input transport for FastAPI WebSocket connections.

    Handles incoming WebSocket messages, deserializes frames, and manages
    connection monitoring with optional session timeouts.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket input transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._monitor_websocket_task = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the input transport with the frame processor setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)

        await self._client.setup(setup)

        if self._params.serializer:
            await self._params.serializer.setup(setup)

    async def cleanup(self):
        """Clean up transport resources."""
        await super().cleanup()
        await self._teardown()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        """Start the input transport and begin message processing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if not self._monitor_websocket_task and self._params.session_timeout:
            self._monitor_websocket_task = self.create_task(
                self._monitor_websocket(self._params.session_timeout)
            )

        await self._client.trigger_client_connected()
        await self.push_frame(ClientConnectedFrame())

        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_messages())

        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        """Stop all running tasks."""
        if self._monitor_websocket_task:
            await self.cancel_task(self._monitor_websocket_task)
            self._monitor_websocket_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def _teardown(self):
        """Cancel the receive/monitor tasks and disconnect the client.

        Idempotent and shared by ``stop()``, ``cancel()``, and ``cleanup()`` so
        the independent receive loop is always stopped and the WebSocket always
        disconnected, regardless of which teardown hook runs.
        """
        await self._stop_tasks()
        await self._client.disconnect()

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._teardown()

    async def _receive_messages(self):
        """Main message receiving loop for WebSocket messages."""
        try:
            async for message in self._client.receive():
                if not self._params.serializer:
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                elif isinstance(frame, InputTransportMessageFrame):
                    await self.broadcast_frame(InputTransportMessageFrame, message=frame.message)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._client.is_closing and not self._client._transfer_in_progress:
            await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self, timeout: int):
        """Wait for ``timeout`` seconds, then trigger the client-timeout event if still open."""
        await asyncio.sleep(timeout)
        await self._client.trigger_client_timeout()


class FastAPIWebsocketOutputTransport(BaseOutputTransport):
    """Output transport for FastAPI WebSocket connections.

    Handles outgoing frame serialization, audio streaming with timing simulation,
    and WebSocket message transmission with optional WAV header generation.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket output transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed during setup.
        self._send_interval = 0
        self._next_send_time = 0

        # Buffer for optional protocol-level audio packetization.
        # Some serializers may emit arbitrarily sized raw PCM payloads, while
        # certain downstream transports or media endpoints require audio to be
        # sent in fixed-size frames. When `params.fixed_audio_packet_size` is set,
        # this buffer accumulates outgoing audio until a full packet can be
        # emitted, preserving any remainder for subsequent sends.
        self._audio_send_buffer = bytearray()

        # Native audio writes run in a media task while urgent/control frames can
        # serialize concurrently. Only the opted-in path takes this lock: its
        # output filter and packet remainder must belong to one exact context.
        self._audio_serialization_lock = asyncio.Lock()
        self._audio_serialization_context = _NO_AUDIO_OUTPUT_CONTEXT
        self._audio_serialization_context_id: str | None = None
        self._audio_serialization_destination: str | None = None
        self._audio_serialization_failed = False

    def validate_audio_output_policy(self, policy: AudioOutputPolicy | None = None) -> None:
        """Preflight actual serializer boundaries before guarded media is built.

        No-argument validation checks this transport's current configuration;
        supplied policies also use the base callback/filter validation. No
        provider request or serializer setup is performed.
        """
        super().validate_audio_output_policy(policy)
        serializer = self._params.serializer
        if serializer is None or serializer.supports_output_audio_boundary is not True:
            raise ValueError("WebSocket serializer does not support guarded audio boundaries")
        if (
            not isinstance(serializer.output_audio_packet_padding, bytes)
            or len(serializer.output_audio_packet_padding) != 1
        ):
            raise ValueError("WebSocket serializer has invalid audio packet padding")
        packet_bytes = self._params.fixed_audio_packet_size
        if packet_bytes is not None and (type(packet_bytes) is not int or packet_bytes <= 0):
            raise ValueError("WebSocket guarded audio packet size must be positive")

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the output transport with the frame processor setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)

        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2

        await self._client.setup(setup)

        if self._params.serializer:
            await self._params.serializer.setup(setup)

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize timing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)

        # Transfer ends the current media WebSocket intentionally; don't report
        # it as a user hangup through on_client_disconnected.
        if isinstance(frame, EndFrame) and frame.reason == EndTaskReason.TRANSFER_CALL.value:
            self._client._transfer_in_progress = True

        await self._write_frame(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cleanup(self):
        """Discard guarded filter/remainders and release the existing transport resources.

        Cleanup never flushes abandoned speech or invokes a provider lifecycle
        request. A reset error does not prevent socket/transport teardown.
        """
        await super().cleanup()
        if self.audio_output_policy_enabled:
            async with self._audio_serialization_lock:
                try:
                    await self._reset_audio_output_boundary()
                except Exception as e:
                    logger.error(f"{self} output cleanup reset failed: {e.__class__.__name__}")
        await self._client.disconnect()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process outgoing frames with special handling for interruptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Drop any partially buffered audio to avoid replaying stale PCM
            if self._params.fixed_audio_packet_size and not self.audio_output_policy_enabled:
                self._audio_send_buffer.clear()

            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message frame.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_transport_frame(self, frame: Frame) -> bool | AudioOutputDiscarded | None:
        """Finish a guarded TTS response at its ordered native completion boundary.

        The native sender invokes this before completed-playback bookkeeping and
        handles returned failures with its bounded-write policy. Only the exact
        retained serializer owner and actual TTS context may finish its tail; a
        delayed stop never flushes another response. Ordinary callers keep the
        inherited no-op behavior, with no new tail flush.
        """
        if not self.audio_output_policy_enabled or not isinstance(frame, TTSStoppedFrame):
            return None
        context = self.capture_audio_output_context(frame)
        async with self._audio_serialization_lock:
            if (
                context != self._audio_serialization_context
                or frame.context_id != self._audio_serialization_context_id
                or frame.transport_destination != self._audio_serialization_destination
            ):
                return AUDIO_OUTPUT_DISCARDED
            if self._client.is_closing or not self._client.is_connected:
                return False
            try:
                return await self._finish_audio_output_boundary()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{self} output response tail failed: {e.__class__.__name__}")
                return False

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool | AudioOutputDiscarded:
        """Write an audio frame to the WebSocket with timing simulation.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the frame was taken, False on I/O/serialization failure, or
            AUDIO_OUTPUT_DISCARDED when its original source became stale. An
            intentional discard is not paced or reported as a writer failure.
        """
        context = None
        context_id = getattr(frame, "context_id", None)
        destination = frame.transport_destination
        interruptible = frame.interruptible
        if self.audio_output_policy_enabled:
            context = self.capture_audio_output_context(frame)
            if not self.audio_output_context_is_current(context):
                return AUDIO_OUTPUT_DISCARDED
        if self._client.is_closing or not self._client.is_connected:
            return False

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )
        if self.audio_output_policy_enabled:
            self.copy_audio_output_context(context, frame)
            frame.context_id = context_id
            frame.transport_destination = destination
            frame.interruptible = interruptible

        if self._params.add_wav_header:
            frame = OutputAudioRawFrame(
                pcm_to_wav(frame.audio, frame.sample_rate, frame.num_channels),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            if self.audio_output_policy_enabled:
                self.copy_audio_output_context(context, frame)
                frame.context_id = context_id
                frame.transport_destination = destination
                frame.interruptible = interruptible

        written = await self._write_frame(frame)
        if written is AUDIO_OUTPUT_DISCARDED:
            return AUDIO_OUTPUT_DISCARDED
        if not written:
            return False

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

        # Already-started sends are not recalled. This stops stale downstream
        # speech accounting after the pacing await without retrying those bytes.
        if self.audio_output_policy_enabled and not self.audio_output_context_is_current(context):
            return AUDIO_OUTPUT_DISCARDED

        return True

    async def _write_frame(self, frame: Frame) -> bool | AudioOutputDiscarded:
        """Serialize and send a frame through the WebSocket.

        Returns:
            Whether the transport took the frame. A serializer that emits no
            payload has still taken it; see :meth:`FrameSerializer.serialize`.
            Guarded stale audio returns the distinct intentional-discard value.
        """
        if self.audio_output_policy_enabled:
            return await self._write_guarded_frame(frame)
        if self._client.is_closing or not self._client.is_connected:
            return False

        if not self._params.serializer:
            return False

        success = True
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                # Optional protocol-level audio packetization:
                # If a downstream WebSocket media endpoint requires fixed-size PCM frames,
                # configure params.fixed_audio_packet_size (e.g. 640 for 20ms @ 16kHz PCM16 mono).
                packet_bytes = self._params.fixed_audio_packet_size

                if packet_bytes and isinstance(payload, (bytes, bytearray)):
                    self._audio_send_buffer.extend(bytes(payload))

                    # Send only full frames; keep remainder for the next call.
                    while len(self._audio_send_buffer) >= packet_bytes:
                        chunk = bytes(self._audio_send_buffer[:packet_bytes])
                        del self._audio_send_buffer[:packet_bytes]
                        await self._client.send(chunk)
                    return True

                await self._client.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")
            success = False

        return success

    async def _reset_audio_output_boundary(self) -> None:
        """Discard this serializer owner's filter and packet remainder together.

        The guarded serialization lock is held by the caller. Buffer metadata is
        cleared even if a custom reset fails; that failure is not permission to
        relabel the retained bytes as another source.
        """
        try:
            await self._params.serializer.reset_output_audio()
        except Exception:
            # A failed reset cannot safely admit another owner on this filter.
            self._audio_serialization_failed = True
            raise
        finally:
            self._audio_send_buffer.clear()
            self._audio_serialization_context = _NO_AUDIO_OUTPUT_CONTEXT
            self._audio_serialization_context_id = None
            self._audio_serialization_destination = None

    async def _send_guarded_payload(
        self, payload: str | bytes, context: object | None, *, packetize: bool
    ) -> bool | AudioOutputDiscarded:
        """Send only payloads belonging to one retained owner under the lock.

        Each actual send rechecks the same context in the native client. A
        started send can be externally uncertain and is never requeued. Packet
        remainders are used only for this owner's audio, never control messages.
        """
        if not self.audio_output_context_is_current(context):
            return AUDIO_OUTPUT_DISCARDED

        def current_source() -> bool:
            """Recheck the retained immutable context at the final socket boundary."""
            return self.audio_output_context_is_current(context)

        packet_bytes = self._params.fixed_audio_packet_size if packetize else None
        if packet_bytes and isinstance(payload, (bytes, bytearray)):
            self._audio_send_buffer.extend(bytes(payload))
            while len(self._audio_send_buffer) >= packet_bytes:
                if not current_source():
                    self._audio_send_buffer.clear()
                    return AUDIO_OUTPUT_DISCARDED
                chunk = bytes(self._audio_send_buffer[:packet_bytes])
                del self._audio_send_buffer[:packet_bytes]
                result = await self._client.send(chunk, send_guard=current_source)
                if result is AUDIO_OUTPUT_DISCARDED or result is False:
                    self._audio_send_buffer.clear()
                    return result
            return True
        return await self._client.send(payload, send_guard=current_source)

    async def _finish_audio_output_boundary(self) -> bool | AudioOutputDiscarded:
        """Finish current old audio with source-owned tail/padding, or discard it.

        The caller holds the guarded lock. Tail flushes and sends recheck the
        previous owner; codec-correct padding cannot authorize neighboring stale
        samples. Output state is reset before another source can serialize.
        """
        context = self._audio_serialization_context
        if context is _NO_AUDIO_OUTPUT_CONTEXT:
            return True
        result: bool | AudioOutputDiscarded = True
        try:
            if not self.audio_output_context_is_current(context):
                result = AUDIO_OUTPUT_DISCARDED
            else:
                payload = await self._params.serializer.flush_output_audio()
                if not self.audio_output_context_is_current(context):
                    result = AUDIO_OUTPUT_DISCARDED
                elif payload:
                    result = await self._send_guarded_payload(payload, context, packetize=True)
                if result is True and self._audio_send_buffer:
                    if not self.audio_output_context_is_current(context):
                        result = AUDIO_OUTPUT_DISCARDED
                    else:
                        packet_bytes = self._params.fixed_audio_packet_size
                        padding = self._params.serializer.output_audio_packet_padding
                        payload = bytes(self._audio_send_buffer)
                        if packet_bytes:
                            payload += padding * (packet_bytes - len(payload))
                        self._audio_send_buffer.clear()
                        result = await self._send_guarded_payload(payload, context, packetize=False)
        except BaseException:
            # A cleanup error must not replace original cancellation/failure.
            try:
                await self._reset_audio_output_boundary()
            except (Exception, asyncio.CancelledError):
                pass
            raise
        else:
            await self._reset_audio_output_boundary()
        return result

    async def _write_guarded_frame(self, frame: Frame) -> bool | AudioOutputDiscarded:
        """Serialize under one original audio owner and fence every derived send.

        Audio, urgent controls and shutdown share the serializer but may arrive
        from different native tasks. Only this opted-in branch serializes their
        filter/buffer transitions. Stale audio never reaches the serializer;
        cancellation drops uncertain retained output and always propagates.
        """
        is_audio = isinstance(frame, OutputAudioRawFrame)
        context = self.capture_audio_output_context(frame) if is_audio else None
        context_id = getattr(frame, "context_id", None)
        async with self._audio_serialization_lock:
            try:
                if self._audio_serialization_failed and is_audio:
                    return False
                if self._client.is_closing or not self._client.is_connected:
                    return False
                boundary_failed = False
                if is_audio:
                    if not self.audio_output_context_is_current(context):
                        old = self._audio_serialization_context
                        if old is not _NO_AUDIO_OUTPUT_CONTEXT and not (
                            self.audio_output_context_is_current(old)
                        ):
                            await self._reset_audio_output_boundary()
                        return AUDIO_OUTPUT_DISCARDED
                    old = self._audio_serialization_context
                    if old is not _NO_AUDIO_OUTPUT_CONTEXT and (
                        old != context
                        or context_id != self._audio_serialization_context_id
                        or frame.transport_destination != self._audio_serialization_destination
                    ):
                        result = await self._finish_audio_output_boundary()
                        if result is False:
                            return False
                        if not self.audio_output_context_is_current(context):
                            return AUDIO_OUTPUT_DISCARDED
                    self._audio_serialization_context = context
                    self._audio_serialization_context_id = context_id
                    self._audio_serialization_destination = frame.transport_destination
                elif isinstance(frame, EndFrame):
                    try:
                        boundary_failed = await self._finish_audio_output_boundary() is False
                    except Exception as e:
                        # Tail I/O/reset failures must not suppress the existing
                        # serializer's End/hangup cleanup. Cancellation still
                        # propagates before starting another lifecycle effect.
                        logger.error(f"{self} output tail cleanup failed: {e.__class__.__name__}")
                        boundary_failed = True
                elif isinstance(frame, (InterruptionFrame, CancelFrame)):
                    try:
                        await self._reset_audio_output_boundary()
                    except Exception as e:
                        logger.error(f"{self} output reset failed: {e.__class__.__name__}")
                        boundary_failed = True

                payload = await self._params.serializer.serialize(frame)
                if is_audio and not self.audio_output_context_is_current(context):
                    await self._reset_audio_output_boundary()
                    return AUDIO_OUTPUT_DISCARDED
                if not payload:
                    return not boundary_failed
                if is_audio:
                    result = await self._send_guarded_payload(payload, context, packetize=True)
                    if result is AUDIO_OUTPUT_DISCARDED:
                        await self._reset_audio_output_boundary()
                    return result
                await self._client.send(payload)
                return not boundary_failed
            except asyncio.CancelledError:
                # Cleanup cannot replace the caller's cancellation or retry a
                # socket write whose external completion is unknown.
                try:
                    await self._reset_audio_output_boundary()
                except (Exception, asyncio.CancelledError):
                    pass
                raise
            except Exception as e:
                logger.error(f"{self} guarded output failed: {e.__class__.__name__}")
                try:
                    await self._reset_audio_output_boundary()
                except Exception:
                    pass
                return False

    async def _write_audio_sleep(self):
        """Simulate audio playback timing with appropriate delays."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class FastAPIWebsocketTransport(BaseTransport):
    """FastAPI WebSocket transport for real-time audio/video streaming.

    Provides bidirectional WebSocket communication with frame serialization,
    session management, and event handling for client connections and timeouts.

    Event handlers available:

    - on_client_connected(transport, websocket): Client WebSocket connected
    - on_client_disconnected(transport, websocket): Client WebSocket disconnected
    - on_session_timeout(transport, websocket): Session timed out

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            ...
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the FastAPI WebSocket transport.

        Raises ``ValueError`` if ``params.allowed_origins`` is set and the
        connection's Origin header is missing or not in the allowed list. The
        caller is responsible for closing the WebSocket in that case.

        Args:
            websocket: The FastAPI WebSocket connection.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        if params.allowed_origins:
            origin = websocket.headers.get("origin", "")
            if not is_origin_allowed(origin, params.allowed_origins):
                raise ValueError(f"WebSocket connection rejected: origin '{origin}' not allowed")

        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = FastAPIWebsocketClient(
            websocket, self._callbacks, ws_close_timeout=self._params.ws_close_timeout
        )

        self._input = FastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> FastAPIWebsocketInputTransport:
        """Get the input transport processor.

        Returns:
            The WebSocket input transport instance.
        """
        return self._input

    def output(self) -> FastAPIWebsocketOutputTransport:
        """Get the output transport processor.

        Returns:
            The WebSocket output transport instance.
        """
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connected event."""
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnected event."""
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        """Handle session timeout event."""
        await self._call_event_handler("on_session_timeout", websocket)

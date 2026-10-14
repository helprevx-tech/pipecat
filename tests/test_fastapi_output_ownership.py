#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise native FastAPI output ownership with an in-memory WebSocket sink.

These tests use the real transport, client and guarded serialization/packet path.
Only the external socket and a byte serializer are scripted; no server, carrier,
provider REST request or playback acceptance is involved. Existing ordinary
serializer/pacing tests remain the policy-free compatibility authority.
"""

import asyncio
import io
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
from loguru import logger
from starlette.websockets import WebSocketState

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSStoppedFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_output import (
    AUDIO_OUTPUT_CONTEXT_ATTRIBUTE,
    AUDIO_OUTPUT_DISCARDED,
    AudioOutputPolicy,
)
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport


class _MemorySocket:
    """Record actual client sends without an ASGI connection or network access."""

    def __init__(self):
        """Start connected; an optional async hook models an in-flight remote write."""
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self.payloads: list[str | bytes] = []
        self.after_send: Callable[[], Awaitable[None]] | None = None
        self.error: Exception | None = None

    async def send_bytes(self, payload: bytes):
        """Record one binary write or raise the configured I/O failure."""
        await self._send(payload)

    async def send_text(self, payload: str):
        """Record one text write through the same actual client boundary."""
        await self._send(payload)

    async def _send(self, payload: str | bytes):
        """Treat recorded bytes as accepted; the hook cannot retract that write."""
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)
        if self.after_send is not None:
            await self.after_send()


class _ByteSerializer(FrameSerializer):
    """Script serialization waits/tails while declaring an explicit reset contract."""

    supports_output_audio_boundary = True

    def __init__(self):
        """Own only this test's output bytes and pause/failure hooks."""
        super().__init__()
        self.frames: list[OutputAudioRawFrame] = []
        self.controls: list[Frame] = []
        self.pending = b""
        self.buffer_audio = False
        self.before_serialize: Callable[[], Awaitable[None]] | None = None
        self.before_flush: Callable[[], Awaitable[None]] | None = None
        self.reset_error: Exception | None = None
        self.reset_count = 0
        self.flush_count = 0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Emit audio bytes, optionally buffer/pause, and retain the actual rebuilt frame."""
        if isinstance(frame, OutputAudioRawFrame):
            self.frames.append(frame)
            if self.before_serialize is not None:
                await self.before_serialize()
            if self.buffer_audio:
                self.pending += frame.audio
                return None
            return frame.audio
        self.controls.append(frame)
        if isinstance(frame, InterruptionFrame):
            return "clear"
        if isinstance(frame, OutputTransportMessageUrgentFrame):
            return "urgent"
        return None

    async def flush_output_audio(self) -> str | bytes | None:
        """Finish only the scripted old output tail without a lifecycle request."""
        self.flush_count += 1
        if self.before_flush is not None:
            await self.before_flush()
        tail, self.pending = self.pending, b""
        return tail or None

    async def reset_output_audio(self) -> None:
        """Discard abandoned output bytes, or model a failed reset explicitly."""
        self.reset_count += 1
        self.pending = b""
        if self.reset_error is not None:
            raise self.reset_error

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """This output-only fixture never decodes incoming wire data."""
        return None


def _audio(data: bytes, owner: str | None = "old") -> OutputAudioRawFrame:
    """Issue a synthetic immutable source on actual output audio, never a latest-owner fallback."""
    frame = OutputAudioRawFrame(audio=data, sample_rate=8000, num_channels=1)
    frame.test_source = owner
    return frame


def _output(*, serializer=None, packet_size=None, wav=False, guarded=True):
    """Construct real output/client state without starting pipeline tasks or sockets.

    The code-owned policy retains immutable string sources. Its mutable allowed
    set models owner invalidation; copying never consults that set to repair a
    missing source. The pacing spy avoids wall-clock playback delays only.
    """
    serializer = serializer or _ByteSerializer()
    socket = _MemorySocket()
    output = FastAPIWebsocketTransport(
        socket,
        FastAPIWebsocketParams(
            audio_out_enabled=True,
            serializer=serializer,
            add_wav_header=wav,
            fixed_audio_packet_size=packet_size,
            allowed_origins=[],
        ),
    ).output()
    allowed = {"old", "new", "native"}

    def capture(frame):
        """Read only a source already issued on the originating test frame."""
        return getattr(frame, "test_source", None)

    def copy(context, frame):
        """Stamp exactly the retained source on native rebuilt output."""
        frame.test_source = context

    def is_current(context):
        """Authorize only a currently allowed explicit synthetic source."""
        return context in allowed

    def native_context(kind, destination):
        """Supply a distinct trusted native producer label only when requested."""
        return "native"

    if guarded:
        output.set_audio_output_policy(AudioOutputPolicy(capture, copy, is_current, native_context))
    output._sample_rate = 8000
    output._write_audio_sleep = AsyncMock()
    return output, socket, serializer, allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("wav", [False, True])
async def test_raw_and_wav_rebuild_preserve_original_context_without_wire_metadata(wav):
    """Both native rebuilds retain one origin; only encoded audio reaches the client."""
    output, socket, serializer, _ = _output(wav=wav)
    original = _audio(b"\x01\x02" * 8)
    original.context_id = "synthetic-tts"
    original.transport_destination = "synthetic-output"
    original.interruptible = False
    assert await output.write_audio_frame(original) is True
    rebuilt = serializer.frames[0]
    assert rebuilt.test_source == "old"
    assert getattr(rebuilt, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE) == "old"
    assert rebuilt.context_id == "synthetic-tts"
    assert rebuilt.transport_destination == "synthetic-output"
    assert rebuilt.interruptible is False
    assert socket.payloads[0] == rebuilt.audio
    assert (socket.payloads[0].startswith(b"RIFF")) is wav
    assert b"old" not in socket.payloads[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "foreign"])
async def test_missing_or_foreign_source_is_discarded_without_serialization_or_pacing(owner):
    """No current producer is borrowed for absent/unrecognized provenance."""
    output, socket, serializer, _ = _output()
    assert await output.write_audio_frame(_audio(b"aa", owner)) is AUDIO_OUTPUT_DISCARDED
    assert serializer.frames == []
    assert socket.payloads == []
    output._write_audio_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_current_packet_remainders_are_never_merged_across_owners():
    """A partial old packet is padded under old ownership before the new bytes begin."""
    output, socket, _, _ = _output(packet_size=4)
    assert await output.write_audio_frame(_audio(b"aa")) is True
    assert await output.write_audio_frame(_audio(b"bb", "new")) is True
    assert socket.payloads == [b"aa\x00\x00"]
    assert await output._write_frame(EndFrame()) is True
    assert socket.payloads == [b"aa\x00\x00", b"bb\x00\x00"]


@pytest.mark.asyncio
async def test_stale_packet_remainder_is_discarded_before_new_owner_audio():
    """Previously buffered bytes cannot become a newly current packet's prefix."""
    output, socket, _, allowed = _output(packet_size=4)
    assert await output.write_audio_frame(_audio(b"aa")) is True
    allowed.remove("old")
    assert await output.write_audio_frame(_audio(b"bbbb", "new")) is True
    assert socket.payloads == [b"bbbb"]
    assert not output._audio_send_buffer


@pytest.mark.asyncio
async def test_current_serializer_tail_is_finished_under_old_source():
    """Owner transitions emit a legitimate old tail before accepting later output."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, _ = _output(serializer=serializer)
    assert await output.write_audio_frame(_audio(b"aa")) is True
    serializer.buffer_audio = False
    assert await output.write_audio_frame(_audio(b"bb", "new")) is True
    assert socket.payloads == [b"aa", b"bb"]
    assert serializer.flush_count == 1
    assert serializer.pending == b""


@pytest.mark.asyncio
async def test_delayed_tts_stop_cannot_flush_another_response_with_the_same_source():
    """Existing TTS context identity distinguishes two responses within one procedure owner."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, _ = _output(serializer=serializer)
    first, second = _audio(b"aa"), _audio(b"bb")
    first.context_id, second.context_id = "first", "second"
    await output.write_audio_frame(first)
    await output.write_audio_frame(second)
    assert socket.payloads == [b"aa"]
    stale_stop = TTSStoppedFrame(context_id="first")
    stale_stop.test_source = "old"
    assert await output.write_transport_frame(stale_stop) is AUDIO_OUTPUT_DISCARDED
    assert serializer.pending == b"bb"
    current_stop = TTSStoppedFrame(context_id="second")
    current_stop.test_source = "old"
    assert await output.write_transport_frame(current_stop) is True
    assert socket.payloads == [b"aa", b"bb"]


@pytest.mark.asyncio
async def test_guarded_tts_stop_tail_failure_remains_real_io_failure():
    """Completion does not classify an unavailable actual sink as a harmless stale drop."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, _ = _output(serializer=serializer)
    await output.write_audio_frame(_audio(b"aa"))
    socket.error = OSError("synthetic unavailable sink")
    stop = TTSStoppedFrame()
    stop.test_source = "old"
    assert await output.write_transport_frame(stop) is False


@pytest.mark.asyncio
async def test_stale_serializer_tail_is_reset_without_emitting_old_samples():
    """Buffered old samples are discarded rather than emerging during new serialization."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, allowed = _output(serializer=serializer)
    await output.write_audio_frame(_audio(b"aa"))
    allowed.remove("old")
    serializer.buffer_audio = False
    assert await output.write_audio_frame(_audio(b"bb", "new")) is True
    assert socket.payloads == [b"bb"]
    assert serializer.flush_count == 0
    assert serializer.reset_count == 1


@pytest.mark.asyncio
async def test_source_change_during_serialization_discards_result_and_filter():
    """The post-serialize fence rejects an originally current frame after its await."""
    output, socket, serializer, allowed = _output()

    async def invalidate():
        """Revoke the actual old owner inside the serialization await."""
        allowed.remove("old")

    serializer.before_serialize = invalidate
    assert await output.write_audio_frame(_audio(b"aaaa")) is AUDIO_OUTPUT_DISCARDED
    assert socket.payloads == []
    assert serializer.reset_count == 1
    output._write_audio_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_change_during_tail_flush_drops_tail_and_keeps_new_source():
    """An async flush cannot promote its old retained bytes to the incoming owner."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, allowed = _output(serializer=serializer)
    await output.write_audio_frame(_audio(b"aa"))

    async def invalidate():
        """Revoke only the old source while its filter is being flushed."""
        allowed.remove("old")

    serializer.before_flush = invalidate
    serializer.buffer_audio = False
    assert await output.write_audio_frame(_audio(b"bb", "new")) is True
    assert socket.payloads == [b"bb"]


@pytest.mark.asyncio
async def test_source_change_between_actual_packet_sends_stops_remaining_packets():
    """Already accepted first bytes remain recorded; unsent old packets are discarded."""
    output, socket, _, allowed = _output(packet_size=4)

    async def invalidate():
        """Invalidate after the first actual client send has accepted its bytes."""
        allowed.remove("old")

    socket.after_send = invalidate
    assert await output.write_audio_frame(_audio(b"aaaabbbbcccc")) is AUDIO_OUTPUT_DISCARDED
    assert socket.payloads == [b"aaaa"]
    assert output._audio_send_buffer == bytearray()


@pytest.mark.asyncio
async def test_actual_client_final_guard_and_io_failure_are_distinct():
    """The final native sink skips intentional stale data and reports real write failure."""
    output, socket, _, _ = _output()

    def stale():
        """Deny this source immediately before the native socket call."""
        return False

    def current():
        """Allow the explicit source so a real I/O error is observable."""
        return True

    assert await output._client.send(b"aa", send_guard=stale) is AUDIO_OUTPUT_DISCARDED
    assert socket.payloads == []
    socket.error = OSError("synthetic unavailable sink")
    assert await output._client.send(b"aa", send_guard=current) is False
    assert socket.payloads == []


@pytest.mark.asyncio
async def test_guarded_client_error_does_not_log_private_payload_or_exception_details():
    """Opted-in failure logs identify I/O class/state without copying private source/data."""
    output, socket, _, _ = _output()
    socket.error = OSError("synthetic-private-exception-detail")
    messages = io.StringIO()
    sink = logger.add(messages, level="WARNING")
    try:
        assert await output.write_audio_frame(_audio(b"synthetic-private-audio")) is False
    finally:
        # Remove only this test's temporary Loguru sink, preserving global logging.
        logger.remove(sink)
    assert "OSError" in messages.getvalue()
    assert "synthetic-private" not in messages.getvalue()


@pytest.mark.asyncio
async def test_pacing_owner_change_cannot_publish_stale_downstream_speech():
    """A completed send is not retried; the subsequent pacing fence reports a stale scope."""
    output, socket, _, allowed = _output()

    async def invalidate():
        """Invalidate during the actual transport pacing await."""
        allowed.remove("old")

    output._write_audio_sleep = invalidate
    assert await output.write_audio_frame(_audio(b"aa")) is AUDIO_OUTPUT_DISCARDED
    assert socket.payloads == [b"aa"]


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [InterruptionFrame, CancelFrame])
async def test_interruption_and_cancel_discard_filter_and_packet_remainder(control):
    """Shutdown/interruption cleanup never flushes abandoned speech as completed output."""
    output, socket, serializer, _ = _output(packet_size=4)
    await output.write_audio_frame(_audio(b"aa"))
    assert await output._write_frame(control()) is True
    assert socket.payloads == (["clear"] if control is InterruptionFrame else [])
    assert serializer.flush_count == 0
    assert serializer.reset_count == 1
    assert not output._audio_send_buffer


@pytest.mark.asyncio
async def test_cancellation_during_serialize_preserves_original_cancellation_if_reset_fails():
    """Secondary cleanup errors do not transform cancellation or reissue uncertain output."""
    output, socket, serializer, _ = _output()

    async def cancel():
        """Cancel the real serialization operation before any payload is emitted."""
        raise asyncio.CancelledError("synthetic original cancellation")

    serializer.before_serialize = cancel
    serializer.reset_error = RuntimeError("synthetic reset failure")
    with pytest.raises(asyncio.CancelledError, match="synthetic original"):
        await output.write_audio_frame(_audio(b"aa"))
    assert socket.payloads == []
    assert await output.write_audio_frame(_audio(b"bb", "new")) is False


@pytest.mark.asyncio
async def test_cancellation_during_boundary_flush_is_not_masked_by_reset_failure():
    """A cancelled old tail cannot become another owner or lose cancellation to cleanup."""
    serializer = _ByteSerializer()
    serializer.buffer_audio = True
    output, socket, _, _ = _output(serializer=serializer)
    await output.write_audio_frame(_audio(b"aa"))

    async def cancel():
        """Cancel the real old-owner flush before any source-owned tail reaches the sink."""
        raise asyncio.CancelledError("synthetic flush cancellation")

    serializer.before_flush = cancel
    serializer.reset_error = RuntimeError("synthetic reset failure")
    with pytest.raises(asyncio.CancelledError, match="synthetic flush"):
        await output.write_audio_frame(_audio(b"bb", "new"))
    assert socket.payloads == []
    assert not output._audio_send_buffer


@pytest.mark.asyncio
async def test_guarded_audio_and_urgent_serialization_share_the_native_owner_lock():
    """Urgent controls cannot mutate one serializer while its audio operation is waiting."""
    output, socket, serializer, _ = _output()
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause():
        """Hold the actual serialization await until the test releases its owner."""
        entered.set()
        await release.wait()

    serializer.before_serialize = pause
    audio = asyncio.create_task(output.write_audio_frame(_audio(b"aa")))
    control = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        control = asyncio.create_task(
            output._write_frame(OutputTransportMessageUrgentFrame(message={"kind": "synthetic"}))
        )
        await asyncio.sleep(0)
        assert socket.payloads == []
        assert not control.done()
        release.set()
        assert await audio is True
        assert await control is True
        assert socket.payloads == [b"aa", "urgent"]
    finally:
        # Dispose only the tasks created in this test, including assertion failures.
        release.set()
        for task in (audio, control):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (audio, control) if task is not None), return_exceptions=True
        )


def test_guarded_preflight_rejects_unsupported_serializer_without_setup_or_provider_io():
    """A base no-op tail contract cannot qualify an unknown stateful serializer."""
    serializer = _ByteSerializer()
    serializer.supports_output_audio_boundary = False
    output, _, _, _ = _output(serializer=serializer, guarded=False)
    with pytest.raises(ValueError, match="does not support guarded"):
        output.validate_audio_output_policy()
    assert not output.audio_output_policy_enabled


@pytest.mark.parametrize("packet_size", [0, -1])
def test_guarded_preflight_rejects_nonpositive_packet_size(packet_size):
    """Invalid packet loops fail during actual guarded capability preflight."""
    output, _, _, _ = _output(packet_size=packet_size, guarded=False)
    with pytest.raises(ValueError, match="packet size must be positive"):
        output.validate_audio_output_policy()


@pytest.mark.asyncio
async def test_failed_output_reset_does_not_suppress_end_or_cancel_lifecycle_cleanup():
    """A poisoned filter denies new audio but retained cleanup still reaches its serializer."""
    output, _, serializer, _ = _output()
    await output.write_audio_frame(_audio(b"aa"))
    serializer.reset_error = RuntimeError("synthetic reset failure")
    assert await output._write_frame(CancelFrame()) is False
    assert any(isinstance(frame, CancelFrame) for frame in serializer.controls)
    assert await output.write_audio_frame(_audio(b"bb", "new")) is False
    assert await output._write_frame(EndFrame()) is True
    assert any(isinstance(frame, EndFrame) for frame in serializer.controls)


@pytest.mark.asyncio
async def test_ordinary_packetizer_preserves_existing_cross_frame_wire_bytes():
    """Policy omission keeps original coalescing and does not introduce flush/reset behavior."""
    output, socket, serializer, _ = _output(packet_size=4, guarded=False)
    assert await output.write_audio_frame(_audio(b"aa")) is True
    assert await output.write_audio_frame(_audio(b"bb", "new")) is True
    assert socket.payloads == [b"aabb"]
    assert serializer.flush_count == serializer.reset_count == 0
    assert not hasattr(serializer.frames[0], AUDIO_OUTPUT_CONTEXT_ATTRIBUTE)

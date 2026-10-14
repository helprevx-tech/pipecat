# SPDX-License-Identifier: BSD-2-Clause

"""Exercise native WebRTC queue ownership without ICE, RTP, providers or sockets.

The real RawAudioTrack, SmallWebRTCClient and output writer retain source context
through ten-millisecond chunks, paced recv and disconnect. A scripted connection
only replaces the external peer; media frames and completion futures are real.
Base sender/resampler behavior and remote playout require their separate tests.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("aiortc")
pytest.importorskip("av")

from aiortc.mediastreams import MediaStreamError  # noqa: E402

from pipecat.frames.frames import CancelFrame, EndFrame, OutputAudioRawFrame  # noqa: E402
from pipecat.transports.base_output import (  # noqa: E402
    AUDIO_OUTPUT_CONTEXT_ATTRIBUTE,
    AUDIO_OUTPUT_DISCARDED,
    AudioOutputPolicy,
)
from pipecat.transports.base_transport import TransportParams  # noqa: E402
from pipecat.transports.smallwebrtc import transport as native  # noqa: E402
from pipecat.transports.smallwebrtc.transport import (  # noqa: E402
    RawAudioTrack,
    SmallWebRTCCallbacks,
    SmallWebRTCClient,
    SmallWebRTCOutputTransport,
)

RATE = 16000
SAMPLES = RATE // 100


@dataclass(frozen=True)
class _Source:
    """Immutable synthetic source identity, independent of the active callback."""

    name: str


class _Policy:
    """Control original-source eligibility without repairing a missing frame source."""

    def __init__(self):
        """Start with one current speech source and independently permitted native silence."""
        self.current = _Source("first")
        self.silence = _Source("native-silence")
        self.silence_allowed = True
        self.captures = []
        self.native_kinds = []
        self.policy = AudioOutputPolicy(self.capture, self.copy, self.is_current, self.native)

    def capture(self, frame):
        """Read only the source explicitly placed on a synthetic input frame."""
        self.captures.append(frame)
        return getattr(frame, "_test_audio_source", None)

    def copy(self, context, frame):
        """Copy the supplied source verbatim onto a derived fixture frame."""
        frame._test_audio_source = context

    def is_current(self, context):
        """Keep native silence separate from exact current speech-source eligibility."""
        return context == self.current or (self.silence_allowed and context == self.silence)

    def native(self, kind, destination):
        """Issue context only for the actual native producer requested by the track."""
        self.native_kinds.append((kind, destination))
        return self.silence


class _Connection:
    """Scripted peer lifecycle; no aiortc peer connection or network resource exists."""

    def __init__(self):
        """Keep event handlers and the actual track installed by the real client."""
        self.connected = True
        self.handlers = {}
        self.track = None
        self.disconnects = 0

    def event_handler(self, name):
        """Register the real client's existing async event callback without dispatch."""

        def register(handler):
            """Retain one callback under the event requested by client construction."""
            self.handlers[name] = handler
            return handler

        return register

    def is_connected(self):
        """Expose the scripted peer's connected state to the native send guard."""
        return self.connected

    def audio_input_track(self):
        """No input media is needed to exercise the outgoing native track."""
        return None

    def video_input_track(self):
        """Keep video input absent for this audio-only fixture."""
        return None

    def screen_video_input_track(self):
        """Keep screen input absent without creating a video transport."""
        return None

    def replace_audio_track(self, track):
        """Retain the actual RawAudioTrack installed by client connection handling."""
        self.track = track

    async def disconnect(self):
        """Dispatch the native closed handler once through the scripted lifecycle."""
        self.disconnects += 1
        self.connected = False
        await self.handlers["closed"](self)


async def _noop(*args):
    """Accept unrelated client events without creating side effects or tasks."""


def _pcm(value, chunks=1):
    """Build known mono PCM for exact native sample and mixed-owner assertions."""
    return np.full(SAMPLES * chunks, value, dtype=np.int16).tobytes()


def _frame(value, source=None, chunks=1):
    """Construct a real output frame and explicitly attach its synthetic source."""
    frame = OutputAudioRawFrame(audio=_pcm(value, chunks), sample_rate=RATE, num_channels=1)
    if source is not None:
        frame._test_audio_source = source
    return frame


def _guarded_track(policy, *, auto_silence=True):
    """Use the real track with independent source-current and native-silence callbacks."""
    return RawAudioTrack(
        RATE,
        auto_silence=auto_silence,
        context_is_current=policy.is_current,
        # This captures the native producer itself, never the active speech source.
        native_silence_context=lambda: policy.native("track_silence", None),
    )


async def _client_output(*, guarded=True):
    """Bind the actual output/client before building its real offline audio track."""
    connection = _Connection()
    callbacks = SmallWebRTCCallbacks(
        on_app_message=_noop,
        on_client_connected=_noop,
        on_client_disconnected=_noop,
        on_track_status=_noop,
    )
    client = SmallWebRTCClient(connection, callbacks)
    params = TransportParams(audio_out_enabled=True, audio_out_sample_rate=RATE)
    output = SmallWebRTCOutputTransport(client, params)
    policy = _Policy()
    if guarded:
        output.validate_audio_output_policy()
        assert not output.audio_output_policy_enabled
        output.set_audio_output_policy(policy.policy)
    client._params = params
    client._out_sample_rate = RATE
    await client._handle_client_connected()
    return output, client, connection, policy


class _PauseSleep:
    """Pause one real native pacing/empty-queue wait at a deterministic await boundary."""

    def __init__(self):
        """Own release events and retain asyncio's real scheduler before patching."""
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.original_sleep = asyncio.sleep
        self.waits = []

    async def __call__(self, duration):
        """Pause the first wait; later waits yield normally without widening timeouts."""
        self.waits.append(duration)
        if len(self.waits) == 1:
            self.entered.set()
            await self.release.wait()
        else:
            await self.original_sleep(0)


@pytest.mark.asyncio
async def test_ordinary_pcm_timestamp_and_terminal_future_are_unchanged():
    """Real default recv emits exact ten-millisecond PCM and settles only its final chunk."""
    track = RawAudioTrack(RATE)
    future = track.add_audio_bytes(_pcm(113, 2))
    first = await track.recv()
    assert first.pts == 0 and first.sample_rate == RATE
    assert first.time_base.numerator == 1 and first.time_base.denominator == RATE
    assert np.all(first.to_ndarray() == 113)
    assert not future.done()
    second = await track.recv()
    assert second.pts == SAMPLES
    assert np.all(second.to_ndarray() == 113)
    assert future.result() is True
    assert not hasattr(second, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE)
    track.stop()


@pytest.mark.asyncio
async def test_default_and_guarded_empty_enqueues_complete_without_orphaned_futures():
    """Zero bytes need no recv and cannot leave a waiting writer behind."""
    plain = RawAudioTrack(RATE)
    guarded = _guarded_track(_Policy())
    assert plain.add_audio_bytes(b"").result() is True
    assert guarded.add_audio_bytes(b"").result() is True
    assert len(plain._chunk_queue) == len(guarded._chunk_queue) == 0
    plain.stop()
    guarded.stop()


@pytest.mark.asyncio
async def test_queue_drops_original_stale_owner_without_mixing_the_next_owner():
    """Separate real enqueue runs never mix old PCM with the next eligible owner's samples."""
    policy = _Policy()
    track = _guarded_track(policy)
    old = track.add_audio_bytes(_pcm(111, 2), context=policy.current)
    policy.current = _Source("second")
    current = track.add_audio_bytes(_pcm(222), context=policy.current)
    frame = await track.recv()
    assert np.all(frame.to_ndarray() == 222)
    assert old.result() is AUDIO_OUTPUT_DISCARDED
    assert current.result() is True
    assert frame.pts == 0
    track.stop()


@pytest.mark.asyncio
async def test_native_pacing_rechecks_after_owner_loss_and_settles_once(monkeypatch):
    """Owner loss during the actual track clock wait discards the remaining chunk, not its future."""
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    policy = _Policy()
    track = _guarded_track(policy)
    future = track.add_audio_bytes(_pcm(111, 2), context=policy.current)
    completions = []
    # Observe the original future, rather than mirroring internal settlement calls.
    future.add_done_callback(lambda completed: completions.append(completed.result()))
    assert np.all((await track.recv()).to_ndarray() == 111)
    pause = _PauseSleep()
    monkeypatch.setattr(native.asyncio, "sleep", pause)
    pending = asyncio.create_task(track.recv())
    await asyncio.wait_for(pause.entered.wait(), 1)
    policy.current = _Source("replacement")
    pause.release.set()
    frame = await asyncio.wait_for(pending, 1)
    assert pause.waits[0] > 0
    assert np.all(frame.to_ndarray() == 0)
    assert frame.pts == SAMPLES
    assert future.result() is AUDIO_OUTPUT_DISCARDED
    await pause.original_sleep(0)
    assert completions == [AUDIO_OUTPUT_DISCARDED]
    assert policy.native_kinds == [("track_silence", None), ("track_silence", None)]
    track.stop()


@pytest.mark.asyncio
async def test_blocking_track_rechecks_after_empty_queue_wait(monkeypatch):
    """A newly queued stale source after recv's actual wait cannot be blessed by the later owner."""
    policy = _Policy()
    track = _guarded_track(policy, auto_silence=False)
    pause = _PauseSleep()
    monkeypatch.setattr(native.asyncio, "sleep", pause)
    pending = asyncio.create_task(track.recv())
    await asyncio.wait_for(pause.entered.wait(), 1)
    old = track.add_audio_bytes(_pcm(111), context=policy.current)
    policy.current = _Source("replacement")
    current = track.add_audio_bytes(_pcm(222), context=policy.current)
    pause.release.set()
    frame = await asyncio.wait_for(pending, 1)
    assert np.all(frame.to_ndarray() == 222)
    assert old.result() is AUDIO_OUTPUT_DISCARDED
    assert current.result() is True
    assert policy.native_kinds == []
    track.stop()


@pytest.mark.asyncio
async def test_missing_source_never_uses_current_speech_for_silence():
    """Missing data source is discarded while actual native silence has its independent context."""
    policy = _Policy()
    track = _guarded_track(policy)
    missing = track.add_audio_bytes(_pcm(111), context=None)
    frame = await track.recv()
    assert np.all(frame.to_ndarray() == 0)
    assert missing.result() is AUDIO_OUTPUT_DISCARDED
    assert policy.native_kinds == [("track_silence", None)]
    track.stop()


@pytest.mark.asyncio
async def test_native_silence_owner_loss_during_pacing_is_not_recaptured(monkeypatch):
    """Silence captures once before the clock await and cannot adopt a replacement call phase."""
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    policy = _Policy()
    track = _guarded_track(policy)
    assert np.all((await track.recv()).to_ndarray() == 0)
    pause = _PauseSleep()
    monkeypatch.setattr(native.asyncio, "sleep", pause)
    pending = asyncio.create_task(track.recv())
    await asyncio.wait_for(pause.entered.wait(), 1)
    policy.silence = _Source("replacement-native-phase")
    pause.release.set()
    with pytest.raises(MediaStreamError):
        await asyncio.wait_for(pending, 1)
    assert policy.native_kinds == [("track_silence", None), ("track_silence", None)]
    assert track.readyState == "ended"


@pytest.mark.asyncio
async def test_guarded_cancelled_writer_discards_all_its_remaining_chunks():
    """Cancelling the original future prevents queued guarded samples from reaching native recv."""
    policy = _Policy()
    track = _guarded_track(policy)
    future = track.add_audio_bytes(_pcm(111, 2), context=policy.current)
    future.cancel()
    frame = await track.recv()
    assert np.all(frame.to_ndarray() == 0)
    assert future.cancelled()
    assert not track._chunk_queue
    track.stop()


@pytest.mark.asyncio
async def test_ordinary_cancelled_future_preserves_legacy_audio_delivery():
    """Policy-free cancellation retains existing already-queued PCM semantics."""
    track = RawAudioTrack(RATE)
    future = track.add_audio_bytes(_pcm(111))
    future.cancel()
    assert np.all((await track.recv()).to_ndarray() == 111)
    assert future.cancelled()
    track.stop()


@pytest.mark.asyncio
async def test_closed_blocking_track_exits_and_settles_original_future_once():
    """Track stop clears native queued references and future completion remains idempotent."""
    policy = _Policy()
    track = _guarded_track(policy, auto_silence=False)
    future = track.add_audio_bytes(_pcm(111, 3), context=policy.current)
    track.stop()
    track.stop()
    assert future.result() is AUDIO_OUTPUT_DISCARDED
    assert not track._chunk_queue
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_native_current_writer_delivers_exact_pcm_and_completes_successfully():
    """Real output/client/track delivery preserves current PCM and its original successful future."""
    output, client, connection, policy = await _client_output()
    frame = _frame(237, policy.current)
    pending = asyncio.create_task(output.write_audio_frame(frame))
    for _ in range(20):
        if connection.track._chunk_queue:
            break
        await asyncio.sleep(0)
    delivered = await connection.track.recv()
    assert np.all(delivered.to_ndarray() == 237)
    assert await asyncio.wait_for(pending, 1) is True
    assert policy.captures == [frame]
    assert not hasattr(delivered, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE)
    await client._handle_client_closed()


@pytest.mark.asyncio
async def test_native_client_captures_before_queue_and_returns_discard_marker():
    """The actual output/client/track handoff retains old source even if the input attribute changes."""
    output, client, connection, policy = await _client_output()
    frame = _frame(111, policy.current)
    pending = asyncio.create_task(output.write_audio_frame(frame))
    # Wait for the native enqueue; no transport callback or queue behavior is mocked.
    for _ in range(20):
        if connection.track._chunk_queue:
            break
        await asyncio.sleep(0)
    assert connection.track._chunk_queue
    policy.current = _Source("replacement")
    frame._test_audio_source = policy.current
    assert np.all((await connection.track.recv()).to_ndarray() == 0)
    assert await asyncio.wait_for(pending, 1) is AUDIO_OUTPUT_DISCARDED
    assert policy.captures == [frame]
    await client._handle_client_closed()


@pytest.mark.asyncio
async def test_native_client_missing_source_is_not_repaired_or_enqueued():
    """Cached missing provenance stays missing even when the same frame later receives a source."""
    output, client, connection, policy = await _client_output()
    frame = _frame(111)
    assert await output.write_audio_frame(frame) is AUDIO_OUTPUT_DISCARDED
    frame._test_audio_source = policy.current
    assert await output.write_audio_frame(frame) is AUDIO_OUTPUT_DISCARDED
    assert policy.captures == [frame]
    assert not connection.track._chunk_queue
    await client._handle_client_closed()


@pytest.mark.asyncio
async def test_disconnect_during_actual_clock_wait_cannot_return_queued_audio(monkeypatch):
    """Native stop while recv is paced settles queued completion and prevents a late media return."""
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    policy = _Policy()
    track = _guarded_track(policy)
    future = track.add_audio_bytes(_pcm(111, 2), context=policy.current)
    await track.recv()
    pause = _PauseSleep()
    monkeypatch.setattr(native.asyncio, "sleep", pause)
    pending = asyncio.create_task(track.recv())
    await asyncio.wait_for(pause.entered.wait(), 1)
    track.stop()
    pause.release.set()
    with pytest.raises(MediaStreamError):
        await asyncio.wait_for(pending, 1)
    assert future.result() is AUDIO_OUTPUT_DISCARDED
    assert not track._chunk_queue


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded", [False, True])
async def test_peer_disconnect_unblocks_actual_waiting_writer(guarded):
    """Peer disconnect settles the real input future before the client drops its track reference."""
    output, client, connection, policy = await _client_output(guarded=guarded)
    pending = asyncio.create_task(output.write_audio_frame(_frame(111, policy.current)))
    for _ in range(20):
        if connection.track._chunk_queue:
            break
        await asyncio.sleep(0)
    track = connection.track
    assert track._chunk_queue
    await client._handle_peer_disconnected()
    assert await asyncio.wait_for(pending, 1) is (AUDIO_OUTPUT_DISCARDED if guarded else True)
    assert client._audio_output_track is None
    assert track.readyState == "ended" and not track._chunk_queue


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["end", "cancel"])
async def test_native_output_end_cancel_release_original_queued_completion(control):
    """Real output control methods and shared-client release close the native track without replay."""
    output, client, connection, policy = await _client_output()
    await client.connect()
    pending = asyncio.create_task(output.write_audio_frame(_frame(111, policy.current, 2)))
    for _ in range(20):
        if connection.track._chunk_queue:
            break
        await asyncio.sleep(0)
    assert connection.track._chunk_queue
    if control == "end":
        await output.stop(EndFrame())
    else:
        await output.cancel(CancelFrame())
    assert await asyncio.wait_for(pending, 1) is AUDIO_OUTPUT_DISCARDED
    assert connection.disconnects == 1
    assert connection.track.readyState == "ended"
    assert client._audio_output_track is None
    await output._teardown()
    assert connection.disconnects == 1


@pytest.mark.asyncio
async def test_replaced_track_settles_old_future_without_resetting_its_owner():
    """A client reconnect builds a new track and releases the old queued writer exactly once."""
    output, client, connection, policy = await _client_output()
    old_track = connection.track
    old = old_track.add_audio_bytes(_pcm(111), context=policy.current)
    await client._handle_client_connected()
    assert connection.track is not old_track
    assert old.result() is AUDIO_OUTPUT_DISCARDED
    assert old_track.readyState == "ended"
    assert not connection.track._chunk_queue
    await client._handle_client_closed()


@pytest.mark.asyncio
async def test_preflight_and_setter_reject_active_unguarded_track_without_installing_policy():
    """An already active legacy track cannot be approved by attaching a policy after media setup."""
    output, client, connection, policy = await _client_output(guarded=False)
    with pytest.raises(ValueError, match="Existing WebRTC track"):
        output.validate_audio_output_policy()
    with pytest.raises(ValueError, match="Existing WebRTC track"):
        output.set_audio_output_policy(policy.policy)
    assert not output.audio_output_policy_enabled
    await client._handle_client_closed()


@pytest.mark.asyncio
async def test_preflight_rejects_unknown_writer_override():
    """Custom native writers require explicit guarded support, even when their override calls super."""

    class CustomOutput(SmallWebRTCOutputTransport):
        """Represent an extension without its own output-capability validation contract."""

        async def write_audio_frame(self, frame):
            """Delegate ordinary output; inherited preflight must not approve unknown code."""
            return await super().write_audio_frame(frame)

    connection = _Connection()
    client = SmallWebRTCClient(
        connection,
        SmallWebRTCCallbacks(
            on_app_message=_noop,
            on_client_connected=_noop,
            on_client_disconnected=_noop,
            on_track_status=_noop,
        ),
    )
    output = CustomOutput(client, TransportParams(audio_out_enabled=True))
    with pytest.raises(ValueError, match="does not support guarded"):
        output.validate_audio_output_policy()
    assert client._params is None and client._audio_output_track is None


@pytest.mark.asyncio
async def test_track_rejects_partial_or_async_callbacks_before_media_creation():
    """Optional guards must be paired and synchronous; malformed hooks cannot fall back to defaults."""
    with pytest.raises(ValueError, match="paired synchronous"):
        RawAudioTrack(RATE, context_is_current=lambda context: True)

    async def asynchronous_guard(context):
        """An async hook cannot make a decision at the native no-await return boundary."""
        return True

    with pytest.raises(ValueError, match="paired synchronous"):
        RawAudioTrack(
            RATE, context_is_current=asynchronous_guard, native_silence_context=lambda: None
        )


@pytest.mark.asyncio
async def test_source_errors_fail_closed_and_keep_the_original_future_terminal():
    """Guard exceptions discard queued speech; they cannot be interpreted as a successful permission."""
    policy = _Policy()

    def guarded(context):
        """Allow only actual native silence, while simulating an invalid speech-source lookup."""
        if context == policy.silence:
            return True
        raise RuntimeError("source unavailable")

    track = RawAudioTrack(
        RATE, context_is_current=guarded, native_silence_context=lambda: policy.silence
    )
    future = track.add_audio_bytes(_pcm(111), context=policy.current)
    assert np.all((await track.recv()).to_ndarray() == 0)
    assert future.result() is AUDIO_OUTPUT_DISCARDED
    track.stop()

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider-free tests of ownership on the real native output queues and resampler.

The sink records accepted writes without a socket/device. Tests exercise the
maintained BaseOutputTransport, clock/audio tasks, queue acknowledgments, SOXR
stream and observer/event forwarding. Opaque string contexts are test sources,
not execution grants; application authorization and remote playout are outside
this transport contract. Each test shuts down its actual processor/task manager.
"""

import asyncio
import dataclasses
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.utils import create_stream_resampler
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    InterruptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    SpeechBoundaryFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_output import (
    AUDIO_OUTPUT_CONTEXT_ATTRIBUTE,
    AUDIO_OUTPUT_DISCARDED,
    AudioOutputPolicy,
    BaseOutputTransport,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager


class _Sources:
    """Issue immutable test contexts and revoke them without changing queued frames."""

    def __init__(self):
        """Create two speech sources and destination-aware native lifecycle contexts."""
        self.allowed = {"one", "two", ("mixer", None), ("end_silence", None)}
        self.native_calls = []
        self.policy = AudioOutputPolicy(self.capture, self.copy, self.is_current, self.native)

    def capture(self, frame):
        """Read only the source already attached by the test producer."""
        return getattr(frame, "test_source", None)

    def copy(self, context, frame):
        """Stamp the same source onto a derived native frame without repairing None."""
        frame.test_source = context

    def is_current(self, context):
        """Return a literal eligibility result for the exact retained test source."""
        return context in self.allowed

    def native(self, kind, destination):
        """Capture the real generated branch, retaining its destination."""
        context = (kind, destination)
        self.native_calls.append(context)
        return context


class _Sink(BaseOutputTransport):
    """Actual output sender with an in-memory final sink and explicit scripted results."""

    def __init__(self, params):
        """Retain accepted bytes/frames; no external I/O or replacement sender is used."""
        super().__init__(params)
        self.writes = []
        self.attempts = []
        self.forwarded = []
        self.results = deque()
        self.tail_results = deque()
        self.tail_wait = None
        self.changed = asyncio.Event()

    async def start(self, frame):
        """Start the existing destination-specific native senders."""
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        """Record a real native write; scripted discard/False remain distinct outcomes."""
        self.attempts.append(frame)
        result = self.results.popleft() if self.results else True
        if result is True:
            self.writes.append(frame)
        self.changed.set()
        return result

    async def write_transport_frame(self, frame):
        """Script only the guarded native TTS completion hook; all other frames remain no-op."""
        if isinstance(frame, TTSStoppedFrame) and self.tail_wait is not None:
            await self.tail_wait.wait()
        if isinstance(frame, TTSStoppedFrame) and self.tail_results:
            return self.tail_results.popleft()
        return None

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Observe forwarding attempts while retaining the actual base push implementation."""
        if self._should_push_frame(frame, direction):
            self.forwarded.append((frame, direction))
        await super().push_frame(frame, direction)


class _PausedMixer(BaseAudioMixer):
    """Pause only real speech mixing, leaving the native background branch runnable."""

    def __init__(self):
        """Expose exact synchronization points without changing output bytes."""
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self, sample_rate):
        """Accept the output rate; this fixture allocates no external resource."""
        pass

    async def stop(self):
        """Release a paused mix so cancellation/teardown cannot strand the test."""
        self.release.set()

    async def process_frame(self, frame):
        """Retain the mixer control interface without installing effects."""
        pass

    async def mix(self, audio):
        """Pause non-silence at the genuine mixer await, then return identical PCM."""
        if any(audio):
            self.entered.set()
            await self.release.wait()
        return audio


class _UnsupportedFilter(BaseAudioFilter):
    """Represent a stateful filter without a provenance-boundary reset contract."""

    async def start(self, sample_rate):
        """Provide the ordinary filter lifecycle; guarded preflight must not call it."""
        pass

    async def stop(self):
        """Provide ordinary teardown without inventing a reset operation."""
        pass

    async def process_frame(self, frame):
        """Accept existing filter controls without granting guarded capability."""
        pass

    async def filter(self, audio):
        """Return PCM while exposing no promise that retained samples can be discarded."""
        return audio


class _Probe(FrameProcessor):
    """Record the actual adjacent queue insertion without connecting a provider."""

    def __init__(self):
        """Collect only frames that pass the native forwarding boundary."""
        super().__init__()
        self.received = []

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Represent the adjacent queue's effect and retain the original object."""
        self.received.append(frame)


class _PausedObserver(BaseObserver):
    """Pause a text push after observation but before actual adjacent queue insertion."""

    def __init__(self):
        """Expose an already-observed boundary which cannot be retrospectively retracted."""
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.observed = []

    async def on_push_frame(self, data: FramePushed):
        """Observe any text payload, including synthesis bypasses, before the queue effect."""
        if isinstance(data.frame, TextFrame):
            self.observed.append(data.frame)
            self.entered.set()
            await self.release.wait()


def _audio(source="one", audio=None, rate=16000, context_id="tts-one"):
    """Create producer-stamped speech; default bytes fill one native 10 ms chunk."""
    frame = TTSAudioRawFrame(
        audio=b"\x01\x02" * 160 if audio is None else audio,
        sample_rate=rate,
        num_channels=1,
        context_id=context_id,
    )
    frame.test_source = source
    return frame


def _stop(source="one"):
    """Create the matching ordered TTS cleanup frame without rewriting provenance."""
    frame = TTSStoppedFrame(context_id="tts-one")
    frame.test_source = source
    return frame


def _text_variants():
    """Return fresh plain/model/aggregated/TTS payloads, including the real skip-TTS path."""
    model = LLMTextFrame("synthetic model response without synthesis")
    model.skip_tts = True
    return (
        TextFrame("synthetic plain response"),
        model,
        AggregatedTextFrame("synthetic aggregated response", text_type="sentence"),
        TTSTextFrame("synthetic TTS response", text_type="sentence"),
    )


class TestAudioOutputPolicy(unittest.IsolatedAsyncioTestCase):
    """Qualify native owner boundaries, stale skips and ordinary compatibility offline."""

    async def asyncSetUp(self):
        """Track every actual sender and manager for deterministic test teardown."""
        self.transports = []
        self.managers = []

    async def asyncTearDown(self):
        """Cancel actual queued tasks and release processors even after an assertion failure."""
        for transport in self.transports:
            await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
            await transport.cleanup()
        for manager in self.managers:
            for task in list(manager.current_tasks()):
                await manager.cancel_task(task)

    async def make_transport(self, sources=None, observer=None, **kwargs):
        """Start maintained output tasks; optional policy installs before setup as required."""
        params = TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=1,
            audio_out_end_silence_secs=0,
            audio_out_sleep_between_failures=0,
            **kwargs,
        )
        transport = _Sink(params)
        if sources is not None:
            transport.set_audio_output_policy(sources.policy)
        manager = TaskManager(loop=asyncio.get_running_loop())
        self.managers.append(manager)
        self.transports.append(transport)
        clock = SystemClock()
        clock.start()
        await transport.setup(
            FrameProcessorSetup(
                clock=clock,
                task_manager=manager,
                pipeline_worker=SimpleNamespace(app_resources=None),
                observer=observer,
            )
        )
        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        return transport

    async def wait_writes(self, transport, count):
        """Wait for actual native write attempts with a finite offline deadline."""
        async with asyncio.timeout(1):
            while len(transport.attempts) < count:
                transport.changed.clear()
                if len(transport.attempts) < count:
                    await transport.changed.wait()

    async def test_current_source_survives_real_chunking_and_context_copy(self):
        """Full chunks retain original source/TTS context outside serialized dataclass fields."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        frame = _audio(audio=b"\x03\x04" * 320)
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        self.assertEqual(b"".join(f.audio for f in transport.writes), frame.audio)
        self.assertTrue(all(f.test_source == "one" for f in transport.writes))
        self.assertTrue(all(f.context_id == "tts-one" for f in transport.writes))
        self.assertNotIn(AUDIO_OUTPUT_CONTEXT_ATTRIBUTE, dataclasses.asdict(transport.writes[0]))

    async def test_missing_source_is_captured_once_and_never_repaired(self):
        """Adding a later source cannot upgrade a frame whose original capture was missing."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        frame = _audio(source=None)
        self.assertIsNone(transport.capture_audio_output_context(frame))
        frame.test_source = "one"
        self.assertIsNone(transport.capture_audio_output_context(frame))
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(transport.attempts, [])

    async def test_partial_chunks_of_two_current_owners_never_mix(self):
        """An eligible old remainder is source-padded before another owner's PCM begins."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        one = b"\x01\x02" * 80
        two = b"\x03\x04" * 80
        await transport.process_frame(_audio("one", one), FrameDirection.DOWNSTREAM)
        await transport.process_frame(_audio("two", two), FrameDirection.DOWNSTREAM)
        await transport.process_frame(_stop("two"), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        self.assertEqual(
            [(f.test_source, f.audio) for f in transport.writes],
            [("one", one + bytes(160)), ("two", two + bytes(160))],
        )

    async def test_stale_partial_remainder_is_not_prefixed_to_new_owner(self):
        """Revoked buffered PCM disappears without padding or relabeling it as current."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        await transport.process_frame(_audio("one", b"\x01\x02" * 80), FrameDirection.DOWNSTREAM)
        sources.allowed.remove("one")
        two = b"\x03\x04" * 80
        await transport.process_frame(_audio("two", two), FrameDirection.DOWNSTREAM)
        await transport.process_frame(_stop("two"), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 1)
        self.assertEqual(
            [(f.test_source, f.audio) for f in transport.writes], [("two", two + bytes(160))]
        )

    async def test_delayed_stop_cannot_flush_later_tts_context_with_same_owner(self):
        """An old context's cleanup cannot consume the newer context's partial audio/filter tail."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        one, two = b"\x01\x02" * 80, b"\x03\x04" * 80
        await transport.process_frame(_audio("one", one), FrameDirection.DOWNSTREAM)
        await transport.process_frame(
            _audio("one", two, context_id="tts-two"), FrameDirection.DOWNSTREAM
        )
        await transport.process_frame(_stop("one"), FrameDirection.DOWNSTREAM)
        self.assertEqual(transport._media_senders[None]._buffered_audio_bytes, len(two))
        final = TTSStoppedFrame(context_id="tts-two")
        final.test_source = "one"
        await transport.process_frame(final, FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        self.assertEqual(
            [(f.context_id, f.audio) for f in transport.writes],
            [("tts-one", one + bytes(160)), ("tts-two", two + bytes(160))],
        )

    async def test_real_resampler_tail_does_not_cross_revoked_owner_boundary(self):
        """A revoked positive tail cannot contaminate current silence beyond exact int16 dither."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        old_audio = b"\xff\x7f" * 40
        await transport.process_frame(
            _audio("one", old_audio, rate=24000), FrameDirection.DOWNSTREAM
        )
        sources.allowed.remove("one")
        silence = bytes(1200 * 2)
        await transport.process_frame(_audio("two", silence, rate=24000), FrameDirection.DOWNSTREAM)
        await transport.process_frame(_stop("two"), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 5)
        async with asyncio.timeout(1):
            await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(len(transport.writes), 5)
        self.assertTrue(all(frame.test_source == "two" for frame in transport.writes))
        self.assertTrue(all(len(frame.audio) == 320 for frame in transport.writes))
        actual = b"".join(frame.audio for frame in transport.writes)
        self.assertEqual(len(actual), 800 * 2)
        # python-soxr v1.0.0 uses default int16 soxr_io_spec; libsoxr's
        # SOXR_TPDF adds ((ran1 & 31) - (ran2 & 31)) / 32 before rounding.
        # For exact zero input, the only possible int16 samples are -1, 0, 1.
        # Independent streams therefore must not be compared byte-for-byte.
        # Primary implementations: dofuuz/python-soxr v1.0.0 src/soxr_ext.cpp
        # (CSoxr), and chirlu/soxr src/rint-clip.h (DITHERING macro).
        self.assertTrue(np.isin(np.frombuffer(actual, dtype=np.int16), [-1, 0, 1]).all())

        # The identical real filter without reset is an independent negative
        # control: it retains old samples and a strong nonzero prefix. Both
        # assertions above reject this failure, even with dither enabled.
        contaminated = create_stream_resampler(clear_after_secs=None)
        leaked = await contaminated.resample(old_audio, 24000, 16000)
        leaked += await contaminated.resample(silence, 24000, 16000)
        leaked += await contaminated.flush()
        self.assertNotEqual(len(leaked), len(actual))
        self.assertGreater(
            int(np.max(np.abs(np.frombuffer(leaked, dtype=np.int16).astype(np.int32)))),
            1000,
        )

    async def test_revocation_during_resample_discards_result_and_tail(self):
        """The original owner is rechecked after the real resampler await before buffering."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        original = sender._resampler.resample
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused(audio, in_rate, out_rate):
            """Pause at resampling, then retain the actual native PCM conversion."""
            entered.set()
            await release.wait()
            return await original(audio, in_rate, out_rate)

        sender._resampler.resample = paused
        task = transport.create_task(sender.handle_audio_frame(_audio()))
        await asyncio.wait_for(entered.wait(), 1)
        sources.allowed.remove("one")
        release.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(sender._buffered_audio_bytes, 0)
        self.assertEqual(transport.attempts, [])

    async def test_revocation_during_mix_drops_speech_without_relabeling_ambience(self):
        """A speech-owned mixture cannot acquire the independently allowed native mixer source."""
        sources = _Sources()
        mixer = _PausedMixer()
        transport = await self.make_transport(sources, audio_out_mixer=mixer)
        await transport.process_frame(_audio(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(mixer.entered.wait(), 1)
        sources.allowed.remove("one")
        mixer.release.set()
        async with asyncio.timeout(1):
            while not any(f.test_source == ("mixer", None) for f in transport.writes):
                await asyncio.sleep(0)
        self.assertFalse(any(f.test_source == "one" for f in transport.writes))
        self.assertTrue(all(not any(f.audio) for f in transport.writes))

    async def test_native_terminal_silence_has_destination_owned_source(self):
        """Actual terminal silence is classified at its native producer, not as procedural speech."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        await transport._media_senders[None]._send_silence(1)
        self.assertEqual(transport.writes[0].test_source, ("end_silence", None))
        self.assertFalse(any(transport.writes[0].audio))
        self.assertIn(("end_silence", None), sources.native_calls)

    async def test_named_destination_retains_independent_pcm_and_native_context(self):
        """Every actual destination sender carries its context, including generated terminal silence."""
        sources = _Sources()
        sources.allowed.add(("end_silence", "aux"))
        transport = await self.make_transport(sources, audio_out_destinations=["aux"])
        frame = _audio("two")
        frame.transport_destination = "aux"
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 1)
        self.assertEqual(transport.writes[0].transport_destination, "aux")
        await transport._media_senders["aux"]._send_silence(1)
        self.assertEqual(transport.writes[-1].test_source, ("end_silence", "aux"))
        self.assertEqual(transport.writes[-1].transport_destination, "aux")

    async def test_intentional_discards_neither_fail_nor_forward_audio(self):
        """Repeated native stale-discard results do not trigger the I/O watchdog or played forwarding."""
        sources = _Sources()
        transport = await self.make_transport(sources, audio_out_max_consecutive_failures=2)
        transport.results.extend([AUDIO_OUTPUT_DISCARDED] * 5)
        await transport.process_frame(_audio(audio=b"\x01\x02" * 800), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 5)
        await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(transport.writes, [])
        self.assertFalse(
            any(
                isinstance(f, (OutputAudioRawFrame, CancelWorkerFrame))
                for f, _ in transport.forwarded
            )
        )
        self.assertTrue(transport.is_usable)

    async def test_discard_does_not_reset_genuine_failure_count(self):
        """A discarded frame between two real False writes cannot hide the existing failure threshold."""
        sources = _Sources()
        transport = await self.make_transport(sources, audio_out_max_consecutive_failures=2)
        transport.results.extend([False, AUDIO_OUTPUT_DISCARDED, False])
        await transport.process_frame(_audio(audio=b"\x01\x02" * 480), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 3)
        async with asyncio.timeout(1):
            while not any(isinstance(f, CancelWorkerFrame) for f, _ in transport.forwarded):
                await asyncio.sleep(0)

    async def test_guarded_tail_failure_notifies_before_ordered_speech_boundary(self):
        """Prior accepted PCM cannot make a failed serializer tail appear PLAYED at completion."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        observed = []

        async def failed(processor, frame):
            """Represent the app's awaited failure mark, retaining the exact failing cleanup owner."""
            observed.append(("failed", frame))

        async def after_output(processor, frame):
            """Represent ordered playback completion after actual output forwarding."""
            if isinstance(frame, SpeechBoundaryFrame):
                observed.append(("boundary", frame))

        transport.add_event_handler("on_audio_output_failed", failed)
        transport.add_event_handler("on_after_push_frame", after_output)
        transport.tail_results.append(False)
        await transport.process_frame(_audio(), FrameDirection.DOWNSTREAM)
        stopped = _stop()
        await transport.process_frame(stopped, FrameDirection.DOWNSTREAM)
        boundary = SpeechBoundaryFrame(speech_id="speech-one", beginning=False)
        await transport.process_frame(boundary, FrameDirection.DOWNSTREAM)
        await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(observed, [("failed", stopped), ("boundary", boundary)])

    async def test_tail_discard_does_not_notify_failure_or_suppress_cleanup(self):
        """A stale tail remains intentional cleanup with no invented write failure."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        failed = []

        async def capture_failure(processor, frame):
            """Collect only actual native failure notifications."""
            failed.append(frame)

        transport.add_event_handler("on_audio_output_failed", capture_failure)
        transport.tail_results.append(AUDIO_OUTPUT_DISCARDED)
        stopped = _stop()
        await transport.process_frame(stopped, FrameDirection.DOWNSTREAM)
        await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(failed, [])
        self.assertTrue(any(f is stopped for f, _ in transport.forwarded))

    async def test_guarded_tail_timeout_is_bounded_and_notifies_failure(self):
        """A paused native completion flush cannot strand shutdown or report successful speech."""
        sources = _Sources()
        transport = await self.make_transport(sources, audio_out_write_timeout_secs=0.01)
        transport.tail_wait = asyncio.Event()
        failed = []

        async def record_failure(processor, frame):
            """Capture the awaited failure before later cleanup gets a chance to complete speech."""
            failed.append(frame)

        transport.add_event_handler("on_audio_output_failed", record_failure)
        stopped = _stop()
        await transport.process_frame(stopped, FrameDirection.DOWNSTREAM)
        async with asyncio.timeout(1):
            await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(failed, [stopped])
        self.assertFalse(transport.is_usable)

    async def test_actual_audio_failure_emits_one_event_but_discard_emits_none(self):
        """Only a genuine failed native write notifies the application playback owner."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        failed = []

        async def record_failure(processor, frame):
            """Retain exact failed audio frames without inventing outcomes for stale skips."""
            failed.append(frame)

        transport.add_event_handler("on_audio_output_failed", record_failure)
        transport.results.extend([False, AUDIO_OUTPUT_DISCARDED])
        await transport.process_frame(_audio(audio=b"\x01\x02" * 320), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        async with asyncio.timeout(1):
            await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(failed, [transport.attempts[0]])

    async def test_clock_wait_rechecks_source_and_keeps_cleanup(self):
        """Timed text becomes stale while waiting; its ordered speech cleanup still reaches output."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        entered = asyncio.Event()
        clock = transport.get_clock()
        original_time = clock.get_time

        def observed_time():
            """Observe the real clock sender immediately before its native PTS sleep."""
            if asyncio.current_task() is sender._clock_task:
                entered.set()
            return original_time()

        clock.get_time = observed_time
        text = TTSTextFrame("old speech", text_type="sentence")
        text.test_source = "one"
        text.pts = transport.get_clock().get_time() + 30_000_000
        await transport.process_frame(text, FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(entered.wait(), 1)
        sources.allowed.remove("one")
        boundary = SpeechBoundaryFrame(speech_id="speech-one", beginning=False)
        boundary.test_source = "one"
        await transport.process_frame(boundary, FrameDirection.DOWNSTREAM)
        await transport.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        self.assertFalse(any(f is text for f, _ in transport.forwarded))
        self.assertTrue(any(f is boundary for f, _ in transport.forwarded))
        self.assertEqual(transport._media_senders[None]._audio_queue._unfinished_tasks, 0)

    async def assert_observed_text_path(self, text, sources=None, *, revoke=False):
        """Exercise the real audio queue/observer/adjacent queue and output callback for one source.

        A revoked payload can remain in the observer's already-recorded input,
        but cannot enter the adjacent queue or signal after-push completion.
        Current guarded and ordinary payloads must reach both exactly once.
        """
        observer = _PausedObserver()
        transport = await self.make_transport(sources, observer)
        probe = _Probe()
        transport.link(probe)
        after = []

        async def after_push(processor, frame):
            """Record the actual output callback used by playback accounting."""
            after.append(frame)

        transport.add_event_handler("on_after_push_frame", after_push)
        if sources is not None:
            text.test_source = "one"
        await transport.process_frame(text, FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(observer.entered.wait(), 1)
        if revoke:
            self.assertIsNotNone(sources)
            sources.allowed.remove("one")
        observer.release.set()
        async with asyncio.timeout(1):
            await transport._media_senders[None]._audio_queue.join()
        self.assertEqual(observer.observed, [text])
        self.assertEqual(probe.received, [] if revoke else [text])
        self.assertEqual(after, [] if revoke else [text])
        if sources is None:
            self.assertFalse(hasattr(text, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE))
        else:
            self.assertEqual(getattr(text, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE), "one")

    async def test_observer_await_blocks_all_stale_text_family_payloads(self):
        """Plain, skip-TTS model, aggregated and TTS text cannot cross a revoked native boundary."""
        for text in _text_variants():
            with self.subTest(frame_type=type(text).__name__):
                await self.assert_observed_text_path(text, _Sources(), revoke=True)

    async def test_observer_await_accepts_all_current_text_family_payloads_once(self):
        """Each current text subtype retains its source and reaches actual queue/callback once."""
        for text in _text_variants():
            with self.subTest(frame_type=type(text).__name__):
                await self.assert_observed_text_path(text, _Sources())

    async def test_ordinary_text_family_forwarding_remains_source_free(self):
        """Without a policy, every text subtype retains its native queue/event behavior."""
        for text in _text_variants():
            with self.subTest(frame_type=type(text).__name__):
                await self.assert_observed_text_path(text)

    async def test_ordinary_output_preserves_bytes_without_policy_metadata(self):
        """The omitted-policy path retains exact PCM and permits original source-free frames."""
        transport = await self.make_transport()
        raw = OutputAudioRawFrame(audio=b"\x05\x06" * 320, sample_rate=16000, num_channels=1)
        await transport.process_frame(raw, FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        self.assertEqual(b"".join(f.audio for f in transport.writes), raw.audio)
        self.assertFalse(any(hasattr(f, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE) for f in transport.writes))

    async def test_policy_cannot_replace_started_sender_or_existing_policy(self):
        """Installation is fixed before setup; active queues cannot be retroactively relabeled."""
        sources = _Sources()
        transport = _Sink(TransportParams(audio_out_enabled=True))
        transport.set_audio_output_policy(sources.policy)
        with self.assertRaises(RuntimeError):
            transport.set_audio_output_policy(_Sources().policy)
        started = await self.make_transport(sources)
        with self.assertRaises(RuntimeError):
            started.set_audio_output_policy(sources.policy)

    async def test_guarded_filter_preflight_is_pure_and_ordinary_filter_remains_available(self):
        """Unsupported reset capability denies before setup, without removing normal filter support."""
        transport = _Sink(
            TransportParams(audio_out_enabled=True, audio_out_filter=_UnsupportedFilter())
        )
        with self.assertRaises(ValueError):
            transport.validate_audio_output_policy()
        with self.assertRaises(ValueError):
            transport.set_audio_output_policy(_Sources().policy)
        self.assertFalse(transport.audio_output_policy_enabled)
        ordinary = await self.make_transport(audio_out_filter=_UnsupportedFilter())
        await ordinary.process_frame(_audio(source=None), FrameDirection.DOWNSTREAM)
        await self.wait_writes(ordinary, 1)

    async def test_invalid_callbacks_and_non_boolean_verdicts_never_authorize(self):
        """Async/non-callable policies reject preflight and a truthy non-bool verdict fails closed."""
        sources = _Sources()
        transport = _Sink(TransportParams(audio_out_enabled=True))

        async def asynchronous(frame):
            """Represent an invalid callback requiring an await at a pure effect boundary."""
            return "one"

        with self.assertRaises(TypeError):
            transport.validate_audio_output_policy(
                AudioOutputPolicy(asynchronous, sources.copy, sources.is_current, sources.native)
            )
        with self.assertRaises(TypeError):
            transport.validate_audio_output_policy(
                AudioOutputPolicy(None, sources.copy, sources.is_current, sources.native)
            )

        def truthy(context):
            """Return an invalid truthy value which cannot grant output eligibility."""
            return "yes"

        transport.set_audio_output_policy(
            AudioOutputPolicy(sources.capture, sources.copy, truthy, sources.native)
        )
        self.assertFalse(transport.audio_output_context_is_current("one"))

    async def test_interruption_preserves_current_uninterruptible_owner_remainder(self):
        """Keep the exact owner on a partial uninterruptible chunk until its real TTS stop."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        audio = _audio(audio=b"\x01\x02" * (sender.audio_chunk_size * 3 // 4))
        audio.interruptible = False
        await transport.process_frame(audio, FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 1)
        self.assertEqual(sender._buffered_audio_bytes, sender.audio_chunk_size // 2)

        await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        self.assertTrue(sender._bot_speaking)
        self.assertEqual(sender._buffered_audio_bytes, sender.audio_chunk_size // 2)
        self.assertEqual(sender._audio_runs[0][2], "one")
        await transport.process_frame(_stop(), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 2)
        self.assertEqual(
            b"".join(frame.audio for frame in transport.writes),
            audio.audio + bytes(sender.audio_chunk_size // 2),
        )
        self.assertTrue(all(frame.test_source == "one" for frame in transport.writes))

    async def test_interruption_does_not_preserve_revoked_uninterruptible_owner(self):
        """An interruption flag cannot make revoked buffered speech eligible for a new visit."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        old = _audio(audio=b"\x01\x02" * (sender.audio_chunk_size // 4))
        old.interruptible = False
        await transport.process_frame(old, FrameDirection.DOWNSTREAM)
        sources.allowed.remove("one")

        await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        self.assertEqual(sender._buffered_audio_bytes, 0)
        await transport.process_frame(_audio("two"), FrameDirection.DOWNSTREAM)
        await self.wait_writes(transport, 1)

        self.assertEqual(len(transport.writes), 1)
        self.assertEqual(transport.writes[0].test_source, "two")
        self.assertEqual(transport.writes[0].audio, _audio("two").audio)

    async def test_revoked_queued_protected_audio_does_not_suppress_interrupted_stop(self):
        """A stale protected queued chunk cannot hide the stop/id while a native write awaits."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        entered, release = asyncio.Event(), asyncio.Event()

        async def write(frame):
            """Pause the real native consumer without replacing its audio queue/lifecycle."""
            entered.set()
            await release.wait()
            return True

        transport.write_audio_frame = write
        try:
            await transport.process_frame(_audio(), FrameDirection.DOWNSTREAM)
            await asyncio.wait_for(entered.wait(), timeout=1)
            queued = _audio()
            queued.interruptible = False
            await transport.process_frame(queued, FrameDirection.DOWNSTREAM)
            self.assertTrue(sender._audio_queue.has_frame(OutputAudioRawFrame))
            sources.allowed.remove("one")
            interruption = InterruptionFrame()

            await transport.process_frame(interruption, FrameDirection.DOWNSTREAM)

            self.assertFalse(sender._bot_speaking)
            self.assertFalse(sender._audio_queue.has_frame(OutputAudioRawFrame))
            stops = [
                frame for frame, _ in transport.forwarded
                if isinstance(frame, BotStoppedSpeakingFrame)
            ]
            self.assertEqual(len(stops), 2)
            self.assertTrue(all(frame.interrupted for frame in stops))
            self.assertTrue(all(
                frame.interruption_id == interruption.interruption_id for frame in stops
            ))
        finally:
            release.set()

    async def test_revoked_protected_clock_work_cannot_block_current_due_text(self):
        """Revoked waiting/queued protected text gives the actual clock to current due text."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        now = transport.get_clock().get_time()
        pending = TTSTextFrame("old waiting", "sentence")
        pending.test_source, pending.interruptible, pending.pts = "one", False, now + 10_000_000_000
        await transport.process_frame(pending, FrameDirection.DOWNSTREAM)
        async with asyncio.timeout(1):
            while sender._clock_frame is not pending:
                await asyncio.sleep(0)
        queued = TTSTextFrame("old queued", "sentence")
        queued.test_source, queued.interruptible, queued.pts = "one", False, now + 30_000_000
        current = TTSTextFrame("current due", "sentence")
        current.test_source, current.interruptible, current.pts = "two", False, now + 20_000_000
        await transport.process_frame(queued, FrameDirection.DOWNSTREAM)
        await transport.process_frame(current, FrameDirection.DOWNSTREAM)
        sources.allowed.remove("one")

        await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        async with asyncio.timeout(0.25):
            while not any(frame is current for frame, _ in transport.forwarded):
                await asyncio.sleep(0.001)
        self.assertFalse(any(frame is pending or frame is queued for frame, _ in transport.forwarded))
        await asyncio.wait_for(sender._clock_queue.join(), timeout=0.25)

    async def test_cancelled_clock_wait_is_acknowledged_once_across_restarts(self):
        """Persistent clock queues can join after repeated cancellation of actual timed dequeues."""
        sources = _Sources()
        transport = await self.make_transport(sources)
        sender = transport._media_senders[None]
        for index in range(2):
            pending = TTSTextFrame(f"waiting {index}", "sentence")
            pending.test_source = "one"
            pending.pts = transport.get_clock().get_time() + 10_000_000_000
            await transport.process_frame(pending, FrameDirection.DOWNSTREAM)
            async with asyncio.timeout(1):
                while sender._clock_frame is not pending:
                    await asyncio.sleep(0)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            await asyncio.wait_for(sender._clock_queue.join(), timeout=0.25)
            self.assertEqual(sender._clock_queue._unfinished_tasks, 0)
            self.assertFalse(sender._clock_task.done())

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base output transport implementation for Pipecat.

This module provides the BaseOutputTransport class which handles audio and video
output processing, including frame buffering, mixing, timing, and media streaming.
An optional per-instance policy preserves opaque audio provenance through native
buffers and queues. Intentional policy discards are separate from transport
failures, and callers which omit the policy retain the ordinary playback path.
"""

import asyncio
import inspect
import itertools
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger
from PIL import Image

from pipecat.audio.dtmf.utils import load_dtmf_audio
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.utils import create_stream_resampler, is_silence, scale_audio
from pipecat.frames.frames import (
    AggregatedTextProgressFrame,
    AssistantImageRawFrame,
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    FilterControlFrame,
    Frame,
    InterruptionFrame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    SpriteFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VolumeFrame,
    VolumeGainFrame,
)
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.deprecation import warn_deprecated
from pipecat.utils.frame_queue import FramePriorityQueue, FrameQueue
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.35
# Public compatibility constant used by Tavus; base transports use per-instance settings.
BOT_VAD_STOP_FALLBACK_SECS = 3

AUDIO_OUTPUT_CONTEXT_ATTRIBUTE = "_pipecat_audio_output_context"
"""Private frame metadata; opaque ownership never enters dataclass wire formats."""

_AUDIO_OUTPUT_UNSET = object()


class AudioOutputDiscarded(Enum):
    """Remaining stale output was discarded, distinct from a failed write.

    Earlier packets may already have been accepted; this result never retracts
    them or proves that remote playout did not happen.
    """

    DISCARDED = "discarded"


AUDIO_OUTPUT_DISCARDED = AudioOutputDiscarded.DISCARDED


@dataclass(frozen=True)
class AudioOutputPolicy:
    """Code-owned callbacks for retaining immutable audio source context.

    The callbacks are synchronous. ``capture`` reads an already issued source;
    it must never repair a missing source from whichever producer is current.
    ``copy`` stamps the exact captured context onto a derived frame. ``is_current``
    checks whether that original source can still be forwarded. ``native_context``
    supplies a source only at an actual native ``mixer``, ``end_silence`` or
    ``track_silence`` producer, scoped to the provided destination. Contexts must
    be immutable and support equality so distinct owners cannot share a chunk.
    This metadata neither grants execution nor records completed delivery.

    Parameters:
        capture: Return the original opaque context, or None for missing provenance.
        copy: Stamp an exact context on a derived frame without changing ownership.
        is_current: Return literal True only while a retained source is eligible.
        native_context: Capture actual call-owned generated output before its awaits.
    """

    capture: Callable[[Frame], object | None]
    copy: Callable[[object | None, Frame], None]
    is_current: Callable[[object | None], bool]
    native_context: Callable[[str, str | None], object | None]


class BaseOutputTransport(FrameProcessor):
    """Base class for output transport implementations.

    Handles audio and video output processing including frame buffering, audio mixing,
    timing coordination, and media streaming. Supports multiple output destinations
    and provides interruption handling for real-time communication.

    Event handlers available:

    - on_audio_output_failed: A guarded native audio/tail write failed. The
      awaited handler runs before later ordered speech completion boundaries.
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the base output transport.

        Args:
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._params = params
        self._audio_output_policy: AudioOutputPolicy | None = None
        self._register_event_handler("on_audio_output_failed", sync=True)

        # Output sample rate. It will be initialized during setup.
        self._sample_rate = 0

        # We write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This helps with interruption handling. It will be
        # initialized during setup.
        self._audio_chunk_size = 0

        # We will have one media sender per output frame destination. This allow
        # us to send multiple streams at the same time if the transport allows
        # it.
        self._media_senders: dict[Any, BaseOutputTransport.MediaSender] = {}

        if params.video_out_bitrate is not None:
            warn_deprecated(
                "`TransportParams.video_out_bitrate` is deprecated since 1.1.0 and will be "
                "removed in 2.0.0. Use provider-specific settings instead.",
                stacklevel=2,
            )

    @property
    def audio_output_policy_enabled(self) -> bool:
        """Whether this instance requires retained output provenance."""
        return self._audio_output_policy is not None

    def validate_audio_output_policy(self, policy: AudioOutputPolicy | None = None) -> None:
        """Preflight guarded capability without starting processors or providers.

        Subclasses extend this check for serializer/track support. Output filters
        have no ownership-boundary reset contract, so they are unsupported only
        on this guarded path. Omitting a policy validates capability without
        installing anything; supplying it additionally validates its callbacks.
        """
        if self._params.audio_out_filter is not None:
            raise ValueError("Guarded audio output does not support stateful output filters")
        if policy is not None and (
            type(policy) is not AudioOutputPolicy
            or not all(
                callable(callback)
                and not inspect.iscoroutinefunction(callback)
                and not inspect.iscoroutinefunction(getattr(callback, "__call__", None))
                for callback in (
                    policy.capture,
                    policy.copy,
                    policy.is_current,
                    policy.native_context,
                )
            )
        ):
            raise TypeError("Audio output policy requires synchronous code-owned callbacks")

    def set_audio_output_policy(self, policy: AudioOutputPolicy) -> None:
        """Install one policy before setup; an active sender cannot be relabeled."""
        if self._sample_rate or self._media_senders:
            raise RuntimeError("Audio output policy must be installed before setup")
        if type(policy) is not AudioOutputPolicy:
            raise TypeError("Audio output policy must be an AudioOutputPolicy")
        if self._audio_output_policy is not None and self._audio_output_policy is not policy:
            raise RuntimeError("Audio output policy cannot be replaced")
        self.validate_audio_output_policy(policy)
        self._audio_output_policy = policy

    def capture_audio_output_context(self, frame: Frame) -> object | None:
        """Capture a frame once, including missing provenance, before any await."""
        if self._audio_output_policy is None:
            return None
        if hasattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE):
            return getattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE)
        context = self._audio_output_policy.capture(frame)
        if inspect.isawaitable(context):
            if inspect.iscoroutine(context):
                context.close()
            raise TypeError("Audio output capture must be synchronous")
        setattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE, context)
        return context

    def copy_audio_output_context(self, context: object | None, frame: Frame) -> None:
        """Carry the exact original context onto a native derived frame."""
        if self._audio_output_policy is not None:
            if (
                hasattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE)
                and getattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE) != context
            ):
                raise ValueError("Derived audio output cannot change its original source")
            result = self._audio_output_policy.copy(context, frame)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Audio output copying must be synchronous")
            if result is not None:
                raise TypeError("Audio output copying must return None")
            setattr(frame, AUDIO_OUTPUT_CONTEXT_ATTRIBUTE, context)

    def audio_output_context_is_current(self, context: object | None) -> bool:
        """Fail closed for stale contexts and policy errors; ordinary output is eligible."""
        if self._audio_output_policy is None:
            return True
        try:
            result = self._audio_output_policy.is_current(context)
            if inspect.iscoroutine(result):
                result.close()
            return result is True
        except Exception:
            return False

    def native_audio_output_context(
        self, kind: str, destination: str | None = None
    ) -> object | None:
        """Capture a trusted native producer without relabeling external speech."""
        if kind not in {"mixer", "end_silence", "track_silence"}:
            raise ValueError("Unknown native audio output producer")
        if self._audio_output_policy is None:
            return None
        context = self._audio_output_policy.native_context(kind, destination)
        if inspect.isawaitable(context):
            if inspect.iscoroutine(context):
                context.close()
            raise TypeError("Native audio output capture must be synchronous")
        return context

    def _audio_output_frame_is_current(self, frame: Frame) -> bool:
        """Fence audio and every text payload while allowing lifecycle cleanup through.

        Text can bypass synthesis or remain aggregated before reaching assistant
        context. Its source must therefore remain eligible independently of
        whether the frame was produced by a TTS service. The application policy
        classifies actual user-transcription/control origins; transport code does
        not infer permission from a text subclass or ``skip_tts`` flag.
        """
        if isinstance(
            frame,
            (
                OutputAudioRawFrame,
                TextFrame,
                AggregatedTextProgressFrame,
                OutputDTMFFrame,
                OutputDTMFUrgentFrame,
            ),
        ):
            return self.audio_output_context_is_current(self.capture_audio_output_context(frame))
        return True

    def _should_push_frame(self, frame: Frame, direction: FrameDirection) -> bool:
        """Recheck original payloads after native observer awaits; keep cleanup flowing."""
        return direction != FrameDirection.DOWNSTREAM or self._audio_output_frame_is_current(frame)

    @property
    def sample_rate(self) -> int:
        """Get the current audio sample rate.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def audio_chunk_size(self) -> int:
        """Get the audio chunk size for output processing.

        Returns:
            The size of audio chunks in bytes.
        """
        return self._audio_chunk_size

    @property
    def capabilities(self) -> BotCapabilities:
        """The media this transport sends to the user.

        Returns:
            The ``audio_out`` and ``video_out`` capabilities.
        """
        return BotCapabilities(
            audio_out=self._params.audio_out_enabled, video_out=self._params.video_out_enabled
        )

    def _should_forward_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Whether the media sender forwards an accepted audio frame downstream.

        Paced transports forward their accepted audio later, on their own clock.
        Keep that choice separate from write success so the failure watchdog does
        not treat accepted, buffered audio as a failed transport write.
        """
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sample_rate = self._params.audio_out_sample_rate or setup.audio_out_sample_rate

        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks

        # Start audio filter. Paired with the stop in cleanup(), which is the
        # only other call guaranteed to happen exactly once.
        if self._params.audio_out_filter:
            await self._params.audio_out_filter.start(self._sample_rate)

    async def cleanup(self):
        """Release output transport resources at teardown."""
        await super().cleanup()
        for _, sender in self._media_senders.items():
            await sender.cleanup()
        if self._params.audio_out_filter:
            await self._params.audio_out_filter.stop()

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Base hook for subclasses, which reach it through ``super()``. The
        transport's own audio configuration is resolved in :meth:`setup`.

        Args:
            frame: The start frame containing initialization parameters.
        """
        pass

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        for _, sender in self._media_senders.items():
            await sender.stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        for _, sender in self._media_senders.items():
            await sender.cancel(frame)

    async def set_transport_ready(self, frame: StartFrame):
        """Called when the transport is ready to stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        # Register destinations.
        for destination in self._params.audio_out_destinations:
            await self.register_audio_destination(destination)

        for destination in self._params.video_out_destinations:
            await self.register_video_destination(destination)

        # Start default media sender.
        self._media_senders[None] = BaseOutputTransport.MediaSender(
            self,
            destination=None,
            sample_rate=self.sample_rate,
            audio_chunk_size=self.audio_chunk_size,
            params=self._params,
        )
        await self._media_senders[None].start(frame)

        # Media senders already send both audio and video, so make sure we only
        # have one media server per shared name.
        destinations = list(
            set(self._params.audio_out_destinations + self._params.video_out_destinations)
        )

        # Start media senders.
        for destination in destinations:
            self._media_senders[destination] = BaseOutputTransport.MediaSender(
                self,
                destination=destination,
                sample_rate=self.sample_rate,
                audio_chunk_size=self.audio_chunk_size,
                params=self._params,
            )
            await self._media_senders[destination].start(frame)

        # Sending a frame indicating that the output transport is ready and able to receive frames.
        await self.push_frame(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message.

        Args:
            frame: The transport message frame to send.
        """
        pass

    async def register_video_destination(self, destination: str):
        """Register a video output destination.

        Args:
            destination: The destination identifier to register.
        """
        pass

    async def register_audio_destination(self, destination: str):
        """Register an audio output destination.

        Args:
            destination: The destination identifier to register.
        """
        pass

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the transport.

        Args:
            frame: The output video frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return False

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool | AudioOutputDiscarded:
        """Write an audio frame to the transport.

        Args:
            frame: The output audio frame to write.

        Returns:
            Whether the transport took the frame. A serializer that emits no
            payload has still taken it; see :meth:`FrameSerializer.serialize`.
            Guarded subclasses may return ``AUDIO_OUTPUT_DISCARDED`` when their
            last native sink check intentionally rejects stale queued output.
        """
        return False

    async def write_dtmf(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Write a DTMF tone using the transport's preferred method.

        Args:
            frame: The DTMF frame to write.
        """
        if not self._audio_output_frame_is_current(frame):
            return
        if self._supports_native_dtmf():
            await self._write_dtmf_native(frame)
        else:
            await self._write_dtmf_audio(frame)

    async def write_transport_frame(self, frame: Frame) -> bool | AudioOutputDiscarded | None:
        """Handle a queued frame after preceding audio has been sent.

        Override in transport subclasses to handle custom frame types that
        flow through the audio queue. Called by the media sender after the
        frame has waited for any preceding audio to finish.

        Args:
            frame: The frame to handle.

        Returns:
            None for the ordinary no-op hook. Guarded implementations can
            return a tail write result for an exact TTSStoppedFrame: True for
            accepted completion, False for an I/O failure, or the intentional
            discard singleton for a stale/mismatched source. Cleanup is not a
            claim that speech played.
        """
        pass

    def _supports_native_dtmf(self) -> bool:
        """Override in transport implementations that support native DTMF.

        Returns:
            True if the transport supports native DTMF, False otherwise.
        """
        return False

    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Override in transport implementations for native DTMF.

        Args:
            frame: The DTMF frame to write.
        """
        raise NotImplementedError("Transport claims native DTMF support but doesn't implement it")

    async def _write_dtmf_audio(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Generate and send audio tones for DTMF.

        Args:
            frame: The DTMF frame to write.
        """
        if not frame.buttons:
            return
        context = self.capture_audio_output_context(frame)
        for button in frame.buttons:
            if not self.audio_output_context_is_current(context):
                return
            dtmf_audio = await load_dtmf_audio(button, sample_rate=self._sample_rate)
            if not self.audio_output_context_is_current(context):
                return
            dtmf_audio_frame = OutputAudioRawFrame(
                audio=dtmf_audio, sample_rate=self._sample_rate, num_channels=1
            )
            self.copy_audio_output_context(context, dtmf_audio_frame)
            if self.audio_output_policy_enabled:
                dtmf_audio_frame.transport_destination = frame.transport_destination
                sender = self._media_senders.get(frame.transport_destination)
                if sender is None:
                    return
                await sender._internal_write_audio_frame(dtmf_audio_frame)
            else:
                await self.write_audio_frame(dtmf_audio_frame)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame downstream.

        Args:
            frame: The audio frame to send.
        """
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame downstream.

        Args:
            frame: The image frame to send.
        """
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle transport-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        self.capture_audio_output_context(frame)
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and not self._audio_output_frame_is_current(
            frame
        ):
            return

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            # Keep pushing EndFrame down so all the pipeline stops nicely.
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._handle_frame(frame)
        elif isinstance(frame, OutputTransportMessageUrgentFrame):
            await self.send_message(frame)
        elif isinstance(frame, OutputDTMFUrgentFrame):
            await self.write_dtmf(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: Frame):
        """Handle frames by routing them to appropriate media senders."""
        if frame.transport_destination not in self._media_senders:
            logger.warning(
                f"{self} destination [{frame.transport_destination}] not registered for frame {frame}"
            )
            return

        sender = self._media_senders[frame.transport_destination]

        if isinstance(frame, InterruptionFrame):
            await sender.handle_interruptions(frame)
        elif isinstance(frame, OutputAudioRawFrame):
            await sender.handle_audio_frame(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
            await sender.handle_image_frame(frame)
            if isinstance(frame, AssistantImageRawFrame):
                # This will push it further, to be handled by the assistant
                # aggregator, say
                await sender.handle_sync_frame(frame)
        elif isinstance(frame, MixerControlFrame):
            await sender.handle_mixer_control_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await sender.handle_tts_stopped(frame)
        elif frame.pts:
            await sender.handle_timed_frame(frame)
        else:
            await sender.handle_sync_frame(frame)

    #
    # Media Sender
    #

    class MediaSender:
        """Handles media streaming for a specific destination.

        Manages audio and video output processing including buffering, timing,
        mixing, and frame delivery for a single output destination.
        """

        def __init__(
            self,
            transport: "BaseOutputTransport",
            *,
            destination: str | None,
            sample_rate: int,
            audio_chunk_size: int,
            params: TransportParams,
        ):
            """Initialize the media sender.

            Args:
                transport: The parent transport instance.
                destination: The destination identifier for this sender.
                sample_rate: The audio sample rate in Hz.
                audio_chunk_size: The size of audio chunks in bytes.
                params: Transport configuration parameters.
            """
            self._transport = transport
            self._destination = destination
            self._sample_rate = sample_rate
            self._audio_chunk_size = audio_chunk_size
            self._params = params

            # This is to resize images. We only need to resize one image at a time.
            self._executor = ThreadPoolExecutor(max_workers=1)

            # Incoming audio waiting to be cut into chunks, as runs of audio that
            # came from frames with the same ``interruptible`` flag, so a chunk
            # keeps the flag of the audio it holds, along with the type of the
            # frames it was built from (so a flushed partial chunk can be
            # reconstructed as the same frame type).
            self._audio_runs: deque[tuple[bytearray, bool, object | None]] = deque()
            self._audio_buffer_cls: type[OutputAudioRawFrame] = OutputAudioRawFrame
            self._audio_buffer_context_id: str | None = None
            self._audio_stream_key: tuple | None = None
            self._resampler_context: object = _AUDIO_OUTPUT_UNSET
            self._resampler_input_rate: int | None = None
            self._resampler_uninterruptible = False
            # Playback resets and ingestion share the actual native filter. The
            # lock prevents an async owner boundary from racing a stop/reset.
            self._audio_ingest_lock = asyncio.Lock()
            self._bot_speech_context: object | None = None
            self._bot_speech_context_id: str | None = None

            # This will be used to resample incoming audio to the output sample
            # rate. Speech boundaries are signalled explicitly (see
            # `_enqueue_flushed_audio_buffer` and `_bot_stopped_speaking`), so
            # the resampler must not discard its filter tail on its own: it runs
            # ahead of playback and would otherwise treat the pause between two
            # TTS chunks as the end of the stream.
            self._resampler = create_stream_resampler(clear_after_secs=None)

            # The user can provide a single mixer, to be used by the default
            # destination, or a destination/mixer mapping.
            self._mixer: BaseAudioMixer | None = None

            # The output filter belongs to the default destination, like a
            # single mixer.
            self._filter = None if destination else params.audio_out_filter

            # The output volume (VolumeFrame) and the gain on it (VolumeGainFrame),
            # updated in order with the audio.
            self._volume = params.audio_out_volume
            self._volume_gain = 1.0

            # These are the images that we should send at our desired framerate.
            self._video_images = None

            # Indicates if the bot is currently speaking.
            self._bot_speaking = False
            # Indicates if TTS audio has been received since the last stop.
            self._tts_audio_received = False
            # Last time a BotSpeakingFrame was pushed.
            self._bot_speaking_frame_time = 0
            # How often a BotSpeakingFrame should be pushed (value should be
            # greater than the audio chunks to have any effect).
            self._bot_speaking_frame_period = 0.2
            # Last time the bot actually spoke.
            self._bot_speech_last_time = 0

            self._audio_task: asyncio.Task | None = None
            self._video_task: asyncio.Task | None = None
            self._clock_task: asyncio.Task | None = None

            # Timed frames as (timestamp, counter, frame), ordered by presentation
            # time. It outlives the clock task so an interruption can keep its
            # uninterruptible frames.
            self._clock_queue = FramePriorityQueue(frame_getter=lambda item: item[2])
            # The frame the clock task is waiting to push.
            self._clock_frame: Frame | None = None

            # If timestamps are equal, use this count to preserve the insertion order
            self._clock_queue_counter = itertools.count()

        @property
        def sample_rate(self) -> int:
            """Get the audio sample rate.

            Returns:
                The sample rate in Hz.
            """
            return self._sample_rate

        @property
        def audio_chunk_size(self) -> int:
            """Get the audio chunk size.

            Returns:
                The size of audio chunks in bytes.
            """
            return self._audio_chunk_size

        async def start(self, frame: StartFrame):
            """Start the media sender and initialize components.

            Args:
                frame: The start frame containing initialization parameters.
            """
            self._clear_audio_buffer()

            # Create all tasks.
            self._create_video_task()
            self._create_clock_task()
            self._create_audio_task()

            # Check if we have an audio mixer for our destination.
            if self._params.audio_out_mixer:
                if isinstance(self._params.audio_out_mixer, Mapping):
                    self._mixer = self._params.audio_out_mixer.get(self._destination, None)
                elif not self._destination:
                    # Only use the default mixer if we are the default destination.
                    self._mixer = self._params.audio_out_mixer

            # Start audio mixer.
            if self._mixer:
                await self._mixer.start(self._sample_rate)

        async def stop(self, frame: EndFrame):
            """Stop the media sender and cleanup resources.

            Args:
                frame: The end frame signaling sender shutdown.
            """
            # Let the sink tasks process the queue until they reach this EndFrame.
            await self._clock_queue.put((float("inf"), next(self._clock_queue_counter), frame))
            await self._audio_queue.put(frame)

            # At this point we have enqueued an EndFrame and we need to wait for
            # that EndFrame to be processed by the audio and clock tasks. We
            # also need to wait for these tasks before cancelling the video task
            # because it might be still rendering.
            #
            # A peer that stopped reading can no longer strand us here: every
            # write is bounded by `audio_out_write_timeout_secs`, and the first
            # timeout costs the transport its usability, so whatever is still
            # queued drains at once instead of paying the bound again.
            if self._audio_task:
                await self._audio_task
            if self._clock_task:
                await self._clock_task

            # Stop audio mixer.
            if self._mixer:
                await self._mixer.stop()

            # We can now cancel the video task.
            await self._cancel_video_task()

        async def cancel(self, frame: CancelFrame):
            """Cancel the media sender and stop all processing.

            Args:
                frame: The cancel frame signaling immediate cancellation.
            """
            await self.cleanup()

        async def cleanup(self):
            """Release media sender resources at teardown."""
            try:
                # Since we are cancelling everything it doesn't matter what task we cancel first.
                await self._cancel_audio_task()
                await self._cancel_clock_task()
                await self._cancel_video_task()

                # Stop audio mixer so it doesn't keep generating frames after cancellation.
                if self._mixer:
                    await self._mixer.stop()
            finally:
                self._executor.shutdown(wait=False)

        async def handle_interruptions(self, frame: InterruptionFrame):
            """Handle interruption events by restarting tasks and clearing buffers.

            Args:
                frame: The interruption that stopped playback.
            """
            # Keep the audio task running but drain all interruptible frames, so
            # the frame being written and the pending uninterruptible ones are
            # still delivered. With a mixer, cancelling the task would also stop
            # mixer-only output during the restart, causing an audible gap in
            # the background audio (made worse by telephony serializers that
            # clear the playout buffer on interruptions).
            self._audio_queue.reset(keep_frame=self._transport._audio_output_frame_is_current)
            # The same goes for the audio still waiting to fill a chunk.
            # Aayiq's runs retain an owner as well as interruptibility; the flag
            # cannot extend a revoked visit's authority to keep playing speech.
            self._audio_runs = deque(
                (audio, flag, context)
                for audio, flag, context in self._audio_runs
                if flag and self._transport.audio_output_context_is_current(context)
            )

            # Protected queued or waiting text loses its right to delay the
            # current visit as soon as its original source is revoked.
            self._clock_queue.reset(keep_frame=self._transport._audio_output_frame_is_current)
            if self._clock_frame and (
                self._clock_frame.interruptible
                or not self._transport._audio_output_frame_is_current(self._clock_frame)
            ):
                await self._cancel_clock_task()

            await self._cancel_video_task()

            # Create tasks.
            self._create_video_task()
            self._create_clock_task()

            # Uninterruptible speech retains playback, but an actual stop still
            # carries the interruption identity used by Aayiq's tool deferral.
            if not (self._audio_runs or self._audio_queue.has_frame(OutputAudioRawFrame)):
                await self._bot_stopped_speaking(
                    interrupted=True,
                    interruption_id=frame.interruption_id,
                )

        async def handle_audio_frame(self, frame: OutputAudioRawFrame):
            """Handle incoming audio frames by buffering and chunking.

            Args:
                frame: The output audio frame to handle.
            """
            if not self._params.audio_out_enabled:
                return
            context = self._transport.capture_audio_output_context(frame)
            if not self._transport.audio_output_context_is_current(context):
                return
            async with self._audio_ingest_lock:
                if not self._transport.audio_output_context_is_current(context):
                    return
                if self._transport.audio_output_policy_enabled:
                    stream_key = (
                        context,
                        type(frame),
                        getattr(frame, "context_id", None),
                        frame.sample_rate,
                        frame.num_channels,
                    )
                    if self._audio_stream_key is not None and stream_key != self._audio_stream_key:
                        await self._enqueue_flushed_audio_buffer()
                    if not self._transport.audio_output_context_is_current(context):
                        return
                    if self._resampler_input_rate not in (None, frame.sample_rate):
                        # Different producers can use different native rates.
                        # The previous source has already been flushed/reset.
                        self._resampler = create_stream_resampler(clear_after_secs=None)
                    self._audio_stream_key = stream_key
                    self._resampler_context = context
                    self._resampler_input_rate = frame.sample_rate
                    self._audio_buffer_context_id = getattr(frame, "context_id", None)
                self._resampler_uninterruptible = not frame.interruptible

                self._audio_buffer_cls = type(frame)
                resampled = await self._resampler.resample(
                    frame.audio, frame.sample_rate, self._sample_rate
                )
                if not self._transport.audio_output_context_is_current(context):
                    self._clear_audio_buffer()
                    await self._resampler.reset()
                    self._resampler_context = _AUDIO_OUTPUT_UNSET
                    self._audio_stream_key = None
                    return
                self._buffer_audio(
                    resampled, uninterruptible=not frame.interruptible, context=context
                )
                while self._buffered_audio_bytes >= self._audio_chunk_size:
                    audio, uninterruptible, chunk_context = self._take_audio_chunk()
                    if len(audio) < self._audio_chunk_size:
                        audio += bytes(self._audio_chunk_size - len(audio))
                    await self._enqueue_audio_chunk(
                        audio, uninterruptible, chunk_context, num_channels=frame.num_channels
                    )

        async def handle_image_frame(self, frame: OutputImageRawFrame | SpriteFrame):
            """Handle incoming image frames for video output.

            Args:
                frame: The output image or sprite frame to handle.
            """
            if not self._params.video_out_enabled:
                return

            if isinstance(frame, OutputImageRawFrame) and frame.sync_with_audio:
                # Route through the audio queue so the image is only
                # displayed after all preceding audio has been sent.
                await self._audio_queue.put(frame)
            elif self._params.video_out_is_live and isinstance(frame, OutputImageRawFrame):
                await self._video_queue.put(frame)
            elif isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            else:
                await self._set_video_images(frame.images)

        async def handle_timed_frame(self, frame: Frame):
            """Handle frames with presentation timestamps.

            Args:
                frame: The frame with timing information to handle.
            """
            self._transport.capture_audio_output_context(frame)
            if self._transport._audio_output_frame_is_current(frame):
                await self._clock_queue.put((frame.pts, next(self._clock_queue_counter), frame))

        async def handle_sync_frame(self, frame: Frame):
            """Handle frames that need synchronized processing.

            Args:
                frame: The frame to handle synchronously.
            """
            self._transport.capture_audio_output_context(frame)
            if self._transport._audio_output_frame_is_current(frame):
                await self._audio_queue.put(frame)

        async def handle_tts_stopped(self, frame: TTSStoppedFrame):
            """Queue a TTSStoppedFrame, flushing any trailing buffered audio first.

            `handle_audio_frame` only queues complete `audio_chunk_size` chunks,
            so up to one chunk's worth of trailing audio can still be sitting in
            `_audio_runs`. Queue it now (padded to a full chunk with silence)
            so it plays before the stop frame is handled, instead of being
            discarded when the buffer is cleared in `_bot_stopped_speaking`.

            Args:
                frame: The TTS stopped frame to queue.
            """
            context = self._transport.capture_audio_output_context(frame)
            async with self._audio_ingest_lock:
                if not self._transport.audio_output_policy_enabled or (
                    context == self._resampler_context
                    and frame.context_id == self._audio_buffer_context_id
                ):
                    await self._enqueue_flushed_audio_buffer()
            await self._audio_queue.put(frame)

        async def handle_mixer_control_frame(self, frame: MixerControlFrame):
            """Handle audio mixer control frames.

            Args:
                frame: The mixer control frame to handle.
            """
            if self._mixer:
                await self._mixer.process_frame(frame)

        #
        # Audio handling
        #

        def _create_audio_task(self):
            """Create the audio processing task."""
            if not self._audio_task:
                self._audio_queue = FrameQueue()
                self._audio_task = self._transport.create_task(self._audio_task_handler())

        async def _cancel_audio_task(self):
            """Cancel and cleanup the audio processing task."""
            if self._audio_task:
                await self._transport.cancel_task(self._audio_task)
                self._audio_task = None

        async def _bot_started_speaking(self):
            """Handle bot started speaking event."""
            if self._bot_speaking:
                return

            self._bot_speaking = True

            logger.debug(
                f"Bot{f' [{self._destination}]' if self._destination else ''} started speaking"
            )

            downstream_frame = BotStartedSpeakingFrame()
            downstream_frame.transport_destination = self._destination
            upstream_frame = BotStartedSpeakingFrame()
            upstream_frame.transport_destination = self._destination

            # Setting the siblings id
            upstream_frame.broadcast_sibling_id = downstream_frame.id
            downstream_frame.broadcast_sibling_id = upstream_frame.id

            await self._transport.push_frame(downstream_frame)
            await self._transport.push_frame(upstream_frame, FrameDirection.UPSTREAM)

        async def _enqueue_flushed_audio_buffer(self):
            """Queue every bit of audio still held back at the end of a speech run.

            That means both what the resampler is holding in its filter and
            what is left in `_audio_runs`, padded with silence to a full
            chunk. The audio goes out as frames of the same type (e.g.
            `TTSAudioRawFrame`) as the audio it was buffered from, on
            `_audio_queue`, so it takes the normal playback path (write, error
            handling, bot speaking tracking) like any other chunk, in order
            relative to whatever is queued after it (e.g. a TTSStoppedFrame).
            """
            # The resampler holds the tail of the audio it was fed, which
            # belongs at the end of this speech run, with the last run's flag.
            context = (
                None if self._resampler_context is _AUDIO_OUTPUT_UNSET else self._resampler_context
            )
            if not self._transport.audio_output_context_is_current(context):
                self._clear_audio_buffer()
                await self._resampler.reset()
                self._resampler_context = _AUDIO_OUTPUT_UNSET
                self._audio_stream_key = None
                return
            tail = await self._resampler.flush()
            if not self._transport.audio_output_context_is_current(context):
                self._clear_audio_buffer()
                await self._resampler.reset()
                self._resampler_context = _AUDIO_OUTPUT_UNSET
                self._audio_stream_key = None
                return
            last_uninterruptible = (
                self._resampler_uninterruptible
                if self._transport.audio_output_policy_enabled
                else self._audio_runs[-1][1]
                if self._audio_runs
                else False
            )
            self._buffer_audio(tail, uninterruptible=last_uninterruptible, context=context)

            if not self._audio_runs:
                return

            # The flushed tail can be longer than a chunk, so send whole chunks
            # first and pad only what's left over.
            while self._buffered_audio_bytes >= self._audio_chunk_size:
                audio, uninterruptible, chunk_context = self._take_audio_chunk()
                if len(audio) < self._audio_chunk_size:
                    audio += bytes(self._audio_chunk_size - len(audio))
                await self._enqueue_audio_chunk(audio, uninterruptible, chunk_context)

            if self._audio_runs:
                audio, uninterruptible, chunk_context = self._take_audio_chunk()
                padding = bytes(self._audio_chunk_size - len(audio))
                await self._enqueue_audio_chunk(audio + padding, uninterruptible, chunk_context)

        @property
        def _buffered_audio_bytes(self) -> int:
            """Count bytes in the existing ordered native audio runs."""
            return sum(len(audio) for audio, _, _ in self._audio_runs)

        def _buffer_audio(
            self, audio: bytes, *, uninterruptible: bool, context: object | None = None
        ):
            """Append PCM while keeping interruptibility and original owner boundaries."""
            if not audio:
                return
            if (
                self._audio_runs
                and self._audio_runs[-1][1] == uninterruptible
                and self._audio_runs[-1][2] == context
            ):
                self._audio_runs[-1][0].extend(audio)
            else:
                self._audio_runs.append((bytearray(audio), uninterruptible, context))

        def _take_audio_chunk(self) -> tuple[bytes, bool, object | None]:
            """Cut one owner's PCM and retain its interruptibility; never merge owners."""
            chunk = bytearray()
            uninterruptible = False
            context = self._audio_runs[0][2] if self._audio_runs else None
            while len(chunk) < self._audio_chunk_size and self._audio_runs:
                audio, run_uninterruptible, run_context = self._audio_runs[0]
                if run_context != context:
                    break
                uninterruptible = uninterruptible or run_uninterruptible
                needed = self._audio_chunk_size - len(chunk)
                if len(audio) <= needed:
                    chunk.extend(audio)
                    self._audio_runs.popleft()
                else:
                    chunk.extend(audio[:needed])
                    del audio[:needed]
            return bytes(chunk), uninterruptible, context

        def _clear_audio_buffer(self):
            """Discard pending PCM runs without assigning them another source."""
            self._audio_runs.clear()

        async def _enqueue_audio_chunk(
            self,
            audio: bytes,
            uninterruptible: bool,
            context: object | None = None,
            *,
            num_channels: int | None = None,
        ):
            """Queue a derived native chunk with the exact source which produced its bytes."""
            if not self._transport.audio_output_context_is_current(context):
                return
            frame = self._audio_buffer_cls(
                audio,
                sample_rate=self._sample_rate,
                num_channels=(
                    self._params.audio_out_channels if num_channels is None else num_channels
                ),
            )
            frame.transport_destination = self._destination
            frame.interruptible = not uninterruptible
            if self._transport.audio_output_policy_enabled and isinstance(frame, TTSAudioRawFrame):
                frame.context_id = self._audio_buffer_context_id
            self._transport.copy_audio_output_context(context, frame)
            await self._audio_queue.put(frame)

        async def _bot_stopped_speaking(
            self, *, interrupted: bool = False, interruption_id: int | None = None
        ):
            """Handle bot stopped speaking event."""
            if not self._bot_speaking:
                return

            self._bot_speaking = False
            self._tts_audio_received = False

            # Any remaining leftover here (e.g. from an interruption) is
            # discarded rather than flushed, since it's no longer wanted. The
            # same goes for the audio still inside the resampler, which would
            # otherwise be prepended to whatever the bot says next.
            if self._transport.audio_output_policy_enabled:
                async with self._audio_ingest_lock:
                    if (
                        self._resampler_context == self._bot_speech_context
                        and self._audio_buffer_context_id == self._bot_speech_context_id
                    ):
                        self._clear_audio_buffer()
                        await self._resampler.reset()
                        self._resampler_context = _AUDIO_OUTPUT_UNSET
                        self._audio_stream_key = None
            else:
                self._clear_audio_buffer()
                await self._resampler.reset()

            logger.debug(
                f"Bot{f' [{self._destination}]' if self._destination else ''} stopped speaking"
            )

            downstream_frame = BotStoppedSpeakingFrame(
                interrupted=interrupted, interruption_id=interruption_id
            )
            downstream_frame.transport_destination = self._destination
            upstream_frame = BotStoppedSpeakingFrame(
                interrupted=interrupted, interruption_id=interruption_id
            )
            upstream_frame.transport_destination = self._destination

            # Setting the siblings id
            upstream_frame.broadcast_sibling_id = downstream_frame.id
            downstream_frame.broadcast_sibling_id = upstream_frame.id

            await self._transport.push_frame(downstream_frame)
            await self._transport.push_frame(upstream_frame, FrameDirection.UPSTREAM)

        async def _bot_currently_speaking(self):
            """Handle bot speaking event."""
            await self._bot_started_speaking()

            diff_time = time.time() - self._bot_speaking_frame_time
            if diff_time >= self._bot_speaking_frame_period:
                await self._transport.broadcast_frame(BotSpeakingFrame)
                self._bot_speaking_frame_time = time.time()

            self._bot_speech_last_time = time.time()

        async def _maybe_bot_currently_speaking(self, frame: SpeechOutputAudioRawFrame):
            if not is_silence(frame.audio):
                await self._bot_currently_speaking()
            else:
                silence_duration = time.time() - self._bot_speech_last_time
                if silence_duration > BOT_VAD_STOP_SECS:
                    await self._bot_stopped_speaking()

        async def _handle_bot_speech(self, frame: Frame):
            """Update speech lifecycle only for audio eligible at this native boundary."""
            if not self._transport._audio_output_frame_is_current(frame):
                return
            self._bot_speech_context = self._transport.capture_audio_output_context(frame)
            self._bot_speech_context_id = getattr(frame, "context_id", None)
            # TTS case.
            if isinstance(frame, TTSAudioRawFrame):
                # We will only trigger bot stopped speaking based on the TTSStoppedFrame,
                # if we have received audio from TTS
                self._tts_audio_received = True
                await self._bot_currently_speaking()
            # Speech stream case.
            elif isinstance(frame, SpeechOutputAudioRawFrame):
                await self._maybe_bot_currently_speaking(frame)

        async def _handle_frame(self, frame: Frame) -> bool | AudioOutputDiscarded | None:
            """Handle various frame types with appropriate processing.

            Args:
                frame: The frame to handle.
            """
            if isinstance(frame, OutputAudioRawFrame):
                await self._handle_bot_speech(frame)
            elif isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            elif isinstance(frame, SpriteFrame):
                await self._set_video_images(frame.images)
            elif isinstance(frame, OutputTransportMessageFrame):
                await self._transport.send_message(frame)
            elif isinstance(frame, OutputDTMFFrame):
                await self._transport.write_dtmf(frame)
            elif isinstance(frame, TTSStoppedFrame):
                # Guarded serializers finish their exact source/context tail
                # before bot-stop accounting. Ordinary transport hooks retain
                # their original dispatch behavior.
                outcome = None
                if self._transport.audio_output_policy_enabled:
                    outcome = await self._internal_write_transport_frame(frame)
                # We will only trigger bot stopped speaking based on the TTSStoppedFrame,
                # if we have received audio from TTS
                if self._tts_audio_received and (
                    not self._transport.audio_output_policy_enabled
                    or (
                        self._transport.capture_audio_output_context(frame)
                        == self._bot_speech_context
                        and frame.context_id == self._bot_speech_context_id
                    )
                ):
                    logger.debug("Bot stopped speaking based on TTSStoppedFrame")
                    await self._bot_stopped_speaking()
                return outcome
            elif isinstance(frame, FilterControlFrame) and self._filter:
                await self._filter.process_frame(frame)
            elif isinstance(frame, VolumeFrame):
                self._volume = frame.volume
            elif isinstance(frame, VolumeGainFrame):
                self._volume_gain = frame.gain
            else:
                await self._transport.write_transport_frame(frame)

        async def _filter_audio(self, frame: Frame):
            """Run the output filter over a frame taken from the audio queue.

            Args:
                frame: The queued frame. Only audio frames are filtered.
            """
            if self._filter and isinstance(frame, OutputAudioRawFrame):
                frame.audio = await self._filter.filter(frame.audio)

        def _apply_volume(self, frame: Frame):
            """Scale an audio frame taken from the audio queue by the volume and its gain."""
            volume = self._volume * self._volume_gain
            if volume != 1.0 and isinstance(frame, OutputAudioRawFrame):
                frame.audio = scale_audio(frame.audio, volume)

        def _next_frame(self) -> AsyncGenerator[Frame, None]:
            """Generate the next frame for audio processing.

            Returns:
                An async generator yielding frames for processing.
            """

            async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                """Consume actual queue items with post-await source checks and balanced acknowledgments."""
                while True:
                    try:
                        frame = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=vad_stop_secs
                        )
                        try:
                            if not self._transport._audio_output_frame_is_current(frame):
                                continue
                            await self._filter_audio(frame)
                            if not self._transport._audio_output_frame_is_current(frame):
                                continue
                            self._apply_volume(frame)
                            yield frame
                        finally:
                            self._audio_queue.task_done()
                    except TimeoutError:
                        # Fallback: notify the bot stopped speaking upstream if necessary based on timeout.
                        await self._bot_stopped_speaking()

            async def with_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                """Keep speech-owned mixtures distinct from trusted mixer-only native output."""
                # Caller below only invokes this when `self._mixer` is set.
                mixer = self._mixer
                assert mixer is not None
                last_frame_time = 0
                silence = b"\x00" * self._audio_chunk_size
                while True:
                    try:
                        frame = self._audio_queue.get_nowait()
                        try:
                            if not self._transport._audio_output_frame_is_current(frame):
                                continue
                            await self._filter_audio(frame)
                            if not self._transport._audio_output_frame_is_current(frame):
                                continue
                            self._apply_volume(frame)
                            if isinstance(frame, OutputAudioRawFrame):
                                frame.audio = await mixer.mix(frame.audio)
                                if not self._transport._audio_output_frame_is_current(frame):
                                    continue
                                last_frame_time = time.time()
                            yield frame
                        finally:
                            self._audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        # Fallback: notify the bot stopped speaking upstream if necessary based on timeout.
                        diff_time = time.time() - last_frame_time
                        if diff_time > vad_stop_secs:
                            await self._bot_stopped_speaking()
                        # Generate an audio frame with only the mixer's part.
                        context = self._transport.native_audio_output_context(
                            "mixer", self._destination
                        )
                        if self._transport.audio_output_context_is_current(context):
                            audio = await mixer.mix(silence)
                            if self._transport.audio_output_context_is_current(context):
                                frame = OutputAudioRawFrame(
                                    audio=audio,
                                    sample_rate=self._sample_rate,
                                    num_channels=self._params.audio_out_channels,
                                )
                                if self._transport.audio_output_policy_enabled:
                                    frame.transport_destination = self._destination
                                self._transport.copy_audio_output_context(context, frame)
                                yield frame
                        # Allow other asyncio tasks to execute by adding a small sleep
                        # Without this sleep, in task cancellation scenarios, this loop would
                        # continuously return without any delay, leading to 100% CPU utilization
                        # and preventing cancel/stop signals from being processed properly
                        await asyncio.sleep(0)

            if self._mixer:
                return with_mixer(self._params.bot_vad_stop_secs)
            else:
                return without_mixer(self._params.bot_vad_stop_secs)

        async def _send_silence(self, secs: int):
            """Send terminal native silence under its own captured call-owned context."""
            if secs <= 0:
                return

            context = self._transport.native_audio_output_context("end_silence", self._destination)
            if not self._transport.audio_output_context_is_current(context):
                return

            sample_width = 2
            silence = b"\x00" * self.sample_rate * sample_width * secs
            silence_frame = OutputAudioRawFrame(
                audio=silence, sample_rate=self.sample_rate, num_channels=1
            )
            if self._transport.audio_output_policy_enabled:
                silence_frame.transport_destination = self._destination
            self._transport.copy_audio_output_context(context, silence_frame)
            await self._internal_write_audio_frame(silence_frame)

        async def _audio_task_handler(self):
            """Main audio processing task handler."""
            frames = self._next_frame()
            try:
                consecutive_failures = 0
                max_consecutive_failures = self._params.audio_out_max_consecutive_failures
                sleep_between_consecutive_failures = self._params.audio_out_sleep_between_failures

                async for frame in frames:
                    # No need to push EndFrame, it's pushed from process_frame().
                    if isinstance(frame, EndFrame):
                        # Send some final silence so words don't cut out.
                        await self._send_silence(self._params.audio_out_end_silence_secs)
                        break

                    # Skip frames with no audio data (e.g. filter is buffering).
                    if isinstance(frame, OutputAudioRawFrame) and not frame.audio:
                        continue

                    if not self._transport._audio_output_frame_is_current(frame):
                        continue

                    # Handle frame.
                    handled_result = await self._handle_frame(frame)
                    if not self._transport._audio_output_frame_is_current(frame):
                        continue

                    # If we are not able to write to the transport we shouldn't
                    # push downstream.
                    tail_write = (
                        self._transport.audio_output_policy_enabled
                        and isinstance(frame, TTSStoppedFrame)
                        and handled_result is not None
                    )
                    write_attempt = isinstance(frame, OutputAudioRawFrame) or tail_write
                    push_downstream = handled_result if tail_write else True

                    # Try to send audio to the transport.
                    try:
                        if isinstance(frame, OutputAudioRawFrame):
                            push_downstream = await self._internal_write_audio_frame(frame)
                    except Exception as e:
                        logger.error(f"{self} Error writing {frame} to transport: {e}")
                        push_downstream = False

                    if push_downstream is AUDIO_OUTPUT_DISCARDED:
                        # Remaining stale output was intentionally stopped;
                        # earlier packets may already be accepted. Do not count
                        # this as a failed/successful write or suppress cleanup.
                        if isinstance(frame, OutputAudioRawFrame):
                            continue
                        push_downstream = True
                        write_attempt = False

                    # Handle write failures
                    if not push_downstream and write_attempt:
                        # A bounded-write timeout already reports a permanent error
                        # and marks the transport unusable. From that point the
                        # worker's ProcessorUnusablePolicy owns disposition; drain
                        # the remaining queue without sleeping or overriding that
                        # policy with the older fork-level cancellation watchdog.
                        if not self._transport.is_usable:
                            continue

                        consecutive_failures += 1
                        # A hangup produces a burst of these; the first one and the
                        # cancellation below are enough above DEBUG.
                        log = logger.warning if consecutive_failures == 1 else logger.debug
                        log(
                            f"Failed to write audio frame (consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                        )

                        # Break out if we've failed too many times consecutively
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(
                                f"Cancelling task after {consecutive_failures} consecutive failures"
                            )

                            # Send bot stopped speaking frame
                            await self._bot_stopped_speaking()

                            # This CancelWorkerFrame would be intercepted upstream
                            # so that we can call end_call_with_reason and dispose
                            # off the call properly
                            await self._transport.push_frame(
                                CancelWorkerFrame(reason="audio_output_write_failed"),
                                FrameDirection.UPSTREAM,
                            )
                            break

                        # Sleep before retrying
                        await asyncio.sleep(sleep_between_consecutive_failures)
                    elif write_attempt:
                        # Reset counter only after a successful audio write.
                        consecutive_failures = 0

                    # If we were able to send to the transport, push the frame
                    # downstream in case anyone else needs it.
                    if push_downstream and (
                        not isinstance(frame, OutputAudioRawFrame)
                        or self._transport._should_forward_audio_frame(frame)
                    ):
                        if self._transport._audio_output_frame_is_current(frame):
                            await self._transport.push_frame(frame)
            finally:
                # Closing the real queue iterator acknowledges an item even
                # when End, a write watchdog or cancellation exits at yield.
                await frames.aclose()

        async def _internal_write_audio_frame(
            self, frame: OutputAudioRawFrame
        ) -> bool | AudioOutputDiscarded:
            """Write a frame to the transport, giving up if the write never returns.

            A client that stops reading is the common way to get there: the
            connection stays up and nothing fails, so the transport has neither
            an error to report nor state to check. That leaves how long the
            write takes as the only signal available.

            Returns:
                Whether the transport took the frame, or the deliberate-discard
                singleton if its retained source became stale before writing.
            """
            if not self._transport._audio_output_frame_is_current(frame):
                return AUDIO_OUTPUT_DISCARDED
            # Reporting the timeout below costs the transport its usability, so
            # a peer is only written off once however much audio is still queued.
            if not self._transport.is_usable:
                return False

            timeout = self._params.audio_out_write_timeout_secs
            try:
                result = await asyncio.wait_for(
                    self._transport.write_audio_frame(frame), timeout=timeout
                )
                if result is False and self._transport.audio_output_policy_enabled:
                    await self._transport._call_event_handler("on_audio_output_failed", frame)
                return result
            except TimeoutError as e:
                if self._transport.audio_output_policy_enabled:
                    await self._transport._call_event_handler("on_audio_output_failed", frame)
                await self._transport.push_error(
                    f"{self._transport} timed out after {timeout}s writing audio to the "
                    "transport; the peer has stopped reading",
                    exception=e,
                    force_treat_as_permanent=True,
                )
                return False
            except Exception:
                if self._transport.audio_output_policy_enabled:
                    await self._transport._call_event_handler("on_audio_output_failed", frame)
                raise

        async def _internal_write_transport_frame(
            self, frame: TTSStoppedFrame
        ) -> bool | AudioOutputDiscarded | None:
            """Bound a guarded serializer tail flush and notify real failures before completion.

            The native hook retains the exact stopped source/context and may
            deliberately discard a stale/mismatched stop. None denotes no tail
            hook. Neither discard nor None becomes a successful audio write.
            """
            if not self._transport.is_usable:
                return False
            timeout = self._params.audio_out_write_timeout_secs
            try:
                result = await asyncio.wait_for(
                    self._transport.write_transport_frame(frame), timeout=timeout
                )
                if result is False:
                    await self._transport._call_event_handler("on_audio_output_failed", frame)
                return result
            except TimeoutError as e:
                await self._transport._call_event_handler("on_audio_output_failed", frame)
                await self._transport.push_error(
                    f"{self._transport} timed out after {timeout}s finishing output audio",
                    exception=e,
                    force_treat_as_permanent=True,
                )
                return False
            except Exception as e:
                await self._transport._call_event_handler("on_audio_output_failed", frame)
                await self._transport.push_error("Output audio completion failed", exception=e)
                return False

        #
        # Video handling
        #

        def _create_video_task(self):
            """Create the video processing task if video output is enabled."""
            if not self._video_task and self._params.video_out_enabled:
                self._video_queue = asyncio.Queue()
                self._video_task = self._transport.create_task(self._video_task_handler())

        async def _cancel_video_task(self):
            """Cancel and cleanup the video processing task."""
            # Stop video output task.
            if self._video_task:
                await self._transport.cancel_task(self._video_task)
                self._video_task = None

        async def _set_video_image(self, image: OutputImageRawFrame):
            """Set a single video image for cycling output.

            Args:
                image: The image frame to cycle for video output.
            """
            self._video_images = itertools.cycle([image])

        async def _set_video_images(self, images: list[OutputImageRawFrame]):
            """Set multiple video images for cycling output.

            Args:
                images: The list of image frames to cycle for video output.
            """
            self._video_images = itertools.cycle(images)

        async def _video_task_handler(self):
            """Main video processing task handler."""
            self._video_start_time = None
            self._video_frame_index = 0
            self._video_frame_duration = 1 / self._params.video_out_framerate
            self._video_frame_reset = self._video_frame_duration * 5
            while True:
                if self._params.video_out_is_live:
                    await self._video_is_live_handler()
                elif self._video_images:
                    image = next(self._video_images)
                    await self._draw_image(image)
                    await asyncio.sleep(self._video_frame_duration)
                else:
                    await asyncio.sleep(self._video_frame_duration)

        async def _video_is_live_handler(self):
            """Handle live video streaming with frame timing."""
            image = await self._video_queue.get()

            # We get the start time as soon as we get the first image.
            if not self._video_start_time:
                self._video_start_time = time.time()
                self._video_frame_index = 0

            # Calculate how much time we need to wait before rendering next image.
            real_elapsed_time = time.time() - self._video_start_time
            real_render_time = self._video_frame_index * self._video_frame_duration
            delay_time = self._video_frame_duration + real_render_time - real_elapsed_time

            if abs(delay_time) > self._video_frame_reset:
                self._video_start_time = time.time()
                self._video_frame_index = 0
            elif delay_time > 0:
                await asyncio.sleep(delay_time)
                self._video_frame_index += 1

            # Render image
            await self._draw_image(image)

            self._video_queue.task_done()

        async def _draw_image(self, frame: OutputImageRawFrame):
            """Draw/render an image frame with resizing if needed.

            Args:
                frame: The image frame to draw.
            """

            def resize_frame(frame: OutputImageRawFrame) -> OutputImageRawFrame:
                # Without a format we can't decode the bytes, so leave the
                # frame as-is and let the transport pass it through unchanged.
                if frame.format is None:
                    return frame

                desired_size = (self._params.video_out_width, self._params.video_out_height)

                # TODO: we should refactor in the future to support dynamic resolutions
                # which is kind of what happens in P2P connections.
                # We need to add support for that inside the DailyTransport
                if frame.size != desired_size:
                    image = Image.frombytes(frame.format, frame.size, frame.image)
                    resized_image = image.resize(desired_size)
                    # logger.warning(f"{frame} does not have the expected size {desired_size}, resizing")
                    frame = OutputImageRawFrame(
                        resized_image.tobytes(), resized_image.size, resized_image.mode
                    )

                return frame

            frame = await self._transport.get_event_loop().run_in_executor(
                self._executor, resize_frame, frame
            )
            await self._transport.write_video_frame(frame)

        #
        # Clock handling
        #

        def _create_clock_task(self):
            """Create the clock/timing processing task."""
            if not self._clock_task:
                self._clock_task = self._transport.create_task(self._clock_task_handler())

        async def _cancel_clock_task(self):
            """Cancel and cleanup the clock processing task."""
            if self._clock_task:
                await self._transport.cancel_task(self._clock_task)
                self._clock_task = None
                self._clock_frame = None

        async def _clock_task_handler(self):
            """Deliver current timed frames and acknowledge each dequeue, including cancellation."""
            while True:
                timestamp, _, frame = await self._clock_queue.get()
                try:
                    if isinstance(frame, EndFrame):
                        return
                    self._clock_frame = frame
                    if not self._transport._audio_output_frame_is_current(frame):
                        continue

                    current_time = self._transport.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Authority may change while the native clock is waiting.
                    if self._transport._audio_output_frame_is_current(frame):
                        await self._transport.push_frame(frame)
                finally:
                    self._clock_frame = None
                    # The queue survives a clock-task restart; a cancelled wait
                    # still owns exactly one unfinished item to acknowledge.
                    self._clock_queue.task_done()

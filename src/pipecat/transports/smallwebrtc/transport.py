#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Small WebRTC transport implementation for Pipecat.

This module provides a WebRTC transport implementation using aiortc for
real-time audio and video communication. It supports bidirectional media
streaming, application messaging, and client connection management.
Optional output ownership remains private to the native audio queue and is
checked again after track pacing; it never enters encoded media or messages.
"""

import asyncio
import fractions
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SpriteFrame,
    StartFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import (
    AUDIO_OUTPUT_DISCARDED,
    AudioOutputDiscarded,
    AudioOutputPolicy,
    BaseOutputTransport,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import (
    AUDIO_TRANSCEIVER_INDEX,
    SCREEN_VIDEO_TRANSCEIVER_INDEX,
    VIDEO_TRANSCEIVER_INDEX,
    SmallWebRTCConnection,
    SmallWebRTCTrack,
)
from pipecat.transports.video_in_sampler import _capture_framerate, _VideoInSamplers
from pipecat.utils.deprecation import warn_deprecated
from pipecat.utils.shared import acquires, releases
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

try:
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use the SmallWebRTC, you need to `uv add "pipecat-ai[webrtc]"`.')
    raise ImportError(f"Missing module: {e}") from e

CAM_VIDEO_SOURCE = "camera"
SCREEN_VIDEO_SOURCE = "screenVideo"
MIC_AUDIO_SOURCE = "microphone"

# SmallWebRTC has one peer, so all its video samplers are kept under one id.
_PEER_ID = "peer"

# The source each of the connection's receivers carries.
_SOURCES_BY_RECEIVER = {
    AUDIO_TRANSCEIVER_INDEX: MIC_AUDIO_SOURCE,
    VIDEO_TRANSCEIVER_INDEX: CAM_VIDEO_SOURCE,
    SCREEN_VIDEO_TRANSCEIVER_INDEX: SCREEN_VIDEO_SOURCE,
}


class SmallWebRTCCallbacks(BaseModel):
    """Callback handlers for SmallWebRTC events.

    Parameters:
        on_app_message: Called when an application message is received.
        on_client_connected: Called when a client establishes connection.
        on_client_disconnected: Called when a client disconnects.
        on_track_status: Called with the source (``"microphone"``, ``"camera"`` or
            ``"screenVideo"``) and whether the peer is sending it, when the peer
            turns a source on or off.
    """

    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_client_connected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_track_status: Callable[[str, bool], Awaitable[None]]


@dataclass(frozen=True)
class _QueuedAudioChunk:
    """Keep one native ten-millisecond chunk with its original source and completion.

    Each chunk retains the input's existing future so cancellation can discard
    its remaining bytes. Only the final chunk settles that future normally.
    """

    audio: bytes
    future: asyncio.Future[bool | AudioOutputDiscarded]
    final: bool
    context: object | None = field(default=None, repr=False)


class RawAudioTrack(AudioStreamTrack):
    """Custom audio stream track for WebRTC output.

    Handles audio frame generation and timing for WebRTC transmission,
    supporting queued audio data with proper synchronization.
    """

    def __init__(
        self,
        sample_rate: int,
        auto_silence: bool = True,
        *,
        context_is_current: Callable[[object | None], bool] | None = None,
        native_silence_context: Callable[[], object | None] | None = None,
    ):
        """Initialize the raw audio track.

        Args:
            sample_rate: The audio sample rate in Hz.
            auto_silence: If True, emit silence when the queue is empty. If False,
                wait until audio data is available.
            context_is_current: Optional synchronous guard for the retained source.
            native_silence_context: Paired callback capturing the actual silence
                producer before pacing. Silence cannot inherit a speech source.

        Raises:
            ValueError: If guarded callbacks are incomplete or noncallable.
        """
        super().__init__()
        if (context_is_current is None) != (native_silence_context is None) or (
            context_is_current is not None
            and any(
                not callable(callback)
                or inspect.iscoroutinefunction(callback)
                or inspect.iscoroutinefunction(getattr(callback, "__call__", None))
                for callback in (context_is_current, native_silence_context)
            )
        ):
            raise ValueError("Guarded audio tracks require paired synchronous callbacks")
        self._sample_rate = sample_rate
        self._auto_silence = auto_silence
        self._context_is_current = context_is_current
        self._native_silence_context = native_silence_context
        self._samples_per_10ms = sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit (2 bytes per sample)
        self._timestamp = 0
        self._start = time.time()
        self._chunk_queue: deque[_QueuedAudioChunk] = deque()

    def add_audio_bytes(
        self, audio_bytes: bytes, *, context: object | None = None
    ) -> asyncio.Future[bool | AudioOutputDiscarded]:
        """Add audio bytes to the buffer for transmission.

        Args:
            audio_bytes: Raw audio data to queue for transmission.
            context: Original opaque source captured by the writer, never a
                replacement inferred when these bytes are consumed.

        Returns:
            The input's completion future. Guarded discarded data resolves to
            AUDIO_OUTPUT_DISCARDED; ordinary completion retains True.

        Raises:
            ValueError: If audio bytes are not a multiple of 10ms size.
        """
        if len(audio_bytes) % self._bytes_per_10ms != 0:
            raise ValueError("Audio bytes must be a multiple of 10ms size.")
        future = asyncio.get_running_loop().create_future()
        if not audio_bytes:
            future.set_result(True)
            return future
        if self.readyState == "ended":
            future.set_result(
                AUDIO_OUTPUT_DISCARDED if self._context_is_current is not None else True
            )
            return future

        # Break input into 10ms chunks
        for i in range(0, len(audio_bytes), self._bytes_per_10ms):
            chunk = audio_bytes[i : i + self._bytes_per_10ms]
            final = i + self._bytes_per_10ms >= len(audio_bytes)
            self._chunk_queue.append(_QueuedAudioChunk(chunk, future, final, context))

        return future

    def mark_pending_futures_done(self):
        """Mark all pending futures as done in _chunk_queue.

        This should be called when the connection closes to unblock
        any callers waiting on add_audio_bytes futures.
        """
        logger.debug(
            f"RawAudioTrack.mark_pending_futures_done: marking futures as done, queue size: {len(self._chunk_queue)}"
        )
        result = AUDIO_OUTPUT_DISCARDED if self._context_is_current is not None else True
        for chunk in self._chunk_queue:
            if chunk.final and not chunk.future.done():
                chunk.future.set_result(result)
        self._chunk_queue.clear()

    def stop(self):
        """Release queued completion futures before ending the native media track."""
        self.mark_pending_futures_done()
        super().stop()

    def _source_is_current(self, context: object | None) -> bool:
        """Apply a synchronous source guard; callback errors cannot permit audio."""
        if self._context_is_current is None:
            return True
        try:
            return self._context_is_current(context) is True
        except Exception:
            return False

    def _settle_chunk(self, chunk: _QueuedAudioChunk, result: bool | AudioOutputDiscarded):
        """Settle the original terminal future once, including cancelled writers."""
        if chunk.final and not chunk.future.done():
            chunk.future.set_result(result)

    async def recv(self):
        """Return the next audio frame for WebRTC transmission.

        Returns:
            An AudioFrame containing the next audio data, or silence if the queue is empty
            and ``auto_silence`` is True.
        """
        if self.readyState == "ended":
            raise MediaStreamError
        silence_context = None
        if self._auto_silence and self._native_silence_context is not None:
            try:
                silence_context = self._native_silence_context()
            except Exception:
                self.stop()
                raise MediaStreamError from None

        # Capture the native silence source before this clock wait, independently
        # from any speech that may later be discarded from the same queue.
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        while True:
            queued = None
            if self.readyState == "ended":
                raise MediaStreamError
            if self._chunk_queue:
                candidate = self._chunk_queue.popleft()
                if self._context_is_current is not None and (
                    candidate.future.cancelled() or not self._source_is_current(candidate.context)
                ):
                    self._settle_chunk(candidate, AUDIO_OUTPUT_DISCARDED)
                    continue
                queued = candidate
                chunk = candidate.audio
            elif self._auto_silence:
                if not self._source_is_current(silence_context):
                    self.stop()
                    raise MediaStreamError
                chunk = bytes(self._bytes_per_10ms)
            else:
                await asyncio.sleep(0.005)
                continue

            samples = np.frombuffer(chunk, dtype=np.int16)
            frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
            frame.sample_rate = self._sample_rate
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, self._sample_rate)
            # No await occurs between the final source check and returning to aiortc.
            if queued is not None:
                if not self._source_is_current(queued.context):
                    self._settle_chunk(queued, AUDIO_OUTPUT_DISCARDED)
                    continue
                self._settle_chunk(queued, True)
            elif not self._source_is_current(silence_context):
                self.stop()
                raise MediaStreamError
            self._timestamp += self._samples_per_10ms
            return frame


class RawVideoTrack(VideoStreamTrack):
    """Custom video stream track for WebRTC output.

    Handles video frame queuing and conversion for WebRTC transmission.
    """

    def __init__(self, width, height):
        """Initialize the raw video track.

        Args:
            width: Video frame width in pixels.
            height: Video frame height in pixels.
        """
        super().__init__()
        self._width = width
        self._height = height
        self._video_buffer = asyncio.Queue()

    def add_video_frame(self, frame):
        """Add a video frame to the transmission buffer.

        Args:
            frame: The video frame to queue for transmission.
        """
        self._video_buffer.put_nowait(frame)

    async def recv(self):
        """Return the next video frame for WebRTC transmission.

        Returns:
            A VideoFrame ready for WebRTC transmission.
        """
        raw_frame = await self._video_buffer.get()

        # Convert bytes to NumPy array
        frame_data = np.frombuffer(raw_frame.image, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")

        # Assign timestamp
        frame.pts, frame.time_base = await self.next_timestamp()

        return frame


class SmallWebRTCClient:
    """WebRTC client implementation for handling connections and media streams.

    Manages WebRTC peer connections, audio/video streaming, and application
    messaging through the SmallWebRTCConnection interface.
    """

    def __init__(self, webrtc_connection: SmallWebRTCConnection, callbacks: SmallWebRTCCallbacks):
        """Initialize the WebRTC client.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            callbacks: Event callbacks for connection and message handling.
        """
        self._webrtc_connection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._video_output_track = None
        self._audio_output_capture: Callable[[Frame], object | None] | None = None
        self._audio_output_context_is_current: Callable[[object | None], bool] | None = None
        self._audio_output_native_context: Callable[[str, str | None], object | None] | None = None
        self._audio_input_track: SmallWebRTCTrack | None = None
        self._video_input_track: SmallWebRTCTrack | None = None
        self._screen_video_track: SmallWebRTCTrack | None = None

        self._params = None
        self._audio_in_channels = 0
        self._in_sample_rate = 0
        self._out_sample_rate = 0

        # Audio resampler - will be configured during setup with target sample rate/layout
        self._audio_in_resampler = None
        self._audio_in_layout = None

        @self._webrtc_connection.event_handler("connected")
        async def on_connected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtc_connection.event_handler("disconnected")
        async def on_disconnected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection lost.")
            await self._handle_peer_disconnected()

        @self._webrtc_connection.event_handler("closed")
        async def on_closed(connection: SmallWebRTCConnection):
            logger.debug("Client connection closed.")
            await self._handle_client_closed()

        @self._webrtc_connection.event_handler("app-message")
        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
            await self._handle_app_message(message, connection.pc_id)

        @self._webrtc_connection.event_handler("track-status")
        async def on_track_status(
            connection: SmallWebRTCConnection, receiver_index: int, enabled: bool
        ):
            source = _SOURCES_BY_RECEIVER.get(receiver_index)
            if source:
                await self._callbacks.on_track_status(source, enabled)

    def _convert_frame(self, frame_array: np.ndarray, format_name: str) -> np.ndarray:
        """Convert a video frame to RGB format based on the input format.

        Args:
            frame_array: The input frame as a NumPy array.
            format_name: The format of the input frame.

        Returns:
            The converted RGB frame as a NumPy array.

        Raises:
            ImportError: If OpenCV is not installed.
            ValueError: If the format is unsupported.
        """
        if format_name.startswith("rgb"):  # Already in RGB, no conversion needed
            return frame_array

        try:
            import cv2
        except ModuleNotFoundError as e:
            raise ImportError(
                "Receiving non-RGB video frames requires OpenCV. Install it with "
                '`uv add "pipecat-ai[webrtc-video]"`.'
            ) from e

        format_conversions = {
            "yuv420p": cv2.COLOR_YUV2RGB_I420,
            "yuvj420p": cv2.COLOR_YUV2RGB_I420,  # OpenCV treats both the same
            "nv12": cv2.COLOR_YUV2RGB_NV12,
            "gray": cv2.COLOR_GRAY2RGB,
        }

        conversion_code = format_conversions.get(format_name)

        if conversion_code is None:
            raise ValueError(f"Unsupported format: {format_name}")

        return cv2.cvtColor(frame_array, conversion_code)

    def video_source_enabled(self, video_source: str) -> bool:
        """Whether the peer is sending a video source.

        Args:
            video_source: ``"camera"`` or ``"screenVideo"``.

        Returns:
            Whether the source's track is present and the peer hasn't turned it off.
        """
        track = (
            self._video_input_track
            if video_source == CAM_VIDEO_SOURCE
            else self._screen_video_track
        )
        return track is not None and track.is_enabled()

    async def read_video_frame(self, video_source: str):
        """Read video frames from the WebRTC connection.

        Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and creates an InputImageRawFrame.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").

        Yields:
            UserImageRawFrame objects containing video data from the peer.
        """
        while True:
            video_track = (
                self._video_input_track
                if video_source == CAM_VIDEO_SOURCE
                else self._screen_video_track
            )
            if video_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(video_track.recv(), timeout=2.0)
            except TimeoutError:
                if (
                    self._webrtc_connection.is_connected()
                    and video_track
                    and video_track.is_enabled()
                ):
                    logger.warning("Timeout: No video frame received within the specified time.")
                    # self._webrtc_connection.ask_to_renegotiate()
                frame = None
            except MediaStreamError:
                # Track is dead — every subsequent `recv()` would raise the same
                # error and busy-loop the generator at ~100Hz. Clear the track
                # reference so the loop parks on the `is None` gate above; a
                # renegotiation that repopulates the track will resume frames.
                logger.warning("Media stream error while reading the video; clearing track.")
                if video_source == CAM_VIDEO_SOURCE:
                    self._video_input_track = None
                else:
                    self._screen_video_track = None
                frame = None

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            format_name = frame.format.name
            # Convert frame to NumPy array in its native format
            frame_array = frame.to_ndarray(format=format_name)
            frame_rgb = self._convert_frame(frame_array, format_name)
            del frame_array  # free intermediate array immediately
            image_bytes = frame_rgb.tobytes()
            del frame_rgb  # free RGB array immediately

            image_frame = UserImageRawFrame(
                user_id=self._webrtc_connection.pc_id,
                image=image_bytes,
                size=(frame.width, frame.height),
                format="RGB",
            )
            image_frame.transport_source = video_source
            image_frame.pts = frame.pts

            del frame  # free original VideoFrame
            del image_bytes  # reference kept in image_frame

            yield image_frame

    async def read_audio_frame(self):
        """Read audio frames from the WebRTC connection.

        Reads 20ms of audio from the given MediaStreamTrack and creates an InputAudioRawFrame.

        Yields:
            InputAudioRawFrame objects containing audio data from the peer.
        """
        # setup() builds the resampler before the reader task is started.
        assert self._audio_in_resampler is not None

        while True:
            if self._audio_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._audio_input_track.recv(), timeout=2.0)
            except TimeoutError:
                if (
                    self._webrtc_connection.is_connected()
                    and self._audio_input_track
                    and self._audio_input_track.is_enabled()
                ):
                    logger.warning("Timeout: No audio frame received within the specified time.")
                frame = None
            except MediaStreamError:
                # Track is dead — every subsequent `recv()` would raise the same
                # error and busy-loop the generator at ~100Hz. Clear the track
                # reference so the loop parks on the `is None` gate above; a
                # renegotiation that repopulates the track will resume frames.
                logger.warning("Media stream error while reading the audio; clearing track.")
                self._audio_input_track = None
                frame = None

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            # Resample if needed, otherwise use the frame as-is. The resampler
            # also converts to the configured channel layout, so frames whose
            # layout doesn't already match must go through it even when the
            # rate already matches (e.g. aiortc decodes to stereo regardless
            # of the source track) — otherwise interleaved bytes get labeled
            # with the wrong channel count.
            frames_to_process = (
                self._audio_in_resampler.resample(frame)
                if frame.sample_rate != self._in_sample_rate
                or frame.layout.name != self._audio_in_layout
                else [frame]
            )

            for processed_frame in frames_to_process:
                # Convert to 16-bit PCM bytes
                pcm_array = processed_frame.to_ndarray().astype(np.int16)
                pcm_bytes = pcm_array.tobytes()
                del pcm_array  # free NumPy array immediately

                audio_frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=self._in_sample_rate,
                    num_channels=self._audio_in_channels,
                )
                audio_frame.pts = frame.pts
                del pcm_bytes  # reference kept in audio_frame

                yield audio_frame

            del frame  # free original AudioFrame

    def set_audio_output_callbacks(
        self,
        capture: Callable[[Frame], object | None],
        is_current: Callable[[object | None], bool],
        native_context: Callable[[str, str | None], object | None],
    ) -> None:
        """Bind one output owner's callbacks before the shared client is set up.

        The native track receives these fixed callbacks, never a mutable current
        agent lookup. Replacing a callback after audio resources exist is denied.
        """
        if self._params is not None or self._audio_output_track is not None:
            raise RuntimeError("Audio output callbacks must be installed before client setup")
        if any(
            not callable(callback)
            or inspect.iscoroutinefunction(callback)
            or inspect.iscoroutinefunction(getattr(callback, "__call__", None))
            for callback in (capture, is_current, native_context)
        ):
            raise TypeError("Audio output callbacks must be synchronous callables")
        if self._audio_output_capture is not None and (
            self._audio_output_capture != capture
            or self._audio_output_context_is_current != is_current
            or self._audio_output_native_context != native_context
        ):
            raise RuntimeError("Audio output callbacks cannot be replaced")
        self._audio_output_capture = capture
        self._audio_output_context_is_current = is_current
        self._audio_output_native_context = native_context

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool | AudioOutputDiscarded:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The audio frame to transmit.

        Returns:
            True after ordinary/current track completion, AUDIO_OUTPUT_DISCARDED
            for an intentional guarded discard, or False for an unavailable sink.
        """
        context = None
        if self._audio_output_capture is not None:
            context = self._audio_output_capture(frame)
            if not self._audio_output_context_is_current(context):
                return AUDIO_OUTPUT_DISCARDED
        if self._can_send() and self._audio_output_track:
            if self._audio_output_capture is None:
                await self._audio_output_track.add_audio_bytes(frame.audio)
                return True
            if not isinstance(self._audio_output_track, RawAudioTrack) or (
                self._audio_output_track._context_is_current is None
            ):
                raise RuntimeError("WebRTC output track does not support guarded audio")
            result = await self._audio_output_track.add_audio_bytes(frame.audio, context=context)
            if result is AUDIO_OUTPUT_DISCARDED:
                return AUDIO_OUTPUT_DISCARDED
            return True
        return False

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        if self._can_send() and self._video_output_track:
            self._video_output_track.add_video_frame(frame)
            return True
        return False

    async def setup(self, _params: TransportParams, setup: FrameProcessorSetup):
        """Set up the client with transport parameters.

        Args:
            _params: Transport configuration parameters.
            setup: Configuration object containing setup parameters.
        """
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or setup.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or setup.audio_out_sample_rate
        self._params = _params
        self._audio_in_layout = "stereo" if self._audio_in_channels == 2 else "mono"
        self._audio_in_resampler = AudioResampler(
            "s16", self._audio_in_layout, self._in_sample_rate
        )

    @acquires("connection")
    async def connect(self):
        """Establish the WebRTC connection."""
        if self._webrtc_connection.is_connected():
            # already initialized
            return

        logger.info("Connecting to Small WebRTC")
        await self._webrtc_connection.connect()

    @releases("connection")
    async def disconnect(self):
        """Disconnect from the WebRTC peer."""
        if self.is_connected and not self.is_closing:
            logger.info("Disconnecting to Small WebRTC")
            self._closing = True
            await self._webrtc_connection.disconnect()
            await self._handle_peer_disconnected()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send an application message through the WebRTC connection.

        Messages sent before the data channel is open (e.g. while the peer
        connection is still being established) are buffered by the connection
        and flushed, in order, once the channel opens.

        Args:
            frame: The message frame to send.
        """
        if self.is_closing:
            message_type = (
                frame.message.get("type", "unknown")
                if isinstance(frame.message, dict)
                else type(frame.message).__name__
            )
            logger.debug(f"Discarding app message '{message_type}': peer connection is closing.")
            return
        self._webrtc_connection.send_app_message(frame.message)

    async def _handle_client_connected(self):
        """Handle client connection establishment."""
        # There is nothing to do here yet, the pipeline is still not ready
        if not self._params:
            return

        self._audio_input_track = self._webrtc_connection.audio_input_track()
        self._video_input_track = self._webrtc_connection.video_input_track()
        self._screen_video_track = self._webrtc_connection.screen_video_input_track()
        if self._params.audio_out_enabled:
            if self._audio_output_track is not None:
                self._audio_output_track.stop()
            track_callbacks = {}
            if self._audio_output_capture is not None:
                guard = self._audio_output_context_is_current
                native_context = self._audio_output_native_context
                if guard is None or native_context is None:
                    raise RuntimeError("WebRTC output callbacks are incomplete")
                track_callbacks = {
                    "context_is_current": guard,
                    # Bind the original output callback, independent of later client references.
                    "native_silence_context": lambda: native_context("track_silence", None),
                }
            self._audio_output_track = RawAudioTrack(
                sample_rate=self._out_sample_rate,
                auto_silence=self._params.audio_out_auto_silence,
                **track_callbacks,
            )
            self._webrtc_connection.replace_audio_track(self._audio_output_track)

        if self._params.video_out_enabled:
            self._video_output_track = RawVideoTrack(
                width=self._params.video_out_width, height=self._params.video_out_height
            )
            self._webrtc_connection.replace_video_track(self._video_output_track)

        await self._callbacks.on_client_connected(self._webrtc_connection)

    async def _handle_peer_disconnected(self):
        """Settle the old track's queued writers before discarding media references."""
        if self._audio_output_track is not None:
            self._audio_output_track.stop()
        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

    async def _handle_client_closed(self):
        """Handle client connection closure."""
        # Set pending futures before clearing the track to unblock write_audio_frame
        if self._audio_output_track:
            self._audio_output_track.stop()

        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._closing:
            await self._callbacks.on_client_disconnected(self._webrtc_connection)

    async def _handle_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        await self._callbacks.on_app_message(message, sender)

    def _can_send(self):
        """Check if the connection is ready for sending data."""
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        """Check if the WebRTC connection is established.

        Returns:
            True if connected to the peer.
        """
        return self._webrtc_connection.is_connected()

    @property
    def is_closing(self) -> bool:
        """Check if the connection is in the process of closing.

        Returns:
            True if the connection is closing.
        """
        return self._closing


class SmallWebRTCInputTransport(BaseInputTransport):
    """Input transport implementation for SmallWebRTC.

    Handles incoming audio and video streams from WebRTC peers,
    including user image requests and application message handling.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC input transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None
        self._receive_screen_video_task = None
        self._video_samplers = _VideoInSamplers()

    def _supports_video_in_source(self, video_source: str) -> bool:
        """Whether this transport captures a video source listed in ``video_in_sources``.

        Args:
            video_source: The video source.

        Returns:
            Whether the source is the camera or the screen share.
        """
        return video_source in (CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and establish the WebRTC connection.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        await self._client.setup(self._params, setup)
        await self._client.connect()

    async def cleanup(self):
        """Release resources during teardown."""
        await super().cleanup()
        await self._teardown()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames including user image requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    async def _stop_tasks(self):
        """Stop all background tasks."""
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None
        if self._receive_screen_video_task:
            await self.cancel_task(self._receive_screen_video_task)
            self._receive_screen_video_task = None

    async def _teardown(self):
        """Cancel receive tasks and disconnect the WebRTC client.

        Idempotent so it can run from ``stop()``, ``cancel()``, and
        ``cleanup()`` without duplicating work.
        """
        await self._stop_tasks()
        await self._client.disconnect()

    async def start(self, frame: StartFrame):
        """Start receiving media from the WebRTC connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task and self._params.video_in_enabled:
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the input transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._teardown()

    async def _receive_audio(self):
        """Background task for receiving audio frames from WebRTC."""
        try:
            audio_iterator = self._client.read_audio_frame()
            async for audio_frame in audio_iterator:
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self, video_source: str):
        """Background task for receiving video frames from WebRTC.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").
        """
        try:
            video_iterator = self._client.read_video_frame(video_source)
            async for video_frame in video_iterator:
                if not video_frame:
                    continue
                due, request_frame = self._video_samplers.sample(_PEER_ID, video_source)
                if not due:
                    continue
                if request_frame:
                    video_frame.text = request_frame.text
                    video_frame.append_to_context = request_frame.append_to_context
                    video_frame.request = request_frame
                await self.push_video_frame(video_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def push_app_message(self, message: Any):
        """Push an application message into the pipeline.

        Args:
            message: The application message to process.
        """
        logger.trace(f"Received app message inside SmallWebRTCInputTransport  {message}")
        await self.broadcast_frame(InputTransportMessageFrame, message=message)

    # Add this method similar to DailyInputTransport.request_participant_image
    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Request an image frame from the participant's video stream.

        When a UserImageRequestFrame is received, this method will store the request
        and the next video frame received will be converted to a UserImageRawFrame.

        Args:
            frame: The user image request frame.
        """
        logger.debug(f"Requesting image from participant: {frame.user_id}")

        # Default to camera if no source specified
        if frame.video_source is None:
            frame.video_source = CAM_VIDEO_SOURCE

        if not self._client.video_source_enabled(frame.video_source):
            error = f"The peer isn't sending {frame.video_source} video."
            await self._answer_image_requests([frame], error)
            return

        # A request for a source that isn't being captured starts receiving it,
        # passing on every frame.
        if not self._video_samplers.is_capturing(_PEER_ID, frame.video_source):
            self._video_samplers.capture(_PEER_ID, frame.video_source, None)
        self._video_samplers.add_request(_PEER_ID, frame.video_source, frame)
        # If we're not already receiving video, try to get a frame now
        if (
            frame.video_source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))
        elif (
            frame.video_source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE)
            )

    async def stop_video(self, video_source: str):
        """Answer the image requests waiting on a video source the peer turned off.

        The source stays captured, so its frames are sampled again if it's turned
        back on.

        Args:
            video_source: ``"camera"`` or ``"screenVideo"``.
        """
        requests = self._video_samplers.take_requests(_PEER_ID, video_source)
        error = f"The peer turned off its {video_source} video."
        await self._answer_image_requests(requests, error)

    async def remove_all_video(self):
        """Stop sampling video and answer the waiting image requests, e.g. on disconnect."""
        requests = self._video_samplers.clear()
        await self._answer_image_requests(requests, "The peer disconnected.")

    async def capture_participant_media(
        self,
        source: str = CAM_VIDEO_SOURCE,
        framerate: int | None = None,
    ):
        """Capture media from a specific participant.

        Args:
            source: Media source to capture from. ("camera", "microphone", or "screenVideo")
            framerate: For a video source, frames per second to pass on. ``0`` passes
                on only the frames that answer image requests, and ``None`` passes on
                every frame.
        """
        if source in (CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE):
            self._video_samplers.capture(_PEER_ID, source, framerate)

        # If we're not already receiving video, try to get a frame now
        if (
            source == MIC_AUDIO_SOURCE
            and not self._receive_audio_task
            and self._params.audio_in_enabled
        ):
            # Start audio reception if it's not already running
            self._receive_audio_task = self.create_task(self._receive_audio())
        elif (
            source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))
        elif (
            source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE)
            )


class SmallWebRTCOutputTransport(BaseOutputTransport):
    """Output transport implementation for SmallWebRTC.

    Handles outgoing audio and video streams to WebRTC peers,
    including transport message sending.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC output transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    def validate_audio_output_policy(self, policy: AudioOutputPolicy | None = None) -> None:
        """Preflight the actual maintained client/track path without creating media.

        Guarded output requires the native writer and paired track callbacks.
        Custom writer overrides must explicitly implement their own capability
        contract rather than inheriting this approval accidentally.
        """
        super().validate_audio_output_policy(policy)
        if (
            type(self).write_audio_frame is not SmallWebRTCOutputTransport.write_audio_frame
            or not isinstance(self._client, SmallWebRTCClient)
            or type(self._client).write_audio_frame is not SmallWebRTCClient.write_audio_frame
            or type(self._client).set_audio_output_callbacks
            is not SmallWebRTCClient.set_audio_output_callbacks
            or type(self._client)._handle_client_connected
            is not SmallWebRTCClient._handle_client_connected
        ):
            raise ValueError("WebRTC transport does not support guarded audio output")
        track = self._client._audio_output_track
        if track is not None and (
            not isinstance(track, RawAudioTrack) or track._context_is_current is None
        ):
            raise ValueError("Existing WebRTC track does not support guarded audio output")
        if policy is not None and self._client._params is not None:
            raise RuntimeError("Audio output policy must be installed before client setup")

    def set_audio_output_policy(self, policy: AudioOutputPolicy) -> None:
        """Install the shared policy and fixed native callbacks before client setup."""
        super().set_audio_output_policy(policy)
        self._client.set_audio_output_callbacks(
            self.capture_audio_output_context,
            self.audio_output_context_is_current,
            self.native_audio_output_context,
        )

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and establish the WebRTC connection.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        await self._client.setup(self._params, setup)
        await self._client.connect()

    async def cleanup(self):
        """Release resources during teardown."""
        await super().cleanup()
        await self._teardown()

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        await self.set_transport_ready(frame)

    async def _teardown(self):
        """Disconnect the WebRTC client.

        Idempotent so it can run from ``stop()``, ``cancel()``, and
        ``cleanup()`` without duplicating work.
        """
        await self._client.disconnect()

    async def stop(self, frame: EndFrame):
        """Stop the output transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._teardown()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message through the WebRTC connection.

        Args:
            frame: The transport message frame to send.
        """
        await self._client.send_message(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool | AudioOutputDiscarded:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The output audio frame to transmit.

        Returns:
            The client's current/ordinary completion, deliberate-discard sentinel,
            or False for an actual unavailable sink.
        """
        return await self._client.write_audio_frame(frame)

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The output video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return await self._client.write_video_frame(frame)


class SmallWebRTCTransport(BaseTransport):
    """WebRTC transport implementation for real-time communication.

    Provides bidirectional audio and video streaming over WebRTC connections
    with support for application messaging and connection event handling.

    Event handlers available:

    - on_client_connected(transport, client): Client connected to WebRTC session
    - on_client_disconnected(transport, client): Client disconnected from WebRTC session
    - on_client_message(transport, message, client): Received a data channel message

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            ...
    """

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: TransportParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the WebRTC transport.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_track_status=self._on_track_status,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input: SmallWebRTCInputTransport | None = None
        self._output: SmallWebRTCOutputTransport | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def get_client_id(self, client: Any) -> str:
        """The id of a client, as passed to ``on_client_connected``.

        Args:
            client: The client, as passed to the transport's client events.

        Returns:
            The peer connection's id.
        """
        return client.pc_id

    def input(self) -> SmallWebRTCInputTransport:
        """Get the input transport processor.

        Returns:
            The input transport for handling incoming media streams.
        """
        if not self._input:
            self._input = SmallWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> SmallWebRTCOutputTransport:
        """Get the output transport processor.

        Returns:
            The output transport for handling outgoing media streams.
        """
        if not self._output:
            self._output = SmallWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame through the transport.

        Args:
            frame: The image frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame through the transport.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def _on_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        if self._input:
            await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_client_connected(self, webrtc_connection):
        """Handle client connection events."""
        # Capture the configured video sources before the event handlers run, so
        # a handler that captures a source itself takes precedence.
        if self._input:
            for video_source, source_params in self._params.video_in_sources.items():
                if self._input._supports_video_in_source(video_source):
                    framerate = 0 if source_params.on_request_only else source_params.framerate
                    await self._input.capture_participant_media(
                        source=video_source, framerate=framerate
                    )

        await self._call_event_handler("on_client_connected", webrtc_connection)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_client_disconnected(self, webrtc_connection):
        """Handle client disconnection events."""
        if self._input:
            await self._input.remove_all_video()
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def _on_track_status(self, source: str, enabled: bool):
        """Handle the peer turning a source on or off."""
        if self._input and source != MIC_AUDIO_SOURCE and not enabled:
            await self._input.stop_video(source)

    async def capture_participant_video(
        self,
        participant_id: str | None = None,
        framerate: int | None | NotGiven = NOT_GIVEN,
        video_source: str = CAM_VIDEO_SOURCE,
        *,
        on_request_only: bool = False,
    ):
        """Capture the peer's camera or screen share at a framerate.

        Args:
            participant_id: The peer's id, as from :meth:`get_client_id`.
                SmallWebRTC has one peer, so the id isn't used to select it.

                .. deprecated:: 1.13.0
                    Pass ``participant_id`` first, as on other transports. Calls
                    without it, ``capture_participant_video(video_source=...)``
                    or ``capture_participant_video("camera")``, keep their old
                    meaning, passing on every frame unless ``framerate`` is given.
                    Will be removed in 2.0.0.

            framerate: Frames per second to pass on, 30 if not given, or ``None``
                for every frame. It doesn't apply with ``on_request_only``.
            video_source: Video source to capture from ("camera" or "screenVideo").
            on_request_only: Pass on only the frames that answer image requests.
        """
        if participant_id is None or participant_id in (CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE):
            warn_deprecated(
                "`SmallWebRTCTransport.capture_participant_video(participant_id=None)` is "
                "deprecated since 1.13.0 and will be removed in 2.0.0. "
                "Use `capture_participant_video(participant_id, framerate, video_source)` "
                "instead.",
                stacklevel=2,
            )
            # The deprecated form is (video_source, framerate=None).
            if participant_id is not None:
                video_source = participant_id
            framerate = framerate if is_given(framerate) else None
        elif not is_given(framerate):
            framerate = 30
        framerate = _capture_framerate(
            framerate, on_request_only, "SmallWebRTCTransport.capture_participant_video"
        )

        if self._input:
            await self._input.capture_participant_media(source=video_source, framerate=framerate)

    async def capture_participant_audio(
        self,
        audio_source: str = MIC_AUDIO_SOURCE,
    ):
        """Capture audio from a specific participant.

        Args:
            audio_source: Audio source to capture from. (currently, "microphone" is the only supported option)
        """
        if self._input:
            await self._input.capture_participant_media(source=audio_source)

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Account TOKEN text only through the provider's explicit usage-report callback.

Real frame processing covers filtering, transforms, turn/end flushes and resets.
Owned fake transport boundaries exercise actual Deepgram/Groq adapters without
provider connections or invoices. A completed local submission is an observation,
not proof of synthesis, delivery or provider billing. Fixtures close generators,
fake sockets and processor-owned tasks even when a scenario fails.
"""

import asyncio
import io
import json
import wave
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq import tts as groq_tts
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_filter import BaseTextFilter
from tests.frame_processor_helpers import frame_processor_setup

SAMPLE_RATE = 16000
AUDIO = b"\x00\x01" * 80
UNICODE = "नमस्ते 🌏 café e\u0301"


def _usage(frames):
    """Return only actual TTS observations, excluding unrelated timing metric frames."""
    return [
        data
        for frame in frames
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, TTSUsageMetricsData)
    ]


class _SubmissionTTS(TTSService):
    """Own a deterministic transport boundary while retaining real TTS frame processing."""

    def __init__(self, *, rejected=(), partial=(), capable=True, **kwargs):
        """Configure submitted/rejected text and metrics capability; create no background work.

        Args:
            rejected: Exact prepared text rejected before its usage callback.
            partial: Submitted text followed by a synthetic downstream audio error.
            capable: Whether this synthetic service supports usage observations.
            **kwargs: Ordinary TTS settings, filters, transforms and aggregation mode.
        """
        super().__init__(
            sample_rate=SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            pause_frame_processing=False,
            append_trailing_space=False,
            stop_frame_timeout_s=0.1,
            max_consecutive_zero_audio_contexts=100,
            settings=TTSSettings(model="synthetic-model", voice="synthetic", language=None),
            **kwargs,
        )
        self.rejected = set(rejected)
        self.partial = set(partial)
        self.capable = capable
        self.submitted = []

    def can_generate_metrics(self) -> bool:
        """Expose the fixture's explicit capability through the ordinary processor gate."""
        return self.capable

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Report only completed fake submissions, retaining accepted usage after later errors."""
        if text in self.rejected:
            yield ErrorFrame(error="Synthetic submission refusal")
            return
        self.submitted.append(text)
        await self.start_tts_usage_metrics(text)
        yield TTSAudioRawFrame(AUDIO, SAMPLE_RATE, 1, context_id=context_id)
        if text in self.partial:
            yield ErrorFrame(error="Synthetic error after submitted audio")


class _DropText(BaseTextFilter):
    """Represent a filter that deliberately removes input before provider submission."""

    async def filter(self, text: str) -> str:
        """Return no speakable text; callers must not infer usage from the original input."""
        return ""


async def _run_frames(service, frames, *, enabled=True):
    """Drive real setup/start/frame/end/cleanup through the existing bounded test pipeline."""
    return await asyncio.wait_for(
        run_test(
            service,
            frames_to_send=frames,
            pipeline_params=PipelineParams(enable_usage_metrics=enabled),
        ),
        timeout=5,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["llm", "end"])
async def test_token_frame_path_reports_one_submitted_unicode_total(finish):
    """LLM completion and EndFrame both flush exactly the accepted prepared characters."""
    service = _SubmissionTTS(text_aggregation_mode=TextAggregationMode.TOKEN)
    texts = ["नमस्ते", " 🌏 café e\u0301."]
    frames = [LLMFullResponseStartFrame(), *[TextFrame(text) for text in texts]]
    if finish == "llm":
        frames.extend([LLMFullResponseEndFrame(), LLMFullResponseEndFrame()])
    down, _ = await _run_frames(service, frames)
    usage = _usage(down)
    assert service.submitted == texts
    assert [item.value for item in usage] == [sum(map(len, texts))]
    assert usage[0].processor == service.name and usage[0].model == "synthetic-model"
    assert usage[0].value != sum(len(text.encode("utf-8")) for text in texts)
    assert service._streamed_text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [TextAggregationMode.SENTENCE, TextAggregationMode.TOKEN])
async def test_independent_speak_frame_is_counted_once_before_shutdown(mode):
    """TTSSpeakFrame restores turn state without losing or duplicating its submitted text."""
    service = _SubmissionTTS(text_aggregation_mode=mode)
    down, _ = await _run_frames(service, [TTSSpeakFrame(UNICODE)])
    assert service.submitted == [UNICODE]
    assert [item.value for item in _usage(down)] == [len(UNICODE)]
    assert service._streamed_text == ""


@pytest.mark.asyncio
async def test_token_mixed_refusal_and_partial_audio_retains_only_reported_text():
    """Failed tokens contribute nothing; an accepted token remains counted after an audio error."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        rejected={" rejected"},
        partial={" accepted"},
    )
    down, up = await _run_frames(
        service,
        [
            LLMFullResponseStartFrame(),
            TextFrame("first"),
            TextFrame(" rejected"),
            TextFrame(" accepted"),
            LLMFullResponseEndFrame(),
        ],
    )
    assert service.submitted == ["first", " accepted"]
    assert [item.value for item in _usage(down)] == [len("first accepted")]
    # AIService routes generator errors upstream, independently of downstream usage.
    assert [frame.error for frame in up if isinstance(frame, ErrorFrame)] == [
        "Synthetic submission refusal",
        "Synthetic error after submitted audio",
    ]
    assert not [frame for frame in down if isinstance(frame, ErrorFrame)]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["filter", "blank", "skip", "unusable", "transform"])
async def test_token_input_that_never_submits_has_no_usage(reason):
    """Pre-provider refusals cannot create usage merely because raw text entered the processor."""
    kwargs = {"text_filters": [_DropText()]} if reason == "filter" else {}

    async def fail_transform(text, text_type):
        """Refuse prepared text before the provider boundary without mutating the input."""
        raise ValueError("Synthetic transform refusal")

    if reason == "transform":
        kwargs["text_transforms"] = [("*", fail_transform)]
    service = _SubmissionTTS(text_aggregation_mode=TextAggregationMode.TOKEN, **kwargs)
    text = TextFrame(" " if reason == "blank" else "never submitted")
    if reason == "skip":
        text.skip_tts = True
    if reason == "unusable":
        await service.set_usable(False)
    down, _ = await _run_frames(
        service,
        [
            LLMFullResponseStartFrame(),
            text,
            LLMFullResponseEndFrame(),
        ],
    )
    assert service.submitted == []
    assert _usage(down) == [] and service._streamed_text == ""


@pytest.mark.asyncio
async def test_token_transform_counts_the_prepared_text_not_the_raw_frame():
    """Unicode transforms change submitted character units without counting original text again."""

    async def transform(text, text_type):
        """Replace each synthetic input with an exact provider-facing Unicode value."""
        return UNICODE

    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, text_transforms=[("*", transform)]
    )
    down, _ = await _run_frames(
        service,
        [
            LLMFullResponseStartFrame(),
            TextFrame("raw"),
            LLMFullResponseEndFrame(),
        ],
    )
    assert service.submitted == [UNICODE]
    assert [item.value for item in _usage(down)] == [len(UNICODE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,capable", [(False, True), (True, False)])
async def test_token_metrics_gates_still_suppress_observations(enabled, capable):
    """The ordinary opt-in/capability gates suppress frames while consuming pending text once."""
    service = _SubmissionTTS(text_aggregation_mode=TextAggregationMode.TOKEN, capable=capable)
    down, _ = await _run_frames(
        service,
        [
            LLMFullResponseStartFrame(),
            TextFrame(UNICODE),
            LLMFullResponseEndFrame(),
        ],
        enabled=enabled,
    )
    assert service.submitted == [UNICODE]
    assert _usage(down) == [] and service._streamed_text == ""


@asynccontextmanager
async def _direct(service, monkeypatch):
    """Own a real processor setup and metric sink without starting provider or pipeline workers.

    The caller may drive native generators and actual control-frame processing.
    Its processor tasks are cancelled/joined before cleanup; missing cleanup is
    an assertion failure rather than a silently ignored owned task.
    """
    manager = TaskManager()
    frames = []

    async def retain(frame, direction=FrameDirection.DOWNSTREAM):
        """Retain frame objects at the processor boundary without downstream network or storage."""
        frames.append(frame)

    monkeypatch.setattr(service, "push_frame", retain)
    await TTSService.setup(service, frame_processor_setup(manager, enable_usage_metrics=True))
    try:
        yield SimpleNamespace(service=service, frames=frames, manager=manager)
    finally:
        try:
            await service._interrupt_audio_contexts()
        finally:
            tasks = list(manager.current_tasks())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await service.cleanup()
        assert not manager.current_tasks()


@pytest.mark.asyncio
async def test_standalone_speak_emits_without_any_future_flush_frame(monkeypatch):
    """A self-contained speak cannot leave its accepted observation waiting for another turn."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        await service.process_frame(TTSSpeakFrame(UNICODE), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]
        assert not service._processing_text and service._streamed_text == ""
        await service.start_tts_usage_metrics("late")
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE), 4]


@pytest.mark.asyncio
async def test_nested_speak_preserves_parent_reported_buffer_until_turn_end(monkeypatch):
    """Nested speak neither flushes nor drops the active parent's accepted characters."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        await service.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        original_context = service._turn_context_id
        service._processing_text = True
        await service.start_tts_usage_metrics("parent")
        await service.process_frame(TTSSpeakFrame(UNICODE), FrameDirection.DOWNSTREAM)
        assert service._turn_context_id == original_context and service._processing_text
        assert service._streamed_text == "parent" + UNICODE and _usage(case.frames) == []
        await service.start_tts_usage_metrics(" tail")
        await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len("parent" + UNICODE + " tail")]
        assert service._streamed_text == ""


@pytest.mark.asyncio
async def test_token_late_report_after_llm_flush_is_immediate_and_not_next_turn(monkeypatch):
    """A real end-frame transition cannot strand an already accepted late callback in the next turn."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        service._processing_text = True
        await service.start_tts_usage_metrics("early")
        assert _usage(case.frames) == []
        await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await service.start_tts_usage_metrics("late")
        assert [item.value for item in _usage(case.frames)] == [5, 4]
        await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [5, 4]
        assert service._streamed_text == ""


@pytest.mark.asyncio
async def test_token_reentrant_report_during_flush_is_not_cleared(monkeypatch):
    """Detach the buffer before its emission await so a late callback remains visible."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        reentered = False

        async def retain_and_report(frame, direction=FrameDirection.DOWNSTREAM):
            """Inject one accepted callback at the exact metric-emission await boundary."""
            nonlocal reentered
            case.frames.append(frame)
            if _usage([frame]) and not reentered:
                reentered = True
                await service.start_tts_usage_metrics("late")

        monkeypatch.setattr(service, "push_frame", retain_and_report)
        service._processing_text = True
        await service.start_tts_usage_metrics("early")
        await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [5, 4]
        assert service._streamed_text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["cancel", "interruption"])
async def test_accepted_token_survives_control_reset_without_duplicate(monkeypatch, control):
    """Cancellation/interruption preserve reported text but do not leak it into a later turn."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        service._processing_text = True
        await service.start_tts_usage_metrics(UNICODE)
        frame = CancelFrame() if control == "cancel" else InterruptionFrame()
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]
        assert service._streamed_text == "" and not service._processing_text
        if control == "interruption":
            await service.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            service._processing_text = True
            await service.start_tts_usage_metrics("next")
            await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            assert [item.value for item in _usage(case.frames)] == [len(UNICODE), 4]
        else:
            await service.cancel(CancelFrame())
            assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["error", "cancellation"])
async def test_cancel_metric_failure_still_joins_audio_owner(monkeypatch, failure_kind):
    """Emission failures propagate unchanged while cancel still releases its audio task.

    The detached observation is never automatically replayed after an unknown
    metric delivery. This tests local task/exception ownership, not durable usage.
    """
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        gate = asyncio.Event()
        task = service.create_task(gate.wait(), name="owned-audio-wait")
        service._audio_context_task = task
        service._processing_text = True
        await service.start_tts_usage_metrics(UNICODE)
        failure = (
            asyncio.CancelledError("Synthetic metric cancellation")
            if failure_kind == "cancellation"
            else RuntimeError("Synthetic metric failure")
        )

        async def refuse_metric(frame, direction=FrameDirection.DOWNSTREAM):
            """Fail only emission of an actual usage frame; retain non-metric cleanup traffic."""
            if _usage([frame]):
                raise failure
            case.frames.append(frame)

        monkeypatch.setattr(service, "push_frame", refuse_metric)
        control = CancelFrame()
        with pytest.raises(type(failure)) as caught:
            await service.process_frame(control, FrameDirection.DOWNSTREAM)
        assert caught.value is failure
        assert task.done() and service._audio_context_task is None
        assert service._streamed_text == "" and not service._processing_text
        assert [frame for frame in case.frames if isinstance(frame, CancelFrame)] == [control]
        await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_interruption_metric_failure_keeps_reset_and_error(monkeypatch):
    """A failed local usage emission cannot bypass interruption's context reset."""
    service = _SubmissionTTS(
        text_aggregation_mode=TextAggregationMode.TOKEN, enable_direct_mode=True
    )
    async with _direct(service, monkeypatch) as case:
        service._processing_text = True
        service._sent_non_whitespace_in_context = True
        await service.start_tts_usage_metrics(UNICODE)
        failure = RuntimeError("Synthetic interruption metric failure")

        async def refuse_metric(frame, direction=FrameDirection.DOWNSTREAM):
            """Raise the exact local emission failure, without replacing lifecycle cleanup."""
            if _usage([frame]):
                raise failure
            case.frames.append(frame)

        monkeypatch.setattr(service, "push_frame", refuse_metric)
        control = InterruptionFrame()
        with pytest.raises(RuntimeError) as caught:
            await service.process_frame(control, FrameDirection.DOWNSTREAM)
        assert caught.value is failure
        assert not service._processing_text and not service._sent_non_whitespace_in_context
        assert service._streamed_text == ""
        assert [frame for frame in case.frames if isinstance(frame, InterruptionFrame)] == [control]


class _Boundary:
    """Own a completed local text submission, never a provider acknowledgement or invoice."""

    def __init__(self, outcome):
        """Start open with one configured outcome and an optional cancellable send gate."""
        self.outcome = outcome
        self.state = State.OPEN
        self.is_active = True
        self.submitted = []
        self.entered = asyncio.Event()
        self.gate = None

    async def submit(self, text):
        """Record text only after an optional gate and before returning a completed submission."""
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.outcome == "error":
            raise RuntimeError("Synthetic submission failure")
        if self.outcome == "cancel":
            raise asyncio.CancelledError("Synthetic send cancellation")
        self.submitted.append(text)

    async def send(self, message):
        """Accept native text messages; control and close messages create no usage."""
        payload = json.loads(message)
        if payload.get("type") == "Speak":
            await self.submit(payload["text"])
        elif payload.get("type") == "text":
            await self.submit(payload["data"]["text"])

    async def send_json(self, message):
        """Mirror the completed SageMaker send boundary without opening its native session."""
        if message.get("type") == "Speak":
            await self.submit(message["text"])

    async def close(self):
        """Release this owned fake connection idempotently without network I/O."""
        self.state = State.CLOSED
        self.is_active = False

    async def close_session(self):
        """Expose the SageMaker cleanup contract using the same owned close state."""
        await self.close()


class _GroqResponse:
    """Own one valid WAV chunk and optional later read failure after request acceptance."""

    def __init__(self, partial):
        """Create a bounded in-memory WAV payload; no SDK transport is allocated."""
        self.partial = partial
        self.closed = False
        payload = io.BytesIO()
        with wave.open(payload, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(AUDIO)
        self.audio = payload.getvalue()

    async def iter_bytes(self):
        """Close the owned response iterator even when a later synthetic read fails."""
        try:
            yield self.audio
            if self.partial:
                raise RuntimeError("Synthetic audio read failure after acceptance")
        finally:
            self.closed = True


@asynccontextmanager
async def _adapter(monkeypatch, provider, mode, outcome="success"):
    """Bind an actual adapter to owned fake transport while retaining real usage processing."""
    boundary = _Boundary(outcome)
    response = _GroqResponse(outcome == "partial")
    common = {"text_aggregation_mode": mode, "enable_direct_mode": True}

    async def create_response(**kwargs):
        """Return native-shaped audio only after the fake request submission completes."""
        await boundary.submit(kwargs["input"])
        return response

    if provider == "websocket":
        service = DeepgramTTSService(api_key="synthetic", sample_rate=SAMPLE_RATE, **common)
        service._websocket = boundary
    elif provider == "sagemaker":
        service = DeepgramSageMakerTTSService(
            endpoint_name="synthetic", region="us-east-1", sample_rate=SAMPLE_RATE, **common
        )
        service._client = boundary
    elif provider == "sarvam":
        service = SarvamTTSService(api_key="synthetic", sample_rate=SAMPLE_RATE, **common)
        service._websocket = boundary
    else:
        client = SimpleNamespace(
            audio=SimpleNamespace(speech=SimpleNamespace(create=create_response))
        )
        # Return only the owned fake client; the native SDK allocates no transport.
        monkeypatch.setattr(groq_tts, "AsyncGroq", lambda **kwargs: client)
        service = groq_tts.GroqTTSService(api_key="synthetic", sample_rate=SAMPLE_RATE, **common)
    try:
        async with _direct(service, monkeypatch) as case:
            yield SimpleNamespace(**vars(case), boundary=boundary, response=response)
    finally:
        await boundary.close()


async def _native_frames(case):
    """Exhaust/close the actual adapter generator without guessing its reported outcome."""
    async with aclosing(case.service.run_tts(UNICODE, "synthetic-context")) as stream:
        return [frame async for frame in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker", "groq"])
@pytest.mark.parametrize("mode", [TextAggregationMode.SENTENCE, TextAggregationMode.TOKEN])
async def test_native_adapter_success_has_one_observation(monkeypatch, provider, mode):
    """All three adapters report one completed submission in sentence and token aggregation."""
    async with _adapter(monkeypatch, provider, mode) as case:
        case.service._processing_text = True
        frames = await _native_frames(case)
        assert not any(isinstance(frame, ErrorFrame) for frame in frames)
        assert case.boundary.submitted == [UNICODE]
        if mode == TextAggregationMode.TOKEN:
            assert _usage(case.frames) == []
        await case.service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]
        await case.service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker", "groq"])
@pytest.mark.parametrize("outcome", ["error", "cancel"])
async def test_native_adapter_failed_submission_reports_nothing(monkeypatch, provider, outcome):
    """Error/cancellation before a completed local submission cannot create a usage callback."""
    async with _adapter(monkeypatch, provider, TextAggregationMode.TOKEN, outcome) as case:
        case.service._processing_text = True
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await _native_frames(case)
        else:
            frames = await _native_frames(case)
            assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)
        await case.service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert case.boundary.submitted == [] and _usage(case.frames) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker", "groq"])
async def test_native_adapter_blocked_send_cancellation_is_owned(monkeypatch, provider):
    """Cancel and join a real blocked adapter generator without counting or retrying its text."""
    async with _adapter(monkeypatch, provider, TextAggregationMode.TOKEN) as case:
        case.boundary.gate = asyncio.Event()
        case.service._processing_text = True
        task = case.service.create_task(_native_frames(case), name="owned-blocked-send")
        try:
            await asyncio.wait_for(case.boundary.entered.wait(), timeout=1)
            assert case.boundary.submitted == [] and _usage(case.frames) == []
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await case.service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert case.boundary.submitted == [] and _usage(case.frames) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker", "groq"])
async def test_native_accepted_callback_survives_later_cancel(monkeypatch, provider):
    """An accepted request is counted once even if synthesis is cancelled before turn flush."""
    async with _adapter(monkeypatch, provider, TextAggregationMode.TOKEN) as case:
        case.service._processing_text = True
        await _native_frames(case)
        assert _usage(case.frames) == [] and case.boundary.submitted == [UNICODE]
        await case.service.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]
        assert case.service._streamed_text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [TextAggregationMode.SENTENCE, TextAggregationMode.TOKEN])
async def test_groq_partial_audio_error_retains_accepted_observation(monkeypatch, mode):
    """A later audio-stream failure cannot erase a request accepted before iteration began."""
    async with _adapter(monkeypatch, "groq", mode, "partial") as case:
        case.service._processing_text = True
        frames = await _native_frames(case)
        assert len(frames) == 2 and isinstance(frames[0], TTSAudioRawFrame)
        assert isinstance(frames[1], ErrorFrame) and case.response.closed
        await case.service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        assert [item.value for item in _usage(case.frames)] == [len(UNICODE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker"])
@pytest.mark.parametrize("failure_kind", ["error", "cancellation"])
async def test_native_cancel_disconnects_before_metric_failure(monkeypatch, provider, failure_kind):
    """A metric error/cancellation cannot bypass either concrete adapter's native teardown."""
    async with _adapter(monkeypatch, provider, TextAggregationMode.TOKEN) as case:
        case.service._processing_text = True
        await _native_frames(case)
        failure = (
            asyncio.CancelledError("Synthetic emission cancellation")
            if failure_kind == "cancellation"
            else RuntimeError("Synthetic emission failure")
        )

        async def fail_after_disconnect(frame, direction=FrameDirection.DOWNSTREAM):
            """Assert actual socket/session shutdown before failing usage transport."""
            if _usage([frame]):
                assert case.boundary.state is State.CLOSED and not case.boundary.is_active
                raise failure
            case.frames.append(frame)

        monkeypatch.setattr(case.service, "push_frame", fail_after_disconnect)
        control = CancelFrame()
        with pytest.raises(type(failure)) as caught:
            await case.service.process_frame(control, FrameDirection.DOWNSTREAM)
        assert caught.value is failure
        assert case.boundary.state is State.CLOSED and not case.boundary.is_active
        assert case.service._streamed_text == ""
        assert [frame for frame in case.frames if isinstance(frame, CancelFrame)] == [control]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["websocket", "sagemaker"])
async def test_native_cancel_slow_metric_sink_cannot_delay_disconnect(monkeypatch, provider):
    """Join cancellation while metric transport blocks only after actual provider cleanup."""
    async with _adapter(monkeypatch, provider, TextAggregationMode.TOKEN) as case:
        case.service._processing_text = True
        await _native_frames(case)
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_metric(frame, direction=FrameDirection.DOWNSTREAM):
            """Expose the emission await without touching the completed transport teardown."""
            if _usage([frame]):
                entered.set()
                await release.wait()
            case.frames.append(frame)

        monkeypatch.setattr(case.service, "push_frame", hold_metric)
        control = CancelFrame()
        task = case.service.create_task(
            case.service.process_frame(control, FrameDirection.DOWNSTREAM),
            name="owned-cancel-metric-wait",
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert not task.done()
            assert case.boundary.state is State.CLOSED and not case.boundary.is_active
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert case.service._streamed_text == ""
        assert [frame for frame in case.frames if isinstance(frame, CancelFrame)] == [control]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["error", "cancellation", "blocked"])
async def test_interruptible_reconnect_precedes_metric_transport(monkeypatch, failure_kind):
    """A concrete Sarvam interruption resets its socket before a failing or blocked metric sink.

    Only creation of the replacement connection is inert. The actual interruptible
    hook and Sarvam disconnect run; no receive/keepalive tasks or provider run.
    """
    async with _adapter(monkeypatch, "sarvam", TextAggregationMode.TOKEN) as case:
        service = case.service
        service._processing_text = True
        await _native_frames(case)
        # A real pipeline sets this on TTSStartedFrame before playback is confirmed.
        service._tts_started = True
        replacement = _Boundary("success")
        connections = []
        entered, release = asyncio.Event(), asyncio.Event()
        failure = (
            asyncio.CancelledError("Synthetic interruption emission cancellation")
            if failure_kind == "cancellation"
            else RuntimeError("Synthetic interruption emission failure")
        )

        async def connect_owned_replacement():
            """Reset native connection flags and provide one owned replacement socket only."""
            await WebsocketService._connect(service)
            service._websocket = replacement
            connections.append(replacement)

        async def metric_after_reconnect(frame, direction=FrameDirection.DOWNSTREAM):
            """Check native disconnect/reconnect order at the actual usage-emission boundary."""
            if _usage([frame]):
                assert case.boundary.state is State.CLOSED and connections == [replacement]
                assert service._websocket is replacement
                entered.set()
                if failure_kind == "blocked":
                    await release.wait()
                else:
                    raise failure
            case.frames.append(frame)

        monkeypatch.setattr(service, "_connect", connect_owned_replacement)
        monkeypatch.setattr(service, "push_frame", metric_after_reconnect)
        control = InterruptionFrame()
        try:
            if failure_kind == "blocked":
                task = service.create_task(
                    service.process_frame(control, FrameDirection.DOWNSTREAM),
                    name="owned-interruption-metric-wait",
                )
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    assert not task.done() and connections == [replacement]
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            else:
                with pytest.raises(type(failure)) as caught:
                    await service.process_frame(control, FrameDirection.DOWNSTREAM)
                assert caught.value is failure
            assert service._streamed_text == "" and connections == [replacement]
            assert [frame for frame in case.frames if isinstance(frame, InterruptionFrame)] == [
                control
            ]
        finally:
            release.set()
            await replacement.close()

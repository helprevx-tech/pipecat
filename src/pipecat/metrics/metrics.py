#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Metrics and one-shot usage observation models for Pipecat framework.

This module defines Pydantic models for various types of metrics data
collected throughout the pipeline, including timing, token usage, and
processing statistics.
"""

from dataclasses import dataclass, fields

from pydantic import BaseModel

from pipecat.utils.deprecation import deprecated


class MetricsData(BaseModel):
    """Base class for all metrics data.

    Parameters:
        processor: Name of the processor generating the metrics.
        model: Optional model name associated with the metrics.
    """

    processor: str
    model: str | None = None


class TTFBMetricsData(MetricsData):
    """Time To First Byte (TTFB) metrics data.

    Parameters:
        value: TTFB measurement in seconds.
    """

    value: float


class TTFAMetricsData(MetricsData):
    """Time To First Audio (TTFA) metrics data.

    Measures the time from a TTS request to the first audible audio sample,
    i.e. time-to-first-byte plus any leading silence the service pads onto the
    start of its response. ``ttfa`` is reported with its breakdown so consumers
    can see how much of the perceived latency is silence padding rather than
    service response time, without correlating a separate ``TTFBMetricsData``.

    Parameters:
        ttfa: TTFA measurement in seconds (``ttfb`` plus ``leading_silence``).
        ttfb: Time-to-first-byte that TTFA builds on, in seconds. This mirrors
            the standalone ``TTFBMetricsData`` (emitted earlier) for convenience;
            it is not a separate measurement, so don't aggregate both.
        leading_silence: Silence padding before the first audible sample, in
            seconds (``ttfa`` minus ``ttfb``).
    """

    ttfa: float
    ttfb: float
    leading_silence: float


class TTFATMetricsData(MetricsData):
    """Time To First Answer Token (TTFAT) metrics data.

    Measures the time from an LLM request to the first token of the answer the
    caller sees, i.e. time-to-first-byte plus any reasoning the model streamed
    first. ``ttfat`` is reported with its breakdown so consumers can see how much
    of the perceived latency is the model thinking rather than responding,
    without correlating a separate ``TTFBMetricsData``.

    A turn that answers with a tool call rather than text ends the measurement at
    the call, taken where it first appears in the stream. Answering from a tool
    result takes a second inference, so such a turn reports twice — once for the
    call, once for the answer built from its result. Each figure covers one
    inference; consumers wanting one per user turn keep the first, which is the
    one that measures how quickly the model began responding at all.

    Reported only by LLM services that answer in text. Speech-to-speech services
    answer in audio, which has no answer token to measure to.

    Parameters:
        ttfat: TTFAT measurement in seconds (``ttfb`` plus ``thinking_time``).
        ttfb: Time-to-first-byte that TTFAT builds on, in seconds. This mirrors
            the standalone ``TTFBMetricsData`` (emitted earlier) for convenience;
            it is not a separate measurement, so don't aggregate both.
        thinking_time: Time between the model's first output and the first answer
            token, in seconds (``ttfat`` minus ``ttfb``). Reasoning is the usual
            reason a model streams something before it starts answering, though a
            model that reasons none still spends a little time here.
    """

    ttfat: float
    ttfb: float
    thinking_time: float


class ProcessingMetricsData(MetricsData):
    """General processing time metrics data.

    Parameters:
        value: Processing time measurement in seconds.
    """

    value: float


@dataclass(frozen=True, slots=True)
class LLMInferenceUsage:
    """Optional provider counters received by an out-of-pipeline inference.

    Missing counters remain None; explicit zero is a measurement. Prompt counts
    retain provider net/gross cache semantics, as with LLMTokenUsage. Totals may
    be independently reported rather than the sum of the visible components.
    Counters use the cross-language safe integer range and contain no request,
    response, timestamp or financial data.

    Parameters:
        prompt_tokens: Known prompt count, including tool input when known.
        completion_tokens: Known completion count, including reasoning when known.
        total_tokens: Reported total or a sum of completely known components.
        cache_read_input_tokens: Known cache-read input count.
        cache_creation_input_tokens: Known cache-write input count.
        reasoning_tokens: Known reasoning portion of generated output.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self):
        """Reject malformed reported counters without coercing booleans or strings."""
        if any(not self._valid_count(getattr(self, item.name)) for item in fields(self)):
            raise ValueError("Invalid one-shot token count")

    @staticmethod
    def _valid_count(value: object) -> bool:
        """Accept unknown or exact nonnegative integers representable by JS clients."""
        return value is None or type(value) is int and 0 <= value <= 9007199254740991

    @staticmethod
    def sum_tokens(*values: int | None) -> int | None:
        """Sum only completely known valid addends; derived overflow stays unknown."""
        if any(not LLMInferenceUsage._valid_count(value) for value in values):
            raise ValueError("Invalid one-shot token count")
        if not values or any(value is None for value in values):
            return None
        total = sum(value for value in values if value is not None)
        return total if LLMInferenceUsage._valid_count(total) else None


class LLMTokenUsage(BaseModel):
    """Token usage statistics for LLM operations.

    Services differ in whether their input count is reported net or gross of the
    prompt cache. Anthropic and Bedrock report ``prompt_tokens`` net, with the
    cache counts alongside it; OpenAI-compatible services report it gross, with
    the cache counts already inside it. ``total_tokens`` is the gross figure
    either way, so it stays comparable across services, and it is therefore not
    always ``prompt_tokens + completion_tokens``. Read the cache fields for the
    breakdown rather than subtracting.

    Per-bucket cost accounting has to account for that difference: adding the
    cache counts to ``prompt_tokens`` double-counts them on a service that
    reports gross.

    Parameters:
        prompt_tokens: Number of tokens in the input prompt, net or gross of the
            cache counts depending on the service. Includes tool results the
            provider feeds back to the model, which are billed as input.
        completion_tokens: Number of tokens in the generated completion,
            including any reasoning tokens.
        total_tokens: Total number of tokens used, including any cached input
            tokens.
        cache_read_input_tokens: Number of tokens read from cache, if applicable.
            Billed below the input rate.
        cache_creation_input_tokens: Number of tokens written to cache, if
            applicable. Billed at or above the input rate.
        reasoning_tokens: Number of completion tokens used for reasoning, if applicable.
        input_audio_tokens: Number of prompt tokens that were audio, if applicable.
        output_audio_tokens: Number of completion tokens that were audio, if applicable.
        cache_read_input_audio_tokens: Number of cache-read tokens that were audio, if applicable.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    cache_read_input_audio_tokens: int | None = None


class LLMUsageMetricsData(MetricsData):
    """LLM token usage metrics data.

    Parameters:
        value: Token usage statistics for the LLM operation.
    """

    value: LLMTokenUsage


class STTUsage(BaseModel):
    """Usage statistics for STT operations.

    Reports raw usage, not cost. Values are incremental deltas since the
    previous usage report; consumers sum them across a session.

    Continuous STT services stream all input audio (including silence), so
    their summed ``audio_seconds`` approximates the stream duration — which
    is what most streaming providers bill. Segmented STT services submit
    only the detected speech segments, so their sum covers just those
    segments.

    Parameters:
        audio_seconds: Client-measured seconds of audio submitted to the
            service since the last usage report.
    """

    audio_seconds: float


class STTUsageMetricsData(MetricsData):
    """Speech-to-Text usage metrics data.

    Parameters:
        value: Usage statistics for the STT operation.
    """

    value: STTUsage


class TTSUsageMetricsData(MetricsData):
    """Text-to-Speech usage metrics data.

    Parameters:
        value: Number of characters processed by TTS.
    """

    value: int


class TextAggregationMetricsData(MetricsData):
    """Text aggregation time metrics data.

    Measures the time from the first LLM token to the first complete sentence,
    representing the latency cost of sentence aggregation in the TTS pipeline.

    Parameters:
        value: Aggregation time in seconds.
    """

    value: float


class TurnMetricsData(MetricsData):
    """Metrics data for turn detection predictions.

    Parameters:
        is_complete: Whether the turn is predicted to be complete.
        probability: Confidence probability of the turn completion prediction.
        e2e_processing_time_ms: End-to-end processing time in milliseconds,
            measured from VAD speech-to-silence transition to turn completion.
    """

    is_complete: bool
    probability: float
    e2e_processing_time_ms: float


@deprecated(
    "`SmartTurnMetricsData` is deprecated since 0.0.104 and will be removed in 2.0.0. "
    "Use `TurnMetricsData` instead."
)
class SmartTurnMetricsData(TurnMetricsData):
    """Metrics data for smart turn predictions.

    .. deprecated:: 0.0.104
        Use :class:`TurnMetricsData` instead. Will be removed in 2.0.0.

    Parameters:
        inference_time_ms: Time taken for inference in milliseconds.
        server_total_time_ms: Total server processing time in milliseconds.
    """

    inference_time_ms: float = 0.0
    server_total_time_ms: float = 0.0

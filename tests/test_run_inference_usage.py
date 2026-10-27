#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""One-shot usage observations keep received counters separate from inference results.

Real services/adapters/events run against inert SDK clients. No provider request,
pipeline metrics, background observer task or mutable SDK usage object is needed
to retain an optional invocation-local observation.
"""

import asyncio
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.metrics.metrics import LLMInferenceUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aayiqagent.llm import AayiqAgentLLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.xai.llm import GrokLLMService

FAMILIES = ("openai", "anthropic", "google", "aws", "responses", "responses-http")
SAFE_INTEGER = 9007199254740991


def raw_usage(family, *, zero=False):
    """Build independent actual-family field shapes; no requested token limit supplies a count."""
    if family in {"openai", "sarvam", "aayiqagent", "grok"}:
        return SimpleNamespace(
            prompt_tokens=0 if zero else 11,
            completion_tokens=0 if zero else 7,
            total_tokens=0 if zero else 18,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=0 if zero else 2, cache_write_tokens=None
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0 if zero else 3),
        )
    if family == "anthropic":
        return SimpleNamespace(
            input_tokens=0 if zero else 11,
            output_tokens=0 if zero else 7,
            cache_read_input_tokens=0 if zero else 3,
            cache_creation_input_tokens=0 if zero else 2,
            output_tokens_details=SimpleNamespace(thinking_tokens=0 if zero else 1),
        )
    if family == "google":
        return SimpleNamespace(
            prompt_token_count=0 if zero else 11,
            tool_use_prompt_token_count=0 if zero else 3,
            candidates_token_count=0 if zero else 5,
            thoughts_token_count=0 if zero else 2,
            total_token_count=0 if zero else 100,
            cached_content_token_count=None,
        )
    if family == "aws":
        return {
            "inputTokens": 0 if zero else 11,
            "outputTokens": 0 if zero else 7,
            "cacheReadInputTokens": 0 if zero else 3,
            "cacheWriteInputTokens": 0 if zero else 2,
            "totalTokens": 0 if zero else 999,
        }
    return SimpleNamespace(
        input_tokens=0 if zero else 11,
        output_tokens=0 if zero else 7,
        total_tokens=0 if zero else 18,
        input_tokens_details=SimpleNamespace(
            cached_tokens=0 if zero else 2, cache_write_tokens=None
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0 if zero else 3),
    )


def prepared(family, usage, text="answer"):
    """Construct a real service/adapter and replace only its SDK transport with an inert response."""
    client, exited = AsyncMock(), None
    if family in {"openai", "sarvam", "aayiqagent", "grok", "responses", "responses-http"}:
        service_type = {
            "openai": OpenAILLMService,
            "sarvam": SarvamLLMService,
            "aayiqagent": AayiqAgentLLMService,
            "grok": GrokLLMService,
            "responses": OpenAIResponsesLLMService,
            "responses-http": OpenAIResponsesHttpLLMService,
        }[family]
        constructor = "_create_client" if family.startswith("responses") else "create_client"
        arguments = (
            {
                "gateway_base_url": "https://gateway.example.invalid",
                "invocation_token": "synthetic-token",
                "request_id": "synthetic-invocation",
                "model": "synthetic-model",
            }
            if family == "aayiqagent"
            else {"api_key": "synthetic-key"}
        )
        with patch.object(service_type, constructor, return_value=client):
            service = service_type(**arguments)
        if family.startswith("responses"):
            response = SimpleNamespace(usage=usage, output_text=text)
            request = client.responses.create
        else:
            response = SimpleNamespace(
                usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )
            request = client.chat.completions.create
    elif family == "anthropic":
        service = AnthropicLLMService(api_key="synthetic-key", client=client)
        response = SimpleNamespace(
            usage=usage, content=[] if text is None else [SimpleNamespace(text=text)]
        )
        request = client.beta.messages.create
    elif family == "google":
        with patch.object(GoogleLLMService, "create_client"):
            service = GoogleLLMService(api_key="synthetic-key")
        service._client = client
        response = SimpleNamespace(
            usage_metadata=usage,
            candidates=[]
            if text is None
            else [
                SimpleNamespace(
                    finish_reason=None, content=SimpleNamespace(parts=[SimpleNamespace(text=text)])
                )
            ],
        )
        request = client.aio.models.generate_content
    else:
        session, context = MagicMock(), AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = False
        session.create_client.return_value = context
        with patch(
            "pipecat.services.aws.llm.aiobotocore.session.get_session", return_value=session
        ):
            service = AWSBedrockLLMService(
                aws_access_key="synthetic-key",
                aws_secret_key="synthetic-secret",
                aws_region="us-east-1",
            )
        response = {
            "usage": usage,
            "output": {"message": {"content": [] if text is None else [{"text": text}]}},
        }
        request, exited = client.converse, context.__aexit__
    request.return_value = response
    service.start_llm_usage_metrics = AsyncMock()
    return service, request, response, exited


def collect(service):
    """Retain only the current service's inline data-only observation for assertions."""
    observations = []

    def observed(owner, usage):
        """A local collector neither dispatches a provider nor borrows another service's observation."""
        assert owner is service
        observations.append(usage)

    service.add_event_handler("on_inference_usage", observed)
    return observations, observed


async def infer(service):
    """Use actual adapter conversion on a minimal human context without tools, files or external dependencies."""
    return await service.run_inference(LLMContext(messages=[{"role": "user", "content": "Hello"}]))


@pytest.mark.asyncio
async def test_one_shot_usage_is_observed_inline_without_an_extra_request():
    """The same completed SDK response produces one observation before the unchanged text return."""
    client = AsyncMock()
    raw = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=None),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=raw, choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    with patch.object(OpenAILLMService, "create_client", return_value=client):
        service = OpenAILLMService(api_key="synthetic-key")
    observations = []

    def observed(owner, usage):
        """Copy only invocation-local immutable counters; the observer cannot issue a request."""
        assert owner is service
        observations.append(usage)

    service.add_event_handler("on_inference_usage", observed)
    answer = await service.run_inference(
        LLMContext(messages=[{"role": "user", "content": "Hello"}])
    )
    assert answer == "answer"
    client.chat.completions.create.assert_awaited_once()
    assert len(observations) == 1
    assert observations[0].prompt_tokens == 11
    assert observations[0].completion_tokens == 7
    assert observations[0].total_tokens == 18
    assert observations[0].cache_read_input_tokens == 2
    assert observations[0].cache_creation_input_tokens is None
    assert observations[0].reasoning_tokens == 3
    assert service._event_tasks == set()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.asyncio
async def test_actual_families_preserve_known_counts_and_one_request(family):
    """One actual response produces one snapshot without pipeline metrics or invented independent-total equality."""
    service, request, _, exited = prepared(family, raw_usage(family))
    observations, _ = collect(service)
    assert await infer(service) == "answer"
    assert service.supports_inference_usage is True
    request.assert_awaited_once()
    service.start_llm_usage_metrics.assert_not_awaited()
    assert service._event_tasks == set() and len(observations) == 1
    usage = observations[0]
    expected = {"anthropic": (11, 7, 23), "google": (14, 7, 100), "aws": (11, 7, 999)}.get(
        family, (11, 7, 18)
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == expected
    if exited is not None:
        exited.assert_awaited_once()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.asyncio
async def test_explicit_zero_and_missing_usage_have_different_observations(family):
    """Zero counters survive while an absent response usage object remains unknown."""
    service, request, response, _ = prepared(family, raw_usage(family, zero=True))
    observations, _ = collect(service)
    assert await infer(service) == "answer"
    assert observations[0] is not None
    assert (
        observations[0].prompt_tokens,
        observations[0].completion_tokens,
        observations[0].total_tokens,
    ) == (0, 0, 0)
    if family == "aws":
        response["usage"] = None
    else:
        setattr(response, "usage_metadata" if family == "google" else "usage", None)
    assert await infer(service) == "answer"
    assert observations == [observations[0], None]
    assert request.await_count == 2


@pytest.mark.parametrize("name", tuple(LLMInferenceUsage.__dataclass_fields__))
@pytest.mark.parametrize(
    "bad", [True, False, "1", 1.0, -1, SAFE_INTEGER + 1, [], {}, float("nan"), float("inf")]
)
def test_snapshot_rejects_malformed_present_counts_without_coercion(name, bad):
    """Every public counter rejects wrong types/ranges rather than promoting malformed evidence to measured values."""
    with pytest.raises(ValueError, match="Invalid one-shot token count"):
        LLMInferenceUsage(**{name: bad})


def test_nullable_snapshot_is_frozen_and_derives_only_known_safe_sums():
    """Unknown, explicit zero and independently reported counters remain distinct under immutable ownership."""
    observed = LLMInferenceUsage(prompt_tokens=0, total_tokens=SAFE_INTEGER)
    assert asdict(observed) == {
        "prompt_tokens": 0,
        "completion_tokens": None,
        "total_tokens": SAFE_INTEGER,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "reasoning_tokens": None,
    }
    with pytest.raises(FrozenInstanceError):
        observed.prompt_tokens = 1
    assert LLMInferenceUsage.sum_tokens(0, 0) == 0
    assert LLMInferenceUsage.sum_tokens(0, SAFE_INTEGER) == SAFE_INTEGER
    assert LLMInferenceUsage.sum_tokens(SAFE_INTEGER, 1) is None
    assert LLMInferenceUsage.sum_tokens(1, None) is None
    assert LLMInferenceUsage.sum_tokens() is None
    with pytest.raises(ValueError):
        LLMInferenceUsage.sum_tokens(None, True)
    assert LLMService.supports_inference_usage is False


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bad", [True, "11", -1, SAFE_INTEGER + 1])
@pytest.mark.asyncio
async def test_malformed_sdk_visible_usage_is_unavailable_without_breaking_the_answer(family, bad):
    """Strict validation happens before converters can coerce actual malformed raw fields."""
    raw = raw_usage(family)
    if family == "aws":
        raw["inputTokens"] = bad
    else:
        setattr(
            raw,
            "prompt_token_count"
            if family == "google"
            else "prompt_tokens"
            if family == "openai"
            else "input_tokens",
            bad,
        )
    service, request, _, _ = prepared(family, raw)
    observations, _ = collect(service)
    assert await infer(service) == "answer"
    assert observations == [None]
    request.assert_awaited_once()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.asyncio
async def test_partial_primary_fields_preserve_other_independent_measurements(family):
    """A missing addend makes its derived count unknown without erasing an actual independent total."""
    raw = raw_usage(family)
    if family == "openai":
        raw.completion_tokens = None
        expected = (11, None, 18)
    elif family == "anthropic":
        raw.cache_read_input_tokens = None
        expected = (11, 7, None)
    elif family == "google":
        raw.tool_use_prompt_token_count = None
        raw.thoughts_token_count = None
        expected = (None, None, 100)
    elif family == "aws":
        raw["inputTokens"] = None
        expected = (None, 7, 999)
    else:
        raw.output_tokens = None
        expected = (11, None, 18)
    service, _, _, _ = prepared(family, raw)
    observations, _ = collect(service)
    await infer(service)
    assert (
        observations[0].prompt_tokens,
        observations[0].completion_tokens,
        observations[0].total_tokens,
    ) == expected


@pytest.mark.parametrize("family", ("openai", "grok"))
@pytest.mark.asyncio
async def test_missing_optional_detail_attributes_do_not_erase_known_primary_counts(family):
    """A lenient SDK object's omitted optional attributes remain unknown without invalidating independent primary measurements."""
    raw = raw_usage(family)
    raw.prompt_tokens_details = SimpleNamespace(cache_write_tokens=5)
    raw.completion_tokens_details = SimpleNamespace()
    service, request, _, _ = prepared(family, raw)
    observations, _ = collect(service)
    await infer(service)
    observed = observations[0]
    assert (observed.prompt_tokens, observed.completion_tokens, observed.total_tokens) == (
        11,
        7,
        18,
    )
    assert observed.cache_read_input_tokens is None and observed.cache_creation_input_tokens == 5
    assert observed.reasoning_tokens is None
    assert not hasattr(raw.prompt_tokens_details, "cached_tokens")
    request.assert_awaited_once()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.asyncio
async def test_malformed_nested_container_is_not_treated_as_missing_optional_usage(family):
    """Primitive nested usage cannot masquerade as an absent detail object while other counters appear valid."""
    raw = raw_usage(family)
    if family == "openai":
        raw.prompt_tokens_details = []
    elif family == "anthropic":
        raw.output_tokens_details = "invalid"
    elif family.startswith("responses"):
        raw.input_tokens_details = True
    else:
        raw = []
    service, request, _, _ = prepared(family, raw)
    observations, _ = collect(service)
    assert await infer(service) == "answer"
    assert observations == [None]
    request.assert_awaited_once()


@pytest.mark.parametrize("family", ("anthropic", "google", "aws"))
@pytest.mark.asyncio
async def test_derived_overflow_keeps_independent_valid_counts(family):
    """Derived counters cannot exceed the shared safe bound or replace a reported total."""
    raw = raw_usage(family)
    if family == "anthropic":
        raw.input_tokens = SAFE_INTEGER
        expected = (SAFE_INTEGER, 7, None)
    elif family == "google":
        raw.prompt_token_count = SAFE_INTEGER
        expected = (None, 7, 100)
    else:
        raw["inputTokens"], raw["totalTokens"] = SAFE_INTEGER, None
        expected = (SAFE_INTEGER, 7, None)
    service, _, _, _ = prepared(family, raw)
    observations, _ = collect(service)
    await infer(service)
    assert (
        observations[0].prompt_tokens,
        observations[0].completion_tokens,
        observations[0].total_tokens,
    ) == expected


@pytest.mark.parametrize(
    "prompt,completion,total,reasoning,expected",
    [
        (32, 9, 135, 94, 103),
        (32, 103, 135, 94, 103),
        (None, 9, 135, 94, None),
        (32, 9, None, 94, None),
    ],
)
@pytest.mark.asyncio
async def test_grok_reasoning_is_normalized_only_when_its_inclusion_is_known(
    prompt, completion, total, reasoning, expected
):
    """The existing complete normalizer remains authoritative; partial inputs never guess reasoning inclusion."""
    raw = raw_usage("grok")
    raw.prompt_tokens, raw.completion_tokens, raw.total_tokens = prompt, completion, total
    raw.completion_tokens_details.reasoning_tokens = reasoning
    service, request, _, _ = prepared("grok", raw)
    observations, _ = collect(service)
    await infer(service)
    assert (
        observations[0].completion_tokens == expected
        and observations[0].reasoning_tokens == reasoning
    )
    assert observations[0].total_tokens == total
    request.assert_awaited_once()


@pytest.mark.parametrize("family", ("sarvam", "aayiqagent"))
@pytest.mark.asyncio
async def test_openai_delegators_emit_once_without_changing_their_validation(family):
    """Delegating provider wrappers inherit capability and the one original response event."""
    service, request, _, _ = prepared(family, raw_usage(family))
    observations, _ = collect(service)
    assert await infer(service) == "answer"
    assert service.supports_inference_usage and len(observations) == 1
    request.assert_awaited_once()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.asyncio
async def test_empty_answer_still_has_received_usage_and_no_second_call(family):
    """Answer absence is not proof that a received provider response consumed no tokens."""
    service, request, _, _ = prepared(family, raw_usage(family), text=None)
    observations, _ = collect(service)
    assert await infer(service) is None
    assert len(observations) == 1 and observations[0] is not None
    request.assert_awaited_once()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_provider_failure_or_cancellation_before_response_has_no_usage(family, cancelled):
    """Unreceived responses never emit a guessed zero/None record or retry the failed SDK request."""
    service, request, _, exited = prepared(family, raw_usage(family))
    failure = asyncio.CancelledError() if cancelled else RuntimeError("synthetic provider failure")
    request.side_effect = failure
    observations, _ = collect(service)
    with pytest.raises(type(failure)):
        await infer(service)
    assert observations == []
    request.assert_awaited_once()
    if exited is not None:
        exited.assert_awaited_once()


@pytest.mark.parametrize("family", ("openai", "aws"))
@pytest.mark.asyncio
async def test_observer_failure_and_cancellation_follow_existing_inline_event_contract(family):
    """Ordinary observer failure leaves the answer intact; cancellation propagates and owned client cleanup still runs."""
    for cancelled in (False, True):
        service, request, _, exited = prepared(family, raw_usage(family))
        observations = []

        def failing(owner, usage, *, observations=observations, cancelled=cancelled):
            """Use a safe synthetic error to exercise current dispatcher semantics, never provider exception contents."""
            observations.append(usage)
            if cancelled:
                raise asyncio.CancelledError
            raise RuntimeError("synthetic observer failure")

        service.add_event_handler("on_inference_usage", failing)
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await infer(service)
        else:
            assert await infer(service) == "answer"
        assert len(observations) == 1
        request.assert_awaited_once()
        assert service._event_tasks == set()
        if exited is not None:
            exited.assert_awaited_once()


@pytest.mark.asyncio
async def test_unobserved_usage_is_not_inspected_and_removed_handler_cannot_capture_later_calls():
    """Optional observation cannot add decoding requirements to existing unobserved one-shot callers."""
    service, request, _, _ = prepared("openai", raw_usage("openai"))
    observations, handler = collect(service)
    assert await infer(service) == "answer"
    service.remove_event_handler("on_inference_usage", handler)

    class UnobservedResponse:
        """Native answer with a usage property that unobserved execution must not touch."""

        choices = [SimpleNamespace(message=SimpleNamespace(content="unchanged"))]

        @property
        def usage(self):
            """Fail if optional bookkeeping introduces a new response requirement."""
            raise AssertionError("Unobserved usage was accessed")

    request.return_value = UnobservedResponse()
    assert await infer(service) == "unchanged"
    assert len(observations) == 1 and request.await_count == 2


@pytest.mark.asyncio
async def test_malformed_answer_preserves_received_usage_before_existing_extraction_failure():
    """A received usage observation is independent of whether the original text extraction succeeds."""
    service, request, response, _ = prepared("openai", raw_usage("openai"))
    response.choices = []
    observations, _ = collect(service)
    with pytest.raises(IndexError):
        await infer(service)
    assert len(observations) == 1 and observations[0].total_tokens == 18
    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_event_handlers_are_awaited_inline_without_background_tasks():
    """The existing event API's synchronous mode owns an async observer until it returns."""
    service, request, _, _ = prepared("openai", raw_usage("openai"))
    observed = []

    async def capture(owner, usage):
        """Yield only to the local loop while retaining the same invocation and immutable counter snapshot."""
        assert owner is service
        observed.append("started")
        await asyncio.sleep(0)
        observed.append(usage)

    service.add_event_handler("on_inference_usage", capture)
    assert await infer(service) == "answer"
    assert observed[0] == "started" and observed[1].total_tokens == 18
    assert service._event_tasks == set()
    request.assert_awaited_once()

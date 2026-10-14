"""Verify native and gateway Flux emit each provider partial once, in turn order.

Synthetic TurnInfo messages exercise the inherited dispatcher without opening a
socket. StartOfTurn must retain the fork's early interruption input, while Update
uses upstream's language/result-rich frame instead of duplicating that partial.
"""

from unittest.mock import AsyncMock, call

import pytest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.aayiqagent.flux.stt import AayiqAgentFluxSTTService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.transcriptions.language import Language


@pytest.fixture(params=["direct", "gateway"])
def service_factory(request):
    """Build either transport without setup; both must use the real shared dispatcher."""

    def create(model="flux-general-en"):
        """Construct an unconnected service with an explicit model and synthetic credentials."""
        if request.param == "gateway":
            return AayiqAgentFluxSTTService(
                gateway_base_url="https://gateway.example",
                invocation_token="test-invocation",
                request_id="test-request",
                model=model,
            )
        return DeepgramFluxSTTService(
            api_key="test-key", settings=DeepgramFluxSTTService.Settings(model=model)
        )

    return create


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,languages,expected_language",
    [("flux-general-en", [], Language.EN), ("flux-general-multi", ["hi"], Language.HI)],
)
async def test_start_and_updates_emit_partials_before_the_single_final(
    service_factory, model, languages, expected_language
):
    """A start, update and final produce two distinct partials and one final on either transport."""
    service = service_factory(model)
    frames = []

    async def push(frame, *args, **kwargs):
        """Capture service output before pipeline routing without changing the frame."""
        frames.append(frame)

    async def broadcast(frame_type, **kwargs):
        """Retain provider turn markers in the same ordered stream as transcript frames."""
        frames.append(frame_type(**kwargs))

    service.push_frame = push
    service.broadcast_frame = broadcast
    service._call_event_handler = AsyncMock()
    service._handle_transcription = AsyncMock()
    service.emit_stt_usage_metrics = AsyncMock()

    update = {
        "type": "TurnInfo",
        "event": "Update",
        "transcript": "Hello there",
        "languages": languages,
    }
    final = {**update, "event": "EndOfTurn", "transcript": "Hello there.", "words": []}
    await service._handle_message(
        {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "Hello"}
    )
    await service._handle_message(update)
    await service._handle_message(final)

    assert [type(frame) for frame in frames] == [
        ProposedUserStartedSpeakingFrame,
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert [frame.text for frame in frames if hasattr(frame, "text")] == [
        "Hello",
        "Hello there",
        "Hello there.",
    ]
    assert frames[2].language == expected_language
    assert frames[2].result == update
    assert frames[-2].language == expected_language
    assert frames[-2].result == final
    assert frames[-2].finalized
    service._call_event_handler.assert_any_await("on_update", "Hello there")
    assert service._call_event_handler.await_args_list.count(call("on_update", "Hello there")) == 1


@pytest.mark.asyncio
async def test_empty_updates_do_not_create_transcription_frames(service_factory):
    """Empty onset/update messages retain the start marker without inventing text or update events."""
    service = service_factory()
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    for event in ("StartOfTurn", "Update"):
        await service._handle_message({"type": "TurnInfo", "event": event, "transcript": ""})
    service.push_frame.assert_not_awaited()
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStartedSpeakingFrame)
    service._call_event_handler.assert_awaited_once_with("on_start_of_turn", "")

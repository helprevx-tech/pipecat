# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Transition acknowledgements retain caller context without starting inference."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import NodeTransitionStartedFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_text", ["", "Please transfer me"])
async def test_transition_cancels_deferred_flush_before_acknowledging(pending_text):
    """Retire the actual delayed flush and commit at most once, even on duplicate frames."""
    context = LLMContext()
    aggregator = LLMUserAggregator(context, _realtime_service_mode=True)
    await aggregator.setup(frame_processor_setup(TaskManager()))
    aggregator.push_frame = AsyncMock()
    aggregator._ttfs_p99_latency = 60
    acknowledged = asyncio.Event()
    committed = []

    @aggregator.event_handler("on_user_turn_message_added")
    async def on_message_added(_, message):
        committed.append(message.content)

    try:
        if pending_text:
            await aggregator._handle_transcription(TranscriptionFrame(pending_text, "", "now"))
        await aggregator._realtime_handoff_flush()
        delayed = aggregator._realtime_handoff_flush_task
        await asyncio.sleep(0)
        frame = NodeTransitionStartedFrame([], acknowledged)
        await aggregator._handle_node_transition_started(frame)
        await aggregator._handle_node_transition_started(frame)
        await asyncio.sleep(0)

        assert acknowledged.is_set()
        assert delayed.done()
        assert aggregator._realtime_handoff_flush_task is None
        assert committed == ([pending_text] if pending_text else [])
        assert context.messages == (
            [{"role": "user", "content": pending_text}] if pending_text else []
        )
        aggregator.push_frame.assert_not_awaited()
    finally:
        await aggregator.cleanup()


@pytest.mark.asyncio
async def test_cancelled_transition_does_not_acknowledge_or_consume_pending_text():
    """Cancellation while retiring the timer propagates without a false acknowledgement."""
    context = LLMContext()
    aggregator = LLMUserAggregator(context, _realtime_service_mode=True)
    aggregator.push_frame = AsyncMock()
    await aggregator._handle_transcription(TranscriptionFrame("Keep this answer", "", "now"))
    aggregator._cancel_realtime_handoff_flush_task = AsyncMock(side_effect=asyncio.CancelledError)
    acknowledged = asyncio.Event()

    with pytest.raises(asyncio.CancelledError):
        await aggregator._handle_node_transition_started(
            NodeTransitionStartedFrame([], acknowledged)
        )

    assert not acknowledged.is_set()
    assert context.messages == []
    assert aggregator.aggregation_string() == "Keep this answer"
    aggregator.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_transition_without_ack_request_leaves_cascade_turn_unchanged():
    """A service that did not request aggregation retains its normal turn-stop owner."""
    aggregator = LLMUserAggregator(LLMContext(), _realtime_service_mode=True)
    aggregator.push_frame = AsyncMock()
    await aggregator._handle_transcription(TranscriptionFrame("Still talking", "", "now"))
    await aggregator._handle_node_transition_started(NodeTransitionStartedFrame([]))
    assert aggregator.aggregation_string() == "Still talking"
    aggregator.push_frame.assert_not_awaited()

# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Public fork context corrections remain compatible with upstream context commits."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from pipecat.frames.frames import NodeTransitionStartedFrame, TextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
)
from pipecat.utils.string import TextPartForConcatenation


@pytest.mark.asyncio
async def test_assistant_correction_applies_once_to_silent_and_turn_end_commits():
    context = LLMContext()
    correct = Mock(side_effect=lambda value: value.upper())
    aggregator = LLMAssistantAggregator(
        context, params=LLMAssistantAggregatorParams(correct_aggregation_callback=correct)
    )
    aggregator.push_frame = AsyncMock()

    await aggregator._handle_text(TextFrame("before file"))
    assert await aggregator._add_aggregation_to_context() == "BEFORE FILE"
    aggregator.push_frame.assert_not_awaited()
    await aggregator._handle_text(TextFrame("after file"))
    assert await aggregator.push_aggregation() == "AFTER FILE"
    assert await aggregator.push_aggregation() == ""

    assert context.messages == [
        {"role": "assistant", "content": "BEFORE FILE"},
        {"role": "assistant", "content": "AFTER FILE"},
    ]
    assert [call.args for call in correct.call_args_list] == [("before file",), ("after file",)]
    assert aggregator.push_frame.await_count == 2


@pytest.mark.asyncio
async def test_node_transition_commits_user_context_before_ack_without_inference():
    context = LLMContext()
    aggregator = LLMUserAggregator(context)
    aggregator._aggregation.append(
        TextPartForConcatenation("pending transcript", includes_inter_part_spaces=False)
    )
    aggregator.push_frame = AsyncMock()
    acknowledged = asyncio.Event()
    await aggregator._handle_node_transition_started(NodeTransitionStartedFrame([], acknowledged))
    assert acknowledged.is_set()
    assert context.messages == [{"role": "user", "content": "pending transcript"}]
    aggregator.push_frame.assert_not_awaited()

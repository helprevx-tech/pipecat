# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Preserve public context, mute-callback and required transition-ACK contracts.

These offline regressions use native aggregators, callback strategy and LLM tool
execution. Transport frame delivery is captured locally and provider inference is
never invoked. Async fixtures own service/task cleanup; timeout and cancellation
cases must terminate before tools are announced, prepared or dispatched.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    NodeTransitionStartedFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.user_mute.callback_user_mute_strategy import CallbackUserMuteStrategy
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.string import TextPartForConcatenation
from tests.frame_processor_helpers import frame_processor_setup


@pytest.mark.asyncio
async def test_assistant_correction_applies_once_to_silent_and_turn_end_commits():
    """Correct each silent or turn-end commit once without emitting duplicate context."""
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
    """Acknowledge only after pending user text is committed, without new inference."""
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


def test_public_mute_callback_alias_remains_importable():
    """Existing integrations can retain their exported callback type annotation."""
    from pipecat.turns.user_mute.callback_user_mute_strategy import ShouldMuteCallback

    predicate: ShouldMuteCallback = lambda frame: True
    assert predicate(Frame()) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("muted", [False, True])
@pytest.mark.parametrize("callback_kind", ["sync", "async", "awaitable"])
async def test_mute_callbacks_keep_sync_async_and_awaitable_contracts(
    monkeypatch, muted, callback_kind
):
    """Evaluate once after the base hook, returning a Boolean for each callback kind."""
    frame = Frame()
    observed = []

    async def base_hook(strategy, received):
        """Record the retained base lifecycle call before the application predicate."""
        observed.append(("base", received))
        return False

    def sync_callback(received):
        """Return a caller-owned immediate decision for the exact incoming frame."""
        observed.append(("callback", received))
        return muted

    async def async_callback(received):
        """Return a caller-owned coroutine decision without starting external work."""
        return sync_callback(received)

    def awaitable_callback(received):
        """Return a settled Future to cover awaitables beyond coroutine functions."""
        result = asyncio.get_running_loop().create_future()
        result.set_result(sync_callback(received))
        return result

    monkeypatch.setattr(BaseUserMuteStrategy, "process_frame", base_hook)
    callbacks = {
        "sync": sync_callback,
        "async": async_callback,
        "awaitable": awaitable_callback,
    }
    strategy = CallbackUserMuteStrategy(should_mute_callback=callbacks[callback_kind])
    await strategy.setup(frame_processor_setup(TaskManager()))
    try:
        result = await strategy.process_frame(frame)
        assert result is muted
        assert observed == [("base", frame), ("callback", frame)]
    finally:
        await strategy.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_kind", ["sync", "async"])
async def test_mute_callback_errors_reach_the_caller(callback_kind):
    """A predicate error must not turn into a fabricated mute or unmute decision."""
    error = RuntimeError("predicate unavailable")

    def sync_callback(frame):
        """Raise the caller's exact synchronous error without owning resources."""
        raise error

    async def async_callback(frame):
        """Raise the caller's exact asynchronous error without owning resources."""
        raise error

    callback = sync_callback if callback_kind == "sync" else async_callback
    strategy = CallbackUserMuteStrategy(should_mute_callback=callback)
    with pytest.raises(RuntimeError) as raised:
        await strategy.process_frame(Frame())
    assert raised.value is error


@pytest.mark.asyncio
async def test_mute_callback_cancellation_propagates():
    """Cancelling caller-owned callback work cannot manufacture a mute decision."""

    async def cancelled_callback(frame):
        """Represent an externally cancelled application decision."""
        raise asyncio.CancelledError

    strategy = CallbackUserMuteStrategy(should_mute_callback=cancelled_callback)
    with pytest.raises(asyncio.CancelledError):
        await strategy.process_frame(Frame())


class _ContextAckService(LLMService):
    """Drive native tool dispatch with an explicit synthetic ACK requirement.

    No inference or provider session exists. The short wait bounds only these
    offline tests; production defaults and the native tool dispatcher stay intact.
    """

    def __init__(self):
        """Initialize native scheduling/settings and a per-test aggregation requirement."""
        super().__init__(settings=LLMSettings(model="offline-context-ack"))
        self.requires_ack = True
        self._NODE_TRANSITION_CONTEXT_AGGREGATION_TIMEOUT_SECONDS = 0.02

    def _requires_node_transition_context_aggregation(self):
        """Expose this test's explicit transition-aggregation policy to the native gate."""
        return self.requires_ack


@pytest_asyncio.fixture
async def native_transition_service():
    """Own a real LLM service/task manager with captured transport and handler effects.

    Actual registration, ACK waiting, prepared/start events and native dispatch
    are retained. Captured broadcast frames replace only transport delivery; an
    opt-in native user aggregator supplies a genuine context acknowledgement.
    Cleanup joins every task created by this service, including failed cases.
    """
    service = _ContextAckService()
    manager = TaskManager()
    await service.setup(
        frame_processor_setup(
            manager,
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        )
    )
    state = SimpleNamespace(
        service=service,
        frames=[],
        handled=[],
        aggregator=None,
        waiting=asyncio.Event(),
        settled=asyncio.Event(),
    )
    # Record announcements without spawning unrelated application event work.
    service._call_event_handler = AsyncMock()

    async def capture_broadcast(frame_class, **kwargs):
        """Capture native lifecycle frames and optionally deliver the transition to its aggregator."""
        frame = frame_class(**kwargs)
        state.frames.append(frame)
        if isinstance(frame, NodeTransitionStartedFrame):
            state.waiting.set()
            if state.aggregator is not None:
                await state.aggregator._handle_node_transition_started(frame)
        if isinstance(frame, FunctionCallResultFrame):
            state.settled.set()

    async def handler(params):
        """Record one actual dispatch and settle it through its native result callback."""
        state.handled.append(params.function_name)
        await params.result_callback({"status": "done"})

    service.broadcast_frame = capture_broadcast
    service.register_function("transition", handler, is_node_transition=True)
    service.register_function("ordinary", handler)
    try:
        yield state
    finally:
        await service.cleanup()


def _function_call(name):
    """Create one synthetic native model call with its own shared-context object."""
    return FunctionCallFromLLM(
        function_name=name,
        tool_call_id=f"call-{name}",
        arguments={},
        context=LLMContext(),
    )


@pytest.mark.asyncio
async def test_missing_required_context_ack_never_announces_prepares_or_dispatches(
    native_transition_service,
):
    """A missing required ACK fails the whole batch before either sibling tool can start."""
    state = native_transition_service
    with pytest.raises(TimeoutError):
        await state.service.run_function_calls(
            [_function_call("ordinary"), _function_call("transition")]
        )
    assert len(state.frames) == 1
    assert isinstance(state.frames[0], NodeTransitionStartedFrame)
    assert not state.frames[0].context_aggregation_event.is_set()
    state.service._call_event_handler.assert_not_awaited()
    assert not state.service._function_call_tasks
    assert state.handled == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requires_ack", "name", "acknowledge"),
    [(True, "transition", True), (False, "transition", False), (True, "ordinary", False)],
    ids=["required-ack", "optional-transition", "ordinary-tool"],
)
async def test_context_ack_gate_preserves_valid_and_nonrequired_dispatch(
    native_transition_service, requires_ack, name, acknowledge
):
    """A real user-context ACK permits dispatch; nonrequired paths do not gain a wait."""
    state = native_transition_service
    state.service.requires_ack = requires_ack
    if acknowledge:
        context = LLMContext()
        state.aggregator = LLMUserAggregator(context)
        state.aggregator._aggregation.append(
            TextPartForConcatenation("pending caller answer", includes_inter_part_spaces=False)
        )
        state.aggregator.push_frame = AsyncMock()
    await state.service.run_function_calls([_function_call(name)])
    await asyncio.wait_for(state.settled.wait(), 1)
    assert state.handled == [name]
    assert any(isinstance(frame, FunctionCallsStartedFrame) for frame in state.frames)
    assert [call.args[0] for call in state.service._call_event_handler.await_args_list] == [
        "on_function_calls_started",
        "on_function_calls_prepared",
    ]
    transitions = [frame for frame in state.frames if isinstance(frame, NodeTransitionStartedFrame)]
    if acknowledge:
        assert transitions[0].context_aggregation_event.is_set()
        assert context.messages == [{"role": "user", "content": "pending caller answer"}]
        state.aggregator.push_frame.assert_not_awaited()
    elif name == "transition":
        assert transitions[0].context_aggregation_event is None
    else:
        assert transitions == []


@pytest.mark.asyncio
async def test_cancellation_during_required_context_ack_does_not_dispatch(
    native_transition_service,
):
    """Cancellation propagates through the actual ACK wait without accepting tool ownership."""
    state = native_transition_service
    state.service._NODE_TRANSITION_CONTEXT_AGGREGATION_TIMEOUT_SECONDS = 60
    pending = asyncio.create_task(state.service.run_function_calls([_function_call("transition")]))
    try:
        await asyncio.wait_for(state.waiting.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        state.service._call_event_handler.assert_not_awaited()
        assert not state.service._function_call_tasks
        assert state.handled == []
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

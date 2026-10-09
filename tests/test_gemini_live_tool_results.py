#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallResultProperties
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _TestGeminiLiveLLMService(GeminiLiveLLMService):
    def create_client(self):
        self._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=None)))


class _FakeSession:
    def __init__(self):
        self.send_tool_response = AsyncMock()
        self.send_realtime_input = AsyncMock()


@pytest.mark.asyncio
async def test_gemini_3_tool_result_sends_only_tool_response():
    service = _TestGeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model="gemini-3.1-flash-live-preview"),
    )
    session = _FakeSession()
    service._session = session

    delivered = await service._tool_result(
        "call-123",
        "get_account_balances",
        {"status": "success", "balances": []},
    )

    assert delivered is True
    session.send_tool_response.assert_awaited_once()
    response = session.send_tool_response.await_args.kwargs["function_responses"]
    assert response.id == "call-123"
    assert response.name == "get_account_balances"
    assert response.response == {"status": "success", "balances": []}
    session.send_realtime_input.assert_not_awaited()


def _async_tool_service():
    service = _TestGeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model="gemini-3.8-live"),
    )
    service._function_is_async = lambda _name: True
    return service


@pytest.mark.asyncio
async def test_reconnect_preserves_intermediate_order_and_silent_final_result():
    service = _async_tool_service()

    for progress, is_final in [(1, False), (2, False), (3, True)]:
        assert not await service._tool_result(
            "call-123", "lookup", {"progress": progress}, is_final=is_final, spoken=False
        )
    assert "call-123" not in service._completed_tool_calls

    service._session = session = _FakeSession()
    await service._drain_pending_tool_results()

    responses = [
        call.kwargs["function_responses"] for call in session.send_tool_response.await_args_list
    ]
    assert [response.response["progress"] for response in responses] == [1, 2, 3]
    assert [response.will_continue for response in responses] == [True, True, False]
    assert all(response.response["scheduling"] == "SILENT" for response in responses)
    assert all(response.scheduling.value == "SILENT" for response in responses)
    assert "call-123" in service._completed_tool_calls
    assert not service._pending_tool_results


@pytest.mark.asyncio
async def test_intermediate_delivery_does_not_complete_the_call():
    service = _async_tool_service()
    service._session = session = _FakeSession()

    assert await service._tool_result(
        "call-123", "lookup", {"progress": 1}, is_final=False, spoken=True
    )

    assert "call-123" not in service._completed_tool_calls
    response = session.send_tool_response.await_args.kwargs["function_responses"]
    assert response.will_continue is True
    assert response.scheduling.value == "WHEN_IDLE"


@pytest.mark.asyncio
async def test_failed_result_is_retried_without_losing_scheduling():
    service = _async_tool_service()
    service._session = session = _FakeSession()
    session.send_tool_response.side_effect = [ConnectionError("disconnected"), None]
    service._handle_send_error = AsyncMock()

    assert not await service._tool_result("call-123", "lookup", {"done": True}, spoken=False)
    assert "call-123" not in service._completed_tool_calls
    service._handle_send_error.assert_awaited_once()

    await service._drain_pending_tool_results()

    responses = [
        call.kwargs["function_responses"] for call in session.send_tool_response.await_args_list
    ]
    assert responses[0] is responses[1]
    assert responses[1].will_continue is False
    assert responses[1].scheduling.value == "SILENT"
    assert "call-123" in service._completed_tool_calls
    assert not service._pending_tool_results


@pytest.mark.asyncio
async def test_context_replay_does_not_duplicate_or_change_a_queued_result_frame():
    service = _async_tool_service()
    service._tool_call_id_to_name["call-123"] = "lookup"
    await service._handle_function_call_result(
        FunctionCallResultFrame(
            function_name="lookup",
            tool_call_id="call-123",
            arguments={},
            result={"done": True},
            properties=FunctionCallResultProperties(run_llm=False),
        )
    )
    assert "call-123" not in service._completed_tool_calls
    service._context = LLMContext(
        messages=[{"role": "tool", "tool_call_id": "call-123", "content": '{"done":true}'}]
    )
    await service._process_completed_function_calls(send_new_results=True)

    service._session = session = _FakeSession()
    await service._drain_pending_tool_results()

    session.send_tool_response.assert_awaited_once()
    response = session.send_tool_response.await_args.kwargs["function_responses"]
    assert response.scheduling.value == "SILENT"
    assert "call-123" in service._completed_tool_calls


@pytest.mark.asyncio
async def test_concurrent_drains_send_each_result_once():
    service = _async_tool_service()
    await service._tool_result("call-123", "lookup", {"done": True})
    service._session = session = _FakeSession()

    async def send(**_kwargs):
        await asyncio.sleep(0)

    session.send_tool_response.side_effect = send
    await asyncio.gather(
        service._drain_pending_tool_results(), service._drain_pending_tool_results()
    )

    session.send_tool_response.assert_awaited_once()
    assert not service._pending_tool_results

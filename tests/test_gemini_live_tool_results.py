#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mock transport client lifecycle events."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.mock_transport import MockTransport


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_frame_type", [EndFrame, CancelFrame])
async def test_local_shutdown_does_not_emit_client_disconnected(terminal_frame_type):
    """A locally generated terminal frame is not a remote peer disconnect."""
    transport = MockTransport()
    on_client_disconnected = AsyncMock()
    transport.add_event_handler("on_client_disconnected", on_client_disconnected)

    input_transport = transport.input()
    input_transport.push_frame = AsyncMock()
    await input_transport.process_frame(terminal_frame_type(), FrameDirection.DOWNSTREAM)

    on_client_disconnected.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_client_emits_client_disconnected():
    """Tests can explicitly model a remote peer disconnect."""
    transport = MockTransport()
    on_client_disconnected = AsyncMock()
    transport.add_event_handler("on_client_disconnected", on_client_disconnected)

    await transport.disconnect_client()
    await transport.cleanup()

    on_client_disconnected.assert_awaited_once_with(transport, None)

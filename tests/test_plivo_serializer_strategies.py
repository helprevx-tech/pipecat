from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import EndFrame
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.utils.enums import EndTaskReason


def _serializer(transfer_strategy, hangup_strategy):
    return PlivoFrameSerializer(
        stream_id="stream-1",
        call_id="call-1",
        auth_id="MA123",
        auth_token="secret",
        transfer_strategy=transfer_strategy,
        hangup_strategy=hangup_strategy,
    )


@pytest.mark.asyncio
async def test_transfer_end_frame_invokes_only_transfer_strategy():
    transfer_strategy = AsyncMock()
    transfer_strategy.execute_transfer.return_value = True
    hangup_strategy = AsyncMock()

    await _serializer(transfer_strategy, hangup_strategy).serialize(
        EndFrame(reason=EndTaskReason.TRANSFER_CALL.value)
    )

    transfer_strategy.execute_transfer.assert_awaited_once_with(
        {"call_id": "call-1", "auth_id": "MA123", "auth_token": "secret"}
    )
    hangup_strategy.execute_hangup.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_end_frame_invokes_only_hangup_strategy():
    transfer_strategy = AsyncMock()
    hangup_strategy = AsyncMock()
    hangup_strategy.execute_hangup.return_value = True

    await _serializer(transfer_strategy, hangup_strategy).serialize(EndFrame())

    hangup_strategy.execute_hangup.assert_awaited_once_with(
        {"call_id": "call-1", "auth_id": "MA123", "auth_token": "secret"}
    )
    transfer_strategy.execute_transfer.assert_not_awaited()

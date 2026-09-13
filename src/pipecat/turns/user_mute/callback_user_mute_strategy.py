#
# Copyright (c) 2026, HelpRevX
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that delegates the decision to a caller-supplied callback."""

import inspect
from typing import Awaitable, Callable, Union

from pipecat.frames.frames import Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy

ShouldMuteCallback = Callable[[Frame], Union[bool, Awaitable[bool]]]


class CallbackUserMuteStrategy(BaseUserMuteStrategy):
    """Mute the user whenever ``should_mute_callback(frame)`` says so.

    The other strategies in this package encode one fixed heuristic each. This
    one hands the per-frame decision to the application, which is what a
    workflow engine needs when the answer depends on its own state — e.g.
    "the bot is speaking AND the current node forbids interruption", or "the
    pipeline is shutting down". The callback receives every frame the strategy
    sees, so it can also use them to track that state.

    Args:
        should_mute_callback: Called with each frame; returns ``True`` to mute
            the user for that frame. May be sync or async.
        **kwargs: Passed through to ``BaseUserMuteStrategy``.
    """

    def __init__(self, *, should_mute_callback: ShouldMuteCallback, **kwargs):
        super().__init__(**kwargs)
        self._should_mute_callback = should_mute_callback

    async def process_frame(self, frame: Frame) -> bool:
        """Return whether the user is muted for this frame, per the callback."""
        result = self._should_mute_callback(frame)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, HelpRevX
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that delegates mute decisions to an external callback."""

import inspect
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


ShouldMuteCallback = Callable[[Frame], bool | Awaitable[bool]]


class CallbackUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that uses an external callback to determine mute state.

    This strategy delegates the mute decision to a callback function provided
    at construction time. The callback is invoked on each frame and should
    return True if the user should be muted, False otherwise.

    This is useful when the mute logic depends on external state (e.g., the
    current workflow node, bot speaking state, function call execution, etc.)
    that is managed outside the strategy itself.

    Example:
        async def should_mute(frame: Frame) -> bool:
            # Return True to mute, False to unmute
            return engine.is_bot_speaking()

        strategy = CallbackUserMuteStrategy(should_mute_callback=should_mute)

    """

    def __init__(
        self,
        should_mute_callback: ShouldMuteCallback,
        **kwargs,
    ):
        """Initialize the callback user mute strategy.

        Args:
            should_mute_callback: A synchronous or awaitable callback that takes a Frame
                and returns True if the user should be muted, False otherwise.
            **kwargs: Additional arguments passed to the base strategy.
        """
        super().__init__(**kwargs)
        self._should_mute_callback = should_mute_callback

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted (True = muted, False = not muted).
        """
        await super().process_frame(frame)
        result = self._should_mute_callback(frame)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

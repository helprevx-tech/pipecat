#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The frame queue reads a frame's ``interruptible`` flag as it is."""

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.utils.frame_queue import FramePriorityQueue, FrameQueue


class TestFrameQueueInterruptibility(unittest.IsolatedAsyncioTestCase):
    def test_a_flag_set_on_a_queued_frame_is_respected(self):
        queue = FrameQueue()
        frame = TextFrame(text="hi")
        queue.put_nowait(frame)
        self.assertFalse(queue.has_uninterruptible)

        frame.interruptible = False
        self.assertTrue(queue.has_uninterruptible)
        queue.get_nowait()
        self.assertFalse(queue.has_uninterruptible)

        end = EndFrame()
        queue.put_nowait(end)
        self.assertTrue(queue.has_uninterruptible)
        end.interruptible = True
        self.assertFalse(queue.has_uninterruptible)

    def test_reset_keeps_what_is_uninterruptible_now(self):
        queue = FrameQueue()
        plain, end = TextFrame(text="hi"), EndFrame()
        queue.put_nowait(plain)
        queue.put_nowait(end)
        plain.interruptible = False
        end.interruptible = True

        queue.reset()

        self.assertEqual(queue.qsize(), 1)
        self.assertIs(queue.get_nowait(), plain)
        self.assertFalse(queue.has_uninterruptible)

    def test_has_frame_and_tuple_items(self):
        queue = FrameQueue(frame_getter=lambda item: item[0])
        queue.put_nowait((TextFrame(text="hi"), "down"))
        queue.put_nowait((EndFrame(), "down"))
        self.assertTrue(queue.has_frame(TextFrame))
        self.assertTrue(queue.has_uninterruptible)
        queue.reset()
        self.assertFalse(queue.has_frame(TextFrame))
        self.assertTrue(queue.has_frame(EndFrame))
        self.assertEqual(queue.get_nowait()[1], "down")


class TestFramePriorityQueue(unittest.IsolatedAsyncioTestCase):
    async def test_reset_preserves_priority_order_and_join(self):
        queue = FramePriorityQueue(frame_getter=lambda item: item[2])
        frames = [TextFrame(text=str(i)) for i in range(4)]
        for frame in frames[:3]:
            frame.interruptible = False
        for priority, sequence, frame in [
            (2, 0, frames[0]),
            (1, 2, frames[2]),
            (1, 1, frames[1]),
            (0, 3, frames[3]),
        ]:
            queue.put_nowait((priority, sequence, frame))

        queue.reset()

        for frame in [frames[1], frames[2], frames[0]]:
            self.assertTrue(queue.has_uninterruptible)
            self.assertIs(queue.get_nowait()[2], frame)
            queue.task_done()
        self.assertFalse(queue.has_uninterruptible)
        await asyncio.wait_for(queue.join(), timeout=1)


class TestFrameQueueRetention(unittest.IsolatedAsyncioTestCase):
    """The optional authority predicate only narrows protected-frame retention."""

    async def test_reset_discards_revoked_protected_frames_and_preserves_join(self):
        """Both real queue types retain current protected work and acknowledge every removal."""
        for queue_type in (FrameQueue, FramePriorityQueue):
            with self.subTest(queue_type=queue_type.__name__):
                queue = queue_type(frame_getter=lambda item: item[2])
                stale, current, normal, end = TextFrame("stale"), TextFrame("current"), TextFrame("normal"), EndFrame()
                stale.interruptible = current.interruptible = False
                for index, frame in enumerate((stale, current, normal, end)):
                    queue.put_nowait((index, index, frame))

                queue.reset(keep_frame=lambda frame: frame is not stale)

                self.assertEqual(queue.qsize(), 2)
                for expected in (current, end):
                    self.assertIs(queue.get_nowait()[2], expected)
                    queue.task_done()
                await asyncio.wait_for(queue.join(), timeout=0.25)
                self.assertEqual(queue._unfinished_tasks, 0)

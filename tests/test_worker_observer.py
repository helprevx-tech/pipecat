#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how WorkerObserver hands pushes to the observers it manages."""

import asyncio
import gc
import unittest

from pipecat.frames.frames import TextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker_observer import ObserverProcessingError, WorkerObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager


class RecordingObserver(BaseObserver):
    """Records every push of a text frame it is told about."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushes: list[tuple[str, bool]] = []

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            self.pushes.append((data.source.name, data.first_push))


class TestWorkerObserverInAPipeline(unittest.IsolatedAsyncioTestCase):
    """A frame is pushed again by every processor that passes it along."""

    async def _run(self, *observers: BaseObserver):
        pipeline = Pipeline(
            [
                IdentityFilter(name="first"),
                IdentityFilter(name="second"),
                IdentityFilter(name="third"),
            ]
        )
        await run_test(
            pipeline,
            frames_to_send=[TextFrame("hello"), SleepFrame(sleep=0.1)],
            expected_down_frames=[TextFrame],
            observers=list(observers),
        )

    async def test_an_observer_that_handles_a_frame_once_is_told_once(self):
        observer = RecordingObserver(observe_every_push=False)
        await self._run(observer)
        self.assertEqual(len(observer.pushes), 1)
        self.assertTrue(observer.pushes[0][1])

    async def test_an_observer_gets_every_hop_by_default(self):
        observer = RecordingObserver()
        await self._run(observer)
        sources = [source for source, _ in observer.pushes]
        for name in ("first", "second", "third"):
            self.assertIn(name, sources)
        # Only the first push is marked as such.
        firsts = [first for _, first in observer.pushes]
        self.assertEqual(firsts, [True] + [False] * (len(firsts) - 1))


class TestWorkerObserverMemory(unittest.IsolatedAsyncioTestCase):
    async def test_first_observer_can_be_added_after_setup(self):
        worker_observer = WorkerObserver()
        await worker_observer.setup(TaskManager())
        self.addAsyncCleanup(worker_observer.cleanup)
        observer = RecordingObserver()
        worker_observer.add_observer(observer)
        source = IdentityFilter(name="source")

        await worker_observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=TextFrame("hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        await asyncio.gather(*(proxy.queue.join() for proxy in worker_observer._proxies.values()))
        self.assertEqual(observer.pushes, [("source", True)])

    async def test_a_frame_is_forgotten_once_the_pipeline_lets_go_of_it(self):
        observer = RecordingObserver()
        worker_observer = WorkerObserver(observers=[observer])
        await worker_observer.setup(TaskManager())
        source = IdentityFilter(name="source")

        async def push(frame):
            await worker_observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=source,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        for _ in range(10):
            await push(TextFrame("hello"))
        self.assertEqual(len(worker_observer._frames_pushed), 10)

        # The proxy holds on to the last push it handled, so end with a frame
        # the test keeps and wait for it to be handed over.
        kept = TextFrame("kept")
        await push(kept)
        await asyncio.gather(*(proxy.queue.join() for proxy in worker_observer._proxies.values()))
        gc.collect()

        self.assertEqual(list(worker_observer._frames_pushed.values()), [kept])
        self.assertEqual(len(observer.pushes), 11)

        await worker_observer.cleanup()


class TestObserverArtifactBarrier(unittest.IsolatedAsyncioTestCase):
    async def test_idle_wait_includes_async_event_handlers(self):
        finished = asyncio.Event()
        release = asyncio.Event()

        class EventObserver(BaseObserver):
            def __init__(self):
                super().__init__()
                self._register_event_handler("on_text")

            async def on_push_frame(self, data):
                await self._call_event_handler("on_text")

        observer = EventObserver()

        @observer.event_handler("on_text")
        async def persist(_observer):
            await release.wait()
            finished.set()

        worker = WorkerObserver(observers=[observer])
        await worker.setup(TaskManager())
        source = IdentityFilter()
        await worker.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=TextFrame("hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        barrier = asyncio.create_task(worker.wait_until_idle())
        await asyncio.sleep(0)
        self.assertFalse(barrier.done())
        release.set()
        await asyncio.wait_for(barrier, 1)
        self.assertTrue(finished.is_set())
        await worker.cleanup()


class CleanupRecordingObserver(BaseObserver):
    """Records its cleanup and fires an event for every text frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cleaned_up = False
        self._register_event_handler("on_text")

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            await self._call_event_handler("on_text")

    async def cleanup(self):
        await super().cleanup()
        self.cleaned_up = True


class TestWorkerObserverRemoval(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.observer = CleanupRecordingObserver()
        self.worker_observer = WorkerObserver(observers=[self.observer])
        await self.worker_observer.setup(TaskManager())
        self.addAsyncCleanup(self.worker_observer.cleanup)

    async def _push_text(self):
        source = IdentityFilter()
        await self.worker_observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=TextFrame("hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def test_a_removed_observer_is_cleaned_up(self):
        await self.worker_observer.remove_observer(self.observer)
        self.assertTrue(self.observer.cleaned_up)

    async def test_an_observer_can_be_removed_from_its_own_event_handler(self):
        removed = asyncio.Event()

        @self.observer.event_handler("on_text")
        async def on_text(observer):
            await self.worker_observer.remove_observer(observer)
            removed.set()

        await self._push_text()
        await asyncio.wait_for(removed.wait(), timeout=1.0)
        self.assertTrue(self.observer.cleaned_up)


class FailingCleanupObserver(BaseObserver):
    async def cleanup(self):
        await super().cleanup()
        raise RuntimeError("cleanup failed")


class TestWorkerObserverShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_an_observer_failing_cleanup_does_not_stop_the_others(self):
        later = CleanupRecordingObserver()
        worker_observer = WorkerObserver(observers=[FailingCleanupObserver(), later])
        await worker_observer.setup(TaskManager())

        await worker_observer.cleanup()

        self.assertTrue(later.cleaned_up)


if __name__ == "__main__":
    unittest.main()


class TestObserverFailureRecovery(unittest.IsolatedAsyncioTestCase):
    def _event(self, text):
        source = IdentityFilter(name="source")
        return FramePushed(
            source=source,
            destination=source,
            frame=TextFrame(text),
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )

    async def test_failed_callback_does_not_strand_later_frames_or_certify_artifacts(self):
        class Flaky(RecordingObserver):
            async def on_push_frame(self, data):
                if data.frame.text == "bad":
                    raise ValueError("synthetic failure")
                await super().on_push_frame(data)

        flaky, healthy = Flaky(), RecordingObserver()
        worker = WorkerObserver(observers=[flaky, healthy])
        await worker.setup(TaskManager())
        self.addAsyncCleanup(worker.cleanup)
        await worker.on_push_frame(self._event("bad"))
        await worker.on_push_frame(self._event("good"))
        with self.assertRaisesRegex(ObserverProcessingError, "processing failed"):
            await asyncio.wait_for(worker.wait_until_idle(), 1)
        self.assertEqual(len(flaky.pushes), 1)
        self.assertEqual(len(healthy.pushes), 2)
        self.assertTrue(all(not proxy.task.done() for proxy in worker._proxies.values()))

    async def test_late_observer_setup_finishes_before_first_frame_and_managed_task(self):
        class Late(RecordingObserver):
            async def setup(self, manager):
                await asyncio.sleep(0.01)
                await super().setup(manager)

            async def on_push_frame(self, data):
                await self.create_task(asyncio.sleep(0))
                await super().on_push_frame(data)

        worker = WorkerObserver()
        manager = TaskManager()
        await worker.setup(manager)
        self.addAsyncCleanup(worker.cleanup)
        observer = Late()
        worker.add_observer(observer)
        await worker.on_push_frame(self._event("immediate"))
        await asyncio.wait_for(worker.wait_until_idle(), 1)
        self.assertIs(observer.task_manager, manager)
        self.assertEqual(len(observer.pushes), 1)

    async def test_late_setup_failure_is_reported_without_hanging_the_barrier(self):
        class FailedSetup(RecordingObserver):
            async def setup(self, manager):
                raise RuntimeError("setup failed")

        worker = WorkerObserver()
        await worker.setup(TaskManager())
        self.addAsyncCleanup(worker.cleanup)
        observer = FailedSetup()
        worker.add_observer(observer)
        await worker.on_push_frame(self._event("queued"))
        with self.assertRaisesRegex(ObserverProcessingError, "setup failed"):
            await asyncio.wait_for(worker.wait_until_idle(), 1)
        self.assertEqual(observer.pushes, [])

    async def test_cancelled_processing_releases_backlog_and_reports_incomplete_drain(self):
        started = asyncio.Event()

        class Blocked(RecordingObserver):
            async def on_push_frame(self, data):
                started.set()
                await asyncio.Event().wait()

        worker = WorkerObserver(observers=[Blocked()])
        await worker.setup(TaskManager())
        await worker.on_push_frame(self._event("running"))
        await started.wait()
        await worker.on_push_frame(self._event("queued"))
        await worker.cleanup()
        # A late push must not enter an orphaned queue after cleanup.
        await worker.on_push_frame(self._event("after cleanup"))
        with self.assertRaisesRegex(ObserverProcessingError, "cancelled"):
            await asyncio.wait_for(worker.wait_until_idle(), 1)

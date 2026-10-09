#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
import weakref
from typing import Any

from attr import dataclass
from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import (
    BaseObserver,
    FrameProcessed,
    FramePushed,
    ProcessorSetUp,
    StartupWarmup,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
    """

    queue: asyncio.Queue
    observer: BaseObserver
    task: asyncio.Task | None = None
    error: str | None = None


class ObserverProcessingError(RuntimeError):
    """An observer could not process all frames required for an artifact snapshot."""


class _PipelineStartedSignal:
    """Internal sentinel queued to observers when the pipeline has started."""

    pass


class WorkerObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline worker.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.

    It also tells a frame's first push from the ones that follow, and only
    passes the first one to observers that want a frame once.
    """

    def __init__(
        self,
        *,
        observers: list[BaseObserver] | None = None,
        **kwargs,
    ):
        """Initialize the WorkerObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._proxies: dict[BaseObserver, Proxy] | None = (
            None  # Becomes a dict after setup() is called
        )
        # Frames pushed so far, held weakly: an entry goes away with its
        # frame, so this tracks the frames in flight, not every frame ever
        # pushed.
        self._frames_pushed: weakref.WeakValueDictionary[int, Frame] = weakref.WeakValueDictionary()

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)

        # After setup(), create the proxy immediately; setup() handles earlier additions.
        if self._proxies is not None:
            proxy = self._create_proxy(observer, setup_observer=True)
            self._proxies[observer] = proxy

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer and clean up its resources.

        Args:
            observer: The observer to remove.
        """
        # If the observer has a proxy, remove it.
        if self._proxies and observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            # Cancel the proxy worker right away.
            await self.cancel_task(proxy.task)

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)

    async def setup(self, task_manager: BaseTaskManager):
        """Set up a proxy for every managed observer.

        Processors report their own setup to observers, so the proxies are in
        place before any of them is set up.

        Args:
            task_manager: The task manager the proxies run their tasks on.
        """
        await super().setup(task_manager)
        self._proxies = self._create_proxies(self._observers)

        for observer in self._proxies:
            await observer.setup(task_manager)

    async def wait_until_idle(self) -> None:
        """Wait until every observer has processed its currently queued frames."""
        if not self._proxies:
            return
        proxies = list(self._proxies.values())
        await asyncio.gather(*(proxy.queue.join() for proxy in proxies))
        # Observer event handlers may outlive on_push_frame; finish them before
        # callers persist transcripts and other artifacts.
        await asyncio.gather(*(BaseObject.cleanup(proxy.observer) for proxy in proxies))
        failures = [proxy.error for proxy in proxies if proxy.error]
        if failures:
            raise ObserverProcessingError("; ".join(failures))

    async def cleanup(self):
        """Cleanup all proxy observers."""
        await super().cleanup()

        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self.cancel_task(proxy.task)

        for observer in self._proxies:
            await observer.cleanup()

    async def on_pipeline_started(self):
        """Forward pipeline started signal to all managed observers."""
        await self._send_to_proxy(_PipelineStartedSignal())

    async def on_process_frame(self, data: FrameProcessed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for the managed observers.

        A repeated push only reaches the observers that want every push.

        Args:
            data: The frame push event data to distribute to observers.
        """
        if not self._proxies:
            return

        frame = data.frame
        data.first_push = frame.id not in self._frames_pushed
        if data.first_push:
            self._frames_pushed[frame.id] = frame

        for proxy in self._proxies.values():
            if data.first_push or proxy.observer.observe_every_push:
                await self._queue_proxy(proxy, data)

    async def on_processor_setup(self, data: ProcessorSetUp):
        """Queue processor setup timing for all managed observers.

        Args:
            data: The processor setup event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    @deprecated(
        "`WorkerObserver.on_startup_warmup` is deprecated since 1.12.0 and will be removed in "
        "2.0.0. No replacement."
    )
    async def on_startup_warmup(self, data: StartupWarmup):
        """Queue deferred-import warming timing for all managed observers.

        .. deprecated:: 1.12.0
            No replacement. Nothing warms deferred imports at startup, so this
            is never called. Will be removed in 2.0.0.

        Args:
            data: The startup warmup event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    def _create_proxy(self, observer: BaseObserver, *, setup_observer: bool = False) -> Proxy:
        """Create a proxy, initializing late observers before delivering any frames."""
        proxy = Proxy(queue=asyncio.Queue(), observer=observer)
        proxy.task = self.create_task(self._proxy_task_handler(proxy, setup_observer))
        return proxy

    def _create_proxies(self, observers: list[BaseObserver]) -> dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any):
        if not self._proxies:
            return
        for proxy in self._proxies.values():
            await self._queue_proxy(proxy, data)

    async def _queue_proxy(self, proxy: Proxy, data: Any):
        if proxy.task is not None and proxy.task.done():
            self._record_proxy_error(proxy, "received events after its processing task stopped")
            return
        await proxy.queue.put(data)

    def _record_proxy_error(self, proxy: Proxy, description: str):
        # Retain one bounded diagnostic per observer; the drain barrier must not
        # certify artifacts after a callback failed or queued frames were lost.
        if proxy.error is None:
            proxy.error = f"{proxy.observer}: {description}"
            logger.warning(proxy.error)

    async def _proxy_task_handler(self, proxy: Proxy, setup_observer: bool = False):
        """Deliver queued events without letting one callback strand the drain barrier."""
        observer, queue = proxy.observer, proxy.queue
        setup_failed = False
        try:
            if setup_observer:
                try:
                    await observer.setup(self.task_manager)
                except Exception as exc:
                    self._record_proxy_error(proxy, f"setup failed ({type(exc).__name__})")
                    setup_failed = True
            while True:
                data = await queue.get()
                try:
                    if setup_failed:
                        continue
                    if isinstance(data, _PipelineStartedSignal):
                        await observer.on_pipeline_started()
                    elif isinstance(data, FramePushed):
                        await observer.on_push_frame(data)
                    elif isinstance(data, FrameProcessed):
                        await observer.on_process_frame(data)
                    elif isinstance(data, ProcessorSetUp):
                        await observer.on_processor_setup(data)
                    elif isinstance(data, StartupWarmup):
                        await observer.on_startup_warmup(data)
                except asyncio.CancelledError:
                    self._record_proxy_error(proxy, "processing cancelled before completion")
                    raise
                except Exception as exc:
                    self._record_proxy_error(proxy, f"processing failed ({type(exc).__name__})")
                finally:
                    queue.task_done()
        finally:
            # Cancellation may leave queued events with no consumer. Release
            # joiners and report the loss rather than leaving finalization stuck.
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()
                self._record_proxy_error(proxy, "stopped before queued events were processed")

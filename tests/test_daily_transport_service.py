#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Current Daily first-participant callback contract without a Daily room.

The historical skipped test used a removed module and synchronous event method.
These cases use the actual transport's async participant producer, registered
callback dispatcher and task drain. Only native Daily/CallClient construction is
inert; no SDK join, media device, credentials or account are used. A scheduled
callback is not complete until its actual event/cleanup boundary is observed.
"""

import asyncio
import unittest
from unittest.mock import patch

from pipecat.transports.daily.transport import DailyParams, DailyTransport


class TestDailyTransport(unittest.IsolatedAsyncioTestCase):
    """Own one synthetic transport and its callback tasks per isolated loop."""

    async def asyncSetUp(self):
        """Patch SDK-only constructors, retaining actual native event behavior."""
        self.daily_patch = patch("pipecat.transports.daily.transport.Daily")
        self.call_patch = patch("pipecat.transports.daily.transport.CallClient")
        self.daily = self.daily_patch.start()
        self.call_client = self.call_patch.start()
        self.addCleanup(self.daily_patch.stop)
        self.addCleanup(self.call_patch.stop)
        self.transport = DailyTransport(
            "https://synthetic.daily.test/unit", None, "unit-bot", params=DailyParams()
        )
        self.addAsyncCleanup(self.transport.cleanup)

    async def asyncTearDown(self):
        """No callback fixture may join a room or create live audio/video devices."""
        self.call_client.return_value.join.assert_not_called()
        self.daily.create_microphone_device.assert_not_called()
        self.daily.create_camera_device.assert_not_called()
        self.daily.create_speaker_device.assert_not_called()

    async def test_event_handler(self):
        """A synchronous user callback runs through the current async dispatcher."""
        called = asyncio.Event()
        observations = []

        @self.transport.event_handler("on_first_participant_joined")
        def event_handler(transport, participant):
            """Capture exact producer identity/data; signal actual callback execution."""
            observations.append((transport, participant))
            called.set()

        participant = {"id": "synthetic-first"}
        await self.transport._on_participant_joined(participant)
        await asyncio.wait_for(called.wait(), timeout=1)
        await self.transport.cleanup()
        self.assertEqual(observations, [(self.transport, participant)])
        self.assertFalse(self.transport._event_tasks)

    async def test_async_first_handler_once_and_all_participant_handlers(self):
        """First-only callback occurs once; every participant retains normal events."""
        first = []
        every = []
        clients = []

        @self.transport.event_handler("on_first_participant_joined")
        async def first_handler(transport, participant):
            """Yield once to exercise scheduled async rather than immediate callbacks."""
            await asyncio.sleep(0)
            first.append((transport, participant["id"]))

        @self.transport.event_handler("on_participant_joined")
        def participant_handler(transport, participant):
            """Retain every original participant event independently of first-only state."""
            every.append(participant["id"])

        @self.transport.event_handler("on_client_connected")
        def client_handler(transport, participant):
            """Keep the actual compatibility client event for each participant."""
            clients.append(participant["id"])

        await self.transport._on_participant_joined({"id": "first"})
        await self.transport._on_participant_joined({"id": "second"})
        await self.transport.cleanup()
        self.assertEqual(first, [(self.transport, "first")])
        self.assertEqual(every, ["first", "second"])
        self.assertEqual(clients, ["first", "second"])
        self.assertFalse(self.transport._event_tasks)

    async def test_missing_participant_id_does_not_consume_first_event(self):
        """Malformed producer data refuses before changing first-participant state."""
        observed = []

        @self.transport.event_handler("on_first_participant_joined")
        def first_handler(transport, participant):
            """Observe only the later valid native participant event."""
            observed.append(participant["id"])

        with self.assertRaises(KeyError):
            await self.transport._on_participant_joined({})
        self.assertFalse(self.transport._other_participant_has_joined)
        self.assertFalse(self.transport._event_tasks)
        await self.transport._on_participant_joined({"id": "valid"})
        await self.transport.cleanup()
        self.assertEqual(observed, ["valid"])

    async def test_handler_error_does_not_drop_other_registered_callbacks(self):
        """An isolated callback failure must not strand subsequent callback tasks."""
        observed = []

        @self.transport.event_handler("on_first_participant_joined")
        async def failed(transport, participant):
            """Exercise the native handler-error path using nonprivate synthetic text."""
            raise RuntimeError("synthetic-handler-failure")

        @self.transport.event_handler("on_first_participant_joined")
        async def survived(transport, participant):
            """A separately registered callback retains its native delivery contract."""
            observed.append(participant["id"])

        await self.transport._on_participant_joined({"id": "first"})
        await self.transport.cleanup()
        self.assertEqual(observed, ["first"])
        self.assertFalse(self.transport._event_tasks)

    async def test_cleanup_waits_for_pending_handler_without_room_or_media_work(self):
        """Actual cleanup drains its original scheduled callback before returning."""
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []

        @self.transport.event_handler("on_first_participant_joined")
        async def pending_handler(transport, participant):
            """Keep one original callback pending until the explicit test barrier opens."""
            entered.set()
            await release.wait()
            completed.append(participant["id"])

        await self.transport._on_participant_joined({"id": "first"})
        await asyncio.wait_for(entered.wait(), timeout=1)
        cleanup = asyncio.create_task(self.transport.cleanup())
        try:
            await asyncio.sleep(0)
            self.assertFalse(cleanup.done())
            self.assertEqual(completed, [])
        finally:
            release.set()
            await asyncio.wait_for(cleanup, timeout=1)
        self.assertEqual(completed, ["first"])
        self.assertFalse(self.transport._event_tasks)

    async def test_removed_event_name_cannot_silently_register_or_run(self):
        """The obsolete API is not mistaken for the current registered event."""
        called = []

        @self.transport.event_handler("on_first_other_participant_joined")
        def obsolete_handler(transport, participant):
            """This unregistered name must never receive a current native event."""
            called.append(participant)

        await self.transport._on_participant_joined({"id": "first"})
        await self.transport.cleanup()
        self.assertEqual(called, [])
        self.assertNotIn("on_first_other_participant_joined", self.transport._event_handlers)


if __name__ == "__main__":
    unittest.main()

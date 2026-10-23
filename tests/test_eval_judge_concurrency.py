"""Qualify local judge admission and cleanup without provider or account effects.

The maintained CLI module and native filesystem remain active. Only judgement
and display boundaries are synthetic; each test owns its temporary output and
patch cleanup. Invalid admission must not truncate existing judgement evidence.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.cli.commands import eval as command


class TestJudgeConcurrency(unittest.IsolatedAsyncioTestCase):
    """Exercise invalid limits, bounded success and cancellation on the actual helper."""

    async def test_invalid_limits_preserve_existing_output(self):
        """Reject wrong types and nonpositive limits before output or judge dispatch."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            for limit in (0, -1, None, "1", 1.0, True, False):
                with self.subTest(limit=limit):
                    output.write_text("Retain existing synthetic evidence\n")
                    with patch.object(command, "_judge_conversation") as judge:
                        with self.assertRaisesRegex(ValueError, "positive integer"):
                            await asyncio.wait_for(
                                command._judge_all(None, [Path("one.json")], output, limit, False),
                                0.2,
                            )
                        judge.assert_not_called()
                    self.assertEqual(output.read_text(), "Retain existing synthetic evidence\n")

    async def test_positive_limit_bounds_dispatch_and_closes_output(self):
        """A single slot serializes two judgements and flushes two durable result lines."""
        active = maximum = 0

        async def judge(simulation, path):
            """Own one synthetic active slot until completion; never contact a service."""
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0)
                return SimpleNamespace(passed=True)
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            with (
                patch.object(command, "_judge_conversation", judge),
                patch.object(
                    command, "_judgement_record", lambda path, result: {"passed": result.passed}
                ),
                patch.object(command, "_print_judgement"),
            ):
                results = await asyncio.wait_for(
                    command._judge_all(
                        None, [Path("one.json"), Path("two.json")], output, 1, False
                    ),
                    1,
                )
            self.assertEqual((len(results), maximum, active), (2, 1, 0))
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()],
                [{"passed": True}, {"passed": True}],
            )

    async def test_cancellation_joins_active_judgements(self):
        """Cancellation propagates and releases the active child before helper completion."""
        active = 0
        entered = asyncio.Event()

        async def judge(simulation, path):
            """Block one synthetic judgement and release its slot on cancellation."""
            nonlocal active
            active += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            with patch.object(command, "_judge_conversation", judge):
                task = asyncio.create_task(
                    command._judge_all(None, [Path("one.json"), Path("two.json")], output, 1, False)
                )
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(active, 0)
            self.assertEqual(output.read_text(), "")


class TestJudgeCommand(unittest.TestCase):
    """Validate Typer admission before the command reads inputs or overwrites output."""

    def test_cli_rejects_zero_without_dispatch(self):
        """The public -c option rejects zero and leaves retained output untouched."""
        from typer.testing import CliRunner

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judged.jsonl"
            output.write_text("Retain existing synthetic evidence\n")
            with patch.object(command, "_judge_all") as judge:
                result = CliRunner().invoke(
                    command.eval_app,
                    ["judge", "scenario.yaml", "one.json", "-c", "0", "-o", str(output)],
                )
                judge.assert_not_called()
            self.assertEqual(result.exit_code, 2)
            self.assertIn("not in the range x>=1", result.output)
            self.assertEqual(output.read_text(), "Retain existing synthetic evidence\n")

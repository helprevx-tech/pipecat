#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Sarvam TTS text gating."""

import unittest

from pipecat.services.sarvam.tts import _SpeakableTextFilter


class TestSpeakableTextFilter(unittest.IsolatedAsyncioTestCase):
    """Sarvam rejects requests holding no target-language character."""

    async def test_punctuation_only_text_is_emptied(self):
        filter = _SpeakableTextFilter()

        # A doubled full stop is split by sentence aggregation, stranding this.
        self.assertEqual(await filter.filter("."), "")
        self.assertEqual(await filter.filter(".."), "")
        self.assertEqual(await filter.filter("…"), "")
        self.assertEqual(await filter.filter(" -- "), "")
        self.assertEqual(await filter.filter("🙂"), "")

    async def test_speakable_text_passes_through_unchanged(self):
        filter = _SpeakableTextFilter()

        for text in (
            "hamare teeno models alag-alag power requirements ke liye bane hain.",
            "दूसरी तरफ, इलेक्ट्रिक मोटर में सीधे बैटरी से ताकत मिलती है।",
            "X45",
            "123",
        ):
            self.assertEqual(await filter.filter(text), text)

    async def test_script_is_not_assumed_to_be_latin(self):
        """Devanagari, Han and Cyrillic are speakable exactly as Latin is."""
        filter = _SpeakableTextFilter()

        for text in ("हाँ", "你好", "да"):
            self.assertEqual(await filter.filter(text), text)


if __name__ == "__main__":
    unittest.main()

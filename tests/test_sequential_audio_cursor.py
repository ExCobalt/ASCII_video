"""Silent checks of the retained decoder experiment, without device access."""

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from media import normalize_audio_timing, probe
from audio_cursor import SequentialAudioCursor
from test_formats import fixture


class SequentialCursorTests(unittest.TestCase):
    def test_cancel_skipped_prefix_invalidates_history_and_replays_older_target(self):
        with SequentialAudioCursor(self.path, self.info.audio_index, history_seconds=1) as cursor:
            first = cursor.frames(0)
            next(first)
            first.close()
            self.assertTrue(cursor._recent)
            calls = 0

            def stopped():
                nonlocal calls
                calls += 1
                return calls >= 20

            self.assertEqual(list(cursor.frames(2, stopped)), [])
            self.assertGreater(cursor.end, 0)
            self.assertFalse(cursor._recent)
            opens = cursor.opens
            request = cursor.frames(0)
            try:
                frame = next(request)
                np.testing.assert_array_equal(frame.to_ndarray(), self.reference[frame.pts])
                self.assertEqual(frame.pts, min(self.reference))
                self.assertEqual(cursor.opens, opens + 1)
            finally:
                request.close()

    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "audio.mp4"
        fixture(self.path, duration=3, tone=True, layout="stereo", audio_options={"aac_pns": "1"})
        self.info = probe(self.path)
        self.reference = {}
        with av.open(str(self.path)) as container:
            end = None
            for frame in container.decode(container.streams[self.info.audio_index]):
                end, _ = normalize_audio_timing(frame, end)
                self.reference[frame.pts] = frame.to_ndarray().copy()
        with patch("audio_cursor.open_media", side_effect=AssertionError("eager open")):
            self.cursor = SequentialAudioCursor(self.path, self.info.audio_index)
        self.addCleanup(self.cursor.close)

    def window(self, target):
        request = self.cursor.frames(Fraction(str(target)))
        try:
            count = 0
            for frame in request:
                np.testing.assert_array_equal(frame.to_ndarray(), self.reference[frame.pts])
                count += 1
                if self.cursor.end >= Fraction(str(target)) + Fraction(1, 10):
                    break
            self.assertGreater(count, 0)
        finally:
            request.close()

    def test_forward_retains_decoder_backward_restarts_and_close_is_final(self):
        for target in (.175, 1.35, 2.7):
            self.window(target)
            self.assertEqual(self.cursor.opens, 1)
        self.window(.175)
        self.assertEqual(self.cursor.opens, 2)
        self.window(2.7)
        self.assertEqual(self.cursor.opens, 2)
        self.cursor.close()
        self.cursor.close()
        self.assertIsNone(self.cursor._container)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            next(self.cursor.frames(0))

    def test_cancel_forward_scan_preserves_cursor(self):
        calls = 0

        def stopped():
            nonlocal calls
            calls += 1
            return calls >= 10

        self.assertEqual(list(self.cursor.frames(2, stopped)), [])
        self.assertGreater(self.cursor.decoded_frames, 0)
        before = self.cursor.decoded_frames
        self.window(2)
        self.assertEqual(self.cursor.opens, 1)
        self.assertGreater(self.cursor.decoded_frames, before)

    def test_already_cancelled_request_does_not_open_and_eof_is_reused(self):
        self.assertEqual(list(self.cursor.frames(0, lambda: True)), [])
        self.assertEqual(self.cursor.opens, 0)
        self.assertEqual(list(self.cursor.frames(10)), [])
        self.assertTrue(self.cursor.complete)
        before = self.cursor.decoded_frames
        self.assertEqual(list(self.cursor.frames(11)), [])
        self.assertEqual(self.cursor.decoded_frames, before)
        self.assertEqual(self.cursor.opens, 1)
        self.window(.175)
        self.assertEqual(self.cursor.opens, 2)


if __name__ == "__main__":
    unittest.main()

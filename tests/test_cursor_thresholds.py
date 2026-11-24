"""Cursor target/history boundaries at fractional samples and changing rates."""
from copy import copy
from fractions import Fraction
from types import SimpleNamespace
import unittest

from audio_cursor import SequentialAudioCursor
from media import normalize_audio_timing


class CursorThresholdTests(unittest.TestCase):
    def test_fractional_boundaries_negative_times_large_times_and_rate_changes(self):
        for origin in (Fraction(-1, 10), Fraction(0), Fraction(2 ** 40)):
            raw = []
            clock = origin
            for rate in (8000, 44100, 48000, 44100, 8000):
                for _ in range(3):
                    pts = clock.numerator * rate // clock.denominator
                    raw.append(SimpleNamespace(pts=pts, time_base=Fraction(1, rate),
                                               samples=rate // 100, sample_rate=rate))
                    clock = Fraction(pts + rate // 100, rate)
            reference, previous = [], None
            for item in raw:
                frame = copy(item)
                previous, _ = normalize_audio_timing(frame, previous)
                reference.append((frame, previous))
            for index in (0, 5, 9, 14):
                boundary = reference[index][1]
                for offset in (Fraction(-1, 192000), Fraction(0), Fraction(1, 192000)):
                    target = boundary + offset
                    with self.subTest(origin=origin, index=index, offset=offset):
                        history = Fraction(1, 10)
                        with SequentialAudioCursor('unused', 0, history_seconds=history) as cursor:
                            cursor._container = SimpleNamespace(close=lambda: None)
                            cursor._decoder = iter(copy(frame) for frame in raw)
                            actual = list(cursor.frames(target))
                            key = lambda frame: (frame.pts, frame.sample_rate, frame.samples)
                            self.assertEqual([key(frame) for frame in actual],
                                             [key(frame) for frame, end in reference if end > target])
                            expected_history = [key(frame) for frame, end in reference
                                                if end > target - history
                                                and Fraction(frame.pts, frame.sample_rate) >= previous - history]
                            self.assertEqual([key(frame) for frame, _, _ in cursor._recent], expected_history)
                            self.assertEqual(cursor.end, previous)


if __name__ == '__main__':
    unittest.main()

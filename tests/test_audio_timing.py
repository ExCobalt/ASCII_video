"""Exact sample-clock normalization at boundaries and very large timestamps."""
from fractions import Fraction
import math
import random
from types import SimpleNamespace
import unittest

from media import normalize_audio_timing


class AudioTimingTests(unittest.TestCase):
    def test_invalid_timestamp_bases_are_rejected_without_mutation(self):
        for pts, base in ((None, Fraction(1, 1000)), (0, None),
                          (0, Fraction(0)), (0, Fraction(-1, 1000)), (0, 0)):
            frame = SimpleNamespace(pts=pts, time_base=base, sample_rate=48000, samples=960)
            with self.assertRaisesRegex(RuntimeError, 'Frame has no valid PTS/time_base'):
                normalize_audio_timing(frame, None)
            self.assertEqual((frame.pts, frame.time_base), (pts, base))

    def test_negative_half_samples_and_large_pts_preserve_exact_rounding(self):
        for pts in (-3, -1, 1, 3, 2 ** 62 + 1, -(2 ** 62) - 1):
            frame = SimpleNamespace(pts=pts, time_base=Fraction(1, 96000), sample_rate=48000, samples=1024)
            end, gap = normalize_audio_timing(frame, None)
            self.assertEqual(frame.pts, (pts + 1) // 2)
            self.assertEqual(end, Fraction(frame.pts + 1024, 48000))
            self.assertFalse(gap)

    def test_rational_reference_covers_tolerance_gaps_overlap_and_rate_changes(self):
        rng = random.Random(6417)
        for _ in range(3000):
            rate = rng.choice((8000, 16000, 44100, 48000, 96000))
            base = rng.choice((Fraction(1, 1000), Fraction(1, rate),
                               Fraction(1, 10 ** 9), Fraction(1001, 48000)))
            pts = rng.choice((-2 ** 55, -17, 0, 31, 2 ** 55)) + rng.randrange(-100, 100)
            start = pts * base
            tolerance = max(Fraction(1, rate), base)
            previous = start + rng.choice((-2, -1, 0, 1, 2)) * tolerance
            previous += rng.choice((-1, 0, 1)) * Fraction(1, 3 * rate)
            # A prior segment may use another sample rate or a coarser clock.
            if rng.randrange(4) == 0:
                previous = Fraction(round(previous * 44100), 44100)
            frame = SimpleNamespace(pts=pts, time_base=base, sample_rate=rate, samples=1024)
            if start < previous - tolerance:
                with self.assertRaisesRegex(RuntimeError, "Overlapping audio timestamps"):
                    normalize_audio_timing(frame, previous)
                self.assertEqual((frame.pts, frame.time_base), (pts, base))
            else:
                snapped = abs(start - previous) <= tolerance
                expected = math.floor((previous if snapped else start) * rate + Fraction(1, 2))
                end, gap = normalize_audio_timing(frame, previous)
                self.assertEqual((frame.pts, frame.time_base, end, gap),
                                 (expected, Fraction(1, rate), Fraction(expected + 1024, rate), not snapped))

from pathlib import Path
import unittest

import numpy as np

from dithering import DITHER_MODES, thresholds


class DitheringTests(unittest.TestCase):
    def test_blue_noise_has_unique_reproducible_ranks_and_suppresses_low_frequencies(self):
        root = Path(__file__).resolve().parents[1]
        ranks = np.load(root / "assets" / "blue-noise.npy", allow_pickle=False)
        np.testing.assert_array_equal(np.sort(ranks, axis=None), np.arange(4096))
        field = thresholds((64, 64), "blue-noise")
        np.testing.assert_array_equal(field, (ranks.astype(float) + .5) / 4096 - .5)
        frequencies = np.fft.fftfreq(64)
        radius2 = frequencies[:, None] ** 2 + frequencies[None, :] ** 2
        low = (radius2 > 0) & (radius2 < .125 ** 2)
        white = thresholds((64, 64), "noise")
        for fraction in (.25, .5, .75):
            blue_power = np.abs(np.fft.fft2(field + .5 < fraction)) ** 2
            white_power = np.abs(np.fft.fft2(white + .5 < fraction)) ** 2
            self.assertLess(blue_power[low].mean(), .2 * white_power[low].mean())

    def test_bayer_retains_legacy_thresholds(self):
        ranks = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                          [3, 11, 1, 9], [15, 7, 13, 5]])
        expected = np.tile((ranks + .5) / 16 - .5, (3, 4))[:9, :13]
        np.testing.assert_array_equal(thresholds((9, 13), "bayer"), expected)

    def test_fields_are_centered_bounded_and_distinct(self):
        fields = [thresholds((128, 128), mode) for mode in DITHER_MODES]
        for field in fields:
            self.assertTrue(np.isfinite(field).all())
            self.assertLess(abs(field.mean()), 1e-15)
            self.assertGreater(field.min(), -.5)
            self.assertLess(field.max(), .5)
        for index, field in enumerate(fields):
            for other in fields[index + 1:]:
                self.assertFalse(np.array_equal(field, other))
        np.testing.assert_array_equal(fields[0], 0)

    def test_resizing_and_repeated_calls_preserve_origin(self):
        for mode in DITHER_MODES:
            small = thresholds((7, 11), mode)
            large = thresholds((173, 291), mode)
            np.testing.assert_array_equal(small, large[:7, :11])
            np.testing.assert_array_equal(small, thresholds((7, 11), mode))
            small[:] = 42
            self.assertTrue(np.all(thresholds((7, 11), mode) < .5))

    def test_uniform_tiles_reproduce_intermediate_quantization_levels(self):
        for mode in DITHER_MODES[1:]:
            field = thresholds((128, 128), mode)
            for fraction in (.125, .25, .5, .75, .875):
                quantized = np.floor(3.5 + fraction + field)
                self.assertAlmostEqual(quantized.mean(), 3 + fraction, places=12)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            thresholds((3, 5), "unknown")


if __name__ == "__main__":
    unittest.main()

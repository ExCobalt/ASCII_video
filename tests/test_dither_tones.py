"""Rendered tone checks independent of the quantizer's internal formulas."""

import unittest

import numpy as np

from diffusion import MODES as DIFFUSION_MODES
from dithering import DITHER_MODES
from rendering import Renderer


MODES = (*DITHER_MODES, *DIFFUSION_MODES)


class DitherToneTests(unittest.TestCase):
    def test_two_level_ordered_dither_preserves_uniform_tone(self):
        # With black/white levels, white area is the reconstructed mean tone.
        # Complete tiles eliminate any ambiguity from partial pattern coverage.
        for mode in DITHER_MODES[1:]:
            renderer = Renderer(palette=" @", dither=mode)
            for gray in (0, 32, 64, 128, 192, 255):
                with self.subTest(mode=mode, gray=gray):
                    chars = renderer.symbols(np.full((128, 128, 3), gray))
                    self.assertLess(abs(np.mean(chars == "@") - gray / 255), .01)

    def test_two_level_diffusion_preserves_uniform_tone(self):
        for mode in DIFFUSION_MODES:
            renderer = Renderer(palette=" @", dither=mode)
            # Atkinson deliberately discards one quarter of the error; its
            # contrast increase is not a brightness-conservation guarantee.
            tolerance = .16 if mode == "atkinson" else .04 if mode == "riemersma" else .025
            for gray in (0, 32, 64, 128, 192, 255):
                with self.subTest(mode=mode, gray=gray):
                    chars = renderer.symbols(np.full((128, 128, 3), gray))
                    self.assertLess(abs(np.mean(chars == "@") - gray / 255), tolerance)

    def test_black_and_white_remain_pure_for_every_mode(self):
        for mode in MODES:
            renderer = Renderer(palette=" !@", dither=mode)
            for gray, expected in ((0, " "), (255, "@")):
                with self.subTest(mode=mode, gray=gray):
                    np.testing.assert_array_equal(
                        renderer.symbols(np.full((17, 23, 3), gray)), expected)

    def test_exact_custom_palette_levels_are_not_mixed(self):
        for mode in MODES:
            renderer = Renderer(palette=" !@", dither=mode)
            for gray, expected in ((0, " "), (127.5, "!"), (255, "@")):
                with self.subTest(mode=mode, gray=gray):
                    np.testing.assert_array_equal(
                        renderer.symbols(np.full((32, 32, 3), gray)), expected)

    def test_no_dither_selects_nearest_custom_level(self):
        # Independent nearest-distance oracle on either side of each midpoint.
        gray = np.array([0, 20, 42.49, 42.51, 90, 127.49, 127.51,
                         180, 212.49, 212.51, 240, 255])
        palette = " !#@"
        levels = (0, 85, 170, 255)
        expected = [palette[min(range(4), key=lambda i: abs(value - levels[i]))]
                    for value in gray]
        grid = np.repeat(gray[None, :, None], 3, axis=2)
        np.testing.assert_array_equal(Renderer(palette=palette).symbols(grid), [expected])

    def test_frames_and_grid_resizing_leave_no_quantization_history(self):
        grid = np.random.default_rng(2026).integers(0, 256, (19, 31, 3))
        for mode in MODES:
            with self.subTest(mode=mode):
                renderer = Renderer(palette=" !#@", dither=mode)
                expected = renderer.symbols(grid)
                renderer.symbols(255 - grid)
                renderer.symbols(np.full((7, 13, 3), 64))
                np.testing.assert_array_equal(renderer.symbols(grid), expected)
                np.testing.assert_array_equal(
                    Renderer(palette=" !#@", dither=mode).symbols(grid), expected)


if __name__ == "__main__":
    unittest.main()

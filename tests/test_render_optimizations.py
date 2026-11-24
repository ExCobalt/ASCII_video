"""Exact arithmetic regressions against the pre-optimization operations."""
import unittest

import numpy as np

from dithering import _FIELDS, thresholds
from rendering import Renderer, _level_indices


def luminance_before(grid):
    return np.clip(.2126 * grid[..., 0] + .7152 * grid[..., 1] + .0722 * grid[..., 2], 0, 255)


def tone_before(y, contrast=1.3, brightness=.2, gamma=1.7):
    t = np.clip(((y / 255 - .5) * contrast + .5) * (1 + brightness), 0, 1)
    return 255 * t ** (1 / gamma)


def ordered_before(values, levels, field):
    lower = np.clip(np.searchsorted(levels, values, side="right") - 1, 0, len(levels) - 1)
    upper = np.minimum(lower + 1, len(levels) - 1)
    span = levels[upper] - levels[lower]
    fraction = np.divide(values - levels[lower], span,
                         out=np.zeros_like(values), where=span > 0)
    return np.where(fraction >= field + .5, upper, lower)


def field_before(shape, field):
    rows, columns = shape
    return field[np.arange(rows)[:, None] % field.shape[0], np.arange(columns)[None, :] % field.shape[1]]


class RenderingOptimizationTests(unittest.TestCase):
    def test_luminance_and_tone_preserve_order_and_inputs(self):
        rng = np.random.default_rng(20261113)
        for dtype in (np.float32, np.float64):
            source = rng.uniform(-400, 500, (19, 37, 3)).astype(dtype)
            for grid in (source, source[::-1, ::-1], source.transpose(1, 0, 2)):
                original = grid.copy()
                expected = luminance_before(grid)
                np.testing.assert_array_equal(Renderer._luminance(grid), expected)
                for contrast, brightness, gamma in ((1.3, .2, 1.7), (.1, -1, .1), (3, 1, 5), (1, 0, 1)):
                    renderer = Renderer(contrast=contrast, brightness=brightness, gamma=gamma)
                    np.testing.assert_array_equal(renderer._tone(expected),
                                                  tone_before(expected, contrast, brightness, gamma)
                                                  if (contrast, brightness, gamma) != (1, 0, 1) else expected)
                np.testing.assert_array_equal(grid, original)

    def test_ordered_endpoints_duplicates_and_strides(self):
        rng = np.random.default_rng(26)
        for levels in (np.array([0., .1, .1, .3, 1.]), np.linspace(0, 1, 95)):
            values = np.r_[rng.uniform(-1, 2, 500), levels, np.nextafter(levels, -np.inf),
                           np.nextafter(levels, np.inf)][None, ::-1]
            field = thresholds(values.shape, "bayer")
            np.testing.assert_array_equal(_level_indices(values, levels, field), ordered_before(values, levels, field))

    def test_saturation_extrema_preserve_exact_colors(self):
        source = np.random.default_rng(21).uniform(-30, 300, (31, 57, 3))
        for grid in (source, source[::-1, ::-1], source.transpose(1, 0, 2)):
            rgb = np.clip(grid, 0, 255)
            y = luminance_before(rgb)
            chroma = rgb - y[..., None]
            upper, lower = np.max(chroma, axis=-1), -np.min(chroma, axis=-1)
            high = np.divide(255 - y, upper, out=np.full_like(y, np.inf), where=upper > 0)
            low = np.divide(y, lower, out=np.full_like(y, np.inf), where=lower > 0)
            for amount in (-1, -.5, .5, 1):
                strength = np.minimum(1 + amount, np.minimum(high, low))
                expected = np.clip(y[..., None] + strength[..., None] * chroma, 0, 255)
                actual = Renderer(color="source", color_saturation=amount)._color_grid(grid)
                np.testing.assert_array_equal(actual, expected)

    def test_threshold_partial_tiles_are_exact_and_independent(self):
        for mode, field in _FIELDS.items():
            for shape in ((1, 1), (3, 5), (90, 320), (129, 131)):
                expected = field_before(shape, field)
                actual = thresholds(shape, mode)
                np.testing.assert_array_equal(actual, expected)
                actual[:] = 0
                np.testing.assert_array_equal(thresholds(shape, mode), expected)


if __name__ == "__main__":
    unittest.main()

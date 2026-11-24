import unittest
from unittest.mock import patch

import numpy as np

import diffusion
from cli_options import PALETTES
from dithering import DITHER_MODES, thresholds
from raster import Rasterizer, _glyph_atlas, glyph_profile
from rendering import Renderer, _level_indices


class GlyphProfileTests(unittest.TestCase):
    def test_profiles_follow_actual_readonly_raster_masks(self):
        atlas, _ = _glyph_atlas(8, 16, 12)
        self.assertIs(_glyph_atlas(8, 16, 12)[0], atlas)
        self.assertFalse(atlas.flags.writeable)
        self.assertIs(Rasterizer(" @", "mono", 16, 16).atlas, atlas)
        with self.assertRaises(ValueError):
            atlas[0, 0, 0] = 1
        for palette in PALETTES.values():
            chars, levels = glyph_profile(palette)
            self.assertIs(glyph_profile(palette)[0], chars)
            self.assertFalse(chars.flags.writeable)
            self.assertFalse(levels.flags.writeable)
            self.assertEqual(set(chars), set(palette))
            density = np.array([atlas[ord(c)-32].sum(dtype=np.uint64) for c in chars])
            np.testing.assert_array_equal(levels, density / density.max())
            self.assertTrue(np.all(np.diff(levels) >= 0))
            self.assertEqual(levels[0], 0)
            self.assertEqual(levels[-1], 1)
        self.assertEqual(''.join(glyph_profile(PALETTES['basic'])[0]), " -.:=+*%#@")

    def test_stored_ramps_are_sorted_and_medium_spans_extended_density(self):
        atlas, _ = _glyph_atlas(8, 16, 12)
        self.assertEqual({name: len(ramp) for name, ramp in PALETTES.items()},
                         {"basic": 10, "medium": 32, "extended": 70})
        for ramp in PALETTES.values():
            density = np.array([atlas[ord(c)-32].mean() for c in ramp])
            self.assertTrue(np.all(np.diff(density) >= 0))
            self.assertEqual(''.join(glyph_profile(ramp)[0]), ramp)
        medium, extended = PALETTES['medium'], PALETTES['extended']
        self.assertTrue(set(medium) <= set(extended))
        self.assertEqual((medium[0], medium[-1]), (extended[0], extended[-1]))
        basic_levels = glyph_profile(PALETTES['basic'])[1]
        medium_levels = glyph_profile(medium)[1]
        self.assertLess(np.diff(medium_levels).max(), np.diff(basic_levels).max())
        self.assertEqual(Renderer().palette.tolist(), list(PALETTES['basic']))

    def test_custom_palette_preserves_order_and_full_endpoint_range(self):
        renderer = Renderer(palette="@. ")
        np.testing.assert_array_equal(renderer.levels, [0, .5, 1])
        grid = np.repeat(np.array([0, 127.5, 255])[None, :, None], 3, axis=2)
        np.testing.assert_array_equal(renderer.symbols(grid), np.array([list("@. ")]))

    def test_nearest_nonuniform_levels_and_monotone_gradient(self):
        levels = np.array([0, .1, .7, 1])
        values = np.array([[0, .049, .051, .399, .401, .849, .851, 1]])
        np.testing.assert_array_equal(_level_indices(values, levels), [[0, 0, 1, 1, 2, 2, 3, 3]])
        for palette in PALETTES.values():
            renderer = Renderer(palette=palette)
            grid = np.repeat(np.linspace(0, 255, 1024)[None, :, None], 3, axis=2)
            chars = renderer.symbols(grid)
            coverage = _glyph_atlas(8, 16, 12)[0][chars.view(np.uint32)-32].mean(axis=(-2, -1))
            self.assertTrue(np.all(np.diff(coverage[0]) >= 0))

    def test_ordered_occupancy_uses_actual_adjacent_interval(self):
        levels = np.array([0, .1, .7, 1])
        field = thresholds((4, 4), 'bayer')
        for fraction, upper_count in ((0, 0), (.25, 4), (.5, 8), (.75, 12), (1, 16)):
            values = np.full((4, 4), .1 + fraction * .6)
            selected = _level_indices(values, levels, field)
            self.assertTrue(np.isin(selected, [1, 2]).all())
            self.assertEqual(np.count_nonzero(selected == 2), upper_count)

    def test_exact_levels_and_duplicate_densities_are_deterministic(self):
        levels = np.array([0, .2, .2, 1])
        for mode in DITHER_MODES:
            field = None if mode == 'none' else thresholds((4, 4), mode)
            with np.errstate(divide='raise', invalid='raise'):
                np.testing.assert_array_equal(_level_indices(np.full((4, 4), .2), levels, field),
                                              np.full((4, 4), 2))
                np.testing.assert_array_equal(_level_indices(np.zeros((4, 4)), levels, field),
                                              np.zeros((4, 4), dtype=int))
                np.testing.assert_array_equal(_level_indices(np.ones((4, 4)), levels, field),
                                              np.full((4, 4), 3))

    def test_diffusion_preserves_exact_measured_levels_in_both_paths(self):
        chars, levels = glyph_profile(PALETTES['basic'])
        values = np.tile(levels, (4, 1))
        for mode in diffusion.MODES:
            expected = np.tile(np.arange(len(levels)), (4, 1))
            np.testing.assert_array_equal(diffusion.quantize(values, levels, mode), expected)
            with patch.object(diffusion, '_native_quantize', None):
                np.testing.assert_array_equal(diffusion.quantize(values, levels, mode), expected)

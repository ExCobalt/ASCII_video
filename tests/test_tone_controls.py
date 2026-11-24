import unittest
from unittest.mock import patch

import numpy as np

import diffusion
from cli_options import PALETTES
from dithering import DITHER_MODES
from raster import _glyph_atlas
from rendering import Renderer


WEIGHTS = np.array([.2126, .7152, .0722])


class ToneControlsTests(unittest.TestCase):
    def test_neutral_controls_preserve_all_palettes_and_dither_modes(self):
        grid = np.random.default_rng(9).uniform(0, 255, (7, 11, 3))
        for palette in (*PALETTES.values(), "@ ."):
            for mode in (*DITHER_MODES, *diffusion.MODES):
                with self.subTest(palette=palette, mode=mode):
                    baseline = Renderer(palette=palette, dither=mode)
                    actual = Renderer(palette=palette, dither=mode, color="source",
                                      color_saturation=0, glyph_correction=0)
                    chars, colors = actual.cells(grid)
                    np.testing.assert_array_equal(chars, baseline.symbols(grid))
                    np.testing.assert_array_equal(colors, grid)

    def test_custom_reverse_still_inverts_without_correction(self):
        chars, colors = Renderer("@ ", color="source").cells(np.array([[[0, 0, 0], [255, 255, 255]]]))
        np.testing.assert_array_equal(chars, [["@", " "]])
        np.testing.assert_array_equal(colors, [[[0, 0, 0], [255, 255, 255]]])

    def test_saturation_grayscale_and_double_chroma(self):
        grid = np.array([[[100., 120., 140.], [80., 90., 100.]]])
        y = grid @ WEIGHTS
        gray = Renderer(color="source", color_saturation=-1).cells(grid)[1]
        np.testing.assert_allclose(gray, np.repeat(y[..., None], 3, axis=-1))
        boosted = Renderer(color="source", color_saturation=1).cells(grid)[1]
        np.testing.assert_allclose(boosted, y[..., None] + 2 * (grid - y[..., None]))

    def test_saturation_preserves_luma_and_chroma_direction_at_gamut_limit(self):
        grid = np.array([[[250., 70., 20.], [10., 20., 200.], [255., 0., 0.], [55., 55., 55.]]])
        y = grid @ WEIGHTS
        result = Renderer(color="source", color_saturation=1).cells(grid)[1]
        np.testing.assert_allclose(result @ WEIGHTS, y, atol=1e-12)
        self.assertTrue(np.all((result >= 0) & (result <= 255)))
        for original, actual, luma in zip(grid[0, :3], result[0, :3], y[0, :3]):
            ratios = (actual - luma) / (original - luma)
            np.testing.assert_allclose(ratios, np.full(3, ratios[0]), atol=1e-12)
        np.testing.assert_allclose(result[0, 2], grid[0, 2])

    def test_saturation_has_no_effect_on_mono_or_tint(self):
        grid = np.array([[[20., 180., 70.]]])
        for color in ("mono", "#123456"):
            renderer = Renderer(color=color, color_saturation=1)
            np.testing.assert_array_equal(renderer.cells(grid)[1], grid)

    def test_tone_known_numeric_vectors_and_clipping(self):
        y = np.array([0., 63.75, 127.5, 191.25, 255.])
        np.testing.assert_array_equal(Renderer(brightness=-1)._tone(y), np.zeros(5))
        np.testing.assert_allclose(Renderer(brightness=1)._tone(y), [0, 127.5, 255, 255, 255])
        np.testing.assert_allclose(Renderer(contrast=2)._tone(y), [0, 0, 127.5, 255, 255])
        np.testing.assert_allclose(Renderer(contrast=.5)._tone(y), [63.75, 95.625, 127.5, 159.375, 191.25])
        np.testing.assert_allclose(Renderer(gamma=2)._tone(y), np.sqrt(y / 255) * 255)
        np.testing.assert_allclose(Renderer(gamma=.5)._tone(y), (y / 255) ** 2 * 255)
        np.testing.assert_array_equal(Renderer()._tone(y), y)

    def test_tone_operations_apply_contrast_then_gain_then_gamma(self):
        result = Renderer(contrast=2, brightness=.5, gamma=2)._tone(np.array([102.]))
        np.testing.assert_allclose(result, [255 * np.sqrt(.45)])

    def test_full_source_compensation_reaches_density_normalized_energy(self):
        grid = np.array([[[64., 64., 64.], [170., 60., 30.], [255., 0., 0.], [20., 70., 190.]]])
        for palette in (*PALETTES.values(), "@ .", " .@"):
            renderer = Renderer(palette, color="source", glyph_correction=1)
            chars, colors = renderer.cells(grid)
            atlas, _ = _glyph_atlas(8, 16, 12)
            coverage = atlas.mean(axis=(1, 2)) / 255
            ink = coverage[chars.view(np.uint32) - 32]
            max_ink = max(coverage[ord(c) - 32] for c in palette)
            np.testing.assert_allclose(ink * (colors @ WEIGHTS), max_ink * (grid @ WEIGHTS), atol=1e-12)
            self.assertTrue(np.all((colors >= 0) & (colors <= 255)))
            ratio = np.divide(colors, grid, out=np.zeros_like(colors), where=grid > 0)
            active = grid > 0
            for a, mask in zip(ratio[0], active[0]):
                np.testing.assert_allclose(a[mask], np.full(mask.sum(), a[mask][0]))

    def test_partial_compensation_interpolates_actual_baseline_energy(self):
        grid = np.array([[[40., 100., 70.], [160., 40., 80.]]])
        atlas, _ = _glyph_atlas(8, 16, 12)
        coverage = atlas.mean(axis=(1, 2)) / 255
        base = Renderer(color="source", gamma=1.2)
        baseline = coverage[base.symbols(grid).view(np.uint32) - 32] * (grid @ WEIGHTS)
        max_ink = max(coverage[ord(c) - 32] for c in PALETTES["basic"])
        full_target = max_ink * base._tone(grid @ WEIGHTS)
        for amount in (.1, .5, .9, 1):
            renderer = Renderer(color="source", gamma=1.2, glyph_correction=amount)
            chars, colors = renderer.cells(grid)
            energy = coverage[chars.view(np.uint32) - 32] * (colors @ WEIGHTS)
            np.testing.assert_allclose(energy, baseline + amount * (full_target - baseline), atol=1e-12)

    def test_black_space_and_lifted_black_are_finite(self):
        grid = np.zeros((3, 5, 3))
        for color in ("mono", "source", "#123456"):
            for contrast in (1, .5):
                for amount in (.1, 1):
                    with np.errstate(all="raise"):
                        chars, colors = Renderer(" @", color=color, contrast=contrast,
                                                 glyph_correction=amount).cells(grid)
                    self.assertTrue(np.isfinite(colors).all())
                    if color == "source":
                        np.testing.assert_array_equal(colors, grid)
                    if contrast == 1:
                        np.testing.assert_array_equal(chars, np.full((3, 5), " "))

    def test_full_mono_compensation_uses_actual_density_for_custom_palette(self):
        grid = np.repeat(np.array([[[0.], [64.], [128.], [255.]]]), 3, axis=-1)
        renderer = Renderer("@ .", glyph_correction=1)
        chars = renderer.cells(grid)[0]
        atlas, _ = _glyph_atlas(8, 16, 12)
        density = {c: atlas[ord(c) - 32].mean() for c in "@ ."}
        ordered = sorted(density, key=density.get)
        expected = [[min(ordered, key=lambda c: (abs(density[c] / density["@"] - value / 255),
                                                -density[c])) for value in (0, 64, 128, 255)]]
        np.testing.assert_array_equal(chars, expected)

    def test_compensation_caps_impossible_highlight_without_changing_rgb_direction(self):
        grid = np.array([[[255., 20., 10.]]])
        renderer = Renderer(color="source", glyph_correction=1, gamma=5)
        chars, colors = renderer.cells(grid)
        np.testing.assert_allclose(colors, grid)
        self.assertEqual(chars[0, 0], renderer.palette[renderer._ink_order[-1]])

    def test_compensation_native_and_python_diffusion_match(self):
        grid = np.random.default_rng(123).uniform(0, 255, (5, 9, 3))
        for mode in diffusion.MODES:
            renderer = Renderer(" @.+", color="source", glyph_correction=.6, dither=mode)
            native = renderer.cells(grid)
            with patch.object(diffusion, "_native_quantize", None):
                pure = renderer.cells(grid)
            np.testing.assert_array_equal(native[0], pure[0])
            np.testing.assert_allclose(native[1], pure[1], atol=1e-12)

    def test_control_range_validation(self):
        for name, values in (("glyph_correction", (-.01, 1.01, np.nan, np.inf)),
                             ("color_saturation", (-1.01, 1.01, np.nan, np.inf))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Renderer(**{name: value})


if __name__ == "__main__":
    unittest.main()

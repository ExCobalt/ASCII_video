import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

import diffusion
from dithering import DITHER_MODES, thresholds
from rendering import BAYER, DEFAULT_PALETTE, Renderer, display_aspect, fit_grid, rotate_rgb


def scalar_sample(rgb, columns, rows):
    # Independent overlap weights in integer coordinates: a target cell has
    # width source_width, while a source pixel has width target_columns.
    height, width = rgb.shape[:2]
    out = np.empty((rows, columns, 3), dtype=np.float64)
    for y in range(rows):
        ya, yb = y * height // rows, ((y+1)*height + rows-1) // rows
        wy = [max(0, min((y+1)*height, (sy+1)*rows)-max(y*height, sy*rows))
              for sy in range(ya, yb)]
        for x in range(columns):
            xa, xb = x * width // columns, ((x+1)*width + columns-1) // columns
            wx = [max(0, min((x+1)*width, (sx+1)*columns)-max(x*width, sx*columns))
                  for sx in range(xa, xb)]
            weights = np.asarray(wy, dtype=np.float64)[:, None] * wx
            out[y, x] = (rgb[ya:yb, xa:xb] * weights[..., None]).sum(axis=(0,1)) / (height*width)
    return out


def scalar_symbols(grid, palette, gamma, contrast, dither, brightness=0):
    from cli_options import PALETTES
    from raster import _glyph_atlas

    if palette in PALETTES.values():
        atlas, _ = _glyph_atlas(8, 16, 12)
        measured = [(c, sum(int(v) for v in atlas[ord(c)-32].flat)) for c in palette]
        measured.sort(key=lambda item: item[1])
        palette = ''.join(c for c, _ in measured)
        levels = [coverage / measured[-1][1] for _, coverage in measured]
    else:
        levels = [i / (len(palette) - 1) for i in range(len(palette))]
    out = np.empty(grid.shape[:2], dtype="U1")
    field = thresholds(grid.shape[:2], dither) if dither != "none" else None
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            r, g, b = grid[y, x]
            light = min(255, max(0, 0.2126*r + 0.7152*g + 0.0722*b)) / 255
            if gamma != 1 or contrast != 1 or brightness != 0:
                light = min(1, max(0, ((light-.5)*contrast+.5)*(1+brightness))) ** (1/gamma)
            if field is None:
                index = min(range(len(levels)), key=lambda i: (abs(levels[i]-light), -i))
            else:
                index = max((i for i, level in enumerate(levels) if level <= light), default=0)
                if index < len(levels)-1:
                    fraction = (light-levels[index]) / (levels[index+1]-levels[index])
                    if fraction >= field[y, x]+.5:
                        index += 1
            out[y, x] = palette[index]
    return out


class RenderingTests(unittest.TestCase):
    def test_additional_dithering_keeps_source_colors_and_has_no_frame_history(self):
        rgb = np.random.default_rng(51).integers(0, 256, (13, 17, 3), dtype=np.uint8)
        original = rgb.copy()
        palette = " .#@"
        for mode in (*DITHER_MODES[2:], *diffusion.MODES):
            with self.subTest(mode=mode), patch.object(diffusion, "_native_quantize", None):
                renderer = Renderer(palette=palette, dither=mode)
                chars, colors = renderer.render(rgb, 7, 5)
                np.testing.assert_allclose(colors, scalar_sample(rgb, 7, 5), rtol=0, atol=1e-10)
                y = .2126 * colors[..., 0] + .7152 * colors[..., 1] + .0722 * colors[..., 2]
                if mode in diffusion.MODES:
                    levels = np.linspace(0, 1, len(palette))
                    indices = diffusion.quantize(y / 255, levels, mode)
                else:
                    expected = scalar_symbols(colors, palette, 1, 1, mode)
                if mode in diffusion.MODES:
                    expected = np.array(list(palette))[indices]
                np.testing.assert_array_equal(chars, expected)
                renderer.render(255 - rgb, 3, 4)
                np.testing.assert_array_equal(renderer.render(rgb, 7, 5)[0], chars)
        np.testing.assert_array_equal(rgb, original)

    def test_rgb8_sum_precision_boundary_and_dtype_fallback(self):
        renderer = Renderer()
        # At 65793 white pixels the sum is 2**24 - 1; one more exceeds
        # float32's exact-integer range. Both sides must retain exact means.
        for width in (65793, 65794):
            rgb = np.full((1, width, 3), 255, dtype=np.uint8)
            rgb[0, -1] = (254, 253, 252)
            actual = renderer.sample(rgb, 1, 1)
            self.assertEqual(actual.dtype, np.float64)
            np.testing.assert_array_equal(actual, scalar_sample(rgb, 1, 1))
        # Same cached geometry with a different dtype must not use RGB8's
        # bound for arbitrary values or lose small fractional contributions.
        rgb = np.full((11, 17, 3), 1e8, dtype=np.float64)
        renderer.sample(rgb.astype(np.uint8), 4, 3)
        rgb[::2, ::3] += .125
        np.testing.assert_allclose(renderer.sample(rgb, 4, 3), scalar_sample(rgb, 4, 3), rtol=0, atol=1e-6)

    def test_automatic_grid_tracks_terminal_without_a_column_cap(self):
        self.assertEqual(fit_grid(None, None, 641, 182, .5, 16 / 9),
                         (640, 180, 640, 181))
        self.assertEqual(fit_grid(None, None, 81, 25, .5, 16 / 9),
                         (80, 23, 80, 24))
        self.assertEqual(fit_grid(120, None, 641, 182, .5, 16 / 9),
                         (120, 34, 120, 181))

    def test_shared_sampler_keeps_each_geometry_during_resize(self):
        rgb = np.random.default_rng(31).integers(0, 256, (121, 213, 3), dtype=np.uint8)
        sizes = [(60, 34), (80, 45), (120, 20), (240, 140)]
        expected = {size: scalar_sample(rgb, *size) for size in sizes}
        renderer = Renderer()

        def check(size):
            np.testing.assert_allclose(renderer.sample(rgb, *size), expected[size], rtol=0, atol=1e-10)

        # Decoder and resize redraw share this sampler while NumPy releases the GIL.
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(check, sizes * 50))

    def test_area_average_and_fractional_buckets(self):
        rng = np.random.default_rng(17)
        renderer = Renderer()
        for h, w in ((1, 1), (2, 3), (3, 2), (7, 11)):
            rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            for c, r in ((1, 1), (2, 2), (5, 8), (13, 9), (w, h)):
                actual = renderer.sample(rgb, c, r)
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_allclose(actual, scalar_sample(rgb, c, r), rtol=0, atol=1e-9)

    def test_scalar_oracle_tones_and_dither(self):
        rgb = np.random.default_rng(23).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
        for gamma in (0.1, 1, 2, 5):
            for contrast in (0.1, 1, 2, 3):
                for dither in ("none", "bayer"):
                    renderer = Renderer(gamma=gamma, contrast=contrast, dither=dither)
                    chars, grid = renderer.render(rgb, 7, 4)
                    expected = scalar_symbols(grid, DEFAULT_PALETTE, gamma, contrast, dither)
                    np.testing.assert_array_equal(chars, expected)
                    np.testing.assert_allclose(grid, scalar_sample(rgb, 7, 4), rtol=0, atol=1e-9)

    def test_equal_height_bands_and_rotated_views(self):
        rgb = np.random.default_rng(40).integers(0, 256, size=(24, 36, 3), dtype=np.uint8)
        renderer = Renderer()
        for rotation in (0, 90, 180, 270):
            frame = rotate_rgb(rgb, rotation)
            for columns, rows in ((5, 6), (12, 12), (40, 30)):
                np.testing.assert_allclose(renderer.sample(frame, columns, rows),
                                           scalar_sample(frame, columns, rows), rtol=0, atol=1e-9)

    def test_unequal_height_bands_at_playback_resolutions(self):
        renderer = Renderer()
        rng = np.random.default_rng(42)
        for height, width in ((1080, 1920), (720, 1280)):
            rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            for columns, rows in ((160, 45), (240, 68)):
                expected = scalar_sample(rgb, columns, rows)
                np.testing.assert_allclose(renderer.sample(rgb, columns, rows), expected, rtol=0, atol=1e-10)

    def test_geometry_and_orientation(self):
        self.assertEqual(fit_grid(160, None, 200, 60, 0.5, 16 / 9), (160, 45, 160, 59))
        self.assertEqual(fit_grid(160, 20, 100, 40, 0.5, 16 / 9), (71, 20, 99, 20))
        self.assertEqual(fit_grid(1, 1, 2, 2, 0.5, 100), (1, 1, 1, 1))
        self.assertAlmostEqual(display_aspect(720, 576, 16 / 15), 4 / 3)
        self.assertAlmostEqual(display_aspect(720, 576, 16 / 15, 90), 3 / 4)
        self.assertEqual(display_aspect(100, 50, None), 2)
        rgb = np.arange(18).reshape(2, 3, 3)
        for rotation in (0, 90, 180, 270):
            np.testing.assert_array_equal(rotate_rgb(rgb, rotation), np.rot90(rgb, -rotation // 90))
        with self.assertRaises(ValueError):
            display_aspect(720, 576, 1, 45)
        with self.assertRaises(ValueError):
            fit_grid(160, None, 1, 10, 0.5, 16 / 9)

    def test_brightness_boundaries(self):
        renderer = Renderer(palette=" .#@")
        values = np.array([0, 255/6-1e-8, 255/6+1e-8, 255/2-1e-8, 255/2+1e-8, 255])
        grid = np.repeat(values[None, :, None], 3, axis=2)
        np.testing.assert_array_equal(renderer.symbols(grid), np.array([list("  ..#@")]))

    def test_tone_and_source_rgb(self):
        palette = " .#%@"
        grid = np.repeat(np.array([63.75, 127.5, 191.25])[None, :, None], 3, axis=2)
        np.testing.assert_array_equal(Renderer(palette=palette, contrast=2).symbols(grid),
                                      np.array([[' ', '#', '@']]))
        self.assertEqual(Renderer(palette=palette, gamma=2).symbols(grid)[0, 0], '#')
        rgb = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
        _, actual = Renderer(gamma=2, contrast=2, dither="bayer").render(rgb, 1, 1)
        np.testing.assert_array_equal(actual, [[[1.5, 2.5, 3.5]]])

    def test_brightness_gain_preserves_geometry_and_source_colors(self):
        rgb = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
        original = rgb.copy()
        for brightness in (-1, -.25, 0, .25, 1):
            symbols, colors = Renderer(brightness=brightness).render(rgb, 3, 1)
            expected = scalar_symbols(rgb, DEFAULT_PALETTE, 1, 1, "none", brightness)
            np.testing.assert_array_equal(symbols, expected)
            np.testing.assert_array_equal(colors, rgb)
            self.assertEqual(symbols.shape, (1, 3))
        np.testing.assert_array_equal(rgb, original)
        # Contrast, gain, clipping, then gamma (independent scalar reference).
        colors = np.repeat(np.array([40, 110, 180])[None, :, None], 3, axis=2)
        renderer = Renderer(gamma=2, contrast=1.3, brightness=0.1)
        expected = scalar_symbols(colors, DEFAULT_PALETTE, 2, 1.3, "none", .1)
        np.testing.assert_array_equal(renderer.symbols(colors), expected)
        for value in (-1.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Renderer(brightness=value)

    def test_bayer_stability_and_tiling(self):
        # Halfway between two custom density levels gives equal occupancy.
        grid = np.full((8, 8, 3), 127.5, dtype=np.float64)
        renderer = Renderer(palette=" .#@", dither="bayer")
        chars = renderer.symbols(grid)
        np.testing.assert_array_equal(chars[:4, :4], renderer.palette[1 + (BAYER < 8).astype(int)])
        self.assertEqual(np.count_nonzero(chars[:4, :4] == renderer.palette[1]), 8)
        np.testing.assert_array_equal(chars[:4, :4], chars[4:, 4:])
        np.testing.assert_array_equal(chars, renderer.symbols(grid))
        self.assertTrue(np.all(Renderer(dither="bayer").symbols(np.zeros((8, 8, 3))) == ' '))
        self.assertTrue(np.all(Renderer(dither="bayer").symbols(np.full((8, 8, 3), 255)) == '@'))

    def test_validation_and_palette_order(self):
        for kwargs in ({"palette": "aa"}, {"palette": "a\n"}, {"palette": "aж"},
                       {"gamma": float("nan")}, {"contrast": float("inf")}, {"dither": "random"}):
            with self.assertRaises(ValueError):
                Renderer(**kwargs)
        chars, _ = Renderer(palette="@ ").render(np.zeros((1, 1, 3), dtype=np.uint8), 1, 1)
        self.assertEqual(chars[0, 0], '@')


if __name__ == "__main__":
    unittest.main()

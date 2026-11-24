from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raster import Rasterizer, font_path
import raster as raster_module


class RasterTests(unittest.TestCase):
    def test_source_export_strips_match_whole_plane_filter_and_retain_frames(self):
        # Independent full-image LANCZOS reference, including arbitrary high
        # frequency pixels at strip seams rather than only real glyph masks.
        rng = np.random.default_rng(20261113)
        planes = rng.integers(0, 256, (3, 1440, 2560), dtype=np.uint8)
        expected = np.stack([np.asarray(Image.fromarray(p).resize(
            (1920, 1080), Image.Resampling.LANCZOS)) for p in planes], axis=-1)
        symbols = np.full((90, 320), '@', dtype='U1')[:, ::-1]
        colors = np.zeros((90, 320, 3), dtype=np.uint8)[:, ::-1]
        symbols.flags.writeable = colors.flags.writeable = False
        raster = Rasterizer(' @', 'source', 1920, 1080)
        raster._source_parallel_strips = True
        boxes = []
        resize = Image.Image.resize
        def observed(image, *args, **kwargs):
            boxes.append(kwargs.get('box'))
            return resize(image, *args, **kwargs)
        with ThreadPoolExecutor(max_workers=12) as pool, \
                patch.object(raster_module, '_compose', object()), \
                patch.object(raster_module, '_compose_planes_sparse', None, create=True), \
                patch.object(raster_module, '_compose_planes', return_value=planes):
            raster._resize_pool = pool
            with patch.object(Image.Image, 'resize', observed):
                actual = raster.draw(symbols, colors)
            self.assertEqual(len(boxes), 12)
            self.assertTrue(all(box is not None for box in boxes))
            np.testing.assert_array_equal(actual, expected)
            planes[:] = 0
            self.assertFalse(raster.draw(symbols, colors).any())
            np.testing.assert_array_equal(actual, expected)

    def test_native_size_mono_and_blank_frames_are_readonly_and_independent(self):
        for symbol in ('@', ' '):
            raster = Rasterizer(' @', 'mono', 16, 16)
            actual = raster.draw(np.full((1, 2), symbol), None)
            expected = actual.copy()
            self.assertFalse(actual.flags.writeable)
            self.assertEqual(actual.strides[-1], 0)
            raster.draw(np.full((1, 2), '#'), None)
            np.testing.assert_array_equal(actual, expected)

    def test_fixed_atlas_cache_observes_replacement_and_mutable_edits(self):
        raster = Rasterizer(' @', '#19ABEF', 8, 16)
        symbols = np.array([['@']])
        raster.draw(symbols, None)
        raster.atlas = raster.atlas.copy()
        for coverage in (31, 197):
            raster.atlas[ord('@') - 32] = coverage
            expected = (coverage * np.asarray(raster.rgb) + 127) // 255
            np.testing.assert_array_equal(raster.draw(symbols, None),
                                          np.broadcast_to(expected, (16, 8, 3)))

    def test_blank_shortcut_validates_colors_and_selected_atlas(self):
        symbols = np.array([[' ']])
        raster = Rasterizer(' @', 'source', 8, 16)
        with self.assertRaises(ValueError):
            raster.draw(symbols, np.zeros((1, 1, 3), dtype=float))
        raster.atlas = raster.atlas.copy()
        raster.atlas[0] = 255
        colors = np.full((1, 1, 3), 73, dtype=np.uint8)
        np.testing.assert_array_equal(raster.draw(symbols, colors), np.full((16, 8, 3), 73))
        # At 1.5x the renderer selects the enlarged font, whose space is empty.
        raster.width, raster.height = 12, 24
        self.assertFalse(raster.draw(symbols, colors).any())

    def test_text_coverage_brightens_edges_without_spreading_or_changing_foreground(self):
        from raster import _TEXT_COVERAGE

        self.assertEqual(int(_TEXT_COVERAGE[0]), 0)
        self.assertEqual(int(_TEXT_COVERAGE[255]), 255)
        self.assertEqual(int(_TEXT_COVERAGE[128]), 188)
        self.assertTrue(np.all(np.diff(_TEXT_COVERAGE.astype(int)) >= 0))
        self.assertTrue(np.all(_TEXT_COVERAGE >= np.arange(256)))
        symbols = np.array([["@", " "]])
        colors = np.array([[[37, 149, 231], [255, 255, 255]]], dtype=np.uint8)
        raster = Rasterizer(" @", "source", 16, 16)
        raster.atlas = raster.atlas.copy()
        raster.atlas[ord("@") - 32, 0, 0] = 255
        original = raster.draw(symbols, colors)
        # Fully covered strokes retain the console's foreground; black and
        # whitespace stay black. Correction belongs to coverage, not RGB.
        full = raster.atlas[ord("@") - 32] == 255
        self.assertTrue(full.any())
        np.testing.assert_array_equal(original[:, :8][full], np.tile(colors[0, 0], (full.sum(), 1)))
        self.assertFalse(original[:, 8:].any())
        self.assertTrue(np.all(original[:, :8] <= colors[0, 0]))

    def test_large_source_strips_match_rgb_lanczos_and_retain_frames(self):
        import av

        rng = np.random.default_rng(117)
        symbols = np.array([chr(code) for code in range(32, 127)], dtype="U1").reshape(5, 19)[::-1, ::-1]
        colors = rng.integers(0, 256, (5, 19, 3), dtype=np.uint8)[::-1, ::-1, ::-1]
        symbols.setflags(write=False)
        colors.setflags(write=False)
        for compose in (raster_module._compose, None):
            with patch.object(raster_module, "_compose", compose):
                for shape, dimensions in (((5, 19), (73, 51)), ((5, 19), (2, 2)),
                                          ((5, 19), (151, 80)), ((1, 19), (151, 16)),
                                          ((5, 1), (8, 79))):
                    with self.subTest(native=compose is not None, shape=shape, size=dimensions):
                        chars = symbols[:shape[0], :shape[1]]
                        rgb = colors[:shape[0], :shape[1]]
                        raster = Rasterizer(" .#@", "source", *dimensions)
                        # Exercise all coverage values, not just font antialiasing.
                        raster.atlas = raster.atlas.copy()
                        raster.atlas[:] = rng.integers(0, 256, raster.atlas.shape, dtype=np.uint8)
                        expected = raster.draw(chars, rgb)
                        raster.SOURCE_RGB_LIMIT = 0
                        actual = raster.draw(chars, rgb)
                        frame = av.VideoFrame.from_ndarray(actual, format="rgb24")
                        raster.draw(chars[::-1], 255 - rgb)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)

    def test_large_source_strips_cross_glyph_row_boundaries(self):
        rng = np.random.default_rng(119)
        symbols = rng.choice(np.array([chr(c) for c in range(32, 127)]), (137, 480))
        colors = rng.integers(0, 256, (137, 480, 3), dtype=np.uint8)
        raster = Rasterizer(" .#@", "source", 1918, 1080)
        expected = raster.draw(symbols, colors)
        raster.SOURCE_RGB_LIMIT = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            raster._resize_pool = pool
            np.testing.assert_array_equal(raster.draw(symbols, colors), expected)

    def test_parallel_channels_preserve_pixels_and_retained_frames(self):
        rng = np.random.default_rng(113)
        symbols = rng.choice(np.array([chr(c) for c in range(32, 127)]), (135, 480))
        colors = rng.integers(0, 256, (135, 480, 3), dtype=np.uint8)
        for compose in (raster_module._compose, None):
            with patch.object(raster_module, "_compose", compose):
                for dimensions in ((1920, 1080), (5000, 2800), (2000, 1200)):
                    with self.subTest(native=compose is not None, dimensions=dimensions):
                        raster = Rasterizer(" .#@", "source", *dimensions)
                        expected = raster.draw(symbols, colors)
                        with ThreadPoolExecutor(max_workers=3) as pool:
                            raster._resize_pool = pool
                            actual = raster.draw(symbols, colors)
                            raster.draw(symbols[::-1], colors[::-1])
                        np.testing.assert_array_equal(actual, expected)

    def test_parallel_resize_skips_small_fixed_and_unscaled_images(self):
        from unittest.mock import Mock
        for mode, shape, dimensions in (("source", (4, 8), (80, 80)),
                                        ("mono", (135, 480), (1920, 1080)),
                                        ("#19ABEF", (135, 480), (1920, 1080)),
                                        ("source", (135, 480), (3840, 2160))):
            with self.subTest(mode=mode, dimensions=dimensions):
                raster = Rasterizer(" .#@", mode, *dimensions)
                raster._resize_pool = Mock()
                raster.draw(np.full(shape, "@"), np.full((*shape, 3), 129, dtype=np.uint8))
                raster._resize_pool.map.assert_not_called()

    @unittest.skipIf(getattr(raster_module, "_compose", None) is None, "optional native module unavailable")
    def test_native_matches_fallback_with_strided_readonly_inputs(self):
        rng = np.random.default_rng(109)
        symbols = np.array([chr(code) for code in range(32, 127)], dtype="U1").reshape(5, 19)[::-1, ::-1]
        colors = rng.integers(0, 256, (5, 19, 3), dtype=np.uint8)[::-1, ::-1, ::-1]
        symbols.setflags(write=False)
        colors.setflags(write=False)
        for dimensions in ((152, 80), (228, 120), (153, 81), (73, 51), (231, 123)):
            for mode in ("mono", "source", "#19ABEF"):
                with self.subTest(dimensions=dimensions, mode=mode):
                    rasterizer = Rasterizer(" .#@", mode, *dimensions)
                    actual = rasterizer.draw(symbols, colors)
                    with patch.object(raster_module, "_compose", None):
                        expected = rasterizer.draw(symbols, colors)
                    np.testing.assert_array_equal(actual, expected)

    @unittest.skipIf(getattr(raster_module, "_compose_planes", None) is None, "optional planar module unavailable")
    def test_planar_composer_rejects_invalid_shapes_and_ascii(self):
        codes = np.array([[64]], dtype=np.uint32)
        colors = np.zeros((1, 1, 3), dtype=np.uint8)
        atlas = Rasterizer(" @", "source", 8, 16).atlas
        for bad_codes, bad_colors, bad_atlas in ((codes, colors[:, :, :2], atlas),
                                                 (codes, colors, atlas[:-1]),
                                                 (np.array([[31]], dtype=np.uint32), colors, atlas),
                                                 (np.array([[127]], dtype=np.uint32), colors, atlas)):
            with self.subTest(codes=bad_codes, colors=bad_colors.shape, atlas=bad_atlas.shape):
                with self.assertRaises(ValueError):
                    raster_module._compose_planes(bad_codes, bad_colors, bad_atlas)

    @unittest.skipIf(getattr(raster_module, "_compose", None) is None, "optional native module unavailable")
    def test_old_native_without_planar_extension_still_renders(self):
        import importlib.util
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location("raster_old_native", raster_module.__file__)
        old = importlib.util.module_from_spec(spec)
        # The old extension exposes only the original three-argument function.
        def old_compose(codes, colors, atlas):
            return raster_module._compose(codes, colors, atlas)
        with patch.dict(sys.modules, {"_glyphs": SimpleNamespace(compose=old_compose)}):
            spec.loader.exec_module(old)
        self.assertIs(old._compose, old_compose)
        self.assertIsNone(old._compose_planes)
        symbols = np.full((90, 320), "@", dtype="U1")
        colors = np.full((90, 320, 3), (37, 149, 231), dtype=np.uint8)
        raster = old.Rasterizer(" @", "source", 1920, 1080)
        expected = raster.draw(symbols, colors)
        with ThreadPoolExecutor(max_workers=3) as pool:
            raster._resize_pool = pool
            np.testing.assert_array_equal(raster.draw(symbols, colors), expected)

    def test_retained_pixels_survive_later_draws_and_pyav_conversion(self):
        import gc
        import av

        symbols = np.array([list(" @"), list("#.")])
        colors = np.array([[[13, 23, 37], [59, 83, 127]],
                           [[149, 173, 191], [211, 233, 251]]], dtype=np.uint8)
        for mode in ("mono", "#00FF88", "source"):
            for dimensions in ((16, 32), (40, 24), (8, 12)):
                with self.subTest(mode=mode, dimensions=dimensions):
                    raster = Rasterizer(" .#@", mode, *dimensions)
                    pixels = raster.draw(symbols, colors)
                    expected = pixels.copy()
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    raster.draw(symbols[::-1, ::-1], 255 - colors)
                    del raster
                    gc.collect()
                    np.testing.assert_array_equal(pixels, expected)
                    del pixels
                    gc.collect()
                    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)

    def test_all_ascii_tiles_match_unclipped_reference(self):
        raster = Rasterizer(" .#@", "mono", 8, 16)
        font = ImageFont.truetype(str(font_path()), 12, layout_engine=ImageFont.Layout.BASIC)
        # Compare every tile with a generous canvas: cropped g/q/y/_/| must fail.
        for code in range(32, 127):
            with self.subTest(character=chr(code)):
                large = Image.new("L", (40, 48))
                ImageDraw.Draw(large).text((raster.origin[0] + 16, raster.origin[1] + 16), chr(code),
                                          font=font, fill=255, anchor="ls")
                reference = np.asarray(large)
                tile = reference[16:32, 16:24]
                self.assertEqual(int(reference.sum()), int(tile.sum()))
                linear = tile.astype(np.float64) / 255
                expected = np.floor(255 * np.where(linear <= .0031308, 12.92 * linear,
                                                  1.055 * linear ** (1 / 2.4) - .055) + .5).astype(np.uint8)
                np.testing.assert_array_equal(raster.atlas[code - 32], expected)
        self.assertFalse(raster.atlas[0].any())

    def test_mono_hex_and_source_colors(self):
        symbols = np.array([["@", "@"]])
        colors = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        mono = Rasterizer(" .#@", "mono", 16, 16).draw(symbols, colors)
        self.assertEqual(mono.dtype, np.uint8)
        np.testing.assert_array_equal(mono[:, :, 0], mono[:, :, 1])
        source = Rasterizer(" .#@", "source", 16, 16).draw(symbols, colors)
        self.assertGreater(int(source[:, :8, 0].sum()), 0)
        self.assertFalse(source[:, :8, 1:].any())
        self.assertGreater(int(source[:, 8:, 1].sum()), 0)
        self.assertFalse(source[:, 8:, (0, 2)].any())
        fixed = Rasterizer(" .#@", "#FF0000", 16, 16).draw(symbols, colors)
        self.assertGreater(int(fixed[:, :, 0].sum()), 0)
        self.assertFalse(fixed[:, :, 1:].any())

    def test_centered_letterbox_and_output_dimensions(self):
        raster = Rasterizer(" .#@", "mono", 32, 16)
        result = raster.draw(np.array([["@"]]), None)
        self.assertEqual(result.shape, (16, 32, 3))
        self.assertFalse(result[:, :12].any())
        self.assertFalse(result[:, 20:].any())
        self.assertTrue(result[:, 12:20].any())
        resized = Rasterizer(" .#@", "mono", 10, 10).draw(np.array([["@"]]), None)
        self.assertEqual(resized.shape, (10, 10, 3))
        self.assertFalse(resized[:, :2].any())
        self.assertFalse(resized[:, 7:].any())

    def test_grid_order_and_blank_frame(self):
        symbols = np.array([["@", " "], [" ", "#"]])
        raster = Rasterizer(" .#@", "mono", 16, 32)
        result = raster.draw(symbols, None)
        np.testing.assert_array_equal(result[:16, :8, 0], raster.atlas[ord("@") - 32])
        self.assertFalse(result[:16, 8:].any())
        self.assertFalse(result[16:, :8].any())
        np.testing.assert_array_equal(result[16:, 8:, 0], raster.atlas[ord("#") - 32])
        self.assertFalse(raster.draw(np.full((2, 2), " "), None).any())

    def test_near_native_upscale_keeps_glyph_pixels_and_centers_padding(self):
        symbols = np.full((50, 100), "@", dtype="U1")
        colors = np.full((50, 100, 3), (37, 149, 231), dtype=np.uint8)
        for mode in ("mono", "source", "#19ABEF"):
            native = Rasterizer("@ ", mode, 800, 800).draw(symbols, colors)
            for size in (801, 808):  # Fractional padding and the inclusive 1% boundary.
                actual = Rasterizer("@ ", mode, size, size).draw(symbols, colors)
                expected = np.zeros((size, size, 3), dtype=np.uint8)
                offset = (size - 800) // 2
                expected[offset:offset + 800, offset:offset + 800] = native
                np.testing.assert_array_equal(actual, expected)
            # Larger enlargement and any reduction retain the Lanczos path.
            for size in (809, 799):
                actual = Rasterizer("@ ", mode, size, size).draw(symbols, colors)
                expected = np.asarray(Image.fromarray(native).resize((size, size), Image.Resampling.LANCZOS))
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_input(self):
        raster = Rasterizer(" .#@", "source", 16, 16)
        with self.assertRaises(ValueError):
            raster.draw(np.array([["░"]]), np.zeros((1, 1, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            raster.draw(np.array([["@"]]), np.zeros((1, 1, 3), dtype=float))
        with self.assertRaises(ValueError):
            Rasterizer(" .#@", "white", 16, 16)

    def test_native_font_matches_unclipped_freetype_and_color_compositing(self):
        symbols = np.array([chr(code) for code in range(32, 127)], dtype="U1").reshape(5, 19)
        colors = np.random.default_rng(51).integers(0, 256, (5, 19, 3), dtype=np.uint8)
        for steps in range(3, 17):  # Every supported half-step from 1.5x through 8x.
            cell_width, cell_height = 4 * steps, 8 * steps
            width, height = 19 * cell_width, 5 * cell_height
            font = ImageFont.truetype(str(font_path()), 6 * steps, layout_engine=ImageFont.Layout.BASIC)
            bounds = [font.getbbox(chr(code), anchor="ls") for code in range(32, 127)]
            left, top = min(b[0] for b in bounds), min(b[1] for b in bounds)
            right, bottom = max(b[2] for b in bounds), max(b[3] for b in bounds)
            origin = ((cell_width - right + left) // 2 - left,
                      (cell_height - bottom + top) // 2 - top)
            large = Image.new("L", (width + 32, height + 32))
            draw = ImageDraw.Draw(large)
            for row in range(5):
                for column in range(19):
                    draw.text((16 + column * cell_width + origin[0],
                               16 + row * cell_height + origin[1]), symbols[row, column],
                              font=font, fill=255, anchor="ls")
            raw = np.asarray(large)[16:16 + height, 16:16 + width]
            self.assertEqual(int(np.asarray(large).sum()), int(raw.sum()))
            linear = raw.astype(np.float64) / 255
            coverage = np.floor(255 * np.where(linear <= .0031308, 12.92 * linear,
                                              1.055 * linear ** (1 / 2.4) - .055) + .5).astype(np.uint8)
            for mode in ("mono", "source", "#19ABEF"):
                with self.subTest(steps=steps, mode=mode):
                    raster = Rasterizer(" .#@", mode, width + 3, height)
                    actual = raster.draw(symbols, colors)
                    foreground = (colors.repeat(cell_height, axis=0).repeat(cell_width, axis=1)
                                  if mode == "source" else np.array(raster.rgb))
                    pixels = ((coverage[..., None].astype(np.uint16) * foreground + 127) // 255).astype(np.uint8)
                    expected = np.zeros((height, width + 3, 3), dtype=np.uint8)
                    expected[:, 1:width + 1] = pixels
                    np.testing.assert_array_equal(actual, expected)
                    cached = raster._scaled_atlas
                    self.assertLess(cached[1].nbytes, 1024 * 1024)
                    raster.draw(symbols, colors)
                    self.assertIs(raster._scaled_atlas, cached)
                    raster.draw(np.repeat(symbols, 2, axis=0), np.repeat(colors, 2, axis=0))
                    if steps >= 6 and steps % 2 == 0:
                        self.assertIsNot(raster._scaled_atlas, cached)
                    else:
                        self.assertIs(raster._scaled_atlas, cached)
                    np.testing.assert_array_equal(actual, expected)

    def test_non_half_step_and_oversized_fonts_retain_lanczos(self):
        symbols = np.array([list("@#%"), list("gj_")])
        for mode in ("mono", "source", "#19ABEF"):
            colors = np.full((2, 3, 3), (37, 149, 231), dtype=np.uint8)
            original = Rasterizer(" .#@", mode, 24, 32).draw(symbols, colors)
            for width, height in ((30, 40), (42, 56), (204, 272), (18, 24)):
                with self.subTest(mode=mode, size=(width, height)):
                    raster = Rasterizer(" .#@", mode, width, height)
                    expected = np.asarray(Image.fromarray(original).resize((width, height), Image.Resampling.LANCZOS))
                    np.testing.assert_array_equal(raster.draw(symbols, colors), expected)
                    self.assertIsNone(raster._scaled_atlas)

    def test_vectorized_coloring_matches_individual_pillow_glyphs(self):
        rng = np.random.default_rng(23)
        symbols = rng.choice([chr(code) for code in range(32, 127)], (3, 7)).astype("U1")[:, ::-1]
        colors = rng.integers(0, 256, (3, 7, 3), dtype=np.uint8)[:, ::-1]
        for mode in ("mono", "source", "#11A3F7"):
            for width, height in ((56, 48), (80, 64), (28, 24), (64, 90)):
                with self.subTest(mode=mode, size=(width, height)):
                    raster = Rasterizer(" .#@", mode, width, height)
                    reference = Image.new("RGB", (56, 48))
                    for row in range(3):
                        for column in range(7):
                            foreground = tuple(int(c) for c in colors[row, column]) if mode == "source" else raster.rgb
                            mask = Image.fromarray(raster.atlas[ord(symbols[row, column]) - 32])
                            glyph = Image.composite(Image.new("RGB", (8, 16), foreground),
                                                    Image.new("RGB", (8, 16)), mask)
                            reference.paste(glyph, (column * 8, row * 16))
                    scale = min(width / 56, height / 48)
                    fitted = (int(56 * scale + .5), int(48 * scale + .5))
                    reference = reference.resize(fitted, Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (width, height))
                    canvas.paste(reference, ((width - fitted[0]) // 2, (height - fitted[1]) // 2))
                    np.testing.assert_array_equal(raster.draw(symbols, colors), np.asarray(canvas))

    def test_fixed_color_lookup_matches_every_coverage_and_channel(self):
        # Synthetic tiles exercise all 256 mask coverages, including rounding
        # values absent from the bundled font, and all 256 foreground levels.
        symbols = np.array([[" ", "!"]], dtype="U1")
        coverage = np.arange(256, dtype=np.uint16).reshape(16, 16)
        for level in range(256):
            raster = Rasterizer(" !", f"#{level:02X}{255-level:02X}A3", 16, 16)
            raster.atlas = raster.atlas.copy()
            raster.atlas[0] = coverage[:, :8]
            raster.atlas[1] = coverage[:, 8:]
            expected = ((coverage[..., None] * np.asarray(raster.rgb, dtype=np.uint16)
                         + 127) // 255).astype(np.uint8)
            np.testing.assert_array_equal(raster.draw(symbols, None), expected)

    def test_mono_resize_matches_rgb_filtering_and_centering(self):
        rng = np.random.default_rng(17)
        symbols = rng.choice([chr(code) for code in range(32, 127)], (5, 11)).astype("U1")[:, ::-1]
        for width, height in ((2, 2), (7, 19), (88, 80), (87, 79), (31, 23), (180, 150), (55, 101)):
            with self.subTest(size=(width, height)):
                raster = Rasterizer(" .#@", "mono", width, height)
                # Synthetic antialias values make this independent of the font.
                raster.atlas = raster.atlas.copy()
                raster.atlas[:] = rng.integers(0, 256, raster.atlas.shape, dtype=np.uint8)
                coverage = raster.atlas[symbols.view(np.uint32) - 32].transpose(0, 2, 1, 3).reshape(80, 88)
                reference = Image.fromarray(np.repeat(coverage[..., None], 3, axis=-1))
                scale = min(width / 88, height / 80)
                fitted = (min(width, max(1, int(88 * scale + .5))),
                          min(height, max(1, int(80 * scale + .5))))
                reference = reference.resize(fitted, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (width, height))
                canvas.paste(reference, ((width - fitted[0]) // 2, (height - fitted[1]) // 2))
                np.testing.assert_array_equal(raster.draw(symbols, None), np.asarray(canvas))

    def test_source_compositing_matches_all_coverage_component_pairs(self):
        raster = Rasterizer(" !", "source", 2048, 32)
        raster.atlas = raster.atlas.copy()
        raster.atlas[0] = np.arange(128, dtype=np.uint8).reshape(16, 8)
        raster.atlas[1] = np.arange(128, 256, dtype=np.uint8).reshape(16, 8)
        symbols = np.array([[" "] * 256, ["!"] * 256], dtype="U1")
        levels = np.arange(256, dtype=np.uint8)
        colors = np.broadcast_to(np.stack((levels, 255 - levels, levels), axis=-1), (2, 256, 3))
        coverage = np.tile(np.arange(256, dtype=np.uint16).reshape(32, 8), (1, 256))
        foreground = colors.repeat(16, axis=0).repeat(8, axis=1).astype(np.uint16)
        expected = ((coverage[..., None] * foreground + 127) // 255).astype(np.uint8)
        np.testing.assert_array_equal(raster.draw(symbols, colors), expected)
        if raster_module._compose_planes is not None:
            for chars, rgb in ((symbols, colors), (symbols[::-1, ::-1], colors[::-1, ::-1, ::-1])):
                chars.setflags(write=False)
                rgb.setflags(write=False)
                codes = chars.view(np.uint32)
                packed = raster_module._compose(codes, rgb, raster.atlas)
                planes = raster_module._compose_planes(codes, rgb, raster.atlas)
                np.testing.assert_array_equal(planes.transpose(1, 2, 0), packed)


if __name__ == "__main__":
    unittest.main()

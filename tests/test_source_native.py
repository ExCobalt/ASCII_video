"""Independent pixel and lifetime contracts for optional native source export."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import raster


AVAILABLE = all(getattr(raster, name, None) is not None for name in
                ('_make_source_tinted_atlas', '_compose_source_tinted', '_resize_source_band'))


def reference_compose(codes, colors, atlas):
    rows, columns = codes.shape
    _, height, width = atlas.shape
    coverage = atlas[codes - 32].astype(np.uint16)
    pixels = ((coverage[..., None] * colors[:, :, None, None, :].astype(np.uint16) + 127) // 255).astype(np.uint8)
    return pixels.transpose(4, 0, 2, 1, 3).reshape(3, rows * height, columns * width)


def make_raster(module=raster, fast=False):
    result = module.Rasterizer(' .#@', 'source', 1920, 1080)
    result._source_parallel_strips = True
    result._source_fast_raster = fast
    return result


@unittest.skipUnless(AVAILABLE, 'optional source kernels unavailable')
class SourceNativeTests(unittest.TestCase):
    def test_all_coverage_color_pairs_and_strided_readonly_composition(self):
        atlas = np.zeros((95, 16, 8), dtype=np.uint8)
        atlas[:2] = np.arange(256, dtype=np.uint8).reshape(2, 16, 8)
        codes = np.array([[32] * 256, [33] * 256], dtype=np.uint32)
        values = np.arange(256, dtype=np.uint8)
        colors = np.broadcast_to(np.stack((values, 255-values, values), axis=-1), (2, 256, 3))
        atlas.flags.writeable = codes.flags.writeable = False
        tinted = raster._make_source_tinted_atlas(atlas)
        self.assertFalse(tinted.flags.writeable)
        for chars, rgb in ((codes, colors), (codes[::-1, ::-1], colors[::-1, ::-1, ::-1])):
            expected = reference_compose(chars, rgb, atlas)
            actual = raster._compose_source_tinted(chars, rgb, tinted)
            np.testing.assert_array_equal(actual, expected)
            raster._compose_source_tinted(chars, np.zeros(rgb.shape, dtype=np.uint8), tinted)
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.shares_memory(actual, tinted))

    @unittest.skipUnless(raster._PILLOW_VERSION in ('12.3.0', '13.0.0'), 'native filter targets Pillow 12.3.0/13.0.0')
    def test_four_bands_equal_whole_pillow_random_sparse_edges_and_strides(self):
        rng = np.random.default_rng(131142)
        random = rng.integers(0, 256, (1440, 2560), dtype=np.uint8)
        sparse = np.zeros_like(random)
        sparse[::17, ::23] = 255
        edges = np.zeros_like(random)
        for boundary in (0, 360, 720, 1080, 1439):
            edges[max(0, boundary-4):min(1440, boundary+5):2] = 255
        edges[:, :5:2] = edges[:, -5::2] = 255
        held = saved = None
        with ThreadPoolExecutor(max_workers=4) as pool:
            for plane in (random, sparse, edges, random[::-1, ::-1]):
                plane.flags.writeable = False
                expected = np.asarray(Image.fromarray(plane).resize((1920, 1080), Image.Resampling.LANCZOS))
                bands = list(pool.map(lambda top: raster._resize_source_band(plane, top), (0, 270, 540, 810)))
                np.testing.assert_array_equal(np.concatenate(bands), expected)
                self.assertTrue(all(band.shape == (270, 1920) for band in bands))
                self.assertFalse(np.shares_memory(bands[0], bands[1]))
                if held is None:
                    held, saved = bands[0], bands[0].copy()
                else:
                    np.testing.assert_array_equal(held, saved)

    @unittest.skipUnless(raster._PILLOW_VERSION in ('12.3.0', '13.0.0'), 'native filter targets Pillow 12.3.0/13.0.0')
    def test_full_raster_fallback_parity_cache_and_retained_frames(self):
        rng = np.random.default_rng(131143)
        symbols = rng.choice(list(' .:-=+*#%@'), (90, 320)).astype('U1')[::-1, ::-1]
        colors = rng.integers(0, 256, (90, 320, 3), dtype=np.uint8)[::-1, ::-1, ::-1]
        symbols.flags.writeable = colors.flags.writeable = False
        fast, fallback = make_raster(fast=True), make_raster()
        with ThreadPoolExecutor(max_workers=12) as pool:
            fast._resize_pool = fallback._resize_pool = pool
            held = fast.draw(symbols, colors)
            expected = fallback.draw(symbols, colors)
            np.testing.assert_array_equal(held, expected)
            cached = fast._source_tinted_atlas
            self.assertIsNotNone(cached)
            fast.draw(symbols, np.zeros(colors.shape, dtype=np.uint8))
            self.assertIs(fast._source_tinted_atlas, cached)
            np.testing.assert_array_equal(held, expected)

    @unittest.skipUnless(raster._PILLOW_VERSION in ('12.3.0', '13.0.0'), 'native filter targets Pillow 12.3.0/13.0.0')
    def test_mutable_atlas_rebuild_then_readonly_cache_transition(self):
        fast, fallback = make_raster(fast=True), make_raster()
        atlas = fast.atlas.copy()
        atlas.flags.writeable = False
        fast.atlas = fallback.atlas = atlas
        symbols = np.full((90, 320), 'A', dtype='U1')
        colors = np.full((90, 320, 3), (33, 147, 250), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=12) as pool:
            fast._resize_pool = fallback._resize_pool = pool
            held = fast.draw(symbols, colors)
            saved = held.copy()
            old_cache = fast._source_tinted_atlas
            old_table = old_cache[1].copy()
            atlas.flags.writeable = True
            atlas[:] = 255 - atlas
            np.testing.assert_array_equal(fast.draw(symbols, colors), fallback.draw(symbols, colors))
            self.assertIsNone(fast._source_tinted_atlas)
            np.testing.assert_array_equal(old_cache[1], old_table)
            np.testing.assert_array_equal(held, saved)
            atlas.flags.writeable = False
            np.testing.assert_array_equal(fast.draw(symbols, colors), fallback.draw(symbols, colors))
            new_cache = fast._source_tinted_atlas
            self.assertIsNot(new_cache, old_cache)
            self.assertIs(new_cache[0], atlas)
            fast.draw(symbols, colors)
            self.assertIs(fast._source_tinted_atlas, new_cache)

    def test_old_extension_and_other_pillow_version_use_exact_fallback(self):
        spec = importlib.util.spec_from_file_location('raster_without_source_native', raster.__file__)
        previous = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_glyphs': SimpleNamespace(compose=raster._compose,
                compose_planes=raster._compose_planes, compose_planes_sparse=raster._compose_planes_sparse)}):
            spec.loader.exec_module(previous)
        self.assertIsNone(previous._make_source_tinted_atlas)
        self.assertIsNone(previous._compose_source_tinted)
        self.assertIsNone(previous._resize_source_band)
        symbols = np.full((90, 320), '@', dtype='U1')
        colors = np.full((90, 320, 3), (241, 96, 37), dtype=np.uint8)
        baseline, old, unsupported = make_raster(), make_raster(previous, True), make_raster(fast=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            baseline._resize_pool = old._resize_pool = unsupported._resize_pool = pool
            expected = baseline.draw(symbols, colors)
            np.testing.assert_array_equal(old.draw(symbols, colors), expected)
            with patch.object(raster, '_PILLOW_VERSION', 'unsupported'), \
                    patch.object(raster, '_make_source_tinted_atlas', side_effect=AssertionError('native called')), \
                    patch.object(raster, '_resize_source_band', side_effect=AssertionError('native called')):
                np.testing.assert_array_equal(unsupported.draw(symbols, colors), expected)
            self.assertIsNone(unsupported._source_tinted_atlas)

    def test_invalid_band_shape_top_and_composition_inputs(self):
        plane = np.zeros((1440, 2560), dtype=np.uint8)
        for top in (-270, 1, 269, 1080):
            with self.subTest(top=top), self.assertRaises(ValueError):
                raster._resize_source_band(plane, top)
        for shape in ((1439, 2560), (1440, 2559), (0, 2560)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                raster._resize_source_band(np.zeros(shape, dtype=np.uint8), 0)
        atlas = np.zeros((95, 16, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            raster._make_source_tinted_atlas(atlas[:-1])
        tinted = raster._make_source_tinted_atlas(atlas)
        for code in (31, 127):
            with self.assertRaises(ValueError):
                raster._compose_source_tinted(np.array([[code]], dtype=np.uint32),
                    np.zeros((1, 1, 3), dtype=np.uint8), tinted)
        with self.assertRaises(ValueError):
            raster._compose_source_tinted(np.array([[32]], dtype=np.uint32),
                np.zeros((1, 1, 2), dtype=np.uint8), tinted)

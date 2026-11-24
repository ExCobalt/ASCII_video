"""Independent equality and safety checks for optional sparse composition."""
import unittest
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import raster


@unittest.skipIf(getattr(raster, '_compose_planes_sparse', None) is None, 'optional sparse kernel unavailable')
class SparsePlaneTests(unittest.TestCase):
    def test_old_extension_with_planes_but_without_sparse_keeps_exact_export(self):
        spec = importlib.util.spec_from_file_location('raster_previous_extension', raster.__file__)
        previous = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_glyphs': SimpleNamespace(
                compose=raster._compose, compose_planes=raster._compose_planes)}):
            spec.loader.exec_module(previous)
        self.assertIsNone(previous._compose_planes_sparse)
        rng = np.random.default_rng(12)
        symbols = rng.choice(list(' .:-=+*#%@'), (90, 320)).astype('U1')
        colors = rng.integers(0, 256, (90, 320, 3), dtype=np.uint8)
        old = previous.Rasterizer(' .#@', 'source', 1920, 1080)
        new = raster.Rasterizer(' .#@', 'source', 1920, 1080)
        old._source_parallel_strips = new._source_parallel_strips = True
        with ThreadPoolExecutor(max_workers=12) as pool:
            old._resize_pool = new._resize_pool = pool
            np.testing.assert_array_equal(old.draw(symbols, colors), new.draw(symbols, colors))

    def test_all_coverage_and_foreground_pairs(self):
        atlas = raster.Rasterizer(' @', 'source', 16, 16).atlas.copy()
        atlas[0] = np.arange(128, dtype=np.uint8).reshape(16, 8)
        atlas[1] = np.arange(128, 256, dtype=np.uint8).reshape(16, 8)
        codes = np.array([[32] * 256, [33] * 256], dtype=np.uint32)
        values = np.arange(256, dtype=np.uint8)
        colors = np.broadcast_to(np.stack((values, 255-values, values), axis=-1), (2, 256, 3))
        coverage = np.tile(np.arange(256, dtype=np.uint16).reshape(32, 8), (1, 256))
        foreground = colors.repeat(16, axis=0).repeat(8, axis=1).astype(np.uint16)
        expected = ((coverage[..., None] * foreground + 127) // 255).astype(np.uint8)
        actual = raster._compose_planes_sparse(codes, colors, atlas)
        np.testing.assert_array_equal(actual.transpose(1, 2, 0), expected)

    def test_strides_readonly_zero_masks_and_retained_outputs(self):
        rng = np.random.default_rng(131128)
        codes = rng.integers(32, 127, (13, 29), dtype=np.uint32)[::-1, ::-1]
        colors = rng.integers(0, 256, (13, 29, 3), dtype=np.uint8)[::-1, ::-1, ::-1]
        atlas = rng.integers(0, 256, (95, 7, 5), dtype=np.uint8)
        atlas[:, ::2, ::2] = 0
        codes.flags.writeable = colors.flags.writeable = atlas.flags.writeable = False
        expected = raster._compose_planes(codes, colors, atlas)
        actual = raster._compose_planes_sparse(codes, colors, atlas)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertFalse(np.shares_memory(actual, colors))
        raster._compose_planes_sparse(codes, np.zeros(colors.shape, dtype=np.uint8), atlas)
        np.testing.assert_array_equal(actual, expected)

    def test_empty_dimensions_and_invalid_inputs(self):
        atlas = np.zeros((95, 16, 8), dtype=np.uint8)
        for rows, columns in ((0, 4), (4, 0), (0, 0)):
            result = raster._compose_planes_sparse(np.full((rows, columns), 32, dtype=np.uint32),
                np.zeros((rows, columns, 3), dtype=np.uint8), atlas)
            self.assertEqual(result.shape, (3, rows*16, columns*8))
        for code in (0, 31, 127, 2**32-1):
            with self.assertRaises(ValueError):
                raster._compose_planes_sparse(np.array([[code]], dtype=np.uint32),
                    np.zeros((1, 1, 3), dtype=np.uint8), atlas)
        for colors, masks in ((np.zeros((1, 1, 2), dtype=np.uint8), atlas),
                              (np.zeros((1, 1, 3), dtype=np.uint8), atlas[:-1])):
            with self.assertRaises(ValueError):
                raster._compose_planes_sparse(np.array([[32]], dtype=np.uint32), colors, masks)


if __name__ == '__main__':
    unittest.main()

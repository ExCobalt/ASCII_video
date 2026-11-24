"""Owned GBRP snapshots, raster layout and production opt-in contracts."""
from concurrent.futures import ThreadPoolExecutor
import gc
import unittest

import numpy as np

import export
import raster


def planar_image(readonly=True):
    owner = np.empty((3, 1080, 1920), dtype=np.uint8)
    owner[0], owner[1], owner[2] = 17, 83, 219
    owner[:, 0, 0] = (253, 7, 129)
    owner[:, -1, -1] = (2, 241, 51)
    owner.flags.writeable = not readonly
    return owner, owner.transpose(1, 2, 0)


class PlanarExportTests(unittest.TestCase):
    def test_owned_gbrp_channel_order_snapshot_and_gc_lifetime(self):
        owner, image = planar_image()
        expected = image.copy()
        frame = export._rgb_frame(image, planar=True)
        self.assertEqual(frame.format.name, 'gbrp')
        for plane, channel in zip(frame.planes, (1, 2, 0), strict=True):
            pixels = np.ndarray((1080, 1920), dtype=np.uint8, buffer=plane,
                                strides=(plane.line_size, 1))
            np.testing.assert_array_equal(pixels, owner[channel])
            self.assertFalse(np.shares_memory(pixels, owner))
        # Deliberate external mutation cannot modify the encoder's snapshot.
        owner.flags.writeable = True
        owner.fill(0)
        del owner, image
        gc.collect()
        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)
        _, other = planar_image()
        another_frame = export._rgb_frame(other, planar=True)
        self.assertTrue(all(a.buffer_ptr != b.buffer_ptr for a, b in zip(frame.planes, another_frame.planes)))
        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

    def test_writable_planar_layout_keeps_original_rgb_snapshot(self):
        owner, image = planar_image(readonly=False)
        expected = image.copy()
        frame = export._rgb_frame(image, planar=True)
        self.assertEqual(frame.format.name, 'rgb24')
        owner.fill(0)
        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

    def test_readonly_interleaved_and_negative_stride_views_fall_back(self):
        _, planar = planar_image()
        interleaved = np.ascontiguousarray(planar)
        interleaved.flags.writeable = False
        for image in (interleaved, planar[::-1], planar[:, ::-1], planar[:, :, ::-1]):
            with self.subTest(strides=image.strides):
                frame = export._rgb_frame(image, planar=True)
                self.assertEqual(frame.format.name, 'rgb24')
                np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), image)

    def test_default_keeps_rgb24_snapshot_even_for_exact_planar_layout(self):
        owner, image = planar_image()
        expected = image.copy()
        frame = export._rgb_frame(image)
        self.assertEqual(frame.format.name, 'rgb24')
        owner.flags.writeable = True
        owner.fill(0)
        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

    @unittest.skipIf(raster._compose is None or raster._compose_planes is None,
                     'optional planar composition unavailable')
    def test_raster_chw_exact_rgb_pixels_and_retained_output(self):
        rng = np.random.default_rng(131151)
        symbols = rng.choice(list(' .:-=+*#%@'), (90, 320)).astype('U1')
        colors = rng.integers(0, 256, (90, 320, 3), dtype=np.uint8)
        hwc = raster.Rasterizer(' .#@', 'source', 1920, 1080)
        chw = raster.Rasterizer(' .#@', 'source', 1920, 1080)
        self.assertFalse(hwc._source_planar_export)
        hwc._source_parallel_strips = chw._source_parallel_strips = True
        chw._source_planar_export = True
        fast_modes = [False]
        if raster._make_source_tinted_atlas is not None and raster._PILLOW_VERSION == '12.3.0':
            fast_modes.append(True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            hwc._resize_pool = chw._resize_pool = pool
            for fast in fast_modes:
                with self.subTest(native_resize=fast):
                    hwc._source_fast_raster = chw._source_fast_raster = fast
                    expected = hwc.draw(symbols, colors)
                    held = chw.draw(symbols, colors)
                    self.assertEqual(held.strides, (1920, 1, 1080 * 1920))
                    self.assertFalse(held.flags.writeable)
                    self.assertFalse(held.base.flags.writeable)
                    self.assertTrue(held.base.flags.owndata)
                    np.testing.assert_array_equal(held, expected)
                    self.assertEqual(held.tobytes(), expected.tobytes())
                    frame = export._rgb_frame(held, planar=True)
                    self.assertEqual(frame.format.name, 'gbrp')
                    chw.draw(symbols, np.zeros_like(colors))
                    np.testing.assert_array_equal(held, expected)
                    np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

"""Owned-frame candidates: logical RGB, lifetime, retention and applicability."""
from concurrent.futures import ThreadPoolExecutor
import gc
import inspect
import textwrap
import unittest
from unittest.mock import patch

import av
import numpy as np

import export
import raster


class _PreparedRgb(np.ndarray):
    """RGB raster retaining an independently owned prepared PyAV snapshot."""


class _OwnedGbrp:
    def __init__(self, frame):
        self.frame = frame
        self.shape = (frame.height, frame.width, 3)
        self.ndim = 3
        self.dtype = np.dtype(np.uint8)
        self.rgb_planes = tuple(np.ndarray(self.shape[:2], dtype=np.uint8,
            buffer=frame.planes[index], strides=(frame.planes[index].line_size, 1))
            for index in (2, 0, 1))
        # Internal kernel adapters, NOT logical RGB: every channel index aliases
        # one selected physical plane. Each task writes only its given channel.
        self.kernel_destinations = tuple(np.ndarray(self.shape, dtype=np.uint8,
            buffer=frame.planes[index], strides=(frame.planes[index].line_size, 1, 0))
            for index in (2, 0, 1))

    def seal(self):
        for plane in (*self.rgb_planes, *self.kernel_destinations):
            plane.flags.writeable = False

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('Logical RGB requires a copy from GBRP planes')
        pixels = np.empty(self.shape, dtype=np.uint8)
        for channel, plane in enumerate(self.rgb_planes):
            pixels[:, :, channel] = plane
        return pixels if dtype is None else pixels.astype(dtype, copy=False)

    def tobytes(self, order='C'):
        return self.__array__().tobytes(order=order)


def apply_variant(export_module, variant):
    if variant not in ('prepare-gbrp', 'direct-gbrp'):
        raise ValueError(variant)
    e = export_module
    activated = e._owned_frame_activation = {variant: 0}
    original_rgb = e._rgb_frame

    if variant == 'prepare-gbrp':
        original_draw = e.Rasterizer.draw

        def draw(self, *args, **kwargs):
            pixels = original_draw(self, *args, **kwargs)
            if self.color != 'source' or pixels.shape != (1080, 1920, 3):
                return pixels
            prepared = original_rgb(pixels, planar=self._source_planar_export)
            tagged = pixels.view(_PreparedRgb)
            tagged._export_frame = prepared
            activated[variant] += 1
            return tagged

        def rgb_frame(image, **kwargs):
            prepared = getattr(image, '_export_frame', None) if isinstance(image, _PreparedRgb) else None
            return prepared if prepared is not None else original_rgb(image, **kwargs)

        e.Rasterizer.draw, e._rgb_frame = draw, rgb_frame
        return

    source = textwrap.dedent(inspect.getsource(e.Rasterizer.draw))
    old_allocation = ('shape = (3, 1080, 1920) if self._source_planar_export else (1080, 1920, 3)\n'
                      '                pixels = np.empty(shape, dtype=np.uint8)')
    new_allocation = ('if self._source_planar_export:\n'
                      '                    pixels = _round3_allocate_gbrp()\n'
                      '                else:\n'
                      '                    pixels = np.empty((1080, 1920, 3), dtype=np.uint8)')
    old_return = ('if self._source_planar_export:\n'
                  '                    pixels.flags.writeable = False\n'
                  '                    return pixels.transpose(1, 2, 0)')
    new_return = ('if self._source_planar_export:\n'
                  '                    pixels.seal()\n'
                  '                    return pixels')
    hooks = ((old_allocation, new_allocation),
             ('pixels[channel, top:top + 270, :] = strip',
              'pixels.rgb_planes[channel][top:top + 270, :] = strip'),
             (old_return, new_return))
    if '_source_direct_resize' in source:
        # Newer baselines already resize directly into their final storage.
        # Preserve that path, substituting only the owned GBRP destination;
        # frozen round-three sources have no such branch and keep the hooks above.
        if 'destination = pixels.transpose(1, 2, 0)' in source:
            hooks += (('destination = pixels.transpose(1, 2, 0) if self._source_planar_export else pixels',
                       'destination = pixels.kernel_destinations[channel] if self._source_planar_export else pixels'),)
        else:
            hooks += (('destination = (pixels[channel, top:top + 270, :] if self._source_planar_export',
                       'destination = (pixels.rgb_planes[channel][top:top + 270, :] if self._source_planar_export'),)
    for old, new in hooks:
        if source.count(old) != 1:
            raise RuntimeError('Expected current production planar strip destination')
        source = source.replace(old, new)

    def allocate():
        activated[variant] += 1
        return _OwnedGbrp(e.av.VideoFrame(1920, 1080, 'gbrp'))

    def rgb_frame(image, **kwargs):
        if isinstance(image, _OwnedGbrp):
            return image.frame
        return original_rgb(image, **kwargs)

    raster._round3_allocate_gbrp = allocate
    namespace = {}
    exec(compile(source, raster.__file__, 'exec'), raster.__dict__, namespace)
    e.Rasterizer.draw = namespace['draw']
    e._rgb_frame = rgb_frame


class OwnedFrameCandidateTests(unittest.TestCase):
    def test_gbrp_wrapper_is_explicitly_not_an_ndarray_and_materializes_rgb(self):
        wrapped = _OwnedGbrp(av.VideoFrame(18, 10, 'gbrp'))
        expected = np.empty((10, 18, 3), dtype=np.uint8)
        for channel, value in enumerate((17, 83, 219)):
            wrapped.rgb_planes[channel][:] = expected[:, :, channel] = value
        wrapped.seal()
        self.assertNotIsInstance(wrapped, np.ndarray)
        self.assertEqual(wrapped.shape, expected.shape)
        self.assertTrue(all(not p.flags.writeable for p in wrapped.rgb_planes))
        self.assertTrue(all(p.shape == expected.shape and p.strides[-1] == 0
                            and not p.flags.writeable for p in wrapped.kernel_destinations))
        np.testing.assert_array_equal(np.asarray(wrapped), expected)
        self.assertEqual(wrapped.tobytes(), expected.tobytes())
        with self.assertRaises(ValueError):
            wrapped.__array__(copy=False)
        inspection = np.asarray(wrapped)
        inspection.fill(0)
        np.testing.assert_array_equal(np.asarray(wrapped), expected)
        frame = wrapped.frame
        del wrapped
        gc.collect()
        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

    @unittest.skipIf(raster._compose is None or raster._compose_planes is None,
                     'optional planar composition unavailable')
    def test_actual_raster_pixels_held_frames_and_nonplanar_format_preserved(self):
        rng = np.random.default_rng(131165)
        symbols = rng.choice(list(' .:-=+*#%@'), (90, 320)).astype('U1')
        colors = rng.integers(0, 256, (90, 320, 3), dtype=np.uint8)
        original_draw, original_rgb = raster.Rasterizer.draw, export._rgb_frame

        def make(pool, planar, direct=False):
            result = raster.Rasterizer(' .#@', 'source', 1920, 1080)
            result._source_parallel_strips = True
            result._source_fast_raster = True
            result._source_planar_export = planar
            result._source_direct_resize = direct
            result._resize_pool = pool
            return result

        with ThreadPoolExecutor(max_workers=12) as pool:
            expected = make(pool, False).draw(symbols, colors)
            layouts = [(False, False), (True, False)]
            direct_kernel = getattr(raster, '_resize_source_band_into', None)
            if direct_kernel is not None and raster._PILLOW_VERSION == '12.3.0':
                layouts += [(False, True), (True, True)]
            for variant in ('prepare-gbrp', 'direct-gbrp'):
                for planar, direct in layouts:
                    with self.subTest(variant=variant, planar=planar, direct_resize=direct), \
                            patch.object(raster.Rasterizer, 'draw', original_draw), \
                            patch.object(export, '_rgb_frame', original_rgb), \
                            patch.object(raster, '_resize_source_band_into', wraps=direct_kernel, create=True) as writes:
                        apply_variant(export, variant)
                        candidate = make(pool, planar, direct)
                        image = candidate.draw(symbols, colors)
                        self.assertEqual(writes.call_count, 12 if direct else 0)
                        np.testing.assert_array_equal(np.asarray(image), expected)
                        self.assertEqual(image.tobytes(), expected.tobytes())
                        frame = export._rgb_frame(image, planar=planar)
                        self.assertEqual(frame.format.name, 'gbrp' if planar else 'rgb24')
                        active = export._owned_frame_activation[variant]
                        if variant == 'direct-gbrp' and not planar:
                            self.assertEqual(active, 0)  # VP9 remains inapplicable.
                            self.assertIsInstance(image, np.ndarray)
                        else:
                            self.assertGreater(active, 0)
                            self.assertIs(export._rgb_frame(image, planar=planar), frame)
                        candidate.draw(symbols, np.zeros_like(colors))
                        np.testing.assert_array_equal(np.asarray(image), expected)
                        del candidate, image
                        gc.collect()
                        np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), expected)

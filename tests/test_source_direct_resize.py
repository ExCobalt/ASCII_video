"""Direct native output regions, old-extension fallback and export selection."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import export
import raster


@unittest.skipIf(raster._resize_source_band_into is None, 'optional direct resize unavailable')
class DirectResizeTests(unittest.TestCase):
    @unittest.skipUnless(raster._PILLOW_VERSION in ('12.3.0', '13.0.0'), 'native filter targets Pillow 12.3.0/13.0.0')
    def test_contiguous_interleaved_padded_and_reversed_destinations(self):
        rng = np.random.default_rng(131171)
        plane = rng.integers(0, 256, (1440, 2560), dtype=np.uint8)[::-1, ::-1]
        plane.flags.writeable = False
        expected = np.asarray(Image.fromarray(plane).resize((1920, 1080), Image.Resampling.LANCZOS))[540:810]
        retained = []
        for layout in ('planar', 'interleaved', 'padded', 'reversed'):
            with self.subTest(layout=layout):
                shape = ((3, 1080, 1920) if layout == 'planar' else
                         (1082, 1927, 3) if layout == 'padded' else (1080, 1920, 3))
                owner = np.full(shape, 171, dtype=np.uint8)
                expected_owner = owner.copy()
                if layout == 'planar':
                    destination = owner.transpose(1, 2, 0)
                    expected_destination = expected_owner.transpose(1, 2, 0)
                else:
                    key = ((slice(1, 1081), slice(3, 1923), slice(None)) if layout == 'padded' else
                           (slice(None, None, -1),) * 3 if layout == 'reversed' else
                           (slice(None),) * 3)
                    destination = owner[key]
                    expected_destination = expected_owner[key]
                expected_destination[540:810, :, 1] = expected
                self.assertIsNone(raster._resize_source_band_into(plane, 540, destination, 1))
                np.testing.assert_array_equal(owner, expected_owner)  # Includes untouched padding/channels.
                retained.append((owner, expected_owner))
        raster._resize_source_band(plane, 0)
        for owner, expected_owner in retained:
            np.testing.assert_array_equal(owner, expected_owner)

    def test_invalid_destination_and_band_are_rejected_before_writes(self):
        plane = np.zeros((1440, 2560), dtype=np.uint8)
        readonly = np.zeros((1080, 1920, 3), dtype=np.uint8)
        readonly.flags.writeable = False
        for destination in (readonly, np.zeros((1, 1920, 3), dtype=np.uint8),
                            np.zeros((1080, 1, 3), dtype=np.uint8),
                            np.zeros((1080, 1920, 2), dtype=np.uint8),
                            np.zeros((1080, 1920, 3), dtype=np.float32)):
            with self.subTest(shape=destination.shape, dtype=destination.dtype), self.assertRaises(ValueError):
                raster._resize_source_band_into(plane, 0, destination, 0)
        destination = np.full((1080, 1920, 3), 171, dtype=np.uint8)
        for source, top, channel in ((plane, -1, 0), (plane, 1080, 0), (plane[:-1], 0, 0),
                                    (plane, 0, -1), (plane, 0, 3)):
            with self.assertRaises(ValueError):
                raster._resize_source_band_into(source, top, destination, channel)
            self.assertTrue(np.all(destination == 171))

    @unittest.skipUnless(raster._PILLOW_VERSION in ('12.3.0', '13.0.0'), 'native filter targets Pillow 12.3.0/13.0.0')
    def test_full_raster_direct_regions_and_retained_chw_hwc_outputs(self):
        rng = np.random.default_rng(131172)
        symbols = rng.choice(list(' .:-=+*#%@'), (90, 320)).astype('U1')
        colors = rng.integers(0, 256, (90, 320, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=12) as pool:
            for planar in (False, True):
                baseline = raster.Rasterizer(' .#@', 'source', 1920, 1080)
                direct = raster.Rasterizer(' .#@', 'source', 1920, 1080)
                for instance in (baseline, direct):
                    instance._source_parallel_strips = instance._source_fast_raster = True
                    instance._source_planar_export = planar
                    instance._resize_pool = pool
                direct._source_direct_resize = True
                expected = baseline.draw(symbols, colors)
                with patch.object(raster, '_resize_source_band_into', wraps=raster._resize_source_band_into) as called:
                    held = direct.draw(symbols, colors)
                self.assertEqual(called.call_count, 12)
                np.testing.assert_array_equal(held, expected)
                direct.draw(symbols, np.zeros_like(colors))
                np.testing.assert_array_equal(held, expected)

    def test_previous_native_extension_and_other_pillow_keep_fallback(self):
        spec = importlib.util.spec_from_file_location('raster_without_direct_resize', raster.__file__)
        previous = importlib.util.module_from_spec(spec)
        old_extension = SimpleNamespace(compose=raster._compose, compose_planes=raster._compose_planes,
            compose_planes_sparse=raster._compose_planes_sparse,
            make_source_tinted_atlas=raster._make_source_tinted_atlas,
            compose_source_tinted=raster._compose_source_tinted, resize_source_band=raster._resize_source_band)
        with patch.dict(sys.modules, {'_glyphs': old_extension}):
            spec.loader.exec_module(previous)
        self.assertIsNone(previous._resize_source_band_into)
        self.assertIs(previous._resize_source_band, raster._resize_source_band)
        symbols = np.full((90, 320), '@', dtype='U1')
        colors = np.full((90, 320, 3), (93, 201, 17), dtype=np.uint8)
        baseline = raster.Rasterizer(' .#@', 'source', 1920, 1080)
        old = previous.Rasterizer(' .#@', 'source', 1920, 1080)
        with ThreadPoolExecutor(max_workers=12) as pool:
            for instance in (baseline, old):
                instance._source_parallel_strips = instance._source_fast_raster = True
                instance._source_planar_export = instance._source_direct_resize = True
                instance._resize_pool = pool
            expected = baseline.draw(symbols, colors)
            np.testing.assert_array_equal(old.draw(symbols, colors), expected)
            with patch.object(raster, '_PILLOW_VERSION', 'unsupported'), \
                    patch.object(raster, '_resize_source_band_into', side_effect=AssertionError('direct called')):
                np.testing.assert_array_equal(baseline.draw(symbols, colors), expected)


class DirectResizeSelectionTests(unittest.TestCase):
    def test_only_measured_nvenc_geometry_enables_direct_resize(self):
        info = SimpleNamespace(aspect=16 / 9, audio_index=None, duration=1 / 30)
        with tempfile.TemporaryDirectory() as directory:
            options = SimpleNamespace(input=Path(directory) / 'in.mp4', output=Path(directory) / 'out.mp4',
                overwrite=True, resolution=(1920, 1080), columns=320, palette=' .#@', gamma=1,
                contrast=1, dither='none', color='source', no_audio=True, container='mp4')
            cases = (('h264', 'nvenc', 24, 320, True), ('h264', 'x264', 24, 320, False),
                     ('hevc', 'x264', 24, 320, False), ('av1', 'x264', 24, 320, False),
                     ('h264', 'nvenc', 8, 320, False), ('h264', 'nvenc', 24, 240, False))
            for codec, encoder, cpus, columns, expected in cases:
                options.codec, options.encoder, options.columns = codec, encoder, columns
                with self.subTest(codec=codec, encoder=encoder, cpus=cpus, columns=columns), \
                        patch('export.probe', return_value=info), patch('export.av.Codec'), \
                        patch('export.os.cpu_count', return_value=cpus), patch('export.Renderer'), \
                        patch('export.Rasterizer') as selected, \
                        patch('export._prepared_videos', side_effect=RuntimeError('selection captured')):
                    with self.assertRaisesRegex(RuntimeError, 'selection captured'):
                        export.render(options, progress=lambda *_: None)
                    self.assertIs(selected.return_value._source_direct_resize, expected)

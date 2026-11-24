from contextlib import closing
from pathlib import Path
import queue
import os
import tempfile
import threading
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageOps

from export import _prepared_videos
from gui import Application, PreviewWindow, build_arguments
from media import frame_at, probe
from preview import FrameSaveJob, PreviewJob, make_preview, save_preview_frame
from raster import Rasterizer
from rendering import Renderer
from test_export import fixture
from test_rotation_integration import make_rotated_video


class PreviewTests(unittest.TestCase):


    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "кадр с пробелами.mp4"
        fixture(self.source)
        _, self.options = build_arguments("preview", {"input": str(self.source),
            "output_width": 128, "output_height": 64, "columns": 16, "color": "source"})

    def test_audio_timeline_matches_export_and_switch_invalidates_source(self):
        fixture(self.source, video_pts=list(range(6, 18)), audio_offset=0, audio_samples=48000)
        cache = {}
        for audio in (True, False, True):
            _, options = build_arguments("preview", {"input": str(self.source), "audio": audio,
                "output_width": 128, "output_height": 64, "columns": 16,
                "palette": "@.", "brightness": .5, "color": "source"})
            self.assertEqual(options.no_audio, not audio)
            info = probe(self.source, audio=audio)
            for fraction in (.1, 0, .5, 1):
                result = make_preview(options, fraction, bounds=(128, 64), cache=cache)
                self.assertEqual(result.duration, info.duration)
                self.assertEqual(result.position, fraction * info.duration)
                expected = np.zeros((64, 128, 3), dtype=np.uint8)
                renderer = Renderer(options.palette, brightness=.5)
                raster = Rasterizer(options.palette, "source", 128, 64)
                with closing(_prepared_videos(info, renderer, raster, result.columns, result.rows,
                                              "source", None)) as frames:
                    for image, start, end in frames:
                        if start > result.position:
                            break
                        expected = image
                np.testing.assert_array_equal(result.converted, expected)
                if audio and fraction <= .1:
                    self.assertFalse(np.asarray(result.original).any())
                    self.assertFalse(np.asarray(result.converted).any())
                    with patch("preview.frame_at", side_effect=AssertionError("Black frame decoded again")):
                        for zoom in (1, 2):
                            detail = make_preview(options, fraction, bounds=(128, 64), zoom=zoom,
                                                  center=(1, 0), cache=cache)
                            self.assertFalse(np.asarray(detail.original).any())
                            self.assertFalse(np.asarray(detail.converted).any())
                else:
                    self.assertTrue(np.asarray(result.original).any())
            # Reuse exactly the same position when the audio setting changes.
            make_preview(options, .1, cache=cache)
        self.assertEqual(list(self.source.parent.iterdir()), [self.source])

    def test_preview_uses_export_pixels_and_never_writes_output(self):
        info = probe(self.source, audio=False)
        for mode in ("none", "bayer8", "blue-noise", "floyd-steinberg", "riemersma"):
            self.options.dither = mode
            for fraction in (0, .5, 1):
                result = make_preview(self.options, fraction, bounds=(96, 48))
                renderer = Renderer(self.options.palette, dither=mode)
                raster = Rasterizer(self.options.palette, "source", 128, 64)
                expected = None
                with closing(_prepared_videos(info, renderer, raster, result.columns, result.rows,
                                              "source", None)) as frames:
                    for image, start, end in frames:
                        if start <= result.position:
                            expected = image
                        else:
                            break
                np.testing.assert_array_equal(result.full_frame, expected)
                expected = ImageOps.pad(Image.fromarray(expected), (96, 48),
                                        method=Image.Resampling.LANCZOS, color="black")
                np.testing.assert_array_equal(result.converted, expected)
                self.assertEqual(result.original.size, (96, 48))
                self.assertEqual((result.columns, result.rows), (16, 4))
        self.assertEqual(list(self.source.parent.iterdir()), [self.source])

    def test_png_saves_full_raster_and_retains_snapshot_after_cache_changes(self):
        cache = {}
        result = make_preview(self.options, .5, bounds=(31, 17), zoom=2, cache=cache)
        self.assertIs(result.full_frame, cache["render"][1])
        self.assertEqual(result.full_frame.size, (128, 64))
        expected = np.asarray(result.full_frame).copy()
        self.options.brightness = .5
        updated = make_preview(self.options, .5, bounds=(31, 17), cache=cache)
        self.assertIsNot(updated.full_frame, result.full_frame)
        target = self.source.with_name("полный кадр.png")
        save_preview_frame(result, target)
        with Image.open(target) as saved:
            self.assertEqual(saved.size, (128, 64))
            self.assertEqual(saved.format, "PNG")
            np.testing.assert_array_equal(saved, expected)

    def test_new_color_controls_match_direct_raster_and_encoded_export_input(self):
        import av
        from fractions import Fraction
        from export import render, _rgb_frame
        # A colored fixture ensures saturation changes are observable.
        rgb = np.empty((32, 64, 3), dtype=np.uint8)
        rgb[..., 0] = np.linspace(20, 180, 64).astype(np.uint8)
        rgb[..., 1] = 70
        rgb[..., 2] = 35
        with av.open(str(self.source), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 30)
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        self.options.no_audio = True
        baseline = make_preview(self.options, 0).full_frame.copy()
        for saturation, correction in ((-1, 0), (.5, .7), (0, 1)):
            self.options.color_saturation = saturation
            self.options.glyph_correction = correction
            result = make_preview(self.options, 0)
            info = probe(self.source, audio=False)
            renderer = Renderer(self.options.palette, color="source",
                                color_saturation=saturation, glyph_correction=correction)
            symbols, foreground = renderer.render(frame_at(info, 0), result.columns, result.rows)
            colors = np.clip(np.floor(foreground + .5), 0, 255).astype(np.uint8)
            expected = Rasterizer(self.options.palette, "source", 128, 64).draw(symbols, colors)
            np.testing.assert_array_equal(result.full_frame, expected)
            self.assertFalse(np.array_equal(result.full_frame, baseline))
            self.options.output = self.source.with_name(f"controls-{saturation}-{correction}.mp4")
            captured = []
            def capture(image, **kwargs):
                captured.append(image.copy())
                return _rgb_frame(image, **kwargs)
            with patch("export._rgb_frame", side_effect=capture):
                render(self.options)
            self.assertTrue(captured)
            np.testing.assert_array_equal(captured[0], expected)
            with av.open(str(self.options.output)) as output:
                self.assertFalse(output.streams.audio)
                self.assertEqual(len(list(output.decode(video=0))), 2)

    def test_mono_l_raster_black_intervals_zoom_and_png_are_rgb_exact(self):
        fixture(self.source, video_pts=list(range(6, 18)), audio_offset=0, audio_samples=48000)
        self.options.color = "mono"
        info = probe(self.source, audio=not self.options.no_audio)
        renderer = Renderer(self.options.palette)
        raster = Rasterizer(self.options.palette, "mono", 128, 64)
        cache = {}
        target = self.source.with_suffix(".png")
        for fraction in (0, .5, 1):
            result = make_preview(self.options, fraction, bounds=(97, 49), cache=cache)
            self.assertIn(result.full_frame.mode, ("L", "RGB"))
            self.assertEqual(result.converted.mode, "RGB")
            rgb = frame_at(info, result.position, allow_before_start=True)
            if rgb is None:
                expected = np.zeros((64, 128, 3), np.uint8)
                self.assertEqual(fraction, 0)
            else:
                grid = renderer.sample(rgb, result.columns, result.rows)
                expected = raster.draw(renderer.symbols(grid), None)
            np.testing.assert_array_equal(result.full_frame.convert("RGB"), expected)
            for zoom in (1, 2):
                detail = make_preview(self.options, fraction, bounds=(97, 49), zoom=zoom, cache=cache)
                self.assertIs(detail.full_frame, result.full_frame)
                self.assertEqual(detail.converted.mode, "RGB")
            save_preview_frame(result, target)
            with Image.open(target) as saved:
                self.assertEqual(saved.mode, result.full_frame.mode)
                np.testing.assert_array_equal(saved.convert("RGB"), expected)

    def test_png_failure_keeps_existing_file_and_removes_temporary(self):
        result = make_preview(self.options)
        target = self.source.with_suffix(".png")
        target.write_bytes(b"previous image")
        before = set(target.parent.iterdir())

        def failed_save(image, stream, **kwargs):
            stream.write(b"incomplete png")
            raise OSError("disk full")

        with patch.object(Image.Image, "save", failed_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_preview_frame(result, target)
        self.assertEqual(target.read_bytes(), b"previous image")
        self.assertEqual(set(target.parent.iterdir()), before)
        with patch("preview.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                save_preview_frame(result, target)
        self.assertEqual(target.read_bytes(), b"previous image")
        self.assertEqual(set(target.parent.iterdir()), before)
        with self.assertRaises(ValueError):
            save_preview_frame(result, target.with_suffix(".jpg"))
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "исходный"):
            save_preview_frame(replace(result, input_path=target), target)

    def test_rotation_and_non_square_pixels_keep_display_aspect(self):
        make_rotated_video(self.source, 90)
        result = make_preview(self.options, 0, bounds=(96, 96))
        self.assertEqual((result.columns, result.rows), (16, 19))
        pixels = np.asarray(result.original)
        self.assertTrue(np.all(pixels[:, :20] == 0))
        self.assertGreater(np.count_nonzero(pixels[:, 30:60]), 0)

    def test_zoom_cache_reuses_decode_and_raster_with_exact_pixels(self):
        views = ((0, (.5, .5)), (1, (.2, .7)), (2, (.8, .3)), (1, (1, 1)))
        expected = [make_preview(self.options, .5, zoom=zoom, center=center) for zoom, center in views]
        cache = {}
        with patch("preview.probe", wraps=probe) as probing, \
                patch("preview.frame_at", wraps=frame_at) as decoding, \
                patch("preview.Rasterizer", wraps=Rasterizer) as raster:
            actual = [make_preview(self.options, .5, zoom=zoom, center=center, cache=cache)
                      for zoom, center in views]
        probing.assert_called_once()
        decoding.assert_called_once()
        raster.assert_called_once()
        self.assertEqual(set(cache), {"metadata", "source", "source_image", "sample", "render"})
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.original, b.original)
            np.testing.assert_array_equal(a.converted, b.converted)
            self.assertEqual(a.viewport, b.viewport)

    def test_each_image_setting_replaces_raster_but_reuses_source(self):
        cache = {}
        make_preview(self.options, .5, cache=cache)
        sample = Renderer.sample
        for name, value in (("palette", " .#@"), ("gamma", 1.4), ("contrast", 1.2),
                            ("brightness", .15), ("color_saturation", .4), ("glyph_correction", .7),
                            ("dither", "bayer8"), ("color", "#00FF88"),
                            ("columns", 12), ("resolution", (64, 128))):
            setattr(self.options, name, value)
            expected = make_preview(self.options, .5)
            old = weakref.ref(cache["render"][1])
            old_grid = weakref.ref(cache["sample"][1])
            old_geometry = cache["sample"][0]
            grid_pixels = cache["sample"][1].copy()
            with patch("preview.frame_at", side_effect=AssertionError("Extra decode")), \
                    patch("preview.Rasterizer", wraps=Rasterizer) as raster, \
                    patch.object(Renderer, "sample", autospec=True, side_effect=sample) as sampling:
                actual = make_preview(self.options, .5, cache=cache)
            raster.assert_called_once()
            if cache["sample"][0] == old_geometry:
                sampling.assert_not_called()
                self.assertIs(cache["sample"][1], old_grid())
                np.testing.assert_array_equal(cache["sample"][1], grid_pixels)
            else:
                sampling.assert_called_once()
                self.assertIsNone(old_grid())
            self.assertIsNone(old())
            np.testing.assert_array_equal(actual.converted, expected.converted)
            self.assertEqual(set(cache), {"metadata", "source", "source_image", "sample", "render"})

    def test_seeks_reuse_metadata_and_release_previous_source_image(self):
        cache = {}
        with patch("preview.probe", wraps=probe) as probing, \
                patch("preview.frame_at", wraps=frame_at) as decoding:
            make_preview(self.options, .1, cache=cache)
            old_image = weakref.ref(cache["source_image"])
            old_source = weakref.ref(cache["source"][2])
            make_preview(self.options, .8, cache=cache)
            self.assertIsNone(old_image())
            self.assertIsNone(old_source())
            self.options.no_audio = not self.options.no_audio
            make_preview(self.options, .8, cache=cache)
        self.assertEqual(probing.call_count, 2)
        self.assertEqual(decoding.call_count, 3)

    def test_pan_reuses_source_image_without_another_full_rgb_copy(self):
        cache = {}
        make_preview(self.options, .5, cache=cache)
        image = cache["source_image"]
        with patch("preview.Image.fromarray", side_effect=AssertionError("Full RGB copied")):
            actual = make_preview(self.options, .5, zoom=2, center=(.2, .7), cache=cache)
        expected = make_preview(self.options, .5, zoom=2, center=(.2, .7))
        self.assertIs(cache["source_image"], image)
        np.testing.assert_array_equal(actual.original, expected.original)
        np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_cancelled_seek_keeps_only_valid_metadata_and_can_retry(self):
        cache, stopped = {}, threading.Event()
        make_preview(self.options, .1, cache=cache)
        old_image = weakref.ref(cache["source_image"])
        def cancelled(*args, **kwargs):
            stopped.set()
            return None
        with patch("preview.frame_at", side_effect=cancelled), self.assertRaises(InterruptedError):
            make_preview(self.options, .8, stop_event=stopped, cache=cache)
        self.assertEqual(set(cache), {"metadata"})
        self.assertIsNone(old_image())
        with patch("preview.probe", side_effect=AssertionError("Metadata probed again")):
            actual = make_preview(self.options, .8, cache=cache)
        expected = make_preview(self.options, .8)
        np.testing.assert_array_equal(actual.original, expected.original)
        np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_position_and_file_metadata_invalidate_cache_and_release_old_frames(self):
        cache = {}
        make_preview(self.options, .1, cache=cache)
        old_source, old_raster = weakref.ref(cache["source"][2]), weakref.ref(cache["render"][1])
        old_grid = weakref.ref(cache["sample"][1])
        make_preview(self.options, .8, cache=cache)
        self.assertIsNone(old_source())
        self.assertIsNone(old_raster())
        self.assertIsNone(old_grid())
        stat = self.source.stat()
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        old_grid = weakref.ref(cache["sample"][1])
        old_image = weakref.ref(cache["source_image"])
        with patch("preview.frame_at", wraps=frame_at) as decoding, \
                patch("preview.probe", wraps=probe) as probing:
            make_preview(self.options, .8, cache=cache)
        decoding.assert_called_once()
        probing.assert_called_once()
        self.assertIsNone(old_grid())
        self.assertIsNone(old_image())
        other = self.source.with_name("other.mp4")
        other.write_bytes(self.source.read_bytes())
        self.options.input = other
        old_grid = weakref.ref(cache["sample"][1])
        with patch("preview.frame_at", wraps=frame_at) as decoding, \
                patch("preview.probe", wraps=probe) as probing:
            make_preview(self.options, .8, cache=cache)
        decoding.assert_called_once()
        probing.assert_called_once()
        self.assertIsNone(old_grid())

    def test_resolution_reuses_sample_only_when_fitted_grid_is_unchanged(self):
        cache = {}
        make_preview(self.options, .5, cache=cache)
        self.options.resolution = (256, 128)
        expected = make_preview(self.options, .5)
        with patch.object(Renderer, "sample", side_effect=AssertionError("Same geometry resampled")):
            actual = make_preview(self.options, .5, cache=cache)
        np.testing.assert_array_equal(actual.converted, expected.converted)
        old_grid = weakref.ref(cache["sample"][1])
        self.options.resolution = (8, 8)
        expected = make_preview(self.options, .5)
        actual = make_preview(self.options, .5, cache=cache)
        self.assertIsNone(old_grid())
        self.assertEqual(cache["sample"][0], (actual.columns, actual.rows))
        np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_same_sample_is_unchanged_across_color_and_dither_modes(self):
        cache = {}
        make_preview(self.options, .5, cache=cache)
        grid = cache["sample"][1]
        original = grid.copy()
        for color in ("mono", "#00FF88", "source"):
            for dither in ("none", "bayer8", "floyd-steinberg", "riemersma"):
                self.options.color, self.options.dither = color, dither
                self.options.gamma = 1.4
                expected = make_preview(self.options, .5)
                with patch.object(Renderer, "sample", side_effect=AssertionError("Sample was not reused")):
                    actual = make_preview(self.options, .5, cache=cache)
                self.assertIs(cache["sample"][1], grid)
                np.testing.assert_array_equal(grid, original)
                np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_cancelled_sample_is_not_published_and_old_geometry_is_released(self):
        cache, stopped = {}, threading.Event()
        make_preview(self.options, .5, cache=cache)
        old_grid = weakref.ref(cache["sample"][1])
        self.options.columns = 12
        sample = Renderer.sample
        def cancelled(renderer, *args):
            grid = sample(renderer, *args)
            stopped.set()
            return grid
        with patch.object(Renderer, "sample", cancelled), self.assertRaises(InterruptedError):
            make_preview(self.options, .5, stop_event=stopped, cache=cache)
        self.assertIsNone(old_grid())
        self.assertEqual(set(cache), {"metadata", "source", "source_image"})
        expected = make_preview(self.options, .5)
        actual = make_preview(self.options, .5, cache=cache)
        np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_cancelled_raster_is_not_published_to_cache(self):
        cache, stopped = {}, threading.Event()
        draw = Rasterizer.draw
        def cancelled(raster, *args):
            result = draw(raster, *args)
            stopped.set()
            return result
        with patch.object(Rasterizer, "draw", cancelled), self.assertRaises(InterruptedError):
            make_preview(self.options, .5, stop_event=stopped, cache=cache)
        self.assertEqual(set(cache), {"metadata", "source", "sample"})
        expected = make_preview(self.options, .5)
        actual = make_preview(self.options, .5, cache=cache)
        np.testing.assert_array_equal(actual.converted, expected.converted)

    def test_unknown_duration_shows_first_frame_and_cancellation_skips_probe(self):
        info = probe(self.source, audio=False)
        unknown = SimpleNamespace(**{**info.__dict__, "duration": None})
        with patch("preview.probe", return_value=unknown):
            result = make_preview(self.options, 1)
        self.assertEqual(result.position, 0)
        self.assertIsNone(result.duration)
        stopped = threading.Event()
        stopped.set()
        with patch("preview.probe") as probing, self.assertRaises(InterruptedError):
            make_preview(self.options, stop_event=stopped)
        probing.assert_not_called()
        for fraction in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_preview(self.options, fraction)

    def test_worker_cancels_between_operations_and_reports_errors(self):
        entered, release = threading.Event(), threading.Event()
        original = probe(self.source, audio=False)
        def probing(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original
        with patch("preview.probe", side_effect=probing), patch("preview.frame_at") as decoding:
            job = PreviewJob(self.options, .5, (96, 48))
            job.start()
            self.assertTrue(entered.wait(1))
            job.cancel()
            release.set()
            job.thread.join(2)
            self.assertFalse(job.thread.is_alive())
            self.assertEqual(job.result.get_nowait(), ("cancelled", None))
            decoding.assert_not_called()
        with patch("preview.make_preview", side_effect=RuntimeError("decode failed")):
            job = PreviewJob(self.options, .5, (96, 48))
            job.start()
            job.thread.join(2)
        self.assertEqual(job.result.get_nowait(), ("error", "decode failed"))

    def window(self):
        window = PreviewWindow.__new__(PreviewWindow)
        window.app = SimpleNamespace(snapshot=lambda: {"input": str(self.source)}, preview=window)
        window.window, window.status, window.position = Mock(), Mock(), Mock()
        window.position.get.return_value = 20
        window.percent, window.percent_box = Mock(), Mock()
        window.percent.get.return_value = "20"
        window.closing, window.pending, window._poll_id = False, None, None
        window._resize_id = None
        window.bounds = (96, 48)
        window.zoom = Mock()
        window.zoom.get.return_value = "Вписать"
        window.center = (.5, .5)
        window.view = None
        window.frame_preview = None
        window.frame_job = None
        window.frame_button = Mock()
        window.source_viewport = None
        window._drag = None
        window._drag_moved = window._pan_pending = False
        window.labels, window.image_items, window.scale = [Mock(), Mock()], [1, 2], Mock()
        window.photos = (object(), object())
        window.cache = {"held": object()}
        window.job = Mock()
        window.job.thread.is_alive.return_value = True
        window.job.stop_event.is_set.return_value = True
        window.job.result = queue.Queue()
        window.job.result.put(("done", Mock()))
        return window

    def test_save_dialog_uses_clicked_result_even_if_new_result_arrives(self):
        window = self.window()
        old = make_preview(self.options, .2)
        new = make_preview(self.options, .8)
        window.frame_preview = old
        window.app.dialogs, window.app.messages = Mock(), Mock()
        target = self.source.with_suffix(".png")

        def dialog(**kwargs):
            self.assertIn(f"{old.position:.3f}s", kwargs["initialfile"])
            window.frame_preview = new
            return str(target)

        window.app.dialogs.asksaveasfilename.side_effect = dialog
        with patch("preview.save_preview_frame") as save:
            window.save_frame()
            window.frame_job.thread.join(2)
        self.assertIs(save.call_args.args[0], old)
        self.assertEqual(save.call_args.args[1], str(target))
        window.status.set.assert_not_called()
        window.app.messages.showerror.assert_not_called()

    def test_save_dialog_cancel_error_and_close(self):
        window = self.window()
        window.app.dialogs, window.app.messages = Mock(), Mock()
        window.save_frame()
        window.app.dialogs.asksaveasfilename.assert_not_called()

        window.frame_preview = make_preview(self.options)
        window.app.dialogs.asksaveasfilename.return_value = ""
        with patch("preview.save_preview_frame") as save:
            window.save_frame()
            save.assert_not_called()
        window.app.dialogs.asksaveasfilename.return_value = str(self.source.with_suffix(".png"))
        with patch("preview.save_preview_frame", side_effect=OSError("disk full")):
            window.save_frame()
            window.frame_job.thread.join(2)
        window.poll()
        window.app.messages.showerror.assert_called_once()
        window.closing = True
        window.app.dialogs.asksaveasfilename.reset_mock()
        window.save_frame()
        window.app.dialogs.asksaveasfilename.assert_not_called()

        window.closing = False
        def closed_during_dialog(**kwargs):
            window.closing = True
            return str(self.source.with_suffix(".png"))
        window.app.dialogs.asksaveasfilename.side_effect = closed_during_dialog
        with patch("preview.save_preview_frame") as save:
            window.save_frame()
            save.assert_not_called()

    def test_png_worker_cancel_preserves_file_and_releases_snapshot(self):
        result = make_preview(self.options)
        reference = weakref.ref(result)
        target = self.source.with_suffix(".png")
        target.write_bytes(b"old png")
        entered, release = threading.Event(), threading.Event()
        original_save = Image.Image.save

        def blocked_save(image, *args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test release missing")
            return original_save(image, *args, **kwargs)

        job = FrameSaveJob(result, target)
        del result
        with patch.object(Image.Image, "save", blocked_save):
            job.start()
            try:
                self.assertTrue(entered.wait(2))
                job.cancel()
            finally:
                release.set()
                job.thread.join(2)
        self.assertFalse(job.thread.is_alive())
        self.assertEqual(job.result.get_nowait(), ("cancelled", None))
        self.assertEqual(target.read_bytes(), b"old png")
        self.assertFalse(list(target.parent.glob(".*.png")))
        self.assertIsNone(job.preview)
        self.assertIsNone(reference())
        job = FrameSaveJob(make_preview(self.options), target)
        job.cancel()
        job.start()
        job.thread.join(2)
        self.assertEqual(job.result.get_nowait(), ("cancelled", None))
        self.assertEqual(target.read_bytes(), b"old png")

    def test_png_cancellation_waits_for_started_atomic_publication(self):
        target = self.source.with_suffix(".png")
        entered, release = threading.Event(), threading.Event()
        cancel_started, cancel_done = threading.Event(), threading.Event()
        real_replace = os.replace
        def blocked_replace(*args):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test release missing")
            return real_replace(*args)
        job = FrameSaveJob(make_preview(self.options), target)
        def cancel():
            cancel_started.set()
            job.cancel()
            cancel_done.set()
        canceller = threading.Thread(target=cancel)
        with patch("preview.os.replace", blocked_replace):
            job.start()
            try:
                self.assertTrue(entered.wait(2))
                canceller.start()
                self.assertTrue(cancel_started.wait(2))
                self.assertFalse(cancel_done.wait(.02))
                self.assertFalse(job.stop_event.is_set())
            finally:
                release.set()
                job.thread.join(2)
                if canceller.ident is not None:
                    canceller.join(2)
        self.assertFalse(job.thread.is_alive())
        self.assertFalse(canceller.is_alive())
        self.assertEqual(job.result.get_nowait(), ("done", str(target)))
        with Image.open(target) as saved:
            saved.load()
        self.assertTrue(cancel_done.is_set())

    def test_close_waits_for_both_workers_and_saving_has_no_queue(self):
        window = self.window()
        window.frame_preview = make_preview(self.options)
        window.app.dialogs, window.app.messages = Mock(), Mock()
        window.app.dialogs.asksaveasfilename.return_value = str(self.source.with_suffix(".png"))
        with patch("preview.FrameSaveJob") as factory:
            window.save_frame()
            window.save_frame()
        factory.assert_called_once()
        saving = window.frame_job
        saving.thread.is_alive.return_value = True
        window.close()
        saving.cancel.assert_called_once()
        window.job.cancel.assert_called_once()
        window.job.thread.is_alive.return_value = False
        window.poll()
        window.window.destroy.assert_not_called()
        saving.thread.is_alive.return_value = False
        saving.result.get_nowait.return_value = ("cancelled", None)
        window.poll()
        window.window.destroy.assert_called_once()
        self.assertIsNone(window.frame_job)
        self.assertIsNone(window.frame_preview)
        window.app.messages.showerror.assert_not_called()

    def test_percent_entry_accepts_fractional_positions_and_rejects_invalid_values(self):
        for text, expected in (("0", 0), ("100", 100), ("12.5", 12.5), ("12,5", 12.5)):
            window = self.window()
            window.percent.get.return_value = text
            with patch.object(window, "refresh") as refresh:
                self.assertEqual(window.seek_percent(), "break")
            window.position.set.assert_called_once_with(expected)
            refresh.assert_called_once()
        for text in ("", "abc", "nan", "inf", "-1", "100.1"):
            window = self.window()
            window.percent.get.return_value = text
            with patch.object(window, "refresh") as refresh:
                self.assertEqual(window.seek_percent(), "break")
            window.position.set.assert_not_called()
            refresh.assert_not_called()
            window.job.cancel.assert_not_called()
        window = self.window()
        window.closing = True
        window.seek_percent()
        window.percent.get.assert_not_called()

    def test_refresh_disables_old_hit_maps_until_current_result_is_displayed(self):
        window = self.window()
        window.view = (1920, 1080), (0, 0, 1920, 1080)
        window.source_viewport = (0, 0, 1920, 1080)
        photos = window.photos
        window.refresh()
        pending = window.pending
        self.assertIsNone(window.view)
        self.assertIsNone(window.source_viewport)
        self.assertIsNone(window.frame_preview)
        self.assertEqual(window.frame_button.configure.call_args.kwargs["state"], "disabled")
        for source in (False, True):
            window.inspect(SimpleNamespace(x=1, y=1), source=source)
        self.assertEqual(window.center, (.5, .5))
        self.assertIs(window.pending, pending)
        self.assertIs(window.photos, photos)
        for label in window.labels:
            label.configure.assert_called_with(cursor="watch")

        result = make_preview(self.options, .2, bounds=window.bounds)
        window.pending = None
        window.job.thread.is_alive.return_value = False
        window.job.stop_event.is_set.return_value = False
        window.job.result = queue.Queue()
        window.job.result.put(("done", result))
        with patch("PIL.ImageTk.PhotoImage"):
            window.poll()
        self.assertEqual(window.view, (result.frame_size, result.viewport))
        self.assertEqual(window.source_viewport, result.original_viewport)
        self.assertIs(window.frame_preview, result)
        self.assertEqual(window.frame_button.configure.call_args.kwargs["state"], "normal")
        for label in window.labels:
            label.configure.assert_called_with(cursor="crosshair")

    def test_photoimage_failure_keeps_polling_and_close_waits_for_png_worker(self):
        from tkinter import TclError

        for failure_index in (0, 1):
            with self.subTest(failure_index=failure_index):
                window = self.window()
                result = make_preview(self.options, .2, bounds=window.bounds)
                old_photos = window.photos
                window.job.thread.is_alive.return_value = False
                window.job.stop_event.is_set.return_value = False
                window.job.result = queue.Queue()
                window.job.result.put(("done", result))
                effects = [object()] * failure_index + [TclError("invalid command name PyImagingPhoto")]
                with patch("PIL.ImageTk.PhotoImage", side_effect=effects):
                    window.poll()
                self.assertIsNone(window.job)
                self.assertIsNone(window.view)
                self.assertIsNone(window.source_viewport)
                self.assertIsNone(window.frame_preview)
                self.assertIs(window.photos, old_photos)
                self.assertEqual(window.frame_button.configure.call_args.kwargs["state"], "disabled")
                self.assertIn("PyImagingPhoto", window.status.set.call_args.args[0])
                for label in window.labels:
                    label.itemconfigure.assert_not_called()
                    label.configure.assert_called_with(cursor="arrow")
                window.window.after.assert_called_once_with(50, window.poll)
                self.assertIsNotNone(window._poll_id)

                window.frame_job = Mock()
                saving = window.frame_job
                saving.thread.is_alive.return_value = True
                window.close()
                saving.cancel.assert_called_once()
                window.poll()
                window.window.destroy.assert_not_called()
                saving.thread.is_alive.return_value = False
                saving.result.get_nowait.return_value = ("cancelled", None)
                window.poll()
                window.window.destroy.assert_called_once()
                self.assertIsNone(window.app.preview)

    def test_photoimage_completion_rechecks_closing_and_pending_state(self):
        for change in ("closing", "pending"):
            with self.subTest(change=change):
                window = self.window()
                result = make_preview(self.options, .2, bounds=window.bounds)
                old_photos = window.photos
                finished = window.job
                finished.thread.is_alive.return_value = False
                finished.stop_event.is_set.return_value = False
                finished.result = queue.Queue()
                finished.result.put(("done", result))
                window.frame_job = Mock()
                window.frame_job.thread.is_alive.return_value = True

                def photo(*args, **kwargs):
                    finished.thread.join.assert_called_once()
                    self.assertIsNone(window.job)
                    if change == "closing":
                        window.close()
                    else:
                        window.pending = ("new request",)
                    return object()

                with patch("PIL.ImageTk.PhotoImage", side_effect=photo) as make_photo, \
                        patch.object(window, "_start_pending") as start:
                    window.poll()
                self.assertEqual(make_photo.call_count, 2)
                self.assertIs(window.photos, old_photos)
                self.assertIsNone(window.frame_preview)
                self.assertIsNone(window.view)
                for label in window.labels:
                    label.itemconfigure.assert_not_called()
                window.window.destroy.assert_not_called()
                window.window.after.assert_called_once_with(50, window.poll)
                if change == "closing":
                    window.frame_job.cancel.assert_called_once()
                    start.assert_not_called()
                else:
                    start.assert_called_once()

    def test_invalid_parameters_and_worker_error_leave_old_images_unselectable(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                window = self.window()
                window.view = (1920, 1080), (0, 0, 1920, 1080)
                window.source_viewport = (0, 0, 1920, 1080)
                photos = window.photos
                if invalid:
                    window.app.snapshot = lambda: {"input": str(self.source), "gamma": "nan"}
                window.refresh()
                if not invalid:
                    window.pending = None
                    window.job.thread.is_alive.return_value = False
                    window.job.stop_event.is_set.return_value = False
                    window.job.result = queue.Queue()
                    window.job.result.put(("error", "decode failed"))
                    window.poll()
                self.assertIsNone(window.view)
                self.assertIsNone(window.source_viewport)
                self.assertIs(window.photos, photos)
                window.inspect(SimpleNamespace(x=1, y=1), source=True)
                self.assertEqual(window.center, (.5, .5))
                for label in window.labels:
                    label.configure.assert_called_with(cursor="arrow")

    def test_resize_coalesces_changes_and_preserves_zoom_center_and_cache(self):
        window = self.window()
        window.labels = [Mock(), Mock()]
        window.status_label, window.hint = Mock(), Mock()
        for label in window.labels:
            label.winfo_width.return_value = 321
            label.winfo_height.return_value = 181
        window.zoom.get.return_value = "200%"
        window.center = (.2, .8)
        window.view = (self.options.resolution, (0, 0, 96, 48))
        held_cache = window.cache.copy()
        window.window.after.side_effect = ("resize-1", "resize-2")
        window._queue_resize(Mock())
        window._queue_resize(Mock())
        window.window.after_cancel.assert_called_once_with("resize-1")
        with patch.object(window, "refresh", wraps=window.refresh) as refresh:
            window.inspect(SimpleNamespace(x=10, y=10))
            refresh.assert_not_called()
            window._resize()
            refresh.assert_called_once()
        self.assertEqual(window.bounds, (321, 181))
        self.assertEqual(window.pending[2:], (2, (.2, .8)))
        self.assertEqual(window.cache, held_cache)
        self.assertIsNone(window.view)
        window.job.cancel.assert_called_once()
        window.job.thread.join.assert_not_called()
        window.window.after.reset_mock()
        window._resize()
        window.window.after.assert_not_called()

    def test_close_cancels_resize_and_late_callback_does_not_restart_work(self):
        window = self.window()
        window._resize_id = "resize-1"
        window.close()
        window.window.after_cancel.assert_called_once_with("resize-1")
        with patch.object(window, "refresh") as refresh:
            window._queue_resize(Mock())
            window._resize()
        refresh.assert_not_called()
        self.assertIsNone(window._resize_id)

    def test_native_zoom_shows_exact_export_pixels_including_edges_and_small_outputs(self):
        info = probe(self.source, audio=False)
        for resolution in ((128, 64), (32, 16)):
            self.options.resolution = resolution
            renderer = Renderer(self.options.palette)
            raster = Rasterizer(self.options.palette, "source", *resolution)
            overview = make_preview(self.options, 0, bounds=(97, 49))
            with closing(_prepared_videos(info, renderer, raster, overview.columns, overview.rows,
                                          "source", None)) as frames:
                exported, _, _ = next(frames)
            for zoom in (1, 2):
                for center in ((.5, .5), (0, 0), (1, 1)):
                    with self.subTest(resolution=resolution, zoom=zoom, center=center):
                        result = make_preview(self.options, 0, bounds=(97, 49), zoom=zoom, center=center)
                        left, top, _, _ = result.viewport
                        canvas = Image.new("RGB", (int(np.ceil(97 / zoom)), int(np.ceil(49 / zoom))))
                        canvas.paste(Image.fromarray(exported), (-left, -top))
                        expected = np.repeat(np.repeat(np.asarray(canvas), zoom, axis=0), zoom, axis=1)[:49, :97]
                        np.testing.assert_array_equal(result.converted, expected)
                        self.assertEqual(result.original.size, (97, 49))

    def test_click_maps_visible_pixels_and_coalesces_zoom_requests(self):
        window = self.window()
        window.view = (1920, 1080), (0, 0, 1920, 1080)
        window.inspect(SimpleNamespace(x=24, y=12))
        self.assertEqual(window.center, (.25, .25))
        window.zoom.set.assert_called_once_with("100%")
        window.zoom.get.return_value = "200%"
        window.view = (1920, 1080), (100, 200, 48, 24)
        window.inspect(SimpleNamespace(x=48, y=24))
        self.assertEqual(window.center, (124 / 1920, 212 / 1080))
        self.assertEqual(window.pending[2:], (2, window.center))
        window.view = (1920, 1080), (100, 200, 48, 24)
        window.source_viewport = (960, 0, 480, 1080)
        window.inspect(SimpleNamespace(x=24, y=12), source=True)
        self.assertEqual(window.center, (1080 / 1920, .25))

    def test_source_zoom_matches_display_aspect_and_padding_without_full_canvas(self):
        make_rotated_video(self.source, 90)
        info = probe(self.source, audio=False)
        self.options.resolution = (128, 96)
        source = Image.fromarray(frame_at(info, 0))
        overview = make_preview(self.options, 0, bounds=(97, 49))
        left, top, span_x, span_y = overview.original_viewport
        # The center of the source thumbnail still maps to the fitted source
        # center in the differently shaped MP4 canvas, within one display pixel.
        self.assertAlmostEqual((left + span_x / 2) / 128, .5, delta=.02)
        self.assertAlmostEqual((top + span_y / 2) / 96, .5, delta=.02)
        width = min(128, max(1, round(96 * info.aspect)))
        height = min(96, max(1, round(width / info.aspect)))
        reference = Image.new("RGB", (128, 96))
        reference.paste(source.resize((width, height), Image.Resampling.LANCZOS),
                        ((128 - width) // 2, (96 - height) // 2))
        for zoom in (1, 2):
            for center in ((.5, .5), (0, 0), (1, 1)):
                result = make_preview(self.options, 0, bounds=(97, 49), zoom=zoom, center=center)
                left, top, _, _ = result.viewport
                crop = reference.crop((left, top, left + int(np.ceil(97 / zoom)),
                                       top + int(np.ceil(49 / zoom))))
                expected = np.repeat(np.repeat(np.asarray(crop), zoom, axis=0), zoom, axis=1)[:49, :97]
                # Pillow 13's region resize differs from whole-image resize by
                # up to two RGB8 levels at fractional filter coordinates.
                error = np.abs(np.asarray(result.original).astype(int) - expected.astype(int))
                self.assertLessEqual(error.max(), 2)

    def test_newest_request_replaces_pending_and_old_result_is_discarded(self):
        window = self.window()
        old_job = window.job
        window.refresh()
        window.position.get.return_value = 80
        window.refresh()
        self.assertEqual(window.pending[1], .8)
        old_job.thread.is_alive.return_value = False
        with patch.object(window, "_start_pending") as start, patch("PIL.ImageTk.PhotoImage") as photo:
            window.poll()
        start.assert_called_once()
        photo.assert_not_called()
        old_job.thread.join.assert_called_once()

    def test_invalid_parameters_cancel_pending_and_close_waits_for_worker(self):
        window = self.window()
        window.pending = (self.options, .8)
        window.app.snapshot = lambda: {"input": str(self.source), "gamma": "nan"}
        window.refresh()
        self.assertIsNone(window.pending)
        window.job.cancel.assert_called_once()
        window.close()
        window.poll()
        window.window.destroy.assert_not_called()
        window.job.thread.is_alive.return_value = False
        window.poll()
        window.window.destroy.assert_called_once()
        self.assertIsNone(window.app.preview)
        self.assertEqual(window.cache, {})

    def test_application_close_waits_for_preview_cleanup(self):
        app = Application.__new__(Application)
        app._preview_update_id = None
        app.job = None
        app.preview, app.root = Mock(), Mock()
        app.close()
        app.preview.close.assert_called_once()
        app.poll()
        app.root.destroy.assert_not_called()
        app.preview = None
        app.poll()
        app.root.destroy.assert_called_once()

    def test_preview_validation_keeps_timeline_without_requiring_output(self):
        values = {"input": str(self.source), "output": "", "audio": True, "brightness": "-.5"}
        _, options = build_arguments("preview", values)
        self.assertFalse(options.no_audio)
        self.assertTrue(build_arguments("preview", {**values, "audio": False})[1].no_audio)
        self.assertEqual(options.brightness, -.5)
        self.assertEqual(values["output"], "")


if __name__ == "__main__":
    unittest.main()

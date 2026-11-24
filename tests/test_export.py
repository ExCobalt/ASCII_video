from fractions import Fraction
import errno
import gc
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from export import _audio_frames, _encode_video, _normalized_audio, _publish, _rgb_frame, render


def fixture(path, video_pts=None, audio_offset=None, audio_samples=24000):
    if video_pts is None:
        video_pts = list(range(12))
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=30)
        video.width, video.height, video.pix_fmt = 64, 32, "yuv420p"
        video.codec_context.time_base = Fraction(1, 30)
        video.options = {"preset": "ultrafast", "tune": "zerolatency"}
        audio = None
        if audio_offset is not None:
            audio = container.add_stream("aac", rate=48000)
            audio.layout = "stereo"
            audio.bit_rate = 192000
            audio.codec_context.format = "fltp"
        for index, pts in enumerate(video_pts):
            image = np.full((32, 64, 3), max(80, 255 - index * 8), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            container.mux(video.encode(frame))
        container.mux(video.encode(None))
        if audio is not None:
            cursor = 0
            while cursor < audio_samples:
                count = min(1024, audio_samples - cursor)
                t = np.arange(cursor, cursor + count) / 48000
                signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(np.stack((signal, signal)), format="fltp", layout="stereo")
                frame.sample_rate = 48000
                frame.pts, frame.time_base = audio_offset + cursor, Fraction(1, 48000)
                container.mux(audio.encode(frame))
                cursor += count
            container.mux(audio.encode(None))


def video_summary(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        size = stream.width, stream.height
        rate = stream.average_rate
        duration = stream.duration * stream.time_base
        frames = list(container.decode(stream))
        times = [frame.pts * frame.time_base for frame in frames]
        means = [frame.to_ndarray(format="rgb24").mean() for frame in frames]
        return size, rate, duration, times, means


class ExportTests(unittest.TestCase):
    def test_exclusive_publish_preserves_racing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "temporary"
            output = Path(directory) / "output"
            temporary.write_bytes(b"finished")
            output.write_bytes(b"winner")
            with self.assertRaises(FileExistsError):
                _publish(temporary, output, False)
            self.assertEqual(output.read_bytes(), b"winner")
            self.assertEqual(temporary.read_bytes(), b"finished")

    def test_exclusive_publish_falls_back_on_unsupported_posix_rename(self):
        import ctypes
        for platform, syscall in (("linux", "renameat2"), ("darwin", "renamex_np")):
            for error in (errno.ENOSYS, errno.EOPNOTSUPP):
                with self.subTest(platform=platform, error=error):
                    libc = Mock()
                    getattr(libc, syscall).return_value = -1
                    with patch("export.os.name", "posix"), patch("export.sys.platform", platform), \
                            patch.object(ctypes, "CDLL", return_value=libc), \
                            patch.object(ctypes, "get_errno", return_value=error), \
                            patch("export.os.link") as link, patch("export.os.unlink") as unlink:
                        _publish("temporary", "output", False)
                        link.assert_called_once_with("temporary", "output")
                        unlink.assert_called_once_with("temporary")

    def test_each_dither_exports_distinct_patterns_with_identical_timestamps(self):
        from dithering import DITHER_MODES
        from diffusion import MODES
        patterns, timeline = [], None
        for mode in (*DITHER_MODES, *MODES):
            options = self.options(dither=mode, palette=" @", no_audio=True,
                                   output=self.directory / f"dither-{mode}.mp4")
            render(options)
            summary = video_summary(options.output)
            if timeline is None:
                timeline = summary[:4]
            self.assertEqual(summary[:4], timeline)
            with av.open(str(options.output)) as container:
                frames = list(container.decode(video=0))
                patterns.append(frames[-1].to_ndarray(format="rgb24").tobytes())
        self.assertEqual(len(set(patterns)), len(DITHER_MODES) + len(MODES))

    def test_quality_sets_each_encoder_and_rejects_before_io(self):
        expected = {"compact": ("30", "27"), "balanced": ("26", "23"), "high": ("20", "19")}
        open_container = av.open
        for quality, values in expected.items():
            for encoder, quantizer in zip(("x264", "nvenc"), values):
                for color in ("mono", "source", "#19ABEF"):
                    with self.subTest(quality=quality, encoder=encoder, color=color):
                        output = Mock()
                        output.__enter__ = Mock(return_value=output)
                        output.__exit__ = Mock(return_value=False)
                        video = output.add_stream.return_value
                        video.encode.return_value = []
                        def open_selected(path, *args, **kwargs):
                            return output if args and args[0] == "w" else open_container(path, *args, **kwargs)
                        with patch("export.av.Codec"), patch("export.av.open", side_effect=open_selected):
                            render(self.options(encoder=encoder, quality=quality, color=color,
                                                no_audio=True, overwrite=True))
                        wanted = quantizer
                        self.assertEqual(video.options["crf" if encoder == "x264" else "cq"], wanted)
                        self.assertEqual(video.options["preset"], "fast" if encoder == "x264" else "p4")
                        self.assertEqual(video.options.get("chromaoffset"),
                                         None)
        with patch("export.av.open") as opened, self.assertRaisesRegex(ValueError, "Quality"):
            render(self.options(quality="bad"))
        opened.assert_not_called()

    def test_quality_profiles_encode_and_default_remains_balanced(self):
        results = {}
        for quality in (None, "compact", "balanced", "high"):
            options = self.options(no_audio=True, overwrite=True)
            if quality is not None:
                options.quality = quality
            render(options)
            results[quality] = options.output.read_bytes()
            with av.open(str(options.output)) as container:
                self.assertFalse(container.streams.audio)
                points = [f.pts * f.time_base for f in container.decode(video=0)]
                self.assertEqual(points, [Fraction(i, 30) for i in range(12)])
        self.assertEqual(results[None], results["balanced"])
        self.assertNotEqual(results["compact"], results["high"])

    def test_nvenc_encode_and_flush_errors_preserve_output_and_cause(self):
        open_container = av.open
        for phase in ("frame", "flush"):
            with self.subTest(phase=phase):
                options = self.options(encoder="nvenc", overwrite=True, no_audio=True)
                options.output.write_bytes(b"previous export")
                output = Mock()
                output.__enter__ = Mock(return_value=output)
                output.__exit__ = Mock(return_value=False)
                video = output.add_stream.return_value
                original = av.error.ArgumentError(22, "Invalid argument", "avcodec_open2(h264_nvenc)")
                def encode(frame):
                    if phase == "frame" or frame is None:
                        raise original
                    return []
                video.encode.side_effect = encode
                def open_selected(path, *args, **kwargs):
                    return output if args and args[0] == "w" else open_container(path, *args, **kwargs)
                with (patch("export.av.Codec"), patch("export.av.open", side_effect=open_selected),
                      self.assertRaises(RuntimeError) as caught):
                    render(options)
                self.assertIs(caught.exception.__cause__, original)
                for detail in ("NVENC", "128x64", "GPU/driver", "x264", str(original)):
                    self.assertIn(detail, str(caught.exception))
                output.add_stream.assert_called_once_with("h264_nvenc", rate=30)
                self.assertEqual(options.output.read_bytes(), b"previous export")
                self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_encoder_context_does_not_relabel_other_errors(self):
        video = Mock(width=128, height=64)
        for encoder, original in (("x264", av.error.ArgumentError(22, "x264 failure")),
                                  ("nvenc", ValueError("invalid Python argument")),
                                  ("nvenc", InterruptedError("cancelled"))):
            with self.subTest(encoder=encoder, error=type(original)):
                video.encode.side_effect = original
                with self.assertRaises(type(original)) as caught:
                    _encode_video(video, None, encoder)
                self.assertIs(caught.exception, original)
        # A muxer failure is outside the video encoder boundary, even for NVENC.
        output = Mock()
        output.__enter__ = Mock(return_value=output)
        output.__exit__ = Mock(return_value=False)
        output.add_stream.return_value.encode.return_value = []
        original = av.error.ArgumentError(22, "muxer failure")
        output.mux.side_effect = original
        open_container = av.open
        def open_selected(path, *args, **kwargs):
            return output if args and args[0] == "w" else open_container(path, *args, **kwargs)
        with (patch("export.av.Codec"), patch("export.av.open", side_effect=open_selected),
              self.assertRaises(av.FFmpegError) as caught):
            render(self.options(encoder="nvenc", no_audio=True))
        self.assertIs(caught.exception, original)
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_encoded_glyph_edges_are_brighter_without_changing_timeline(self):
        import raster

        # Isolate raster coverage from calibrated palette selection and its cache.
        # Both exports select the same glyphs from this explicit custom ramp.
        with patch.object(raster, "_TEXT_COVERAGE", np.arange(256, dtype=np.uint8)):
            raw_atlas = raster._glyph_atlas.__wrapped__(8, 16, 12)
        for mode in ("mono", "source"):
            with self.subTest(mode=mode):
                baseline = self.options(output=self.directory / f"raw-{mode}.mp4", color=mode,
                                        palette=" @", no_audio=True)
                with patch.object(raster, "_glyph_atlas", return_value=raw_atlas):
                    render(baseline)
                corrected = self.options(output=self.directory / f"corrected-{mode}.mp4", color=mode,
                                         palette=" @", no_audio=True)
                render(corrected)
                before, after = video_summary(baseline.output), video_summary(corrected.output)
                self.assertEqual(before[:4], after[:4])
                self.assertEqual(after[3], [Fraction(index, 30) for index in range(12)])
                self.assertTrue(np.all(np.asarray(after[4]) > np.asarray(before[4]) * 1.15))

    def test_rgb_snapshot_padding_lifetime_and_yuv_parity(self):
        frames = []
        backend = av.VideoFrame
        def allocate(*args):
            frame = backend(*args)
            plane = frame.planes[0]
            plane.update(bytes([165]) * plane.buffer_size)
            return frame
        for width, height in ((2, 2), (14, 8), (18, 10)):
            original = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
            image = original[::-1, ::-1, ::-1]
            expected = image.copy()
            image.setflags(write=False)
            with patch("export.av", SimpleNamespace(VideoFrame=allocate)):
                frame = _rgb_frame(image)
            original[:] = 0
            del original, image
            gc.collect()
            plane = frame.planes[0]
            raw = np.frombuffer(plane, dtype=np.uint8)
            rows = raw[:height * plane.line_size].reshape(height, plane.line_size)
            self.assertTrue(np.all(rows[:, width * 3:] == 165))
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)
            np.testing.assert_array_equal(frame.reformat(format="yuv420p").to_ndarray(),
                backend.from_ndarray(expected, format="rgb24").reformat(format="yuv420p").to_ndarray())
            frames.append(frame)
        self.assertEqual(len({frame.planes[0].buffer_ptr for frame in frames}), 3)
        del frame, plane, raw, rows
        frames.clear()
        gc.collect()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "видео с пробелом.mp4"
        fixture(self.source)
        self.stderr = io.StringIO()
        self.capture = patch("sys.stderr", self.stderr)
        self.capture.start()
        self.addCleanup(self.capture.stop)

    def options(self, **kwargs):
        values = dict(input=self.source, output=self.directory / "готовое видео.mp4",
                      overwrite=False, resolution=(128, 64), columns=16,
                      palette=" .:-=+*#%@", color="mono", gamma=1, contrast=1,
                      dither="none", no_audio=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_unavailable_nvenc_preserves_output_without_fallback(self):
        options = self.options(encoder="nvenc", overwrite=True, no_audio=True)
        options.output.write_bytes(b"previous export")
        missing = av.codec.codec.UnknownCodecError("h264_nvenc")
        with (patch("export.av.Codec", side_effect=missing) as codec,
              patch("export.tempfile.mkstemp") as temporary,
              self.assertRaisesRegex(RuntimeError, "h264_nvenc.*unavailable") as caught):
            render(options)
        codec.assert_called_once_with("h264_nvenc", "w")
        temporary.assert_not_called()
        self.assertIs(caught.exception.__cause__, missing)
        self.assertEqual(options.output.read_bytes(), b"previous export")
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_full_hd_columns_change_grid_and_saved_pixels(self):
        from raster import Rasterizer
        source = self.directory / "wide.mp4"
        with av.open(str(source), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv420p"
            y, x = np.indices((90, 160))
            gray = ((x * 13 + y * 17) % 256).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.repeat(gray[..., None], 3, axis=2), format="rgb24")
            frame.pts, frame.time_base = 0, Fraction(1, 30)
            container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        shapes, images, summaries = [], [], []
        draw = Rasterizer.draw
        def capture(rasterizer, symbols, colors):
            shapes.append(symbols.shape)
            return draw(rasterizer, symbols, colors)
        for columns in (240, 480):
            options = self.options(input=source, resolution=(1920, 1080), columns=columns,
                                   no_audio=True, output=self.directory / f"columns-{columns}.mp4")
            with patch.object(Rasterizer, "draw", capture):
                render(options)
            with av.open(str(options.output)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 1)
                summaries.append((frames[0].width, frames[0].height, frames[0].pts * frames[0].time_base))
                images.append(frames[0].to_ndarray(format="rgb24"))
        self.assertEqual(shapes, [(68, 240), (135, 480)])
        self.assertEqual(summaries, [(1920, 1080, 0), (1920, 1080, 0)])
        self.assertFalse(np.array_equal(*images))

    def test_brightness_reaches_export_without_changing_size_or_timing(self):
        summaries = []
        for brightness in (-1, 0, 1):
            options = self.options(brightness=brightness, no_audio=True,
                                   output=self.directory / f"brightness-{brightness}.mp4")
            render(options)
            summaries.append(video_summary(options.output))
        self.assertEqual(summaries[0][:4], summaries[1][:4])
        self.assertEqual(summaries[1][:4], summaries[2][:4])
        self.assertLess(max(summaries[0][4]), min(summaries[1][4]))
        self.assertNotEqual(summaries[1][4], summaries[2][4])
        default = self.options(no_audio=True)
        render(default)
        self.assertEqual(default.output.read_bytes(),
                         (self.directory / "brightness-0.mp4").read_bytes())

    def test_manual_brightness_keeps_near_black_letterbox_empty(self):
        from raster import Rasterizer

        with av.open(str(self.source), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
            pixels = np.full((32, 64, 3), 1, dtype=np.uint8)
            pixels[8:24] = 120
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = 0, Fraction(1, 30)
            container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        options = self.options(brightness=0.1, no_audio=True)
        drawn = []
        draw = Rasterizer.draw

        def capture(instance, symbols, colors):
            self.assertTrue(np.all(symbols[[0, -1]] == " "))
            image = draw(instance, symbols, colors)
            drawn.append(image)
            return image

        with patch.object(Rasterizer, "draw", capture):
            render(options)
        self.assertTrue(drawn)
        for image in drawn:
            self.assertFalse(image[:16].any())
            self.assertFalse(image[-16:].any())
        with av.open(str(options.output)) as container:
            pixels = next(container.decode(video=0)).to_ndarray(format="rgb24")
        # H.264 may introduce a few RGB code values beside the bright content;
        # the exact assertions above verify that no letterbox glyph was drawn.
        self.assertLessEqual(pixels[:16].max(), 4)
        self.assertLessEqual(pixels[-16:].max(), 4)
        self.assertGreater(pixels[16:-16].max(), 30)

    def test_silent_export_dimensions_fps_duration_unicode_and_progress(self):
        options = self.options()
        updates = []
        render(options, progress=lambda *values: updates.append(values))
        size, rate, duration, times, _ = video_summary(options.output)
        self.assertEqual(size, (128, 64))
        self.assertEqual(rate, 30)
        self.assertEqual(duration, Fraction(2, 5))
        self.assertEqual(times, [Fraction(i, 30) for i in range(12)])
        self.assertEqual(updates[-1], (12, 0.4))
        self.assertEqual(self.stderr.getvalue(), "")
        with av.open(str(options.output)) as container:
            self.assertEqual(len(container.streams.audio), 0)
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_identical_cells_reuse_raster_without_changing_mp4(self):
        from raster import Rasterizer

        # The fixture fades, then holds a constant decoded gray. Exercise
        # cache invalidation before the hold, VFR timing and an audio tail.
        fixture(self.source, video_pts=[index * 2 for index in range(40)],
                audio_offset=0, audio_samples=144000)
        original = Rasterizer.draw
        for color in ("mono", "source", "#19ABEF"):
            with self.subTest(color=color):
                options = self.options(color=color, overwrite=True)
                with patch("export.np.array_equal", return_value=False):
                    render(options)
                reference = options.output.read_bytes()
                calls = []

                def draw(instance, symbols, colors):
                    calls.append(1)
                    return original(instance, symbols, colors)

                with patch.object(Rasterizer, "draw", draw):
                    render(options)
                self.assertGreater(len(calls), 1)
                self.assertLess(len(calls), 40)
                self.assertEqual(options.output.read_bytes(), reference)

    def test_export_tags_match_actual_yuv_for_fresh_and_held_frames(self):
        fixture(self.source, video_pts=[0, 3, 6])
        colors = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255],
                           [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        image = np.repeat(np.repeat(colors[None], 64, axis=0), 32, axis=1)
        options = self.options(resolution=(192, 64), color="source", no_audio=True)
        # Isolate the RGB -> encoder boundary with broad constant color bars.
        with patch("export.Rasterizer.draw", return_value=image):
            render(options)
        expected = [[16, 126, 235, 81, 145, 41],
                    [128, 128, 128, 90, 54, 240],
                    [128, 128, 128, 240, 34, 110]]
        with av.open(str(options.output)) as container:
            stream = container.streams.video[0]
            self.assertEqual((stream.codec_context.colorspace,
                              stream.codec_context.color_range), (6, 1))
            count = 0
            for frame in container.decode(stream):
                self.assertEqual((frame.colorspace, frame.color_range), (6, 1))
                self.assertEqual(frame.pts * frame.time_base, Fraction(count, 30))
                for plane, values in zip(frame.planes, expected):
                    data = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
                    centers = np.arange(6) * (plane.width // 6) + plane.width // 12
                    np.testing.assert_allclose(data[plane.height // 2, centers], values, atol=2, rtol=0)
                count += 1
            self.assertEqual(count, 7)

    def test_gui_job_cancels_real_encoder_and_preserves_existing_mp4(self):
        from gui import ExportJob

        # Two VFR frames describe ten minutes without a large input fixture.
        # Cancel through the same cross-thread Event used by the GUI button.
        fixture(self.source, video_pts=[0, 18000])
        options = self.options(overwrite=True, no_audio=True, allow_long_export=True)
        original = self.source.read_bytes()
        options.output.write_bytes(original)
        job = ExportJob(options)
        job.start()
        try:
            frames, media_time = job.progress.get(timeout=10)
            self.assertGreaterEqual(frames, 1)
            self.assertGreaterEqual(media_time, 0)
            self.assertTrue(job.thread.is_alive())
            self.assertTrue(list(self.directory.glob(".*.mp4")))
        finally:
            job.cancel()
            job.thread.join(10)
        self.assertFalse(job.thread.is_alive())
        self.assertEqual(job.result.get_nowait(), ("cancelled", None))
        self.assertTrue(job.result.empty())
        self.assertLessEqual(job.progress.qsize(), 1)
        self.assertEqual(options.output.read_bytes(), original)
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])
        self.assertEqual(self.stderr.getvalue(), "")

    def test_vfr_holds_active_frame_and_avoids_repeated_rasterization(self):
        fixture(self.source, video_pts=[0, 3, 9])
        options = self.options(color="source")
        from raster import Rasterizer
        original = Rasterizer.draw
        calls = []
        def draw(instance, symbols, colors):
            calls.append(1)
            return original(instance, symbols, colors)
        with patch.object(Rasterizer, "draw", draw):
            render(options)
        _, _, duration, times, means = video_summary(options.output)
        self.assertEqual(len(calls), 3)
        self.assertEqual(times, [Fraction(i, 30) for i in range(10)])
        self.assertEqual(duration, Fraction(1, 3))
        # A held image can be quantized differently in I/P/B pictures; the
        # decoded mean need not be identical even though the input is held.
        self.assertAlmostEqual(means[0], means[2], delta=2)
        self.assertAlmostEqual(means[3], means[8], delta=2)

    def test_audio_offset_and_longer_audio_extend_video(self):
        fixture(self.source, audio_offset=9600, audio_samples=24000)
        options = self.options()
        render(options)
        _, _, duration, times, means = video_summary(options.output)
        self.assertGreaterEqual(float(duration), 0.7)
        self.assertLess(float(duration), 0.75)
        self.assertAlmostEqual(means[11], means[-1], delta=1)
        with av.open(str(options.output)) as container:
            audio = container.streams.audio[0]
            self.assertEqual(audio.codec_context.name, "aac")
            self.assertEqual(audio.codec_context.sample_rate, 48000)
            self.assertEqual(audio.codec_context.layout.name, "stereo")
            samples = np.concatenate([f.to_ndarray() for f in container.decode(audio)], axis=1)
        self.assertLess(np.abs(samples[:, :8500]).max(), 0.002)
        self.assertGreater(np.abs(samples[:, 11000:18000]).mean(), 0.05)

    def test_early_audio_end_and_no_audio_option(self):
        fixture(self.source, audio_offset=0, audio_samples=4800)
        options = self.options()
        render(options)
        self.assertEqual(video_summary(options.output)[2], Fraction(2, 5))
        with av.open(str(options.output)) as container:
            stream = container.streams.audio[0]
            self.assertLess(float(stream.duration * stream.time_base), 0.15)
        options.overwrite, options.no_audio = True, True
        render(options)
        with av.open(str(options.output)) as container:
            self.assertEqual(len(container.streams.audio), 0)

    def test_one_second_audio_delay_is_preserved(self):
        fixture(self.source, audio_offset=48000, audio_samples=24000)
        options = self.options()
        render(options)
        _, _, duration, _, means = video_summary(options.output)
        self.assertGreaterEqual(float(duration), 1.5)
        self.assertLess(float(duration), 1.55)
        self.assertAlmostEqual(means[11], means[-1], delta=1)
        with av.open(str(options.output)) as container:
            samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        self.assertLess(np.abs(samples[:, :46000]).max(), .002)
        self.assertGreater(np.abs(samples[:, 50000:56000]).mean(), .05)

    def test_audio_before_video_produces_black_initial_frames(self):
        fixture(self.source, video_pts=list(range(6, 18)), audio_offset=0, audio_samples=4800)
        options = self.options()
        render(options)
        _, _, duration, times, means = video_summary(options.output)
        self.assertEqual(duration, Fraction(3, 5))
        self.assertEqual(times, [Fraction(i, 30) for i in range(18)])
        self.assertTrue(all(value < 1 for value in means[:6]))
        self.assertGreater(means[6], 10)

    def test_overwrite_and_same_file_protection(self):
        options = self.options()
        options.output.write_bytes(b"original")
        with self.assertRaises(ValueError):
            render(options)
        self.assertEqual(options.output.read_bytes(), b"original")
        options.overwrite = True
        render(options)
        self.assertEqual(video_summary(options.output)[0], options.resolution)
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            render(self.options(output=self.source, overwrite=True))
        self.assertEqual(self.source.read_bytes(), original)

    def test_destination_created_during_render_is_not_replaced(self):
        options = self.options()
        other_writer = b"created by another process after the initial existence check"
        updates = []

        def create_destination(*values):
            updates.append(values)
            if len(updates) == 1:
                options.output.write_bytes(other_writer)

        with self.assertRaises(FileExistsError):
            render(options, progress=create_destination)
        self.assertTrue(updates)
        self.assertEqual(options.output.read_bytes(), other_writer)
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_cancel_and_failure_clean_temporary_preserve_original(self):
        options = self.options(overwrite=True)
        options.output.write_bytes(b"original")
        stop = threading.Event()
        with self.assertRaisesRegex(InterruptedError, "Rendering cancelled"):
            render(options, stop_event=stop, progress=lambda *_: stop.set())
        self.assertEqual(options.output.read_bytes(), b"original")
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])
        with patch("export.Rasterizer.draw", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                render(options)
        self.assertEqual(options.output.read_bytes(), b"original")
        self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_channel_resize_failure_and_cancel_join_workers_preserve_output(self):
        from PIL import Image

        options = self.options(overwrite=True, color="source", columns=480,
                               resolution=(1920, 1080), no_audio=True)
        resize = Image.Image.resize
        for fail in (True, False):
            with self.subTest(fail=fail):
                options.output.write_bytes(b"original")
                stop = threading.Event()
                workers = set()

                def interrupted_resize(image, size, resample=None, *args, **kwargs):
                    if (image.mode == "L" and image.width >= 1920
                            and resample == Image.Resampling.LANCZOS):
                        workers.add(threading.current_thread())
                        if fail:
                            raise RuntimeError("channel resize failed")
                        stop.set()
                    return resize(image, size, resample, *args, **kwargs)

                with patch.object(Image.Image, "resize", interrupted_resize):
                    with self.assertRaisesRegex(RuntimeError if fail else InterruptedError,
                                                "channel resize failed" if fail else "Rendering cancelled"):
                        render(options, stop_event=stop)
                self.assertTrue(workers)
                self.assertTrue(all(worker.name.startswith("ascii-resize") for worker in workers))
                self.assertTrue(all(not worker.is_alive() for worker in workers))
                self.assertFalse(any(t.name.startswith("ascii-resize") for t in threading.enumerate()))
                self.assertEqual(options.output.read_bytes(), b"original")
                self.assertEqual(list(self.directory.glob(".*.mp4")), [])

    def test_audio_gap_chunks_and_overlap_rejection_without_hardware(self):
        from export import _audio_frame
        first = _audio_frame(np.ones((2, 1200), dtype=np.float32), 100)
        second = _audio_frame(np.ones((2, 100), dtype=np.float32), 3000)
        class Container:
            streams = [object()]
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def decode(self, _):
                return iter((first, second))
        info = SimpleNamespace(path="unused", audio_index=0, origin=Fraction(0))
        with patch("export.open_media", return_value=Container()):
            frames = list(_audio_frames(info))
        self.assertTrue(all(frame.samples <= 1024 for frame in frames))
        self.assertEqual(frames[0].pts, 0)
        self.assertEqual(frames[-1].pts + frames[-1].samples, 3100)
        merged = np.concatenate([frame.to_ndarray() for frame in frames], axis=1)
        np.testing.assert_array_equal(merged[:, :100], 0)
        np.testing.assert_array_equal(merged[:, 100:1300], 1)
        np.testing.assert_array_equal(merged[:, 1300:3000], 0)
        with self.assertRaisesRegex(RuntimeError, "Overlapping"):
            list(_normalized_audio(second, 4000, Fraction(0)))
        second.pts = 1000
        with patch("export.open_media", return_value=Container()):
            with self.assertRaisesRegex(RuntimeError, "Overlapping audio timestamps"):
                list(_audio_frames(info))


if __name__ == "__main__":
    unittest.main()

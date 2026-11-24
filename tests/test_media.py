from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import queue
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

import av
import numpy as np

from media import decode_video_from, frame_at, open_media, probe, timestamp, video_intervals, VideoDecoder
from rendering import Renderer


def make_video(path, codec="libx264", times=None):
    times = times or [0, 40, 100, 180, 220]
    with av.open(str(path), "w") as output:
        stream = output.add_stream(codec, rate=25)
        stream.width, stream.height = 64, 36
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        if codec == "libx264":
            stream.options = {"preset": "ultrafast", "crf": "28"}
        for index, pts in enumerate(times):
            rgb = np.full((36, 64, 3), (index * 40) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


class MediaTests(unittest.TestCase):
    def test_demuxer_receives_only_the_selected_file_and_releases_it_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "медиа с пробелом.mp4"
            path.touch()
            with patch("media.av.open", side_effect=ValueError("invalid media")) as opened:
                with self.assertRaisesRegex(ValueError, "invalid media"), open_media(path):
                    self.fail("Invalid input opened")
            source, = opened.call_args.args
            self.assertEqual(Path(source.name), path)
            self.assertTrue(source.closed)
            self.assertEqual(opened.call_args.kwargs["container_options"]["protocol_whitelist"], "")
            with self.assertRaisesRegex(ValueError, "self-contained"):
                opened.call_args.kwargs["io_open"]("additional file", 1, {})

    def test_silent_matroska_uses_video_end_even_when_audio_is_longer(self):
        from test_export import fixture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long-audio.mkv"
            fixture(path, video_pts=[0], audio_offset=0, audio_samples=48000)
            silent, audible = probe(path, audio=False), probe(path)
            self.assertAlmostEqual(silent.duration, 1 / 30, delta=.002)
            self.assertGreaterEqual(audible.duration, 1)
            # Keep the valid segment duration, but remove the optional track tags.
            path.write_bytes(path.read_bytes().replace(b"DURATION", b"UNTIMED_"))
            self.assertIsNone(probe(path, audio=False).duration)
            self.assertGreaterEqual(probe(path).duration, 1)

    def test_seek_cancellation_stops_before_another_seek_or_decode(self):
        stream = SimpleNamespace(time_base=Fraction(1), start_time=0)
        late = SimpleNamespace(pts=1001, time_base=Fraction(1))
        for stage in ("before_seek", "seek", "decode", "first_frame"):
            with self.subTest(stage=stage):
                cancel = threading.Event()
                container = Mock()

                def seek(*args, **kwargs):
                    if stage == "seek":
                        cancel.set()

                def decode(selected):
                    if stage == "decode":
                        cancel.set()

                    def frames():
                        if stage == "first_frame":
                            cancel.set()
                        yield late

                    return frames()

                container.seek.side_effect = seek
                container.decode.side_effect = decode
                if stage == "before_seek":
                    cancel.set()
                self.assertEqual(list(decode_video_from(container, stream, Fraction(1000),
                                                       cancel.is_set)), [])
                self.assertEqual(container.seek.call_count, stage != "before_seek")
                self.assertEqual(container.decode.call_count, stage in ("decode", "first_frame"))

    def test_seek_cancellation_after_yield_does_not_decode_next_frame(self):
        cancel = threading.Event()
        stream = SimpleNamespace(time_base=Fraction(1), start_time=0)
        first = SimpleNamespace(pts=0, time_base=Fraction(1))
        frames = MagicMock(__next__=Mock(side_effect=[first, AssertionError("decoded after cancel")]))
        container = Mock(decode=Mock(return_value=frames))
        request = decode_video_from(container, stream, Fraction(0), cancel.is_set)
        self.assertIs(next(request), first)
        cancel.set()
        self.assertEqual(list(request), [])
        self.assertEqual(frames.__next__.call_count, 1)

    def test_frame_at_cancelled_seek_returns_none_without_decode_error(self):
        cancel = threading.Event()
        info = SimpleNamespace(path="unused", origin=Fraction(0), video_index=0)
        stream = SimpleNamespace(time_base=Fraction(1), start_time=0)
        container = Mock(streams=[stream])
        container.seek.side_effect = lambda *args, **kwargs: cancel.set()
        with patch("media.open_media") as opened:
            opened.return_value.__enter__.return_value = container
            self.assertIsNone(frame_at(info, 1000, cancel))
            container.seek.assert_called_once()
            container.decode.assert_not_called()
            opened.reset_mock()
            self.assertIsNone(frame_at(info, 1000, cancel))
            opened.assert_not_called()

    def test_video_worker_passes_seek_cancellation_without_stopping_session(self):
        info = SimpleNamespace(path="unused", origin=Fraction(0), video_index=0)
        stream = SimpleNamespace(time_base=Fraction(1), start_time=0, average_rate=Fraction(25))
        stop = threading.Event()
        worker = VideoDecoder(info, Mock(), (4, 1, 4, 1), lambda: 0, stop)
        worker.seek_time = 1000
        container = Mock(streams=[stream])
        container.seek.side_effect = lambda *args, **kwargs: worker._cancel.set()
        with patch("media.open_media") as opened:
            opened.return_value.__enter__.return_value = container
            worker.start()
            worker.close()
        container.seek.assert_called_once()
        container.decode.assert_not_called()
        self.assertIsNone(worker.error)
        self.assertFalse(stop.is_set())
        self.assertTrue(worker.queue.empty())

    def test_cancelled_video_iterator_does_not_invent_final_frame_duration(self):
        cancel = threading.Event()
        frame = SimpleNamespace(pts=0, time_base=Fraction(1), duration=0)

        def frames():
            yield frame
            cancel.set()

        self.assertEqual(list(video_intervals(frames(), None, cancel.is_set)), [])

    def test_tagged_yuv_range_and_matrix_reach_rgb_grid(self):
        # FFmpeg AVColorSpace: BT709=1, SMPTE170M/BT601=6;
        # AVColorRange: limited/MPEG=1, full/JPEG=2. Lossless YUV444
        # keeps the input bars exact so the check isolates YUV -> RGB.
        cases = [(1, 1, (63, 102, 240), (255, 1, 0)),
                 (1, 2, (54, 99, 255), (254, 0, 0)),
                 (6, 1, (81, 90, 240), (254, 0, 0)),
                 (6, 2, (76, 85, 255), (254, 0, 0))]
        with tempfile.TemporaryDirectory() as directory:
            for space, color_range, red, expected_red in cases:
                with self.subTest(colorspace=space, color_range=color_range):
                    path = Path(directory) / f"bars-{space}-{color_range}.mp4"
                    gray = (16, 126, 235) if color_range == 1 else (0, 128, 255)
                    bars = [(level, 128, 128) for level in gray] + [red]
                    frame = av.VideoFrame(64, 16, "yuv444p")
                    for channel, plane in enumerate(frame.planes):
                        pixels = np.full((plane.height, plane.line_size), 128, np.uint8)
                        for index, bar in enumerate(bars):
                            pixels[:, index * 16:(index + 1) * 16] = bar[channel]
                        plane.update(pixels)
                    frame.colorspace, frame.color_range = space, color_range
                    frame.pts, frame.time_base = 0, Fraction(1, 30)
                    with av.open(str(path), "w") as output:
                        stream = output.add_stream("libx264", rate=30)
                        stream.width, stream.height, stream.pix_fmt = 64, 16, "yuv444p"
                        stream.codec_context.colorspace = space
                        stream.codec_context.color_range = color_range
                        stream.options = {"preset": "ultrafast", "crf": "0", "tune": "zerolatency"}
                        output.mux(stream.encode(frame))
                        output.mux(stream.encode(None))
                    info = probe(path, audio=False)
                    expected = np.array([[[0, 0, 0], [128, 128, 128],
                                          [255, 255, 255], expected_red]], dtype=np.float64)
                    worker = VideoDecoder(info, Renderer(), (4, 1, 4, 1), lambda: 0, threading.Event())
                    worker.start()
                    try:
                        prepared = worker.queue.get(timeout=3)
                        np.testing.assert_allclose(prepared.grid, expected, atol=1, rtol=0)
                    finally:
                        worker.close()
                    self.assertIsNone(worker.error)
                    resized = Renderer().sample(frame_at(info, 0), 4, 1)
                    np.testing.assert_allclose(resized, expected, atol=1, rtol=0)

    def test_vfr_and_equal_timestamps(self):
        frames = [SimpleNamespace(pts=p, time_base=Fraction(1, 1000), duration=40)
                  for p in [0, 40, 40, 100]]
        result = list(video_intervals(frames, Fraction(25)))
        self.assertEqual([row[1] for row in result], [0, Fraction(1, 25), Fraction(1, 10)])
        self.assertIs(result[1][0], frames[2])
        self.assertEqual(result[-1][2], Fraction(14, 100))
        self.assertTrue(result[-1][3])

    def test_missing_decreasing_and_final_duration(self):
        frame = SimpleNamespace(pts=None, time_base=Fraction(1, 1000), duration=0)
        with self.assertRaises(RuntimeError):
            timestamp(frame)
        frame.pts = 20
        earlier = SimpleNamespace(pts=10, time_base=frame.time_base, duration=0)
        with self.assertRaises(RuntimeError):
            list(video_intervals([frame, earlier], Fraction(25)))
        with self.assertRaises(RuntimeError):
            list(video_intervals([frame], None))
        self.assertEqual(list(video_intervals([frame], Fraction(25)))[0][2], Fraction(6, 100))

    def test_real_formats_and_bounded_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension, codec in [("mp4", "libx264"), ("mkv", "libx264"), ("webm", "libvpx-vp9")]:
                path = Path(directory) / f"видео с пробелом.{extension}"
                make_video(path, codec)
                info = probe(path)
                self.assertIsNone(info.audio_index)
                self.assertAlmostEqual(info.aspect, 16 / 9)
                stop = threading.Event()
                worker = VideoDecoder(info, Renderer(), (16, 5, 16, 5), lambda: 0.0, stop)
                worker.start()
                result = []
                try:
                    while not worker.eof or not worker.queue.empty():
                        try:
                            result.append(worker.queue.get(timeout=0.2))
                        except queue.Empty:
                            if worker.error:
                                raise worker.error
                    self.assertEqual(len(result), 5)
                    self.assertEqual(worker.queue.maxsize, 3)
                    self.assertEqual(result[0].grid.shape, (5, 16, 3))
                    self.assertAlmostEqual(result[2].start, 0.1)
                    self.assertTrue(result[-1].last)
                finally:
                    worker.close()
                self.assertFalse(worker.thread.is_alive())

    def test_expired_frames_skip_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movie.mp4"
            make_video(path, times=list(range(0, 500, 40)))
            renderer = Renderer()
            count = 0
            original = renderer.sample
            def sample(*args):
                nonlocal count
                count += 1
                return original(*args)
            renderer.sample = sample
            stop = threading.Event()
            worker = VideoDecoder(probe(path, audio=False), renderer, (16, 5, 16, 5), lambda: 1.0, stop)
            try:
                worker.start()
                worker.thread.join(timeout=2)
                self.assertFalse(worker.thread.is_alive())
                self.assertGreater(worker.dropped, 5)
                self.assertEqual(count, 1)  # Last image is retained past video EOF.
                self.assertTrue(worker.queue.get_nowait().last)
            finally:
                worker.close()

    def test_future_fps_deadline_skips_sampling_but_retains_crossing_and_last_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movie.mp4"
            make_video(path, times=[0, 20, 40, 60, 80, 100, 140, 160])
            counts = []
            for deadline in (0.0, .1, 1.0):
                renderer = Renderer()
                sampled = []
                original = renderer.sample
                def sample(*args):
                    sampled.append(1)
                    return original(*args)
                renderer.sample = sample
                worker = VideoDecoder(probe(path, audio=False), renderer,
                                      (16, 5, 16, 5), lambda: 0.0, threading.Event())
                worker.present_after = deadline
                result = []
                try:
                    worker.start()
                    while not worker.eof or not worker.queue.empty():
                        try:
                            result.append(worker.queue.get(timeout=.2))
                        except queue.Empty:
                            if worker.error:
                                raise worker.error
                    self.assertTrue(result[-1].last)
                    if deadline == .1:
                        self.assertAlmostEqual(result[0].start, .1)
                        self.assertAlmostEqual(result[0].end, .14)
                    counts.append(len(sampled))
                finally:
                    worker.close()
            self.assertEqual(counts, [8, 3, 1])


if __name__ == "__main__":
    unittest.main()

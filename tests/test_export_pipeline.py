"""Bounded export prefetch, cleanup, and source frames between output ticks."""

from fractions import Fraction
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from export import _prepared_videos, render
from test_export import fixture, video_summary


def speed_up_video(source, output):
    """Turn the fixture's 30fps video into 120fps without changing its audio."""
    with av.open(str(source)) as incoming, av.open(str(output), "w") as outgoing:
        streams = {stream.index: outgoing.add_stream_from_template(stream) for stream in incoming.streams}
        for packet in incoming.demux():
            if packet.dts is None:
                continue
            index = packet.stream.index
            if packet.stream.type == "video":
                packet.time_base /= 4
            packet.stream = streams[index]
            outgoing.mux(packet)


class ExportPipelineTests(unittest.TestCase):
    def test_source_strip_pool_is_limited_to_measured_geometry_and_cpu(self):
        from concurrent.futures import ThreadPoolExecutor
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        info = SimpleNamespace(aspect=16 / 9, audio_index=None, duration=1 / 30)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            options = SimpleNamespace(input=directory / 'input.mp4', output=directory / 'output.mp4',
                overwrite=True, resolution=(1920, 1080), columns=320, palette=' .#@',
                gamma=1, contrast=1, dither='none', color='source', no_audio=True)
            for cpus, color, columns, expected in ((24, 'source', 320, True),
                    (8, 'source', 320, False), (None, 'source', 320, False),
                    (24, 'mono', 320, False), (24, 'source', 240, False)):
                options.color, options.columns = color, columns
                # Isolate pool selection from decoding/raster work; one black
                # frame still exercises the real encoder and context cleanup.
                def prepared(*_):
                    yield image, Fraction(0), Fraction(1, 30)
                with patch('export.probe', return_value=info), \
                        patch('export.os.cpu_count', return_value=cpus), \
                        patch('export._prepared_videos', prepared), \
                        patch('export.Rasterizer') as raster, \
                        patch('export.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as executor:
                    render(options, progress=lambda *_: None)
                    self.assertIs(raster.return_value._source_parallel_strips, expected)
                    self.assertEqual(executor.call_args.kwargs['max_workers'], 12 if expected else 3)

    def test_source_rounding_preserves_readonly_grid_and_half_boundaries(self):
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        def source(_):
            yield frame, Fraction(0), Fraction(1, 30), 0, True
        values = [-.6, 0., .499, .5, 1.5, 127.499, 127.5, 254.5, 255.6]
        for dtype in (np.float32, np.float64):
            grid = np.array(values, dtype=dtype).reshape(1, 3, 3)
            grid.flags.writeable = False
            original = grid.copy()
            renderer = Mock()
            renderer.render.return_value = (np.array([list("abc")]), grid)
            rasterizer = Mock()
            rasterizer.draw.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
            with patch("export._video_frames", source):
                results = list(_prepared_videos(None, renderer, rasterizer, 3, 1, "source", None))
            self.assertEqual(len(results), 1)
            colors = rasterizer.draw.call_args.args[1]
            np.testing.assert_array_equal(colors, [[[0, 0, 0], [1, 2, 127], [128, 255, 255]]])
            self.assertEqual(colors.dtype, np.uint8)
            np.testing.assert_array_equal(grid, original)
            self.assertFalse(np.shares_memory(colors, grid))

    def test_duration_callback_reuses_probe_before_first_progress(self):
        from media import probe
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, output = Path(directory) / "input.mp4", Path(directory) / "output.mp4"
            fixture(source)
            options = SimpleNamespace(input=source, output=output, overwrite=False,
                resolution=(64, 32), columns=8, palette=" .:-=+*#%@", gamma=1, contrast=1,
                dither="none", color="mono", no_audio=True)
            events = []
            with patch("export.probe", wraps=probe) as inspect:
                render(options, on_duration=lambda duration: events.append(("duration", duration)),
                       progress=lambda *args: events.append(("progress", args)))
            inspect.assert_called_once()
            self.assertEqual(events[0][0], "duration")
            self.assertAlmostEqual(events[0][1], .4)
            self.assertTrue(all(event[0] == "progress" for event in events[1:]))

    def test_bounded_prefetch_uses_one_worker_and_closes_source(self):
        closed, second = threading.Event(), threading.Event()
        threads = []
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        def source(_):
            try:
                for index in range(10):
                    yield frame, Fraction(index, 30), Fraction(index + 1, 30), 0, index == 9
            finally:
                closed.set()
        renderer = Mock()
        renderer.render.side_effect = [(np.array([[chr(65 + index)]]), None) for index in range(10)]
        def draw(*_):
            threads.append(threading.current_thread())
            if len(threads) == 2:
                second.set()
            return np.zeros((2, 2, 3), dtype=np.uint8)
        rasterizer = Mock(draw=Mock(side_effect=draw))
        with patch("export._video_frames", source):
            prepared = _prepared_videos(None, renderer, rasterizer, 1, 1, "mono", None)
            try:
                first = next(prepared)
                self.assertEqual(first[1:3], (Fraction(0), Fraction(1, 30)))
                self.assertTrue(second.wait(3))
                self.assertEqual(len(threads), 2)
                self.assertIs(threads[0], threads[1])
                self.assertIsNot(threads[0], threading.current_thread())
            finally:
                prepared.close()
        self.assertTrue(closed.is_set())
        self.assertFalse(threads[0].is_alive())

    def test_worker_failure_and_cancellation_close_source_and_join(self):
        for cancel in (False, True):
            closed, ready = threading.Event(), threading.Event()
            stop = threading.Event()
            threads = []
            frame = Mock()
            frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
            def source(_):
                try:
                    for index in range(3):
                        yield frame, Fraction(index, 30), Fraction(index + 1, 30), 0, index == 2
                finally:
                    closed.set()
            renderer = Mock()
            renderer.render.side_effect = [(np.array([[chr(65 + index)]]), None) for index in range(3)]
            def draw(*_):
                threads.append(threading.current_thread())
                if len(threads) == 2:
                    ready.set()
                    if not cancel:
                        raise RuntimeError("prefetched raster failed")
                return np.zeros((2, 2, 3), dtype=np.uint8)
            with patch("export._video_frames", source):
                prepared = _prepared_videos(None, renderer, Mock(draw=Mock(side_effect=draw)), 1, 1, "mono", stop)
                try:
                    next(prepared)
                    self.assertTrue(ready.wait(3))
                    if cancel:
                        stop.set()
                    with self.assertRaisesRegex(InterruptedError if cancel else RuntimeError,
                                               "Rendering cancelled" if cancel else "prefetched raster failed"):
                        next(prepared)
                finally:
                    prepared.close()
            self.assertTrue(closed.is_set())
            self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_high_fps_skips_unused_rasters_and_holds_final_between_ticks(self):
        # Isolate skipped timestamp intervals from equal-cell reuse, which has
        # its own end-to-end byte comparison in test_export.py.
        with tempfile.TemporaryDirectory() as directory, patch("export.np.array_equal", return_value=False):
            directory = Path(directory)
            original, source, output = (directory / name for name in ("original.mp4", "input.mp4", "output.mp4"))
            options = SimpleNamespace(input=source, output=output, overwrite=True,
                resolution=(64, 32), columns=8, palette=" .:-=+*#%@", gamma=1, contrast=1,
                dither="none", color="mono", no_audio=True)
            fixture(original, video_pts=list(range(60)))
            speed_up_video(original, source)
            from raster import Rasterizer
            with patch.object(Rasterizer, "draw", autospec=True, side_effect=Rasterizer.draw) as draw:
                render(options, progress=lambda *_: None)
                self.assertEqual(draw.call_count, 16)  # Fifteen output ticks plus the final source frame.
            self.assertEqual(video_summary(output)[3], [Fraction(index, 30) for index in range(15)])

            fixture(original, video_pts=[0, 1], audio_offset=0, audio_samples=24000)
            speed_up_video(original, source)
            with av.open(str(source)) as container:
                self.assertEqual([frame.pts * frame.time_base for frame in container.decode(video=0)],
                                 [Fraction(0), Fraction(1, 120)])
            options.no_audio = False
            images = [np.zeros((32, 64, 3), dtype=np.uint8), np.full((32, 64, 3), 255, dtype=np.uint8)]
            with patch.object(Rasterizer, "draw", side_effect=images) as draw:
                render(options, progress=lambda *_: None)
                self.assertEqual(draw.call_count, 2)
            means = video_summary(output)[4]
            self.assertGreaterEqual(len(means), 15)
            self.assertLess(means[0], 2)
            self.assertTrue(all(value > 250 for value in means[1:]))


if __name__ == "__main__":
    unittest.main()

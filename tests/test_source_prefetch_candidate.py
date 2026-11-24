"""Two-future candidate contracts and bounds; run only outside timing probes."""
from fractions import Fraction
import inspect
import threading
from unittest.mock import Mock, patch

import numpy as np

import export
import test_export_pipeline as original


def enable_two_frame_prefetch(export_module):
    source = inspect.getsource(export_module._prepared_videos)
    original = '''                    pending = executor.submit(prepare_next)
                    try:
                        while True:
                            _check_stop(stop_event)
                            interval = pending.result()
                            if interval is None:
                                break
                            pending = executor.submit(prepare_next)
                            yield interval
                    finally:
                        pending.cancel()'''
    replacement = '''                    from collections import deque
                    queued = deque(executor.submit(prepare_next) for _ in range(2))
                    pending = None
                    try:
                        while queued:
                            _check_stop(stop_event)
                            pending = queued.popleft()
                            interval = pending.result()
                            if interval is None:
                                break
                            queued.append(executor.submit(prepare_next))
                            yield interval
                    finally:
                        if pending is not None:
                            pending.cancel()
                        for future in queued:
                            future.cancel()'''
    if source.count(original) != 1:
        raise RuntimeError('Expected current one-future raster prefetch loop')
    source = source.replace(original, replacement)
    exec(compile(source, export_module.__file__, 'exec'), export_module.__dict__)


class RasterPrefetchTests(original.ExportPipelineTests):
    def setUp(self):
        restore = patch.object(export, '_prepared_videos', export._prepared_videos)
        restore.start()
        self.addCleanup(restore.stop)
        enable_two_frame_prefetch(export)
        candidate = export._prepared_videos
        def enabled(info, renderer, rasterizer, *args):
            rasterizer._source_sample_ahead = True
            return candidate(info, renderer, rasterizer, *args)
        export._prepared_videos = enabled
        imported = patch.object(original, '_prepared_videos', enabled)
        imported.start()
        self.addCleanup(imported.stop)

    def test_bounded_prefetch_uses_one_worker_and_closes_source(self):
        # Replace only the original one-raster bound with the candidate's bound.
        # Inherited rounding, cancellation/failure and high-FPS/audio tests remain.
        closed, fourth_sampled, third_drawn = (threading.Event() for _ in range(3))
        sample_threads, raster_threads, issued = [], [], []
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)

        def source(_):
            try:
                for index in range(20):
                    yield frame, Fraction(index, 30), Fraction(index + 1, 30), 0, index == 19
            finally:
                closed.set()

        def sample(*_):
            sample_threads.append(threading.current_thread())
            index = len(sample_threads)
            if index == 4:
                fourth_sampled.set()
            return np.array([[chr(64 + index)]]), None

        def draw(*_):
            raster_threads.append(threading.current_thread())
            image = np.full((2, 2, 3), len(raster_threads), dtype=np.uint8)
            issued.append(image)
            if len(raster_threads) == 3:
                third_drawn.set()
            return image

        renderer = Mock(render=Mock(side_effect=sample))
        rasterizer = Mock(draw=Mock(side_effect=draw))
        with patch('export._video_frames', source):
            prepared = export._prepared_videos(None, renderer, rasterizer, 1, 1, 'mono', None)
            try:
                first = next(prepared)
                self.assertTrue(third_drawn.wait(3))
                self.assertTrue(fourth_sampled.wait(3))
                # Consumer holds frame1; futures hold frames2/3; sample4 awaits raster.
                self.assertEqual(len(raster_threads), 3)
                self.assertEqual(len(sample_threads), 4)
                self.assertEqual(first[1:], (Fraction(0), Fraction(1, 30)))
                np.testing.assert_array_equal(first[0], np.ones((2, 2, 3), dtype=np.uint8))
                self.assertEqual(len({id(t) for t in raster_threads}), 1)
                self.assertEqual(len({id(t) for t in sample_threads}), 1)
                self.assertIsNot(raster_threads[0], sample_threads[0])
                self.assertTrue(all(not np.shares_memory(issued[0], image) for image in issued[1:]))
            finally:
                prepared.close()
        self.assertTrue(closed.is_set())
        self.assertTrue(all(not t.is_alive() for t in raster_threads + sample_threads))

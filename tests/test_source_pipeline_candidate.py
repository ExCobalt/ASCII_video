"""Production sample-ahead semantics, bounds, failure, and cleanup checks."""
from fractions import Fraction
import threading
from unittest.mock import Mock, patch

import numpy as np

import export
import test_export_pipeline as original


class SampleAheadTests(original.ExportPipelineTests):
    def setUp(self):
        # Reuse the ordinary contract tests with the measured path enabled.
        production = export._prepared_videos
        def enabled(info, renderer, rasterizer, *args):
            rasterizer._source_sample_ahead = True
            return production(info, renderer, rasterizer, *args)
        restore = patch.object(export, '_prepared_videos', export._prepared_videos)
        restore.start()
        self.addCleanup(restore.stop)
        export._prepared_videos = enabled
        imported = patch.object(original, '_prepared_videos', export._prepared_videos)
        imported.start()
        self.addCleanup(imported.stop)

    def test_only_one_additional_sample_is_prefetched_and_cache_is_preserved(self):
        closed, sampled_third = threading.Event(), threading.Event()
        sample_threads, raster_threads = [], []
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)

        def source(_):
            try:
                for index in range(20):
                    yield frame, Fraction(index, 30), Fraction(index + 1, 30), 0, index == 19
            finally:
                closed.set()

        symbols = np.array([['a']])
        colors = np.zeros((1, 1, 3), dtype=np.float64)

        def sample(*_):
            sample_threads.append(threading.current_thread())
            if len(sample_threads) == 3:
                sampled_third.set()
            return symbols, colors

        image = np.zeros((2, 2, 3), dtype=np.uint8)

        def draw(*_):
            raster_threads.append(threading.current_thread())
            return image

        renderer, rasterizer = Mock(render=Mock(side_effect=sample)), Mock(draw=Mock(side_effect=draw))
        with patch('export._video_frames', source):
            prepared = export._prepared_videos(None, renderer, rasterizer, 1, 1, 'source', None)
            try:
                first = next(prepared)
                self.assertTrue(sampled_third.wait(3))
                self.assertEqual(len(sample_threads), 3)
                self.assertIs(first[0], image)
                second = next(prepared)
                self.assertIs(second[0], image)
                self.assertEqual(rasterizer.draw.call_count, 1)
            finally:
                prepared.close()
        self.assertTrue(closed.is_set())
        self.assertTrue(all(not t.is_alive() for t in sample_threads + raster_threads))
        self.assertIsNot(sample_threads[0], raster_threads[0])

    def test_sample_failure_propagates_and_joins_both_stages(self):
        closed = threading.Event()
        workers = []
        frame = Mock()
        frame.to_ndarray.return_value = np.zeros((2, 2, 3), dtype=np.uint8)

        def source(_):
            try:
                for index in range(3):
                    yield frame, Fraction(index, 30), Fraction(index + 1, 30), 0, index == 2
            finally:
                closed.set()

        def draw(*_):
            workers.append(threading.current_thread())
            return np.zeros((2, 2, 3), dtype=np.uint8)

        # Keep a separate count because raster and sample complete concurrently.
        count = 0
        def failing_sample(*args):
            nonlocal count
            count += 1
            workers.append(threading.current_thread())
            if count == 2:
                raise RuntimeError('sample failed')
            return np.array([['a']]), None

        with patch('export._video_frames', source):
            prepared = export._prepared_videos(None, Mock(render=Mock(side_effect=failing_sample)),
                Mock(draw=Mock(side_effect=draw)), 1, 1, 'mono', None)
            try:
                next(prepared)
                with self.assertRaisesRegex(RuntimeError, 'sample failed'):
                    next(prepared)
            finally:
                prepared.close()
        self.assertTrue(closed.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in workers))

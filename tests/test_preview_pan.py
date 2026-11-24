"""Shared preview dragging uses captured geometry and bounded pending work."""
from types import SimpleNamespace
import queue
import unittest
from unittest.mock import patch
import test_preview as fixtures

def event(x, y):
    return SimpleNamespace(x=x, y=y)

class PreviewPanTests(unittest.TestCase):
    def setUp(self):
        fixtures.PreviewTests.setUp(self)

    def window(self, zoom='100%'):
        window = fixtures.PreviewTests.window(self)
        window.zoom.get.return_value = zoom
        window.bounds = 200, 100
        window.view = (1000, 500), (400, 200, 200, 100)
        window.source_viewport = (200, 100, 200, 100)
        return window

    def test_fit_click_is_immediate_detail_click_waits_for_release_and_ignores_jitter(self):
        window = self.window('Вписать')
        with patch.object(window, 'inspect') as inspect:
            window.start_pan(event(40, 20), source=True)
            inspect.assert_called_once()
            self.assertIsNone(window._drag)
        window = self.window()
        with patch.object(window, 'inspect') as inspect:
            window.start_pan(event(40, 20), source=True)
            window.pan(event(43, 22))
            inspect.assert_not_called()
            self.assertEqual(window.center, (.5, .5))
            window.finish_pan(event(43, 22))
            inspect.assert_called_once_with(unittest.mock.ANY, source=True)
        self.assertIsNone(window._drag)

    def test_drag_translation_uses_captured_source_or_mp4_viewport_and_zoom(self):
        for source in (False, True):
            for zoom, factor in (('100%', 1), ('200%', 2)):
                window = self.window(zoom)
                window.start_pan(event(100, 50), source=source)
                window.view = window.source_viewport = None
                window.pan(event(140, 70))
                base = (.3, .3) if source else (.5, .5)
                self.assertEqual(window.center, (base[0] - .04 / factor, base[1] - .04 / factor))
                self.assertTrue(window._pan_pending)
                window.job.cancel.assert_not_called()

    def test_edges_use_visible_center_and_small_axes_cannot_pan(self):
        window = self.window()
        window.center = (0, 0)
        window.view = (1000, 50), (0, -25, 200, 100)
        window.start_pan(event(100, 50))
        window.pan(event(90, 40))
        self.assertEqual(window.center, (.11, .5))
        window.pan(event(-10000, -10000))
        self.assertEqual(window.center, (.9, .5))
        window.pan(event(10000, 10000))
        self.assertEqual(window.center, (.1, .5))

    def test_poll_displays_completed_pan_then_requests_only_latest_center(self):
        window = self.window()
        window.start_pan(event(100, 50))
        held_cache = window.cache.copy()
        for x in range(105, 160):
            window.pan(event(x, 50))
        running = window.job
        running.thread.is_alive.return_value = False
        running.stop_event.is_set.return_value = False
        running.result = queue.Queue()
        result = SimpleNamespace(frame_size=(1000, 500), viewport=(400, 200, 200, 100),
            original_viewport=(400, 200, 200, 100), original=object(), converted=object(),
            duration=1., position=.2, columns=320, rows=90)
        running.result.put(('done', result))
        with patch('PIL.ImageTk.PhotoImage'), patch('preview.PreviewJob') as job:
            window.poll()
        running.cancel.assert_not_called()
        running.thread.join.assert_called_once()
        self.assertEqual(job.call_count, 1)
        self.assertEqual(job.call_args.kwargs['center'], (.441, .5))
        self.assertIsNotNone(window._drag)
        self.assertFalse(window._pan_pending)
        self.assertEqual(window.cache, held_cache)
        for label in window.labels:
            label.itemconfigure.assert_called_once()

    def test_release_flushes_final_position_without_extra_worker(self):
        window = self.window()
        window.start_pan(event(100, 50))
        window.pan(event(140, 70))
        running = window.job
        with patch('preview.PreviewJob') as job:
            window.finish_pan(event(140, 70))
        job.assert_not_called()
        running.cancel.assert_called_once()
        self.assertEqual(window.pending[2:], (1, (.46, .46)))
        self.assertIsNone(window._drag)
        self.assertFalse(window._pan_pending)
        with patch.object(window, 'refresh') as refresh:
            window.finish_pan(event(160, 80))
        refresh.assert_not_called()

    def test_release_uses_last_coordinates_even_without_a_motion_event(self):
        window = self.window()
        window.start_pan(event(100, 50))
        window.finish_pan(event(160, 80))
        self.assertEqual(window.center, (.44, .44))
        self.assertEqual(window.pending[2:], (1, (.44, .44)))
        window.job.cancel.assert_called_once()

    def test_external_refresh_resize_close_and_invalid_parameters_end_gesture(self):
        for action in ('refresh', 'resize', 'close', 'invalid'):
            window = self.window()
            window.start_pan(event(100, 50))
            window.pan(event(140, 70))
            center = window.center
            if action == 'resize':
                window._queue_resize(None)
            elif action == 'close':
                window.close()
            else:
                if action == 'invalid':
                    window.app.snapshot = lambda: {'input': str(self.source), 'gamma': 'nan'}
                window.refresh()
            window.pan(event(-1000, -1000))
            window.finish_pan(event(-1000, -1000))
            self.assertEqual(window.center, center)
            self.assertIsNone(window._drag)
            self.assertFalse(window._pan_pending)

    def test_invalid_parameters_during_release_do_not_leave_a_gesture_or_raise(self):
        window = self.window()
        window.job = None
        window.start_pan(event(100, 50))
        window.app.snapshot = lambda: {'input': str(self.source), 'gamma': 'nan'}
        window.finish_pan(event(160, 80))
        self.assertIsNone(window._drag)
        self.assertIsNone(window.pending)
        self.assertFalse(window._pan_pending)

if __name__ == '__main__':
    unittest.main()

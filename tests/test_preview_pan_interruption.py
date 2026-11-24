"""Focus loss, unmap and Escape finish a pan without synthesizing a click."""
import unittest
from unittest.mock import patch
import test_preview_pan as fixtures

class PreviewPanInterruptionTests(unittest.TestCase):
    def setUp(self):
        fixtures.fixtures.PreviewTests.setUp(self)

    def window(self):
        return fixtures.PreviewPanTests.window(self)

    def test_interruption_without_motion_never_selects_a_new_center(self):
        window = self.window()
        window.start_pan(fixtures.event(100, 50))
        with patch.object(window, 'inspect') as inspect, patch.object(window, 'refresh') as refresh:
            window.end_pan()
            window.finish_pan(fixtures.event(190, 95))
            window.pan(fixtures.event(190, 95))
        self.assertEqual(window.center, (.5, .5))
        self.assertIsNone(window._drag)
        self.assertFalse(window._pan_pending)
        inspect.assert_not_called()
        refresh.assert_not_called()
        window.job.cancel.assert_not_called()

    def test_interruption_flushes_only_last_known_center_once_and_preserves_cache(self):
        window = self.window()
        held_cache = window.cache.copy()
        window.start_pan(fixtures.event(100, 50))
        window.pan(fixtures.event(140, 70))
        running = window.job
        with patch('preview.PreviewJob') as job:
            window.end_pan()
            window.end_pan()
            window.pan(fixtures.event(190, 95))
            window.finish_pan(fixtures.event(190, 95))
        running.cancel.assert_called_once()
        job.assert_not_called()
        self.assertEqual(window.center, (.46, .46))
        self.assertEqual(window.pending[1:], (.2, 1, (.46, .46)))
        self.assertEqual(window.cache, held_cache)
        self.assertIsNone(window._drag)
        self.assertFalse(window._pan_pending)

    def test_ready_view_restores_cursor_and_unrelated_focus_events_do_nothing(self):
        window = self.window()
        window.job = None
        window.start_pan(fixtures.event(100, 50))
        with patch.object(window, 'refresh') as refresh:
            window.end_pan()
            window.end_pan()
        refresh.assert_not_called()
        for label in window.labels:
            label.configure.assert_called_once_with(cursor='crosshair')
        window = self.window()
        window.closing = True
        with patch.object(window, 'refresh') as refresh:
            window.end_pan()
        refresh.assert_not_called()
        window.job.cancel.assert_not_called()

if __name__ == '__main__':
    unittest.main()

"""Cross-platform wheel zoom preserves the selected preview and bounded queue."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_preview as fixtures


class PreviewWheelTests(unittest.TestCase):
    def setUp(self):
        fixtures.PreviewTests.setUp(self)

    def window(self, zoom='Вписать'):
        window = fixtures.PreviewTests.window(self)
        value = [zoom]
        window.zoom.get.side_effect = lambda: value[0]
        window.zoom.set.side_effect = lambda new: value.__setitem__(0, new)
        window.center = (.2, .7)
        return window

    def test_vertical_events_step_once_clamp_and_preserve_center(self):
        for event in (SimpleNamespace(delta=120), SimpleNamespace(delta=1),
                      SimpleNamespace(delta=240), SimpleNamespace(num=4, delta=0)):
            window = self.window()
            with patch.object(window, 'refresh') as refresh:
                for expected in ('100%', '200%', '200%'):
                    self.assertEqual(window.wheel_zoom(event), 'break')
                    self.assertEqual(window.zoom.get(), expected)
                    self.assertEqual(window.center, (.2, .7))
                self.assertEqual(refresh.call_count, 2)
        for event in (SimpleNamespace(delta=-120), SimpleNamespace(delta=-1),
                      SimpleNamespace(delta=-240), SimpleNamespace(num=5, delta=0)):
            window = self.window('200%')
            with patch.object(window, 'refresh') as refresh:
                for expected in ('100%', 'Вписать', 'Вписать'):
                    self.assertEqual(window.wheel_zoom(event), 'break')
                    self.assertEqual(window.zoom.get(), expected)
                self.assertEqual(refresh.call_count, 2)

    def test_horizontal_zero_and_late_events_do_not_request_a_frame(self):
        window = self.window()
        with patch.object(window, 'refresh') as refresh:
            for event in (SimpleNamespace(delta=0), SimpleNamespace(num=6),
                          SimpleNamespace(delta=120, state=1),
                          SimpleNamespace(num=4, state=1)):
                self.assertIsNone(window.wheel_zoom(event))
            window.closing = True
            window.wheel_zoom(SimpleNamespace(delta=120))
        refresh.assert_not_called()
        window.zoom.set.assert_not_called()

    def test_wheel_burst_replaces_pending_request_without_stale_hit_maps(self):
        window = self.window()
        running = window.job
        held_cache = window.cache.copy()
        window.view = None
        window.source_viewport = None
        with patch('preview.PreviewJob') as job:
            for delta in (120, 120, -120, 120):
                window.wheel_zoom(SimpleNamespace(delta=delta))
        job.assert_not_called()
        self.assertIs(window.job, running)
        self.assertEqual(running.cancel.call_count, 4)
        self.assertEqual(window.pending[1:], (.2, 2, (.2, .7)))
        self.assertEqual(window.zoom.get(), '200%')
        self.assertIsNone(window.view)
        self.assertIsNone(window.source_viewport)
        self.assertEqual(window.cache, held_cache)


if __name__ == '__main__':
    unittest.main()

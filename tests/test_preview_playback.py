from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np

from gui import build_arguments
from preview import PreviewPlaybackJob, make_preview
from test_export import fixture


class PreviewPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name) / 'movie.mp4'
        fixture(self.source, audio_offset=0)
        _, self.options = build_arguments('preview', {'input': str(self.source),
            'output_width': 128, 'output_height': 64, 'columns': 16, 'color': 'source'})

    def finish(self, job):
        job.thread.join(10)
        if job.thread.is_alive():
            job.cancel()
            job.thread.join(10)
            self.fail('Playback failed to finish')
        kind, value = job.result.get_nowait()
        self.assertEqual(kind, 'done', value)

    def test_muted_sequential_playback_matches_static_raster_and_is_bounded(self):
        job = PreviewPlaybackJob(self.options, 0, (96, 48))
        with patch('audio.AudioPlayer', side_effect=AssertionError('Muted preview opened audio')):
            job.start()
            self.finish(job)
        self.assertEqual(job.frames.qsize(), 1)
        last = job.frames.get_nowait()
        self.assertEqual(last.position, last.duration)
        expected = make_preview(self.options, 1, (96, 48))
        np.testing.assert_array_equal(last.full_frame, expected.full_frame)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [self.source])

    def test_moving_frames_match_random_access_with_video_delay_and_audio_tail(self):
        fixture(self.source, video_pts=[3, 4, 7, 9], audio_offset=0, audio_samples=28800)
        observed = []
        def render(*args, **kwargs):
            value = make_preview(*args, **kwargs)
            observed.append((value.position / value.duration, np.asarray(value.full_frame).copy()))
            return value
        with patch('preview.make_preview', side_effect=render):
            job = PreviewPlaybackJob(self.options, 0, (96, 48))
            job.start()
            self.finish(job)
        self.assertGreater(len(observed), 3)
        for fraction, raster in observed:
            expected = make_preview(self.options, fraction, (96, 48))
            np.testing.assert_array_equal(raster, expected.full_frame)

    def test_sound_uses_media_clock_and_closes_device(self):
        fake = Mock(error=None)
        fake.time.return_value = 1000
        with patch('audio.AudioPlayer', return_value=fake) as factory:
            job = PreviewPlaybackJob(self.options, .25, (96, 48), sound=True)
            job.start()
            self.finish(job)
        factory.assert_called_once()
        fake.prepare.assert_called_once_with()
        fake.start.assert_called_once_with()
        fake.close.assert_called_once_with()
        self.assertGreater(fake.seek_time, 0)
        self.assertEqual(job.frames.get_nowait().position,
                         make_preview(self.options, 1).duration)

    def test_cancel_releases_worker_without_consumer(self):
        job = PreviewPlaybackJob(self.options, 0, (96, 48))
        job.start()
        job.frames.get(timeout=10)
        job.cancel()
        job.thread.join(10)
        self.assertFalse(job.thread.is_alive())
        self.assertIn(job.result.get_nowait()[0], ('done', 'cancelled'))

    def test_device_failure_is_reported_and_closed(self):
        fake = Mock(error=None)
        fake.prepare.side_effect = RuntimeError('device unavailable')
        with patch('audio.AudioPlayer', return_value=fake):
            job = PreviewPlaybackJob(self.options, 0, (96, 48), sound=True)
            job.start()
            job.thread.join(10)
        self.assertFalse(job.thread.is_alive())
        self.assertEqual(job.result.get_nowait(), ('error', 'device unavailable'))
        fake.close.assert_called_once()


    def test_gui_pause_seek_sound_toggle_and_close(self):
        import tkinter as tk
        from gui import Application
        from ttkbootstrap import Style
        from ttkbootstrap.publisher import Publisher
        # Other GUI modules create and destroy independent Tcl interpreters.
        Style.instance = None
        Publisher.clear_subscribers()
        root = tk.Tk()
        root.withdraw()
        app = Application(root)
        app.values['input'].set(str(self.source))
        app.values['output_width'].set('128')
        app.values['output_height'].set('64')
        app.values['columns'].set('16')
        app.open_preview()
        preview = app.preview

        def wait_for(predicate):
            deadline = time.monotonic() + 10
            while not predicate() and time.monotonic() < deadline:
                root.update()
                time.sleep(.005)
            self.assertTrue(predicate())

        try:
            wait_for(lambda: preview.job is None)
            self.assertFalse(preview.preview_sound.get())
            preview.toggle_playback()
            self.assertIsNotNone(preview.playback)
            preview.toggle_playback()
            wait_for(lambda: preview.playback is None)
            preview.percent.set('50')
            preview.seek_percent()
            wait_for(lambda: preview.job is None)
            self.assertAlmostEqual(preview.position.get(), 50)
            fake = Mock(error=None)
            fake.time.return_value = .25
            with patch('audio.AudioPlayer', return_value=fake):
                preview.toggle_playback()
                preview.preview_sound.set(True)
                preview.change_preview_sound()
                wait_for(lambda: fake.start.called)
                preview.close()
                wait_for(lambda: app.preview is None)
                fake.close.assert_called_once()
        finally:
            app.close()
            if app.preview is not None:
                wait_for(lambda: app.preview is None)

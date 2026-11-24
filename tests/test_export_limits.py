from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from cli_options import ExportLimitError, parser
from export import _check_export_span, _normalized_audio, render
from gui import build_arguments, ExportJob, Application
from media import probe
from test_export import fixture


class ExportLimitTests(unittest.TestCase):
    def test_cli_consent_is_explicit_and_render_only(self):
        command = ['render', 'input.mp4', '--output', 'output.mp4']
        self.assertFalse(parser().parse_args(command).allow_long_export)
        self.assertTrue(parser().parse_args([*command, '--allow-long-export']).allow_long_export)

    def test_thresholds_and_explicit_consent(self):
        _check_export_span(0, 60)
        _check_export_span(21600, 21600)
        for start, end in ((0, Fraction(60001, 1000)), (21600, Fraction(648001, 30))):
            with self.assertRaises(ExportLimitError):
                _check_export_span(start, end)
            _check_export_span(start, end, True)

    def test_tiny_day_long_video_rejected_even_with_unknown_or_false_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.mp4'
            output = Path(directory) / 'output.mp4'
            fixture(source, video_pts=[0, 30 * 86400])
            self.assertLess(source.stat().st_size, 4096)
            _, options = build_arguments('render', dict(input=str(source), output=str(output),
                output_width=128, output_height=64, columns=16, audio=False, overwrite=True))
            info = probe(source, audio=False)
            for duration in (info.duration, None, 1):
                with self.subTest(duration=duration):
                    output.write_bytes(b'previous output')
                    with patch('export.probe', return_value=replace(info, duration=duration)), \
                            patch('export._encode_video', side_effect=AssertionError('Must reject before encoding')), \
                            self.assertRaises(ExportLimitError):
                        render(options)
                    self.assertEqual(output.read_bytes(), b'previous output')
                    self.assertFalse(list(Path(directory).glob('.output.mp4.*')))

    def test_long_initial_video_delay_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.mp4'
            fixture(source, video_pts=[30 * 61, 30 * 61 + 1], audio_offset=0)
            _, options = build_arguments('render', dict(input=str(source),
                output=str(Path(directory) / 'output.mp4'), output_width=128, output_height=64, columns=16))
            with self.assertRaises(ExportLimitError):
                render(options)

    def test_audio_gap_rejected_before_allocating_silence_and_can_be_confirmed(self):
        block = av.AudioFrame.from_ndarray(np.zeros((2, 32), dtype=np.float32), format='fltp', layout='stereo')
        block.sample_rate = 48000
        block.time_base = Fraction(1, 48000)
        block.pts = 48000 * 61
        with patch('export._audio_frame', side_effect=AssertionError('No silence allocation')), \
                self.assertRaises(ExportLimitError):
            next(_normalized_audio(block, 0, Fraction(0)))
        permitted = _normalized_audio(block, 0, Fraction(0), allow_long_export=True)
        self.assertEqual(next(permitted).samples, 1024)
        permitted.close()

    def test_consent_allows_same_video_under_a_lower_test_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.mp4'
            fixture(source)
            _, options = build_arguments('render', dict(input=str(source),
                output=str(Path(directory) / 'output.mp4'), output_width=128, output_height=64, columns=16, audio=False))
            with patch('export.MAX_EXPORT_SECONDS', 0), self.assertRaises(ExportLimitError):
                render(options)
            options.allow_long_export = True
            with patch('export.MAX_EXPORT_SECONDS', 0):
                render(options, progress=lambda *_: None)
            with av.open(str(options.output)) as container:
                self.assertEqual(len(list(container.decode(video=0))), 12)

    def test_gui_confirmation_reuses_snapshot_and_never_retries_after_cancel(self):
        for consent, cancelled, closing in ((True, False, False), (False, False, False),
                                            (True, True, False), (True, False, True)):
            with self.subTest(consent=consent, cancelled=cancelled, closing=closing):
                options = parser().parse_args(['render', 'input.mp4', '--output', 'output.mp4'])
                job = ExportJob(options)
                with patch('export.render', side_effect=ExportLimitError('large timeline')):
                    job._run()
                job.thread = Mock()
                job.thread.is_alive.return_value = False
                if cancelled:
                    job.cancel()
                app = Application.__new__(Application)
                app.job, app.closing, app.preview = job, closing, None
                app.root, app.status, app.messages = Mock(), Mock(), Mock()
                app.play_button, app.save_button, app.cancel_button = Mock(), Mock(), Mock()
                app.show_progress, app._start_export = Mock(), Mock()
                app.messages.askyesno.return_value = consent
                app.poll()
                if consent and not cancelled and not closing:
                    app._start_export.assert_called_once_with(options)
                    self.assertTrue(options.allow_long_export)
                else:
                    app._start_export.assert_not_called()
                    self.assertFalse(options.allow_long_export)
                if cancelled or closing:
                    app.messages.askyesno.assert_not_called()


if __name__ == '__main__':
    unittest.main()

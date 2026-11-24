"""Retained audio decoder with real resampling and fake output devices."""

from fractions import Fraction
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from check_container_seeking import pcm
from cursor_audio_adapter import experimental_module, pcm_window
from media import probe
from audio_seek_prototype import insert_gap
from sequential_audio_cursor import SequentialAudioCursor
from test_formats import FORMATS, fixture
from test_seeking import SilentDevice


class CursorAudioTests(unittest.TestCase):
    def test_bounded_history_replays_nearby_requests_and_survives_gap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source, path = Path(folder) / "source.mkv", Path(folder) / "history.mkv"
            fixture(source, duration=8, tone=True, layout="stereo")
            insert_gap(source, path, 3, .25)
            info = probe(path)
            reference, _ = pcm(path, info, 0, 44100, channels=2)
            with SequentialAudioCursor(path, info.audio_index, history_seconds=1) as cursor:
                module = experimental_module(cursor)
                for target in (2.7, 2.8, 3.350125, 3.45, 5.7, 5.8, .175, 5.7):
                    with self.subTest(target=target):
                        opens = cursor.opens
                        actual, _ = pcm_window(module, path, info, target, 44100)
                        start = round(target * 44100)
                        np.testing.assert_array_equal(actual, reference[start:start + round(.2 * 44100)])
                        if target in (2.8, 3.45, 5.8):
                            self.assertEqual(cursor.opens, opens)
                        if cursor._recent:
                            first = cursor._recent[0][0]
                            self.assertLessEqual(cursor.end - Fraction(first.pts, first.sample_rate), 1)
                cursor.close()
                self.assertFalse(cursor._recent)

    def test_full_tail_and_eof_after_repeated_seeks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            for extension, video, audio in FORMATS:
                path = Path(folder) / ("tail." + extension)
                fixture(path, video, audio, duration=3, tone=True, layout="stereo")
                info = probe(path)
                for rate in (44100, 48000):
                    reference, end = pcm(path, info, 0, rate, channels=2)
                    with SequentialAudioCursor(path, info.audio_index, history_seconds=1) as cursor:
                        module = experimental_module(cursor)
                        for target in (.175, 2.350125, .175, end - .025, end):
                            with self.subTest(extension=extension, rate=rate, target=target):
                                with patch("check_container_seeking.AudioPlayer", module.AudioPlayer):
                                    actual, actual_end = pcm(path, info, target, rate, channels=2)
                                np.testing.assert_array_equal(actual, reference[round(target * rate):])
                                self.assertEqual(actual_end, end)

    def test_gap_resets_resampler_and_preserves_media_clock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source, path = Path(folder) / "source.mkv", Path(folder) / "gap.mkv"
            fixture(source, duration=6, tone=True, layout="stereo")
            insert_gap(source, path, 2, .25)
            info = probe(path)
            reference, _ = pcm(path, info, 0, 44100, channels=2)
            with SequentialAudioCursor(path, info.audio_index) as cursor:
                module = experimental_module(cursor)
                for target in (1.95, 2.05, 2.350125, 4.7):
                    with self.subTest(target=target):
                        actual, _ = pcm_window(module, path, info, target, 44100)
                        start = round(target * 44100)
                        np.testing.assert_array_equal(actual, reference[start:start + round(.2 * 44100)])

    def test_formats_rates_channels_pause_and_repeated_requests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            for extension, video, audio in FORMATS:
                path = Path(folder) / ("cursor." + extension)
                fixture(path, video, audio, duration=6, tone=True, layout="stereo",
                        audio_options={"aac_pns": "1"} if audio == "aac" else None)
                info = probe(path)
                for rate in (44100, 48000):
                    for channels in (1, 2):
                        reference, _ = pcm(path, info, 0, rate, channels=channels)
                        if channels == 1:
                            reference = reference[:, None]
                        with SequentialAudioCursor(path, info.audio_index) as cursor:
                            module = experimental_module(cursor)
                            for target in (0, .175, 3.350125, .175, 4.7):
                                with self.subTest(extension=extension, rate=rate, channels=channels, target=target):
                                    actual, metrics = pcm_window(module, path, info, target, rate,
                                                                 channels=channels, pause=True)
                                    start = round(target * rate)
                                    np.testing.assert_array_equal(actual, reference[start:start + round(.2 * rate)])
                                    self.assertLessEqual(metrics["peak_queued_samples"], 2 * rate)

    def test_cancel_preparation_then_reuse_cursor_without_stale_pcm(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "cancel.mp4"
            fixture(path, duration=8, tone=True, layout="stereo")
            info = probe(path)
            reference, _ = pcm(path, info, 0, 44100, channels=2)
            with SequentialAudioCursor(path, info.audio_index) as cursor:
                module = experimental_module(cursor)
                audio = module.AudioPlayer(str(path), info.audio_index, info.origin, threading.Event())
                audio.seek_time = 3.350125
                held, release = threading.Event(), threading.Event()
                frames = cursor.frames

                def delayed(*args, **kwargs):
                    for frame in frames(*args, **kwargs):
                        held.set()
                        if not release.wait(3):
                            raise RuntimeError("test decoder watchdog")
                        yield frame

                try:
                    with patch("audio.sd.query_devices", return_value={"max_output_channels": 2,
                                                                     "default_samplerate": 44100}), \
                            patch("audio.sd.OutputStream", SilentDevice), patch.object(cursor, "frames", delayed):
                        audio.prepare(wait=False)
                        self.assertTrue(held.wait(3))
                        audio._cancel.set()
                        release.set()
                        audio.close()
                    self.assertEqual(audio._written, 0)
                    self.assertIsNone(audio.error)
                    self.assertIsNone(audio._producer)
                    actual, _ = pcm_window(module, path, info, 5.350125, 44100)
                    start = round(5.350125 * 44100)
                    np.testing.assert_array_equal(actual, reference[start:start + round(.2 * 44100)])
                    self.assertEqual(cursor.opens, 1)
                finally:
                    release.set()
                    audio.close()

    def test_decoder_failure_reaches_player_and_closes_device(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "error.mp4"
            fixture(path, duration=1, tone=True)
            info = probe(path)
            with SequentialAudioCursor(path, info.audio_index) as cursor:
                module = experimental_module(cursor)
                audio = module.AudioPlayer(str(path), info.audio_index, info.origin, threading.Event())
                device = SilentDevice()
                try:
                    with patch("audio.sd.query_devices", return_value={"max_output_channels": 2,
                                                                     "default_samplerate": 44100}), \
                            patch("audio.sd.OutputStream", return_value=device), \
                            patch.object(cursor, "frames", side_effect=RuntimeError("decoder failed")):
                        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                            audio.prepare()
                    self.assertTrue(device.closed)
                    self.assertTrue(audio.stop_event.is_set())
                    self.assertIsNone(audio._producer)
                finally:
                    audio.close()

    def test_cancel_when_pcm_ring_is_full_then_reuse_decoder(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "full.mp4"
            fixture(path, duration=8, tone=True, layout="stereo")
            info = probe(path)
            reference, _ = pcm(path, info, 0, 44100, channels=2)
            with SequentialAudioCursor(path, info.audio_index) as cursor:
                module = experimental_module(cursor)
                audio = module.AudioPlayer(str(path), info.audio_index, info.origin, threading.Event())
                waiting = threading.Event()
                wait_capacity = audio._wait_capacity

                def observed_wait(count):
                    if audio._capacity - (audio._written - audio._read) < count:
                        waiting.set()
                    return wait_capacity(count)

                try:
                    with patch("audio.sd.query_devices", return_value={"max_output_channels": 2,
                                                                     "default_samplerate": 44100}), \
                            patch("audio.sd.OutputStream", SilentDevice), \
                            patch.object(audio, "_wait_capacity", observed_wait):
                        audio.prepare(wait=False)
                        self.assertTrue(waiting.wait(3))
                        self.assertLessEqual(audio._written - audio._read, 2 * 44100)
                        audio.close()
                    self.assertIsNone(audio.error)
                    self.assertIsNone(audio._producer)
                    actual, _ = pcm_window(module, path, info, 5.350125, 44100)
                    start = round(5.350125 * 44100)
                    np.testing.assert_array_equal(actual, reference[start:start + round(.2 * 44100)])
                    self.assertEqual(cursor.opens, 1)
                finally:
                    audio.close()


if __name__ == "__main__":
    unittest.main()

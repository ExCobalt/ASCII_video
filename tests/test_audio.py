from contextlib import closing
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from audio import AudioPlayer


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.time = 100.0
        self.closed = False
        self.started = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.started = True
        self.start_calls += 1

    def stop(self, ignore_errors=True):
        self.started = False
        self.stop_calls += 1

    def close(self):
        self.closed = True


class AudioTests(unittest.TestCase):
    def make_player(self, rate=10, channels=1):
        player = AudioPlayer("unused", 0, 0, threading.Event())
        player.rate = rate
        player.channels = channels
        player._capacity = 2 * rate
        player._ring = np.empty((2 * rate, channels), np.float32)
        player._stream = FakeStream(callback=player._callback)
        self.addCleanup(player.close)
        return player

    def block(self, start, values, rate=10):
        frame = av.AudioFrame.from_ndarray(
            np.array([values], dtype=np.float32), format="flt", layout="mono"
        )
        frame.sample_rate = rate
        frame.time_base = Fraction(1, rate)
        frame.pts = start
        return frame

    def callback(self, player, count, underflow=False, dac=100):
        output = np.empty((count, player.channels), np.float32)
        player._callback(
            output, count, SimpleNamespace(outputBufferDacTime=dac),
            SimpleNamespace(output_underflow=underflow),
        )
        return output

    def test_gap_eof_drain_and_dac_clock(self):
        player = self.make_player()
        end = player._enqueue(self.block(3, [.2, .3, .4]), 0)
        player._eof = True
        player._end_time = end / player.rate
        np.testing.assert_allclose(self.callback(player, 5)[:, 0], [0, 0, 0, .2, .3])
        self.assertEqual(player.time(), 0)
        player._stream.time = 100.25
        self.assertAlmostEqual(player.time(), .25)
        np.testing.assert_allclose(self.callback(player, 5, dac=100.5)[:, 0], [.4, 0, 0, 0, 0])
        self.assertEqual(player.end_time, .6)
        self.assertIsNone(player.error)
        self.assertEqual(player._read, 3)

    def test_underrun_and_device_underflow_signal_stop(self):
        for underflow in (False, True):
            with self.subTest(underflow=underflow):
                player = self.make_player()
                output = self.callback(player, 2, underflow)
                self.assertTrue(player.stop_event.is_set())
                self.assertIsInstance(player.error, RuntimeError)
                np.testing.assert_array_equal(output, 0)

    def test_clock_tracks_dac_latency_not_prequeued_pcm(self):
        player = self.make_player(rate=100)
        player._enqueue(self.block(0, [1] * 200, rate=100), 0)
        player._eof = True
        player._stream.time = 100
        self.callback(player, 10, dac=100.2)
        self.assertEqual(player.time(), 0)
        player._stream.time = 100.25
        self.assertAlmostEqual(player.time(), .05)
        # The next callback may run early; its DAC timestamp retains the clock.
        self.callback(player, 10, dac=100.3)
        self.assertAlmostEqual(player.time(), .05)
        player._stream.time = 100.4
        self.assertAlmostEqual(player.time(), .2)
        self.assertGreater(player._written / player.rate, player.time())
        # Silence after EOF continues the same clock without resetting it.
        self.callback(player, 200, dac=100.4)
        player._stream.time = 102.5
        self.assertAlmostEqual(player.time(), 2.3)
        self.assertIsNone(player.error)

    def test_pause_drains_freezes_clock_and_resume_reanchors_without_losing_pcm(self):
        player = self.make_player(rate=10)
        player._enqueue(self.block(0, list(range(20))), 0)
        player._eof = True
        player.start()
        np.testing.assert_array_equal(self.callback(player, 5)[:, 0], list(range(5)))
        stream = player._stream
        original_stop = stream.stop

        def stop_and_drain(ignore_errors=True):
            self.assertFalse(ignore_errors)
            # Simulate the final callback consumed during PortAudio stop/drain.
            np.testing.assert_array_equal(self.callback(player, 2, dac=100.5)[:, 0], [5, 6])
            original_stop(ignore_errors)

        stream.stop = stop_and_drain
        player.pause()
        self.assertTrue(player.paused)
        self.assertEqual(player.time(), .7)
        self.assertEqual(player._read, 7)
        self.assertFalse(player.stop_event.is_set())
        stream.time = 1000
        self.assertEqual(player.time(), .7)
        player.pause()
        self.assertEqual(stream.stop_calls, 1)
        player.resume()
        self.assertFalse(player.paused)
        self.assertEqual(player.time(), .7)
        self.assertEqual(stream.start_calls, 2)
        player.resume()
        self.assertEqual(stream.start_calls, 2)
        np.testing.assert_array_equal(self.callback(player, 3, dac=1000.2)[:, 0], [7, 8, 9])
        self.assertEqual(player.time(), .7)  # New DAC buffer has not sounded yet.
        stream.time = 1000.35
        self.assertAlmostEqual(player.time(), .85)
        self.assertIsNone(player.error)

    def test_pause_resume_device_failures_are_reported_and_stop_producer(self):
        for method in ("pause", "resume"):
            with self.subTest(method=method):
                player = self.make_player()
                if method == "pause":
                    player._stream.stop = unittest.mock.Mock(side_effect=RuntimeError("stop failed"))
                else:
                    player._paused = True
                    player._stream.start = unittest.mock.Mock(side_effect=RuntimeError("start failed"))
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    getattr(player, method)()
                self.assertIsInstance(player.error, RuntimeError)
                self.assertTrue(player.stop_event.is_set())

    def test_paused_pcm_producer_stays_bounded_and_close_cancels_it(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "pause.wav"
            self.write_audio(path, frames=50)
            with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 1, "default_samplerate": 8000,
            }), patch("audio.sd.OutputStream", FakeStream):
                player = AudioPlayer(path, 0, 0, threading.Event())
                self.addCleanup(player.close)
                player.prepare()
                player.start()
                self.callback(player, 800)
                player.pause()
                time.sleep(.03)
                self.assertFalse(player.stop_event.is_set())
                self.assertTrue(player.paused)
                self.assertEqual(player.time(), .1)
                self.assertLessEqual(player._written - player._read, 16000)
                producer = player._producer
                player.close()
                player.close()
                self.assertFalse(producer.is_alive())

    def test_origin_and_overlap_tolerance(self):
        player = self.make_player()
        player.origin = Fraction(2)
        end = player._enqueue(self.block(20, [1, 2, 3]), 0)
        self.assertEqual(end, 3)
        self.assertEqual(player._enqueue(self.block(22, [4, 5]), end), 4)
        with self.assertRaisesRegex(RuntimeError, "Overlapping"):
            player._enqueue(self.block(20, [6]), 4)
        player._eof = True
        np.testing.assert_array_equal(self.callback(player, 4)[:, 0], [1, 2, 3, 5])

    def test_capacity_and_cancel_wake(self):
        player = self.make_player()
        player._enqueue(self.block(0, list(range(20))), 0)
        waiting = threading.Thread(target=player._wait_capacity, args=(1,))
        waiting.start()
        time.sleep(.02)
        self.assertTrue(waiting.is_alive())
        player.stop_event.set()
        waiting.join(1)
        self.assertFalse(waiting.is_alive())
        self.assertEqual(player._written - player._read, player._capacity)
        with self.assertRaisesRegex(RuntimeError, "PCM limit"):
            player._wait_capacity(21)

    def test_close_joins_producer_even_if_device_close_fails(self):
        player = self.make_player()
        player._written = player._capacity
        producer = threading.Thread(target=player._wait_capacity, args=(1,))
        player._producer = producer
        producer.start()
        player._stream.close = unittest.mock.Mock(side_effect=RuntimeError("device gone"))
        with self.assertRaisesRegex(RuntimeError, "device gone"):
            player.close()
        self.assertFalse(producer.is_alive())
        self.assertIsNone(player._stream)

    def write_audio(self, path, frames=8):
        with av.open(str(path), "w") as container:
            stream = container.add_stream("pcm_s16le", rate=8000)
            stream.layout = "mono"
            for i in range(frames):
                frame = av.AudioFrame.from_ndarray(
                    np.full((1, 800), 4096, dtype=np.int16), format="s16", layout="mono"
                )
                frame.sample_rate = 8000
                frame.time_base = Fraction(1, 8000)
                frame.pts = i * 800
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def test_real_decode_resample_prebuffer_and_eof(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sound.wav"
            self.write_audio(path)
            with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 2, "default_samplerate": 16000,
            }), patch("audio.sd.OutputStream", FakeStream):
                player = AudioPlayer(path, 0, 0, threading.Event())
                self.addCleanup(player.close)
                player.prepare()
                player.start()
                self.assertTrue(player._stream.started)
                deadline = time.monotonic() + 2
                while not player.eof and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertTrue(player.eof)
                self.assertIsNone(player.error)
                self.assertAlmostEqual(player.end_time, .8, places=4)
                self.assertLessEqual(player._written - player._read, 32000)
                output = self.callback(player, 12800)
                self.assertTrue(np.all(output > 0))
                np.testing.assert_array_equal(self.callback(player, 100), 0)
                self.assertIsNone(player.error)

    def test_short_audio_eof_before_half_second(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "short.wav"
            self.write_audio(path, frames=1)
            with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 1, "default_samplerate": 8000,
            }), patch("audio.sd.OutputStream", FakeStream):
                player = AudioPlayer(path, 0, 0, threading.Event())
                self.addCleanup(player.close)
                player.prepare()
                self.assertTrue(player.eof)
                self.assertEqual(player.end_time, .1)

    def test_resampler_preserves_start_delay_and_gap(self):
        frames = [self.block(8000, [.5] * 800, 8000),
                  self.block(16000, [.5] * 800, 8000)]
        container = SimpleNamespace(
            streams=[SimpleNamespace(type="audio")],
            decode=lambda stream: iter(frames),
            close=unittest.mock.Mock(),
        )
        with patch("audio_cursor.open_media", return_value=closing(container)), patch(
            "audio.sd.query_devices", return_value={
                "max_output_channels": 1, "default_samplerate": 16000,
            }
        ), patch("audio.sd.OutputStream", FakeStream):
            player = AudioPlayer("fake", 0, 0, threading.Event())
            self.addCleanup(player.close)
            player.prepare()
            deadline = time.monotonic() + 2
            while not player.eof and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(player.eof, str(player.error))
            self.assertAlmostEqual(player.end_time, 2.1)
            out = self.callback(player, 33600)[:, 0]
            np.testing.assert_array_equal(out[:16000], 0)
            self.assertTrue(np.all(out[16000:17600] > .4))
            np.testing.assert_array_equal(out[17600:32000], 0)
            self.assertTrue(np.all(out[32000:] > .4))

    def test_long_producer_bounded_and_close_cancels_wait(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "long.wav"
            self.write_audio(path, frames=50)
            with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 1, "default_samplerate": 8000,
            }), patch("audio.sd.OutputStream", FakeStream):
                player = AudioPlayer(path, 0, 0, threading.Event())
                self.addCleanup(player.close)
                player.prepare()
                time.sleep(.03)
                self.assertFalse(player.eof)
                self.assertLessEqual(player._written - player._read, 16000)
                producer = player._producer
                player.close()
                self.assertFalse(producer.is_alive())

    def test_open_device_and_decoder_errors_propagate(self):
        with patch("audio.sd.query_devices", side_effect=RuntimeError("no device")):
            player = AudioPlayer("missing", 0, 0, threading.Event())
            with self.assertRaisesRegex(RuntimeError, "no device"):
                player.prepare()
            self.assertTrue(player.stop_event.is_set())
        with patch("audio.sd.query_devices", return_value={
            "max_output_channels": 1, "default_samplerate": 8000,
        }), patch("audio.sd.OutputStream", FakeStream):
            player = AudioPlayer("missing", 0, 0, threading.Event())
            with self.assertRaises(Exception):
                player.prepare()
            self.assertIsNotNone(player.error)


if __name__ == "__main__":
    unittest.main()

"""Real demux/decode seeks; the audio output device is always replaced."""

from fractions import Fraction
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

from audio import AudioPlayer
from media import VideoDecoder, probe, timestamp, video_intervals


class SilentDevice:
    def __init__(self, **kwargs):
        self.time = 100.0
        self.closed = False

    def start(self):
        pass

    def stop(self, ignore_errors=True):
        pass

    def close(self):
        self.closed = True


class CaptureRGB:
    def sample(self, rgb, width, height):
        return rgb.copy()


class SeekingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        device = patch("audio.sd.query_devices", return_value={
            "max_output_channels": 1, "default_samplerate": 8000,
        })
        device.start()
        self.addCleanup(device.stop)
        stream = patch("audio.sd.OutputStream", SilentDevice)
        stream.start()
        self.addCleanup(stream.stop)

    def video_file(self):
        path = self.directory / "seek vfr.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
            stream.codec_context.time_base = Fraction(1, 30)
            stream.options = {"preset": "veryfast", "g": "4", "bf": "2"}
            for index, pts in enumerate((0, 3, 6, 12, 20, 30, 45, 60)):
                pixels = np.full((32, 64, 3), 20 + index * 25, np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = pts, Fraction(1, 30)
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        return path

    def decode_video(self, info, target, stop=None):
        stop = stop if stop is not None else threading.Event()
        decoder = VideoDecoder(info, CaptureRGB(), (8, 4), lambda: target, stop)
        decoder.seek_time = target
        self.addCleanup(decoder.close)
        decoder.start()
        frames = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                frames.append(decoder.queue.get(timeout=.01))
            except queue.Empty:
                if decoder.eof or decoder.error:
                    break
        decoder.close()
        self.assertIsNone(decoder.error)
        self.assertTrue(decoder.eof, "video decoder failed to finish")
        self.assertFalse(stop.is_set(), "closing a seek decoder stopped the session")
        return decoder, frames

    def test_video_seek_forward_backward_vfr_gop_matches_full_decode(self):
        path = self.video_file()
        info = probe(path, audio=False)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            expected = [(float(start), float(end), frame.to_ndarray(format="rgb24"), last)
                        for frame, start, end, last in
                        video_intervals(container.decode(stream), stream.average_rate)]
        for target in (1.25, .15, 1.75, 0.0, .4):
            with self.subTest(target=target):
                decoder, frames = self.decode_video(info, target)
                wanted = [entry for entry in expected if entry[1] > target or entry[3]]
                self.assertEqual(len(frames), len(wanted))
                for actual, (start, end, pixels, last) in zip(frames, wanted):
                    self.assertEqual((actual.start, actual.end, actual.last), (start, end, last))
                    np.testing.assert_array_equal(actual.grid, pixels)
                self.assertAlmostEqual(decoder.end_time, expected[-1][1])

    def test_video_seek_into_audio_tail_keeps_final_image(self):
        info = probe(self.video_file(), audio=False)
        with av.open(str(info.path)) as container:
            last = list(container.decode(video=0))[-1]
            expected = last.to_ndarray(format="rgb24")
            final_start = float(timestamp(last))
        decoder, frames = self.decode_video(info, final_start + 5)
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].last)
        np.testing.assert_array_equal(frames[0].grid, expected)
        np.testing.assert_array_equal(decoder.last_rgb, expected)

    def test_full_video_queue_close_allows_next_seek_with_same_session_stop(self):
        info = probe(self.video_file(), audio=False)
        stop = threading.Event()
        decoder = VideoDecoder(info, CaptureRGB(), (8, 4), lambda: 0, stop)
        self.addCleanup(decoder.close)
        decoder.start()
        deadline = time.monotonic() + 3
        while not decoder.queue.full() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(decoder.queue.full())
        decoder.close()
        self.assertFalse(decoder.thread.is_alive())
        self.assertFalse(stop.is_set())
        _, frames = self.decode_video(info, 1.25, stop)
        self.assertTrue(frames)

    def audio_file(self, rate=8000, channels=1, codec="pcm_s16le"):
        path = self.directory / ("seek audio.m4a" if codec == "aac" else "seek audio.wav")
        indices = np.arange(round(rate * .95))
        signal = (12000 * np.sin(indices * .073) + 2000 * np.sin(indices * .013)).astype(np.int16)
        with av.open(str(path), "w") as container:
            stream = container.add_stream(codec, rate=rate)
            stream.layout = "mono" if channels == 1 else "stereo"
            if codec == "aac":
                # PNS uses decoder random state, which legitimately differs
                # after seeking. Disable it to compare the sample-clock phase.
                stream.options = {"aac_pns": "0"}
            for cursor in range(0, len(signal), 317):
                values = np.repeat(signal[cursor:cursor + 317, None], channels, axis=1)
                if codec == "aac":
                    values = np.ascontiguousarray(values.T.astype(np.float32) / 32768)
                    format = "fltp"
                else:
                    values = values.reshape(1, -1)
                    format = "s16"
                frame = av.AudioFrame.from_ndarray(values, format=format,
                                                 layout=stream.layout.name)
                frame.sample_rate = rate
                frame.pts, frame.time_base = cursor, Fraction(1, rate)
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        return path

    def test_paused_seek_does_not_stop_a_fresh_inactive_device(self):
        class InitiallyStoppedDevice(SilentDevice):
            stopped = True

            def stop(self, ignore_errors=True):
                raise RuntimeError("cannot stop an inactive stream")

            def start(self):
                self.stopped = False

        stop = threading.Event()
        player = AudioPlayer(self.audio_file(), 0, 0, stop)
        player.seek_time = .5
        self.addCleanup(player.close)
        with patch("audio.sd.OutputStream", InitiallyStoppedDevice):
            player.prepare()
        player.pause()
        self.assertTrue(player.paused)
        self.assertEqual(player.time(), .5)
        self.assertTrue(player._stream.stopped)
        self.assertFalse(stop.is_set())
        player.resume()
        self.assertFalse(player.paused)
        self.assertFalse(player._stream.stopped)
        self.assertEqual(player.time(), .5)

    def decode_audio(self, path, target, rate=8000):
        stop = threading.Event()
        player = AudioPlayer(path, 0, 0, stop)
        player.seek_time = target
        self.addCleanup(player.close)
        with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 1, "default_samplerate": rate}):
            player.prepare()
        self.assertEqual(player._cursor, round(target * rate))
        self.assertAlmostEqual(player.time(), round(target * rate) / rate)
        player._producer.join(timeout=5)
        self.assertFalse(player._producer.is_alive(), "short audio did not reach EOF")
        self.assertIsNone(player.error)
        self.assertTrue(player.eof)
        count = round(player.end_time * rate) - player._cursor
        pcm = np.empty((count, 1), np.float32)
        player._callback(pcm, count, SimpleNamespace(outputBufferDacTime=100.0),
                         SimpleNamespace(output_underflow=False))
        self.assertIsNone(player.error)
        self.assertEqual(player._read, player._written)
        device = player._stream
        player.close()
        self.assertTrue(device.closed)
        self.assertFalse(stop.is_set(), "closing audio for a seek stopped the session")
        return pcm[:, 0], player.end_time

    def test_audio_forward_backward_seek_exact_pcm_and_media_cursor(self):
        path = self.audio_file()
        reference, end = self.decode_audio(path, 0)
        for target in (.673125, .125, .85, 0.0, .500125):
            with self.subTest(target=target):
                actual, actual_end = self.decode_audio(path, target)
                self.assertEqual(actual_end, end)
                np.testing.assert_array_equal(actual, reference[round(target * 8000):])

    def test_audio_seek_resampling_has_correct_length_and_filter_continuation(self):
        path = self.audio_file()
        reference, end = self.decode_audio(path, 0, rate=12000)
        for target in (.35, .125, .7, .500125):
            with self.subTest(target=target):
                actual, actual_end = self.decode_audio(path, target, rate=12000)
                expected = reference[round(target * 12000):]
                self.assertEqual(actual_end, end)
                self.assertEqual(len(actual), len(expected))
                # A newly started resampling filter may differ only at its head.
                np.testing.assert_allclose(actual[64:], expected[64:], atol=2e-6, rtol=0)

    def test_stereo_packed_and_planar_aac_seek_preserves_resampling_phase(self):
        for codec in ("pcm_s16le", "aac"):
            with self.subTest(codec=codec):
                path = self.audio_file(rate=44100, channels=2, codec=codec)
                reference, end = self.decode_audio(path, 0, rate=48000)
                for target in (.35, .500125, .7):
                    with self.subTest(target=target):
                        actual, actual_end = self.decode_audio(path, target, rate=48000)
                        expected = reference[round(target * 48000):]
                        self.assertEqual(actual_end, end)
                        self.assertEqual(len(actual), len(expected))
                        np.testing.assert_allclose(actual[64:], expected[64:],
                                                   atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()

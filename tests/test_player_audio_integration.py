"""Real decode/PCM/callback integration under slow video drawing, without sound.

Only the device query and OutputStream are replaced in the audio path. The
threaded sink supplies documented DAC timestamps and calls AudioPlayer's real
callback. This checks scheduling and PCM continuity, not physical A/V sync.
https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html
"""

from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audio
from cli import parser
from media import VideoDecoder
import player
from test_export import fixture


class SilentCallbackStream:
    """Simulated device clock and periodic callbacks; no PortAudio stream opens."""
    def __init__(self, **options):
        self.callback = options["callback"]
        self.owner = self.callback.__self__
        self.rate, self.channels = options["samplerate"], options["channels"]
        self.block_frames = self.rate // 50
        self.done = threading.Event()
        self.thread = None
        self.started = None
        self.closed = False
        self.error = None
        self.outputs = []
        self.callback_count = 0
        self.maximum_pcm_samples = 0
        self.prebuffer_samples = 0
        self.producer = None
        self.clock_at_close = None

    @property
    def time(self):
        return 100.0 + (time.perf_counter() - self.started if self.started is not None else 0)

    def start(self):
        # More than the runtime's 0.5s minimum makes the wall-clock test tolerant
        # of loaded CI hosts; the real producer fills the real two-second ring.
        deadline = time.perf_counter() + 2
        while self.owner._written - self.owner._read < self.rate * .75:
            if self.owner.error or time.perf_counter() >= deadline:
                raise RuntimeError("test audio producer did not prebuffer 0.75 seconds")
            self.done.wait(.005)
        self.prebuffer_samples = self.owner._written - self.owner._read
        self.producer = self.owner._producer
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name="silent-test-callback")
        self.thread.start()

    def _run(self):
        sent = 0
        try:
            while not self.done.is_set():
                deadline = self.started + sent / self.rate
                if self.done.wait(max(0, deadline - time.perf_counter())):
                    break
                output = np.empty((self.block_frames, self.channels), np.float32)
                self.callback(output, self.block_frames, SimpleNamespace(
                    currentTime=self.time,
                    outputBufferDacTime=100.0 + sent / self.rate + .04,
                ), SimpleNamespace(output_underflow=False))
                self.outputs.append(output)
                self.callback_count += 1
                self.maximum_pcm_samples = max(
                    self.maximum_pcm_samples, self.owner._written - self.owner._read)
                sent += self.block_frames
                if self.owner.error:
                    raise self.owner.error
                if sent > 10 * self.rate:
                    raise RuntimeError("test callback exceeded ten-second watchdog")
        except BaseException as exception:
            self.error = exception
            self.owner.stop_event.set()

    def close(self):
        self.clock_at_close = self.owner.time()
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.closed = True


def decoded_pcm(path):
    """Independent complete reference for this small fixed-duration fixture."""
    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="flt", layout="stereo", rate=48000)
        for frame in container.decode(container.streams.audio[0]):
            for block in resampler.resample(frame):
                chunks.append(block.to_ndarray().reshape(-1, 2))
        for block in resampler.resample(None):
            chunks.append(block.to_ndarray().reshape(-1, 2))
    return np.concatenate(chunks)


class PlayerAudioIntegrationTests(unittest.TestCase):
    def test_real_audio_callbacks_continue_during_slow_video_and_drain_at_eof(self):
        streams, videos, draws = [], [], []
        started = time.perf_counter()
        maximum_queue = 0

        def stream_factory(**options):
            stream = SilentCallbackStream(**options)
            streams.append(stream)
            return stream

        def decoder_factory(*args):
            video = VideoDecoder(*args)
            videos.append(video)
            return video

        class SlowTerminal:
            def __init__(self, color):
                self.exited = False

            def size(self):
                if time.perf_counter() - started > 10:
                    raise RuntimeError("test player exceeded ten-second watchdog")
                return 33, 17

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.exited = True

            def draw(self, symbols, colors, width, height):
                nonlocal maximum_queue
                before = streams[0].callback_count if streams else 0
                time.sleep(.1)
                after = streams[0].callback_count if streams else 0
                draws.append((before, after))
                maximum_queue = max(maximum_queue, videos[0].queue.qsize())
            def draw_timeline(self, *args):
                return None

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "slow drawing with real audio.mp4"
            fixture(path, video_pts=list(range(90)), audio_offset=0, audio_samples=144000)
            expected = decoded_pcm(path)
            options = parser().parse_args(["play", str(path), "--width", "32", "--height", "16"])
            self.assertFalse(options.no_audio)
            keyboard = MagicMock()
            keyboard.__enter__.return_value.space_pressed.return_value = False
            keyboard.__enter__.return_value.poll.return_value = []
            began = time.perf_counter()
            with patch("audio.sd.query_devices", return_value={
                "max_output_channels": 2, "default_samplerate": 48000,
            }) as query, patch("audio.sd.OutputStream", side_effect=stream_factory), \
                    patch("player.VideoDecoder", side_effect=decoder_factory), \
                    patch("player.Terminal", SlowTerminal), patch("player.Keyboard", return_value=keyboard), \
                    patch("player.sys.stdout.isatty", return_value=True), \
                    patch.dict("os.environ", TERM="xterm-256color"):
                player.play(options)
            elapsed = time.perf_counter() - began
        query.assert_called_once_with(kind="output")
        self.assertEqual(len(streams), 1)
        self.assertEqual(len(videos), 1)
        stream, video = streams[0], videos[0]
        audio_player = stream.owner
        duration = len(expected) / 48000
        self.assertGreaterEqual(elapsed, duration)
        self.assertLess(elapsed, duration + 3)
        self.assertTrue(audio_player.eof and video.eof)
        self.assertIsNone(audio_player.error)
        self.assertIsNone(video.error)
        self.assertIsNone(stream.error)
        self.assertAlmostEqual(audio_player.end_time, duration, places=6)
        self.assertAlmostEqual(video.end_time, 3.0, places=6)
        self.assertGreaterEqual(stream.clock_at_close, duration)
        self.assertLess(stream.clock_at_close, duration + 1)
        self.assertGreaterEqual(stream.prebuffer_samples, 36000)
        self.assertEqual(audio_player._written, len(expected))
        self.assertEqual(audio_player._read, len(expected))
        output = np.concatenate(stream.outputs)
        self.assertGreaterEqual(len(output), len(expected))
        np.testing.assert_array_equal(output[:len(expected)], expected)
        np.testing.assert_array_equal(output[len(expected):], 0)
        self.assertTrue(any(after - before >= 2 for before, after in draws[1:]))
        self.assertGreater(video.presented, 0)
        self.assertGreater(video.skipped + video.dropped, 0)
        self.assertLess(video.presented, 90)
        self.assertEqual(video.presented + video.skipped + video.dropped, 90)
        self.assertEqual(video.queue.maxsize, 3)
        self.assertLessEqual(maximum_queue, 3)
        self.assertEqual(audio_player._ring.shape, (96000, 2))
        self.assertLessEqual(stream.maximum_pcm_samples, audio_player._capacity)
        self.assertTrue(stream.closed)
        self.assertFalse(stream.thread.is_alive())
        self.assertFalse(stream.producer.is_alive())
        self.assertFalse(video.thread.is_alive())


if __name__ == "__main__":
    unittest.main()

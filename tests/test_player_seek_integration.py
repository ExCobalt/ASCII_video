"""Coordinated real video/audio seeks with a silent callback sink and fake screen."""

from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import parser
from media import VideoDecoder
import player
from test_export import fixture
from test_player_audio_integration import decoded_pcm
from audio import AudioPlayer


class SteppedSilentStream:
    """Call the real audio callback on a deterministic fake device clock."""
    def __init__(self, **options):
        self.callback = options["callback"]
        self.owner = self.callback.__self__
        self.rate, self.channels = options["samplerate"], options["channels"]
        self.time = 100.0
        self.active = self.closed = False
        self.outputs = []
        self.producer = None

    def start(self):
        self.active = True

    def stop(self, ignore_errors=False):
        self.active = False

    def step(self):
        if not self.active:
            return
        count = self.rate // 50
        output = np.empty((count, self.channels), np.float32)
        self.callback(output, count, SimpleNamespace(currentTime=self.time, outputBufferDacTime=self.time),
                      SimpleNamespace(output_underflow=False))
        self.outputs.append(output)
        self.time += count / self.rate

    def close(self):
        self.clock_at_close = self.owner.time()
        self.active = False
        self.closed = True
        self.producer = self.owner._producer


class PlayerSeekIntegrationTests(unittest.TestCase):
    def test_pending_seek_accepts_pause_replacement_and_interrupt(self):
        for held in ("audio", "video"):
            for interrupt in (False, True):
                with self.subTest(held=held, interrupt=interrupt):
                    self.run_pending_seek(held, interrupt)

    def test_pending_seek_worker_errors_restore_terminal_and_close_threads(self):
        for held in ("audio", "video"):
            with self.subTest(held=held):
                self.run_pending_seek(held, False, fail=True)

    def test_initial_buffering_accepts_pause_seek_interrupt_and_worker_errors(self):
        for held in ("audio", "video"):
            for interrupt, fail in ((False, False), (True, False), (False, True)):
                with self.subTest(held=held, interrupt=interrupt, fail=fail):
                    self.run_pending_seek(held, interrupt, fail=fail, initial=True)

    def test_initial_buffering_applies_pause_after_ready_and_resumes_complete_pcm(self):
        for held in ("audio", "video"):
            with self.subTest(held=held):
                self.run_pending_seek(held, False, initial=True, settle=True)

    def run_pending_seek(self, held, interrupt, *, fail=False, audio_type=AudioPlayer,
                         initial=False, settle=False):
        streams, videos = [], []
        blocked = threading.Event()
        release_error = threading.Event()
        release = threading.Event()
        screen = SimpleNamespace(exited=False, keyboard_restored=False, timelines=[], draws=0, nonblank=False)
        began = time.perf_counter()
        test = self

        class DelayedAudio(audio_type):
            def _produce(self):
                if held == "audio" and self.seek_time == (0 if initial else .4):
                    blocked.set()
                    while not self._stopped() and not release.is_set():
                        if time.perf_counter() - began > 5:
                            self._fail(RuntimeError("initial buffering keyboard watchdog"))
                            return
                        if release_error.is_set():
                            self._fail(RuntimeError("pending decode error"))
                            return
                        self.stop_event.wait(.001)
                    if self._stopped():
                        return
                super()._produce()

        class DelayedVideo(VideoDecoder):
            def __init__(self, *args):
                super().__init__(*args)
                videos.append(self)

            def _produce(self):
                if held == "video" and self.seek_time == (0 if initial else .4):
                    blocked.set()
                    while not self._cancel.is_set() and not self.stop_event.is_set() and not release.is_set():
                        if time.perf_counter() - began > 5:
                            self.error = RuntimeError("initial buffering keyboard watchdog")
                            self.stop_event.set()
                            self.ready.set()
                            return
                        if release_error.is_set():
                            self.error = RuntimeError("pending decode error")
                            self.stop_event.set()
                            self.ready.set()
                            return
                        self.stop_event.wait(.001)
                    if self._cancel.is_set() or self.stop_event.is_set():
                        return
                super()._produce()

        class Screen:
            def __init__(self, color):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                screen.exited = True

            def size(self):
                if time.perf_counter() - began > 5:
                    raise RuntimeError("pending seek watchdog")
                return 17, 9

            def draw(self, symbols, *args):
                screen.draws += 1
                screen.nonblank |= bool(np.any(symbols != " "))

            def draw_timeline(self, position, duration, paused):
                screen.timelines.append((position, duration, paused))
                return 2, 10, 9

        class Keyboard:
            stage = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                screen.keyboard_restored = True

            def poll(self):
                stream = streams[-1]
                if self.stage == 0:
                    if initial and not blocked.is_set():
                        return []
                    self.stage = 1
                    return [("pause", None)] if initial else [("pause", None), ("seek", .4)]
                if self.stage == 1:
                    if not blocked.is_set():
                        return []
                    test.assertFalse(stream.active)
                    test.assertEqual(len(streams), 1 if initial else 2)
                    if interrupt:
                        raise KeyboardInterrupt
                    if fail:
                        release_error.set()
                        return []
                    self.stage = 2
                    if settle:
                        release.set()
                        return []  # Keep pause intent through initial readiness.
                    return [("pause", None)]  # Resume intent while still preparing.
                if self.stage == 2:
                    test.assertFalse(stream.active)
                    if settle:
                        if not stream.owner.paused:
                            return []
                        test.assertTrue(videos[-1].ready.is_set() and stream.owner.ready)
                        test.assertGreater(screen.draws, 1)
                        test.assertTrue(screen.nonblank)
                        test.assertEqual(stream.owner.time(), 0)
                        test.assertTrue(any(p == 0 and paused for p, _, paused in screen.timelines))
                        self.stage, self.target = 3, 0
                        return [("pause", None)]
                    test.assertTrue(any(p == (0 if initial else .4) and not paused
                                        for p, _, paused in screen.timelines))
                    self.stage = 3
                    if held == "audio":
                        self.target = .2 if initial else .6
                        return [("seek", .2)]
                    self.target = screen.timelines[-1][1] * .75
                    return [("click", (7, 8))]
                if not stream.active:
                    return []
                test.assertEqual(len(streams), 1 if settle else 2 if initial else 3)
                test.assertAlmostEqual(stream.owner.seek_time, self.target)
                if not settle:
                    old = 0 if initial else 1
                    test.assertTrue(streams[old].closed)
                    test.assertFalse(streams[old].producer.is_alive())
                    test.assertFalse(videos[old].thread.is_alive())
                    test.assertEqual(streams[old].outputs, [])
                stream.step()
                return []

        def stream_factory(**options):
            stream = SteppedSilentStream(**options)
            streams.append(stream)
            return stream

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "pending.mp4"
            fixture(path, video_pts=list(range(48)), audio_offset=0, audio_samples=76800)
            expected = decoded_pcm(path)
            options = parser().parse_args(["play", str(path)])
            with patch("audio.AudioPlayer", DelayedAudio), \
                    patch("audio.sd.query_devices", return_value={"max_output_channels": 2, "default_samplerate": 48000}), \
                    patch("audio.sd.OutputStream", side_effect=stream_factory), \
                    patch("player.VideoDecoder", DelayedVideo), patch("player.Terminal", Screen), \
                    patch("player.Keyboard", Keyboard), patch("player.sys.stdout.isatty", return_value=True), \
                    patch.dict("os.environ", TERM="xterm-256color"):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "pending decode error"):
                        player.play(options)
                elif interrupt:
                    with self.assertRaises(KeyboardInterrupt):
                        player.play(options)
                else:
                    player.play(options)
        self.assertTrue(screen.exited and screen.keyboard_restored)
        self.assertTrue(all(s.closed and not s.producer.is_alive() for s in streams))
        self.assertTrue(all(not v.thread.is_alive() for v in videos))
        if not interrupt and not fail:
            actual = np.concatenate(streams[-1].outputs)
            start = round(streams[-1].owner.seek_time * 48000)
            self.assertGreaterEqual(len(actual), len(expected) - start)
            count = min(len(actual), len(expected) - start)
            np.testing.assert_array_equal(actual[:count], expected[start:start + count])
            np.testing.assert_array_equal(actual[count:], 0)

    def test_real_decode_audio_and_video_seek_together_preserving_pause_and_pcm(self):
        streams, videos, screen = [], [], SimpleNamespace(exited=False, colors=[], timelines=[])
        observations = []
        began = time.perf_counter()

        def stream_factory(**options):
            stream = SteppedSilentStream(**options)
            streams.append(stream)
            return stream

        class ObservedDecoder(VideoDecoder):
            def __init__(self, *args):
                super().__init__(*args)
                videos.append(self)
                self.first_prepared = None
                put = self.queue.put

                def observed_put(frame, *args, **kwargs):
                    if self.first_prepared is None:
                        self.first_prepared = frame
                    return put(frame, *args, **kwargs)

                self.queue.put = observed_put

        class Screen:
            def __init__(self, color):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                screen.exited = True

            def size(self):
                if time.perf_counter() - began > 5:
                    raise RuntimeError("seek integration exceeded watchdog")
                return 17, 9

            def draw(self, symbols, colors, width, height):
                if colors is not None:
                    screen.colors.append(colors.copy())

            def draw_timeline(self, position, duration, paused):
                screen.timelines.append((position, duration, paused))
                return 2, 10, 9

        test = self

        class Keyboard:
            stage = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                screen.keyboard_restored = True

            def inspect_seek(self):
                stream, video = streams[-1], videos[-1]
                test.assertEqual(len(streams), len(videos))
                test.assertTrue(stream.owner.paused)
                test.assertFalse(stream.active)
                test.assertAlmostEqual(stream.owner.time(), stream.owner.seek_time, delta=1 / 48000)
                test.assertEqual(stream.owner.seek_time, video.seek_time)
                test.assertIsNotNone(video.first_prepared)
                test.assertLessEqual(video.first_prepared.start, video.seek_time + 1e-8)
                test.assertGreater(video.first_prepared.end, video.seek_time)
                np.testing.assert_array_equal(screen.colors[-1],
                    np.clip(np.floor(video.first_prepared.grid + .5), 0, 255).astype(np.uint8))
                for old in streams[:-1]:
                    test.assertTrue(old.closed)
                    test.assertIsNotNone(old.producer)
                    test.assertFalse(old.producer.is_alive())
                for old in videos[:-1]:
                    test.assertFalse(old.thread.is_alive())
                observations.append((video.seek_time, stream.owner.time(), len(stream.outputs)))

            def poll(self):
                stream = streams[-1]
                if self.stage in (2, 3, 5) and not stream.owner.paused:
                    return []  # Seek preparation now remains interactive.
                if self.stage == 0:
                    stream.step()
                    if stream.owner.time() >= .08:
                        self.stage = 1
                        self.pause_position = stream.owner.time()
                        return [("pause", None)]
                elif self.stage == 1:
                    test.assertTrue(stream.owner.paused)
                    before = stream.owner.time()
                    stream.step()
                    test.assertEqual(stream.owner.time(), before)
                    self.forward_target = before + .5
                    self.stage = 2
                    return [("seek", .5)]
                elif self.stage == 2:
                    self.inspect_seek()
                    test.assertAlmostEqual(stream.owner.seek_time, self.forward_target, places=7)
                    self.backward_target = stream.owner.time() - .25
                    self.stage = 3
                    return [("seek", -.25)]
                elif self.stage == 3:
                    self.inspect_seek()
                    test.assertAlmostEqual(stream.owner.seek_time, self.backward_target, places=7)
                    self.stage = 4
                    self.before_outside = len(streams)
                    return [("click", (7, 0))]  # Video area: no seek.
                elif self.stage == 4:
                    test.assertEqual(len(streams), self.before_outside)
                    self.click_target = screen.timelines[-1][1] * .75
                    self.stage = 5
                    return [("click", (7, 8))]  # One-based x=8 in bar[2,10].
                elif self.stage == 5:
                    self.inspect_seek()
                    test.assertAlmostEqual(stream.owner.seek_time, self.click_target, places=7)
                    self.stage = 6
                    return [("pause", None)]
                else:
                    test.assertFalse(stream.owner.paused)
                    stream.step()
                return []

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "seek with real sound track.mp4"
            fixture(path, video_pts=list(range(48)), audio_offset=0, audio_samples=76800)
            expected = decoded_pcm(path)
            options = parser().parse_args(["play", str(path), "--color", "source"])
            with patch("audio.sd.query_devices", return_value={"max_output_channels": 2,
                                                                "default_samplerate": 48000}), \
                    patch("audio.sd.OutputStream", side_effect=stream_factory), \
                    patch("player.VideoDecoder", ObservedDecoder), patch("player.Terminal", Screen), \
                    patch("player.Keyboard", Keyboard), patch("player.sys.stdout.isatty", return_value=True), \
                    patch.dict("os.environ", TERM="xterm-256color"):
                player.play(options)

        self.assertEqual(len(streams), 4)
        self.assertEqual(len(observations), 3)
        self.assertTrue(screen.exited and screen.keyboard_restored)
        self.assertTrue(any(paused for _, _, paused in screen.timelines))
        cursor = streams[0].owner._audio_cursor
        self.assertTrue(cursor.closed)
        self.assertEqual(cursor.history_seconds, 8)
        for stream, video in zip(streams, videos):
            self.assertIs(stream.owner._audio_cursor, cursor)
            self.assertTrue(stream.closed)
            self.assertFalse(stream.producer.is_alive())
            self.assertFalse(video.thread.is_alive())
            self.assertIsNone(stream.owner.error)
            self.assertIsNone(video.error)
            sample = round(stream.owner.seek_time * stream.rate)
            if stream.outputs:
                actual = np.concatenate(stream.outputs)
                count = min(len(actual), len(expected) - sample)
                np.testing.assert_array_equal(actual[:count], expected[sample:sample + count])
                np.testing.assert_array_equal(actual[count:], 0)
            else:
                # Never-started paused generations still hold only their new PCM.
                start, count, physical = stream.owner._blocks[0]
                self.assertEqual(start, sample)
                np.testing.assert_array_equal(stream.owner._ring[physical:physical + count],
                                              expected[sample:sample + count])
        self.assertGreater(streams[-1].clock_at_close, observations[-1][0])
        self.assertTrue(videos[-1].eof and streams[-1].owner.eof)


if __name__ == "__main__":
    unittest.main()

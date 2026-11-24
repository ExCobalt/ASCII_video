"""Playback scheduling checks without a physical terminal or audio device."""

from fractions import Fraction
from concurrent.futures import Future
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import player
from cli import parser
from media import PreparedFrame


class PlaybackTests(unittest.TestCase):
    def test_resize_worker_is_bounded_nonblocking_and_discards_old_result(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def decode(info, start, cancel):
            calls.append(start)
            entered.set()
            if start == 0:
                if not release.wait(2):
                    raise AssertionError("resize worker was not released")
            return None if cancel.is_set() else np.full((4, 8, 3), 180, dtype=np.uint8)
        first = PreparedFrame(0, .1, None, (4, 2, 4, 2), False)
        second = PreparedFrame(.1, .2, None, (4, 2, 4, 2), False)
        resizer = player.FrameResizer(None, player.Renderer())
        try:
            with patch("player.frame_at", side_effect=decode):
                self.assertFalse(resizer.resize(first, (2, 1, 2, 1), None))
                self.assertTrue(entered.wait(1))
                first_job = resizer.future
                for _ in range(10):
                    self.assertFalse(resizer.resize(second, (3, 1, 3, 1), None))
                    self.assertIs(resizer.future, first_job)
                self.assertEqual(calls, [0])
                release.set()
                first_job.result(timeout=1)
                self.assertFalse(resizer.resize(second, (3, 1, 3, 1), None))
                resizer.future.result(timeout=1)
                self.assertTrue(resizer.resize(second, (3, 1, 3, 1), None))
                self.assertEqual(first.geometry, (4, 2, 4, 2))
                self.assertEqual(second.grid.shape, (1, 3, 3))
                self.assertEqual(calls, [0, .1])
        finally:
            release.set()
            resizer.close()
        self.assertIsNone(resizer.future)

    def run_player(self, intervals, *, resize=False, error=None, interrupt=False, fps=60,
                   pause_times=(), entry_size=None, seek_events=(), seek_delay=0, race_draw=None,
                   image_options=(), honor_sleep=False):
        options = parser().parse_args(["play", "fake.mp4", "--no-audio", "--width", "4", "--height", "2",
                                      "--cell-aspect", "1", "--fps", str(fps), "--palette", " .#@", *image_options])
        clock = [0.0]
        draws = []
        state = SimpleNamespace(exited=False, closed=False, timelines=[], color_draws=[],
                                sleeps=[], size_queries=0)
        keys = list(pause_times)
        seeks = list(seek_events)

        class ImmediateExecutor:
            def __init__(self, **kwargs):
                pass
            def submit(self, function, *args):
                result = Future()
                try:
                    result.set_result(function(*args))
                except BaseException as error:
                    result.set_exception(error)
                return result
            def shutdown(self, **kwargs):
                pass

        class Keyboard:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                state.keyboard_restored = True
            def space_pressed(self):
                if keys and clock[0] >= keys[0]:
                    keys.pop(0)
                    return True
                return False
            def poll(self):
                events = [("pause", None)] if self.space_pressed() else []
                if seeks and clock[0] >= seeks[0][0]:
                    _, action, value = seeks.pop(0)
                    events.append((action, value))
                return events

        class Stop:
            stopped = False
            def is_set(self):
                return self.stopped
            def set(self):
                self.stopped = True
            def wait(self, timeout):
                state.sleeps.append((clock[0], timeout))
                clock[0] += timeout if honor_sleep else 0.005
                if interrupt:
                    raise KeyboardInterrupt
                if clock[0] > 1:
                    raise AssertionError("playback did not terminate")

        class Screen:
            def __init__(self, color):
                self.entered = False
                self.race_size = None
                self.attempts = 0
            def size(self):
                state.size_queries += 1
                if self.race_size is not None:
                    return self.race_size
                if self.entered and entry_size is not None:
                    return entry_size
                return (4, 2) if resize and clock[0] >= 0.025 else (5, 3)
            def __enter__(self):
                self.entered = True
                return self
            def __exit__(self, *exc):
                state.exited = True
            def draw(self, symbols, colors, bw, bh):
                columns, rows = self.size()
                self.attempts += 1
                if self.attempts == race_draw:
                    self.race_size = columns - 1, rows
                    columns, rows = self.race_size
                if not (symbols.shape[1] <= bw < columns and symbols.shape[0] <= bh < rows):
                    raise ValueError("frame does not fit terminal drawing area")
                draws.append((clock[0], symbols.copy(), bw, bh))
                state.color_draws.append(None if colors is None else colors.copy())
            def draw_timeline(self, *args):
                state.timelines.append((clock[0], args, self.size()))
                return (2, 12, self.size()[1])

        class Decoder:
            def __init__(self, info, renderer, geometry, get_clock, stop):
                self.geometry = geometry
                self.queue = queue.Queue(maxsize=3)
                self.eof = True
                self.ready_at = 0
                self.ready = SimpleNamespace(is_set=lambda: clock[0] >= self.ready_at)
                self.error = error
                self.dropped = 0
                self.skipped = self.presented = 0
                state.decoder = self
                self.end_time = intervals[-1][1]
                self.last_rgb = np.full((4, 8, 3), intervals[-1][2], dtype=np.uint8)
                for index, (start, end, level) in enumerate(intervals):
                    self.queue.put(PreparedFrame(start, end, np.full((geometry[1], geometry[0], 3), level, dtype=float),
                                                 geometry, index == len(intervals) - 1))
            def start(self, *, wait=True):
                target = getattr(self, "seek_time", 0)
                if target > 0:
                    self.ready_at = clock[0] + seek_delay
                retained = []
                while not self.queue.empty():
                    frame = self.queue.get_nowait()
                    if frame.end > target or frame.last:
                        retained.append(frame)
                for frame in retained:
                    self.queue.put(frame)
            def close(self):
                state.closed = True

        info = SimpleNamespace(aspect=2, path=Path("fake.mp4"), origin=Fraction(0), audio_index=None,
                               duration=intervals[-1][1])
        def frame_at(info, timestamp, cancel=None):
            level = next(level for start, end, level in intervals if start <= timestamp < end)
            return np.full((4, 8, 3), level, dtype=np.uint8)

        with patch("player.Terminal", Screen), patch("player.VideoDecoder", Decoder), patch("player.probe", return_value=info), \
                patch("player.ThreadPoolExecutor", ImmediateExecutor), \
                patch("player.Keyboard", Keyboard), \
                patch("player.frame_at", side_effect=frame_at), \
                patch("player.threading.Event", Stop), patch("player.time.perf_counter", side_effect=lambda: clock[0]), \
                patch("player.time.sleep", side_effect=Stop().wait), \
                patch("player.sys.stdout.isatty", return_value=True), patch.dict("os.environ", TERM="xterm"):
            try:
                player.play(options)
            except BaseException as caught:
                state.error = caught
        return draws, state, clock[0]

    def test_player_uses_corrected_glyphs_and_foreground_from_raw_grid(self):
        level = (130, 70, 30)
        draws, state, finish = self.run_player([(0, .1, level)], image_options=(
            "--color", "source", "--color-saturation", "-1", "--glyph-correction", "1"))
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        renderer = player.Renderer(" .#@", color="source", color_saturation=-1, glyph_correction=1)
        self.assertTrue(draws)
        active = 0
        for draw, colors in zip(draws, state.color_draws):
            if np.all(draw[1] == " "):
                self.assertFalse(colors.any())
                continue
            active += 1
            grid = np.broadcast_to(level, (*draw[1].shape, 3))
            symbols, foreground = renderer.cells(grid)
            np.testing.assert_array_equal(draw[1], symbols)
            np.testing.assert_array_equal(colors, np.clip(np.floor(foreground + .5), 0, 255).astype(np.uint8))
            np.testing.assert_array_equal(colors[..., 0], colors[..., 1])
            np.testing.assert_array_equal(colors[..., 1], colors[..., 2])
        self.assertGreater(active, 0)

    def test_resize_race_during_startup_pause_and_seek_does_not_abort(self):
        scenarios = [dict(race_draw=1),
                     dict(resize=True, pause_times=(.02, .12), race_draw=3),
                     dict(pause_times=(.01, .12), seek_events=((.02, "seek", .1),), race_draw=3)]
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                draws, state, finish = self.run_player([(0, .15, 100), (.15, .2, 255)], **scenario)
                self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
                self.assertTrue(state.exited and state.closed)
                self.assertGreaterEqual(finish, .2)
                self.assertTrue(draws)

    def test_vfr_all_frames_and_last_held_until_end(self):
        draws, state, finish = self.run_player([(0, 0.04, 100), (0.04, 0.1, 180), (0.1, 0.2, 255)])
        self.assertFalse(hasattr(state, "error"))
        # First draw is the startup black screen, followed by each active VFR frame.
        self.assertEqual([str(symbols[0, 0]) for _, symbols, _, _ in draws], [" ", ".", "#", "@"])
        self.assertGreaterEqual(finish, 0.2)
        self.assertLess(finish, 0.21)
        self.assertTrue(state.closed and state.exited)

    def test_quiet_vfr_waits_for_known_pts_without_delaying_frames(self):
        intervals = [(0, .037, 100), (.037, .143, 180), (.143, .9, 255)]
        draws, state, finish = self.run_player(intervals, honor_sleep=True)
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        self.assertEqual([str(symbols[0, 0]) for _, symbols, _, _ in draws], [" ", ".", "#", "@"])
        for (at, *_), (start, _, _) in zip(draws[1:], intervals):
            self.assertGreaterEqual(at, start)
            self.assertLessEqual(at - start, .0021)
        self.assertAlmostEqual(finish, .9, delta=.0021)
        self.assertTrue(any(delay > .002 for _, delay in state.sleeps))
        self.assertTrue(all(.002 <= delay <= .01 for _, delay in state.sleeps))
        self.assertLess(len(state.sleeps), 110)

    def test_decode_readiness_keeps_short_poll_interval(self):
        draws, state, _ = self.run_player([(0, .5, 100), (.5, .9, 180)],
            seek_events=((.02, "seek", .1),), seek_delay=.03, honor_sleep=True)
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        pending_waits = [delay for at, delay in state.sleeps if .02 <= at < .05]
        self.assertTrue(pending_waits)
        self.assertTrue(all(delay == .002 for delay in pending_waits))
        self.assertTrue(any(.05 <= at < .0521 for at, *_ in draws))

    def test_pending_video_without_audio_resumes_clock_only_after_readiness(self):
        draws, state, finish = self.run_player([(0, .2, 100), (.2, .4, 180)],
            pause_times=(.02, .08), seek_events=((.025, "seek", .1),), seek_delay=.1)
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        pending = [(at, position, paused) for at, (position, _, paused), _ in state.timelines
                   if .025 <= at < .125]
        self.assertTrue(pending)
        for _, position, _ in pending:
            self.assertAlmostEqual(position, .12)
        self.assertTrue(any(not paused for _, _, paused in pending))
        self.assertTrue(any(at >= .125 for at, *_ in draws))
        self.assertGreaterEqual(finish, .405)
        self.assertLess(finish, .42)
        self.assertTrue(state.closed and state.exited and state.keyboard_restored)

    def test_space_pause_freezes_media_time_without_skipping_frames(self):
        draws, state, finish = self.run_player([
            (0, .04, 100), (.04, .1, 180), (.1, .2, 255)
        ], pause_times=(.02, .12))
        self.assertFalse(hasattr(state, "error"))
        self.assertEqual([str(symbols[0, 0]) for _, symbols, _, _ in draws], [" ", ".", "#", "@"])
        self.assertFalse(any(.02 <= timestamp < .12 for timestamp, *_ in draws))
        self.assertGreaterEqual(draws[2][0], .13)
        self.assertAlmostEqual(finish, .3, delta=.011)
        self.assertEqual(state.decoder.skipped, 0)
        self.assertTrue(state.keyboard_restored and state.closed and state.exited)

    def test_seek_while_paused_previews_new_frame_and_retains_pause(self):
        draws, state, finish = self.run_player([(0, .08, 100), (.08, .15, 180), (.15, .2, 255)],
            pause_times=(.01, .05), seek_events=((.02, "seek", .1),))
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        preview = [(t, str(s[0, 0])) for t, s, *_ in draws if .02 <= t < .05]
        self.assertEqual(preview, [(.02, "#")])
        self.assertLess(finish, .16)
        self.assertTrue(state.closed and state.exited)

    def test_resize_during_pause_redraws_frozen_frame_and_timeline(self):
        draws, state, finish = self.run_player([(0, .15, 100), (.15, .2, 255)],
            resize=True, pause_times=(.02, .12))
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        held = [(t, s) for t, s, *_ in draws if .025 <= t < .12]
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0][1].shape, (1, 2))
        self.assertEqual(str(held[0][1][0, 0]), ".")
        bars = [(t, args) for t, args, size in state.timelines if .025 <= t < .12]
        self.assertTrue(bars)
        self.assertTrue(all(abs(args[0] - .02) < .001 and args[2] for _, args in bars))
        self.assertAlmostEqual(finish, .3, delta=.011)

    def test_resize_and_seek_during_pause_keep_new_preview(self):
        draws, state, _ = self.run_player([(0, .08, 100), (.08, .15, 180), (.15, .2, 255)],
            resize=True, pause_times=(.01, .12), seek_events=((.04, "seek", .1),))
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        after_seek = [s for t, s, *_ in draws if .04 <= t < .12]
        self.assertEqual(len(after_seek), 1)
        self.assertEqual(after_seek[0].shape, (1, 2))
        self.assertEqual(str(after_seek[0][0, 0]), "#")

    def test_backward_and_mouse_seek_ignore_clicks_outside_bar(self):
        intervals = [(0, .08, 100), (.08, .15, 180), (.15, .2, 255)]
        for events, expected_finish in (
            (((.1, "seek", -.08),), .28),
            (((.02, "click", (6, 2)),), .12),
            (((.02, "click", (6, 0)),), .2),
        ):
            draws, state, finish = self.run_player(intervals, seek_events=events)
            self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
            self.assertAlmostEqual(finish, expected_finish, delta=.012)
            self.assertTrue(state.closed and state.exited)

    def test_rate_limit_counts_unshown_and_expired_frames_once(self):
        for middle_end in (.005, .009):
            with self.subTest(middle_end=middle_end):
                draws, state, _ = self.run_player([
                    (0, .004, 100), (.004, middle_end, 180), (middle_end, .2, 255)
                ], fps=10)
                self.assertFalse(hasattr(state, "error"))
                self.assertEqual(state.decoder.presented, 2)
                self.assertEqual(state.decoder.skipped, 1)
                self.assertEqual(state.decoder.dropped, 0)
                self.assertEqual(len(draws), 3)  # Initial black screen is not a video frame.
                self.assertGreaterEqual(draws[-1][0] - draws[1][0], .1)

    def test_delayed_video_keeps_black_screen_until_first_pts(self):
        draws, state, _ = self.run_player([(0.05, 0.12, 100), (0.12, 0.2, 255)])
        self.assertFalse(hasattr(state, "error"))
        self.assertEqual(str(draws[0][1][0, 0]), " ")
        self.assertGreaterEqual(draws[1][0], 0.05)
        self.assertEqual(str(draws[1][1][0, 0]), ".")

    def test_error_and_cancel_restore_terminal_and_close_decoder(self):
        for error, interrupt, expected in [(RuntimeError("decode failed"), False, RuntimeError), (None, True, KeyboardInterrupt)]:
            with self.subTest(interrupt=interrupt):
                _, state, _ = self.run_player([(0, 0.2, 100)], error=error, interrupt=interrupt)
                self.assertIsInstance(state.error, expected)
                self.assertTrue(state.exited and state.closed)

    def test_resize_redraws_active_vfr_frame_before_next_pts(self):
        draws, state, _ = self.run_player([(0, 0.15, 100), (0.15, 0.2, 255)], resize=True)
        self.assertFalse(hasattr(state, "error"))
        resized_active = [(timestamp, symbols) for timestamp, symbols, bw, bh in draws
                          if 0.025 <= timestamp < 0.15 and bw == 3 and bh == 1]
        self.assertTrue(resized_active, "active VFR frame must redraw after terminal resize")
        self.assertEqual(str(resized_active[0][1][0, 0]), ".")

    def test_font_setup_remeasures_geometry_before_initial_frame(self):
        draws, state, finish = self.run_player([(0, .1, 100), (.1, .2, 255)],
                                              entry_size=(4, 2))
        self.assertFalse(hasattr(state, "error"), getattr(state, "error", None))
        self.assertEqual(draws[0][1].shape, (1, 2))
        self.assertEqual(draws[0][2:], (3, 1))
        self.assertTrue(all(symbols.shape == (1, 2) for _, symbols, _, _ in draws))
        self.assertEqual([str(symbols[0, 0]) for _, symbols, _, _ in draws], [" ", ".", "@"])
        self.assertGreaterEqual(finish, .2)
        self.assertTrue(state.exited and state.closed)

    def test_real_short_mp4_without_audio(self):
        import av
        draws = []
        class Screen:
            def __init__(self, color):
                pass
            def size(self):
                return 9, 5
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                pass
            def draw(self, symbols, colors, bw, bh):
                draws.append(symbols.copy())
            def draw_timeline(self, *args):
                return None

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "short movie.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=20)
                stream.width, stream.height, stream.pix_fmt = 16, 8, "yuv420p"
                stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
                for index, level in enumerate((100, 180, 255)):
                    frame = av.VideoFrame.from_ndarray(np.full((8, 16, 3), level, dtype=np.uint8), format="rgb24")
                    frame.pts = index
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            options = parser().parse_args(["play", str(path), "--no-audio", "--width", "8", "--palette", " .#@"])
            started = time.monotonic()
            with patch("player.Terminal", Screen), patch("player.sys.stdout.isatty", return_value=True), patch.dict("os.environ", TERM="xterm"):
                player.play(options)
            self.assertLess(time.monotonic() - started, 2)
        self.assertEqual([str(symbols[0, 0]) for symbols in draws], [" ", ".", "#", "@"])


if __name__ == "__main__":
    unittest.main()

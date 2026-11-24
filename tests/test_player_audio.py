"""Audio/video coordination with fake audio, clock and terminal only."""

from fractions import Fraction
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import player
from cli import parser
from media import PreparedFrame


class AudioPlaybackTests(unittest.TestCase):
    def run_player(self, *, video_end=.2, audio_end=.4, video_start=0,
                   underflow_at=None, interrupt_at=None, space_times=()):
        options = parser().parse_args([
            "play", "fake.mp4", "--width", "4", "--height", "2", "--cell-aspect", "1",
            "--fps", "60", "--palette", " .#@",
        ])
        media_clock = [0.0]
        wall_clock = [0.0]
        draws = []
        calls = []
        state = SimpleNamespace(audio_closed=False, video_closed=False, exited=False,
                                audio_times=[], error=None, paused=False,
                                pause_events=[], draw_wall_times=[], keyboard_exited=False,
                                cursor_closed=False)
        cursor = SimpleNamespace(close=lambda: setattr(state, 'cursor_closed', True))

        def advance(seconds):
            wall_clock[0] += seconds
            if not state.paused:
                media_clock[0] += seconds
            if interrupt_at is not None and wall_clock[0] >= interrupt_at:
                raise KeyboardInterrupt
            if wall_clock[0] > 1:
                raise AssertionError("player did not terminate using the fake audio clock")

        class Keyboard:
            def __init__(self):
                self.pending = list(space_times)

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                state.keyboard_exited = True

            def space_pressed(self):
                if self.pending and wall_clock[0] >= self.pending[0]:
                    self.pending.pop(0)
                    return True
                return False
            def poll(self):
                return [("pause", None)] if self.space_pressed() else []

        class Stop:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, seconds):
                advance(seconds)

        class Screen:
            def __init__(self, color):
                pass

            def size(self):
                return 5, 3

            def __enter__(self):
                calls.append("screen_enter")
                return self

            def __exit__(self, *exception):
                state.exited = True
                state.exit_time = media_clock[0]
                state.wall_finish = wall_clock[0]

            def draw(self, symbols, colors, bw, bh):
                draws.append((media_clock[0], str(symbols[0, 0])))
                state.draw_wall_times.append(wall_clock[0])
                calls.append("draw")
            def draw_timeline(self, *args):
                return None

        class Decoder:
            def __init__(self, info, renderer, geometry, get_clock, stop):
                self.geometry = geometry
                self.queue = queue.Queue(maxsize=3)
                self.eof = True
                self.error = None
                self.ready = SimpleNamespace(is_set=lambda: True)
                self.end_time = video_end
                self.dropped = 0
                self.skipped = self.presented = 0
                self.last_rgb = np.full((4, 8, 3), 255, dtype=np.uint8)
                midpoint = (video_start + video_end) / 2
                for start, end, level, last in [
                    (video_start, midpoint, 100, False), (midpoint, video_end, 255, True),
                ]:
                    self.queue.put(PreparedFrame(
                        start, end, np.full((geometry[1], geometry[0], 3), level, dtype=float),
                        geometry, last,
                    ))

            def start(self, *, wait=True):
                calls.append("video_start")

            def close(self):
                state.video_closed = True

        class FakeAudio:
            def __init__(self, path, stream_index, origin, stop, *, cursor=None):
                state.audio_arguments = (path, stream_index, origin)
                state.audio_cursor = cursor
                self.stop = stop
                self.ready = True

            def prepare(self, *, wait=True):
                calls.append("audio_prepare")

            def start(self):
                calls.append("audio_start")

            def time(self):
                state.audio_times.append(media_clock[0])
                return media_clock[0]

            def pause(self):
                state.paused = True
                calls.append("audio_pause")
                state.pause_events.append(("pause", wall_clock[0], media_clock[0]))

            def resume(self):
                state.paused = False
                calls.append("audio_resume")
                state.pause_events.append(("resume", wall_clock[0], media_clock[0]))

            @property
            def error(self):
                if underflow_at is not None and media_clock[0] >= underflow_at:
                    self.stop.set()
                    return RuntimeError("Audio device output underflow")
                return None

            @property
            def end_time(self):
                # EOF can be decoded ahead of playback, but is unknown at startup.
                return audio_end if media_clock[0] >= .01 else None

            def close(self):
                state.audio_closed = True
                state.audio_close_time = media_clock[0]
                state.audio_closed_while_paused = state.paused

        info = SimpleNamespace(aspect=2, path=Path("fake.mp4"), audio_index=2,
                               origin=Fraction(5, 2))
        with patch("player.Terminal", Screen), patch("player.VideoDecoder", Decoder), \
                patch("player.Keyboard", Keyboard), \
                patch("player.probe", return_value=info), \
                patch("player.threading.Event", Stop), \
                patch("player.time.sleep", side_effect=advance), \
                patch("player.time.monotonic", return_value=12345), \
                patch("player.time.perf_counter", side_effect=lambda: 1000 + wall_clock[0]), \
                patch("player.sys.stdout.isatty", return_value=True), \
                patch.dict("os.environ", TERM="xterm"), \
                patch("audio_cursor.create_audio_cursor", return_value=cursor) as make_cursor, \
                patch.dict(sys.modules, {"audio": SimpleNamespace(AudioPlayer=FakeAudio)}):
            try:
                player.play(options)
            except BaseException as error:
                state.error = error
        make_cursor.assert_called_once_with("fake.mp4", 2, history_seconds=8)
        self.assertIs(state.audio_cursor, cursor)
        self.assertTrue(state.cursor_closed)
        return draws, calls, state, media_clock[0]

    def test_video_ends_first_and_last_frame_held_to_audio_end(self):
        draws, calls, state, finish = self.run_player(video_end=.2, audio_end=.4)
        self.assertIsNone(state.error)
        self.assertEqual([symbol for _, symbol in draws], [" ", ".", "@"])
        self.assertGreaterEqual(finish, .4)
        self.assertLess(finish, .405)
        self.assertEqual(state.audio_arguments, ("fake.mp4", 2, Fraction(5, 2)))
        self.assertLess(calls.index("audio_prepare"), calls.index("screen_enter"))
        self.assertLess(calls.index("draw"), calls.index("audio_start"))
        self.assertTrue(state.audio_closed and state.video_closed and state.exited)

    def test_audio_ends_first_and_its_clock_continues_to_video_end(self):
        draws, _, state, finish = self.run_player(video_end=.3, audio_end=.1)
        self.assertIsNone(state.error)
        self.assertGreaterEqual(finish, .3)
        self.assertLess(finish, .305)
        self.assertTrue(any(timestamp >= .3 for timestamp in state.audio_times))
        self.assertEqual(draws[-1][1], "@")
        self.assertTrue(state.audio_closed and state.video_closed and state.exited)

    def test_delayed_video_uses_black_screen_before_audio_clock_reaches_pts(self):
        draws, _, state, _ = self.run_player(video_start=.05, video_end=.2, audio_end=.25)
        self.assertIsNone(state.error)
        self.assertEqual(draws[0], (0, " "))
        self.assertGreaterEqual(draws[1][0], .05)
        self.assertEqual(draws[1][1], ".")

    def test_underflow_and_ctrl_c_close_audio_video_and_restore_screen(self):
        for arguments, exception in [
            ({"underflow_at": .08}, RuntimeError),
            ({"interrupt_at": .08}, KeyboardInterrupt),
        ]:
            with self.subTest(arguments=arguments):
                _, _, state, finish = self.run_player(**arguments)
                self.assertIsInstance(state.error, exception)
                if exception is RuntimeError:
                    self.assertIn("underflow", str(state.error))
                self.assertLess(finish, .1)
                self.assertTrue(state.audio_closed and state.video_closed and state.exited)

    def test_space_pause_freezes_audio_clock_and_resume_preserves_frames_and_eof(self):
        draws, calls, state, finish = self.run_player(space_times=(.04, .14))
        self.assertIsNone(state.error)
        self.assertEqual([symbol for _, symbol in draws], [" ", ".", "@"])
        self.assertEqual(calls.count("audio_pause"), 1)
        self.assertEqual(calls.count("audio_resume"), 1)
        paused, resumed = state.pause_events
        self.assertEqual(paused[2], resumed[2])
        # Key arrivals need not coincide with a poll (including float clock
        # rounding at .14); each must be handled within the 10 ms interval.
        for event, requested in zip((paused, resumed), (.04, .14)):
            self.assertGreaterEqual(event[1], requested)
            self.assertLessEqual(event[1] - requested, .010001)
        pause_duration = resumed[1] - paused[1]
        self.assertFalse(any(paused[1] <= wall < resumed[1] for wall in state.draw_wall_times))
        self.assertGreaterEqual(finish, .4)
        self.assertLess(finish, .405)
        self.assertAlmostEqual(state.wall_finish - finish, pause_duration)
        self.assertTrue(state.audio_closed and state.video_closed and state.exited)
        self.assertTrue(state.keyboard_exited)

    def test_ctrl_c_during_pause_closes_both_streams_and_keyboard(self):
        _, _, state, _ = self.run_player(space_times=(.04, .14), interrupt_at=.08)
        self.assertIsInstance(state.error, KeyboardInterrupt)
        self.assertTrue(state.audio_closed_while_paused)
        self.assertTrue(state.audio_closed and state.video_closed and state.exited)
        self.assertTrue(state.keyboard_exited)


if __name__ == "__main__":
    unittest.main()

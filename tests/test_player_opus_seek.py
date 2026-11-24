"""Real Opus session cursor across paused seeks and cancellation; no device I/O."""
from fractions import Fraction
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

import audio_cursor
from cli import parser
from media import normalize_audio_timing, probe
from opus_cursor import OpusAudioCursor, codec_for
import player
from test_player_seek_integration import SteppedSilentStream


def fixture(path):
    with av.open(str(path), 'w', format='matroska') as output:
        video = output.add_stream('libx264', rate=30)
        video.width, video.height, video.pix_fmt = 64, 32, 'yuv420p'
        video.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
        audio = output.add_stream('libopus', rate=48000)
        audio.layout = 'stereo'
        for index in range(60):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), 160, np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 30)
            output.mux(video.encode(frame))
        output.mux(video.encode(None))
        t = np.arange(96000) / 48000
        values = np.column_stack((.2 * np.sin(2 * np.pi * 233 * t),
                                  .15 * np.sin(2 * np.pi * 397 * t))).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format='flt', layout='stereo')
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def reference(path):
    chunks, end = [], None
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        codec = codec_for(stream)
        for packet in container.demux(stream):
            for frame in codec.decode(packet):
                end, _ = normalize_audio_timing(frame, end)
                chunks.append(frame.to_ndarray().reshape(-1, 2).copy())
    return np.concatenate(chunks)


class PlayerOpusSeekTests(unittest.TestCase):
    def test_shared_cursor_preserves_pcm_through_paused_backward_seek_and_interrupt(self):
        streams, cursors = [], []
        state = SimpleNamespace(screen_closed=False, keyboard_closed=False)
        began = time.perf_counter()
        test = self

        class Screen:
            def __init__(self, color):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                state.screen_closed = True

            def size(self):
                if time.perf_counter() - began > 5:
                    raise RuntimeError('Opus player seek watchdog')
                return 17, 9

            def draw(self, *args):
                pass

            def draw_timeline(self, *args):
                return 2, 10, 9

        class Keyboard:
            stage = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                state.keyboard_closed = True

            def poll(self):
                stream = streams[-1]
                if self.stage == 0 and stream.active:
                    stream.step()
                    self.stage = 1
                    return [('pause', None)]
                if self.stage == 1 and stream.owner.paused:
                    self.stage = 2
                    return [('seek', .8)]
                if self.stage == 2 and len(streams) == 2 and stream.owner.paused:
                    self.forward = stream.owner.seek_time
                    self.stage = 3
                    return [('seek', -.5)]
                if self.stage == 3 and len(streams) == 3 and stream.owner.paused:
                    test.assertAlmostEqual(stream.owner.seek_time, self.forward - .5)
                    test.assertFalse(stream.active)
                    self.stage = 4
                    return [('pause', None)]
                if self.stage == 4 and stream.active:
                    stream.step()
                    self.stage = 5
                    return []
                if self.stage == 5:
                    raise KeyboardInterrupt
                return []

        def stream_factory(**options):
            stream = SteppedSilentStream(**options)
            streams.append(stream)
            return stream

        real_factory = audio_cursor.create_audio_cursor

        def cursor_factory(*args, **kwargs):
            cursor = real_factory(*args, **kwargs)
            cursors.append(cursor)
            return cursor

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / 'build') as directory:
            path = Path(directory) / 'seek.mkv'
            fixture(path)
            self.assertEqual(probe(path).origin, 0)
            expected = reference(path)
            options = parser().parse_args(['play', str(path)])
            with patch('audio_cursor.create_audio_cursor', side_effect=cursor_factory), \
                    patch('audio.sd.query_devices', return_value={
                        'max_output_channels': 2, 'default_samplerate': 48000}), \
                    patch('audio.sd.OutputStream', side_effect=stream_factory), \
                    patch('player.Terminal', Screen), patch('player.Keyboard', Keyboard), \
                    patch('player.sys.stdout.isatty', return_value=True), \
                    patch.dict('os.environ', TERM='xterm-256color'):
                with self.assertRaises(KeyboardInterrupt):
                    player.play(options)
        self.assertEqual(len(cursors), 1)
        cursor = cursors[0]
        self.assertIsInstance(cursor, OpusAudioCursor)
        self.assertGreater(cursor.restores, 0)
        self.assertTrue(cursor.closed)
        self.assertIsNone(cursor._container)
        self.assertIsNone(cursor._decoder)
        self.assertFalse(cursor.points)
        self.assertTrue(state.screen_closed and state.keyboard_closed)
        self.assertEqual(len(streams), 3)
        for stream in streams:
            self.assertIs(stream.owner._audio_cursor, cursor)
            self.assertTrue(stream.closed)
            self.assertFalse(stream.producer.is_alive())
            self.assertIsNone(stream.owner.error)
            start = round(stream.owner.seek_time * 48000)
            if stream.outputs:
                actual = np.concatenate(stream.outputs)
                np.testing.assert_array_equal(actual, expected[start:start + len(actual)])
            else:
                block_start, count, physical = stream.owner._blocks[0]
                self.assertEqual(block_start, start)
                np.testing.assert_array_equal(stream.owner._ring[physical:physical + count],
                                              expected[start:start + count])


if __name__ == '__main__':
    unittest.main()

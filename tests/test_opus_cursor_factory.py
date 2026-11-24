"""Production decoder selection and lifetime checks without an audio device."""
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import av
import numpy as np

from audio_cursor import SequentialAudioCursor, create_audio_cursor
from media import normalize_audio_timing
from opus_cursor import Decoder, OpusAudioCursor, codec_for, library


ROOT = Path(__file__).resolve().parents[1]


def encode(path, codec, layout, family=None):
    channels = len(av.AudioLayout(layout).channels)
    t = np.arange(96000) / 48000
    values = np.column_stack([
        .1 * np.sin(2 * np.pi * (137 + 83 * channel) * t)
        for channel in range(channels)]).astype(np.float32)
    options = {} if family is None else {'mapping_family': str(family)}
    with av.open(str(path), 'w') as output:
        stream = output.add_stream(codec, rate=48000, options=options)
        stream.layout = layout
        frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format='flt', layout=layout)
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        output.mux(stream.encode(frame))
        output.mux(stream.encode(None))


def signature(frame):
    return (frame.pts, frame.samples, frame.layout.name,
            hashlib.sha256(frame.to_ndarray()).hexdigest())


def reference(path, *, explicit_opus=False):
    result, end = [], None
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if explicit_opus:
            codec = codec_for(stream)
            frames = (frame for packet in container.demux(stream) for frame in codec.decode(packet))
        else:
            frames = container.decode(stream)
        for frame in frames:
            end, _ = normalize_audio_timing(frame, end)
            result.append(signature(frame))
    return result


class OpusCursorFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.mono = Path(cls.directory.name) / 'mono.webm'
        cls.surround = Path(cls.directory.name) / 'surround.webm'
        cls.aac = Path(cls.directory.name) / 'audio.m4a'
        encode(cls.mono, 'libopus', 'mono', 0)
        encode(cls.surround, 'libopus', '5.1', 1)
        encode(cls.aac, 'aac', 'stereo')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def assert_native_fallback(self, path, history_seconds=8):
        expected = reference(path)
        self.assertTrue(expected)
        cursor = create_audio_cursor(str(path), 0, history_seconds=history_seconds)
        self.addCleanup(cursor.close)
        self.assertIsInstance(cursor, SequentialAudioCursor)
        self.assertEqual(cursor.history_seconds, history_seconds)
        target = Fraction(expected[0][0], 48000)
        for repeat in range(2):
            self.assertEqual([signature(frame) for frame in cursor.frames(target)], expected)

    def test_supported_mapping_keeps_one_codec_across_backward_requests(self):
        for path in (self.mono, self.surround):
            with self.subTest(layout=path.stem):
                expected = reference(path, explicit_opus=True)
                cursor = create_audio_cursor(str(path), 0)
                try:
                    self.assertIsInstance(cursor, OpusAudioCursor)
                    for repeat in range(2):
                        self.assertEqual([signature(frame) for frame in cursor.frames(cursor.origin)], expected)
                    self.assertGreater(cursor.restores, 0)
                finally:
                    cursor.close()

    def test_macos_uses_libopus_from_pyav_wheel(self):
        package = Path(self.directory.name) / 'av'
        dylib = package / '.dylibs' / 'libopus.0.dylib'
        dylib.parent.mkdir(parents=True)
        dylib.touch()
        with patch('opus_cursor.sys.platform', 'darwin'), patch.object(av, '__file__', str(package / '__init__.py')), \
                patch('opus_cursor.ct.CDLL') as load:
            lib, path = library()
        load.assert_called_once_with(str(dylib))
        self.assertEqual(path, dylib)
        self.assertIs(lib, load.return_value)

    def test_aac_uses_native_decoder_and_requested_history_without_loading_opus(self):
        with patch('opus_cursor.library', side_effect=AssertionError('Unexpected libopus load')):
            self.assert_native_fallback(self.aac, history_seconds=3)

    def test_missing_library_falls_back_before_any_pcm(self):
        with patch('opus_cursor.library', side_effect=OSError('Library unavailable')) as load:
            self.assert_native_fallback(self.mono)
        load.assert_called_once_with()

    def test_untested_runtime_uses_native_decoder_without_loading_opus(self):
        with patch.dict(av.library_versions, {'libavcodec': (62, 0, 0)}), patch(
                'opus_cursor.library', side_effect=AssertionError('Unexpected libopus load')):
            self.assert_native_fallback(self.mono)

    def test_unsupported_mapping_uses_native_decoder_without_loading_opus(self):
        path = Path(self.directory.name) / 'family255.mkv'
        with av.open(str(self.mono)) as original, av.open(str(path), 'w') as output:
            stream = output.add_stream_from_template(original.streams.audio[0])
            head = bytearray(stream.codec_context.extradata)
            head[18] = 255
            stream.codec_context.extradata = bytes(head) + bytes((1, 0, 0))
            for packet in original.demux(original.streams.audio[0]):
                if packet.dts is not None:
                    packet.stream = stream
                    output.mux(packet)
        with av.open(str(path)) as container:
            self.assertEqual(container.streams.audio[0].codec_context.extradata[18], 255)
        with patch('opus_cursor.library', side_effect=AssertionError('Unexpected libopus load')):
            self.assert_native_fallback(path)

    def test_malformed_header_is_an_error_not_a_backend_change(self):
        head = bytearray(b'OpusHead' + bytes((1, 1)) + bytes(8) + bytes((0,)))
        family0_three_channels = head.copy()
        family0_three_channels[9] = 3
        truncated_family1 = head.copy()
        truncated_family1[18] = 1
        for invalid in (b'', b'NotOpus!' + bytes(11), bytes(family0_three_channels), bytes(truncated_family1)):
            with self.subTest(header=invalid):
                stream = SimpleNamespace(index=0, type='audio',
                    codec_context=SimpleNamespace(name='opus', extradata=invalid))
                container = MagicMock()
                container.__enter__.return_value.streams = [stream]
                with patch('opus_cursor.open_media', return_value=container), patch('opus_cursor.library') as load:
                    with self.assertRaises(RuntimeError):
                        create_audio_cursor('malformed.webm', 0)
                load.assert_not_called()

    def test_snapshot_size_checked_and_closed_decoder_cannot_access_native_memory(self):
        lib, _ = library()
        decoder = Decoder(lib, 1)
        state = decoder.snapshot()
        try:
            for invalid in (state[:-1], state + b'\0', bytearray(state)):
                with self.subTest(size=len(invalid), type=type(invalid).__name__):
                    with self.assertRaises(ValueError):
                        decoder.restore(invalid)
                    self.assertEqual(decoder.snapshot(), state)
        finally:
            decoder.__exit__()
        decoder.__exit__()
        self.assertIsNone(decoder.state)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            decoder.decode(b'', 960, lost=True)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            decoder.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            decoder.restore(state)


if __name__ == '__main__':
    unittest.main()

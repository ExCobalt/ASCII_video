"""Malformed native/container Opus data must never expose an unfilled PCM buffer."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, library
from opus_cursor import Decoder
from test_opus_multistream import encode_source


class InvalidPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.sources = []
        for layout in ('mono', 'stereo', '5.1'):
            path = Path(cls.directory.name) / f'{layout}.mkv'
            encode_source(path, layout)
            cls.sources.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_negative_duration_and_native_decode_failure(self):
        for channels, mapping in ((1, None), (2, None), (6, (4, 2, (0, 4, 1, 2, 3, 5)))):
            with self.subTest(channels=channels):
                with Decoder(self.lib, channels, multistream=mapping) as decoder:
                    samples = self.lib.opus_packet_get_nb_samples(b'\xff', 1, 48000)
                    self.assertEqual(samples, -4)
                    with self.assertRaisesRegex(ValueError, 'sample count'):
                        decoder.decode(b'\xff', samples)
                    samples = self.lib.opus_packet_get_nb_samples(b'\xfe', 1, 48000)
                    self.assertEqual(samples, 1920)
                    with self.assertRaisesRegex(RuntimeError, 'decode Opus packet'):
                        decoder.decode(b'\xfe', samples)

    def test_real_malformed_packets_fail_and_release_cursor(self):
        for source in self.sources:
            for raw in (b'\xff', b'\xfe'):
                path = source.with_stem(source.stem + '-' + raw.hex())
                with av.open(str(source)) as original, av.open(str(path), 'w') as output:
                    stream = output.add_stream_from_template(original.streams.audio[0])
                    for index, packet in enumerate(original.demux(original.streams.audio[0])):
                        if packet.dts is None:
                            continue
                        if index == 20:
                            damaged = av.Packet(raw)
                            damaged.pts, damaged.dts = packet.pts, packet.dts
                            damaged.duration, damaged.time_base = packet.duration, packet.time_base
                            packet = damaged
                        packet.stream = stream
                        output.mux(packet)
                with self.subTest(path=path.name):
                    # The default FFmpeg parser removes these framing errors
                    # before decoding. Disable it to exercise the real native
                    # boundary, rather than relying on assert or fabricated PCM.
                    original_open = av.open
                    options = {'fflags': '+noparse+nofillin'}
                    with original_open(str(path), options=options) as container:
                        codec = codec_for(container.streams.audio[0])
                        with self.assertRaises(av.error.InvalidDataError):
                            for packet in container.demux(container.streams.audio[0]):
                                codec.decode(packet)
                    with patch('opus_cursor.av.open',
                               side_effect=lambda *args, **kwargs: original_open(
                                   *args, **kwargs, options=options)):
                        cursor = LazyCursor(path, self.lib)
                        try:
                            for repeat in range(2):
                                with self.assertRaisesRegex((ValueError, RuntimeError), 'Opus'):
                                    list(cursor.frames(cursor.origin))
                                self.assertIsNone(cursor._container)
                                self.assertIsNone(cursor._decoder)
                                self.assertTrue(cursor.request_closed)
                        finally:
                            cursor.close()

    def test_errors_survive_python_optimization(self):
        code = """
import sys
sys.path.insert(0, 'src')
from opus_cursor import Decoder, library
lib, _ = library()
for raw, error_type in ((b'\\xff', ValueError), (b'\\xfe', RuntimeError)):
    with Decoder(lib, 1) as decoder:
        count = lib.opus_packet_get_nb_samples(raw, len(raw), 48000)
        try:
            decoder.decode(raw, count)
        except error_type:
            print(error_type.__name__)
        else:
            raise RuntimeError('Malformed packet exposed PCM under -O')
"""
        result = subprocess.run([sys.executable, '-O', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['ValueError', 'RuntimeError'])


if __name__ == '__main__':
    unittest.main()

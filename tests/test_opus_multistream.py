"""Silent family-1 multichannel PCM, stereo downmix and checkpoint checks."""
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from opus_cursor import Decoder
from test_opus_lazy_preskip import fixture
from media import normalize_audio_timing


def encode_source(path, layout, seconds=2):
    channels = len(av.AudioLayout(layout).channels)
    t = np.arange(seconds * 48000) / 48000
    values = np.column_stack([.1 * np.sin(2 * np.pi * (137 + 83 * i) * t)
                              for i in range(channels)]).astype(np.float32)
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libopus', rate=48000,
            options={'mapping_family': '1', 'frame_duration': '20'})
        stream.layout = layout
        frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format='flt', layout=layout)
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


class MultistreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.sources = []
        for layout in ('mono', 'stereo', '3.0', 'quad', '5.0', '5.1', '6.1', '7.1'):
            path = Path(cls.directory.name) / f'{layout}.webm'
            encode_source(path, layout)
            cls.sources.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def compare_frames(self, path):
        expected = []
        with av.open(str(path)) as container:
            codec = codec_for(container.streams.audio[0])
            end = None
            for packet in container.demux(container.streams.audio[0]):
                for frame in codec.decode(packet):
                    end, _ = normalize_audio_timing(frame, end)
                    expected.append((frame.pts, frame.samples, frame.layout.name,
                        hashlib.sha256(frame.to_ndarray()).hexdigest()))
        self.assertTrue(expected)
        cursor = LazyCursor(path, self.lib)
        try:
            self.assertIsNotNone(cursor.multistream)
            # Force several compactions in a short input, including 7.1 states.
            cursor._interval = 480
            for repeat in range(2):
                actual = [(frame.pts, frame.samples, frame.layout.name,
                    hashlib.sha256(frame.to_ndarray()).hexdigest())
                    for frame in cursor.frames(cursor.origin)]
                self.assertEqual(actual, expected)
            self.assertGreater(cursor.compactions, 0)
            self.assertGreater(cursor.restores, 0)
            self.assertLessEqual(cursor.max_snapshot_bytes, 2 * 1024 * 1024)
            self.assertLessEqual(len(cursor.points), 64)
        finally:
            cursor.close()
        self.assertIsNone(cursor._decoder)
        self.assertFalse(cursor.points)

    def test_speaker_order_and_bounded_snapshots(self):
        for source in self.sources:
            for container_format in ('ogg', 'webm'):
                with self.subTest(source=source.name, format=container_format):
                    path = Path(self.directory.name) / 'mapped.opus'
                    fixture(source, path, 3840, container_format, gain=1536)
                    self.compare_frames(path)

    def test_stereo_downmix_and_repeated_seeks(self):
        for source in self.sources:
            for container_format in ('ogg', 'webm'):
                path = Path(self.directory.name) / 'player.opus'
                fixture(source, path, 3840, container_format, gain=-1536)
                for rate in (44100, 48000):
                    with self.subTest(source=source.name, format=container_format, rate=rate):
                        cursor = LazyCursor(path, self.lib)
                        try:
                            reference = bytearray()
                            baseline = drain(cursor, 0, rate,
                                lambda block: reference.extend(block.tobytes()), indexed=False)
                            for target in (0, .731, .113, .731, baseline['end_time'] - .101, 0):
                                actual = hashlib.sha256()
                                result = drain(cursor, target, rate,
                                    lambda block: actual.update(block.tobytes()))
                                expected = memoryview(reference)[round(target * rate) * 8:]
                                self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                                self.assertEqual(result['samples'], len(expected) // 8)
                                self.assertEqual(result['end_time'], baseline['end_time'])
                        finally:
                            cursor.close()

    def test_duplicate_and_silent_channel_mapping(self):
        for source in (self.sources[2], self.sources[-1]):
            with av.open(str(source)) as container:
                head = container.streams.audio[0].codec_context.extradata
            mapping = bytearray(head[21:21 + head[9]])
            for silent in (False, True):
                with self.subTest(source=source.name, silent=silent):
                    mapping[-1] = 255 if silent else mapping[0]
                    path = Path(self.directory.name) / 'custom.webm'
                    fixture(source, path, 312, 'webm', mapping=mapping)
                    self.compare_frames(path)

    def test_multistream_snapshot_with_packet_loss(self):
        for source in self.sources:
            cursor = LazyCursor(source, self.lib)
            try:
                with av.open(str(source)) as container:
                    packets = container.demux(container.streams.audio[0])
                    payloads = [bytes(next(packets)) for _ in range(3)]
                samples = self.lib.opus_packet_get_nb_samples(payloads[0], len(payloads[0]), 48000)
                sequence = ((payloads[1], False), (b'', True), (b'', True), (payloads[2], False))
                with Decoder(self.lib, cursor.channels, gain=1536, multistream=cursor.multistream) as decoder:
                    decoder.decode(payloads[0], samples)
                    snapshot = decoder.snapshot()
                    expected = [hashlib.sha256(decoder.decode(raw, samples, lost)).digest()
                                for raw, lost in sequence]
                with Decoder(self.lib, cursor.channels, multistream=cursor.multistream) as restored:
                    restored.restore(snapshot)
                    actual = [hashlib.sha256(restored.decode(raw, samples, lost)).digest()
                              for raw, lost in sequence]
                    self.assertEqual(actual, expected)
            finally:
                cursor.close()

    def test_invalid_mapping_is_rejected_before_native_decode(self):
        with av.open(str(self.sources[-1])) as container:
            head = container.streams.audio[0].codec_context.extradata
        variants = [head[:18], head[:21]]
        for offset, value in ((9, 0), (18, 0), (18, 2), (19, 0), (20, 255), (21, 200)):
            invalid = bytearray(head)
            invalid[offset] = value
            variants.append(bytes(invalid))
        for invalid in variants:
            with self.subTest(header=invalid.hex()):
                stream = SimpleNamespace(index=0, type='audio',
                                         codec_context=SimpleNamespace(name='opus', extradata=invalid))
                container = SimpleNamespace(streams=SimpleNamespace(audio=[stream]))
                with patch('opus_cursor.open_media') as opened, \
                     patch.object(self.lib, 'opus_multistream_decoder_create') as create:
                    opened.return_value.__enter__.return_value = container
                    with self.assertRaises(RuntimeError):
                        LazyCursor('unused', self.lib)
                    create.assert_not_called()


if __name__ == '__main__':
    unittest.main()

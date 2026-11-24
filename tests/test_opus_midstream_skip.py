"""Real WebM DiscardPadding, including negative padding across packets."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from media import normalize_audio_timing
from test_opus_multistream import encode_source


EVENTS = {10: (480, 0), 20: (3072, 0), 30: (0, 240), 40: (0, 960), 50: (0, 1200)}
OVERRIDES = {10: (4096, 0), 11: (0, 120), 20: (65535, 0), 21: (480, 0)}


def padded_fixture(source, destination, events=EVENTS):
    negative_values = []
    with av.open(str(source)) as original, av.open(str(destination), 'w',
            options={'cluster_time_limit': '50'}) as output:
        stream = output.add_stream_from_template(original.streams.audio[0])
        dtype = av.packet.packet_sidedata_type_from_literal('skip_samples')
        for index, packet in enumerate(original.demux(original.streams.audio[0])):
            if packet.dts is None:
                continue
            if index in events:
                skip, discard = events[index]
                padding = skip or discard
                data = av.packet.PacketSideData(dtype, 10)
                data.update(bytes(4) + padding.to_bytes(4, 'little') + bytes(2))
                packet.set_sidedata(data)
                if skip:
                    negative_values.append(round(skip * 1_000_000_000 / 48000))
            packet.stream = stream
            output.mux(packet)
    # FFmpeg's muxer writes positive padding only. Negate those known fixture
    # integers without changing any EBML length, packet offset or payload.
    data = destination.read_bytes()
    for value in negative_values:
        width = max(1, (value.bit_length() + 8) // 8)
        marker = b'\x75\xa2' + bytes((0x80 | width,)) + value.to_bytes(width, 'big', signed=True)
        if data.count(marker) != 1:
            raise AssertionError('Expected one generated DiscardPadding element')
        data = data.replace(marker, marker[:3] + (-value).to_bytes(width, 'big', signed=True))
    destination.write_bytes(data)


class MidstreamSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.paths = []
        cls.events = {}
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}-source.webm'
            destination = Path(cls.directory.name) / f'{layout}-padded.webm'
            encode_source(source, layout)
            padded_fixture(source, destination)
            cls.paths.append(destination)
            cls.events[destination] = EVENTS
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}-source.webm'
            destination = Path(cls.directory.name) / f'{layout}-override.webm'
            padded_fixture(source, destination, OVERRIDES)
            cls.paths.append(destination)
            cls.events[destination] = OVERRIDES

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_demuxed_side_data_and_native_restore(self):
        for path in self.paths:
            with self.subTest(path=path.name):
                expected, observed = [], {}
                with av.open(str(path)) as container:
                    stream = container.streams.audio[0]
                    codec, end = codec_for(stream), None
                    for index, packet in enumerate(container.demux(stream)):
                        if index in self.events[path]:
                            side = packet.get_sidedata('skip_samples')
                            raw = bytes(side)
                            observed[index] = (int.from_bytes(raw[:4], 'little'),
                                               int.from_bytes(raw[4:8], 'little'))
                        for frame in codec.decode(packet):
                            end, _ = normalize_audio_timing(frame, end)
                            expected.append((frame.pts, frame.samples,
                                hashlib.sha256(frame.to_ndarray()).hexdigest()))
                self.assertEqual(observed, self.events[path])
                cursor = LazyCursor(path, self.lib)
                try:
                    cursor._interval = 480
                    for repeat in range(2):
                        actual = [(frame.pts, frame.samples,
                            hashlib.sha256(frame.to_ndarray()).hexdigest())
                            for frame in cursor.frames(cursor.origin)]
                        self.assertEqual(actual, expected)
                    self.assertLessEqual(cursor.max_snapshot_bytes, 2 * 1024 * 1024)
                finally:
                    cursor.close()

    def test_player_gaps_resampling_and_repeated_seek(self):
        for path in self.paths:
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        cursor._interval = 480
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for target in (0, .19, .21, .39, .42, .47, .61, .81, .47, .19, 0):
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate,
                                lambda block: actual.update(block.tobytes()))
                            expected = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                            self.assertEqual(result['samples'], len(expected) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                    finally:
                        cursor.close()

    def test_restore_partly_consumed_midstream_skip(self):
        for path in self.paths[:3]:
            with self.subTest(path=path.name):
                cursor = LazyCursor(path, self.lib)
                try:
                    cursor._interval = 480
                    request = cursor.frames(cursor.origin)
                    try:
                        for frame in request:
                            point = next((point for point in cursor.points
                                          if point.skip_remaining == 192), None)
                            if point is not None:
                                break
                        else:
                            self.fail('No snapshot with partly consumed skip')
                    finally:
                        request.close()
                    expected = []
                    with av.open(str(path)) as container:
                        stream = container.streams.audio[0]
                        codec, end = codec_for(stream), None
                        for packet in container.demux(stream):
                            for frame in codec.decode(packet):
                                end, _ = normalize_audio_timing(frame, end)
                                if frame.pts >= round(point.start * 48000):
                                    expected.append((frame.pts, frame.samples,
                                        hashlib.sha256(frame.to_ndarray()).hexdigest()))
                    before = cursor.restores
                    actual = [(frame.pts, frame.samples,
                        hashlib.sha256(frame.to_ndarray()).hexdigest())
                        for frame in cursor.frames(point.start)]
                    self.assertEqual(actual, expected)
                    self.assertEqual(cursor.restores, before + 1)
                finally:
                    cursor.close()


if __name__ == '__main__':
    unittest.main()

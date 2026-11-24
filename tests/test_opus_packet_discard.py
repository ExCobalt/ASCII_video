"""Silent real fragmented-MP4 overlap/discard and checkpoint checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from media import normalize_audio_timing
from test_opus_multistream import encode_source


def overlapping_fixture(source, destination, overlap, preskip=None):
    with av.open(str(source)) as original, av.open(str(destination), 'w', options={
            'movflags': 'empty_moov+frag_keyframe', 'frag_duration': '200000'}) as output:
        stream = output.add_stream_from_template(original.streams.audio[0])
        for packet in original.demux(original.streams.audio[0]):
            if packet.dts is not None:
                packet.stream = stream
                output.mux(packet)
    # Move two known fragment decode times backwards, preserving every box
    # length, packet offset and compressed payload. MOV marks overlaps discard.
    data = bytearray(destination.read_bytes())
    positions, start = [], 0
    while True:
        position = data.find(b'tfdt', start)
        if position < 0:
            break
        width = 8 if data[position + 4] == 1 else 4
        positions.append((position + 8, width))
        start = position + 4
    assert len(positions) == 11
    for index in (2, 5):
        position, width = positions[index]
        value = int.from_bytes(data[position:position + width], 'big')
        assert value == index * 9600
        data[position:position + width] = (value - overlap).to_bytes(width, 'big')
    if preskip is not None:
        # dOps stores the OpusHead pre-skip in big-endian order. The empty
        # moov has no edit list or initial skip side data to override it.
        position = data.find(b'dOps')
        assert position >= 0 and data.count(b'dOps') == 1
        data[position + 6:position + 8] = preskip.to_bytes(2, 'big')
    destination.write_bytes(data)


class PacketDiscardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.paths = []
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}-source.mp4'
            encode_source(source, layout)
            for overlap in (1920, 9600):
                destination = Path(cls.directory.name) / f'{layout}-{overlap}.mp4'
                overlapping_fixture(source, destination, overlap)
                cls.paths.append((destination, overlap))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_native_frames_and_restored_state(self):
        for path, overlap in self.paths:
            with self.subTest(path=path.name):
                expected, flagged = [], []
                with av.open(str(path)) as container:
                    stream = container.streams.audio[0]
                    codec, end = codec_for(stream), None
                    for index, packet in enumerate(container.demux(stream)):
                        if packet.is_discard:
                            flagged.append(index)
                        frames = codec.decode(packet)
                        if packet.is_discard:
                            self.assertEqual(frames, [])
                        for frame in frames:
                            end, _ = normalize_audio_timing(frame, end)
                            expected.append((frame.pts, frame.samples,
                                hashlib.sha256(frame.to_ndarray()).hexdigest()))
                count = overlap // 960
                self.assertEqual(flagged, list(range(20, 20 + count)) +
                                         list(range(50, 50 + count)))
                cursor = LazyCursor(path, self.lib)
                try:
                    cursor._interval = 480
                    for repeat in range(2):
                        actual = [(frame.pts, frame.samples,
                            hashlib.sha256(frame.to_ndarray()).hexdigest())
                            for frame in cursor.frames(cursor.origin)]
                        self.assertEqual(actual, expected)
                    self.assertGreater(cursor.restores, 0)
                    self.assertGreater(cursor.compactions, 0)
                    self.assertLessEqual(cursor.max_snapshot_bytes, 2 * 1024 * 1024)
                finally:
                    cursor.close()

    def test_player_resampling_forward_backward_and_repeat(self):
        for path, overlap in self.paths:
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        cursor._interval = 480
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for target in (0, .35, .39, .41, .95, 1.01, .39, 0, 1.95):
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate,
                                lambda block: actual.update(block.tobytes()))
                            expected = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                            self.assertEqual(result['samples'], len(expected) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                    finally:
                        cursor.close()

    def test_discard_consumes_pending_preskip_and_restores_partial_packet(self):
        for original, overlap in self.paths:
            path = original.with_stem(original.stem + '-pending')
            source = original.with_name(original.name.split('-')[0] + '-source.mp4')
            overlapping_fixture(source, path, overlap, preskip=65535)
            with self.subTest(path=path.name):
                expected, flagged = [], 0
                with av.open(str(path)) as container:
                    stream = container.streams.audio[0]
                    self.assertEqual(int.from_bytes(stream.codec_context.extradata[10:12],
                                                   'little'), 65535)
                    codec, end = codec_for(stream), None
                    for index, packet in enumerate(container.demux(stream)):
                        if index == 0:
                            self.assertFalse(packet.has_sidedata('skip_samples'))
                        flagged += packet.is_discard
                        for frame in codec.decode(packet):
                            end, _ = normalize_audio_timing(frame, end)
                            expected.append((frame.pts, frame.samples,
                                hashlib.sha256(frame.to_ndarray()).hexdigest()))
                self.assertEqual(flagged, 2 * overlap // 960)
                self.assertEqual(expected[0][:2], (65535, 705))
                cursor = LazyCursor(path, self.lib)
                try:
                    cursor._interval = 480
                    for repeat in range(2):
                        actual = [(frame.pts, frame.samples,
                            hashlib.sha256(frame.to_ndarray()).hexdigest())
                            for frame in cursor.frames(cursor.origin)]
                        self.assertEqual(actual, expected)
                    self.assertEqual(cursor.points[0].skip_remaining, 255)
                    self.assertGreater(cursor.restores, 0)
                    for rate in (44100, 48000):
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for target in (0, .01, .19, .41, .01, 0):
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate,
                                lambda block: actual.update(block.tobytes()))
                            tail = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(tail).digest())
                            self.assertEqual(result['samples'], len(tail) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                finally:
                    cursor.close()


if __name__ == '__main__':
    unittest.main()

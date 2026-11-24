"""Real multi-page Opus packets, missing fragments and bounded seek recovery."""
import ctypes as ct
import hashlib
from pathlib import Path
import tempfile
import unittest

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from test_opus_lacing import frames
from test_opus_multistream import encode_source
from test_opus_page_loss import pages


def continued_fixture(source, destination, lib):
    pad = lib.opus_multistream_packet_pad
    pad.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
    pad.restype = ct.c_int
    with av.open(str(source)) as original, av.open(str(destination), 'w') as output:
        stream = output.add_stream_from_template(original.streams.audio[0])
        head = stream.codec_context.extradata
        for index, packet in enumerate(original.demux(original.streams.audio[0])):
            if packet.dts is None:
                continue
            if index == 60:
                raw = bytes(packet)
                buffer = ct.create_string_buffer(raw, 160000)
                assert pad(buffer, len(raw), len(buffer), head[19]) == 0
                padded = av.Packet(buffer.raw)
                padded.pts, padded.dts = packet.pts, packet.dts
                padded.duration, padded.time_base = packet.duration, packet.time_base
                packet = padded
            packet.stream = stream
            output.mux(packet)


class ContinuedPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.valid, cls.invalid = [], []
        cls.originals = []
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}.ogg'
            complete = source.with_stem(source.stem + '-continued')
            encode_source(source, layout, seconds=4)
            continued_fixture(source, complete, cls.lib)
            original = pages(complete)
            first = next(i for i, page in enumerate(original)
                         if int.from_bytes(page[6:14], 'little', signed=True) == -1)
            assert not original[first][5] & 1
            assert original[first + 1][5] & 1 and original[first + 2][5] & 1
            assert int.from_bytes(original[first + 1][6:14], 'little', signed=True) == -1
            assert int.from_bytes(original[first + 2][6:14], 'little', signed=True) >= 0
            cls.originals.append((source, complete))
            cls.valid.append((complete, False))
            for fragment in range(3):
                path = source.with_stem(source.stem + f'-missing-fragment-{fragment}')
                index = first + fragment
                path.write_bytes(b''.join(original[:index] + original[index + 1:]))
                if fragment == 0:
                    cls.valid.append((path, True))
                else:
                    cls.invalid.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_complete_packet_pcm_and_missing_start(self):
        for source, complete in self.originals:
            with self.subTest(path=complete.name):
                self.assertEqual(frames(complete), frames(source))
                with av.open(str(complete)) as container:
                    sizes = [packet.size for packet in container.demux(container.streams.audio[0])]
                self.assertEqual(sizes.count(160000), 1)
        for path, missing in self.valid:
            with self.subTest(path=path.name):
                expected = frames(path)
                self.assertEqual(len(expected), 200 if missing else 201)
                gaps = [following[0] - (previous[0] + previous[1])
                        for previous, following in zip(expected, expected[1:])
                        if following[0] > previous[0] + previous[1]]
                self.assertEqual(gaps, [960] if missing else [])
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

    def test_player_tails_around_continued_packet_and_gap(self):
        for path, missing in self.valid:
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        cursor._interval = 480
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for index, target in enumerate((0, .97, 1.19, 1.20, 1.21, 1.23, 3.7, 1.20, 0)):
                            if index == 7:
                                cursor._release()
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate,
                                lambda block: actual.update(block.tobytes()))
                            tail = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(tail).digest())
                            self.assertEqual(result['samples'], len(tail) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                    finally:
                        cursor.close()

    def test_missing_middle_or_end_fails_like_pinned_ffmpeg(self):
        # The pinned FFmpeg does not inspect Ogg sequence numbers here. Missing
        # middle/end fragments reach the codec as corrupt data. Preserve the
        # decoder error and release resources instead of returning invalid PCM.
        for path in self.invalid:
            with self.subTest(path=path.name):
                with av.open(str(path)) as container:
                    codec = codec_for(container.streams.audio[0])
                    with self.assertRaises(av.error.InvalidDataError):
                        for packet in container.demux(container.streams.audio[0]):
                            codec.decode(packet)
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


if __name__ == '__main__':
    unittest.main()

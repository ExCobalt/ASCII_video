"""Real missing Ogg pages: exact surviving PCM, silence gaps and restored seeks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from opus_test_helpers import LazyCursor, ROOT, drain, library
from test_opus_lacing import frames
from test_opus_multistream import encode_source


def pages(path):
    data, offset, result = path.read_bytes(), 0, []
    while offset < len(data):
        assert data[offset:offset + 4] == b'OggS'
        count = data[offset + 26]
        laces = data[offset + 27:offset + 27 + count]
        size = 27 + count + sum(laces)
        result.append(data[offset:offset + size])
        offset += size
    assert offset == len(data)
    return result


class PageLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.paths = []
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}.ogg'
            encode_source(source, layout, seconds=4)
            original = pages(source)
            assert len(original) >= 7
            # These generated pages end at packet boundaries. Loss of a page
            # containing a continued packet is a separate case.
            assert all(not page[5] & 1 and page[26] and page[26 + page[26]] < 255
                       for page in original)
            for lost_count in (1, 2):
                path = source.with_stem(source.stem + f'-lost-{lost_count}')
                path.write_bytes(b''.join(original[:3] + original[3 + lost_count:]))
                before = int.from_bytes(original[2][6:14], 'little')
                after = int.from_bytes(original[2 + lost_count][6:14], 'little')
                missing = sum(sum(value < 255 for value in page[27:27 + page[26]])
                              for page in original[3:3 + lost_count])
                cls.paths.append((path, original, lost_count, before, after, missing))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_real_sequence_holes_and_native_checkpoint_pcm(self):
        for path, original, lost_count, before, after, missing in self.paths:
            with self.subTest(path=path.name):
                remaining = pages(path)
                self.assertEqual(remaining, original[:3] + original[3 + lost_count:])
                sequences = [int.from_bytes(page[18:22], 'little') for page in remaining]
                self.assertEqual(sequences[3] - sequences[2], lost_count + 1)
                expected = frames(path)
                self.assertEqual(len(expected), 201 - missing)
                gaps = [following[0] - (previous[0] + previous[1])
                        for previous, following in zip(expected, expected[1:])
                        if following[0] > previous[0] + previous[1]]
                self.assertEqual(gaps, [after - before])
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

    def test_player_gap_silence_and_seek_on_both_sides(self):
        for path, original, lost_count, before, after, missing in self.paths:
            start, end = (before - 312) / 48000, (after - 312) / 48000
            middle = (start + end) / 2
            targets = (0, .25, start - .01, start + .01, middle, end + .01, 3.7, middle, 0)
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        cursor._interval = 480
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        sample = round(middle * rate)
                        self.assertEqual(reference[sample * 8:(sample + 100) * 8], bytes(800))
                        for index, target in enumerate(targets):
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


if __name__ == '__main__':
    unittest.main()

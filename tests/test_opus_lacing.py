"""Real Matroska Xiph/EBML/fixed lacing and shared-position Opus anchors."""
from collections import Counter
import ctypes as ct
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from media import normalize_audio_timing
from test_opus_multistream import encode_source


def vint(value):
    for width in range(1, 9):
        if 0 <= value < (1 << (7 * width)) - 1:
            return (value | (1 << (7 * width))).to_bytes(width, 'big')
    raise ValueError('Fixture VINT too large')


def signed_vint(value):
    for width in range(1, 9):
        bias = (1 << (7 * width - 1)) - 1
        if -bias <= value <= bias:
            return (value + bias | (1 << (7 * width))).to_bytes(width, 'big')
    raise ValueError('Fixture signed VINT too large')


def element(identifier, value):
    if isinstance(value, int):
        value = value.to_bytes(max(1, (value.bit_length() + 7) // 8), 'big')
    return bytes.fromhex(identifier) + vint(len(value)) + value


def laced_fixture(source, destination, mode, grouped, lib):
    with av.open(str(source)) as container:
        stream = container.streams.audio[0]
        head = stream.codec_context.extradata
        packets = [packet for packet in container.demux(stream) if packet.size]
        padding = int.from_bytes(bytes(packets[-1].get_sidedata('skip_samples'))[4:8], 'little')
        raw = [bytes(packet) for packet in packets]
    header = element('1a45dfa3', b''.join(element(key, value) for key, value in (
        ('4286', 1), ('42f7', 1), ('42f2', 4), ('42f3', 8), ('4282', b'matroska'),
        ('4287', 4), ('4285', 2))))
    info = element('1549a966', element('2ad7b1', 1_000_000) +
        element('4d80', b'silent-fixture') + element('5741', b'silent-fixture') +
        element('4489', struct.pack('>d', len(raw) * 20.0)))
    track = element('1654ae6b', element('ae', b''.join(element(key, value)
        for key, value in (('d7', 1), ('73c5', 1), ('83', 2), ('9c', 1),
        ('86', b'A_OPUS'), ('63a2', head), ('23e383', 20_000_000),
        ('56aa', 6_500_000), ('56bb', 80_000_000), ('e1',
        element('b5', struct.pack('>d', 48000.0)) + element('9f', head[9]))))))
    pad = lib.opus_multistream_packet_pad
    pad.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
    pad.restype = ct.c_int
    clusters, index, group_index = [], 0, 0
    while index < len(raw):
        # Keep the final packet outside a lace: its DiscardPadding applies
        # to that packet only. Exercise 2, 3 and 7 packets sharing one position.
        count = min((2, 3, 7)[group_index % 3], len(raw) - 1 - index)
        count = max(1, count)
        block_raw = raw[index:index + count]
        if mode == 'fixed' and count > 1:
            width = max(map(len, block_raw)) + 16
            padded = []
            for packet in block_raw:
                buffer = ct.create_string_buffer(packet, width)
                assert pad(buffer, len(packet), width, head[19]) == 0
                padded.append(buffer.raw)
            block_raw = padded
        flags, sizes = 0, b''
        if count > 1:
            flags = {'xiph': 2, 'fixed': 4, 'ebml': 6}[mode]
            sizes = bytes((count - 1,))
            if mode == 'xiph':
                sizes += b''.join(b'\xff' * (len(packet) // 255) + bytes((len(packet) % 255,))
                                  for packet in block_raw[:-1])
            elif mode == 'ebml':
                sizes += vint(len(block_raw[0])) + b''.join(
                    signed_vint(len(block_raw[i]) - len(block_raw[i - 1]))
                    for i in range(1, count - 1))
        payload = b'\x81\x00\x00' + bytes((flags,)) + sizes + b''.join(block_raw)
        last = index + count == len(raw)
        if grouped or last:
            block = element('a1', payload) + element('9b', count * 20)
            if last and padding:
                nanoseconds = round(padding * 1_000_000_000 / 48000)
                width = max(1, (nanoseconds.bit_length() + 8) // 8)
                block += element('75a2', nanoseconds.to_bytes(width, 'big', signed=True))
            block = element('a0', block)
        else:
            block = element('a3', payload[:3] + bytes((flags | 128,)) + payload[4:])
        clusters.append(element('1f43b675', element('e7', index * 20) + block))
        index += count
        group_index += 1
    destination.write_bytes(header + bytes.fromhex('1853806701ffffffffffffff') +
                            info + track + b''.join(clusters))


def frames(path):
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        codec, end = codec_for(stream), None
        for packet in container.demux(stream):
            for frame in codec.decode(packet):
                end, _ = normalize_audio_timing(frame, end)
                result.append((frame.pts, frame.samples,
                    hashlib.sha256(frame.to_ndarray()).hexdigest()))
    return result


class LacingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.paths = []
        for layout in ('mono', 'stereo', '5.1'):
            source = Path(cls.directory.name) / f'{layout}-source.webm'
            encode_source(source, layout)
            for mode in ('xiph', 'ebml', 'fixed'):
                for grouped in (False, True):
                    path = source.with_name(f'{layout}-{mode}-{grouped}.mkv')
                    laced_fixture(source, path, mode, grouped, cls.lib)
                    cls.paths.append((path, source))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_real_laces_preserve_pcm_and_restore_shared_positions(self):
        for path, source in self.paths:
            with self.subTest(path=path.name):
                expected = frames(source)
                self.assertEqual(frames(path), expected)
                with av.open(str(path)) as container:
                    packets = [packet for packet in container.demux(container.streams.audio[0])
                               if packet.size]
                self.assertEqual(len(packets), 101)
                self.assertEqual(max(Counter(packet.pos for packet in packets).values()), 7)
                self.assertEqual(len(set((packet.pos, packet.pts) for packet in packets)), 101)
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

    def test_player_tails_and_reopen_at_laced_anchor(self):
        for path, source in self.paths:
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        cursor._interval = 480
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for index, target in enumerate((0, .031, .071, .33, .91, .071, 0)):
                            if index == 5:
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

"""Silent native Opus regression: pre-skip may cover several packets."""
from fractions import Fraction
import ctypes as ct
import hashlib
from itertools import chain
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from opus_cursor import Decoder
from media import normalize_audio_timing


def fixture(source, destination, preskip, container_format='ogg', codec_delay=None, gain=0, mapping=None):
    with av.open(str(source)) as original, av.open(str(destination), 'w', format=container_format) as output:
        stream = output.add_stream_from_template(original.streams.audio[0])
        original_delay = round(original.streams.audio[0].codec_context.delay * 1_000_000_000 / 48000)
        head = bytearray(stream.codec_context.extradata)
        head[10:12] = preskip.to_bytes(2, 'little')
        head[16:18] = gain.to_bytes(2, 'little', signed=True)
        if mapping is not None:
            head[21:21 + head[9]] = bytes(mapping)
        stream.codec_context.extradata = bytes(head)
        for packet in original.demux(original.streams.audio[0]):
            if packet.dts is None:
                continue
            if packet.pts * packet.time_base >= 4:
                break
            packet.stream = stream
            output.mux(packet)
    if codec_delay is not None:
        # Change only the known fixture's fixed-width CodecDelay integer.
        # Keeping its width preserves all enclosing EBML lengths and offsets.
        width = max(1, (original_delay.bit_length() + 7) // 8)
        marker = b'\x56\xaa' + bytes((0x80 | width,)) + original_delay.to_bytes(width, 'big')
        data = destination.read_bytes()
        if data.count(marker) != 1:
            raise AssertionError('Expected one generated CodecDelay element')
        delay_ns = round(codec_delay * 1_000_000_000 / 48000)
        destination.write_bytes(data.replace(marker,
            marker[:3] + delay_ns.to_bytes(width, 'big')))


class LongPreskipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.sources = []
        for channels, duration in ((1, 20), (2, 60), (1, 2.5)):
            path = Path(cls.directory.name) / f'source-{channels}-{duration}.webm'
            layout = 'mono' if channels == 1 else 'stereo'
            t = np.arange(4 * 48000, dtype=np.float64) / 48000
            values = np.column_stack([.2 * np.sin(2 * np.pi * (233 + 71 * i) * t)
                                      for i in range(channels)]).astype(np.float32)
            with av.open(str(path), 'w') as output:
                stream = output.add_stream('libopus', rate=48000,
                                           options={'frame_duration': str(duration)})
                stream.layout = layout
                frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format='flt', layout=layout)
                frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
                for packet in stream.encode(frame):
                    output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
            cls.sources.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_packet_clock_and_restoration(self):
        for source in self.sources[:2]:
            for preskip in (0, 312, 960, 3840, 65535):
                with self.subTest(source=source.name, preskip=preskip):
                    path = Path(self.directory.name) / 'input.opus'
                    fixture(source, path, preskip)
                    expected = []
                    with av.open(str(path)) as container:
                        stream = container.streams.audio[0]
                        self.assertEqual(int.from_bytes(stream.codec_context.extradata[10:12], 'little'), preskip)
                        codec, end = codec_for(stream), None
                        for packet in container.demux(stream):
                            for frame in codec.decode(packet):
                                end, _ = normalize_audio_timing(frame, end)
                                expected.append((frame.pts, frame.samples, hashlib.sha256(frame.to_ndarray()).hexdigest()))
                    self.assertTrue(expected)
                    cursor = LazyCursor(path, self.lib)
                    try:
                        for repeat in range(2):
                            actual = [(frame.pts, frame.samples, hashlib.sha256(frame.to_ndarray()).hexdigest())
                                      for frame in cursor.frames(cursor.origin)]
                            self.assertEqual(actual, expected)
                        self.assertGreater(cursor.restores, 0)
                        self.assertLessEqual(cursor.max_snapshot_bytes, 2 * 1024 * 1024)
                    finally:
                        cursor.close()

    def test_player_resampling_and_repeated_seek(self):
        source = self.sources[0]
        for preskip in (960, 3840, 65535):
            path = Path(self.directory.name) / 'player.opus'
            fixture(source, path, preskip)
            for rate in (44100, 48000):
                with self.subTest(preskip=preskip, rate=rate):
                    cursor = LazyCursor(path, self.lib)
                    try:
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate, lambda block: reference.extend(block.tobytes()), indexed=False)
                        for target in (0, .731, .113, .731, baseline['end_time'] - .101, 0):
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate, lambda block: actual.update(block.tobytes()))
                            expected = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                            self.assertEqual(result['samples'], len(expected) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                    finally:
                        cursor.close()

    def test_cancel_and_reopen_before_first_audible_packet(self):
        path = Path(self.directory.name) / 'cancel.opus'
        fixture(self.sources[0], path, 65535)
        reference = LazyCursor(path, self.lib)
        try:
            expected = [(frame.pts, frame.samples, hashlib.sha256(frame.to_ndarray()).hexdigest())
                        for frame in reference.frames(reference.origin)]
        finally:
            reference.close()
        for reopen in (False, True):
            with self.subTest(reopen=reopen):
                cursor = LazyCursor(path, self.lib)
                try:
                    self.assertEqual(list(cursor.frames(cursor.origin,
                        lambda: cursor.decoded_frames >= 5)), [])
                    self.assertEqual(cursor.decoded_frames, 5)
                    self.assertFalse(cursor.points)
                    self.assertIsNone(cursor.end)
                    if reopen:
                        cursor._release()
                    actual = [(frame.pts, frame.samples, hashlib.sha256(frame.to_ndarray()).hexdigest())
                              for frame in cursor.frames(cursor.origin)]
                    self.assertEqual(actual, expected)
                    self.assertTrue(cursor.request_closed)
                finally:
                    cursor.close()

    def test_container_skip_overrides_header(self):
        for source in self.sources[:2]:
            for preskip in (0, 960, 65535):
                with self.subTest(source=source.name, header_preskip=preskip):
                    path = Path(self.directory.name) / 'conflicting.webm'
                    fixture(source, path, preskip, 'webm')
                    expected = []
                    with av.open(str(path)) as container:
                        stream = container.streams.audio[0]
                        packets = container.demux(stream)
                        first = next(packets)
                        skip_data = next(bytes(side) for side in first.iter_sidedata()
                                         if side.data_type == 'skip_samples')
                        effective = int.from_bytes(skip_data[:4], 'little')
                        self.assertEqual(effective, 312)
                        self.assertNotEqual(effective, preskip)
                        codec, end = codec_for(stream), None
                        for packet in chain((first,), packets):
                            for frame in codec.decode(packet):
                                end, _ = normalize_audio_timing(frame, end)
                                expected.append((frame.pts, frame.samples,
                                    hashlib.sha256(frame.to_ndarray()).hexdigest()))
                    cursor = LazyCursor(path, self.lib)
                    try:
                        self.assertEqual(cursor.preskip, effective)
                        for repeat in range(2):
                            actual = [(frame.pts, frame.samples,
                                       hashlib.sha256(frame.to_ndarray()).hexdigest())
                                      for frame in cursor.frames(cursor.origin)]
                            self.assertEqual(actual, expected)
                    finally:
                        cursor.close()

    def test_real_container_delay_covers_multiple_packets(self):
        path = Path(self.directory.name) / 'delay.webm'
        source = self.sources[-1]
        fixture(source, path, 65535, 'webm', codec_delay=480)
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            first = next(container.demux(stream))
            side = next(bytes(side) for side in first.iter_sidedata()
                        if side.data_type == 'skip_samples')
            self.assertEqual(int.from_bytes(side[:4], 'little'), 480)
            self.assertEqual(self.lib.opus_packet_get_nb_samples(bytes(first), first.size, 48000), 120)
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                cursor = LazyCursor(path, self.lib)
                try:
                    self.assertEqual(cursor.preskip, 480)
                    reference = bytearray()
                    baseline = drain(cursor, 0, rate,
                        lambda block: reference.extend(block.tobytes()), indexed=False)
                    for target in (0, .731, .113, .731, 0):
                        actual = hashlib.sha256()
                        result = drain(cursor, target, rate,
                            lambda block: actual.update(block.tobytes()))
                        expected = memoryview(reference)[round(target * rate) * 8:]
                        self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                        self.assertEqual(result['samples'], len(expected) // 8)
                        self.assertEqual(result['end_time'], baseline['end_time'])
                finally:
                    cursor.close()

    def test_header_gain_matches_ffmpeg_and_survives_restore(self):
        for source in self.sources[:2]:
            for container_format in ('ogg', 'webm'):
                for gain in (-32768, -1536, -1, 0, 1, 1536, 32767):
                    with self.subTest(source=source.name, format=container_format, gain=gain):
                        path = Path(self.directory.name) / 'gain.opus'
                        fixture(source, path, 3840, container_format, gain=gain)
                        expected = []
                        with av.open(str(path)) as container:
                            stream = container.streams.audio[0]
                            self.assertEqual(int.from_bytes(stream.codec_context.extradata[16:18],
                                'little', signed=True), gain)
                            codec, end = codec_for(stream), None
                            for packet in container.demux(stream):
                                for frame in codec.decode(packet):
                                    end, _ = normalize_audio_timing(frame, end)
                                    expected.append((frame.pts, frame.samples,
                                        hashlib.sha256(frame.to_ndarray()).hexdigest()))
                        cursor = LazyCursor(path, self.lib)
                        try:
                            for repeat in range(2):
                                actual = [(frame.pts, frame.samples,
                                    hashlib.sha256(frame.to_ndarray()).hexdigest())
                                    for frame in cursor.frames(cursor.origin)]
                                self.assertEqual(actual, expected)
                            self.assertGreater(cursor.restores, 0)
                        finally:
                            cursor.close()

    def test_header_gain_in_player_resampling(self):
        for source in self.sources[:2]:
            for container_format in ('ogg', 'webm'):
                for gain in (-1536, 1536, 32767):
                    path = Path(self.directory.name) / 'gain-player.opus'
                    fixture(source, path, 65535, container_format, gain=gain)
                    for rate in (44100, 48000):
                        with self.subTest(source=source.name, format=container_format, gain=gain, rate=rate):
                            cursor = LazyCursor(path, self.lib)
                            try:
                                reference = bytearray()
                                baseline = drain(cursor, 0, rate,
                                    lambda block: reference.extend(block.tobytes()), indexed=False)
                                for target in (0, .731, .113, .731, 0):
                                    actual = hashlib.sha256()
                                    result = drain(cursor, target, rate,
                                        lambda block: actual.update(block.tobytes()))
                                    expected = memoryview(reference)[round(target * rate) * 8:]
                                    self.assertEqual(actual.digest(), hashlib.sha256(expected).digest())
                                    self.assertEqual(result['samples'], len(expected) // 8)
                                    self.assertEqual(result['end_time'], baseline['end_time'])
                            finally:
                                cursor.close()

    def test_gain_is_part_of_copied_decoder_state(self):
        for channels, source in enumerate(self.sources[:2], 1):
            with av.open(str(source)) as container:
                packets = container.demux(container.streams.audio[0])
                payloads = [bytes(next(packets)) for _ in range(3)]
            samples = self.lib.opus_packet_get_nb_samples(payloads[0], len(payloads[0]), 48000)
            sequence = ((payloads[1], False), (b'', True), (b'', True), (payloads[2], False))
            for gain in (-32768, -1536, 1536, 32767):
                with self.subTest(channels=channels, gain=gain):
                    with Decoder(self.lib, channels, gain=gain) as decoder:
                        decoder.decode(payloads[0], samples)
                        state = decoder.snapshot()
                        expected = [hashlib.sha256(decoder.decode(raw, samples, lost)).digest()
                                    for raw, lost in sequence]
                    with Decoder(self.lib, channels) as restored:
                        restored.restore(state)
                        actual_gain = ct.c_int32()
                        # OPUS_GET_GAIN(&value) is request 4045, not 4035.
                        self.assertEqual(self.lib.opus_decoder_ctl(restored.state, 4045,
                            ct.byref(actual_gain)), 0)
                        self.assertEqual(actual_gain.value, gain)
                        actual = [hashlib.sha256(restored.decode(raw, samples, lost)).digest()
                                  for raw, lost in sequence]
                        self.assertEqual(actual, expected)

    def test_gain_errors_do_not_leak_decoder(self):
        for gain in (-32769, 32768):
            with patch.object(self.lib, 'opus_decoder_create') as create:
                with self.assertRaises(ValueError):
                    Decoder(self.lib, 1, gain=gain)
                create.assert_not_called()
        with patch.object(self.lib, 'opus_decoder_ctl', return_value=-1), \
             patch.object(self.lib, 'opus_decoder_destroy', wraps=self.lib.opus_decoder_destroy) as destroy:
            with self.assertRaisesRegex(RuntimeError, 'Cannot set Opus decoder gain'):
                Decoder(self.lib, 1, gain=1536)
            destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()

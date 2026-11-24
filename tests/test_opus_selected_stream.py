"""Silent selected-track Opus checkpoints in real multistream containers."""
import hashlib
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import av

from opus_test_helpers import LazyCursor, ROOT, codec_for, drain, library
from media import normalize_audio_timing, probe
from test_formats import fixture


class SelectedStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib, _ = library()
        (ROOT / 'build').mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        source = Path(cls.directory.name) / 'two-tracks.webm'
        fixture(source, 'libvpx-vp9', 'libopus', tracks=2, default=1,
                duration=2, tone=True, layout='stereo')
        cls.paths = []
        for default in (1, 2):
            path = source.with_name(f'default-{default}.mkv')
            with av.open(str(source)) as original, av.open(str(path), 'w',
                    options={'cluster_time_limit': '50'}) as output:
                streams = []
                for item in original.streams:
                    stream = output.add_stream_from_template(item)
                    if item.type == 'audio':
                        head = bytearray(stream.codec_context.extradata)
                        head[16:18] = (1536 if item.index == 2 else -1536).to_bytes(
                            2, 'little', signed=True)
                        stream.codec_context.extradata = bytes(head)
                        stream.disposition = (av.stream.Disposition.default
                            if item.index == default else av.stream.Disposition(0))
                    streams.append(stream)
                for packet in original.demux():
                    if packet.dts is not None:
                        if packet.stream_index == 2:
                            delay = round(Fraction(1, 8) / packet.time_base)
                            packet.pts += delay
                            packet.dts += delay
                        packet.stream = streams[packet.stream_index]
                        output.mux(packet)
            cls.paths.append((path, default))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_selected_frames_headers_origin_and_restore(self):
        for path, default in self.paths:
            self.assertEqual(probe(path).audio_index, default)
            pcm = []
            for index in (1, 2):
                with self.subTest(path=path.name, stream=index):
                    expected = []
                    with av.open(str(path)) as container:
                        stream = container.streams[index]
                        codec, end = codec_for(stream), None
                        for packet in container.demux(stream):
                            for frame in codec.decode(packet):
                                end, _ = normalize_audio_timing(frame, end)
                                expected.append((frame.pts, frame.samples,
                                    hashlib.sha256(frame.to_ndarray()).hexdigest()))
                    pcm.append(expected[0][2])
                    cursor = LazyCursor(path, self.lib, stream_index=index)
                    try:
                        self.assertEqual(cursor.stream_index, index)
                        self.assertEqual(cursor.gain, 1536 if index == 2 else -1536)
                        self.assertEqual(round(cursor.origin * 48000), expected[0][0])
                        cursor._interval = 480
                        for repeat in range(2):
                            actual = [(frame.pts, frame.samples,
                                hashlib.sha256(frame.to_ndarray()).hexdigest())
                                for frame in cursor.frames(cursor.origin)]
                            self.assertEqual(actual, expected)
                        self.assertGreater(cursor.restores, 0)
                    finally:
                        cursor.close()
            self.assertNotEqual(pcm[0], pcm[1])

    def test_default_selected_player_tails(self):
        for path, default in self.paths:
            info = probe(path)
            for rate in (44100, 48000):
                with self.subTest(path=path.name, rate=rate):
                    cursor = LazyCursor(path, self.lib, stream_index=info.audio_index)
                    try:
                        # AudioPlayer uses the media origin, which includes video
                        # before a delayed default audio track.
                        if default == 2:
                            self.assertGreater(cursor.origin, info.origin)
                        cursor.origin = info.origin
                        reference = bytearray()
                        baseline = drain(cursor, 0, rate,
                            lambda block: reference.extend(block.tobytes()), indexed=False)
                        for target in (0, .05, .13, .71, .05, 0):
                            actual = hashlib.sha256()
                            result = drain(cursor, target, rate,
                                lambda block: actual.update(block.tobytes()))
                            tail = memoryview(reference)[round(target * rate) * 8:]
                            self.assertEqual(actual.digest(), hashlib.sha256(tail).digest())
                            self.assertEqual(result['samples'], len(tail) // 8)
                            self.assertEqual(result['end_time'], baseline['end_time'])
                    finally:
                        cursor.close()

    def test_reject_video_and_missing_stream(self):
        path, _ = self.paths[0]
        with self.assertRaisesRegex(ValueError, 'not audio'):
            LazyCursor(path, self.lib, stream_index=0)
        with self.assertRaises(IndexError):
            LazyCursor(path, self.lib, stream_index=99)


if __name__ == '__main__':
    unittest.main()

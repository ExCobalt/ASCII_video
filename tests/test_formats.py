"""Real required codecs/containers with synthetic media and fake audio devices."""

from fractions import Fraction
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

from audio import AudioPlayer
from cli import main, parser
from export import render
from media import normalize_audio_timing, probe, timestamp


FORMATS = (("mp4", "libx264", "aac"), ("mkv", "libx264", "aac"),
           ("webm", "libvpx-vp9", "libopus"))


def fixture(path, video_codec="libx264", audio_codec="aac", *, video=True, tracks=1, default=0,
            duration=.3, tone=False, layout="mono", audio_options=None):
    with av.open(str(path), "w") as output:
        video_stream = None
        if video:
            video_stream = output.add_stream(video_codec, rate=30)
            video_stream.width, video_stream.height, video_stream.pix_fmt = 64, 32, "yuv420p"
            video_stream.codec_context.time_base = Fraction(1, 30)
            if video_codec == "libx264":
                video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
            else:
                video_stream.options = {"deadline": "realtime", "cpu-used": "8"}
            if tone:
                video_stream.options["g"] = "30"
        audios = []
        for index in range(tracks):
            stream = output.add_stream(audio_codec, rate=48000)
            stream.layout = layout
            stream.bit_rate = 96000
            stream.codec_context.format = "fltp" if audio_codec == "aac" else "flt"
            stream.codec_context.time_base = Fraction(1, 48000)
            if tone and audio_codec == "aac":
                stream.options = {"aac_pns": "0"}
            if audio_options:
                stream.options = {**stream.options, **audio_options}
            stream.disposition = av.stream.Disposition.default if index == default else av.stream.Disposition(0)
            audios.append(stream)
        packets = []
        if video_stream is not None:
            gradient = np.linspace(0, 255, 64, dtype=np.uint8)
            for index in range(round(duration * 30)):
                image = np.empty((32, 64, 3), np.uint8)
                image[..., 0] = gradient
                image[..., 1] = (index * 25) % 256
                image[..., 2] = gradient[::-1]
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 30)
                packets.extend(video_stream.encode(frame))
            packets.extend(video_stream.encode(None))
        for stream in audios:
            samples = round(duration * 48000)
            for start in range(0, samples, 1024):
                count = min(1024, samples - start)
                values = np.zeros((1, count), dtype=np.float32)
                # A short artificial impulse, never played through hardware.
                positions = np.arange(start, start + count)
                values[0, (positions >= 4800) & (positions < 4900)] = .5
                if tone:
                    values[0] = .25 * np.sin(positions * .073) + .05 * np.sin(positions * .013)
                if layout == "stereo":
                    right = .2 * np.cos(positions * .047) if tone else values[0] * .5
                    values = np.stack((values[0], right)).astype(np.float32)
                    if not stream.codec_context.format.is_planar:
                        values = values.T.reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(values, format=stream.codec_context.format.name, layout=layout)
                frame.pts, frame.time_base, frame.sample_rate = start, Fraction(1, 48000), 48000
                packets.extend(stream.encode(frame))
            packets.extend(stream.encode(None))
        packets.sort(key=lambda packet: packet.dts * packet.time_base)
        for packet in packets:
            output.mux(packet)


class FakeOutputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.time = 100.0
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True


class FormatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def source(self, extension, video_codec, audio_codec, **kwargs):
        path = self.directory / f"синтетический ролик.{extension}"
        fixture(path, video_codec, audio_codec, **kwargs)
        return path

    def test_required_containers_decode_and_probe_preserves_first_stream_origin(self):
        for extension, video_codec, audio_codec in FORMATS:
            with self.subTest(extension=extension):
                path = self.source(extension, video_codec, audio_codec)
                info = probe(path)
                with av.open(str(path)) as source:
                    video = source.streams.video[0]
                    video_frames = list(source.decode(video))
                    self.assertEqual(video.codec_context.name, "h264" if video_codec == "libx264" else "vp9")
                    self.assertEqual(len(video_frames), 9)
                    self.assertTrue(all((frame.width, frame.height) == (64, 32) for frame in video_frames))
                    self.assertEqual(info.video_index, video.index)
                with av.open(str(path)) as source:
                    audio = source.streams.audio[0]
                    frames = list(source.decode(audio))
                    self.assertEqual(audio.codec_context.name, "aac" if audio_codec == "aac" else "opus")
                    self.assertEqual(info.audio_index, audio.index)
                    self.assertTrue(all(frame.sample_rate == 48000 for frame in frames))
                    self.assertTrue(all(frame.pts is not None for frame in frames))
                    self.assertGreater(max(np.abs(frame.to_ndarray()).max() for frame in frames), .01)
                self.assertEqual(info.origin, min(timestamp(video_frames[0]), timestamp(frames[0])))

    def test_audio_default_stream_selection_and_lowest_index_fallback(self):
        for default in (0, 1, None):
            with self.subTest(default=default):
                path = self.source("mkv", "libx264", "aac", tracks=2, default=default)
                with av.open(str(path)) as source:
                    audios = list(source.streams.audio)
                    flagged = [stream for stream in audios if stream.disposition & av.stream.Disposition.default]
                    expected = min(flagged or audios, key=lambda stream: stream.index).index
                    if default == 1:
                        self.assertEqual(expected, audios[1].index)
                self.assertEqual(probe(path).audio_index, expected)
                self.assertIsNone(probe(path, audio=False).audio_index)

    def test_container_clock_quantization_snaps_but_real_gaps_and_overlaps_remain(self):
        previous_end = None
        for raw_pts, normalized_pts in ((0, 0), (21, 1024), (43, 2048), (64, 3072), (85, 4096)):
            frame = SimpleNamespace(pts=raw_pts, time_base=Fraction(1, 1000),
                                    sample_rate=48000, samples=1024)
            previous_end, gap = normalize_audio_timing(frame, previous_end)
            self.assertFalse(gap)
            self.assertEqual(frame.pts, normalized_pts)
            self.assertEqual(frame.time_base, Fraction(1, 48000))
        frame = SimpleNamespace(pts=200, time_base=Fraction(1, 1000), sample_rate=48000, samples=1024)
        previous_end, gap = normalize_audio_timing(frame, previous_end)
        self.assertTrue(gap)
        self.assertEqual(frame.pts, 9600)
        frame = SimpleNamespace(pts=190, time_base=Fraction(1, 1000), sample_rate=48000, samples=1024)
        with self.assertRaisesRegex(RuntimeError, "Overlapping"):
            normalize_audio_timing(frame, previous_end)
        for pts, base, rate in ((None, Fraction(1, 1000), 48000),
                                (0, Fraction(0), 48000), (0, Fraction(1, 1000), 0)):
            with self.subTest(pts=pts, base=base, rate=rate):
                frame = SimpleNamespace(pts=pts, time_base=base, sample_rate=rate, samples=1024)
                with self.assertRaises(RuntimeError):
                    normalize_audio_timing(frame, None)

    def test_audio_player_decodes_all_required_formats_through_fake_device(self):
        for extension, video_codec, audio_codec in FORMATS:
            with self.subTest(extension=extension):
                path = self.source(extension, video_codec, audio_codec)
                info = probe(path)
                with patch("audio.sd.query_devices", return_value={
                    "max_output_channels": 2, "default_samplerate": 48000,
                }), patch("audio.sd.OutputStream", FakeOutputStream):
                    audio = AudioPlayer(str(path), info.audio_index, info.origin, threading.Event())
                    try:
                        audio.prepare()
                        self.assertTrue(audio.eof)
                        self.assertIsNone(audio.error)
                        self.assertGreaterEqual(audio.end_time, .29)
                        self.assertLess(audio.end_time, .36)
                        audio.start()
                        output = np.zeros((int(audio.end_time * audio.rate) + 100, 2), np.float32)
                        audio._callback(output, len(output), SimpleNamespace(outputBufferDacTime=100),
                                        SimpleNamespace(output_underflow=False))
                        self.assertIsNone(audio.error)
                        self.assertGreater(np.abs(output).max(), .01)
                    finally:
                        audio.close()

    def test_all_required_sources_export_mp4_aac_with_ordered_video_audio_pts(self):
        for extension, video_codec, audio_codec in FORMATS:
            with self.subTest(extension=extension):
                path = self.source(extension, video_codec, audio_codec)
                result = self.directory / f"export-{extension}.mp4"
                options = parser().parse_args([
                    "render", str(path), "--output", str(result), "--resolution", "128x64", "--columns", "16",
                ])
                render(options, progress=lambda *_: None)
                with av.open(str(result)) as output:
                    video = output.streams.video[0]
                    self.assertEqual((video.width, video.height, video.codec_context.name), (128, 64, "h264"))
                    frames = list(output.decode(video))
                    self.assertEqual([timestamp(frame) for frame in frames], [Fraction(i, 30) for i in range(len(frames))])
                    self.assertGreaterEqual(len(frames), 9)
                    self.assertLessEqual(len(frames), 11)
                with av.open(str(result)) as output:
                    audio = output.streams.audio[0]
                    self.assertEqual((audio.codec_context.name, audio.codec_context.sample_rate,
                                      audio.codec_context.layout.name), ("aac", 48000, "stereo"))
                    frames = list(output.decode(audio))
                    times = [timestamp(frame) for frame in frames]
                    self.assertEqual(times[0], 0)
                    self.assertEqual(times, sorted(times))
                    self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
                    self.assertLess(abs(float(times[-1]) + frames[-1].samples / 48000 - .3), .06)
                    self.assertGreater(max(np.abs(frame.to_ndarray()).max() for frame in frames), .01)

    def test_invalid_missing_audio_only_and_corrupt_inputs_return_one_without_traceback(self):
        audio_only = self.directory / "audio-only.mkv"
        fixture(audio_only, video=False)
        corrupt = self.directory / "corrupt.mp4"
        corrupt.write_bytes(b"this is not a media container")
        for path in (self.directory / "missing.mp4", audio_only, corrupt):
            for command in ("play", "render"):
                with self.subTest(path=path.name, command=command):
                    stderr = io.StringIO()
                    args = [command, str(path)]
                    if command == "play":
                        args += ["--no-audio"]
                    else:
                        args += ["--output", str(self.directory / "invalid-output.mp4")]
                    with patch("sys.stderr", stderr), patch("sys.stdout.isatty", return_value=True), \
                            patch.dict("os.environ", TERM="xterm"), patch("player.Terminal.size", return_value=(161, 46)), \
                            patch("audio.sd.query_devices", side_effect=AssertionError("hardware access forbidden")), \
                            patch("audio.sd.OutputStream", side_effect=AssertionError("hardware access forbidden")):
                        result = main(args)
                    self.assertEqual(result, 1)
                    self.assertIn("ascii-video:", stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

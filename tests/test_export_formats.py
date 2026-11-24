from fractions import Fraction
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from cli_options import EXPORT_CONTAINERS, export_format, parser
from export import render
from gui import Application, build_arguments
from test_export import fixture
import cli


class ExportFormatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "source.mp4"
        fixture(self.source, audio_offset=4800, audio_samples=12000)

    def options(self, container="mp4", codec=None, **kwargs):
        options = parser().parse_args(["render", str(self.source), "--output",
            str(self.directory / f"{codec or 'default'}.{container}"), "--container", container,
            "--resolution", "128x64", "--columns", "16", *(["--codec", codec] if codec else [])])
        for key, value in kwargs.items():
            setattr(options, key, value)
        return options

    def test_matrix_suffix_and_encoder_rejected_before_media_io(self):
        for container, codec, encoder, suffix in (
                ("webm", "hevc", "x264", "webm"), ("mp4", "vp9", "x264", "mp4"),
                ("webm", "vp9", "nvenc", "webm"), ("mp4", "av1", "x264", "webm")):
            options = self.options()
            options.container, options.codec, options.encoder = container, codec, encoder
            options.output = self.directory / f"result.{suffix}"
            with patch("export.probe") as probe, self.assertRaises(ValueError):
                render(options)
            probe.assert_not_called()
        for codec, resolution in (("av1", (32, 64)), ("hevc", (8, 16))):
            with self.assertRaises(ValueError):
                export_format("out.mp4", "mp4", codec, resolution=resolution)

    def test_cli_rejects_cross_options_before_importing_export(self):
        with patch.dict("sys.modules", export=None), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["render", "source", "--output", "out.webm",
                                       "--container", "webm", "--codec", "hevc"]), 2)
            self.assertEqual(cli.main(["render", "source", "--output", "out.webm"]), 2)

    def test_real_container_codec_audio_and_timestamps(self):
        for container, codecs in EXPORT_CONTAINERS.items():
            for codec in codecs:
                for no_audio in (False, True):
                    with self.subTest(container=container, codec=codec, no_audio=no_audio):
                        options = self.options(container, codec, no_audio=no_audio, overwrite=True)
                        render(options, progress=lambda *args: None)
                        with av.open(str(options.output)) as output:
                            video = output.streams.video[0]
                            self.assertEqual(video.codec_context.codec.id, av.Codec(codec, "r").id)
                            self.assertEqual(len(output.streams.audio), 0 if no_audio else 1)
                            if not no_audio:
                                self.assertEqual(output.streams.audio[0].codec_context.name,
                                                 "aac" if container == "mp4" else "opus")
                            frames = list(output.decode(video=0))
                        self.assertEqual(len(frames), 12)
                        times = [f.pts * f.time_base for f in frames]
                        self.assertEqual(times[0], 0)
                        tolerance = Fraction(1, 2000) if container == "webm" else 0
                        self.assertTrue(all(abs(t-Fraction(i, 30)) <= tolerance for i, t in enumerate(times)))
                        if not no_audio:
                            with av.open(str(options.output)) as output:
                                decoded = list(output.decode(audio=0))
                            self.assertTrue(decoded)
                            self.assertEqual(decoded[0].sample_rate, 48000)
                            if container == "webm":
                                self.assertLessEqual(abs(decoded[0].pts * decoded[0].time_base), Fraction(1, 1000))
                                # Leading silence is explicit, not a timestamp shift of the video.
                                samples = np.concatenate([f.to_ndarray() for f in decoded], axis=-1)
                                self.assertLess(float(np.abs(samples[..., :3500]).max()), .005)
                                self.assertGreater(float(np.abs(samples[..., 6000:]).max()), .05)

    def test_gui_uses_matrix_and_preview_ignores_export_codec(self):
        for container, codecs in EXPORT_CONTAINERS.items():
            for codec in codecs:
                values = dict(input=str(self.source), output=str(self.directory / f"out.{container}"),
                              container=container, codec=codec)
                _, options = build_arguments("render", values)
                self.assertEqual((options.container, options.codec), (container, codec))
                _, preview = build_arguments("preview", {**values, "encoder": "nvenc"})
                self.assertEqual((preview.container, preview.codec, preview.encoder), ("mp4", "h264", "x264"))
        values["codec"] = "hevc"
        with self.assertRaises(ValueError):
            build_arguments("render", {**values, "container": "webm"})

    def test_cancel_and_failure_preserve_webm_and_remove_temporary(self):
        import threading
        options = self.options("webm", "vp9", overwrite=True, no_audio=True)
        options.output.write_bytes(b"existing")
        stop = threading.Event()
        def encode(video, frame, encoder):
            stop.set()
            return video.encode(frame)
        with patch("export._encode_video", side_effect=encode), self.assertRaises(InterruptedError):
            render(options, stop_event=stop)
        self.assertEqual(options.output.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.webm")), [])
        with patch("export._encode_video", side_effect=RuntimeError("failed")), self.assertRaises(RuntimeError):
            render(options)
        self.assertEqual(options.output.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.webm")), [])

    def test_gui_container_change_preserves_valid_codec_and_updates_suffix(self):
        app = Application.__new__(Application)
        app.codec_box, app.encoder_box = Mock(), Mock()
        app.values = {}
        for key, value in dict(container="webm", codec="hevc", encoder="nvenc", output="result.mp4").items():
            variable = Mock()
            variable.get.return_value = value
            variable.set.side_effect = lambda value, variable=variable: setattr(variable.get, "return_value", value)
            app.values[key] = variable
        app.update_export_container()
        self.assertEqual(app.values["codec"].get(), "vp9")
        self.assertEqual(app.values["encoder"].get(), "x264")
        self.assertEqual(app.values["output"].get(), "result.webm")
        app.encoder_box.configure.assert_called_with(state="disabled")
        app.values["codec"].set("av1")
        app.values["container"].set("mp4")
        app.update_export_container()
        self.assertEqual(app.values["codec"].get(), "av1")
        self.assertEqual(app.values["output"].get(), "result.mp4")
        app.values["codec"].set("h264")
        app.update_export_codec()
        app.encoder_box.configure.assert_called_with(state="readonly")

    def test_missing_encoder_preserves_target_before_preparation(self):
        options = self.options("webm", "vp9", overwrite=True)
        options.output.write_bytes(b"existing")
        with patch("export.av.Codec", side_effect=av.codec.codec.UnknownCodecError("missing")), \
                patch("export.Renderer") as renderer, self.assertRaisesRegex(RuntimeError, "unavailable"):
            render(options)
        renderer.assert_not_called()
        self.assertEqual(options.output.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.webm")), [])

    def test_webm_late_video_keeps_black_lead_and_audio_origin(self):
        fixture(self.source, video_pts=list(range(6,18)), audio_offset=0, audio_samples=24000)
        for codec in ("vp9", "av1"):
            options = self.options("webm", codec)
            render(options, progress=lambda *args: None)
            with av.open(str(options.output)) as output:
                frames = list(output.decode(video=0))
            self.assertEqual(len(frames), 18)
            self.assertEqual(frames[0].pts * frames[0].time_base, 0)
            self.assertTrue(all(f.to_ndarray(format="rgb24").max() <= 1 for f in frames[:6]))
            self.assertGreater(frames[6].to_ndarray(format="rgb24").max(), 30)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cli


class CliTests(unittest.TestCase):
    def test_unavailable_audio_device_reports_a_playback_error(self):
        from sounddevice import PortAudioError

        def unavailable(options):
            raise PortAudioError("Device unavailable")

        code, _, error = self.run_main(["play", "movie.mkv"], player=unavailable)
        self.assertEqual(code, 1)
        self.assertIn("Device unavailable", error)
        self.assertIn("--no-audio", error)

    def test_all_dithering_algorithms_available_in_both_commands(self):
        from dithering import DITHER_MODES
        from diffusion import MODES
        for command in (["play", "x"], ["render", "x", "--output", "out.mp4"]):
            for mode in (*DITHER_MODES, *MODES):
                self.assertEqual(cli.parser().parse_args([*command, "--dither", mode]).dither, mode)

    def test_quality_selection_is_export_only(self):
        received = []
        with patch.dict(sys.modules, {"export": types.SimpleNamespace(render=lambda options, **kwargs: received.append(options))}):
            for quality in ("compact", "balanced", "high"):
                self.assertEqual(self.run_main(["render", "x", "--output", "a.mp4", "--quality", quality])[0], 0)
                self.assertEqual(received[-1].quality, quality)
        self.assertEqual(self.run_main(["render", "x", "--output", "a.mp4", "--quality", "bad"])[0], 2)
        self.assertEqual(self.run_main(["play", "x", "--quality", "compact"])[0], 2)

    def test_named_palettes_for_play_and_render(self):
        for command in (["play", "x"], ["render", "x", "--output", "a.mp4"]):
            for name, ramp in cli.PALETTES.items():
                options = cli.parser().parse_args([*command, "--palette", name])
                self.assertEqual(options.palette, ramp)
                self.assertEqual(len(set(ramp)), len(ramp))
                self.assertTrue(ramp.startswith(" "))
                self.assertTrue(all(32 <= ord(c) <= 126 for c in ramp))
        self.assertEqual(cli.palette(" .#"), " .#")

    def run_main(self, arguments, player=None):
        module = types.SimpleNamespace(play=player or (lambda options: None))
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, player=module), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = cli.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def test_defaults_and_all_options(self):
        options = cli.parser().parse_args(["play", "кино с пробелами.mp4"])
        self.assertEqual(options.input, Path("кино с пробелами.mp4"))
        self.assertEqual((options.width, options.height, options.fps), (None, None, 30))
        self.assertEqual((options.gamma, options.contrast, options.dither), (1.0, 1.0, "none"))
        self.assertEqual(options.brightness, 0.0)
        self.assertEqual((options.color_saturation, options.glyph_correction), (0.0, 0.0))

        self.assertFalse(options.show_resolution)
        self.assertTrue(cli.parser().parse_args(["play", "x", "--show-resolution"]).show_resolution)
        for command in (["play", "x"], ["render", "x", "--output", "a.mp4"]):
            for brightness in (-1, -0.3, 0, 0.5, 1):
                self.assertEqual(cli.parser().parse_args([
                    *command, "--brightness", str(brightness)]).brightness, brightness)
        options = cli.parser().parse_args(["play", "x", "--width", "200", "--height", "56", "--cell-aspect", "0.7",
                                          "--color", "#abcdef", "--palette", " .#", "--gamma", "2", "--contrast", "1.2",
                                          "--dither", "bayer", "--fps", "60", "--no-audio"])
        self.assertTrue(options.no_audio)
        self.assertEqual((options.width, options.height, options.color), (200, 56, "#abcdef"))

    def test_color_controls_bounds_apply_to_both_commands(self):
        for command in (["play", "x"], ["render", "x", "--output", "x.mp4"]):
            for flag, name, accepted, rejected in (
                    ("--color-saturation", "color_saturation", (-1, -.5, 0, .5, 1), (-1.01, 1.01, "nan", "inf")),
                    ("--glyph-correction", "glyph_correction", (0, .5, 1), (-.01, 1.01, "nan", "inf"))):
                for value in accepted:
                    self.assertEqual(getattr(cli.parser().parse_args([*command, flag, str(value)]), name), value)
                for value in rejected:
                    with self.subTest(command=command[0], flag=flag, value=value):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                            cli.parser().parse_args([*command, flag, str(value)])
                        self.assertEqual(error.exception.code, 2)

    def test_invalid_options_return_two(self):
        cases = [("--width", "0"), ("--height", "1001"), ("--fps", "61"), ("--gamma", "nan"),
                 ("--contrast", "inf"), ("--cell-aspect", "-inf"), ("--gamma", "5.1"), ("--contrast", "0"),
                 ("--brightness", "nan"), ("--brightness", "inf"), ("--brightness", "-1.01"), ("--brightness", "1.01"),
                 ("--color", "white"), ("--palette", "aa"), ("--palette", "░#"), ("--palette", "\t#"),
                 ("--dither", "random"), ("--wid", "100")]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value):
                code, _, error = self.run_main(["play", "x", f"{flag}={value}"])
                self.assertEqual(code, 2)
                self.assertNotIn("Traceback", error)
        self.assertEqual(self.run_main(["render", "x"])[0], 2)

    def test_help_and_version_do_not_import_player(self):
        original = __import__
        def guard(name, *args, **kwargs):
            if name in ("player", "export", "raster", "gui", "PIL", "av", "sounddevice"):
                raise AssertionError(f"unexpected import: {name}")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            self.assertEqual(self.run_main(["--help"])[0], 0)
            self.assertIn("truecolor", self.run_main(["play", "--help"])[1])
            self.assertEqual(self.run_main(["--version"])[1].strip(), "ascii-video 0.3.1")
            self.assertIn("resolution", self.run_main(["render", "--help"])[1])
            self.assertEqual(self.run_main(["gui", "--help"])[0], 0)

    def test_gui_dispatch_explicit_and_double_click(self):
        calls = []
        with patch.dict(sys.modules, {"gui": types.SimpleNamespace(launch=lambda: calls.append("gui"))}):
            self.assertEqual(self.run_main(["gui"])[0], 0)
            self.assertEqual(self.run_main([])[0], 0)
        self.assertEqual(calls, ["gui", "gui"])

    def test_render_resolution_and_dispatch_without_playback(self):
        received = []
        with patch.dict(sys.modules, {"export": types.SimpleNamespace(render=lambda options, **kwargs: received.append(options))}):
            code, _, _ = self.run_main([
                "render", "x.mp4", "--output", "ascii.mp4", "--resolution", "1920x1080",
                "--color", "source", "--columns", "200", "--overwrite"
            ], lambda options: self.fail("export must not play audio or enter the terminal"))
        self.assertEqual(code, 0)
        self.assertEqual(received[0].resolution, (1920, 1080))
        self.assertEqual(received[0].columns, 200)
        self.assertTrue(received[0].overwrite)
        self.assertEqual(received[0].encoder, "x264")
        self.assertEqual(cli.parser().parse_args([
            "render", "x", "--output", "a.mp4"
        ]).resolution, (1920, 1080))
        self.assertEqual(cli.parser().parse_args([
            "render", "x", "--output", "a.mp4"
        ]).columns, 320)
        for size in ("0x720", "1281x720", "1280x719", "9000x720", "1280*720", "nanx720"):
            with self.subTest(size=size):
                self.assertEqual(self.run_main([
                    "render", "x", "--output", "a.mp4", "--resolution", size
                ])[0], 2)

    def test_encoder_selection_is_export_only(self):
        received = []
        with patch.dict(sys.modules, {"export": types.SimpleNamespace(render=lambda options, **kwargs: received.append(options))}):
            self.assertEqual(self.run_main(["render", "x", "--output", "a.mp4", "--encoder", "nvenc"])[0], 0)
        self.assertEqual(received[0].encoder, "nvenc")
        self.assertEqual(self.run_main(["render", "x", "--output", "a.mp4", "--encoder", "unknown"])[0], 2)
        self.assertEqual(self.run_main(["play", "x", "--encoder", "nvenc"])[0], 2)

    def test_exit_codes(self):
        for exception, expected in [(ValueError("bad media"), 1), (RuntimeError("bad audio"), 1), (KeyboardInterrupt(), 130)]:
            def play(options):
                raise exception
            code, _, error = self.run_main(["play", "x"], play)
            self.assertEqual(code, expected)
            self.assertNotIn("Traceback", error)
        self.assertEqual(self.run_main(["play", "x"])[0], 0)

    def test_temporary_font_scale_default_and_range(self):
        self.assertEqual(cli.parser().parse_args(["play", "x"]).font_scale, .8)
        self.assertEqual(cli.parser().parse_args(["play", "x", "--font-scale", "1"]).font_scale, 1)
        for scale in ("0.24", "2.01", "nan", "inf"):
            self.assertEqual(self.run_main(["play", "x", "--font-scale", scale])[0], 2)

    def test_dropped_path_relaunch_keeps_scale_and_normal_audio(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "перенос с пробелами.mp4"
            path.touch()
            with patch("console_font.relaunch_classic", return_value=0) as relaunch:
                code, _, _ = self.run_main([str(path), "--font-scale", "0.8"],
                                           lambda options: self.fail("parent must not play"))
            self.assertEqual(code, 0)
            relaunch.assert_called_once_with(["play", str(path), "--font-scale", "0.8"], .8)

    def test_direct_video_path_and_explorer_arguments(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "кино с пробелами.mp4"
            path.touch()
            received = []
            self.assertEqual(self.run_main([str(path), "--no-audio"], received.append)[0], 0)
            self.assertEqual(received[0].command, "play")
            self.assertEqual(received[0].input, path)
            self.assertTrue(received[0].no_audio)
            self.assertIsNone(received[0].width)
            # Explorer supplies the dropped file as argv[1], without the play command.
            with patch.object(sys, "argv", ["ascii-video.exe", str(path)]):
                self.assertEqual(self.run_main(None, received.append)[0], 0)
            self.assertFalse(received[-1].no_audio)
            self.assertEqual(self.run_main([str(path), str(path)], received.append)[0], 2)
            self.assertEqual(self.run_main([str(Path(directory))], received.append)[0], 2)


if __name__ == "__main__":
    unittest.main()

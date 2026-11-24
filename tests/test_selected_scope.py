"""Keep deferred effects outside the current application contract."""

import contextlib
import io
import unittest

from cli_options import DEFAULTS, PALETTES, parser


class SelectedScopeTests(unittest.TestCase):
    def test_release_defaults_and_deferred_options_stay_out_of_both_commands(self):
        self.assertEqual(set(PALETTES), {"basic", "medium", "extended"})
        self.assertEqual(DEFAULTS["columns"], 320)
        self.assertEqual(DEFAULTS["brightness"], 0)
        self.assertEqual(DEFAULTS["dither"], "none")
        deferred = {
            "--saturation": "0", "--tone-mode": "standard",
            "--shadows": "0", "--highlights": "0", "--denoise": "0",
            "--detail": "0", "--stabilization-strength": "0.5",
            "--color-limit": "16", "--color-transition": "balanced2",
            "--background": "#FFFFFF", "--dither-strength": "1",
            "--stabilize": None, "--adaptive-colors": None,
            "--adaptive-glyphs": None, "--invert": None,
        }
        for command in (["play", "x"], ["render", "x", "--output", "x.mp4"]):
            options = parser().parse_args(command)
            self.assertEqual((options.color_saturation, options.glyph_correction), (0.0, 0.0))
            self.assertFalse(set(vars(options)) & {
                "saturation", "tone_mode", "stabilize", "adaptive_colors", "dither_strength"})
            for flag, value in deferred.items():
                with self.subTest(command=command[0], flag=flag):
                    arguments = [*command, flag, *([] if value is None else [value])]
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        parser().parse_args(arguments)
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

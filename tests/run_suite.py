"""Run the test suite with physical audio disabled before importing tests."""

import argparse
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class PortAudioError(Exception):
    """Audio error type used by the CLI tests."""


def forbid_audio(*args, **kwargs):
    raise AssertionError("Physical audio is forbidden in tests; use a fake device")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-python", action="store_true")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sounddevice = ModuleType("sounddevice")
    sounddevice.PortAudioError = PortAudioError
    names = ("query_devices", "Stream", "InputStream", "OutputStream", "RawStream", "RawInputStream",
             "RawOutputStream", "play", "rec", "playrec")
    for name in names:
        setattr(sounddevice, name, forbid_audio)
    with patch.dict(sys.modules, sounddevice=sounddevice):
        if options.pure_python:
            import raster
            raster._compose = None
            raster._compose_planes = None
            raster._compose_planes_sparse = None
            raster._make_source_tinted_atlas = None
            raster._compose_source_tinted = None
            raster._resize_source_band = None
            raster._resize_source_band_into = None
            import diffusion
            diffusion._native_quantize = None
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=options.pattern)
        if suite.countTestCases() == 0:
            parser.error(f"No tests matched pattern: {options.pattern}")
        result = unittest.TextTestRunner(verbosity=2 if options.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import dithering
import raster
import resources


class ResourceTests(unittest.TestCase):
    def test_supported_installation_layouts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            for layout, module, assets in (
                ("checkout", root / "checkout/src", root / "checkout/assets"),
                ("frozen", root / "frozen", root / "frozen/assets"),
                ("venv", root / "venv/Lib/site-packages", root / "venv/share/ascii-video"),
                ("user", root / "user/Lib/site-packages", root / "user/share/ascii-video"),
                ("target", root / "target", root / "target/share/ascii-video"),
            ):
                with self.subTest(layout=layout):
                    assets.mkdir(parents=True)
                    expected = assets / "fixture.dat"
                    expected.touch()
                    with patch.object(resources, "__file__", str(module / "resources.py")), \
                            patch.object(resources.sys, "prefix", str(root / "venv")), \
                            patch.object(resources.site, "getuserbase", return_value=str(root / "user")):
                        self.assertEqual(resources.resource_path("fixture.dat"), expected)
                    expected.unlink()

    def test_user_install_prefers_its_resources_to_global_install(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            for prefix in ("global", "user"):
                assets = root / prefix / "share/ascii-video"
                assets.mkdir(parents=True)
                (assets / "fixture.dat").touch()
            with patch.object(resources, "__file__", str(root / "user/Lib/site-packages/resources.py")), \
                    patch.object(resources.sys, "prefix", str(root / "global")), \
                    patch.object(resources.site, "getuserbase", return_value=str(root / "user")):
                self.assertEqual(resources.resource_path("fixture.dat"), root / "user/share/ascii-video/fixture.dat")

    def test_missing_resources_preserve_specific_errors(self):
        with patch.object(raster, "resource_path", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Bundled DejaVu Sans Mono font is missing"):
                raster.font_path()
        dithering._blue_noise.cache_clear()
        try:
            with patch.object(dithering, "resource_path", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Bundled blue-noise rank array is missing"):
                    dithering._blue_noise()
        finally:
            dithering._blue_noise.cache_clear()


if __name__ == "__main__":
    unittest.main()

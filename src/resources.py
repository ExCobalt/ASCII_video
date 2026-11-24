"""Find bundled assets in source, frozen and installed layouts."""

from pathlib import Path
import site
import sys


def resource_path(name):
    directory = Path(__file__).resolve().parent
    prefixes = (Path(sys.prefix), Path(site.getuserbase()))
    if directory.is_relative_to(prefixes[1]):
        prefixes = prefixes[::-1]
    roots = (directory / "assets", directory.parent / "assets",
             directory / "share" / "ascii-video") + tuple(
                 prefix / "share" / "ascii-video" for prefix in prefixes)
    return next((root / name for root in roots if (root / name).is_file()), None)

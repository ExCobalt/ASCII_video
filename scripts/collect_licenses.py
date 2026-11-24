"""Collect Windows dependency notices; this does not certify binary compliance."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "av", "numpy", "pillow", "sounddevice", "darkdetect", "ttkbootstrap",
    "cffi", "pycparser", "packaging", "pyinstaller", "setuptools",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory_path(directory, name):
    """Keep manifest paths inside the cache or the installed runtime."""
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe native notice path: {name}")
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"Unsafe native notice path: {name}")
    return path


def check_native_inventory(inventory):
    for name, expected in inventory["packages"].items():
        if metadata.version(name) != expected:
            raise ValueError(f"Native notice inventory needs review for the installed {name} version")
    for entry in inventory["binaries"]:
        directory = (Path(sys.base_prefix) if entry["distribution"] == "python" else
                     Path(metadata.distribution(entry["distribution"]).locate_file("")))
        binary = inventory_path(directory, entry["path"])
        if not binary.is_file() or sha256(binary) != entry["sha256"]:
            raise ValueError(f"Native binary inventory mismatch: {entry['path']}; review scripts/native_notices.json for this runtime")


def verified_notice(entry, cache):
    path = inventory_path(cache, entry["filename"])
    if not path.is_file():
        line_range = f" (notice lines {entry['lines'][0]}-{entry['lines'][1]})" if "lines" in entry else ""
        if not entry.get("url"):
            source = entry["source"]
            raise RuntimeError(f"Missing cached notice {path}. Restore {source['member']}{line_range} from {source['url']} (archive SHA256 {source['sha256']})")
        try:
            with urllib.request.urlopen(entry["url"], timeout=30) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            source = entry["source"]
            raise RuntimeError(f"Cannot fetch native notice {entry['filename']}. Restore {path} from {source['member']}{line_range} in {source['url']} (archive SHA256 {source['sha256']}); keep the pinned notice hash unchanged") from error
        if entry.get("download_sha256") and hashlib.sha256(data).hexdigest() != entry["download_sha256"]:
            raise ValueError(f"Native notice source checksum mismatch: {entry['filename']}")
        if "lines" in entry:
            start, end = entry["lines"]
            data = b"".join(data.splitlines(keepends=True)[start - 1:end])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Native notice checksum mismatch: {entry['filename']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Native notice checksum mismatch: {entry['filename']}")
    return path


def collect(destination):
    if sys.platform != "win32":
        raise RuntimeError("This collector includes Windows runtime notices; Linux/macOS require their own binary and source inventory")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    previous_files = (json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
                      if manifest_path.is_file() else {})
    manifest = {"platform": "win32",
                "scope": "Windows notices only; corresponding source must be supplied separately",
                "packages": {}, "files": {}}

    def copy(source, relative):
        target = inventory_path(destination, Path(relative).as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["files"][Path(relative).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()

    for name in PACKAGES:
        distribution = metadata.distribution(name)
        notices = []
        for file in distribution.files or ():
            # Include entire license directories (AUTHORS/PATENTS included), but
            # never confuse packaging/licenses Python modules with legal texts.
            parts = Path(file).parts
            if any(part.lower() == "licenses" for part in parts[:-1]) or Path(file).name.lower().startswith(("license", "copying", "notice", "authors", "patents")):
                if Path(file).suffix.lower() in (".py", ".pyc", ".pyd"):
                    continue
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    copy(source, Path(name) / file)
                    notices.append(str(file))
        if not notices:
            raise RuntimeError(f"No installed license notices found for {name}")
        manifest["packages"][name] = {"version": distribution.version, "notices": notices}

    copy(ROOT / "LICENSE", "LICENSE")
    copy(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    copy(ROOT / "DISTRIBUTION.md", "DISTRIBUTION.md")
    copy(Path(sys.base_prefix) / "LICENSE.txt", "Python-LICENSE.txt")
    manifest["python"] = sys.version
    # Wheel metadata does not cover all DLLs bundled by PyAV or the Python runtime.
    inventory_file = ROOT / "scripts/native_notices.json"
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    check_native_inventory(inventory)
    for entry in inventory["files"]:
        source = verified_notice(entry, ROOT / "build/license-sources/native-notices")
        copy(source, Path("native") / entry["filename"])
        if entry["filename"] == "ffmpeg/COPYING.GPLv3":
            copy(source, "COPYING.GPLv3")
    copy(inventory_file, "native-notices.json")
    for name in previous_files.keys() - manifest["files"].keys():
        inventory_path(destination, name)
        (destination / name).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/release-licenses")
    args = parser.parse_args()
    result = collect(args.output)
    print(f"Collected {len(result['files'])} notices in {args.output}; source compliance remains a separate check.")

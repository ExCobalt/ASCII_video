"""Record and package a reviewed Linux or macOS PyInstaller bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import tarfile

from package_windows import ROOT, check_sources, sha256, stage_release


NATIVE_HEADERS = {b"\x7fELF", *[bytes.fromhex(magic) for magic in (
    "feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca", "cafebabf", "bfbafeca")]}


def inventory(bundle):
    if sys.platform not in ("linux", "darwin"):
        raise ValueError("Inspect the bundle on its target operating system")
    bundle = bundle.resolve(strict=True)
    files = {}
    for path in sorted(bundle.rglob("*")):
        mode = path.lstat().st_mode
        entry = {"mode": stat.S_IMODE(mode)}
        if path.is_symlink():
            target = Path(os.readlink(path))
            if target.is_absolute() or not path.resolve(strict=True).is_relative_to(bundle):
                raise ValueError(f"Non-portable bundle link: {path}")
            entry["link"] = str(target)
        elif path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) in NATIVE_HEADERS:
                    entry["native"] = True
                stream.seek(0)
                entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        elif not path.is_dir():
            raise ValueError(f"Unsupported bundle entry: {path}")
        files[path.relative_to(bundle).as_posix()] = entry
    if not files:
        raise ValueError("Empty application bundle")
    return {"platform": sys.platform, "architecture": platform.machine(),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"), "files": files}


def check_components(review, current):
    native = {path for path, entry in current["files"].items() if entry.get("native")}
    covered = set()
    for name, group in review.get("components", {}).items():
        binaries = set(group["binaries"])
        if (not binaries or not group["sources"] or not group["notices"] or binaries & covered
                or not set(group["sources"]).issubset(review["files"])
                or not set(group["notices"]).issubset(review["license_files"])):
            raise ValueError(f"Incomplete or overlapping component review: {name}")
        covered.update(binaries)
    if covered != native or not native:
        raise ValueError(f"Native component review differs from bundle: {sorted(native ^ covered)}")


def pack(bundle, sources, output, name):
    review, files = check_sources(sources)
    current = inventory(bundle)
    manifest = sources / "bundle-inventory.json"
    if (manifest.name not in review["files"] or
            json.loads(manifest.read_text(encoding="utf-8")) != current):
        raise ValueError("Source review does not describe this bundle, architecture and dependency lock")
    notices = review.get("license_files", [])
    required = {"LICENSE", "COPYING.GPLv3", "DISTRIBUTION.md", "THIRD_PARTY_NOTICES.md"}
    if not required.issubset(notices) or not set(notices).issubset(review["files"]):
        raise ValueError("Missing reviewed license documents")
    check_components(review, current)
    for filename in ("LICENSE", "DISTRIBUTION.md", "THIRD_PARTY_NOTICES.md"):
        if review["files"][filename] != sha256(ROOT / filename):
            raise ValueError(f"Distribution terms differ from current {filename}")
    pins = json.loads((ROOT / "scripts/native_notices.json").read_text(encoding="utf-8"))
    gpl = next(entry["sha256"] for entry in pins["files"] if entry["filename"] == "ffmpeg/COPYING.GPLv3")
    if review["files"]["COPYING.GPLv3"] != gpl:
        raise ValueError("GPLv3 license checksum mismatch")

    with stage_release(output, name) as stage:
        source_tar = stage / f"{name}-sources.tar.gz"
        with tarfile.open(source_tar, "x:gz") as archive:
            for path, filename in files:
                archive.add(path, arcname=f"sources/{filename}", recursive=False)
            archive.add(sources / "release-review.json", arcname="sources/release-review.json")
        binary_tar = stage / f"{name}.tar.gz"
        with tarfile.open(binary_tar, "x:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
            for path, filename in files:
                if filename in notices:
                    archive.add(path, arcname=f"licenses/{filename}", recursive=False)
            readme = stage / "README.txt"
            if current["platform"] == "darwin":
                start = f'Open "{bundle.name}" in Finder to launch the graphical interface.'
                command = f'./"{bundle.name}"/Contents/MacOS/ASCII-Video'
            else:
                command = f'./"{bundle.name}"/ascii-video'
                start = f"Keep the extracted folder together. Run {command} to open the graphical interface."
            readme.write_text(f"{start}\nFrom a terminal in the extraction directory:\n"
                              f'  {command} play "movie.mp4"\n'
                              f'  {command} render "movie.mp4" --output "movie_ascii.mp4"\n'
                              f"  {command} --help\n\n"
                              "Use media from trusted sources. Malformed files can cause the bundled FFmpeg decoder to hang or crash.\n\n"
                              f"Corresponding source: {source_tar.name}\nSHA-256: {sha256(source_tar)}\n"
                              "See licenses/DISTRIBUTION.md for terms.\n", encoding="utf-8")
            archive.add(readme, arcname=readme.name)
            readme.unlink()
        (stage / f"{name}-SHA256SUMS.txt").write_text(
            "".join(f"{sha256(p)}  {p.name}\n" for p in (binary_tar, source_tar)), encoding="ascii")
    print(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inventory")
    inspect.add_argument("bundle", type=Path)
    inspect.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("package")
    release.add_argument("bundle", type=Path)
    release.add_argument("--sources", type=Path, required=True)
    release.add_argument("--name", required=True)
    release.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if args.command == "inventory":
        data = inventory(args.bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        pack(args.bundle, args.sources, args.output, args.name)


if __name__ == "__main__":
    main()

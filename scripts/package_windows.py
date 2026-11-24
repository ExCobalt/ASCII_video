"""Package a reviewed Windows binary and its corresponding-source companion."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ROOT_NOTICES = ("DISTRIBUTION.md", "COPYING.GPLv3", "LICENSE", "THIRD_PARTY_NOTICES.md")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def stage_release(output, name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Release name must be a filename")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"Release already exists: {destination}")
    # Expose all three finished files in one rename, even after a crash or retry.
    with tempfile.TemporaryDirectory(dir=output, prefix=".release-") as work:
        stage = Path(work) / name
        stage.mkdir()
        yield stage
        stage.rename(destination)


def check_sources(directory):
    """Reject incomplete reviews, missing files and changed source archives."""
    directory = directory.resolve()
    report = json.loads((directory / "release-review.json").read_text(encoding="utf-8"))
    if report.get("blockers") or report.get("status") != "ready":
        raise ValueError("Corresponding-source review is incomplete: " + "; ".join(report.get("blockers", [])))
    if not report.get("files"):
        raise ValueError("Source review contains no files")
    result = []
    for filename, expected in report["files"].items():
        path = (directory / filename).resolve()
        relative = PurePosixPath(filename)
        if (relative.as_posix() != filename or ".." in relative.parts or ":" in filename or "\\" in filename
                or relative.is_absolute() or not path.is_relative_to(directory) or not path.is_file()):
            raise ValueError(f"Missing or unsafe source path: {filename}")
        if sha256(path) != expected:
            raise ValueError(f"Source checksum mismatch: {filename}")
        result.append((path, filename))
    return report, result


def check_distribution_terms(notices):
    for filename in ("DISTRIBUTION.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (notices / filename).read_bytes() != (ROOT / filename).read_bytes():
            raise ValueError(f"Distribution terms differ from current {filename}; rebuild and review the executable")
    inventory = json.loads((ROOT / "scripts/native_notices.json").read_text(encoding="utf-8"))
    expected = next(entry["sha256"] for entry in inventory["files"]
                    if entry["filename"] == "ffmpeg/COPYING.GPLv3")
    for filename in ("COPYING.GPLv3", "native/ffmpeg/COPYING.GPLv3"):
        if sha256(notices / filename) != expected:
            raise ValueError(f"Combined-program GPL text differs from the pinned GPLv3 notice: {filename}")


def check_executable(executable, notices, native_evidence=None):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(executable))
    embedded = {name.replace("\\", "/"): name for name in archive.toc}
    manifest = json.loads((notices / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("manifest.json", *manifest["files"]):
        key = embedded.get("licenses/" + filename)
        if key is None or archive.extract(key) != (notices / filename).read_bytes():
            raise ValueError(f"Executable distribution terms differ or are missing: {filename}; rebuild and review it")

    inventory = json.loads((ROOT / "scripts/native_notices.json").read_text(encoding="utf-8"))
    expected = {Path(entry["path"]).name.lower(): entry["sha256"] for entry in inventory["binaries"]}
    seen = set()
    observed = {}
    for filename, key in embedded.items():
        name = Path(filename).name.lower()
        if not name.endswith((".dll", ".pyd")):
            continue
        digest = hashlib.sha256(archive.extract(key)).hexdigest()
        observed[filename] = digest
        if not name.endswith(".dll") and name not in expected:
            continue
        if name in seen or digest != expected.get(name):
            raise ValueError(f"Duplicate, unreviewed or changed embedded library: {filename}")
        seen.add(name)
    if expected.keys() != seen:
        raise ValueError(f"Missing embedded libraries: {sorted(expected.keys() - seen)}")
    if native_evidence is not None:
        reviewed = {path: item["sha256"] for path, item in native_evidence.items() if item.get("matched")}
        if observed != reviewed:
            raise ValueError("Embedded DLLs and Python extensions differ from the corresponding-source review")


def package(executable, sources, notices, name, output):
    review, files = check_sources(sources)
    if sha256(executable) != review.get("executable_sha256"):
        raise ValueError("Source review does not describe this executable")
    notice_manifest = json.loads((notices / "manifest.json").read_text(encoding="utf-8"))
    required = {*ROOT_NOTICES, "native/ffmpeg/COPYING.GPLv3"}
    if not required.issubset(notice_manifest.get("files", {})):
        raise ValueError("Required distribution notices are missing")
    for filename, expected in notice_manifest["files"].items():
        path = (notices / filename).resolve()
        if not path.is_relative_to(notices.resolve()) or sha256(path) != expected:
            raise ValueError(f"License checksum mismatch: {filename}")
    check_distribution_terms(notices)
    check_review_context(review, notices / "manifest.json", ROOT / "uv.lock")
    evidence_name = "embedded-native-evidence.json"
    if evidence_name not in review["files"]:
        raise ValueError("Missing corresponding-source review of embedded DLLs and Python extensions")
    evidence = json.loads((sources / evidence_name).read_text(encoding="utf-8"))
    check_executable(executable, notices, evidence)
    with stage_release(output, name) as stage:
        write_archives(executable, sources, files, notices, notice_manifest, name, stage)
    print(output / name)


def write_archives(executable, sources, files, notices, notice_manifest, name, output):
    source_zip = output / f"{name}-sources.zip"
    binary_zip = output / f"{name}.zip"
    with ZipFile(source_zip, "x", ZIP_DEFLATED) as archive:
        for path, filename in files:
            archive.write(path, f"{name}-sources/{filename}")
        archive.write(sources / "release-review.json", f"{name}-sources/release-review.json")
    with ZipFile(binary_zip, "x", ZIP_DEFLATED) as archive:
        archive.write(executable, f"{name}/ascii-video.exe")
        for filename in ROOT_NOTICES:
            archive.write(notices / filename, f"{name}/{filename}")
        for filename in notice_manifest["files"]:
            archive.write(notices / filename, f"{name}/licenses/{filename}")
        archive.write(notices / "manifest.json", f"{name}/licenses/manifest.json")
        archive.writestr(f"{name}/README.txt", (
            "Extract the archive and double-click ascii-video.exe to open the graphical interface.\n"
            "To use PowerShell, open it in the extracted folder. Examples:\n"
            '  .\\ascii-video.exe play "movie.mp4"\n'
            '  .\\ascii-video.exe render "movie.mp4" --output "movie_ascii.mp4"\n'
            "Run .\\ascii-video.exe --help for all commands.\n\n"
            "Use media from trusted sources. Malformed files can cause the bundled FFmpeg decoder to hang or crash.\n\n"
            "ASCII Video combined Windows program: GNU General Public License version 3 (GPL-3.0-only).\n"
            "See DISTRIBUTION.md and COPYING.GPLv3 for the combined program's terms.\n"
            "Original source files: MIT License (see LICENSE).\n"
            "Third-party components and independent resources retain their own terms; see THIRD_PARTY_NOTICES.md and licenses/.\n"
            f"Corresponding source: {source_zip.name}\n"
            f"Source archive SHA256: {sha256(source_zip)}\n"
            "Publish both archives together with equal access to the sources.\n"
        ))
    for path in (source_zip, binary_zip):
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt archive: {path}")
    (output / f"{name}-SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in (binary_zip, source_zip)), encoding="ascii")


def check_review_context(review, notice_manifest, lockfile):
    if review.get("uv_lock_sha256") != sha256(lockfile):
        raise ValueError("Dependency lock changed or was not reviewed")
    if review.get("license_manifest_sha256") != sha256(notice_manifest):
        raise ValueError("License manifest changed or was not reviewed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "dist/ascii-video.exe")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    parser.add_argument("--sources", type=Path, default=ROOT / "build/license-sources" / f"windows-x64-{version}")
    parser.add_argument("--notices", type=Path, default=ROOT / "build/release-licenses")
    parser.add_argument("--name")
    parser.add_argument("--check-executable", action="store_true", help="Verify embedded libraries and notices without packaging")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if args.check_executable:
        check_executable(args.executable, args.notices)
    elif args.name:
        package(args.executable, args.sources, args.notices, args.name, args.output)
    else:
        parser.error("provide --name or --check-executable")

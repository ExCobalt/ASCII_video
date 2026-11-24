# Building and publishing ASCII Video

Run commands from the repository root. Build desktop bundles on their target OS and architecture; the scripts do not cross-compile. A successful build establishes neither clean-system compatibility nor completion of the third-party source review.

## Build environment

Use Python 3.12 with Tk and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```console
uv sync --locked --python 3.12 --group build
```

`uv.lock` pins runtime and build-group packages. Isolated Python builds use the setuptools and Cython versions pinned in `pyproject.toml`. Keep those versions aligned with the build environment and retain the logs and tool versions with the artifacts.

If several Python installations exist, pass the full path to the intended Python 3.12 interpreter to `uv sync --python`; an interpreter bundled with another application may lack Tk. Check the selected environment with `uv run --locked --group build python -c "import tkinter; print(tkinter.TkVersion)"`.

| Platform | Native build prerequisites | Desktop bundle |
| --- | --- | --- |
| Windows x64 | Visual Studio Build Tools with Desktop development with C++, Windows SDK, Python 3.12 with Tcl/Tk | `dist/ascii-video.exe` |
| Linux x64 | C compiler and matching Python development headers, Tk, PortAudio, ALSA data, libxcb | `dist/linux/ascii-video/` |
| macOS x64/arm64 | Xcode Command Line Tools and a matching Python 3.12 build with Tk | `dist/macos/ASCII-Video.app` |

For a distribution-provided Python on Debian/Ubuntu, prerequisite packages include `build-essential`, the matching Python development and Tk packages, `libportaudio2`, `libxcb1`, and `libasound2-data`. A different Python installation must supply its own matching headers and Tk support. The Linux build expects `/usr/share/alsa/alsa.conf` and Tk with Xft. On a headless builder install `xvfb` and `xauth` so the build can verify Unicode font support. Build on the oldest Linux system you intend to support; do not infer a glibc baseline from older release notes.

An ordinary source installation can use Python fallbacks when a compiler is unavailable. Set `ASCII_VIDEO_PURE_PYTHON=1` before installation to skip compiling both native extensions. This does not prevent importing `.pyd`/`.so` files already present in `src/`; use a clean checkout for a Python-only installation, or the test runner's `--pure-python` flag to exercise the fallback:

```powershell
# PowerShell; remove the variable before building native release binaries.
$env:ASCII_VIDEO_PURE_PYTHON = "1"
uv sync --locked --python 3.12 --reinstall-package ascii-video
Remove-Item Env:ASCII_VIDEO_PURE_PYTHON
```

```sh
# POSIX shell; the variable applies to this command only.
ASCII_VIDEO_PURE_PYTHON=1 uv sync --locked --python 3.12 --reinstall-package ascii-video
```

## Wheel and source archive

```console
uv run --locked --group build python -m build --no-isolation --outdir build/packages
```

The default build creates a source archive, then builds a wheel from that archive. This catches missing source files that a wheel built directly from the checkout might overlook. Verify both `.pyx` files, all four assets (including `blue-noise.npy`), the license, notices, build scripts, and linked documentation. Generated C files and machine-specific native binaries are excluded from the source archive.

For a quick wheel installation check on Windows PowerShell:

```powershell
uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file build/runtime-requirements.txt
uv venv --python 3.12 build/package-check
$packagePython = (Resolve-Path build/package-check/Scripts/python.exe).Path
$packageWheel = (Get-ChildItem build/packages/*.whl | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
uv pip install --python $packagePython --require-hashes -r build/runtime-requirements.txt
uv pip install --python $packagePython --no-deps $packageWheel
Push-Location build/package-check
try {
    & $packagePython -I -m cli --version
    & $packagePython -I -c "from resources import resource_path; names = ['DejaVuSansMono.ttf', 'blue-noise.npy', 'ascii-video.ico', 'ascii-video.png']; paths = [resource_path(n) for n in names]; assert all(p is not None and p.is_file() for p in paths), paths; print(*paths, sep='\n')"
} finally {
    Pop-Location
}
```

For POSIX shells, the environment's interpreter is `build/package-check/bin/python`. Run it from a directory outside `src/`, without adding the checkout to `PYTHONPATH`. Inspect the resource paths: they should point inside the installed environment. Also perform a short `render --no-audio` export using the installed executable. Testing only imports in the source checkout is insufficient.

## Windows executable

For v0.3.1, build this executable for local use. The public Windows release offers the source archive while the project confirms redistribution rights for the Microsoft Visual C++ Runtime. See the [distribution terms](DISTRIBUTION.md).

The current notice inventory was checked with **python-build-standalone CPython 3.12.16+20261110, Windows x64**. For the same release environment, download the [inspected runtime archive](https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-x86_64-pc-windows-msvc-install_only_stripped.tar.gz), verify SHA-256 `355ee61b8207b91574491b1c6dc9b0ebb4408fc56bfafe498f3ef439314b52dc`, and extract it into `.python/windows-release/`. Its interpreter is `.python/windows-release/python/python.exe`.

```powershell
uv sync --locked --python .python/windows-release/python/python.exe --group build
.\scripts\build_windows.ps1
.\dist\ascii-video.exe --version
```

Another Python 3.12 runtime or changed dependency wheel may have different library fingerprints and needs an updated notice inventory before building the EXE. Ordinary source and wheel installations do not require this exact runtime build.

Use `-PythonPath` to select another prepared Python interpreter. The script rebuilds both native extensions, collects required dependency notices under the ignored `build/release-licenses/` directory, packages a one-file windowed executable, checks help/version, and runs a silent export. The notices, [combined-program terms](DISTRIBUTION.md), and full GPLv3 text are embedded in the EXE and copied into the release archive. Downloaded dependency notices and the separate GPLv3 copy stay under `build/`, outside Git. It requires both accelerators to import from the local `src/` directory. Only the non-ASIO PortAudio DLL is bundled. The frozen app omits the unused Python OpenSSL bindings and Cython compiler. The build checks the executable version against project metadata and compares the embedded DLLs with the native inventory.

The collector uses [scripts/native_notices.json](scripts/native_notices.json) to identify the inspected Windows libraries and their notice files. It checks package versions and DLL fingerprints, then obtains the required texts from `build/license-sources/native-notices/` or the pinned download URLs. Every notice is checked against its SHA-256 before inclusion. Git contains the inventory and download instructions, while the downloaded license texts stay under `build/`.

For an offline build, prepare that cache first. If an upstream notice URL becomes unavailable, each entry records the source archive URL, its hash, and the original member path so the same text can be recovered; apply any recorded `notice_lines` range for header excerpts. A changed library or runtime requires reviewing and updating the inventory; do not replace expected hashes merely to bypass a mismatch. This notice collection does not produce or approve the corresponding-source release archive.

## Linux bundle

```sh
uv sync --locked --python /path/to/python3.12 --group build
bash scripts/build_linux.sh
./dist/linux/ascii-video/ascii-video --version
```

An optional first argument selects the Python interpreter. Distribute the complete `ascii-video/` directory, including `_internal`; preserve executable permissions. The script includes PortAudio, libxcb, ALSA configuration, Tcl/Tk and assets. It checks the bundled Tk extension with ldd and runs a silent export. If the compiler selected by Python is absent, the script uses cc. Set CC explicitly to select another installed compiler. GUI use requires a graphical session (X11 or XWayland) and a Tk build with Xft for Cyrillic labels. The version 0.3.1 x64 binary was built on Ubuntu 22.04 and also passed the offline standalone checks on Debian 12. On Alpine Linux use a Python source or wheel installation. In addition to a matching Tk package, install `binutils` and `portaudio-dev` so Python can discover the PortAudio shared library on musl.

On Ubuntu 22.04 the verified builder used CPython 3.12.16 linked to the system Tcl/Tk with Xft. Pass the absolute path to that interpreter when creating the environment. The native CI job shows the matching interpreter setup. The builder checks the Tk font system before compiling.

## macOS bundle

```sh
uv sync --locked --python 3.12 --group build
bash scripts/build_macos.sh
./dist/macos/ASCII-Video.app/Contents/MacOS/ASCII-Video --version
```

Set `PYTHON` to select another prepared interpreter. Build separately for Intel and Apple Silicon. The pinned PyAV wheels require macOS 11+ on Intel and 14+ on Apple Silicon. By default PyInstaller applies an ad-hoc signature. To build for a publisher, set `ASCII_VIDEO_CODESIGN_IDENTITY` to the Developer ID Application identity in the build user's keychain before running the script. With the publisher's authorization, submit the signed app using a stored notarytool keychain profile, staple the ticket and verify the final app before inventorying it. [Apple notarization guide](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution) and [PyInstaller signing](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing)

## Platform checks

The [CI workflow](.github/workflows/ci.yml) checks installation of the Python wheel and has separate native builds for Windows x64, Ubuntu 22.04 x64, and macOS Intel and Apple Silicon. The Linux runtime check uses a separate offline container without Python or FFmpeg. Inspect the job results for the exact revision before marking a target accepted. Audio device tests use substitutes. In private repositories, CI also transfers each macOS build to a fresh runner of the same architecture for a second check. To run this check locally, use `bash tests/check_macos_packaged.sh /path/to/ASCII-Video.app /path/to/short-video-with-audio.mp4 /path/to/results`.

For GUI inspection on the target desktop, run both themes separately; these diagnostics disable audio:

```console
uv run --locked --group build python tests/check_platform_gui.py --expected-platform win32 --theme light --output build/gui-light
uv run --locked --group build python tests/check_platform_gui.py --expected-platform win32 --theme dark --output build/gui-dark
```

Use `linux` or `darwin` on those systems. Linux automation may use Xvfb, but screenshots still require visual review and Xvfb does not validate a real desktop. Check DPI, both preview frames, playback, seeking, zoom/pan, PNG, cancellation, minimize/restore, and shutdown. Test the packaged application on a clean machine without Python, FFmpeg, or network access. Current clean-system and desktop acceptance on all target platforms remains incomplete.

## Before publishing binaries

1. Record the exact source revision, lockfile, toolchain, target OS, and architecture. Rebuild from that source; do not reuse old executables for a new tag.
2. If using Windows Authenticode, sign the final EXE with the publisher's certificate and timestamp service. Verify it with `signtool verify /pa /v dist/ascii-video.exe` and record the signed file's hash in the source review. [Microsoft SignTool](https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool)
3. Run affected tests with the audio-blocking runner, verify native and fallback paths where relevant, and perform clean-system checks. Record skips and pending acceptance honestly.
4. Apply [DISTRIBUTION.md](DISTRIBUTION.md) to the combined desktop programs. They are conveyed under GNU GPL version 3 (`GPL-3.0-only`). The original source files retain MIT permissions. Review each target's bundled components and preserve their notices.
5. Gather corresponding sources and build information for the exact binary libraries. `uv run --locked --group build python scripts/download_python_sources.py` fetches hash-verified Python source archives from `uv.lock`; it does not gather FFmpeg/codecs, Python/Tcl/Tk, every other native library, or the application's own source archive.
6. On Windows, record each embedded DLL and Python extension in `embedded-native-evidence.json`, keyed by its path inside the executable. Each entry contains its `sha256` and a `matched` reference to its verified wheel member, runtime file or project source. Include the evidence file in the source review. Complete `build/license-sources/windows-x64-0.3.1/release-review.json` after reviewing that source set. Its fields are `status` (`ready` after review), `blockers`, `files` (relative source paths mapped to SHA-256), `executable_sha256`, `uv_lock_sha256`, and `license_manifest_sha256`. [Windows packager](scripts/package_windows.py).
7. Package the reviewed Windows build with `uv run --locked --group build python scripts/package_windows.py --name ascii-video-0.3.1-windows-x64`. It checks the EXE, actual embedded license documents, current distribution terms, source hashes, and lock; it stages the binary/source ZIPs, verifies them, and publishes both with a checksum file under `dist/<name>/`. The binary ZIP carries `DISTRIBUTION.md`, the full `COPYING.GPLv3`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` at its root. Its `README.txt` identifies the source companion and checksum. Offer both archives at the same download location, with equivalent access and no further charge for the source, following GPLv3 section 6(d). Linux and macOS build scripts generate a bundle inventory. Review the matching sources and component notices on that architecture, then run `python scripts/package_posix.py package BUNDLE --sources REVIEW_DIRECTORY --name RELEASE_NAME` to create the archive pair. See [distribution terms](DISTRIBUTION.md).

## Publish the repository

Review `git status` and the final diff. Commit source, assets, public documentation, lockfile, the project license, and third-party references; keep local videos, environments, generated files, and diagnostic output excluded. Add the actual repository URL to project metadata once a public host is chosen, then push the intended branch and release tags. A remote is not preconfigured in this checkout. Confirm the CI run before using it as a required branch check.

These steps follow [GitHub's repository recommendations](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories), the [Open Source Guides](https://opensource.guide/starting-a-project/), and the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/). Packaging formats have technical specifications; there is no single mandatory checklist for every open-source project.

#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
python=${PYTHON:-"$root/.venv/bin/python"}
"$python" -c 'import sys; assert sys.platform == "darwin", "Build requires macOS"'
"$python" -c 'from pathlib import Path; [p.unlink() for pattern in ("_glyphs*.so", "_diffusion*.so") for p in Path("src").glob(pattern)]'
"$python" setup.py build_ext --inplace --force \
    --build-lib build/macos-native/lib --build-temp build/macos-native/temp
"$python" -c 'import sys; sys.path.insert(0, "src"); import raster, diffusion; assert raster._compose is not None and diffusion._native_quantize is not None, "Native accelerators required"'

set --
if [[ -n ${ASCII_VIDEO_CODESIGN_IDENTITY:-} ]]; then
    set -- --codesign-identity "$ASCII_VIDEO_CODESIGN_IDENTITY"
fi
"$python" -m PyInstaller --noconfirm --clean --onedir --windowed --noupx \
    "$@" \
    --exclude-module Cython --exclude-module cython \
    --name ASCII-Video --paths "$root/src" --hidden-import PIL._tkinter_finder \
    --icon "$root/assets/ascii-video.png" --add-data "$root/assets:assets" \
    --add-data "$root/LICENSE:." --add-data "$root/THIRD_PARTY_NOTICES.md:." \
    --workpath "$root/build/macos" --specpath "$root/build/macos" --distpath "$root/dist/macos" "$root/src/main.py"
executable="$root/dist/macos/ASCII-Video.app/Contents/MacOS/ASCII-Video"
if [[ -n ${ASCII_VIDEO_CODESIGN_IDENTITY:-} ]]; then
    codesign --verify --deep --strict "$root/dist/macos/ASCII-Video.app"
fi
"$executable" --help
"$executable" --version
"$python" tests/check_packaged_export.py --executable "$executable" --all-formats
"$python" scripts/package_posix.py inventory "$root/dist/macos/ASCII-Video.app" --output "$root/build/macos/bundle-inventory.json"
printf '%s\n' "Diagnostic bundle: $root/dist/macos/ASCII-Video.app"

#!/usr/bin/env bash
# Build locally on Linux. A successful build is not a license/source release review.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${1:-"$root/.venv/bin/python"}
cd "$root"
"$python" -c 'import platform,sys; assert sys.platform == "linux" and platform.machine() == "x86_64", "Requires Linux x86_64"; import tkinter'
gui_python=("$python")
if [[ -z ${DISPLAY:-} ]]; then
    command -v xvfb-run >/dev/null || { echo 'Install xvfb and xauth for a headless GUI build check.' >&2; exit 1; }
    gui_python=(xvfb-run -a "$python")
fi
"${gui_python[@]}" - <<'PY'
import tkinter
root = tkinter.Tk()
try:
    if root.tk.call("tk::pkgconfig", "get", "fontsystem") != "xft":
        raise SystemExit("Use Python with Xft-enabled Tk for Cyrillic GUI labels on Linux.")
finally:
    root.destroy()
PY

mkdir -p build/linux dist/linux
"$python" - <<'PY'
from pathlib import Path
for pattern in ("_glyphs*.so", "_diffusion*.so"):
    for path in Path("src").glob(pattern):
        path.unlink()
PY
native=$(mktemp -d "$root/build/native-linux.XXXXXX")
if [[ -z ${CC:-} ]]; then
    configured_cc=$("$python" -c 'import shlex, sysconfig; print(shlex.split(sysconfig.get_config_var("CC"))[0])')
    if ! command -v "$configured_cc" >/dev/null; then
        export CC=cc
    fi
fi
"$python" setup.py build_ext --inplace --force --build-lib "$native/lib" --build-temp "$native/temp"
"$python" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "src")
import _glyphs, _diffusion
assert hasattr(_glyphs, "resize_source_band_into")
assert all(Path(module.__file__).resolve().parent == Path("src").resolve()
           for module in (_glyphs, _diffusion))
PY

mapfile -t libraries < <("$python" - <<'PY'
import ctypes
from pathlib import Path
for name in ("libportaudio.so.2", "libxcb.so.1"):
    ctypes.CDLL(name)
    paths = {line.split(maxsplit=5)[5].strip() for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6 and Path(line.split(maxsplit=5)[5].strip()).name.startswith(name)}
    if len(paths) != 1:
        raise RuntimeError(f"Cannot resolve required library {name}: {paths}")
    library = Path(paths.pop()).parent / name
    if not library.is_file():
        raise RuntimeError(f"Required SONAME symlink is missing: {library}")
    print(library)
PY
)
if [[ ${#libraries[@]} -ne 2 ]]; then
    echo 'Install libportaudio2 and libxcb1 before building.' >&2
    exit 1
fi
test -f /usr/share/alsa/alsa.conf
# Standalone CPython finds its private Tcl/Tk through its executable's RPATH.
# PyInstaller resolves _tkinter's ELF dependencies in a separate ldd process.
python_lib=$("$python" -c 'from pathlib import Path; import sys; print(Path(sys.base_prefix) / "lib")')
LD_LIBRARY_PATH="$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$python" -m PyInstaller --noconfirm --clean --onedir --noupx \
    --exclude-module Cython --exclude-module cython --exclude-module _ssl --exclude-module _hashlib --exclude-module readline \
    --name ascii-video --paths "$root/src" --hidden-import PIL._tkinter_finder \
    --runtime-hook "$root/scripts/pyinstaller_linux_runtime.py" \
    --add-data "$root/assets:assets" --add-data '/usr/share/alsa:alsa' \
    --add-data "$root/LICENSE:." --add-data "$root/THIRD_PARTY_NOTICES.md:." \
    --add-binary "${libraries[0]}:." --add-binary "${libraries[1]}:." \
    --workpath "$root/build/linux/pyinstaller" --specpath "$root/build/linux/pyinstaller" \
    --distpath "$root/dist/linux" "$root/src/main.py"
executable="$root/dist/linux/ascii-video/ascii-video"
internal="$(dirname "$executable")/_internal"
tk_module=$(find "$internal" -name '_tkinter*.so' -type f -print -quit)
test -n "$tk_module"
tk_deps=$(LD_LIBRARY_PATH="$internal" ldd "$tk_module")
if grep -q 'not found' <<<"$tk_deps"; then
    printf '%s\n' "$tk_deps" >&2
    exit 1
fi
while read -r dependency; do
    if [[ "$dependency" =~ ^lib(tcl|tk).*\.so && "$dependency" != *" => $internal/"* ]]; then
        echo "Tk dependency is outside the bundle: $dependency" >&2
        exit 1
    fi
done <<<"$tk_deps"
"$executable" --help >/dev/null
"$executable" --version
"$python" tests/check_packaged_export.py --executable "$executable" --all-formats
"$python" scripts/package_posix.py inventory "$(dirname "$executable")" --output "$root/build/linux/bundle-inventory.json"
printf 'Local Linux build: %s\nBefore publication, review Linux-specific native sources and licenses.\n' "$executable"

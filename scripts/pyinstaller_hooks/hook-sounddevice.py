"""Bundle only the Windows x64 PortAudio backend used by this application."""

import importlib.util
from pathlib import Path


module = importlib.util.find_spec("_sounddevice_data")
if module is None or module.submodule_search_locations is None:
    raise RuntimeError("The Windows sounddevice wheel with PortAudio is required")

directory = Path(next(iter(module.submodule_search_locations))) / "portaudio-binaries"
library = directory / "libportaudio64bit.dll"
if not library.is_file():
    raise RuntimeError(f"Required non-ASIO Windows x64 PortAudio DLL is missing: {library}")

binaries = [(str(library), "_sounddevice_data/portaudio-binaries")]
datas = [(str(directory / "README.md"), "_sounddevice_data/portaudio-binaries")]

"""Resolve bundled fonts and PortAudio before Tk/sounddevice initialize."""
import ctypes.util
import os
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree

bundle = Path(sys._MEIPASS)
# Debian's Tcl 8.6 stores .tm packages directly in this directory.
# Tcl's compiled search roots still point at /usr/share/tcltk after relocation.
tcl_modules = bundle / "_tcl_data" / "tcl8"
if tcl_modules.is_dir():
    os.environ["TCL8_6_TM_PATH"] = str(tcl_modules)
sys._ascii_font_directory = tempfile.TemporaryDirectory(prefix="ascii-video-fonts-")
config = Path(sys._ascii_font_directory.name) / "fonts.conf"
fonts = ElementTree.Element("fontconfig")
ElementTree.SubElement(fonts, "include", ignore_missing="yes").text = "/etc/fonts/fonts.conf"
ElementTree.SubElement(fonts, "dir").text = str(bundle / "assets")
ElementTree.SubElement(fonts, "cachedir", prefix="xdg").text = "fontconfig/ascii-video"
ElementTree.ElementTree(fonts).write(config, encoding="utf-8", xml_declaration=True)
os.environ["FONTCONFIG_FILE"] = str(config)

# Minimal Linux installations may have the ALSA library without its configuration.
if "ALSA_CONFIG_PATH" not in os.environ and not Path("/usr/share/alsa/alsa.conf").is_file():
    alsa = bundle / "alsa"
    os.environ.setdefault("ALSA_CONFIG_DIR", str(alsa))
    os.environ["ALSA_CONFIG_PATH"] = str(alsa / "alsa.conf")

_system_find_library = ctypes.util.find_library


def _find_library(name):
    portaudio = bundle / "libportaudio.so.2"
    if name == "portaudio" and portaudio.is_file():
        return str(portaudio)
    return _system_find_library(name)


ctypes.util.find_library = _find_library

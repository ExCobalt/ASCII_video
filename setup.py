"""Optional native acceleration; pure Python remains available without a compiler."""
import os
from setuptools import Extension, setup

setup(ext_modules=[] if os.environ.get("ASCII_VIDEO_PURE_PYTHON") == "1" else [
    Extension("_glyphs", ["src/_glyphs.pyx"], optional=True),
    Extension("_diffusion", ["src/_diffusion.pyx"], optional=True),
])

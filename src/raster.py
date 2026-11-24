"""Portable ASCII glyph atlas and RGB frames for file export."""

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as _PILLOW_VERSION

from resources import resource_path

try:
    from _glyphs import compose as _compose
except ImportError:
    _compose = None

try:
    from _glyphs import compose_planes as _compose_planes
except ImportError:
    _compose_planes = None

try:
    from _glyphs import compose_planes_sparse as _compose_planes_sparse
except ImportError:
    _compose_planes_sparse = None

try:
    from _glyphs import make_source_tinted_atlas as _make_source_tinted_atlas
    from _glyphs import compose_source_tinted as _compose_source_tinted
    from _glyphs import resize_source_band as _resize_source_band
except ImportError:
    _make_source_tinted_atlas = _compose_source_tinted = _resize_source_band = None

try:
    from _glyphs import resize_source_band_into as _resize_source_band_into
except ImportError:
    _resize_source_band_into = None

# FreeType returns linear pixel coverage, not display-encoded brightness.
# On black, an sRGB coverage curve approximates linear-light text blending
# without changing the foreground RGB or doing per-frame power operations.
_coverage = np.arange(256) / 255
_TEXT_COVERAGE = np.floor(255 * np.where(_coverage <= .0031308, 12.92 * _coverage,
                                      1.055 * _coverage ** (1 / 2.4) - .055) + .5).astype(np.uint8)


def font_path():
    path = resource_path("DejaVuSansMono.ttf")
    if path is not None:
        return path
    raise RuntimeError("Bundled DejaVu Sans Mono font is missing")


@lru_cache(maxsize=8)
def _glyph_atlas(cell_width, cell_height, font_size):
    font = ImageFont.truetype(str(font_path()), font_size, layout_engine=ImageFont.Layout.BASIC)
    characters = [chr(code) for code in range(32, 127)]
    bounds = [font.getbbox(character, anchor="ls") for character in characters]
    left = min(box[0] for box in bounds)
    top = min(box[1] for box in bounds)
    right = max(box[2] for box in bounds)
    bottom = max(box[3] for box in bounds)
    if right - left > cell_width or bottom - top > cell_height:
        raise RuntimeError("Bundled font glyphs do not fit the cell")
    origin = ((cell_width - (right - left)) // 2 - left,
              (cell_height - (bottom - top)) // 2 - top)
    atlas = np.empty((95, cell_height, cell_width), dtype=np.uint8)
    for index, character in enumerate(characters):
        tile = Image.new("L", (cell_width, cell_height))
        ImageDraw.Draw(tile).text(origin, character, font=font, fill=255, anchor="ls")
        atlas[index] = np.asarray(tile)
    atlas = _TEXT_COVERAGE[atlas]
    atlas.flags.writeable = False
    return atlas, origin


@lru_cache(maxsize=3)
def glyph_profile(palette):
    """Order a built-in ramp by its actual export-mask density."""
    atlas, _ = _glyph_atlas(Rasterizer.CELL_WIDTH, Rasterizer.CELL_HEIGHT, Rasterizer.FONT_SIZE)
    characters = np.array(list(palette), dtype="U1")
    coverage = atlas[characters.view(np.uint32) - 32].mean(axis=(1, 2))
    order = np.argsort(coverage, kind="stable")
    characters, levels = characters[order], coverage[order] / coverage.max()
    characters.flags.writeable = levels.flags.writeable = False
    return characters, levels


class Rasterizer:
    CELL_WIDTH = 8
    CELL_HEIGHT = 16
    FONT_SIZE = 12
    # Large RGB rasters also need Pillow copies and three resize planes.
    # Above this size, compose/resize one channel in horizontal strips.
    SOURCE_RGB_LIMIT = 64 * 1024 * 1024

    def __init__(self, palette, color, width, height):
        if not 2 <= len(palette) <= 95 or len(set(palette)) != len(palette) or any(not 32 <= ord(c) <= 126 for c in palette):
            raise ValueError("palette requires 2–95 unique printable ASCII characters")
        if width < 1 or height < 1:
            raise ValueError("output width and height must be positive")
        self.width, self.height = width, height
        self.color = color
        self.rgb = None
        if color == "mono":
            self.rgb = (255, 255, 255)
        elif color != "source":
            if len(color) != 7 or color[0] != "#":
                raise ValueError("expected mono, source or #RRGGBB")
            try:
                self.rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            except ValueError:
                raise ValueError("expected mono, source or #RRGGBB") from None
        self.tint_palette = (bytes((coverage * channel + 127) // 255
                                   for coverage in range(256) for channel in self.rgb)
                             if color not in ("mono", "source") else None)
        self.atlas, self.origin = _glyph_atlas(self.CELL_WIDTH, self.CELL_HEIGHT, self.FONT_SIZE)
        self._scaled_atlas = None
        self._resize_pool = None
        self._source_parallel_strips = False
        self._source_fast_raster = False
        self._source_planar_export = False
        self._source_direct_resize = False
        self._source_tinted_atlas = None
        self._tinted_atlas = None

    def _draw_large_source(self, codes, colors, atlas, fitted):
        rows, columns = codes.shape
        _, cell_height, cell_width = atlas.shape
        raster_width, raster_height = columns * cell_width, rows * cell_height
        # Keep each gather/compositing temporary near 1 MiB; a single glyph
        # row is the minimum. Pillow's horizontal pass is independent by row.
        strip_rows = max(1, (1024 * 1024) // (raster_width * cell_height))
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        left, top = (self.width - fitted[0]) // 2, (self.height - fitted[1]) // 2
        def draw_channel(channel):
            with Image.new("L", (fitted[0], raster_height)) as intermediate:
                for start in range(0, rows, strip_rows):
                    end = min(rows, start + strip_rows)
                    coverage = atlas[codes[start:end] - 32].astype(np.uint16)
                    coverage *= colors[start:end, :, channel, None, None]
                    coverage += 127
                    coverage //= 255
                    pixels = coverage.astype(np.uint8).transpose(0, 2, 1, 3).reshape(
                        (end - start) * cell_height, raster_width)
                    with Image.fromarray(pixels) as strip:
                        with strip.resize((fitted[0], strip.height), Image.Resampling.LANCZOS) as resized:
                            intermediate.paste(resized, (0, start * cell_height))
                # Same uint8 horizontal-then-vertical passes as RGB LANCZOS,
                # without retaining the full oversampled RGB image.
                with intermediate.resize(fitted, Image.Resampling.LANCZOS) as resized:
                    return np.asarray(resized)
        channels = (map(draw_channel, range(3)) if self._resize_pool is None
                    else self._resize_pool.map(draw_channel, range(3)))
        for channel, pixels in enumerate(channels):
            canvas[top:top + fitted[1], left:left + fitted[0], channel] = pixels
        return canvas

    def draw(self, symbols, colors):
        if symbols.ndim != 2 or symbols.dtype.kind != "U" or symbols.dtype.itemsize != 4 or 0 in symbols.shape:
            raise ValueError("symbols must be a nonempty U1 matrix")
        codes = symbols.astype("U1", copy=False).view(np.uint32)
        if np.any((codes < 32) | (codes > 126)):
            raise ValueError("symbols must be printable ASCII")
        rows, columns = symbols.shape
        cell_width, cell_height = self.CELL_WIDTH, self.CELL_HEIGHT
        atlas = self.atlas
        scale = min(self.width / (columns * cell_width), self.height / (rows * cell_height))
        # Exact half-step enlargement can rasterize the font at its final size.
        # Keep only the last scaled atlas, capped below 1 MiB at scale 8.
        if 1.5 <= scale <= 8 and (scale * 2).is_integer():
            steps = int(scale * 2)
            scaled = self._scaled_atlas
            if scaled is None or scaled[0] != steps:
                enlarged, _ = _glyph_atlas(4 * steps, 8 * steps, 6 * steps)
                scaled = (steps, enlarged)
                self._scaled_atlas = scaled
            atlas = scaled[1]
            cell_width, cell_height = 4 * steps, 8 * steps
        raster_width, raster_height = columns * cell_width, rows * cell_height
        native_scale = min(self.width / raster_width, self.height / raster_height)
        fitted = (min(self.width, max(1, math.floor(raster_width * native_scale + 0.5))),
                  min(self.height, max(1, math.floor(raster_height * native_scale + 0.5))))
        # Near-native enlargement keeps the glyph pixel grid and adds padding.
        if 1 <= native_scale <= 1.01:
            fitted = (raster_width, raster_height)
        if codes[0, 0] == 32 and np.all(codes == 32) and not atlas[0].any():
            # Blank frames need no composition or LANCZOS. Check the selected
            # atlas as callers can replace masks, including the space glyph.
            if self.color == "source" and (colors.shape != (*symbols.shape, 3) or colors.dtype != np.uint8):
                raise ValueError("source colors must be an H×C×3 uint8 matrix")
            if self.color == "mono":
                blank = np.zeros((self.height, self.width), dtype=np.uint8)
                return np.broadcast_to(blank[..., None], (self.height, self.width, 3))
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if (self.color == "source" and native_scale < 1
                and raster_width * raster_height * 3 > self.SOURCE_RGB_LIMIT):
            if colors.shape != (rows, columns, 3) or colors.dtype != np.uint8:
                raise ValueError("source colors must be an H×C×3 uint8 matrix")
            return self._draw_large_source(codes, colors, atlas, fitted)
        if self.tint_palette is not None and (raster_width, raster_height) == (self.width, self.height):
            # Cache only the last immutable atlas, including its scaled form.
            # A mutable caller-supplied atlas is rebuilt to observe edits.
            cached = self._tinted_atlas
            if cached is None or cached[0] is not atlas or atlas.flags.writeable:
                tinted = ((atlas[..., None].astype(np.uint16) * np.asarray(self.rgb, dtype=np.uint16) + 127) // 255).astype(np.uint8)
                scanlines = tinted.reshape(95, cell_height, cell_width * 3).view(
                    np.dtype((np.void, cell_width * 3))).reshape(95, cell_height)
                cached = (atlas, scanlines)
                self._tinted_atlas = cached
            return cached[1][(codes - 32)[:, None, :],
                             np.arange(cell_height)[None, :, None]].view(np.uint8).reshape(
                self.height, self.width, 3)
        if _compose is not None and self.color != "mono" and 1 <= native_scale <= 1.01:
            if self.color == "source":
                if colors.shape != (rows, columns, 3) or colors.dtype != np.uint8:
                    raise ValueError("source colors must be an H×C×3 uint8 matrix")
                foreground = colors
            else:
                foreground = np.broadcast_to(np.asarray(self.rgb, dtype=np.uint8), (rows, columns, 3))
            pixels = _compose(codes, foreground, atlas)
            if (raster_width, raster_height) == (self.width, self.height):
                return pixels
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            left, top = (self.width - raster_width) // 2, (self.height - raster_height) // 2
            canvas[top:top + raster_height, left:left + raster_width] = pixels
            return canvas
        if _compose is not None and self.color == "source" and native_scale < 1:
            if colors.shape != (rows, columns, 3) or colors.dtype != np.uint8:
                raise ValueError("source colors must be an H×C×3 uint8 matrix")
            if (_compose_planes is not None and self._resize_pool is not None
                    and raster_width * raster_height >= 1280 * 720):
                # The cached native path is limited to validated geometry
                # and Pillow version; old extensions keep sparse composition.
                native_source = (self._source_fast_raster and self._source_parallel_strips
                                 and (raster_width, raster_height) == (2560, 1440)
                                 and fitted == (self.width, self.height) == (1920, 1080)
                                 and _make_source_tinted_atlas is not None
                                 and _PILLOW_VERSION in ("12.3.0", "13.0.0"))
                if native_source:
                    cached = self._source_tinted_atlas
                    if atlas.flags.writeable:
                        tinted = _make_source_tinted_atlas(atlas)
                        self._source_tinted_atlas = None
                    else:
                        if cached is None or cached[0] is not atlas:
                            cached = (atlas, _make_source_tinted_atlas(atlas))
                            self._source_tinted_atlas = cached
                        tinted = cached[1]
                    planes = _compose_source_tinted(codes, colors, tinted)
                else:
                    composer = (_compose_planes_sparse if self._source_parallel_strips
                                and _compose_planes_sparse is not None else _compose_planes)
                    planes = composer(codes, colors, atlas)
                if (self._source_parallel_strips and (raster_width, raster_height) == (2560, 1440)
                        and fitted == (self.width, self.height) == (1920, 1080)):
                    # Export's 12-worker pool divides each channel into four
                    # strips. Integer 4:3 box boundaries preserve the full-image
                    # LANCZOS coefficients, including support across strip edges.
                    direct_resize = (self._source_direct_resize and _resize_source_band_into is not None
                                     and _PILLOW_VERSION in ("12.3.0", "13.0.0"))
                    images = None if native_source or direct_resize else [Image.fromarray(plane) for plane in planes]
                    shape = (3, 1080, 1920) if self._source_planar_export else (1080, 1920, 3)
                    pixels = np.empty(shape, dtype=np.uint8)
                    def resize_strip(task):
                        channel, top = task
                        if direct_resize:
                            destination = pixels.transpose(1, 2, 0) if self._source_planar_export else pixels
                            _resize_source_band_into(planes[channel], top, destination, channel)
                            return None
                        if native_source:
                            return channel, top, _resize_source_band(planes[channel], top)
                        image = images[channel].resize(
                            (1920, 270), Image.Resampling.LANCZOS,
                            box=(0, top * 4 // 3, 2560, (top + 270) * 4 // 3))
                        return channel, top, np.asarray(image)
                    tasks = [(channel, top) for channel in range(3) for top in range(0, 1080, 270)]
                    for result in self._resize_pool.map(resize_strip, tasks):
                        if result is None:
                            continue
                        channel, top, strip = result
                        if self._source_planar_export:
                            pixels[channel, top:top + 270, :] = strip
                        else:
                            pixels[top:top + 270, :, channel] = strip
                    if self._source_planar_export:
                        pixels.flags.writeable = False
                        return pixels.transpose(1, 2, 0)
                    return pixels
                def resize_plane(plane):
                    return Image.fromarray(plane).resize(fitted, Image.Resampling.LANCZOS)
                image = Image.merge("RGB", tuple(self._resize_pool.map(resize_plane, planes)))
                del planes
            else:
                image = Image.fromarray(_compose(codes, colors, atlas))
        else:
            # Gather whole scanlines in image order, avoiding a full tile array
            # followed by a second full-size copy during transpose/reshape.
            scanlines = atlas.view(np.dtype((np.void, cell_width))).reshape(95, cell_height)
            coverage = scanlines[(codes - 32)[:, None, :],
                                 np.arange(cell_height)[None, :, None]].view(np.uint8).reshape(
                rows * cell_height, columns * cell_width)
            if self.color == "mono" and (raster_width, raster_height) == (self.width, self.height):
                # The gathered frame owns its memory; no Pillow byte copy is
                # needed, and later draws cannot overwrite retained pixels.
                coverage.flags.writeable = False
                return np.broadcast_to(coverage[..., None], (self.height, self.width, 3))
            image = Image.fromarray(coverage)
            if self.color == "source":
                if colors.shape != (rows, columns, 3) or colors.dtype != np.uint8:
                    raise ValueError("source colors must be an H×C×3 uint8 matrix")
                foreground = Image.fromarray(colors).resize(image.size, Image.Resampling.NEAREST)
                canvas = Image.new("RGB", image.size)
                canvas.paste(foreground, (0, 0), image)
                image = canvas
            else:
                # Constant colors need only one coverage channel. A fixed 256-entry
                # palette reproduces integer RGB compositing without frame arithmetic.
                if self.tint_palette is not None:
                    image.putpalette(self.tint_palette)
                    image = image.convert("RGB")
        if image.size != fitted:
            if (self._resize_pool is not None and self.color == "source"
                    and image.width * image.height >= 1280 * 720):
                def resize_channel(channel):
                    return channel.resize(fitted, Image.Resampling.LANCZOS)
                channels = self._resize_pool.map(resize_channel, image.split())
                image = Image.merge("RGB", tuple(channels))
            else:
                image = image.resize(fitted, Image.Resampling.LANCZOS)
        if fitted != (self.width, self.height):
            canvas = Image.new(image.mode, (self.width, self.height))
            canvas.paste(image, ((self.width - fitted[0]) // 2, (self.height - fitted[1]) // 2))
            image = canvas
        pixels = np.asarray(image)
        if image.mode == "L":
            # A readonly RGB view retains the independent grayscale frame.
            # Consumers such as Pillow/PyAV accept its repeated channel stride.
            return np.broadcast_to(pixels[..., None], (self.height, self.width, 3))
        return pixels

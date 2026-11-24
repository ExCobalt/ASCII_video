"""Pure NumPy conversion of RGB frames to ASCII cells."""

import math

import numpy as np


from cli_options import PALETTES
from dithering import BAYER, DITHER_MODES, thresholds
from diffusion import MODES as DIFFUSION_MODES, quantize


DEFAULT_PALETTE = PALETTES["basic"]


def fit_grid(width, height, columns, rows, cell_aspect, display_aspect):
    """Return image columns/rows and the enclosing terminal rectangle."""
    if columns < 2 or rows < 2:
        raise ValueError("Terminal must have at least two columns and rows")
    if (width is not None and width < 1) or (height is not None and height < 1):
        raise ValueError("Grid bounds must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (cell_aspect, display_aspect)):
        raise ValueError("Aspect ratios must be finite and positive")
    bw = min(width if width is not None else columns - 1, columns - 1)
    bh = min(height if height is not None else rows - 1, rows - 1)
    c = max(1, math.floor(min(bw, bh * display_aspect / cell_aspect)))
    h = min(bh, max(1, math.floor(c * cell_aspect / display_aspect + 0.5)))
    return c, h, bw, bh


def display_aspect(width, height, sample_aspect=1.0, rotation=0):
    """Display ratio including pixel shape and clockwise rotation."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if sample_aspect is None:
        sample_aspect = 1.0
    if not math.isfinite(sample_aspect) or sample_aspect <= 0:
        raise ValueError("Sample aspect must be finite and positive")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Only right-angle rotations are supported")
    aspect = width * float(sample_aspect) / height
    return 1 / aspect if rotation in (90, 270) else aspect


def rotate_rgb(rgb, rotation=0):
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Only right-angle rotations are supported")
    return np.rot90(rgb, -(rotation // 90)) if rotation else rgb


def _axis_regions(source, target):
    positions = np.arange(target + 1, dtype=np.int64) * source
    bounds = positions // target
    sizes = np.diff(bounds)
    return bounds[:-1], sizes, (positions % target) / target


def _fractional_edges(sums, source, starts, sizes, fractions, axis):
    """Correct integer buckets to the exact box boundaries in float64."""
    sums = sums.astype(np.float64, copy=False)
    shape = [1] * source.ndim
    shape[axis] = len(starts)
    sums *= (sizes > 0).reshape(shape)
    sums -= np.take(source, starts, axis=axis) * fractions[:-1].reshape(shape)
    ends = np.minimum(starts + sizes, source.shape[axis] - 1)
    sums += np.take(source, ends, axis=axis) * fractions[1:].reshape(shape)
    return sums


def _level_indices(values, levels, field=None):
    """Quantize tone, interpolating adjacent densities for ordered dither."""
    if field is None:
        return np.searchsorted((levels[:-1] + levels[1:]) / 2, values, side="right")
    lower = np.clip(np.searchsorted(levels, values, side="right") - 1, 0, len(levels) - 1)
    upper = np.minimum(lower + 1, len(levels) - 1)
    span = levels[upper] - levels[lower]
    fraction = np.divide(values - levels[lower], span,
                         out=np.zeros_like(values), where=span > 0)
    return np.where(fraction >= field + .5, upper, lower)


class Renderer:
    def __init__(self, palette=DEFAULT_PALETTE, gamma=1.0, contrast=1.0, dither="none", brightness=0.0,
                 color="mono", color_saturation=0.0, glyph_correction=0.0):
        if not (2 <= len(palette) <= 95 and len(set(palette)) == len(palette)
                and all(32 <= ord(c) <= 126 for c in palette)):
            raise ValueError("Palette must contain 2–95 unique printable ASCII characters")
        if not math.isfinite(gamma) or not 0.1 <= gamma <= 5:
            raise ValueError("Gamma must be between 0.1 and 5")
        if not math.isfinite(contrast) or not 0.1 <= contrast <= 3:
            raise ValueError("Contrast must be between 0.1 and 3")
        if not math.isfinite(brightness) or not -1 <= brightness <= 1:
            raise ValueError("Brightness must be between -1 and 1")
        if not math.isfinite(color_saturation) or not -1 <= color_saturation <= 1:
            raise ValueError("Color saturation must be between -1 and 1")
        if not math.isfinite(glyph_correction) or not 0 <= glyph_correction <= 1:
            raise ValueError("Glyph correction must be between 0 and 1")
        if dither not in (*DITHER_MODES, *DIFFUSION_MODES):
            raise ValueError("Unknown dither mode")
        if palette in PALETTES.values():
            from raster import glyph_profile
            self.palette, self.levels = glyph_profile(palette)
        else:
            self.palette = np.array(list(palette), dtype="U1")
            self.levels = np.linspace(0, 1, len(palette))
        self.gamma = gamma
        self.contrast = contrast
        self.brightness = brightness
        self.dither = dither
        self.color = color
        self.color_saturation = color_saturation
        self.glyph_correction = glyph_correction
        if glyph_correction:
            from raster import _glyph_atlas
            atlas, _ = _glyph_atlas(8, 16, 12)
            self._coverage = atlas.mean(axis=(1, 2)) / 255
            ink = self._coverage[self.palette.view(np.uint32) - 32]
            self._ink_order = np.argsort(ink, kind="stable")
            self._ink = ink[self._ink_order]
            self._max_ink = self._ink[-1]
        self._regions = None
        self._dither_shape = None

    def sample(self, rgb, columns, rows):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Expected a nonempty RGB frame")
        if columns < 1 or rows < 1:
            raise ValueError("Grid dimensions must be positive")
        geometry = rgb.shape[:2] + (columns, rows)
        regions = self._regions
        if regions is None or geometry != regions[0]:
            y, dy, fy = _axis_regions(rgb.shape[0], rows)
            x, dx, fx = _axis_regions(rgb.shape[1], columns)
            band_height = rgb.shape[0] // rows
            band_indices = extra = None
            if band_height and rgb.shape[0] % rows:
                band_indices = y[:, None] + np.arange(band_height)
                extra = np.flatnonzero(dy > band_height)
            # RGB8 sums fit exactly in float32 while the whole cell sum is
            # <= 2**24. Upsampling keeps the existing float64 path.
            small_cells = (columns <= rgb.shape[1] and rows <= rgb.shape[0]
                           and int(dy.max()) * int(dx.max()) <= (2**24 // 255))
            regions = (geometry, y, dy, fy, x, dx, fx, band_indices, extra, small_cells)
            self._regions = regions
        _, y, dy, fy, x, dx, fx, band_indices, extra, small_cells = regions
        accumulator = np.float32 if rgb.dtype == np.uint8 and small_cells else np.float64
        # First sum integer buckets; fractional endpoint pixels are then
        # shared proportionally between adjacent cells, including upsampling.
        if rgb.shape[0] % rows == 0:
            # Equal-height bands let NumPy sum contiguous rows instead of
            # reducing one strided column at a time. RGB8 sums stay exact.
            bands = rgb.reshape(rows, rgb.shape[0] // rows, rgb.shape[1], 3)
            grid = bands.sum(axis=1, dtype=accumulator)
        elif band_indices is not None:
            # RGB8 sums are exact integers here. Add one scanline per band
            # at a time to avoid gathering another full-resolution frame.
            # Keep long bands and arbitrary floating inputs on NumPy's sum.
            band_height = band_indices.shape[1]
            if accumulator == np.float32 and band_height <= 32:
                grid = rgb[y].astype(accumulator)
                for offset in range(1, band_height):
                    grid += rgb[y + offset]
            else:
                grid = rgb[band_indices].sum(axis=1, dtype=accumulator)
            grid[extra] += rgb[y[extra] + band_height]
        else:
            grid = np.add.reduceat(rgb, y, axis=0, dtype=accumulator)
        horizontal = np.add.reduceat(grid, x, axis=1, dtype=grid.dtype)
        if rgb.shape[1] % columns:
            horizontal = _fractional_edges(horizontal, grid, x, dx, fx, 1)
        grid = horizontal
        if rgb.shape[0] % rows:
            # Reduce the boundary scanlines horizontally before promoting.
            # The correction needs only rows×columns RGB, not rows×source width.
            edges = rgb[np.minimum(np.r_[y, rgb.shape[0]], rgb.shape[0] - 1)]
            horizontal = np.add.reduceat(edges, x, axis=1, dtype=accumulator)
            if rgb.shape[1] % columns:
                horizontal = _fractional_edges(horizontal, edges, x, dx, fx, 1)
            grid = grid.astype(np.float64, copy=False)
            grid *= (dy > 0)[:, None, None]
            grid -= horizontal[:-1] * fy[:-1, None, None]
            grid += horizontal[1:] * fy[1:, None, None]
        # Avoid a full-resolution float64 frame or cumulative-sum buffer.
        # Integer ratios retain exact RGB8 accumulation.
        grid = grid.astype(np.float64, copy=False)
        grid /= (rgb.shape[0] / rows) * (rgb.shape[1] / columns)
        return grid

    @staticmethod
    def _luminance(grid):
        y = 0.2126 * grid[..., 0]
        y += 0.7152 * grid[..., 1]
        y += 0.0722 * grid[..., 2]
        return np.clip(y, 0, 255, out=y)

    def _tone(self, y):
        if self.gamma != 1 or self.contrast != 1 or self.brightness != 0:
            # Preserve floating-point operation order, reusing only our own buffer.
            y = y / 255
            y -= 0.5
            y *= self.contrast
            y += 0.5
            y *= 1 + self.brightness
            np.clip(y, 0, 1, out=y)
            np.power(y, 1 / self.gamma, out=y)
            y *= 255
        return y

    def _indices(self, values, levels, minimum=None):
        if self.dither in DIFFUSION_MODES:
            return quantize(values, levels, self.dither, minimum)
        field = None
        if self.dither != "none":
            if values.shape != self._dither_shape:
                self._thresholds = thresholds(values.shape, self.dither)
                self._dither_shape = values.shape
            field = self._thresholds
        indices = _level_indices(values, levels, field)
        return indices if minimum is None else np.maximum(indices, minimum)

    def _color_grid(self, grid):
        """Scale encoded-RGB chroma, preserving luma and its hue direction."""
        if self.color != "source" or self.color_saturation == 0:
            return grid
        rgb = np.clip(grid, 0, 255)
        y = self._luminance(rgb)
        chroma = rgb - y[..., None]
        upper = np.maximum(np.maximum(chroma[..., 0], chroma[..., 1]), chroma[..., 2])
        lower = -np.minimum(np.minimum(chroma[..., 0], chroma[..., 1]), chroma[..., 2])
        high = np.divide(255 - y, upper, out=np.full_like(y, np.inf), where=upper > 0)
        low = np.divide(y, lower, out=np.full_like(y, np.inf), where=lower > 0)
        strength = np.minimum(1 + self.color_saturation, np.minimum(high, low))
        return np.clip(y[..., None] + strength[..., None] * chroma, 0, 255)

    def cells(self, grid):
        """Return shared glyphs and float foreground RGB for every output."""
        grid = self._color_grid(np.asarray(grid, dtype=np.float64))
        raw_y = self._luminance(grid)
        y = self._tone(raw_y)
        strength = self.glyph_correction
        if strength != 1:
            symbols = self.palette[self._indices(y / 255, self.levels)]
            if not strength:
                return symbols, grid
            baseline_ink = self._coverage[symbols.view(np.uint32) - 32]
            target = baseline_ink / self._max_ink
            target += strength * (y / 255 - target)
        else:
            target = y / 255
        minimum = None
        source_target = None
        if self.color == "source":
            source_target = self._max_ink * y
            if strength != 1:
                baseline = baseline_ink * raw_y
                source_target = baseline + strength * (source_target - baseline)
            peak = np.maximum(np.maximum(grid[..., 0], grid[..., 1]), grid[..., 2])
            required = np.divide(source_target * peak, 255 * raw_y,
                                 out=np.zeros_like(raw_y), where=raw_y > 0)
            minimum = np.minimum(np.searchsorted(self._ink, required), len(self._ink) - 1)
        if self.color == "source" and strength == 1 and not np.any((raw_y <= 0) & (y > 0)):
            # Full source correction needs only enough ink to carry the target.
            positions = minimum
        else:
            positions = self._indices(target, self._ink / self._max_ink, minimum)
        symbols = self.palette[self._ink_order[positions]]
        if self.color == "source":
            ink = self._ink[positions]
            gain = np.divide(source_target, ink * raw_y, out=np.zeros_like(raw_y),
                             where=(ink > 0) & (raw_y > 0))
            limit = np.divide(255, peak, out=np.zeros_like(peak), where=peak > 0)
            grid = np.clip(grid * np.minimum(gain, limit)[..., None], 0, 255)
        return symbols, grid

    def symbols(self, grid):
        return self.cells(grid)[0]

    def render(self, rgb, columns, rows):
        grid = self.sample(rgb, columns, rows)
        return self.cells(grid)

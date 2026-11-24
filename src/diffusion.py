"""Sequential multi-level error diffusion; optional Cython acceleration.

Matrices follow https://surma.dev/things/ditherpunk/; Riemersma's bounded
history follows https://www.compuphase.com/riemer.htm (32 samples, ratio 1/8).
The newest raw (original minus quantized) error has weight 1, the oldest 1/8;
weights are not normalized and history does not include its own feedback.
"""

from functools import lru_cache
from bisect import bisect_right

import numpy as np

try:
    from _diffusion import quantize as _native_quantize
except ImportError:
    _native_quantize = None


_KERNELS = {
    "simple": ((0, 1, .5), (1, 0, .5)),
    "floyd-steinberg": ((0, 1, 7/16), (1, -1, 3/16), (1, 0, 5/16), (1, 1, 1/16)),
    "jarvis": ((0, 1, 7/48), (0, 2, 5/48),
                            (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48),
                            (1, 1, 5/48), (1, 2, 3/48), (2, -2, 1/48),
                            (2, -1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)),
    "atkinson": ((0, 1, 1/8), (0, 2, 1/8), (1, -1, 1/8),
                 (1, 0, 1/8), (1, 1, 1/8), (2, 0, 1/8)),
}
MODES = (*_KERNELS, "riemersma")
_HISTORY_WEIGHTS = (1/8) ** (np.arange(31, -1, -1, dtype=np.float64) / 31)


@lru_cache(maxsize=8)
def _hilbert_path(rows, columns):
    """Crop a Hilbert square to a rectangle, pruning invisible quadrants."""
    size = 1 << (max(rows, columns, 1) - 1).bit_length()
    points = np.empty((rows * columns, 2), dtype=np.intp)
    coordinates = memoryview(points)
    count = 0

    def visit(x, y, xi, xj, yi, yj, side):
        nonlocal count
        xs = (x, x + xi, x + yi, x + xi + yi)
        ys = (y, y + xj, y + yj, y + xj + yj)
        if min(xs) >= columns or max(xs) <= 0 or min(ys) >= rows or max(ys) <= 0:
            return
        if side == 1:
            px, py = x + (xi + yi) // 2, y + (xj + yj) // 2
            if 0 <= px < columns and 0 <= py < rows:
                coordinates[count, 0] = py
                coordinates[count, 1] = px
                count += 1
            return
        a, b, c, d = xi // 2, xj // 2, yi // 2, yj // 2
        half = side // 2
        visit(x, y, c, d, a, b, half)
        visit(x + a, y + b, a, b, c, d, half)
        visit(x + a + c, y + b + d, a, b, c, d, half)
        visit(x + a + yi, y + b + yj, -c, -d, -a, -b, half)

    visit(0, 0, size, 0, 0, size, size)
    coordinates.release()
    points.flags.writeable = False
    return points


def quantize(values, levels, mode, minimum=None):
    """Return nearest-level indices, propagating error only to future cells.

    Input arrays are never modified. Out-of-image weights are discarded.
    Optional minimum indices constrain selection before propagating error.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown diffusion mode: {mode}")
    work = np.array(values, dtype=np.float64, order="C", copy=True)
    palette = np.ascontiguousarray(levels, dtype=np.float64)
    if work.ndim != 2 or palette.ndim != 1 or not len(palette):
        raise ValueError("Expected a 2D image and a nonempty 1D palette")
    if not np.isfinite(work).all() or not np.isfinite(palette).all() or np.any(np.diff(palette) < 0):
        raise ValueError("Values must be finite and palette levels sorted")
    if minimum is None:
        lower = np.zeros(work.shape, dtype=np.intp)
    else:
        lower = np.ascontiguousarray(minimum, dtype=np.intp)
        if lower.shape != work.shape or np.any(lower < 0) or np.any(lower >= len(palette)):
            raise ValueError("Minimum indices must match the image and palette")
    midpoints = np.ascontiguousarray((palette[:-1] + palette[1:]) / 2)
    kernel = np.asarray(_KERNELS.get(mode, ()), dtype=np.float64).reshape(-1, 3)
    path = _hilbert_path(*work.shape) if mode == "riemersma" else np.empty((0, 2), dtype=np.intp)
    if _native_quantize is not None:
        return _native_quantize(work, palette, midpoints, kernel, path, _HISTORY_WEIGHTS, lower)
    result = np.empty(work.shape, dtype=np.intp)
    if mode == "riemersma":
        history = np.zeros(32)
        for y, x in path:
            adjusted = work[y, x]
            for error, weight in zip(history, _HISTORY_WEIGHTS):
                adjusted += error * weight
            index = max(bisect_right(midpoints, adjusted), lower[y, x])
            result[y, x] = index
            history[:-1] = history[1:]
            history[-1] = work[y, x] - palette[index]
        return result
    rows, columns = work.shape
    for y in range(rows):
        for x in range(columns):
            index = max(bisect_right(midpoints, work[y, x]), lower[y, x])
            result[y, x] = index
            error = work[y, x] - palette[index]
            for dy, dx, weight in _KERNELS[mode]:
                ny, nx = y + dy, x + dx
                if ny < rows and 0 <= nx < columns:
                    work[ny, nx] += error * weight
    return result

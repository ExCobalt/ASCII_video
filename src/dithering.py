"""Static, centered threshold fields for ASCII palette quantization."""

from functools import lru_cache

import numpy as np

from resources import resource_path


DITHER_MODES = ("none", "bayer", "bayer8", "clustered", "noise", "blue-noise")
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                  [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64)
_BAYER8 = np.block([[4 * BAYER, 4 * BAYER + 2],
                    [4 * BAYER + 3, 4 * BAYER + 1]])
# Orthogonal clustered-dot ranks (ImageMagick thresholds.xml, h4x4o).
# https://github.com/ImageMagick/ImageMagick/blob/main/config/thresholds.xml
# Adapted from 1-based to 0-based ranks; see THIRD_PARTY_NOTICES.md for the license.
_CLUSTERED = np.array([[6, 12, 10, 3], [11, 15, 13, 7],
                       [9, 14, 5, 1], [4, 8, 2, 0]], dtype=np.float64)
# A shuffled uniform rank tile avoids short ordered patterns and frame flicker.
# Its fixed seed and explicit generator make resizing independent of RNG state.
_NOISE = np.random.Generator(np.random.PCG64(0)).permutation(128 * 128).reshape(128, 128)
_FIELDS = {name: (rank + .5) / rank.size - .5 for name, rank in
           (("bayer", BAYER), ("bayer8", _BAYER8),
            ("clustered", _CLUSTERED), ("noise", _NOISE))}


@lru_cache(maxsize=1)
def _blue_noise():
    path = resource_path("blue-noise.npy")
    if path is None:
        raise RuntimeError("Bundled blue-noise rank array is missing")
    ranks = np.load(path, allow_pickle=False)
    if (ranks.shape != (64, 64) or ranks.dtype != np.uint16
            or not np.array_equal(np.sort(ranks, axis=None), np.arange(4096))):
        raise RuntimeError("Bundled blue-noise rank array is invalid")
    return (ranks.astype(np.float64) + .5) / ranks.size - .5


def thresholds(shape, mode):
    """Return a field anchored at the grid origin, unchanged across frames.

    Values are offsets in palette steps; the caller caches its current shape.
    All modes have zero mean over a complete tile, but partial tiles need not.
    """
    if mode == "none":
        return np.zeros(shape, dtype=np.float64)
    if mode == "blue-noise":
        field = _blue_noise()
    else:
        try:
            field = _FIELDS[mode]
        except KeyError:
            raise ValueError(f"Unknown dither mode: {mode}") from None
    rows, columns = shape
    if mode == "bayer":
        # Repeating the small tile avoids a grid-sized advanced-index gather.
        return np.tile(field, ((rows + 3) // 4, (columns + 3) // 4))[:rows, :columns]
    return field[np.arange(rows)[:, None] % field.shape[0],
                 np.arange(columns)[None, :] % field.shape[1]]

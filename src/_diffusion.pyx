# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""GIL-free sequential error diffusion on the small ASCII cell grid."""

import numpy as np


cdef Py_ssize_t nearest(double value, const double[::1] midpoints) noexcept nogil:
    cdef Py_ssize_t lo = 0, hi = midpoints.shape[0], middle
    while lo < hi:
        middle = (lo + hi) // 2
        if value < midpoints[middle]:
            hi = middle
        else:
            lo = middle + 1
    return lo


def quantize(double[:, ::1] work, const double[::1] levels,
             const double[::1] midpoints, const double[:, ::1] kernel,
             const Py_ssize_t[:, ::1] path, const double[::1] weights,
             const Py_ssize_t[:, ::1] minimum):
    cdef Py_ssize_t rows = work.shape[0], columns = work.shape[1]
    if (levels.shape[0] < 1 or midpoints.shape[0] != levels.shape[0] - 1
            or kernel.shape[1] != 3 or path.shape[1] != 2 or weights.shape[0] != 32
            or minimum.shape[0] != rows or minimum.shape[1] != columns):
        raise ValueError("Invalid diffusion buffer shape")
    if np.any(np.asarray(minimum) < 0) or np.any(np.asarray(minimum) >= levels.shape[0]):
        raise ValueError("Invalid minimum palette index")
    if path.shape[0]:
        coordinates = np.asarray(path)
        if (np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] >= rows)
                or np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] >= columns)):
            raise ValueError("Invalid diffusion path coordinate")
    taps = np.asarray(kernel)
    if (not np.isfinite(taps).all() or np.any(taps[:, 0] < 0) or np.any(taps[:, 0] > 2)
            or np.any(taps[:, 1] < -2) or np.any(taps[:, 1] > 2)):
        raise ValueError("Invalid diffusion kernel offset")
    output = np.empty((rows, columns), dtype=np.intp)
    cdef Py_ssize_t[:, ::1] result = output
    cdef Py_ssize_t y, x, ny, nx, k, i, index
    cdef double value, error, history[64]
    cdef Py_ssize_t head = 0
    with nogil:
        if path.shape[0]:
            for k in range(64):
                history[k] = 0
            for i in range(path.shape[0]):
                y, x = path[i, 0], path[i, 1]
                value = work[y, x]
                for k in range(32):
                    value += history[head + k] * weights[k]
                index = nearest(value, midpoints)
                if index < minimum[y, x]:
                    index = minimum[y, x]
                result[y, x] = index
                error = work[y, x] - levels[index]
                # Mirror the ring to retain contiguous oldest-to-newest sums.
                history[head] = error
                history[head + 32] = error
                head = (head + 1) & 31
        else:
            for y in range(rows):
                for x in range(columns):
                    index = nearest(work[y, x], midpoints)
                    if index < minimum[y, x]:
                        index = minimum[y, x]
                    result[y, x] = index
                    error = work[y, x] - levels[index]
                    for k in range(kernel.shape[0]):
                        ny = y + <Py_ssize_t>kernel[k, 0]
                        nx = x + <Py_ssize_t>kernel[k, 1]
                        if ny < rows and nx >= 0 and nx < columns:
                            work[ny, nx] += error * kernel[k, 2]
    return output

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from functools import lru_cache
from libc.math cimport sin
from libc.stdint cimport uint8_t, int32_t
from libc.string cimport memcpy
import numpy as np

def compose(const unsigned int[:, :] codes, const unsigned char[:, :, :] colors,
            const unsigned char[:, :, ::1] atlas, bint planar=False):
    cdef Py_ssize_t rows=codes.shape[0], cols=codes.shape[1]
    cdef Py_ssize_t h=atlas.shape[1], w=atlas.shape[2]
    if atlas.shape[0] != 95 or colors.shape[0] != rows or colors.shape[1] != cols or colors.shape[2] != 3:
        raise ValueError('invalid glyph/color shape')
    if np.any(np.asarray(codes) < 32) or np.any(np.asarray(codes) > 126):
        raise ValueError('invalid ASCII code')
    if planar:
        pixels=np.empty((3,rows*h,cols*w),dtype=np.uint8)
    else:
        pixels=np.empty((rows*h,cols*w,3),dtype=np.uint8)
    cdef unsigned char[:, :, ::1] result=pixels
    cdef Py_ssize_t row,col,y,x,yy,xx,glyph
    cdef unsigned int coverage,r,g,b
    with nogil:
        for row in range(rows):
            for y in range(h):
                yy=row*h+y
                for col in range(cols):
                    glyph=codes[row,col]-32
                    r=colors[row,col,0]
                    g=colors[row,col,1]
                    b=colors[row,col,2]
                    for x in range(w):
                        coverage=atlas[glyph,y,x]
                        xx=col*w+x
                        if planar:
                            result[0,yy,xx]=(coverage*r+127)//255
                            result[1,yy,xx]=(coverage*g+127)//255
                            result[2,yy,xx]=(coverage*b+127)//255
                        else:
                            result[yy,xx,0]=(coverage*r+127)//255
                            result[yy,xx,1]=(coverage*g+127)//255
                            result[yy,xx,2]=(coverage*b+127)//255
    return pixels



def compose_planes(codes, colors, atlas):
    """Contiguous channels for Pillow's independent LANCZOS passes."""
    return compose(codes, colors, atlas, True)


def compose_planes_sparse(const unsigned int[:, :] codes, const unsigned char[:, :, :] colors,
                    const unsigned char[:, :, ::1] atlas):
    cdef Py_ssize_t rows=codes.shape[0], cols=codes.shape[1]
    cdef Py_ssize_t h=atlas.shape[1], w=atlas.shape[2]
    if atlas.shape[0] != 95 or colors.shape[0] != rows or colors.shape[1] != cols or colors.shape[2] != 3:
        raise ValueError('invalid glyph/color shape')
    if np.any(np.asarray(codes) < 32) or np.any(np.asarray(codes) > 126):
        raise ValueError('invalid ASCII code')
    pixels=np.zeros((3,rows*h,cols*w),dtype=np.uint8)
    cdef unsigned char[:, :, ::1] result=pixels
    cdef Py_ssize_t row,col,y,x,yy,xx,glyph
    cdef unsigned int coverage,r,g,b
    cdef unsigned char *pr
    cdef unsigned char *pg
    cdef unsigned char *pb
    cdef const unsigned char *mask
    if rows == 0 or cols == 0 or h == 0 or w == 0:
        return pixels
    with nogil:
        for row in range(rows):
            for y in range(h):
                yy=row*h+y
                pr=&result[0,yy,0]
                pg=&result[1,yy,0]
                pb=&result[2,yy,0]
                for col in range(cols):
                    glyph=codes[row,col]-32
                    mask=&atlas[glyph,y,0]
                    r=colors[row,col,0]
                    g=colors[row,col,1]
                    b=colors[row,col,2]
                    xx=col*w
                    for x in range(w):
                        coverage=mask[x]
                        if coverage:
                            pr[xx+x]=(coverage*r+127)//255
                            pg[xx+x]=(coverage*g+127)//255
                            pb[xx+x]=(coverage*b+127)//255
    return pixels


def make_source_tinted_atlas(const unsigned char[:, :, ::1] atlas):
    """Immutable exact RGB8 coverage table; the caller owns its cache lifetime."""
    if atlas.shape[0] != 95:
        raise ValueError('invalid atlas shape')
    coverage = np.asarray(atlas).astype(np.uint16)
    values = np.arange(256, dtype=np.uint16)
    result = ((coverage[:, None] * values[None, :, None, None] + 127) // 255).astype(np.uint8)
    result.flags.writeable = False
    return result


def compose_source_tinted(const unsigned int[:, :] codes, const unsigned char[:, :, :] colors,
                          const unsigned char[:, :, :, ::1] tinted):
    cdef Py_ssize_t rows=codes.shape[0], cols=codes.shape[1]
    cdef Py_ssize_t h=tinted.shape[2], w=tinted.shape[3]
    if tinted.shape[0] != 95 or tinted.shape[1] != 256 or colors.shape[0] != rows or colors.shape[1] != cols or colors.shape[2] != 3:
        raise ValueError('invalid glyph/color shape')
    if np.any(np.asarray(codes) < 32) or np.any(np.asarray(codes) > 126):
        raise ValueError('invalid ASCII code')
    pixels = np.empty((3, rows*h, cols*w), dtype=np.uint8)
    cdef unsigned char[:, :, ::1] result=pixels
    cdef Py_ssize_t row,col,y,yy,xx,glyph
    cdef unsigned int r,g,b
    if rows == 0 or cols == 0 or h == 0 or w == 0:
        return pixels
    with nogil:
        for row in range(rows):
            for y in range(h):
                yy=row*h+y
                for col in range(cols):
                    glyph=codes[row,col]-32
                    r=colors[row,col,0]
                    g=colors[row,col,1]
                    b=colors[row,col,2]
                    xx=col*w
                    if w == 8:
                        # Literal lengths let MSVC replace each copy with one load/store.
                        memcpy(&result[0,yy,xx], &tinted[glyph,r,y,0], 8)
                        memcpy(&result[1,yy,xx], &tinted[glyph,g,y,0], 8)
                        memcpy(&result[2,yy,xx], &tinted[glyph,b,y,0], 8)
                    else:
                        memcpy(&result[0,yy,xx], &tinted[glyph,r,y,0], w)
                        memcpy(&result[1,yy,xx], &tinted[glyph,g,y,0], w)
                        memcpy(&result[2,yy,xx], &tinted[glyph,b,y,0], w)
    return pixels


# Fixed-ratio arithmetic follows Pillow 12.3.0 src/libImaging/Resample.c:
# https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
# Copyright Secret Labs AB, Fredrik Lundh and contributors, Jeffrey 'Alex' Clark
# and contributors. Full attribution and MIT-CMU notice:
# THIRD_PARTY_NOTICES.md (Pillow Resample).
cdef double _source_sinc(double x) noexcept nogil:
    if x == 0.0:
        return 1.0
    x = x * 3.14159265358979323846264338327950288
    return sin(x) / x


cdef double _source_lanczos(double x) noexcept nogil:
    if -3.0 <= x < 3.0:
        return _source_sinc(x) * _source_sinc(x / 3.0)
    return 0.0


cdef uint8_t _source_clip(int32_t value) noexcept nogil:
    value = value >> 22
    if value < 0:
        return 0
    if value > 255:
        return 255
    return <uint8_t>value


def _source_coefficients(int source_size, int start, int end, int output_size):
    """Fixed support4, Q22 coefficients; preserve C truncation and double order."""
    cdef double scale = <double>(end - start) / output_size
    cdef double inverse = 1.0 / scale
    cdef double center, total, value
    cdef double weights[9]
    cdef int pixel, tap, first, last, count
    cdef int32_t[:, ::1] bounds, coefficients
    if scale != 4.0 / 3.0:
        raise ValueError("Only fixed 4:3 reduction is supported")
    bounds_array = np.empty((output_size, 2), dtype=np.int32)
    coefficients_array = np.zeros((output_size, 9), dtype=np.int32)
    bounds, coefficients = bounds_array, coefficients_array
    for pixel in range(output_size):
        center = start + (pixel + 0.5) * scale
        first = <int>(center - 4.0 + 0.5)
        last = <int>(center + 4.0 + 0.5)
        if first < 0:
            first = 0
        if last > source_size:
            last = source_size
        count = last - first
        bounds[pixel, 0], bounds[pixel, 1] = first, count
        total = 0.0
        for tap in range(count):
            value = _source_lanczos((tap + first - center + 0.5) * inverse)
            weights[tap] = value
            total += value
        for tap in range(count):
            value = weights[tap]
            if total != 0.0:
                value /= total
            if value < 0.0:
                coefficients[pixel, tap] = <int32_t>(-0.5 + value * (1 << 22))
            else:
                coefficients[pixel, tap] = <int32_t>(0.5 + value * (1 << 22))
    bounds_array.flags.writeable = coefficients_array.flags.writeable = False
    return bounds_array, coefficients_array


@lru_cache(maxsize=5)
def _source_prepare(int top):
    """Cache the measured tables lazily for the four actual source boxes."""
    if top not in (0, 270, 540, 810):
        raise ValueError("Expected one of four 270-row bands")
    horizontal = _source_coefficients(2560, 0, 2560, 1920)
    vertical = _source_coefficients(1440, top * 4 // 3, (top + 270) * 4 // 3, 270)
    return (top, 270, *horizontal, *vertical)


def resize_source_band(const uint8_t[:, :] plane, int top=0):
    """Exact LANCZOS for one 1920x270 output band of a 2560x1440 source plane."""
    cdef const int32_t[:, ::1] xb, xk, yb, yk
    cdef uint8_t[:, ::1] intermediate, output
    cdef int height=270
    cdef int first_row, last_row, row, column, first, count, tap
    cdef int32_t accumulator
    cdef int32_t row_sums[1920]
    cdef int32_t coefficient
    cdef const uint8_t *scanline
    cdef uint8_t *output_line
    if plane.shape[0] != 1440 or plane.shape[1] != 2560:
        raise ValueError("Expected a 1440x2560 grayscale plane")
    prepared = _source_prepare(top)
    xb, xk, yb, yk = prepared[2:]
    first_row = yb[0, 0]
    last_row = yb[height - 1, 0] + yb[height - 1, 1]
    temporary_array = np.empty((last_row - first_row, 1920), dtype=np.uint8)
    result = np.empty((height, 1920), dtype=np.uint8)
    intermediate, output = temporary_array, result
    with nogil:
        for row in range(last_row - first_row):
            for column in range(1920):
                first, count = xb[column, 0], xb[column, 1]
                accumulator = 1 << 21
                for tap in range(count):
                    accumulator += plane[row + first_row, first + tap] * xk[column, tap]
                intermediate[row, column] = _source_clip(accumulator)
        for row in range(height):
            first, count = yb[row, 0] - first_row, yb[row, 1]
            for column in range(1920):
                row_sums[column] = 1 << 21
            for tap in range(count):
                coefficient = yk[row, tap]
                scanline = &intermediate[first + tap, 0]
                for column in range(1920):
                    row_sums[column] += scanline[column] * coefficient
            output_line = &output[row, 0]
            for column in range(1920):
                output_line[column] = _source_clip(row_sums[column])
    return result


def resize_source_band_into(const uint8_t[:, :] plane, int top,
                    uint8_t[:, :, :] destination, int channel):
    """Same fixed Q22 stages, writing final bytes into a disjoint RGB region."""
    cdef const int32_t[:, ::1] xb, xk, yb, yk
    cdef uint8_t[:, ::1] intermediate
    cdef int first_row, last_row, row, column, first, count, tap
    cdef int32_t accumulator, coefficient
    cdef int32_t row_sums[1920]
    cdef const uint8_t *scanline
    if plane.shape[0] != 1440 or plane.shape[1] != 2560:
        raise ValueError("Expected a 1440x2560 grayscale plane")
    if (destination.shape[0] != 1080 or destination.shape[1] != 1920
            or destination.shape[2] != 3 or channel < 0 or channel > 2):
        raise ValueError("Expected a 1080x1920x3 writable destination and RGB channel")
    prepared = _source_prepare(top)
    xb, xk, yb, yk = prepared[2:]
    first_row = yb[0, 0]
    last_row = yb[269, 0] + yb[269, 1]
    temporary_array = np.empty((last_row - first_row, 1920), dtype=np.uint8)
    intermediate = temporary_array
    with nogil:
        for row in range(last_row - first_row):
            for column in range(1920):
                first, count = xb[column, 0], xb[column, 1]
                accumulator = 1 << 21
                for tap in range(count):
                    accumulator += plane[row + first_row, first + tap] * xk[column, tap]
                intermediate[row, column] = _source_clip(accumulator)
        for row in range(270):
            first, count = yb[row, 0] - first_row, yb[row, 1]
            for column in range(1920):
                row_sums[column] = 1 << 21
            for tap in range(count):
                coefficient = yk[row, tap]
                scanline = &intermediate[first + tap, 0]
                for column in range(1920):
                    row_sums[column] += scanline[column] * coefficient
            for column in range(1920):
                destination[top+row, column, channel] = _source_clip(row_sums[column])

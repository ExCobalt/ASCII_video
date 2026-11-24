"""Streaming ASCII MP4/WebM export with timestamped AAC/Opus audio."""

from contextlib import closing, nullcontext
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import errno
import math
import os
from pathlib import Path
import tempfile
import sys
import time

import av
import numpy as np

from cli_options import DEFAULTS, EXPORT_QUALITIES, VIDEO_CODECS, export_format, ExportLimitError
from media import normalize_audio_timing, open_media, probe, rotation_of, timestamp, video_intervals
from raster import Rasterizer
from rendering import Renderer, fit_grid, rotate_rgb

FPS = 30
AUDIO_RATE = 48000
CHUNK = 1024
MAX_EXPORT_SECONDS = 6 * 60 * 60
MAX_GAP_SECONDS = 60


def _check_export_span(start, end, allow_long_export=False):
    if allow_long_export:
        return
    if end > MAX_EXPORT_SECONDS:
        raise ExportLimitError("Длительность экспорта превышает 6 часов. Это может создать очень большой файл.")
    if end - start > MAX_GAP_SECONDS:
        raise ExportLimitError("Разрыв временных меток превышает 60 секунд. Это может создать очень большой файл.")


def _rgb_frame(image, *, planar=False):
    owner = getattr(image, "base", None)
    if (planar and image.dtype == np.uint8 and image.shape == (1080, 1920, 3)
            and image.strides == (1920, 1, 1080 * 1920) and not image.flags.writeable
            and isinstance(owner, np.ndarray) and owner.dtype == np.uint8
            and owner.shape == (3, 1080, 1920) and owner.flags.owndata
            and owner.flags.c_contiguous and not owner.flags.writeable):
        frame = av.VideoFrame(1920, 1080, "gbrp")
        for plane, channel in zip(frame.planes, (1, 2, 0), strict=True):
            target = np.ndarray((1080, 1920), dtype=np.uint8, buffer=plane,
                                strides=(plane.line_size, 1))
            np.copyto(target, image[:, :, channel])
        return frame
    # Let PyAV own alignment and padding; avoid from_ndarray's temporary bytes.
    height, width, _ = image.shape
    frame = av.VideoFrame(width, height, "rgb24")
    plane = frame.planes[0]
    rgb = np.ndarray(image.shape, dtype=np.uint8, buffer=plane,
                     strides=(plane.line_size, 3, 1))
    np.copyto(rgb, image)
    return frame


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Rendering cancelled")


def _encode_video(video, frame, encoder):
    try:
        return video.encode(frame)
    except av.FFmpegError as error:
        if encoder != "nvenc":
            raise
        raise RuntimeError(
            f"NVENC (h264_nvenc) failed for {video.width}x{video.height}. "
            "Check the NVIDIA GPU/driver and supported frame size, or select x264. "
            f"Original error: {error}"
        ) from error


def _audio_frame(data, cursor):
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data), format="fltp", layout="stereo")
    frame.sample_rate = AUDIO_RATE
    frame.pts = cursor
    frame.time_base = Fraction(1, AUDIO_RATE)
    return frame


def _normalized_audio(block, cursor, origin, stop_event=None, allow_long_export=False):
    position = (timestamp(block) - origin) * AUDIO_RATE
    start = math.floor(position + Fraction(1, 2))
    if start < cursor - 1:
        raise RuntimeError("Overlapping resampled audio timestamps")
    _check_export_span(Fraction(cursor, AUDIO_RATE), Fraction(start, AUDIO_RATE), allow_long_export)
    _check_export_span(Fraction(start, AUDIO_RATE), Fraction(start + block.samples, AUDIO_RATE), allow_long_export)
    while cursor < start:
        _check_stop(stop_event)
        count = min(CHUNK, start - cursor)
        yield _audio_frame(np.zeros((2, count), dtype=np.float32), cursor)
        cursor += count
    data = block.to_ndarray()
    offset = max(0, cursor - start)
    while offset < block.samples:
        _check_stop(stop_event)
        count = min(CHUNK, block.samples - offset)
        yield _audio_frame(data[:, offset:offset + count], cursor)
        cursor += count
        offset += count
    return cursor


def _audio_frames(info, stop_event=None, allow_long_export=False):
    if info.audio_index is None:
        return
    with open_media(info.path) as container:
        stream = container.streams[info.audio_index]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
        cursor = 0
        input_end = None
        for frame in container.decode(stream):
            _check_stop(stop_event)
            input_end, discontinuity = normalize_audio_timing(frame, input_end)
            if discontinuity:
                for block in resampler.resample(None):
                    cursor = yield from _normalized_audio(block, cursor, info.origin, stop_event, allow_long_export)
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
            for block in resampler.resample(frame):
                cursor = yield from _normalized_audio(block, cursor, info.origin, stop_event, allow_long_export)
        for block in resampler.resample(None):
            cursor = yield from _normalized_audio(block, cursor, info.origin, stop_event, allow_long_export)


def _video_frames(info):
    with open_media(info.path) as container:
        stream = container.streams[info.video_index]
        for frame, start, end, final in video_intervals(container.decode(stream), stream.average_rate):
            yield frame, start - info.origin, end - info.origin, rotation_of(frame, stream), final


def _prepared_videos(info, renderer, rasterizer, columns, rows, color, stop_event):
    """Prepare one interval ahead while the caller encodes and muxes."""
    previous_symbols = previous_colors = previous_image = None
    with closing(_video_frames(info)) as videos:
        def sample_next():
            _check_stop(stop_event)
            interval = next(videos, None)
            if interval is None:
                return None
            frame, start, end, rotation, final = interval
            # Intervals without a 30fps tick cannot be visible. Keep the final
            # image available even between ticks: audio may outlast the video.
            if not final and Fraction(math.ceil(start * FPS), FPS) >= end:
                return None, None, start, end
            rgb = rotate_rgb(frame.to_ndarray(format="rgb24"), rotation)
            _check_stop(stop_event)
            symbols, grid = renderer.render(rgb, columns, rows)
            _check_stop(stop_event)
            colors = None
            if color == "source":
                # Renderer returns float64; keep its grid intact and reuse
                # only the private rounding temporary before the RGB8 cast.
                rounded = grid + 0.5
                np.floor(rounded, out=rounded)
                np.clip(rounded, 0, 255, out=rounded)
                colors = rounded.astype(np.uint8)
            return symbols, colors, start, end

        # One extra sampled grid overlaps decoding/sampling with rasterization.
        parallel = getattr(rasterizer, "_source_sample_ahead", False) is True
        with (ThreadPoolExecutor(max_workers=1, thread_name_prefix="ascii-sample")
              if parallel else nullcontext()) as sampler:
            sampled = sampler.submit(sample_next) if parallel else None

            def prepare_next():
                nonlocal sampled, previous_symbols, previous_colors, previous_image
                _check_stop(stop_event)
                cells = sampled.result() if parallel else sample_next()
                if cells is None:
                    return None
                if parallel:
                    sampled = sampler.submit(sample_next)
                symbols, colors, start, end = cells
                if symbols is None:
                    return None, start, end
                unchanged = (previous_symbols is not None
                             and np.array_equal(symbols, previous_symbols)
                             and (color != "source" or np.array_equal(colors, previous_colors)))
                image = previous_image if unchanged else rasterizer.draw(symbols, colors)
                previous_symbols, previous_colors, previous_image = symbols, colors, image
                _check_stop(stop_event)
                return image, start, end

            try:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ascii-export") as executor:
                    pending = executor.submit(prepare_next)
                    try:
                        while True:
                            _check_stop(stop_event)
                            interval = pending.result()
                            if interval is None:
                                break
                            pending = executor.submit(prepare_next)
                            yield interval
                    finally:
                        pending.cancel()
            finally:
                # Join raster before cancelling/joining its sampler, then close
                # the source outside both executors (never while it is running).
                if sampled is not None:
                    sampled.cancel()


def _publish(temporary, output, overwrite):
    if overwrite:
        os.replace(temporary, output)
    elif os.name == "nt":
        os.rename(temporary, output)
    else:
        # Exclusive rename works on exFAT and other volumes without hard links.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        source, destination = os.fsencode(temporary), os.fsencode(output)
        if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            status = libc.renamex_np(source, destination, 0x4)  # RENAME_EXCL
        elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            status = libc.renameat2(-100, source, -100, destination, 1)  # AT_FDCWD, RENAME_NOREPLACE
        else:
            status = None
        if status == 0:
            return
        if status is not None:
            error = ctypes.get_errno()
            if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise OSError(error, os.strerror(error), output)
        # Creating the final hard link fails atomically if another writer won.
        os.link(temporary, output)
        os.unlink(temporary)


def _add_video_stream(container, video_codec, codec, encoder, quantizer, width, height):
    video = container.add_stream(video_codec, rate=FPS)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    video.codec_context.time_base = Fraction(1, FPS)
    # PyAV's default RGB -> YUV conversion is BT.601 limited.
    # Declare it so players need not guess a matrix from resolution.
    video.codec_context.colorspace = 6  # AVCOL_SPC_SMPTE170M
    video.codec_context.color_range = 1  # AVCOL_RANGE_MPEG
    if codec == "h264" and encoder == "x264":
        # File export can use lookahead and B-frames; low-latency
        # tuning wastes bitrate here. Keep fine glyph strokes at a
        # size/quality compromise.
        video.options = {"preset": "fast", "crf": quantizer}
    elif codec == "h264":
        video.options = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": quantizer}
        video.bit_rate = 0
        video.codec_context.max_b_frames = 0
    elif codec == "hevc":
        video.options = {"preset": "fast", "crf": quantizer,
                         "x265-params": "log-level=error"}
        video.codec_context.codec_tag = "hvc1"
    elif codec == "vp9":
        video.options = {"crf": quantizer, "deadline": "good", "cpu-used": "4"}
        video.bit_rate = 0
    else:
        video.options = {"preset": "8", "crf": quantizer, "svtav1-params": "lp=4"}
        video.bit_rate = 0
    return video


def _add_audio_stream(container, audio_codec, container_name):
    audio_stream = container.add_stream(audio_codec, rate=AUDIO_RATE)
    audio_stream.layout = "stereo"
    audio_stream.bit_rate = 192000
    audio_stream.codec_context.format = "fltp" if container_name == "mp4" else "flt"
    audio_stream.codec_context.time_base = Fraction(1, AUDIO_RATE)
    return audio_stream


def render(options, *, stop_event=None, progress=None, on_duration=None):
    _check_stop(stop_event)
    encoder = getattr(options, "encoder", "x264")
    container_name, codec = export_format(options.output, getattr(options, "container", "mp4"),
                                          getattr(options, "codec", None), encoder, options.resolution)
    quality = getattr(options, "quality", DEFAULTS["quality"])
    if quality not in EXPORT_QUALITIES:
        raise ValueError("Quality must be compact, balanced or high")
    backend = encoder if codec == "h264" else codec
    profile = EXPORT_QUALITIES[quality]
    quantizer = profile[backend]
    video_codec = "h264_nvenc" if encoder == "nvenc" else VIDEO_CODECS[codec]
    audio_codec = "aac" if container_name == "mp4" else "libopus"
    input_path, output = Path(options.input).resolve(), Path(options.output).resolve()
    if input_path == output:
        raise ValueError("Input and output files must be different")
    if output.exists() and not options.overwrite:
        raise ValueError("Output exists; use --overwrite to replace it")
    width, height = options.resolution
    if min(width, height) < 2 or width % 2 or height % 2:
        raise ValueError("Output dimensions must be positive even integers")
    info = probe(input_path, audio=not options.no_audio)
    allow_long_export = getattr(options, "allow_long_export", False)
    if info.duration is not None:
        _check_export_span(info.duration, info.duration, allow_long_export)
    for name in (video_codec, *(() if info.audio_index is None else (audio_codec,))):
        try:
            av.Codec(name, "w")
        except av.codec.codec.UnknownCodecError as error:
            raise RuntimeError(f"Encoder {name} is unavailable in this PyAV build") from error
    if on_duration is not None:
        on_duration(info.duration)
    renderer = Renderer(options.palette, options.gamma, options.contrast, options.dither,
                        brightness=getattr(options, "brightness", 0.0), color=options.color,
                        color_saturation=getattr(options, "color_saturation", 0.0),
                        glyph_correction=getattr(options, "glyph_correction", 0.0))
    rasterizer = Rasterizer(options.palette, options.color, width, height)
    # The atlas cell size is a rasterization detail, not an ASCII density cap.
    # Only limit cells to output pixels; Rasterizer scales the glyph image.
    columns, rows, _, _ = fit_grid(options.columns, None, width + 1,
                                 height + 1, 0.5, info.aspect)
    # Enable the 1080p/source path only for exact 4:3 strips
    # with enough CPU concurrency.
    parallel_strips = (options.color == "source" and (width, height, columns, rows) == (1920, 1080, 320, 90)
                       and (os.cpu_count() or 1) >= 12)
    rasterizer._source_parallel_strips = parallel_strips
    rasterizer._source_sample_ahead = parallel_strips
    rasterizer._source_fast_raster = parallel_strips and codec != "vp9"
    rasterizer._source_planar_export = parallel_strips and codec != "vp9"
    rasterizer._source_direct_resize = parallel_strips and codec == "h264" and encoder == "nvenc"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=f".{container_name}", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ThreadPoolExecutor(max_workers=12 if parallel_strips else 3, thread_name_prefix="ascii-resize") as resize_pool, \
                closing(_prepared_videos(info, renderer, rasterizer, columns, rows,
                                     options.color, stop_event)) as videos, closing(_audio_frames(info, stop_event, allow_long_export)) as audios:
            rasterizer._resize_pool = resize_pool
            interval, audio = next(videos, None), next(audios, None)
            if interval is None:
                raise RuntimeError("Video stream contains no frames")
            _check_export_span(0, interval[1], allow_long_export)
            _check_export_span(interval[1], interval[2], allow_long_export)
            video_end, audio_end = interval[2], Fraction(0)
            last = previous = frame = None
            image = np.zeros((height, width, 3), dtype=np.uint8)
            mux_options = {"options": {"avoid_negative_ts": "disabled"}} if container_name == "webm" else {}
            with av.open(str(temporary), "w", format=container_name, **mux_options) as container:
                video = _add_video_stream(container, video_codec, codec, encoder, quantizer, width, height)
                audio_stream = (_add_audio_stream(container, audio_codec, container_name)
                                if info.audio_index is not None else None)
                index = 0
                last_report = time.monotonic() - 1
                while True:
                    _check_stop(stop_event)
                    tick = Fraction(index, FPS)
                    while interval is not None and tick >= interval[2]:
                        _check_stop(stop_event)
                        last = interval
                        interval = next(videos, None)
                        if interval is not None:
                            _check_export_span(interval[1], interval[2], allow_long_export)
                            video_end = interval[2]
                    if audio is not None and timestamp(audio) <= tick:
                        audio_end = Fraction(audio.pts + audio.samples, AUDIO_RATE)
                        container.mux(audio_stream.encode(audio))
                        audio = next(audios, None)
                        continue
                    if interval is None and audio is None and tick >= max(video_end, audio_end):
                        break
                    _check_export_span(tick, tick + Fraction(1, FPS), allow_long_export)
                    active = interval if interval is not None and tick >= interval[1] else last
                    if active is not None and active is not previous:
                        if image is not active[0]:
                            frame = None
                        image = active[0]
                        previous = active
                    if frame is None:
                        frame = _rgb_frame(image, planar=rasterizer._source_planar_export)
                    elif frame.format.name != "yuv420p":
                        # The first encode initializes PyAV's shared converter.
                        # Only a held image needs a reusable YUV frame; fresh
                        # images keep the ordinary RGB encode path.
                        frame = video.codec_context.reformatter.reformat(
                            frame, format="yuv420p", threads=video.codec_context.thread_count)
                    frame.pts, frame.time_base = index, Fraction(1, FPS)
                    container.mux(_encode_video(video, frame, encoder))
                    index += 1
                    now = time.monotonic()
                    if now - last_report >= 1:
                        if progress is not None:
                            progress(index, float(tick))
                        else:
                            print(f"ascii-video: rendered {index} frames, media {float(tick):.2f}s", file=sys.stderr)
                        last_report = now
                if audio_stream is not None:
                    container.mux(audio_stream.encode(None))
                container.mux(_encode_video(video, None, encoder))
        _check_stop(stop_event)
        _publish(temporary, output, options.overwrite)
        if progress is not None:
            progress(index, float(Fraction(index, FPS)))
        else:
            print(f"ascii-video: saved {output}", file=sys.stderr)
    finally:
        temporary.unlink(missing_ok=True)

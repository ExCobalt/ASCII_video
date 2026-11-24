"""Still frames and playback using the export renderer; optional playback audio."""

from dataclasses import dataclass
from contextlib import ExitStack, nullcontext
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from media import frame_at, probe
from raster import Rasterizer
from rendering import Renderer, fit_grid


@dataclass(frozen=True)
class Preview:
    original: Image.Image
    converted: Image.Image
    position: float
    duration: float | None
    columns: int
    rows: int
    frame_size: tuple[int, int]
    viewport: tuple[float, float, float, float]
    original_viewport: tuple[float, float, float, float]
    full_frame: Image.Image
    input_path: Path


def save_preview_frame(preview, output, *, stop_event=None, publish_lock=None):
    """Publish a lossless full raster, preserving the old file on failure."""
    output = Path(output).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("Выберите файл с расширением .png")
    if output == preview.input_path.resolve():
        raise ValueError("Нельзя перезаписать исходный файл")
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("PNG saving cancelled")
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".png", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            preview.full_frame.save(stream, format="PNG", compress_level=1)
        with publish_lock if publish_lock is not None else nullcontext():
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("PNG saving cancelled")
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


class FrameSaveJob:
    """One snapshot and one terminal result; no Tk calls from this worker."""
    def __init__(self, preview, output):
        self.preview, self.output = preview, output
        self.stop_event = threading.Event()
        self.publish_lock = threading.Lock()
        self.result = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="ascii-preview-png", daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        with self.publish_lock:
            self.stop_event.set()

    def _run(self):
        try:
            save_preview_frame(self.preview, self.output, stop_event=self.stop_event,
                               publish_lock=self.publish_lock)
            self.result.put(("done", str(self.output)))
        except InterruptedError:
            self.result.put(("cancelled", None))
        except Exception as error:
            self.result.put(("error", str(error)))
        finally:
            self.preview = None


def _viewport(size, bounds, zoom, center):
    width, height = size
    if zoom == 0:
        scale = min(bounds[0] / width, bounds[1] / height)
        cw, ch = bounds[0] / scale, bounds[1] / scale
        return (width - cw) / 2, (height - ch) / 2, cw, ch
    cw, ch = (math.ceil(length / zoom) for length in bounds)
    left, top = (min(max(round(c * length - span / 2), 0), length - span)
                 if length >= span else (length - span) // 2
                 for length, span, c in zip(size, (cw, ch), center))
    return left, top, cw, ch


def _detail(image, viewport, bounds, zoom):
    left, top, width, height = viewport
    image = image.crop((left, top, left + width, top + height))
    return image.resize((width * zoom, height * zoom), Image.Resampling.NEAREST).crop((0, 0, *bounds))


def _source_detail(image, size, fitted, viewport, bounds, zoom):
    # Resize only the visible source region; an 8K preview must not allocate
    # a second full-size canvas just to show a few hundred pixels.
    left, top, width, height = viewport
    ox, oy = ((length - fit) // 2 for length, fit in zip(size, fitted))
    x0, y0 = max(left, ox), max(top, oy)
    x1, y1 = min(left + width, ox + fitted[0]), min(top + height, oy + fitted[1])
    result = Image.new("RGB", bounds)
    if x1 > x0 and y1 > y0:
        box = tuple(value * length / fit for value, length, fit in zip(
            (x0 - ox, y0 - oy, x1 - ox, y1 - oy), image.size * 2, fitted * 2))
        region = image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=box)
        if zoom != 1:
            region = region.resize((region.width * zoom, region.height * zoom), Image.Resampling.NEAREST)
        result.paste(region, ((x0 - left) * zoom, (y0 - top) * zoom))
    return result


def make_preview(options, fraction=0.1, bounds=(480, 270), stop_event=None, *,
                 zoom=0, center=(.5, .5), cache=None, frame_reader=None):
    """Build a viewport; optional cache belongs to one serialized preview window."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Preview position must be between 0 and 1")
    if zoom not in (0, 1, 2) or len(center) != 2 or any(not math.isfinite(c) or not 0 <= c <= 1 for c in center):
        raise ValueError("Invalid preview zoom or center")

    def check_stop():
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Preview cancelled")

    check_stop()
    cache = {} if cache is None else cache
    path = Path(options.input).resolve()
    stat = path.stat()
    metadata_key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino,
                    options.no_audio)
    metadata = cache.get("metadata")
    if metadata is None or metadata[0] != metadata_key:
        cache.clear()
        # Still-frame generation reads audio timing without opening an audio device.
        info = probe(path, audio=not options.no_audio)
        check_stop()
        cache["metadata"] = metadata_key, info
    else:
        _, info = metadata
    source = cache.get("source")
    if source is None or source[0] != fraction:
        for key in ("source", "sample", "render", "source_image"):
            cache.pop(key, None)
        source = None
        position = fraction * info.duration if info.duration is not None else 0.0
        rgb = (frame_reader(info, position) if frame_reader is not None else
               frame_at(info, position, cancel=stop_event, allow_before_start=True))
        check_stop()
        cache["source"] = fraction, info, rgb
    else:
        _, info, rgb = source
    position = fraction * info.duration if info.duration is not None else 0.0
    check_stop()
    width, height = options.resolution
    render_key = (options.palette, options.gamma, options.contrast, options.dither,
                  options.brightness, options.color, options.color_saturation,
                  options.glyph_correction, width, height, options.columns)
    rendered = cache.get("render")
    if rendered is None or rendered[0] != render_key:
        cache.pop("render", None)
        rendered = None
        columns, rows, _, _ = fit_grid(options.columns, None, width + 1, height + 1, .5, info.aspect)
        if rgb is None:
            # Export uses literal black before the first video interval,
            # independently of the palette and tone adjustments.
            converted = Image.new("L" if options.color == "mono" else "RGB", (width, height))
        else:
            renderer = Renderer(options.palette, options.gamma, options.contrast, options.dither,
                                brightness=options.brightness, color=options.color,
                                color_saturation=options.color_saturation,
                                glyph_correction=options.glyph_correction)
            sample_key = (columns, rows)
            sampled = cache.get("sample")
            if sampled is None or sampled[0] != sample_key:
                cache.pop("sample", None)
                grid = renderer.sample(rgb, columns, rows)
                check_stop()
                sampled = sample_key, grid
                cache["sample"] = sampled
            symbols, foreground = renderer.cells(sampled[1])
            colors = np.clip(np.floor(foreground + .5), 0, 255).astype(np.uint8) if options.color == "source" else None
            check_stop()
            # Cache the actual export raster, before shrinking/cropping or H.264.
            rasterizer = Rasterizer(options.palette, options.color, width, height)
            converted = Image.fromarray(rasterizer.draw(symbols, colors))
        check_stop()
        cache["render"] = render_key, converted, columns, rows
    else:
        _, converted, columns, rows = rendered
    check_stop()
    viewport = _viewport((width, height), bounds, zoom, center)
    full_frame = converted
    resampling = Image.Resampling.LANCZOS
    converted = (_detail(converted, viewport, bounds, zoom) if zoom else
                 ImageOps.pad(converted, bounds, method=resampling, color="black"))
    if converted.mode != "RGB":
        converted = converted.convert("RGB")
    original = cache.get("source_image")
    if original is None:
        # Keep only this frame's PIL copy: panning must not copy the full RGB
        # buffer again. It is released with the sampled frame on the next seek.
        original = Image.new("RGB", (1, 1)) if rgb is None else Image.fromarray(rgb)
        check_stop()
        cache["source_image"] = original
    # Apply display aspect (including non-square pixels), as the export does.
    target = (width, height) if zoom else bounds
    ow = min(target[0], max(1, round(target[1] * info.aspect)))
    oh = min(target[1], max(1, round(ow / info.aspect)))
    if zoom:
        original = _source_detail(original, target, (ow, oh), viewport, bounds, zoom)
        viewport = viewport[:2] + (bounds[0] / zoom, bounds[1] / zoom)
        original_viewport = viewport
    else:
        original = original.resize((ow, oh), Image.Resampling.LANCZOS)
        original = ImageOps.pad(original, bounds, method=Image.Resampling.LANCZOS, color="black")
        fw = min(width, max(1, round(height * info.aspect)))
        fh = min(height, max(1, round(fw / info.aspect)))
        # The source overview fills its own thumbnail, while MP4 may have a
        # different canvas aspect. Map clicks through the source's fitted area.
        original_viewport = ((width - fw) // 2 - ((bounds[0] - ow) // 2) * fw / ow,
                             (height - fh) // 2 - ((bounds[1] - oh) // 2) * fh / oh,
                             bounds[0] * fw / ow, bounds[1] * fh / oh)
    check_stop()
    return Preview(original, converted, position, info.duration, columns, rows,
                   (width, height), viewport, original_viewport, full_frame, path)


class PreviewJob:
    def __init__(self, options, fraction, bounds, *, zoom=0, center=(.5, .5), cache=None):
        self.options, self.fraction, self.bounds = options, fraction, bounds
        self.zoom, self.center = zoom, center
        self.cache = cache
        self.stop_event = threading.Event()
        self.result = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="ascii-preview", daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.stop_event.set()

    def _run(self):
        try:
            value = make_preview(self.options, self.fraction, self.bounds, self.stop_event,
                                 zoom=self.zoom, center=self.center, cache=self.cache)
            self.result.put(("done", value))
        except InterruptedError:
            self.result.put(("cancelled", None))
        except Exception as error:
            self.result.put(("error", str(error)))

class PreviewPlaybackJob:
    """Sequential decode, latest frame only, optional DAC clock; never calls Tk."""
    def __init__(self, options, fraction, bounds, *, zoom=0, center=(.5, .5), sound=False):
        self.options, self.fraction, self.bounds = options, fraction, bounds
        self.zoom, self.center, self.sound = zoom, center, sound
        self.stop_event = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.result = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="preview-playback", daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.stop_event.set()

    def _run(self):
        from fractions import Fraction
        from media import decode_video_from, open_media, timestamp, rotation_of
        from rendering import rotate_rgb
        audio = None
        media_input = ExitStack()
        cache = {}
        try:
            info = probe(self.options.input, audio=not self.options.no_audio)
            if info.duration is None or info.duration <= 0:
                raise ValueError("Для воспроизведения предпросмотра нужна известная длительность.")
            position = self.fraction * info.duration
            container = media_input.enter_context(open_media(info.path))
            stream = container.streams[info.video_index]
            frames = decode_video_from(container, stream, info.origin + Fraction(str(position)),
                                       self.stop_event.is_set)
            pending = next(frames, None)
            current = None

            def read_frame(metadata, target):
                nonlocal pending, current
                absolute = metadata.origin + Fraction(str(target))
                while pending is not None and timestamp(pending) <= absolute:
                    current = pending
                    pending = next(frames, None)
                    if self.stop_event.is_set():
                        raise InterruptedError
                return (None if current is None else
                        rotate_rgb(current.to_ndarray(format="rgb24"), rotation_of(current, stream)))

            if self.sound and info.audio_index is not None:
                from audio import AudioPlayer
                audio = AudioPlayer(str(info.path), info.audio_index, info.origin, self.stop_event)
                audio.seek_time = position
                audio.prepare()
            first = True
            started = time.perf_counter()
            while not self.stop_event.is_set():
                tick = time.perf_counter()
                if not first:
                    position = min(info.duration, audio.time() if audio else
                                   self.fraction * info.duration + tick - started)
                value = make_preview(self.options, position / info.duration, self.bounds,
                                     self.stop_event, zoom=self.zoom, center=self.center,
                                     cache=cache, frame_reader=read_frame)
                if first:
                    started = time.perf_counter()
                    if audio is not None:
                        audio.start()
                    first = False
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(value)
                if audio is not None and audio.error:
                    raise audio.error
                if position >= info.duration:
                    break
                self.stop_event.wait(max(0, 1 / 30 - (time.perf_counter() - tick)))
            if audio is not None and audio.error:
                raise audio.error
            self.result.put(("done", None))
        except InterruptedError:
            self.result.put(("error", str(audio.error)) if audio is not None and audio.error
                            else ("cancelled", None))
        except Exception as error:
            self.result.put(("error", str(error)))
        finally:
            try:
                if audio is not None:
                    audio.close()
            finally:
                media_input.close()
                cache.clear()

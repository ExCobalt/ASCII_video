"""Coordinate video, DAC time and reversible terminal drawing."""

import os
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import threading
import time

import numpy as np

from controls import Keyboard
from media import VideoDecoder, frame_at, probe
from rendering import Renderer, fit_grid
from terminal import Terminal


class FrameResizer:
    """One resize job at a time; obsolete results never change a newer frame."""
    def __init__(self, info, renderer):
        self.info, self.renderer = info, renderer
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-resize")
        self.future = None
        self.request = None

    def _sample(self, start, size, rgb, cancel):
        if cancel.is_set():
            return None
        if rgb is None:
            rgb = frame_at(self.info, start, cancel)
        if rgb is None or cancel.is_set():
            return None
        return self.renderer.sample(rgb, size[0], size[1])

    def resize(self, frame, size, last_rgb):
        if self.request is not None:
            previous, previous_size, cancel = self.request
            if previous is not frame or previous_size != size:
                cancel.set()
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            previous, previous_size, cancel = self.request
            self.request = None
            if previous is frame and previous_size == size and not cancel.is_set():
                frame.grid = future.result()
                frame.geometry = size
        if frame.geometry == size:
            return True
        if self.future is None:
            cancel = threading.Event()
            self.request = frame, size, cancel
            self.future = self.pool.submit(self._sample, frame.start, size,
                                           last_rgb if frame.last else None, cancel)
        return False

    def cancel(self):
        if self.request is not None:
            self.request[2].set()

    def close(self):
        self.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = self.request = None


def _toggle_pause(audio, seeking_at, started, paused_at):
    if paused_at is None:
        if audio is not None and seeking_at is None:
            audio.pause()
        return started, time.perf_counter()
    if audio is not None and seeking_at is None:
        audio.resume()
    elif audio is None and seeking_at is None:
        started += time.perf_counter() - paused_at
    return started, None


def _seek_target(action, value, *, clock, terminal, timeline_span, timeline_size, duration, fps):
    if action == "seek":
        target = clock() + value
    elif action == "click":
        if timeline_span is None or duration is None:
            return None
        if timeline_size != terminal.size():
            return None
        x, y = value[0] + 1, value[1] + 1
        x1, x2, row = timeline_span
        if y != row or not x1 <= x <= x2 or x2 <= x1:
            return None
        target = duration * (x - x1) / (x2 - x1)
    else:
        return None
    target = max(0.0, target)
    if duration is not None:
        target = min(target, max(0.0, duration - 1 / fps))
    return target


def play(options):
    if not sys.stdout.isatty():
        raise RuntimeError("stdout must be a terminal")
    if os.environ.get("TERM", "").lower() == "dumb":
        raise RuntimeError("TERM=dumb does not support ANSI output")
    terminal = Terminal(options.color)
    terminal.font_scale = getattr(options, "font_scale", .8)
    terminal.show_resolution = getattr(options, "show_resolution", False)
    info = probe(options.input, audio=not options.no_audio)
    renderer = Renderer(options.palette, options.gamma, options.contrast, options.dither,
                        brightness=getattr(options, "brightness", 0.0), color=options.color,
                        color_saturation=getattr(options, "color_saturation", 0.0),
                        glyph_correction=getattr(options, "glyph_correction", 0.0))
    stop = threading.Event()
    audio = None
    audio_cursor = None
    started = None
    paused_at = None
    seeking_at = 0.0

    def clock():
        if seeking_at is not None:
            return seeking_at
        if started is None:
            return 0.0
        if audio is not None:
            return audio.time()
        return (paused_at if paused_at is not None else time.perf_counter()) - started

    def geometry():
        return fit_grid(options.width, options.height, *terminal.size(),
                        options.cell_aspect, info.aspect)

    def frame_cells(grid, *, active=True):
        if not active:
            return np.full(grid.shape[:2], " "), np.zeros(grid.shape, dtype=np.uint8)
        symbols, foreground = renderer.cells(grid)
        colors = (np.clip(np.floor(foreground + .5), 0, 255).astype(np.uint8)
                  if options.color == "source" else None)
        return symbols, colors

    def draw(symbols, colors, size):
        try:
            terminal.draw(symbols, colors, size[2], size[3])
        except ValueError:
            if geometry() == size:
                raise
            return False
        return True

    resizer = FrameResizer(info, renderer)

    video = VideoDecoder(info, renderer, geometry(), clock, stop)
    try:
        video.start(wait=False)
        if info.audio_index is not None:
            from audio import AudioPlayer
            from audio_cursor import create_audio_cursor
            audio_cursor = create_audio_cursor(str(info.path), info.audio_index, history_seconds=8)
            audio = AudioPlayer(str(info.path), info.audio_index, info.origin, stop, cursor=audio_cursor)
            audio.prepare(wait=False)
        with terminal, Keyboard() as keyboard:
            video.geometry = geometry()
            c, h, bw, bh = video.geometry
            visible = draw(np.full((h, c), " "), np.zeros((h, c, 3), dtype=np.uint8), video.geometry)
            image_size = terminal.size() if visible else None
            pending = current = None
            next_draw = 0.0
            drawn = None
            accounted = True
            timeline_span = None
            next_timeline = 0.0
            timeline_size = None
            duration = getattr(info, "duration", None)
            while True:
                if video.error:
                    raise video.error
                if audio is not None and audio.error:
                    raise audio.error
                if stop.is_set():
                    raise RuntimeError("Playback stopped unexpectedly")
                for action, value in keyboard.poll():
                    if action == "pause":
                        started, paused_at = _toggle_pause(audio, seeking_at, started, paused_at)
                        next_timeline = 0.0
                        continue
                    target = _seek_target(
                        action, value, clock=clock, terminal=terminal,
                        timeline_span=timeline_span, timeline_size=timeline_size,
                        duration=duration, fps=options.fps)
                    if target is None:
                        continue
                    seeking_at = target
                    resizer.cancel()
                    # Old PCM and decoder queues must not cross the seek boundary.
                    if audio is not None:
                        audio.close()
                        audio = None
                    video.close()
                    video = VideoDecoder(info, renderer, geometry(), clock, stop)
                    video.seek_time = target
                    video.start(wait=False)
                    if info.audio_index is not None:
                        from audio import AudioPlayer
                        audio = AudioPlayer(str(info.path), info.audio_index, info.origin, stop, cursor=audio_cursor)
                        audio.seek_time = target
                        audio.prepare(wait=False)
                    pending = current = drawn = None
                    accounted = True
                    next_draw = next_timeline = 0.0
                if seeking_at is not None:
                    if not video.ready.is_set() or (audio is not None and not audio.ready):
                        if time.perf_counter() >= next_timeline:
                            timeline_span = terminal.draw_timeline(clock(), duration, paused_at is not None)
                            timeline_size = terminal.size()
                            next_timeline = time.perf_counter() + .1
                        time.sleep(0.002)
                        continue
                    if video.error:
                        raise video.error
                    if audio is not None and audio.error:
                        raise audio.error
                    target = seeking_at
                    was_paused = paused_at is not None
                    initializing = started is None
                    started = time.perf_counter() - target
                    paused_at = time.perf_counter() if was_paused else None
                    if audio is not None:
                        if was_paused:
                            audio.pause()
                        else:
                            audio.start()
                    seeking_at = None
                    pending = current = drawn = None
                    accounted = True
                    next_draw = next_timeline = 0.0
                    if initializing and not was_paused:
                        # The normal draw path accounts for the first frame once.
                        continue
                    # Show the requested image immediately, including while paused.
                    size = geometry()
                    try:
                        pending = video.queue.get_nowait()
                    except queue.Empty:
                        pending = None
                    ready = (pending is not None and pending.start <= target
                             and resizer.resize(pending, size, video.last_rgb))
                    grid = (pending.grid if ready else
                            np.zeros((size[1], size[0], 3), dtype=float))
                    symbols, colors = frame_cells(grid, active=ready)
                    visible = draw(symbols, colors, size)
                    image_size = terminal.size() if visible and (ready or pending is None or pending.start > target) else None
                if paused_at is not None and image_size != terminal.size():
                    size = geometry()
                    video.geometry = size
                    held = current if current is not None else pending
                    held_active = held is not None and held.start <= clock()
                    if held_active:
                        grid = held.grid if resizer.resize(held, size, video.last_rgb) else None
                    else:
                        grid = np.zeros((size[1], size[0], 3), dtype=float)
                    if grid is not None:
                        symbols, colors = frame_cells(grid, active=held_active)
                        visible = draw(symbols, colors, size)
                        image_size = terminal.size() if visible else None
                        next_timeline = 0.0
                if time.perf_counter() >= next_timeline:
                    timeline_span = terminal.draw_timeline(clock(), duration, paused_at is not None)
                    timeline_size = terminal.size()
                    next_timeline = time.perf_counter() + .1
                if paused_at is not None:
                    time.sleep(0.01)
                    continue
                size = geometry()
                video.geometry = size
                now = clock()
                # Drain expired work, retaining the current frame and one future frame.
                while True:
                    if pending is None:
                        try:
                            pending = video.queue.get_nowait()
                        except queue.Empty:
                            break
                    if pending.start > now:
                        break
                    if not accounted:
                        video.skipped += 1
                    current, pending = pending, None
                    accounted = False
                    if current.end > now or current.last:
                        break
                    video.skipped += 1
                    accounted = True
                if current is not None and current.geometry != size:
                    resizer.resize(current, size, video.last_rgb)
                    drawn = None
                if (current is not None and current.geometry == size and now >= next_draw
                        and (current.end > now or current.last) and drawn is not current):
                    symbols, colors = frame_cells(current.grid)
                    after = clock()
                    if (after < current.end or current.last) and draw(symbols, colors, size):
                        drawn = current
                        if image_size != terminal.size():
                            next_timeline = 0.0
                        image_size = terminal.size()
                        if not accounted:
                            video.presented += 1
                            accounted = True
                        next_draw = after + 1 / options.fps
                        video.present_after = next_draw
                video_done = video.eof and pending is None and video.queue.empty()
                audio_end = audio.end_time if audio is not None else 0.0
                if video_done and audio_end is not None:
                    finish = max(video.end_time or 0.0, audio_end)
                    if now >= finish:
                        if not accounted:
                            video.skipped += 1
                        break
                # Event.wait() uses a coarse Windows timeout; sleep uses Python's
                # high-resolution timer. Once drawn, the image remains valid to
                # the next PTS; poll input/resize at least every 10 ms meanwhile.
                delay = .002
                if current is not None and drawn is current:
                    deadline = pending.start if pending is not None else current.end
                    if video_done and audio_end is not None:
                        deadline = finish
                    delay = min(.01, max(.002, deadline - clock()),
                                max(.002, next_timeline - time.perf_counter()))
                time.sleep(delay)
    finally:
        stop.set()
        try:
            video.close()
        finally:
            try:
                if audio is not None:
                    audio.close()
            finally:
                try:
                    resizer.close()
                finally:
                    if audio_cursor is not None:
                        audio_cursor.close()

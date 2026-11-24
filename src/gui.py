"""Small Tk front end for terminal playback and cancellable MP4 export."""

import contextlib
import io
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

from progress import clock, estimate
from resources import resource_path
from cli_options import DEFAULTS, VERSION, EXPORT_CONTAINERS, export_format, ExportLimitError, parser

EXPORT_QUALITY_LABELS = {"Компактный": "compact", "Обычный": "balanced", "Высокое": "high"}

DITHER_LABELS = {"Без дизеринга": "none", "Bayer 4×4": "bayer", "Bayer 8×8": "bayer8",
                 "Кластерный": "clustered", "Шумовой": "noise", "Синий шум": "blue-noise",
                 "Простое рассеяние": "simple", "Флойд—Стейнберг": "floyd-steinberg",
                 "Джарвис—Джудис—Нинке": "jarvis", "Аткинсон": "atkinson", "Римерсма": "riemersma"}


def round_controls(style, root):
    from PIL import Image, ImageDraw, ImageTk

    # Tk tiles the center. Use a large tile to keep the number of draws low.
    radius = max(3, round(5 * float(root.tk.call("tk", "scaling")) / (96 / 72)))
    width, height = 2 * radius + 64, 2 * radius + 8
    images = []
    for name in ("TButton", "secondary.TButton", "TEntry", "TCombobox", "TSpinbox"):
        button = name.endswith("TButton")
        states = ("disabled", "pressed", "hover focus", "hover", "focus") if button else (
            "disabled", "readonly invalid", "invalid", "readonly focus", "readonly hover",
            "focus", "hover", "readonly")
        assets = []
        for state in ("", *states):
            flags = state.split()
            fill = style.lookup(name, "background" if button else "fieldbackground", flags)
            border = fill if button and "focus" not in flags else style.lookup(
                name, "focuscolor" if button else "bordercolor", flags)
            # Opaque corners avoid Tk's slow alpha blending on Windows.
            image = Image.new("RGB", (width * 3, height * 3), style.colors.bg)
            ImageDraw.Draw(image).rounded_rectangle(
                (0, 0, width * 3 - 1, height * 3 - 1), radius=radius * 3,
                fill=fill, outline=border, width=3)
            photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS), master=root)
            images.append(photo)
            assets.append((*flags, photo) if flags else photo)
        [(original, options)] = style.layout(name)
        element = f"Ascii.{name}.{original}"
        style.element_create(element, "image", *assets,
                             border=radius, padding=1, sticky="nsew")
        options.pop("border", None)
        style.layout(name, [(element, options)])
    return images


def apply_title_theme(window):
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    with contextlib.suppress(OSError):
        get_ancestor = ctypes.windll.user32.GetAncestor
        get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
        get_ancestor.restype = wintypes.HWND
        hwnd = get_ancestor(window.winfo_id(), 2)  # GA_ROOT includes Tk's outer frame.
        enabled = wintypes.BOOL(True)
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        set_attribute.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
        if set_attribute(hwnd, 20, ctypes.byref(enabled), ctypes.sizeof(enabled)):
            set_attribute(hwnd, 19, ctypes.byref(enabled), ctypes.sizeof(enabled))


def clipboard_shortcut(event):
    """Dispatch before Tk's class bindings so paste runs exactly once."""
    action = {"c": "Copy", "v": "Paste", "cyrillic_es": "Copy", "cyrillic_em": "Paste"}.get(
        event.keysym.lower())
    if sys.platform == "win32":
        action = {67: "Copy", 86: "Paste"}.get(event.keycode, action)
    if action:
        event.widget.event_generate(f"<<{action}>>")
        return "break"


def hide_gui_console():
    """Hide a console belonging only to this app; preserve an existing shell."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleWindow.restype = wintypes.HWND
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    window = kernel.GetConsoleWindow()
    if not window:
        return False
    processes = (wintypes.DWORD * 64)()
    count = kernel.GetConsoleProcessList(processes, len(processes))
    owned = {os.getpid()}
    if getattr(sys, "frozen", False):
        owned.add(os.getppid())  # The onefile bootloader waits for its child.
    if not 0 < count <= len(processes) or not set(processes[:count]) <= owned:
        return False
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.ShowWindow.restype = wintypes.BOOL
    user.ShowWindow(window, 0)  # SW_HIDE
    return True


def build_arguments(action, values):
    """Validate a snapshot of form values with the same parser as the CLI."""

    input_path = Path(values.get("input", "")).expanduser()
    if not input_path.is_file():
        raise ValueError("Выберите существующий видеофайл.")
    if action == "preview":
        # The export parser validates image settings; no file will be written.
        values = {**values, "output": str(input_path.with_name(input_path.name + ".preview.mp4")),
                  "encoder": "x264", "quality": DEFAULTS["quality"], "container": "mp4", "codec": "h264"}
        action = "render"
    if action not in ("play", "render"):
        raise ValueError("Неизвестное действие.")
    arguments = [action, str(input_path.resolve())]
    for key in ("palette", "gamma", "contrast", "brightness", "color_saturation", "glyph_correction", "dither", "color"):
        arguments.append(f"--{key.replace('_', '-')}={values.get(key, DEFAULTS[key])}")
    if not values.get("audio", True):
        arguments.append("--no-audio")
    if action == "play":
        if values.get("show_resolution", False):
            arguments.append("--show-resolution")
        for key in ("fps", "cell_aspect", "font_scale"):
            value = str(values.get(key, "")).strip()
            if value:
                arguments.append(f"--{key.replace('_', '-')}={value}")
    else:
        if not values.get("output", "").strip():
            raise ValueError("Выберите файл для сохранения видео.")
        output_path = Path(values["output"]).expanduser().resolve()
        if output_path == input_path.resolve():
            raise ValueError("Исходный файл и результат должны быть разными файлами.")
        width, height = DEFAULTS["resolution"]
        resolution = str(values.get("output_width", width)).strip() + "x" + str(values.get("output_height", height)).strip()
        arguments.extend((f"--output={output_path}", f"--resolution={resolution}",
                          f"--columns={values.get('columns', DEFAULTS['columns'])}",
                          f"--encoder={values.get('encoder', DEFAULTS['encoder'])}",
                          f"--container={values.get('container', DEFAULTS['container'])}",
                          f"--quality={values.get('quality', DEFAULTS['quality'])}"))
        if values.get("codec"):
            arguments.append(f"--codec={values['codec']}")
        if values.get("overwrite", False):
            arguments.append("--overwrite")
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        try:
            options = parser().parse_args(arguments)
        except SystemExit:
            raise ValueError(errors.getvalue().split("error: ")[-1].strip()) from None
    if action == "render":
        export_format(options.output, options.container, options.codec, options.encoder, options.resolution)
    return arguments, options


def player_command(arguments):
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, str(Path(__file__).resolve().with_name("main.py")), *arguments]


def terminal_command(arguments, platform=None):
    command = player_command(arguments)
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        from console_font import classic_terminal_command
        return classic_terminal_command(command)
    if platform == "darwin":
        shell_line = shlex.join(command)
        escaped = shell_line.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
        script = 'tell application "Terminal"\nactivate\ndo script "' + escaped + '"\nend tell'
        return ["/usr/bin/osascript", "-e", script], {}
    from console_font import linux_terminal_command
    return linux_terminal_command(command, parser().parse_args(arguments).font_scale)


class ExportJob:
    """One export thread; only the GUI thread reads queues and updates widgets."""
    def __init__(self, options):
        self.options = options
        self.duration = None
        self.started = None
        self.finished = None
        self.frames = 0
        self.media_time = 0
        self.stop_event = threading.Event()
        self.progress = queue.Queue(maxsize=1)
        self.result = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="mp4-export")

    def start(self):
        self.started = time.monotonic()
        self.thread.start()

    def elapsed(self):
        if self.started is None:
            return 0.0
        return max(0.0, (self.finished if self.finished is not None else time.monotonic()) - self.started)

    def cancel(self):
        self.stop_event.set()

    def _progress(self, frames, media_time):
        try:
            self.progress.get_nowait()
        except queue.Empty:
            pass
        self.progress.put_nowait((frames, media_time))

    def _duration(self, duration):
        self.duration = duration

    def _run(self):
        if self.started is None:
            self.started = time.monotonic()
        try:
            from export import render
            render(self.options, stop_event=self.stop_event, progress=self._progress,
                   on_duration=self._duration)
        except (KeyboardInterrupt, InterruptedError):
            self.result.put(("cancelled", None))
        except ExportLimitError as error:
            self.result.put(("confirm-long-export", str(error)))
        except Exception as error:
            self.result.put(("error", str(error)))
        else:
            self.result.put(("done", str(self.options.output)))
        finally:
            self.finished = time.monotonic()


class PreviewWindow:
    def __init__(self, app):
        import tkinter as tk
        from tkinter import ttk

        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.withdraw()
        self.window.title("Предпросмотр видео")
        self.job = self.pending = None
        self.playback = None
        self._restart_playback = False
        self.closing = False
        self._poll_id = None
        self._resize_id = None
        self.view = None
        self.frame_preview = None
        self.frame_job = None
        self.source_viewport = None
        self._drag = None
        self._drag_moved = self._pan_pending = False
        self.cache = {}
        self.center = (.5, .5)
        self.bounds = (max(160, min(480, (app.root.winfo_screenwidth() - 80) // 2)),
                       max(90, min(270, app.root.winfo_screenheight() - 240)))
        self.photos = tuple(tk.PhotoImage(master=self.window, width=self.bounds[0], height=self.bounds[1])
                            for _ in range(2))
        form = ttk.Frame(self.window, padding=12)
        form.pack(fill="both", expand=True)
        form.rowconfigure(1, weight=1)
        for column in range(2):
            form.columnconfigure(column, weight=1, uniform="preview")
        self.labels = []
        self.image_items = []
        for column, title in enumerate(("Исходный кадр", "ASCII · кадр видео")):
            ttk.Label(form, text=title).grid(row=0, column=column, sticky="w", pady=(0, 6))
            label = tk.Canvas(form, width=self.bounds[0], height=self.bounds[1],
                              background="black", borderwidth=0, highlightthickness=0,
                              cursor="crosshair")
            label.grid(row=1, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
            self.image_items.append(label.create_image(0, 0, image=self.photos[column], anchor="nw"))
            self.labels.append(label)
            label.bind("<Button-1>", lambda event, index=column: self.start_pan(event, source=index == 0))
            label.bind("<B1-Motion>", self.pan)
            label.bind("<ButtonRelease-1>", self.finish_pan)
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                label.bind(event, self.wheel_zoom)
            label.bind("<Configure>", self._queue_resize)
        zoom_row = ttk.Frame(form)
        zoom_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(zoom_row, text="Масштаб видео").pack(side="left")
        self.zoom = tk.StringVar(value="Вписать")
        zoom_choice = ttk.Combobox(zoom_row, textvariable=self.zoom,
                                  values=("Вписать", "100%", "200%"), state="readonly", width=10)
        zoom_choice.pack(side="left", padx=8)
        zoom_choice.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        self.frame_button = ttk.Button(zoom_row, text="Сохранить кадр PNG…", command=self.save_frame, state="disabled")
        self.frame_button.pack(side="left", padx=(8, 0))
        self.playback_button = ttk.Button(zoom_row, text="Воспроизвести", command=self.toggle_playback)
        self.playback_button.pack(side="left", padx=(8, 0))
        self.preview_sound = tk.BooleanVar(value=False)
        ttk.Checkbutton(zoom_row, text="Звук", variable=self.preview_sound,
                        command=self.change_preview_sound).pack(side="left", padx=8)
        controls = ttk.Frame(form)
        controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(controls, text="Позиция в видео").pack(side="left")
        self.position = tk.DoubleVar(value=10)
        self.scale = ttk.Scale(controls, from_=0, to=100, variable=self.position)
        self.scale.pack(side="left", fill="x", expand=True, padx=10)
        self.scale.bind("<Button-1>", self.seek_scale)
        self.scale.bind("<B1-Motion>", self.seek_scale)
        self.scale.bind("<ButtonRelease-1>", lambda event: self.seek_scale(event, refresh=True))
        self.scale.bind("<KeyRelease>", lambda event: self.refresh())
        self.percent = tk.StringVar(value="10")
        self.percent_box = ttk.Spinbox(controls, from_=0, to=100, increment=.1,
                                       textvariable=self.percent, width=9, command=self.seek_percent)
        self.percent_box.pack(side="left")
        ttk.Label(controls, text="%").pack(side="left", padx=(4, 10))
        self.percent_box.bind("<Return>", self.seek_percent)
        tags = self.percent_box.bindtags()
        self.percent_box.bindtags((tags[0], "AsciiClipboard", *tags[1:]))
        self.status = tk.StringVar(value="Подготовка кадра…")
        self.status_label = ttk.Label(form, textvariable=self.status, wraplength=self.bounds[0] * 2)
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="w")
        self.hint = ttk.Label(form, text="Изменения параметров основного окна применяются автоматически. "
                  "Позицию можно ввести в процентах и применить Enter; шаг стрелок — 1%. "
                  "Щелчок выбирает участок, колесо меняет масштаб; крупный план можно перетаскивать. Esc завершает жест. "
                  "Шрифт в терминале может отличаться.",
                  wraplength=self.bounds[0] * 2)
        self.hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        for event in ("<FocusOut>", "<Unmap>", "<Escape>"):
            self.window.bind(event, self.end_pan)
        self.window.update_idletasks()
        minimum_width = max(640, controls.winfo_reqwidth() + 24,
                            zoom_row.winfo_reqwidth() + 24)
        for label in (self.hint, self.status_label):
            label.configure(wraplength=minimum_width - 24)
        self.window.update_idletasks()
        minimum_height = max(360, self.window.winfo_reqheight() - self.bounds[1] + 90)
        self.window.minsize(minimum_width, minimum_height)
        for label in (self.hint, self.status_label):
            label.configure(wraplength=self.bounds[0] * 2)
        self.window.update_idletasks()
        width, height = self.window.winfo_reqwidth(), self.window.winfo_reqheight()
        screen_width, screen_height = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        x, y = app.root.winfo_rootx() + app.root.winfo_width() + 12, app.root.winfo_rooty()
        if x + width > screen_width:
            x = min(app.root.winfo_rootx(), screen_width - width - 20)
            y += app.root.winfo_height() + 12
        if y + height > screen_height:
            y = (screen_height - height) // 2
        # Save an explicit normal size so changes in label wrapping cannot resize the window on restore.
        self.window.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        self.window.deiconify()
        apply_title_theme(self.window)
        self.refresh()
        self._poll_id = self.window.after(50, self.poll)

    def stop_playback(self):
        self._restart_playback = False
        playback = getattr(self, "playback", None)
        if playback is not None:
            playback.cancel()
            self.playback_button.configure(text="Остановка…", state="disabled")

    def change_preview_sound(self):
        active = getattr(self, "playback", None) is not None
        self.stop_playback()
        self._restart_playback = active

    def toggle_playback(self):
        if self.closing:
            return
        if self.playback is not None:
            self.stop_playback()
            return
        if self.job is not None or self.pending is not None:
            return
        try:
            from preview import PreviewPlaybackJob
            _, options = build_arguments("preview", self.app.snapshot())
            # Sound is independent of the export checkbox, but uses the same timeline.
            options.no_audio = False
            fraction = self.position.get() / 100
            if fraction >= 1:
                fraction = 0
            self.playback = PreviewPlaybackJob(options, fraction, self.bounds,
                zoom={"Вписать": 0, "100%": 1, "200%": 2}[self.zoom.get()],
                center=self.center, sound=self.preview_sound.get())
            self.cache.clear()
            self.playback.start()
            self.playback_button.configure(text="Пауза")
            self.status.set("Подготовка воспроизведения…")
        except Exception as error:
            self.playback = None
            self.status.set(str(error))

    def _poll_playback(self):
        playback = getattr(self, "playback", None)
        if playback is None:
            return
        try:
            value = playback.frames.get_nowait()
        except queue.Empty:
            value = None
        if value is not None and not self.closing and not playback.stop_event.is_set():
            try:
                from PIL import ImageTk
                photos = tuple(ImageTk.PhotoImage(image, master=self.window)
                               for image in (value.original, value.converted))
                self.photos = photos
                for label, item, photo in zip(self.labels, self.image_items, photos):
                    label.itemconfigure(item, image=photo)
                self.frame_preview = value
                self.view = value.frame_size, value.viewport
                self.source_viewport = value.original_viewport
                self.position.set(value.position / value.duration * 100)
                self.percent.set(f"{self.position.get():.8g}")
                self._update_frame_button()
                self.status.set(f"Позиция: {clock(value.position, milliseconds=True)} / "
                                f"{clock(value.duration, milliseconds=True)} · Сетка: {value.columns}×{value.rows}")
            except Exception as error:
                self.stop_playback()
                self.status.set(f"Не удалось показать кадр: {error}")
        if not playback.thread.is_alive():
            playback.thread.join()
            kind, value = playback.result.get_nowait()
            self.playback = None
            self.playback_button.configure(text="Воспроизвести", state="normal")
            if kind == "error" and not self.closing:
                self.status.set(f"Ошибка воспроизведения: {value}")
            if self.pending is not None and not self.closing and self.job is None:
                self._start_pending()
            elif getattr(self, "_restart_playback", False) and not self.closing:
                self._restart_playback = False
                self.toggle_playback()

    def wheel_zoom(self, event):
        if self.closing or getattr(event, "state", 0) & 1:
            return
        button = getattr(event, "num", None)
        delta = (1 if button == 4 else -1) if button in (4, 5) else getattr(event, "delta", 0)
        if not delta:
            return
        choices = ("Вписать", "100%", "200%")
        index = choices.index(self.zoom.get())
        target = min(2, max(0, index + (1 if delta > 0 else -1)))
        if target != index:
            self.zoom.set(choices[target])
            self.refresh()
        return "break"

    def _queue_resize(self, event):
        if self.closing:
            return
        self._drag = None
        self._pan_pending = False
        if self._resize_id is not None:
            self.window.after_cancel(self._resize_id)
        self._resize_id = self.window.after(120, self._resize)

    def _resize(self):
        self._resize_id = None
        if self.closing:
            return
        bounds = (min(label.winfo_width() for label in self.labels),
                  min(label.winfo_height() for label in self.labels))
        if min(bounds) < 2 or bounds == self.bounds:
            return
        self.bounds = bounds
        self.view = None
        for label in (self.status_label, self.hint):
            label.configure(wraplength=2 * bounds[0])
        self.refresh()

    def seek_scale(self, event, *, refresh=False):
        if not self.closing and not self.scale.instate(("disabled",)):
            self.stop_playback()
            percent = max(0, min(100, float(self.scale.get(event.x, event.y))))
            self.position.set(percent)
            self.percent.set(f"{percent:.8g}")
            if refresh:
                self.refresh()
        return "break"

    def seek_percent(self, event=None):
        if self.closing:
            return "break"
        try:
            percent = float(self.percent.get().replace(",", "."))
            if not 0 <= percent <= 100:
                raise ValueError
        except ValueError:
            self.status.set("Укажите позицию от 0 до 100%.")
            return "break"
        self.position.set(percent)
        self.refresh()
        return "break"

    def refresh(self, *, keep_drag=False):
        if self.closing:
            return
        self.stop_playback()
        if not keep_drag:
            self._drag = None
        self._pan_pending = False
        self.view = self.source_viewport = None
        self.frame_preview = None
        self._update_frame_button()
        try:
            _, options = build_arguments("preview", self.app.snapshot())
            options.no_audio = False
        except (OSError, RuntimeError, ValueError) as error:
            self._drag = None
            self.pending = None
            if self.job is not None:
                self.job.cancel()
            for label in self.labels:
                label.configure(cursor="arrow")
            self.status.set(str(error))
            return
        fraction = self.position.get() / 100
        self.percent.set(f"{fraction * 100:.8g}")
        self.pending = options, fraction, {"Вписать": 0, "100%": 1, "200%": 2}[self.zoom.get()], self.center
        self.status.set("Подготовка кадра… Выбор участка временно недоступен.")
        for label in self.labels:
            label.configure(cursor="fleur" if keep_drag else "watch")
        if self.job is not None:
            self.job.cancel()
        else:
            self._start_pending()

    def _start_pending(self):
        if getattr(self, "playback", None) is not None:
            return
        from preview import PreviewJob
        options, fraction, zoom, center = self.pending
        self.pending = None
        self.job = PreviewJob(options, fraction, self.bounds, zoom=zoom, center=center, cache=self.cache)
        self.job.start()

    def inspect(self, event, *, source=False):
        if self.closing or self.view is None or self._resize_id is not None:
            return
        size, viewport = self.view
        left, top, width, height = self.source_viewport if source else viewport
        self.center = tuple(min(1, max(0, value / length)) for value, length in zip(
            (left + event.x * width / self.bounds[0], top + event.y * height / self.bounds[1]), size))
        if self.zoom.get() == "Вписать":
            self.zoom.set("100%")
        self.refresh()

    def start_pan(self, event, *, source=False):
        self._drag = None
        self._drag_moved = self._pan_pending = False
        if self.closing or self.view is None or self._resize_id is not None:
            return
        if self.zoom.get() == "Вписать":
            self.inspect(event, source=source)
            return
        size, viewport = self.view
        left, top, width, height = self.source_viewport if source else viewport
        center = tuple((offset + span / 2) / length for offset, span, length in
                       zip((left, top), (width, height), size))
        limits = tuple(min(.5, span / (2 * length)) for span, length in zip((width, height), size))
        self._drag = event.x, event.y, center, limits, size, {"100%": 1, "200%": 2}[self.zoom.get()], source

    def pan(self, event):
        if self.closing or self._drag is None:
            return
        x, y, center, limits, size, zoom, _ = self._drag
        delta = event.x - x, event.y - y
        if not self._drag_moved and max(map(abs, delta)) < 4:
            return
        self._drag_moved = True
        target = tuple(min(1 - low, max(low, value - distance / (zoom * length)))
                       for value, distance, length, low in zip(center, delta, size, limits))
        if target == self.center:
            return
        self.center = target
        self._pan_pending = True
        if self.job is None:
            self.refresh(keep_drag=True)

    def finish_pan(self, event):
        if self.closing or self._drag is None:
            return
        self.pan(event)
        if self._drag is None:
            return
        source = self._drag[-1]
        moved = self._drag_moved
        self.end_pan()
        if not moved:
            self.inspect(event, source=source)

    def end_pan(self, event=None):
        if self.closing or self._drag is None:
            return
        pending = self._pan_pending
        self._drag = None
        self._pan_pending = False
        if pending:
            self.refresh()
        elif self.job is None:
            for label in self.labels:
                label.configure(cursor="crosshair")

    def save_frame(self):
        from preview import FrameSaveJob
        preview = self.frame_preview
        if self.closing or preview is None or self.frame_job is not None:
            return
        filename = self.app.dialogs.asksaveasfilename(
            parent=self.window, title="Сохранить ASCII-кадр", defaultextension=".png",
            filetypes=(("PNG", "*.png"),),
            initialfile=f"{preview.input_path.stem}-ascii-{preview.position:.3f}s.png")
        if not filename or self.closing or self.frame_job is not None:
            return
        self.frame_job = FrameSaveJob(preview, filename)
        self._update_frame_button()
        self.frame_job.start()

    def _update_frame_button(self):
        self.frame_button.configure(
            state="normal" if not self.closing and self.frame_preview is not None and self.frame_job is None else "disabled",
            text="Сохранение PNG…" if self.frame_job is not None else "Сохранить кадр PNG…")

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.stop_playback()
        self._drag = None
        self._pan_pending = False
        if self._resize_id is not None:
            self.window.after_cancel(self._resize_id)
            self._resize_id = None
        self.pending = None
        if self.frame_job is not None:
            self.frame_job.cancel()
        self._update_frame_button()
        if self.job is not None:
            self.job.cancel()
        elif self.frame_job is None and getattr(self, "playback", None) is None:
            self._destroy()

    def _destroy(self):
        self.frame_preview = None
        self.cache.clear()
        if self._poll_id is not None:
            self.window.after_cancel(self._poll_id)
            self._poll_id = None
        self.app.preview = None
        self.window.destroy()

    def poll(self):
        self._poll_id = None
        self._poll_playback()
        if self.frame_job is not None and not self.frame_job.thread.is_alive():
            self._finish_frame_save()
        if self.job is not None and not self.job.thread.is_alive():
            self._finish_preview()
        if (self.closing and self.job is None and self.frame_job is None
                and getattr(self, "playback", None) is None):
            self._destroy()
            return
        if not self.closing and self._pan_pending and self._drag is not None and self.job is None:
            self.refresh(keep_drag=True)
        self._poll_id = self.window.after(15 if getattr(self, "playback", None) is not None else 50, self.poll)

    def _finish_frame_save(self):
        self.frame_job.thread.join()
        kind, value = self.frame_job.result.get_nowait()
        self.frame_job = None
        self._update_frame_button()
        if self.closing:
            return
        if kind == "done":
            self.status.set(f"Кадр сохранён: {value}")
        elif kind == "error":
            self.app.messages.showerror("Не удалось сохранить кадр", value, parent=self.window)

    def _finish_preview(self):
        self.job.thread.join()
        kind, value = self.job.result.get_nowait()
        cancelled = self.job.stop_event.is_set()
        self.job = None
        if not self.closing and self.pending is None and kind == "done" and not cancelled:
            try:
                from PIL import ImageTk
                photos = tuple(ImageTk.PhotoImage(image, master=self.window)
                               for image in (value.original, value.converted))
            except Exception as error:
                kind, value = "error", str(error)
        if self.closing:
            return
        if self.pending is not None:
            self._start_pending()
            return
        if cancelled:
            return
        if kind == "done":
            self.view = value.frame_size, value.viewport
            self.source_viewport = value.original_viewport
            self.frame_preview = value
            self._update_frame_button()
            self.photos = photos
            for label, item, photo in zip(self.labels, self.image_items, self.photos):
                label.itemconfigure(item, image=photo)
                label.configure(cursor="fleur" if self._drag is not None else "crosshair")
            for control in (self.scale, self.percent_box):
                control.configure(state="normal" if value.duration is not None else "disabled")
            duration = clock(value.duration, milliseconds=True)
            self.status.set(f"Позиция: {clock(value.position, milliseconds=True)} / {duration} · "
                            f"Сетка: {value.columns}×{value.rows}")
        elif kind == "error":
            self._drag = None
            self._pan_pending = False
            self.view = self.source_viewport = self.frame_preview = None
            self._update_frame_button()
            for label in self.labels:
                label.configure(cursor="arrow")
            self.status.set(f"Не удалось подготовить кадр: {value}")


class Application:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox

        import darkdetect
        import ttkbootstrap as ttk
        from ttkbootstrap.dialogs import ColorChooserDialog
        from cli_options import PALETTES

        self.root = root
        dark = darkdetect.isDark()
        ttk.install_legacy_themes()
        style = ttk.Style("darkly" if dark else "litera", default_button="primary")
        # Tk can try to resolve ttkbootstrap's empty default focus color while drawing.
        style.configure(".", focuscolor=style.colors.fg)
        self.dialogs = filedialog
        self.color_dialog = ColorChooserDialog
        self.messages = messagebox
        self.job = None
        self.preview = None
        self.closing = False
        root.title("ASCII Video")
        icon = resource_path("ascii-video.png")
        if icon is not None:
            self.icon = tk.PhotoImage(file=str(icon))
            root.iconphoto(True, self.icon)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.form_view = tk.Canvas(root, width=1, height=1, background=style.colors.bg,
                                   borderwidth=0, highlightthickness=0)
        self.form_view.grid(row=0, column=0, sticky="nsew")
        self.form_scroll = ttk.Scrollbar(root, orient="vertical", command=self.form_view.yview)
        self.form_view.configure(yscrollcommand=self.form_scroll.set)
        self.form = form = ttk.Frame(self.form_view, padding=14)
        form_item = self.form_view.create_window(0, 0, anchor="nw", window=form)
        self.form_view.bind("<Configure>", lambda event: self.form_view.itemconfigure(form_item, width=event.width))
        root.bind("<FocusIn>", self.reveal_control, add="+")
        form.columnconfigure(1, weight=1)
        initial = {key: DEFAULTS[key] for key in (
            "palette", "gamma", "contrast", "brightness", "color_saturation", "glyph_correction", "fps", "cell_aspect",
            "font_scale", "columns", "encoder", "container")}
        initial["codec"] = "h264"
        initial.update(input="", output="", output_width=DEFAULTS["resolution"][0],
                       output_height=DEFAULTS["resolution"][1])
        self.values = {key: tk.StringVar(value=str(value)) for key, value in initial.items()}
        self.color_mode = tk.StringVar(value="Белый")
        self.custom_color = tk.StringVar(value="#00FF88")
        self.export_quality = tk.StringVar(value="Обычный")
        self.dither = tk.StringVar(value="Без дизеринга")
        self.audio = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=False)
        self.show_resolution = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Выберите видео и способ вывода.")

        ttk.Label(form, text="Видео").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(form, textvariable=self.values["input"]).grid(row=0, column=1, sticky="ew")
        ttk.Button(form, text="Открыть…", command=self.choose_input).grid(row=0, column=2, padx=(8, 0))

        image = ttk.Labelframe(form, text="Изображение", padding=10)
        image.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 5))
        for column in (1, 3, 5):
            image.columnconfigure(column, weight=1)

        ttk.Combobox(image, textvariable=self.values["palette"], values=tuple(PALETTES.values()), width=11).grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=3)
        ttk.Combobox(image, textvariable=self.color_mode, values=("Белый", "Исходные цвета", "Выбранный цвет"),
                     state="readonly", width=16).grid(row=0, column=3, sticky="ew", padx=(0, 10))
        color_row = ttk.Frame(image)
        color_row.grid(row=0, column=4, columnspan=2, sticky="e")
        self.color_swatch = tk.Label(color_row, width=2, background=self.custom_color.get(),
                                     relief="solid", borderwidth=1)
        self.color_swatch.pack(side="left", padx=(0, 6))
        self.palette_button = ttk.Button(color_row, text="Палитра цвета…", command=self.choose_color,
                                         bootstyle="secondary")
        self.palette_button.pack(side="left")

        for row, column, label in (
            (0, 0, "Палитра"), (0, 2, "Цвет"),
            (1, 0, "Яркость"), (1, 2, "Гамма"), (1, 4, "Контраст"),
            (2, 0, "Насыщенность"), (2, 2, "Компенсация"), (3, 0, "Дизеринг"),
        ):
            ttk.Label(image, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6))

        for column, key, low, high, step in (
            (1, "brightness", -1.0, 1.0, 0.05),
            (3, "gamma", 0.1, 5.0, 0.1),
            (5, "contrast", 0.1, 3.0, 0.1),
        ):
            ttk.Spinbox(image, from_=low, to=high, increment=step,
                        textvariable=self.values[key], width=7).grid(
                row=1, column=column, sticky="ew", padx=(0, 10 if column < 5 else 0), pady=3)
        self.saturation_box = ttk.Spinbox(image, from_=-1.0, to=1.0, increment=0.05,
                                         textvariable=self.values["color_saturation"], width=7)
        self.saturation_box.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=3)
        self.correction_box = ttk.Spinbox(image, from_=0.0, to=1.0, increment=0.05,
                                         textvariable=self.values["glyph_correction"], width=7)
        self.correction_box.grid(row=2, column=3, sticky="ew", padx=(0, 10))
        self.color_mode.trace_add("write", self.update_color_controls)
        self.update_color_controls()
        ttk.Button(image, text="Сброс эффектов", command=self.reset_effects).grid(
            row=2, column=4, columnspan=2, sticky="e")
        ttk.Combobox(image, textvariable=self.dither, values=tuple(DITHER_LABELS), state="readonly", width=23).grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=(0, 10), pady=(3, 0))
        ttk.Checkbutton(image, text="Звук", variable=self.audio).grid(row=3, column=4, columnspan=2, sticky="w")

        terminal = ttk.Labelframe(form, text="Терминал", padding=10)
        terminal.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        for column, label, key, low, high, step in (
            (0, "Ширина / высота символа", "cell_aspect", 0.1, 2.0, 0.1),
            (1, "Масштаб шрифта", "font_scale", 0.25, 2.0, 0.05),
            (2, "Предел кадров/с", "fps", 1, 60, 1),
        ):
            terminal.columnconfigure(column, weight=1)
            ttk.Label(terminal, text=label).grid(row=0, column=column, sticky="w", padx=(0, 10))
            ttk.Spinbox(terminal, from_=low, to=high, increment=step,
                        textvariable=self.values[key], width=7).grid(
                row=1, column=column, sticky="ew", padx=(0, 10), pady=(3, 0))
        ttk.Checkbutton(terminal, text="Отображать разрешение", variable=self.show_resolution).grid(
            row=1, column=3, sticky="w")

        output = ttk.Labelframe(form, text="Сохранение видео", padding=10)
        output.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        for column in range(4):
            output.columnconfigure(column, weight=1, uniform="export")
        self.codec_box = ttk.Combobox(output, textvariable=self.values["codec"],
                                     values=EXPORT_CONTAINERS["mp4"], state="readonly", width=8)
        self.encoder_box = ttk.Combobox(output, textvariable=self.values["encoder"], values=("x264", "nvenc"),
                                       state="readonly", width=8)
        for column, (label, box) in enumerate((
            ("Контейнер", ttk.Combobox(output, textvariable=self.values["container"],
                                       values=tuple(EXPORT_CONTAINERS), state="readonly", width=8)),
            ("Кодек", self.codec_box),
            ("Кодировщик", self.encoder_box),
            ("Качество", ttk.Combobox(output, textvariable=self.export_quality,
                                      values=tuple(EXPORT_QUALITY_LABELS), state="readonly", width=12)),
        )):
            gap = (0, 8 if column < 3 else 0)
            ttk.Label(output, text=label).grid(row=0, column=column, sticky="w", padx=gap)
            box.grid(row=1, column=column, sticky="ew", padx=gap, pady=(3, 8))
        self.values["container"].trace_add("write", self.update_export_container)
        self.values["codec"].trace_add("write", self.update_export_codec)

        for row, column, label, key, low, high, step in (
            (2, 0, "Столбцы ASCII", "columns", 1, 1000, 1),
            (4, 0, "Ширина, px", "output_width", 2, 8192, 2),
            (4, 1, "Высота, px", "output_height", 2, 8192, 2),
        ):
            ttk.Label(output, text=label).grid(row=row, column=column, sticky="w")
            ttk.Spinbox(output, from_=low, to=high, increment=step,
                        textvariable=self.values[key], width=8).grid(
                row=row + 1, column=column, sticky="ew", padx=(0, 8), pady=(3, 8))
        ttk.Checkbutton(output, text="Заменить файл", variable=self.overwrite).grid(row=3, column=3, sticky="w", pady=(3, 8))
        ttk.Entry(output, textvariable=self.values["output"]).grid(row=6, column=0, columnspan=3, sticky="ew", padx=(0, 8))
        ttk.Button(output, text="Сохранить как…", command=self.choose_output).grid(row=6, column=3, sticky="ew")

        actions = ttk.Frame(form)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        self.play_button = ttk.Button(actions, text="Открыть в терминале", command=self.open_terminal, bootstyle="secondary")
        self.save_button = ttk.Button(actions, text="Сохранить видео", command=self.save)
        self.cancel_button = ttk.Button(actions, text="Отмена", command=self.cancel, state="disabled", bootstyle="secondary")
        for column, button in enumerate((
            self.play_button, ttk.Button(actions, text="Предпросмотр", command=self.open_preview, bootstyle="secondary"),
            self.save_button, self.cancel_button,
        )):
            actions.columnconfigure(column, weight=1, uniform="action")
            button.grid(row=0, column=column, sticky="ew", padx=(0, 8 if column < 3 else 0))
        self.progress_line = tk.Canvas(form, height=22, background=style.colors.inputbg,
                                       highlightthickness=0)
        self.progress_line.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.progress_fill = self.progress_line.create_rectangle(
            0, 0, 0, 22, fill="#254875" if dark else "#a9c9ff", outline="")
        self.progress_text = self.progress_line.create_text(0, 11, text="0%", fill=style.colors.inputfg)
        self.progress_value = 0
        self.progress_line.bind("<Configure>", lambda event: self.show_progress(self.progress_value))
        footer = ttk.Frame(form)
        footer.grid(row=6, column=0, columnspan=3, sticky="ew")
        footer.columnconfigure(0, weight=1)
        status_label = ttk.Label(footer, textvariable=self.status, wraplength=550)
        status_label.grid(row=0, column=0, sticky="w")
        form.bind("<Configure>", self.fit_minimum)
        ttk.Label(footer, text=f"v{VERSION}").grid(
            row=0, column=1, sticky="e", padx=(8, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind_class("AsciiClipboard", "<Control-KeyPress>", clipboard_shortcut)
        pending = list(root.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if widget.winfo_class() in ("TEntry", "TSpinbox", "TCombobox", "Entry", "Text"):
                tags = widget.bindtags()
                widget.bindtags((tags[0], "AsciiClipboard", *tags[1:]))
        self._control_images = round_controls(style, root)
        root.update_idletasks()
        apply_title_theme(root)
        self.fit_minimum()
        self.watch_preview_settings()
        self._poll_id = root.after(100, self.poll)

    def watch_preview_settings(self):
        self._preview_update_id = None
        names = ("input", "palette", "gamma", "contrast", "brightness", "color_saturation", "glyph_correction",
                 "output_width", "output_height", "columns")
        for variable in (*(self.values[name] for name in names), self.color_mode,
                         self.custom_color, self.dither, self.audio):
            variable.trace_add("write", self.queue_preview_update)

    def queue_preview_update(self, *args):
        if self.closing or self.preview is None or self.preview.closing:
            return
        if self._preview_update_id is not None:
            self.root.after_cancel(self._preview_update_id)
        self._preview_update_id = self.root.after(180, self.update_preview)

    def update_color_controls(self, *args):
        mode = self.color_mode.get()
        state = "normal" if mode == "Исходные цвета" else "disabled"
        self.saturation_box.configure(state=state)
        self.correction_box.configure(state=state)
        self.palette_button.configure(state="normal" if mode == "Выбранный цвет" else "disabled")

    def reset_effects(self):
        for name in ("brightness", "gamma", "contrast", "color_saturation", "glyph_correction"):
            value = str(DEFAULTS[name])
            if self.values[name].get() != value:
                self.values[name].set(value)
        if self.dither.get() != "Без дизеринга":
            self.dither.set("Без дизеринга")

    def update_preview(self):
        self._preview_update_id = None
        if not self.closing and self.preview is not None and not self.preview.closing:
            self.preview.refresh()

    def fit_minimum(self, event=None):
        if self.closing:
            return
        width, height = self.form.winfo_reqwidth(), self.form.winfo_reqheight()
        # Some window managers report the full screen as their maximum window size.
        chrome = round(self.root.winfo_fpixels("0.75i"))
        available = min(self.root.maxsize()[1], self.root.winfo_screenheight() - chrome)
        visible = min(height, max(1, available))
        self.form_view.configure(width=width, height=visible, scrollregion=(0, 0, width, height))
        if height > visible:
            self.form_scroll.grid(row=0, column=1, sticky="ns")
            width += self.form_scroll.winfo_reqwidth()
        else:
            self.form_scroll.grid_remove()
        self.root.minsize(max(660, width), visible)

    def reveal_control(self, event):
        if not self.form_scroll.winfo_ismapped() or event.widget in (self.root, self.form_scroll):
            return
        top = event.widget.winfo_rooty() - self.form.winfo_rooty()
        y = self.form_view.canvasy(0)
        overflow = top + event.widget.winfo_height() - (y + self.form_view.winfo_height())
        if top < y or overflow > 0:
            offset = top if top < y else y + overflow
            self.form_view.yview_moveto(offset / self.form.winfo_height())

    def choose_input(self):
        filename = self.dialogs.askopenfilename(parent=self.root, title="Выбрать видео",
                                               filetypes=[("Видео", "*.mp4 *.mkv *.mov *.avi *.webm"), ("Все файлы", "*")])
        if filename:
            self.values["input"].set(filename)
            if not self.values["output"].get():
                path = Path(filename)
                self.values["output"].set(str(path.with_name(path.stem + "_ascii." + self.values["container"].get())))

    def update_export_container(self, *args):
        container = self.values["container"].get()
        codecs = EXPORT_CONTAINERS[container]
        self.codec_box.configure(values=codecs)
        if self.values["codec"].get() not in codecs:
            self.values["codec"].set(codecs[0])
        output = self.values["output"].get()
        if Path(output).name:
            self.values["output"].set(str(Path(output).with_suffix(f".{container}")))
        self.update_export_codec()

    def update_export_codec(self, *args):
        h264 = self.values["codec"].get() == "h264"
        self.encoder_box.configure(state="readonly" if h264 else "disabled")
        if not h264:
            self.values["encoder"].set("x264")

    def choose_output(self):
        container = self.values["container"].get()
        filename = self.dialogs.asksaveasfilename(parent=self.root, title="Сохранить ASCII-видео",
                                                defaultextension=f".{container}",
                                                filetypes=[(container.upper(), f"*.{container}")], confirmoverwrite=False)
        if filename:
            self.values["output"].set(filename)

    def choose_color(self):
        dialog = self.color_dialog(parent=self.root, title="Цвет символов", initialcolor=self.custom_color.get())
        dialog.show()
        if dialog.result is not None:
            value = dialog.result.hex
            self.color_swatch.configure(background=value)
            self.custom_color.set(value)
            self.color_mode.set("Выбранный цвет")

    def snapshot(self):
        values = {key: value.get() for key, value in self.values.items()}
        values["color"] = {"Белый": "mono", "Исходные цвета": "source", "Выбранный цвет": self.custom_color.get()}[self.color_mode.get()]
        if values["color"] != "source":
            values["color_saturation"] = values["glyph_correction"] = 0.0
        values["dither"] = DITHER_LABELS[self.dither.get()]
        values["audio"], values["overwrite"] = self.audio.get(), self.overwrite.get()
        values["quality"] = EXPORT_QUALITY_LABELS[self.export_quality.get()]
        values["show_resolution"] = self.show_resolution.get()
        return values

    def open_preview(self):
        if self.closing:
            return
        if self.preview is None:
            self.preview = PreviewWindow(self)
        else:
            self.preview.window.deiconify()
            self.preview.window.lift()
            self.preview.refresh()

    def open_terminal(self):
        if self.closing:
            return
        try:
            arguments, _ = build_arguments("play", self.snapshot())
            command, options = terminal_command(arguments)
            subprocess.Popen(command, shell=False, **options)
        except (ValueError, RuntimeError, OSError) as error:
            self.messages.showerror("Не удалось запустить видео", str(error), parent=self.root)
        else:
            self.status.set("Плеер запускается в терминале. Пробел — пауза, ←/→ и щелчок по полосе — перемотка, Ctrl+C — выход.")

    def save(self):
        if self.job is not None or self.closing:
            return
        try:
            _, options = build_arguments("render", self.snapshot())
        except (OSError, RuntimeError, ValueError) as error:
            self.messages.showerror("Проверьте параметры", str(error), parent=self.root)
            return
        self._start_export(options)

    def _start_export(self, options):
        self.job = ExportJob(options)
        self.play_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set(f"Сохранение {options.container.upper()}…")
        self.show_progress(0)
        self.job.start()

    def cancel(self):
        if self.job is not None:
            self.job.cancel()
            self.status.set("Остановка сохранения…")
            self.cancel_button.configure(state="disabled")

    def close(self):
        self.closing = True
        if self._preview_update_id is not None:
            self.root.after_cancel(self._preview_update_id)
            self._preview_update_id = None
        if self.preview is not None:
            self.preview.close()
        if self.job is not None:
            self.cancel()
        elif self.preview is None:
            if self._poll_id is not None:
                self.root.after_cancel(self._poll_id)
                self._poll_id = None
            self.root.destroy()

    def show_progress(self, percent):
        self.progress_value = percent
        width = self.progress_line.winfo_width()
        if percent is None:
            # Unknown duration: a moving segment, never an invented percentage.
            left = (time.monotonic() % 2) / 2 * width
            right = min(width, left + width / 5)
        else:
            left, right = 0, width * percent / 100
        self.progress_line.coords(self.progress_fill, left, 0, right, 22)
        self.progress_line.coords(self.progress_text, width / 2, 11)
        text = "—" if percent is None else ("≈" if 0 < percent < 100 else "") + f"{percent}%"
        self.progress_line.itemconfigure(self.progress_text, text=text)

    def poll(self):
        self._poll_id = None
        if self.job is not None:
            try:
                self.job.frames, self.job.media_time = self.job.progress.get_nowait()
            except queue.Empty:
                pass
            if not self.job.stop_event.is_set():
                elapsed = self.job.elapsed()
                percent, remaining = estimate(self.job.duration, self.job.media_time, elapsed)
                self.show_progress(percent)
                eta = "—" if remaining is None else "≈" + clock(remaining)
                self.status.set(f"Обработано кадров: {self.job.frames} · Видео: {clock(self.job.media_time)}\n"
                                f"Прошло: {clock(elapsed)} · Осталось: {eta}")
            if not self.job.thread.is_alive():
                self._finish_export()
        if self.closing and self.job is None and self.preview is None:
            self.root.destroy()
            return
        self._poll_id = self.root.after(100, self.poll)

    def _finish_export(self):
        self.job.thread.join()
        kind, value = self.job.result.get_nowait()
        elapsed = self.job.elapsed()
        percent, _ = estimate(self.job.duration, self.job.media_time, elapsed)
        self.show_progress(100 if kind == "done" else (percent or 0))
        options = self.job.options
        cancelled = self.job.stop_event.is_set()
        self.job = None
        if self.closing:
            return
        self.play_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if kind == "confirm-long-export":
            self.status.set("Сохранение остановлено: требуется подтверждение длительности.")
            self._confirm_long_export(options, value, cancelled)
        elif kind == "error":
            self.status.set(f"Сохранение не завершено. · Прошло: {clock(elapsed)}")
            self.messages.showerror("Не удалось сохранить видео", value, parent=self.root)
        elif kind == "cancelled":
            self.status.set(f"Сохранение отменено. · Прошло: {clock(elapsed)}")
        else:
            self.status.set(f"Сохранено: {value}\nПрошло: {clock(elapsed)} · Осталось: 00:00")

    def _confirm_long_export(self, options, reason, cancelled):
        if cancelled or not self.messages.askyesno(
                "Необычная длительность видео",
                reason + "\n\nПродолжить без ограничений длительности? Экспорт начнётся заново.",
                parent=self.root, default="no"):
            return
        if self.closing or self.job is not None:
            return
        options.allow_long_export = True
        self._start_export(options)


def launch():
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("Tkinter недоступен. Установите Python с поддержкой Tk.") from error
    if sys.platform == "win32":
        import ctypes

        # Let Tk render at the system DPI before Windows creates any GUI windows.
        ctypes.windll.user32.SetProcessDPIAware()
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError("Не удалось открыть окно. Проверьте наличие графического рабочего стола.") from error
    application = None
    try:
        root.withdraw()
        application = Application(root)
        hide_gui_console()
        root.deiconify()
        root.mainloop()
    except tk.TclError as error:
        raise RuntimeError(f"Ошибка графического интерфейса: {error}") from error
    finally:
        if application is not None and application.job is not None:
            application.job.cancel()
            application.job.thread.join()
        preview = getattr(application, "preview", None)
        if preview is not None:
            for worker in (preview.job, preview.playback, preview.frame_job):
                if worker is not None:
                    worker.cancel()
                    worker.thread.join()
            preview.cache.clear()
        try:
            root.destroy()
        except tk.TclError:
            pass

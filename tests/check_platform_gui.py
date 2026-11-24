"""Capture a real source GUI on the requested OS; never opens an audio device.

Run each theme in a separate process. Screenshots still need visual review.
Linux example: xvfb-run -a -s '-screen 0 1920x1080x24' python
tests/check_platform_gui.py --expected-platform linux --theme light --output build/gui-light
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

sys.modules["sounddevice"] = None
sys.modules["audio"] = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fixture(path):
    import av
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        stream.options = {"preset": "ultrafast"}
        y, x = np.indices((180, 320))
        for index in range(90):
            pixels = np.stack(((x + index * 3) % 256, (y * 2) % 256,
                               ((x // 20 + y // 20) % 2) * 230 + 20), axis=-1).astype("uint8")
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def run(theme, output):
    import tkinter as tk
    from PIL import Image, ImageGrab
    import gui

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()

    output.mkdir(parents=True, exist_ok=True)
    source = output / "input.mp4"
    fixture(source)
    errors = []
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda kind, value, tb: errors.append(
        "".join(traceback.format_exception(kind, value, tb)))
    application = None

    def pump_until(predicate, description, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            root.update()
            if errors:
                raise AssertionError("Tk callback failed: " + errors[0])
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError(f"Timed out: {description}")

    def capture(window, name):
        window.lift()
        root.update()
        left, top = window.winfo_rootx(), window.winfo_rooty()
        width, height = window.winfo_width(), window.winfo_height()
        if not (0 <= left and 0 <= top and left + width <= window.winfo_screenwidth()
                and top + height <= window.winfo_screenheight()):
            raise AssertionError(f"Window extends beyond screen: {name}")
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            import os

            user32 = ctypes.windll.user32
            user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
            user32.GetAncestor.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
            handle = user32.GetAncestor(window.winfo_id(), 2)  # GA_ROOT: Tk's native toplevel.
            process = wintypes.DWORD()
            user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
            if not handle or process.value != os.getpid():
                raise AssertionError(f"Cannot identify application's window: {name}")
            shot = ImageGrab.grab(window=handle).convert("RGB")
        else:
            shot = ImageGrab.grab(bbox=(left, top, left + width, top + height)).convert("RGB")
        if len(shot.getcolors(8) or range(9)) <= 2:
            raise AssertionError(f"Blank screenshot: {name}; check desktop capture permissions")
        shot.save(output / name)
        return {"width": width, "height": height, "screenshot": name}

    try:
        with patch("darkdetect.isDark", return_value=theme == "dark"):
            application = gui.Application(root)
        application.audio.set(False)
        application.values["input"].set(str(source.resolve()))
        application.values["columns"].set("48")
        application.color_mode.set("Исходные цвета")
        root.geometry("+10+10")
        root.deiconify()
        pump_until(lambda: root.winfo_viewable(), "main window")
        root.update_idletasks()
        main = capture(root, "main.png")

        application.open_preview()
        preview = application.preview
        pump_until(lambda: preview.frame_preview is not None and preview.job is None
                   and preview.pending is None, "both preview images")
        if preview.preview_sound.get():
            raise AssertionError("Preview must default to muted")
        for image in (preview.frame_preview.original, preview.frame_preview.converted):
            if len(image.convert("RGB").getcolors(8) or range(9)) <= 2:
                raise AssertionError("Preview image is blank")
        preview_geometry = capture(preview.window, "preview.png")
        position = preview.position.get()
        preview.toggle_playback()
        pump_until(lambda: preview.position.get() > position + 2, "muted playback advancing")
        preview.stop_playback()
        pump_until(lambda: preview.playback is None and preview.frame_preview is not None,
                   "playback pause")

        png = output / "frame.png"
        with patch.object(application.dialogs, "asksaveasfilename", return_value=str(png.resolve())):
            preview.save_frame()
        pump_until(lambda: png.is_file() and preview.frame_job is None, "PNG save")
        with Image.open(png) as saved:
            saved.verify()
        preview.close()
        pump_until(lambda: application.preview is None, "preview close")
        application.close()
        report = {"status": "passed", "mode": "source-gui", "platform": sys.platform,
                  "platform_version": platform.platform(), "python": platform.python_version(),
                  "theme": theme, "main": main, "preview": preview_geometry,
                  "checks": ["main window", "two preview images", "muted playback",
                             "pause", "PNG save", "close"],
                  "physical_audio": "disabled by blocked imports",
                  "visual_review": "Screenshots require human/visual inspection; this is not packaged-GUI acceptance"}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        if application is not None:
            preview = application.preview
            if preview is not None:
                for job in (preview.job, preview.playback, preview.frame_job):
                    if job is not None:
                        job.cancel()
                        job.thread.join(timeout=10)
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-platform", required=True, choices=("linux", "darwin", "win32"))
    parser.add_argument("--theme", required=True, choices=("light", "dark"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != args.expected_platform:
        parser.error(f"Requested {args.expected_platform}, running on {sys.platform}; no cross-platform result recorded")
    try:
        run(args.theme, args.output)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps({
            "status": "failed", "mode": "source-gui", "platform": sys.platform,
            "theme": args.theme, "error": traceback.format_exc(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise

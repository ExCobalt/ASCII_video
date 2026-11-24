import tkinter as tk
import unittest
from unittest.mock import Mock

from gui import Application


class AutoPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tcl = tk.Tcl()
        self.app = Application.__new__(Application)
        self.app.root = Mock()
        self.app.closing = False
        self.app.preview = Mock(closing=False)
        self.app.values = {name: tk.StringVar(self.tcl, "0") for name in (
            "input", "palette", "gamma", "contrast", "brightness", "color_saturation", "glyph_correction",
            "output_width", "output_height", "columns", "output", "encoder", "fps")}
        for name in ("color_mode", "custom_color", "dither", "audio"):
            setattr(self.app, name, tk.StringVar(self.tcl, "0"))
        self.app.export_quality = tk.StringVar(self.tcl, "Обычный")
        self.callbacks = {}
        self.serial = 0

        def after(delay, callback):
            self.assertEqual(delay, 180)
            self.serial += 1
            self.callbacks[self.serial] = callback
            return self.serial

        self.app.root.after.side_effect = after
        self.app.root.after_cancel.side_effect = self.callbacks.pop
        self.app.watch_preview_settings()

    def flush(self):
        callbacks = list(self.callbacks.values())
        self.callbacks.clear()
        for callback in callbacks:
            callback()

    def test_real_variable_traces_coalesce_all_image_settings(self):
        for name, variable in self.app.values.items():
            if name not in ("output", "encoder", "fps"):
                variable.set("1")
                self.assertEqual(len(self.callbacks), 1)
        for name in ("color_mode", "custom_color", "dither", "audio"):
            getattr(self.app, name).set("1")
            self.assertEqual(len(self.callbacks), 1)
        self.app.preview.refresh.assert_not_called()
        self.flush()
        self.app.preview.refresh.assert_called_once_with()
        self.assertIsNone(self.app._preview_update_id)


    def test_reset_effects_repairs_fields_and_coalesces_without_changing_other_settings(self):
        from cli_options import DEFAULTS
        names = ("brightness", "gamma", "contrast", "color_saturation", "glyph_correction")
        for name in names:
            self.app.values[name].set("invalid")
        self.app.dither.set("Синий шум")
        retained = {name: variable.get() for name, variable in self.app.values.items() if name not in names}
        retained.update({name: getattr(self.app, name).get() for name in ("color_mode", "custom_color", "audio", "export_quality")})
        self.app.reset_effects()
        for name in names:
            self.assertEqual(self.app.values[name].get(), str(DEFAULTS[name]))
        self.assertEqual(self.app.dither.get(), "Без дизеринга")
        for name, value in retained.items():
            variable = self.app.values[name] if name in self.app.values else getattr(self.app, name)
            self.assertEqual(variable.get(), value)
        self.assertEqual(len(self.callbacks), 1)
        self.flush()
        self.app.preview.refresh.assert_called_once_with()
        self.app.reset_effects()
        self.assertFalse(self.callbacks)

    def test_unrelated_settings_do_not_refresh(self):
        self.app.export_quality.set("Высокое")
        for name in ("output", "encoder", "fps"):
            self.app.values[name].set("1")
        self.assertFalse(self.callbacks)

    def test_closed_preview_and_app_ignore_pending_update(self):
        for state in ("preview", "app", "missing"):
            self.app.preview = Mock(closing=False)
            preview = self.app.preview
            self.app.closing = False
            self.app.values["brightness"].set("1")
            if state == "preview":
                preview.closing = True
            elif state == "app":
                self.app.closing = True
            else:
                self.app.preview = None
            self.flush()
            preview.refresh.assert_not_called()
            self.app.values["brightness"].set("2")
            self.assertFalse(self.callbacks)

    def test_invalid_intermediate_input_is_replaced_before_refresh(self):
        self.app.values["brightness"].set("-")
        self.app.values["brightness"].set("-0.5")
        self.flush()
        self.app.preview.refresh.assert_called_once_with()
        self.assertEqual(self.app.values["brightness"].get(), "-0.5")


class PreviewScaleTests(unittest.TestCase):


    def test_window_exposes_only_selected_effects_and_preserves_export_choices(self):
        root = tk.Tk()
        root.withdraw()
        app = Application(root)
        self.addCleanup(app.close)
        self.assertEqual(set(app.values), {"input", "output", "palette", "gamma", "contrast", "brightness", "color_saturation", "glyph_correction",
            "fps", "cell_aspect", "font_scale", "columns", "encoder", "container", "codec",
            "output_width", "output_height"})
        app.values["output"].set("example.mp4")
        app.values["container"].set("webm")
        self.assertEqual(app.values["output"].get(), "example.webm")
        self.assertEqual(app.values["codec"].get(), "vp9")
        self.assertEqual(str(app.encoder_box["state"]), "disabled")
        app.values["brightness"].set(".3")
        app.values["color_saturation"].set(".7")
        app.values["glyph_correction"].set(".8")
        self.assertEqual(str(app.saturation_box["state"]), "disabled")
        app.color_mode.set("Исходные цвета")
        self.assertEqual(str(app.saturation_box["state"]), "normal")
        app.color_mode.set("Выбранный цвет")
        self.assertEqual(str(app.saturation_box["state"]), "disabled")
        app.dither.set("Синий шум")
        app.reset_effects()
        self.assertEqual(app.values["brightness"].get(), "0.0")
        self.assertEqual(app.values["color_saturation"].get(), "0.0")
        self.assertEqual(app.values["glyph_correction"].get(), "0.0")
        self.assertEqual(app.dither.get(), "Без дизеринга")
        self.assertEqual(app.values["container"].get(), "webm")

    def test_click_and_drag_select_cursor_position_without_class_step(self):
        from tkinter import ttk
        from gui import PreviewWindow
        root = tk.Tk()
        self.addCleanup(root.destroy)
        root.withdraw()
        window = tk.Toplevel(root)
        preview = PreviewWindow.__new__(PreviewWindow)
        preview.closing = False
        preview.position = tk.DoubleVar(root, 10)
        preview.percent = tk.StringVar(root, "10")
        preview.refresh = Mock()
        preview.scale = ttk.Scale(window, from_=0, to=100, variable=preview.position, length=400)
        preview.scale.pack()
        preview.scale.bind("<Button-1>", preview.seek_scale)
        preview.scale.bind("<B1-Motion>", preview.seek_scale)
        preview.scale.bind("<ButtonRelease-1>", lambda event: preview.seek_scale(event, refresh=True))
        window.update()
        for fraction in (.8, .2, .5):
            x, y = int(400 * fraction), preview.scale.winfo_height() // 2
            expected = float(preview.scale.get(x, y))
            preview.scale.event_generate("<Button-1>", x=x, y=y)
            self.assertAlmostEqual(preview.position.get(), expected)
            preview.scale.event_generate("<ButtonRelease-1>", x=x, y=y)
            self.assertAlmostEqual(preview.position.get(), expected)
            self.assertAlmostEqual(float(preview.percent.get()), expected, places=5)
        self.assertEqual(preview.refresh.call_count, 3)
        for x, expected in ((-20, 0), (500, 100)):
            preview.scale.event_generate("<B1-Motion>", x=x, y=y)
            self.assertEqual(preview.position.get(), expected)
        preview.scale.state(("disabled",))
        preview.scale.event_generate("<Button-1>", x=30, y=y)
        self.assertEqual(preview.position.get(), 100)
        preview.scale.state(("!disabled",))
        preview.closing = True
        preview.scale.event_generate("<ButtonRelease-1>", x=30, y=y)
        self.assertEqual(preview.position.get(), 100)
        self.assertEqual(preview.refresh.call_count, 3)

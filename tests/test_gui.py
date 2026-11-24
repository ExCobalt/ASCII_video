from pathlib import Path
import json
import shlex
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gui import Application, ExportJob, build_arguments, clipboard_shortcut, hide_gui_console, launch, player_command, terminal_command


class GuiTests(unittest.TestCase):

    def test_short_desktop_can_scroll_to_export_actions(self):
        import tkinter as tk
        from ttkbootstrap import Style
        from ttkbootstrap.publisher import Publisher

        Style.instance = None
        Publisher.clear_subscribers()
        self.addCleanup(setattr, Style, "instance", None)
        self.addCleanup(Publisher.clear_subscribers)
        root = tk.Tk()
        self.addCleanup(root.destroy)
        root.withdraw()
        with patch.object(root, "maxsize", return_value=(1366, 500)):
            app = Application(root)
            root.deiconify()
            root.update()
            self.assertLessEqual(root.winfo_height(), 500)
            self.assertTrue(app.form_scroll.winfo_ismapped())
            app.save_button.focus_force()
            root.update()
            top = app.save_button.winfo_rooty() - root.winfo_rooty()
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(top + app.save_button.winfo_height(), root.winfo_height())
            self.assertGreater(app.form_view.yview()[0], 0)
            app.root.after_cancel(app._poll_id)

    def test_switch_container_with_incomplete_output_path(self):
        app = Application.__new__(Application)
        app.values = {key: Mock() for key in ("container", "codec", "output")}
        app.values["container"].get.return_value = "webm"
        app.values["output"].get.return_value = "."
        app.codec_box = Mock()
        app.update_export_codec = Mock()
        app.update_export_container()
        app.values["output"].set.assert_not_called()
        app.update_export_codec.assert_called_once_with()



    def test_close_cancels_pending_main_poll_before_destroy(self):
        app = Application.__new__(Application)
        app.root = Mock()
        app.preview = None
        app.job = None
        app._preview_update_id = None
        app._poll_id = "pending-main-poll"
        app.close()
        app.root.after_cancel.assert_called_once_with("pending-main-poll")
        app.root.destroy.assert_called_once_with()
        self.assertIsNone(app._poll_id)

    def test_dither_names_and_form_snapshot_reach_both_commands(self):
        from gui import DITHER_LABELS
        from dithering import DITHER_MODES
        from diffusion import MODES
        self.assertEqual(set(DITHER_LABELS.values()), set((*DITHER_MODES, *MODES)))
        app = Application.__new__(Application)
        app.values = {name: Mock() for name in ("input", "output", "brightness", "color_saturation", "glyph_correction")}
        for name, value in (("input", str(self.input)), ("output", str(self.input.with_name("out.mp4"))),
                            ("brightness", "0.5"), ("color_saturation", "-.5"), ("glyph_correction", ".75")):
            app.values[name].get.return_value = value
        app.export_quality = Mock()
        app.export_quality.get.return_value = "Обычный"
        app.color_mode, app.custom_color, app.dither = Mock(), Mock(), Mock()
        app.color_mode.get.return_value = "Исходные цвета"
        app.audio, app.overwrite, app.show_resolution = Mock(), Mock(), Mock()
        app.audio.get.return_value = False
        app.overwrite.get.return_value = False
        app.show_resolution.get.return_value = False
        for label, mode in DITHER_LABELS.items():
            app.dither.get.return_value = label
            for action in ("play", "render"):
                _, options = build_arguments(action, app.snapshot())
                self.assertEqual(options.dither, mode)
                self.assertEqual(options.brightness, .5)
                self.assertEqual(options.color_saturation, -.5)
                self.assertEqual(options.glyph_correction, .75)

        app.saturation_box, app.correction_box, app.palette_button = Mock(), Mock(), Mock()
        app.custom_color.get.return_value = "#00ff88"
        for color in ("Белый", "Выбранный цвет", "Исходные цвета"):
            app.color_mode.get.return_value = color
            app.update_color_controls()
            state = "normal" if color == "Исходные цвета" else "disabled"
            app.saturation_box.configure.assert_called_with(state=state)
            app.correction_box.configure.assert_called_with(state=state)
            app.palette_button.configure.assert_called_with(
                state="normal" if color == "Выбранный цвет" else "disabled")
            values = app.snapshot()
            self.assertEqual(float(values["glyph_correction"]), .75 if state == "normal" else 0)
            self.assertEqual(float(values["color_saturation"]), -.5 if state == "normal" else 0)
        app.values["glyph_correction"].set.assert_not_called()


    def test_clipboard_shortcuts_in_latin_and_russian_layouts(self):
        for keysym, code, action in (("c", 67, "Copy"), ("Cyrillic_es", 67, "Copy"),
                                    ("v", 86, "Paste"), ("Cyrillic_em", 86, "Paste"),
                                    ("unknown", 86, "Paste")):
            widget = Mock()
            with patch("gui.sys.platform", "win32"):
                result = clipboard_shortcut(SimpleNamespace(widget=widget, keysym=keysym, keycode=code))
            self.assertEqual(result, "break")
            widget.event_generate.assert_called_once_with(f"<<{action}>>")
        widget = Mock()
        with patch("gui.sys.platform", "win32"):
            self.assertIsNone(clipboard_shortcut(SimpleNamespace(widget=widget, keysym="z", keycode=90)))
        widget.event_generate.assert_not_called()

    def test_console_hiding_preserves_shell_and_handles_onefile(self):
        import ctypes
        kernel, user = Mock(), Mock()
        kernel.GetConsoleWindow.return_value = 123
        def console_processes(buffer, capacity):
            for index, pid in enumerate(attached):
                buffer[index] = pid
            return len(attached)
        kernel.GetConsoleProcessList.side_effect = console_processes
        with patch("gui.sys.platform", "win32"), patch("gui.os.getpid", return_value=11), \
                patch("gui.os.getppid", return_value=22), patch.object(sys, "frozen", True, create=True), \
                patch.object(ctypes, "WinDLL", side_effect=lambda name, **kwargs: kernel if name == "kernel32" else user, create=True):
            attached = [11, 22, 33]  # An existing command shell shares the console.
            self.assertFalse(hide_gui_console())
            user.ShowWindow.assert_not_called()
            attached = [11, 22]
            self.assertTrue(hide_gui_console())
            user.ShowWindow.assert_called_once_with(123, 0)
            kernel.GetConsoleWindow.return_value = None
            self.assertFalse(hide_gui_console())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.input = Path(self.temporary.name) / "кино с пробелами.mp4"
        self.input.touch()

    def test_play_arguments_keep_terminal_grid_automatic(self):
        arguments, options = build_arguments("play", {"input": str(self.input), "audio": False,
                                                      "color": "#00ff88", "fps": "60", "cell_aspect": "0.7", "font_scale": "0.8"})
        self.assertEqual(options.input, self.input.resolve())
        self.assertIsNone(options.width)
        self.assertIsNone(options.height)
        self.assertTrue(options.no_audio)
        self.assertEqual((options.color, options.fps, options.cell_aspect), ("#00ff88", 60, 0.7))
        self.assertEqual(options.font_scale, 0.8)
        self.assertNotIn("--width", arguments)
        self.assertFalse(build_arguments("play", {"input": str(self.input)})[1].no_audio)
        default_options = build_arguments("play", {"input": str(self.input), "fps": "", "cell_aspect": "", "font_scale": ""})[1]
        self.assertEqual((default_options.fps, default_options.cell_aspect), (30, 0.5))
        self.assertEqual(default_options.font_scale, 0.8)
        for value in ("0.24", "2.01", "nan", "inf", "abc"):
            with self.subTest(font_scale=value), self.assertRaises(ValueError):
                build_arguments("play", {"input": str(self.input), "font_scale": value})

    def test_export_formats_share_validation_with_cli(self):
        from cli_options import EXPORT_CONTAINERS
        for container, codecs in EXPORT_CONTAINERS.items():
            for codec in codecs:
                values = dict(input=str(self.input), output=str(self.input.with_name(f"out.{container}")),
                              container=container, codec=codec, audio=False)
                _, options = build_arguments("render", values)
                self.assertEqual((options.container, options.codec), (container, codec))
        with self.assertRaises(ValueError):
            build_arguments("render", dict(input=str(self.input), output=str(self.input.with_name("out.webm")),
                                           container="webm", codec="h264"))

    def test_export_quality_is_validated_for_file_only(self):
        from gui import EXPORT_QUALITY_LABELS
        for quality in EXPORT_QUALITY_LABELS.values():
            values = dict(input=str(self.input), output=str(self.input.with_name("out.mp4")), quality=quality)
            arguments, options = build_arguments("render", values)
            self.assertEqual(options.quality, quality)
            self.assertIn(f"--quality={quality}", arguments)
            self.assertFalse(any(a.startswith("--quality") for a in build_arguments("play", values)[0]))
        values["quality"] = "unknown"
        with self.assertRaises(ValueError):
            build_arguments("render", values)
        self.assertEqual(build_arguments("preview", values)[1].quality, "balanced")

    def test_render_arguments_and_validation(self):
        arguments, options = build_arguments("play", {"input": str(self.input), "show_resolution": True})
        self.assertIn("--show-resolution", arguments)
        self.assertTrue(options.show_resolution)
        self.assertFalse(build_arguments("play", {"input": str(self.input)})[1].show_resolution)
        output = Path(self.temporary.name) / "результат.mp4"
        _, options = build_arguments("render", {"input": str(self.input), "output": str(output), "output_width": "640", "output_height": "360",
                                                "columns": "80", "overwrite": True, "audio": False, "gamma": "1.4"})
        self.assertEqual(options.resolution, (640, 360))
        self.assertEqual((options.output, options.columns), (output.resolve(), 80))
        self.assertTrue(options.overwrite and options.no_audio)
        self.assertEqual(options.encoder, "x264")
        arguments, options = build_arguments("render", {"input": str(self.input), "output": str(output), "encoder": "nvenc", "show_resolution": True})
        self.assertNotIn("--show-resolution", arguments)
        self.assertEqual(options.encoder, "nvenc")
        self.assertEqual((options.resolution, options.columns), ((1920, 1080), 320))
        self.assertIn("--encoder=nvenc", arguments)
        self.assertNotIn("--encoder", build_arguments("play", {"input": str(self.input), "encoder": "nvenc"})[0])
        invalid = [{"gamma": "nan"}, {"palette": "░@"}, {"output_width": "641"}, {"output_height": "361"},
                   {"output_width": ""}, {"output_height": "no"}, {"output_height": "8194"}, {"output_width": "0"},
                   {"color": "blue"}, {"output": str(self.input)}, {"output": ""}, {"encoder": "unknown"}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_arguments("render", {"input": str(self.input), "output": str(output), **changes})
        with self.assertRaises(ValueError):
            build_arguments("play", {"input": self.temporary.name})

    def test_option_like_palette_and_negative_exponent_survive_gui_validation(self):
        for action in ("play", "render"):
            with self.subTest(action=action):
                arguments, options = build_arguments(action, {
                    "input": str(self.input), "output": str(self.input.with_name("out.mp4")),
                    "palette": "-+", "brightness": "-1e-1",
                })
                self.assertEqual(options.palette, "-+")
                self.assertEqual(options.brightness, -0.1)
                self.assertIn("--palette=-+", arguments)

    def test_gui_and_cli_share_numeric_and_image_defaults(self):
        from cli_options import DEFAULTS, parser
        for action in ("play", "render"):
            output = str(self.input.with_name("out.mp4"))
            _, gui = build_arguments(action, {"input": str(self.input), "output": output})
            cli = parser().parse_args([action, str(self.input)] + (
                ["--output", output] if action == "render" else []))
            for name in DEFAULTS:
                if hasattr(cli, name):
                    self.assertEqual(getattr(gui, name), getattr(cli, name), name)

    def test_player_command_frozen_and_source(self):
        arguments = ["play", str(self.input), "--no-audio"]
        with patch("sys.frozen", True, create=True), patch("sys.executable", "C:/Program Files/ASCII Video/ascii-video.exe"):
            self.assertEqual(player_command(arguments), ["C:/Program Files/ASCII Video/ascii-video.exe", *arguments])
        with patch("sys.frozen", False, create=True), patch("sys.executable", "python-test"):
            command = player_command(arguments)
            self.assertEqual(command[0], "python-test")
            self.assertEqual(Path(command[1]).name, "main.py")
            self.assertEqual(command[2:], arguments)

    def test_terminal_commands_no_shell_or_quote_loss(self):
        arguments = ["play", "movie with 'quote' \"double\" $(bad) \\path\nline.mp4", "--no-audio"]
        with patch("sys.frozen", True, create=True), patch("sys.executable", "/tmp/my player"):
            windows, options = terminal_command(arguments, "win32")
            self.assertEqual(Path(windows[0]).name, "conhost.exe")
            self.assertEqual(windows[1:], ["/tmp/my player", *arguments])
            self.assertEqual(options["creationflags"], 0x10)
            mac, options = terminal_command(arguments, "darwin")
            expected = shlex.join(["/tmp/my player", *arguments])
            self.assertEqual(shlex.split(expected), ["/tmp/my player", *arguments])
            escaped = expected.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
            self.assertEqual(mac[:2], ["/usr/bin/osascript", "-e"])
            self.assertIn('do script "' + escaped + '"', mac[2])
            literal = mac[2].split('do script "', 1)[1].split('"\nend tell', 1)[0]
            self.assertEqual(shlex.split(json.loads('"' + literal + '"')), ["/tmp/my player", *arguments])
            self.assertEqual(options, {})
            with patch("console_font.shutil.which", side_effect=lambda name: "/usr/bin/konsole" if name == "konsole" else None):
                for scale in ("0.8", "1"):
                    linux, options = terminal_command([*arguments, f"--font-scale={scale}"], "linux")
                    self.assertEqual(linux[:3], ["/usr/bin/konsole", "-e", "env"])
                    self.assertIn("PYINSTALLER_RESET_ENVIRONMENT=1", linux)
                    self.assertEqual(linux[-len(arguments)-2:], ["/tmp/my player", *arguments, f"--font-scale={scale}"])
            with patch("console_font.shutil.which", return_value=None), self.assertRaises(RuntimeError):
                terminal_command(arguments, "linux")

    def test_worker_progress_is_bounded_and_success_reported(self):
        def render(options, stop_event, progress, on_duration):
            for index in range(1000):
                progress(index, index / 30)
        with patch.dict(sys.modules, export=SimpleNamespace(render=render)):
            job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            job.start()
            job.thread.join(2)
        self.assertFalse(job.thread.is_alive())
        self.assertEqual(job.progress.qsize(), 1)
        self.assertEqual(job.progress.get_nowait(), (999, 999 / 30))
        self.assertEqual(job.result.get_nowait(), ("done", "result.mp4"))

    def test_close_cancels_and_waits_for_cleanup_before_destroy(self):
        ready, cleanup = threading.Event(), threading.Event()
        def render(options, stop_event, progress, on_duration):
            ready.set()
            stop_event.wait(2)
            cleanup.wait(2)
            raise KeyboardInterrupt
        application = Application.__new__(Application)
        application.preview = None
        application.root = Mock()
        application.status = Mock()
        application.progress_line = Mock()
        application.progress_line.winfo_width.return_value = 200
        application.progress_fill = 1
        application.progress_text = 2
        application.cancel_button = Mock()
        application._preview_update_id = None
        application.closing = False
        application.messages = Mock()
        with patch.dict(sys.modules, export=SimpleNamespace(render=render)):
            application.job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            application.job.start()
            self.assertTrue(ready.wait(1))
            application.close()
            self.assertTrue(application.job.stop_event.is_set())
            application.poll()
            application.root.destroy.assert_not_called()
            cleanup.set()
            application.job.thread.join(2)
            application.poll()
        application.root.destroy.assert_called_once()
        application.messages.showerror.assert_not_called()
        self.assertIsNone(application.job)

    def test_worker_error_reports_without_console_output(self):
        def render(options, stop_event, progress, on_duration):
            raise RuntimeError("decode failed")
        with patch.dict(sys.modules, export=SimpleNamespace(render=render)):
            job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            job.start()
            job.thread.join(2)
        self.assertEqual(job.result.get_nowait(), ("error", "decode failed"))

    def test_conversion_percent_estimate_and_completion(self):
        application = Application.__new__(Application)
        application.preview = None
        application.root = Mock()
        application.status = Mock()
        application.progress_line = Mock()
        application.progress_line.winfo_width.return_value = 200
        application.progress_fill = 1
        application.progress_text = 2
        application._preview_update_id = None
        application.closing = False
        application.play_button = Mock()
        application.save_button = Mock()
        application.cancel_button = Mock()
        application.messages = Mock()
        for duration, media_time, expected in ((10, 5, "≈50%"), (10, 12, "≈99%"),
                                                (None, 5, "—")):
            job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            job.duration = duration
            job.started = 100
            job.thread = Mock()
            job.thread.is_alive.return_value = True
            job._progress(150, media_time)
            application.job = job
            with patch("gui.time.monotonic", return_value=104):
                application.poll()
            self.assertTrue(application.status.set.call_args.args[0].startswith("Обработано кадров"))
            application.progress_line.itemconfigure.assert_called_with(2, text=expected)
            application.progress_line.coords.assert_any_call(2, 100, 11)
            self.assertIn("Прошло: 00:04", application.status.set.call_args.args[0])
            self.assertIn("Осталось: " + ("≈00:04" if media_time == 5 and duration == 10 else "—"),
                          application.status.set.call_args.args[0])
            if duration is not None:
                application.progress_line.coords.assert_any_call(
                    1, 0, 0, 100 if media_time == 5 else 198, 22)
        job.stop_event.set()
        job._progress(200, 20)
        application.status.reset_mock()
        application.poll()
        application.status.set.assert_not_called()
        job.thread.is_alive.return_value = False
        job.result.put(("cancelled", None))
        application.poll()
        self.assertTrue(application.status.set.call_args.args[0].startswith("Сохранение отменено."))
        self.assertNotIn("100%", application.status.set.call_args.args[0])
        application.job = job
        job.result.put(("done", "result.mp4"))
        application.poll()
        self.assertTrue(application.status.set.call_args.args[0].startswith("Сохранено: result.mp4"))
        application.progress_line.itemconfigure.assert_called_with(2, text="100%")
        application.progress_line.coords.assert_any_call(1, 0, 0, 200, 22)

    def test_worker_elapsed_freezes_at_completion(self):
        with patch.dict(sys.modules, export=SimpleNamespace(render=Mock())), \
                patch("gui.time.monotonic", side_effect=[100, 104]):
            job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            job._run()
        self.assertEqual(job.elapsed(), 4)

    def test_export_confirmation_rechecks_state_after_modal_dialog(self):
        for change in ("closing", "new-job"):
            with self.subTest(change=change):
                options = SimpleNamespace(output=Path("result.mp4"), allow_long_export=False)
                finished = ExportJob(options)
                finished.thread = Mock()
                finished.thread.is_alive.return_value = False
                finished.result.put(("confirm-long-export", "long timeline"))
                app = Application.__new__(Application)
                app.job, app.closing, app.preview = finished, False, None
                app.root, app.status, app.messages = Mock(), Mock(), Mock()
                app.play_button, app.save_button, app.cancel_button = Mock(), Mock(), Mock()
                app.show_progress, app._start_export = Mock(), Mock()
                replacement = Mock()

                def confirm(*args, **kwargs):
                    finished.thread.join.assert_called_once()
                    self.assertIsNone(app.job)
                    if change == "closing":
                        app.closing = True
                    else:
                        app.job = replacement
                    return True

                app.messages.askyesno.side_effect = confirm
                app.poll()
                app.messages.askyesno.assert_called_once()
                app._start_export.assert_not_called()
                self.assertFalse(options.allow_long_export)
                if change == "closing":
                    app.root.destroy.assert_called_once()
                    app.root.after.assert_not_called()
                else:
                    self.assertIs(app.job, replacement)
                    app.root.destroy.assert_not_called()
                    app.root.after.assert_called_once_with(100, app.poll)

    def test_duration_comes_from_export_without_an_extra_container_open(self):
        for duration in (10.0, None):
            def render(options, stop_event, progress, on_duration):
                on_duration(duration)
                progress(1, 0)
            backend = SimpleNamespace(open=Mock(side_effect=AssertionError("Extra probe")))
            with patch.dict(sys.modules, av=backend, export=SimpleNamespace(render=render)):
                job = ExportJob(SimpleNamespace(input="input.mp4", output=Path("result.mp4")))
                job._run()
            self.assertEqual(job.duration, duration)
            backend.open.assert_not_called()
            self.assertEqual(job.result.get_nowait()[0], "done")

    def test_backend_interrupted_error_is_cancellation(self):
        def render(options, stop_event, progress, on_duration):
            raise InterruptedError("Rendering cancelled")
        with patch.dict(sys.modules, export=SimpleNamespace(render=render)):
            job = ExportJob(SimpleNamespace(output=Path("result.mp4")))
            job.start()
            job.thread.join(2)
        self.assertEqual(job.result.get_nowait(), ("cancelled", None))

    def test_color_picker_applies_only_confirmed_color(self):
        application = Application.__new__(Application)
        application.root = Mock()
        application.custom_color = Mock()
        application.custom_color.get.return_value = "#00FF88"
        application.color_mode = Mock()
        application.color_swatch = Mock()
        dialog = Mock(result=None)
        application.color_dialog = Mock(return_value=dialog)
        application.choose_color()
        application.custom_color.set.assert_not_called()
        application.color_mode.set.assert_not_called()
        application.color_swatch.configure.assert_not_called()

        dialog.result = SimpleNamespace(hex="#ff0000")
        application.choose_color()
        self.assertEqual(application.color_dialog.call_args.kwargs["initialcolor"], "#00FF88")
        self.assertEqual(dialog.show.call_count, 2)
        application.color_swatch.configure.assert_called_once_with(background="#ff0000")
        application.custom_color.set.assert_called_once_with("#ff0000")
        application.color_mode.set.assert_called_once_with("Выбранный цвет")

    def test_missing_display_gives_runtime_error(self):
        class TclError(Exception):
            pass
        tkinter = SimpleNamespace(Tk=Mock(side_effect=TclError("no display")), TclError=TclError)
        with patch.dict(sys.modules, tkinter=tkinter), self.assertRaises(RuntimeError):
            launch()

    def test_mainloop_interrupt_cancels_worker_before_destroy(self):
        class TclError(Exception):
            pass
        root = Mock()
        root.mainloop.side_effect = KeyboardInterrupt
        job = Mock()
        tkinter = SimpleNamespace(Tk=lambda: root, TclError=TclError)
        with patch.dict(sys.modules, tkinter=tkinter), patch("gui.Application", return_value=SimpleNamespace(job=job)), \
                self.assertRaises(KeyboardInterrupt):
            launch()
        job.cancel.assert_called_once()
        job.thread.join.assert_called_once()
        root.destroy.assert_called_once()

    def test_mainloop_interrupt_joins_preview_before_destroy(self):
        class TclError(Exception):
            pass
        root, job, frame_job = Mock(), Mock(), Mock()
        root.mainloop.side_effect = KeyboardInterrupt
        calls = Mock()
        calls.attach_mock(job.cancel, "cancel")
        calls.attach_mock(job.thread.join, "join")
        calls.attach_mock(frame_job.cancel, "cancel_png")
        calls.attach_mock(frame_job.thread.join, "join_png")
        calls.attach_mock(root.destroy, "destroy")
        application = SimpleNamespace(job=None, preview=SimpleNamespace(
            job=job, playback=None, frame_job=frame_job, cache={"held": object()}))
        tkinter = SimpleNamespace(Tk=lambda: root, TclError=TclError)
        with patch.dict(sys.modules, tkinter=tkinter), patch("gui.Application", return_value=application), \
                self.assertRaises(KeyboardInterrupt):
            launch()
        self.assertEqual([call[0] for call in calls.mock_calls], ["cancel", "join", "cancel_png", "join_png", "destroy"])
        self.assertEqual(application.preview.cache, {})


if __name__ == "__main__":
    unittest.main()

"""Silent Linux CLI acceptance through a real PTY, with bounded disk capture."""

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import pty
import re
import select
import signal
import subprocess
import sys
import termios
import time


ROOT = Path(__file__).resolve().parents[1]
FRAME_START = b"\x1b[H\x1b[40m\x1b[97m"
FRAME_END = b"\x1b[0m"
CHILD = """import runpy,sys
sys.modules['audio']=None
sys.modules['sounddevice']=None
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""


def controlling_terminal():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Session:
    def __init__(self, name, fixture, output):
        self.master, self.slave = pty.openpty()
        termios.tcsetwinsize(self.slave, (30, 120))
        self.before = termios.tcgetattr(self.slave)
        self.raw = (output / (name + ".ansi")).open("wb")
        self.events = (output / (name + ".frames.jsonl")).open("w", encoding="utf-8")
        self.raw_digest = hashlib.sha256()
        self.pending = b""
        self.sequence_tail = b""
        self.counts = {"alt_enter": 0, "alt_leave": 0, "cursor_hide": 0,
                       "cursor_show": 0, "clear": 0}
        self.bytes = self.frames = 0
        self.last_geometry = None
        self.peak_pending = 0
        self.last_bytes = b""
        self.started = time.monotonic()
        environment = os.environ.copy()
        environment.update(TERM="xterm-256color", PYTHONPATH=str(ROOT / "src"),
                           PYTHONUNBUFFERED="1", LC_ALL="C.UTF-8")
        self.command = [sys.executable, "-c", CHILD, str(ROOT / "src/main.py"),
                        str(fixture), "--no-audio"]
        self.process = subprocess.Popen(self.command, stdin=self.slave, stdout=self.slave,
                                        stderr=self.slave, env=environment,
                                        preexec_fn=controlling_terminal, cwd=ROOT)

    def receive(self, chunk):
        self.raw.write(chunk)
        self.raw_digest.update(chunk)
        self.bytes += len(chunk)
        self.last_bytes = (self.last_bytes + chunk)[-4096:]
        sequences = {"alt_enter": b"\x1b[?1049h", "alt_leave": b"\x1b[?1049l",
                     "cursor_hide": b"\x1b[?25l", "cursor_show": b"\x1b[?25h",
                     "clear": b"\x1b[2J"}
        data = self.sequence_tail + chunk
        boundary = len(self.sequence_tail)
        for name, sequence in sequences.items():
            self.counts[name] += sum(match.end() > boundary
                                     for match in re.finditer(re.escape(sequence), data))
        self.sequence_tail = data[-16:]
        self.pending += chunk
        self.peak_pending = max(self.peak_pending, len(self.pending))
        if len(self.pending) > 65536:
            raise AssertionError("Frame parser exceeded 64 KiB")
        while True:
            start = self.pending.find(FRAME_START)
            if start < 0:
                self.pending = self.pending[-len(FRAME_START):]
                break
            self.pending = self.pending[start:]
            end = self.pending.find(FRAME_END, len(FRAME_START))
            if end < 0:
                break
            frame = self.pending[len(FRAME_START):end]
            self.pending = self.pending[end + len(FRAME_END):]
            rows = re.split(rb"\r*\n", frame)
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise AssertionError("Nonrectangular mono PTY frame")
            self.last_geometry = [widths.pop(), len(rows)]
            self.frames += 1
            self.events.write(json.dumps({"frame": self.frames,
                                          "elapsed": round(time.monotonic() - self.started, 6),
                                          "geometry": self.last_geometry,
                                          "sha256": hashlib.sha256(frame).hexdigest()}) + "\n")

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            readable, _, _ = select.select([self.master], [], [], min(.05, max(0, end - time.monotonic())))
            if readable:
                try:
                    chunk = os.read(self.master, 16384)
                except OSError as error:
                    if error.errno == errno.EIO:
                        return
                    raise
                if chunk:
                    self.receive(chunk)

    def until(self, predicate, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            self.pump(.05)
            if self.process.poll() is not None:
                self.pump(.1)
                return predicate()
        return predicate()

    def finish(self, timeout=20):
        if not self.until(lambda: self.process.poll() is not None, timeout):
            raise TimeoutError("Owned CLI process did not exit")
        self.pump(.15)
        after = termios.tcgetattr(self.slave)
        return {"exit_code": self.process.returncode, "elapsed": time.monotonic() - self.started,
                "frames": self.frames, "bytes": self.bytes,
                "raw_sha256": self.raw_digest.hexdigest(), "last_geometry": self.last_geometry,
                "ansi": self.counts.copy(), "termios_restored_exactly": self.before == after,
                "peak_parser_pending_bytes": self.peak_pending,
                "cleanup_at_end": self.last_bytes.endswith(b"\x1b[0m\x1b[?25h\x1b[?1049l"),
                "tail_error_text": re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", self.last_bytes[-200:]).decode("utf-8", "replace")}

    def close(self):
        if self.process.poll() is None:
            # This is only our own timed-out diagnostic child, never a user process.
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.raw.close()
        self.events.close()
        os.close(self.master)
        os.close(self.slave)


def run(fixture, output):
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must be inside this project")
    output.mkdir(parents=True, exist_ok=True)
    alias = output / "Пример видео с пробелами.mp4"
    if not alias.exists():
        alias.symlink_to(fixture.resolve())
    if alias.resolve() != fixture.resolve():
        raise ValueError("Existing Unicode fixture alias points elsewhere")
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "fixture": str(fixture), "utf8_input": str(alias), "audio": "--no-audio and blocked imports",
              "limits": "Real Linux PTY and CLI; no physical terminal emulator, display, or audio device.",
              "cases": {}}
    for name in ("pause_resize_eof", "ctrl_c", "resize_error"):
        session = Session(name, alias, output)
        try:
            if not session.until(lambda: session.frames >= 4):
                raise AssertionError("CLI did not produce four initial ASCII frames")
            attrs = termios.tcgetattr(session.slave)
            detail = {"initial_frame_geometry": session.last_geometry,
                      "cbreak_active": not bool(attrs[3] & (termios.ICANON | termios.ECHO)),
                      "isig_preserved": bool(attrs[3] & termios.ISIG)}
            if name == "pause_resize_eof":
                os.write(session.master, b" ")
                session.pump(.25)
                paused_frames = session.frames
                session.pump(.8)
                detail["paused_observation_seconds"] = .8
                detail["frames_during_settled_pause"] = session.frames - paused_frames
                os.write(session.master, b" ")
                detail["resume_produced_frames"] = session.until(lambda: session.frames >= paused_frames + 3, 3)
                termios.tcsetwinsize(session.slave, (40, 160))
                detail["resize_produced_expected_geometry"] = session.until(lambda: session.last_geometry == [159, 39], 3)
                detail["resized_frame_geometry"] = session.last_geometry
            elif name == "ctrl_c":
                os.write(session.master, b"\x03")
                detail["interrupt_source"] = "PTY VINTR byte 0x03, not direct signal"
            else:
                termios.tcsetwinsize(session.slave, (1, 1))
            detail.update(session.finish())
            report["cases"][name] = detail
        finally:
            session.close()
    cases = report["cases"]
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, case in cases.items():
        expected = 0 if name == "pause_resize_eof" else 130 if name == "ctrl_c" else 1
        assert case["exit_code"] == expected, (name, case["exit_code"])
        assert case["termios_restored_exactly"] and case["cbreak_active"] and case["isig_preserved"], name
        assert case["ansi"]["alt_enter"] == case["ansi"]["alt_leave"] == 1, name
        assert case["ansi"]["cursor_hide"] == case["ansi"]["cursor_show"] == 1, name
    normal = cases["pause_resize_eof"]
    assert normal["frames_during_settled_pause"] == 0
    assert normal["resume_produced_frames"] and normal["resize_produced_expected_geometry"]
    assert normal["ansi"]["clear"] == 2
    assert normal["cleanup_at_end"] and cases["ctrl_c"]["cleanup_at_end"]
    report["passed"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / ".test-media/muse-8s.mp4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".test-media/linux-pty")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT):
        parser.error("--output-dir must be inside this project")
    try:
        report = run(args.input, args.output_dir)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "failure.json").write_text(json.dumps({"error": repr(error)}), encoding="utf-8")
        raise
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

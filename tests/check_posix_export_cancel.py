"""Interrupt an installed POSIX CLI after real MP4 writes, with physical audio off."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(options):
    directory = Path(tempfile.mkdtemp(prefix="cancel-", dir=options.output_json.parent))
    output = directory / "existing result.mp4"
    sentinel = b"previous result must survive interruption\n"
    output.write_bytes(sentinel)
    log = directory / "cli.log"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [str(options.cli), "render", str(options.input), "--output", str(output),
               "--color", "source", "--no-audio", "--overwrite"]
    report = {"command": command, "audio": "physical audio off: --no-audio",
              "cli_sha256": digest(options.cli), "diagnostic_sha256": digest(Path(__file__)),
              "input": str(options.input), "directory": str(directory), "pythonpath_removed": True}
    # Inspect with the same clean environment as the CLI, rather than the
    # observer's possible PYTHONPATH pointing at the source checkout.
    inspect = """import hashlib, importlib.util, json
from pathlib import Path
result = {}
for name in ('cli', 'export', 'player', 'raster'):
    path = Path(importlib.util.find_spec(name).origin)
    result[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(result))
"""
    report["installed_modules"] = json.loads(subprocess.check_output(
        [sys.executable, "-c", inspect], env=environment, cwd=directory, text=True))
    signal_sent = False
    with log.open("wb") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, env=environment)
        report["pid"] = child.pid
        try:
            deadline = time.monotonic() + 30
            previous_size = 0
            while child.poll() is None and time.monotonic() < deadline:
                temporary = list(directory.glob(f".{output.name}.*.mp4"))
                size = sum(path.stat().st_size for path in temporary)
                if previous_size >= 4096 and size > previous_size:
                    report["temporary_bytes_before_signal"] = size
                    report["previous_observed_temporary_bytes"] = previous_size
                    child.send_signal(signal.SIGINT)
                    signal_sent = True
                    break
                previous_size = size
                time.sleep(.05)
            report["exit_code"] = child.wait(timeout=30)
        except BaseException:
            # This is exclusively the child created above, never another task.
            child.terminate()
            child.wait(timeout=10)
            raise
    report["sigint_sent_after_growth"] = signal_sent
    report["existing_output_preserved"] = output.read_bytes() == sentinel
    report["temporary_files_remaining"] = [str(path) for path in directory.glob(f".{output.name}.*.mp4")]
    report["traceback_present"] = "Traceback" in log.read_text(errors="replace")
    report["passed"] = (signal_sent and report["exit_code"] == 130
                        and report["existing_output_preserved"]
                        and not report["temporary_files_remaining"] and not report["traceback_present"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    options = parser.parse_args()
    if os.name != "posix":
        parser.error("POSIX is required")
    options.cli = options.cli.resolve(strict=True)
    if options.cli.parent != Path(sys.executable).absolute().parent:
        parser.error("--cli must belong to the Python environment running this diagnostic")
    options.input = options.input.resolve(strict=True)
    options.output_json = options.output_json.resolve()
    if options.output_json.exists():
        parser.error("report exists; choose a new --output-json")
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = run(options)
    options.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

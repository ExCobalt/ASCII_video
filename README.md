# ASCII Video

Play local video as ASCII art in your terminal with sound, compare the original and ASCII output in a graphical preview, and export to MP4, WebM, or PNG.

Current version: **v0.3.1** · Python **3.12** · Original source: [MIT](LICENSE) · Combined desktop builds: [GPLv3](DISTRIBUTION.md).

## Features

- **Terminal player** with synchronized audio, pause, seeking, and automatic grid sizing.
- **Graphical interface** for selecting a video and adjusting the character set, color, brightness, gamma, contrast, and dithering.
- **Preview** with the original and ASCII frames side by side, playback, seeking, 100% and 200% zoom, and panning. A checkbox enables audio, which is off by default.
- **Video export** to MP4 with H.264, HEVC, or AV1, or to WebM with VP9 or AV1. MP4 uses AAC audio and WebM uses Opus.
- **Frame export** to lossless PNG at full resolution.
- **Color** options include white characters, a custom color, and the source video's colors. Source-color mode also offers saturation and glyph brightness compensation.

ASCII Video processes one self-contained local file per run. For playlists or media that reference separate files or URLs, first save the video as a single file. Batch processing and playback speed adjustment are unavailable.

Use media from trusted sources. Malformed files can cause the bundled FFmpeg decoder to hang or crash.

The GUI labels are currently in Russian; command-line help is in English. Most of this project's code was generated with ChatGPT under the author's direction.

## Install and run

Install Python **3.12** with Tk and [uv](https://docs.astral.sh/uv/getting-started/installation/). Clone or download this repository, open a terminal in its root directory (the one containing `pyproject.toml`), and run:

```console
uv sync --locked --python 3.12
uv run --locked ascii-video
```

On Linux, install the system PortAudio library (`libportaudio2` on Debian and Ubuntu) and use a Python build with Tkinter support. For distribution-provided Python, Tk may be a separate `python3-tk` package. PyAV wheels bundle FFmpeg libraries, so a separate FFmpeg executable is unnecessary. A C compiler enables optional native acceleration; installation can fall back to Python without it.

The target platforms are Windows 10/11 x64, Linux x64, and macOS x64/arm64. The pinned PyAV wheels require macOS 11 or later on Intel and macOS 14 or later on Apple Silicon. Version 0.3.1 passed offline standalone tests on Windows 11 x64, Ubuntu 22.04 x64 and Debian 12 x64, including GUI preview, PNG output, terminal playback and all five software video export formats with audio. On Alpine 3.22 x64, the source package and Python wheel passed offline installation, preview and export checks. Install `binutils` and `portaudio-dev` alongside Python 3.12 with Tk so sounddevice can find PortAudio. Windows 10 and native macOS acceptance remain open. See the [build instructions and platform checks](BUILDING.md).

Windows v0.3.1 is distributed as source while the right to redistribute the bundled Microsoft Visual C++ Runtime awaits confirmation. Use the Python installation steps above, or [build a standalone EXE locally](BUILDING.md#windows-executable). Run the local EXE without arguments to open the GUI, or drop a single video file onto it to play the file in the terminal.

For a Linux x64 binary distribution, extract the entire archive, open a terminal in the extracted directory, and run `./ascii-video/ascii-video`. Keep the executable and its `_internal` directory together. The same executable accepts the CLI commands below. See the [build instructions](BUILDING.md).

## Command line

Play a video in its original colors.

```console
uv run --locked ascii-video play "movie.mp4" --color source
```

Play without audio and keep the terminal's original font scale.

```console
uv run --locked ascii-video play "movie.mp4" --font-scale 1 --no-audio
```

Export a video with one of these commands.

```console
uv run --locked ascii-video render "movie.mp4" --output "movie_ascii.mp4" --color source
uv run --locked ascii-video render "movie.mp4" --output "movie_ascii.webm" --container webm --codec vp9
uv run --locked ascii-video render "movie.mp4" --output "movie_hevc.mp4" --codec hevc --quality high
```

By default, export uses MP4/H.264, 1920×1080 resolution, 320 ASCII columns, 30 fps, and the `balanced` quality preset. Export includes audio when the source has it. Use `--no-audio` to exclude audio from both the output and the timeline. Export runs without opening an audio device. For H.264, `--encoder nvenc` requires a compatible NVIDIA GPU and driver. If NVENC fails, the export does not fall back to software encoding.

Use `--overwrite` to replace an existing output file. If an error occurs or you cancel before the new output is finalized, the existing file stays intact.

Useful options for both `play` and `render`:

| Option | Purpose |
| --- | --- |
| `--palette basic`, `medium`, or `extended` | Choose the character set; a custom ramp of unique printable ASCII characters is also accepted |
| `--color mono`, `source`, or `"#00FF88"` | White, original colors, or a fixed color on black |
| `--brightness`, `--gamma`, `--contrast` | Adjust the tone |
| `--color-saturation`, `--glyph-correction` | Adjust source-color saturation and compensate for glyph coverage |
| `--dither bayer` or `--dither floyd-steinberg` | Examples of the available dithering algorithms |

For export, `--resolution 1280x720 --columns 160` changes the output size and ASCII detail. Both pixel dimensions must be even. PNG output is available through the GUI preview's save button, not the `render` command.

Display the version and full command-line help:

```console
uv run --locked ascii-video --version
uv run --locked ascii-video play --help
uv run --locked ascii-video render --help
```

With a locally built Windows app, replace `uv run --locked ascii-video` with the path to `ascii-video.exe` (for example, `.\ascii-video.exe` in PowerShell).

## Controls

| Terminal action | Key or gesture |
| --- | --- |
| Pause or resume | Space |
| Seek backward or forward by 5 seconds | ← / → |
| Jump to a position | Click the timeline at the bottom |
| Quit | Ctrl+C |

Terminal playback requires ANSI/VT support and interactive standard output. Color output requires truecolor support. Use `--show-resolution` to display the current grid size. The default font scale is 0.8. Font scaling uses the classic Windows console or GNOME Terminal in a graphical Linux session. On Linux without GNOME Terminal, playback keeps the terminal’s current font size. Use `--font-scale 1` to keep the original scale.

In the preview, scroll to switch zoom levels, click to select an area, and drag to pan the zoomed frame. PNG export saves the full ASCII image at its original raster size. The preview audio checkbox applies only to the preview.

On a short display, use the main window's vertical scrollbar to reach the export buttons. Tab also scrolls the focused control into view.

## Limits for long exports

Exports longer than **6 hours** require explicit confirmation, as do video frames held for more than **60 seconds** and timestamp gaps longer than **60 seconds**. The app checks both the metadata and the actual timeline during processing.

The GUI prompts you to confirm a retry with these limits waived. On the command line, use `--allow-long-export` if you know the input is valid. The limits prevent accidentally oversized exports and apply to duration, frame holds, and timestamp gaps. Memory usage and file size are outside their scope.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Tkinter import fails or no GUI appears | Use Python with Tk support and a graphical desktop session. If several Python installations exist, select the intended interpreter by its full path with `uv sync --python`. Headless export uses `render` |
| PortAudio is missing on Linux | Install `libportaudio2` with your distribution's package manager |
| Terminal output is garbled or playback reports non-interactive output | Run in an ANSI/VT terminal without redirecting stdout; use a monospaced font and `--font-scale 1` if automatic scaling causes trouble |
| Audio device cannot open | Check the OS output device, or use `--no-audio` for terminal playback. Preview audio is a separate checkbox |
| NVENC fails | Check the NVIDIA GPU and driver, or choose `--encoder x264` |
| Output file already exists | Choose a new name or explicitly pass `--overwrite` |
| Native extension fails to build | See the compiler prerequisites and Python fallback in [BUILDING.md](BUILDING.md) |

## Development

Application code lives directly in `src/` and tests live in `tests/`. `uv.lock` pins runtime and build-group dependencies. See [BUILDING.md](BUILDING.md) for wheel, source archive, and native application builds.

Use the test runner to check the affected code. It disables physical audio output and fails if the pattern selects no tests.

```console
uv run --locked python tests/run_suite.py --pattern test_export_limits.py
uv run --locked python tests/run_suite.py --pattern test_raster.py --pure-python
```

Run tests for the changed logic and directly related scenarios. Run the full suite only when explicitly requested.

## License

Original project code is available under the [MIT License](LICENSE), including its warranty disclaimer. Fonts, adapted algorithms, and dependencies retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The combined Windows, Linux, and macOS programs with PyAV/FFmpeg and x264/x265 are distributed under **GNU GPL version 3**. Each binary release must offer the source companion that matches its executable. The original source files retain their MIT permissions. See the [distribution terms](DISTRIBUTION.md) and [release preparation instructions](BUILDING.md#before-publishing-binaries).

Installing PyAV from a wheel with `uv sync` also installs the FFmpeg libraries and codecs in that wheel. The desktop build bundles those installed libraries.

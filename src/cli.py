"""Command-line dispatch; help does not load media or audio libraries."""

from pathlib import Path
import sys

# Preserve imports used by existing diagnostics and callers.
from cli_options import (PALETTES, VERSION, bounded_float, bounded_integer, color,
                         image_options, palette, parser, resolution, export_format, ExportLimitError)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["gui"]
    if arguments[0] not in ("play", "render", "gui") and not arguments[0].startswith("-"):
        if Path(arguments[0]).is_file():
            arguments.insert(0, "play")
    try:
        argument_parser = parser()
        options = argument_parser.parse_args(arguments)
        if options.command == "render":
            try:
                export_format(options.output, options.container, options.codec, options.encoder, options.resolution)
            except ValueError as error:
                argument_parser.error(str(error))
    except SystemExit as error:
        return int(error.code)
    try:
        if options.command == "play":
            from console_font import relaunch_classic, relaunch_linux
            status = relaunch_classic(arguments, options.font_scale)
            if status is not None:
                return status
            status = relaunch_linux(arguments, options.font_scale)
            if status is not None:
                return status
            from player import play
            play(options)
        elif options.command == "render":
            from export import render
            from progress import ConsoleProgress
            progress = ConsoleProgress()
            try:
                render(options, progress=progress, on_duration=progress.set_duration)
            except BaseException:
                progress.finish()
                raise
            progress.finish(options.output)
        else:
            from gui import launch
            launch()
    except KeyboardInterrupt:
        return 130
    except ExportLimitError as error:
        print(f"ascii-video: {error} Use --allow-long-export to confirm.", file=sys.stderr)
        return 1
    # Only audible playback imports sounddevice; help and silent export stay independent.
    except getattr(sys.modules.get("sounddevice"), "PortAudioError", ()) as error:
        print(f"ascii-video: Audio output failed: {error}. Check the output device or use --no-audio.",
              file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"ascii-video: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

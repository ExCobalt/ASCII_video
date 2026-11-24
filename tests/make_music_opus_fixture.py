"""Create silent diagnostic H.264/Opus MKV copies of local music videos."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av

from media import normalize_audio_timing, probe


def video_digest(path):
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        video = container.streams[probe(path).video_index]
        for packet in container.demux(video):
            if packet.dts is not None:
                payload = bytes(packet)
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
                count += 1
    return {"packets": count, "sha256": digest.hexdigest()}


def convert(source, destination, duration):
    info = probe(source)
    if info.audio_index is None:
        raise ValueError("Input has no audio")
    with av.open(str(source)) as container, av.open(str(destination), "w") as output:
        video = container.streams[info.video_index]
        audio = container.streams[info.audio_index]
        copied_video = output.add_stream_from_template(video)
        encoded_audio = output.add_stream("libopus", rate=48000)
        encoded_audio.layout = "stereo"
        encoded_audio.bit_rate = 160000
        encoded_audio.codec_context.time_base = Fraction(1, 48000)
        encoded_audio.options = {"frame_duration": duration}
        resampler = av.AudioResampler(format="flt", layout="stereo", rate=48000)
        previous = None

        def encode(frames):
            for frame in frames:
                for packet in encoded_audio.encode(frame):
                    output.mux(packet)

        for packet in container.demux(video, audio):
            if packet.stream.index == video.index:
                if packet.dts is not None:
                    packet.stream = copied_video
                    output.mux(packet)
            else:
                for frame in packet.decode():
                    previous, discontinuity = normalize_audio_timing(frame, previous)
                    if discontinuity:
                        raise ValueError("Fixture creator requires continuous source audio")
                    encode(resampler.resample(frame))
        encode(resampler.resample(None))
        for packet in encoded_audio.encode(None):
            output.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-duration", choices=("20", "60"), default="20")
    args = parser.parse_args()
    sources = [source.resolve() for source in args.input]
    destinations = [args.directory / (source.stem + "-opus" + args.frame_duration + ".mkv")
                    for source in args.input]
    resolved_destinations = [path.resolve() for path in destinations]
    if len(set(resolved_destinations)) != len(destinations):
        parser.error("Inputs would produce the same fixture path")
    for destination in destinations:
        if destination.exists() or destination.resolve() in sources:
            parser.error(f"Fixture path already exists or matches an input: {destination}")
    for protected in sources + resolved_destinations:
        if (args.output.resolve() == protected
                or (args.output.exists() and protected.exists() and args.output.samefile(protected))):
            parser.error("Report path must differ from every input and fixture")
    args.directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"physical_audio": False, "pyav": av.__version__, "passed": False,
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    for source, destination in zip(args.input, destinations):
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        convert(source, destination, args.frame_duration)
        original_video, copied_video = video_digest(source), video_digest(destination)
        assert original_video == copied_video, "Remux changed video payloads"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        with av.open(str(destination)) as container:
            stream = container.streams.audio[0]
            assert stream.codec_context.name == "opus"
            assert stream.codec_context.layout.name == "stereo"
            assert stream.codec_context.sample_rate == 48000
            output_duration = container.duration / av.time_base
        report["cases"].append({"source": str(source), "input_sha256": source_hash,
            "fixture": str(destination), "fixture_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "video_payloads_exact": True, "video": copied_video, "duration": output_duration,
            "audio_codec": "opus", "layout": "stereo", "rate": 48000,
            "frame_duration_ms": args.frame_duration})
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

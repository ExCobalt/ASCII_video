"""Single-consumer Opus decoding with bounded, process-local checkpoints.

Uses the public libopus state-copy API from the installed PyAV binary wheel.
Snapshots never cross libraries, processes or sessions. No audio device I/O.
"""
import ctypes as ct
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import chain
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np

from media import normalize_audio_timing, open_media


class UnsupportedOpusCursor(RuntimeError):
    """Select the ordinary decoder before any session PCM is produced."""


class Decoder:
    def __init__(self, lib, channels, *, gain=0, multistream=None):
        if not -32768 <= gain <= 32767:
            raise ValueError('Opus gain must fit signed Q8 dB')
        self.lib, self.channels = lib, channels
        error = ct.c_int()
        if multistream is None:
            self._decode, self._destroy, ctl = lib.opus_decode_float, lib.opus_decoder_destroy, lib.opus_decoder_ctl
            self.state = lib.opus_decoder_create(48000, channels, ct.byref(error))
            self.size = lib.opus_decoder_get_size(channels)
        else:
            streams, coupled, mapping = multistream
            if len(mapping) != channels:
                raise ValueError('Opus mapping must specify each output channel')
            table = (ct.c_ubyte * channels)(*mapping)
            self._decode = lib.opus_multistream_decode_float
            self._destroy, ctl = lib.opus_multistream_decoder_destroy, lib.opus_multistream_decoder_ctl
            self.state = lib.opus_multistream_decoder_create(48000, channels, streams,
                coupled, table, ct.byref(error))
            self.size = lib.opus_multistream_decoder_get_size(streams, coupled)
        try:
            if not self.state or error.value != 0 or self.size <= 0:
                raise RuntimeError(f'Cannot create Opus decoder: {error.value}')
            if gain:
                # OPUS_SET_GAIN(gain): request 4034 plus signed opus_int32.
                result = ctl(self.state, 4034, ct.c_int32(gain))
                if result != 0:
                    raise RuntimeError(f'Cannot set Opus decoder gain: {result}')
            self.buffer = np.empty((5760, channels), dtype=np.float32)
            self._output = self.buffer.ctypes.data_as(ct.POINTER(ct.c_float))
        except Exception:
            self.__exit__()
            raise

    def decode(self, raw, samples, lost=False):
        if self.state is None:
            raise RuntimeError('Opus decoder closed')
        if not 0 < samples <= 5760:
            raise ValueError(f'Invalid Opus sample count: {samples}')
        count = self._decode(self.state, None if lost else raw,
            0 if lost else len(raw), self._output,
            samples if lost else 5760, 0)
        if count < 0:
            raise RuntimeError(f'Cannot decode Opus packet: {count}')
        if count != samples:
            raise RuntimeError(f'Unexpected Opus sample count: {count}, expected {samples}')
        return self.buffer[:count]

    def snapshot(self):
        if self.state is None:
            raise RuntimeError('Opus decoder closed')
        return ct.string_at(self.state, self.size)

    def restore(self, snapshot):
        if self.state is None:
            raise RuntimeError('Opus decoder closed')
        if not isinstance(snapshot, bytes) or len(snapshot) != self.size:
            raise ValueError('Invalid Opus decoder snapshot')
        ct.memmove(self.state, snapshot, self.size)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.state is not None:
            self._destroy(self.state)
            self.state = None


def library():
    package = Path(av.__file__).parent
    if sys.platform == 'darwin':
        paths = list((package / '.dylibs').glob('libopus.*.dylib'))
    else:
        directory = package.parent / 'av.libs'
        paths = list(directory.glob('*opus*.dll' if sys.platform == 'win32' else '*opus*.so*'))
    if len(paths) != 1:
        raise OSError('Cannot locate the bundled libopus library')
    lib = ct.CDLL(str(paths[0]))
    lib.opus_decoder_create.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_int)]
    lib.opus_decoder_create.restype = ct.c_void_p
    lib.opus_decoder_destroy.argtypes = [ct.c_void_p]
    lib.opus_decoder_destroy.restype = None
    # Only the fixed arguments of the public variadic CTL entry point.
    lib.opus_decoder_ctl.argtypes = [ct.c_void_p, ct.c_int]
    lib.opus_decoder_ctl.restype = ct.c_int
    lib.opus_decoder_get_size.argtypes = [ct.c_int]
    lib.opus_decoder_get_size.restype = ct.c_int
    lib.opus_decode_float.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int,
                                    ct.POINTER(ct.c_float), ct.c_int, ct.c_int]
    lib.opus_decode_float.restype = ct.c_int
    lib.opus_multistream_decoder_create.argtypes = [ct.c_int, ct.c_int, ct.c_int,
        ct.c_int, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_int)]
    lib.opus_multistream_decoder_create.restype = ct.c_void_p
    lib.opus_multistream_decoder_get_size.argtypes = [ct.c_int, ct.c_int]
    lib.opus_multistream_decoder_get_size.restype = ct.c_int
    lib.opus_multistream_decoder_destroy.argtypes = [ct.c_void_p]
    lib.opus_multistream_decoder_destroy.restype = None
    lib.opus_multistream_decoder_ctl.argtypes = [ct.c_void_p, ct.c_int]
    lib.opus_multistream_decoder_ctl.restype = ct.c_int
    lib.opus_multistream_decode_float.argtypes = lib.opus_decode_float.argtypes
    lib.opus_multistream_decode_float.restype = ct.c_int
    lib.opus_packet_get_nb_samples.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
    lib.opus_packet_get_nb_samples.restype = ct.c_int
    lib.opus_get_version_string.restype = ct.c_char_p
    return lib, paths[0]


@dataclass(frozen=True)
class Anchor:
    position: int
    pts: int
    prefix_samples: int
    payload_sha256: str
    state: bytes



@dataclass(frozen=True)
class Checkpoint:
    anchor: Anchor
    previous_end: Fraction | None
    phase: int | None
    start: Fraction
    skip_remaining: int = 0
    metadata_sha256: str = ''


def packet_metadata(packet, skip_remaining):
    digest = hashlib.sha256(str((packet.time_base, packet.is_discard)).encode('ascii'))
    discard = 0
    for side in packet.iter_sidedata():
        if side.data_type == 'skip_samples':
            data = bytes(side)
            if len(data) < 10:
                raise RuntimeError('Invalid skip_samples side data')
            incoming = int.from_bytes(data[:4], 'little', signed=True)
            if incoming:
                skip_remaining = max(0, incoming)
            discard = int.from_bytes(data[4:8], 'little')
            continue
        digest.update(side.data_type.encode('ascii'))
        digest.update(bytes(side))
    # Demuxers can omit/reissue equivalent skip side data after a seek.
    # Compare its effective trimming, not the presence of the side-data record.
    digest.update(str((skip_remaining, discard)).encode('ascii'))
    return digest.hexdigest()


def codec_for(stream):
    codec = av.CodecContext.create('libopus', 'r')
    codec.extradata, codec.sample_rate, codec.layout = stream.codec_context.extradata, 48000, stream.layout
    codec.options = {'request_sample_fmt': 'flt', 'pkt_timebase': str(stream.time_base)}
    return codec


class OpusAudioCursor:
    """One consumer, one live container/decoder, bounded lazily built snapshots."""
    def __init__(self, path, lib=None, *, stream_index=None):
        if lib is None and (sys.platform not in ('win32', 'linux', 'darwin') or
                any(av.library_versions[name][0] != 63 for name in ('libavcodec', 'libavformat'))):
            raise UnsupportedOpusCursor('Untested Opus runtime')
        self.path, self.lib = path, lib
        with open_media(path) as container:
            stream = (container.streams.audio[0] if stream_index is None
                      else container.streams[stream_index])
            if stream.type != 'audio':
                raise ValueError('Selected stream is not audio')
            if stream.codec_context.name != 'opus':
                raise UnsupportedOpusCursor('Not an Opus stream')
            self.stream_index = stream.index
            head = stream.codec_context.extradata
            if not head or len(head) < 19 or head[:8] != b'OpusHead' or head[9] == 0:
                raise RuntimeError('Invalid or unsupported Opus header')
            if head[8] > 15 or head[18] not in (0, 1):
                raise UnsupportedOpusCursor('Unsupported Opus channel mapping family')
            if head[9] > 8:
                raise RuntimeError('Invalid Opus channel count for family 0/1')
            if 'libopus' not in av.codecs_available:
                raise UnsupportedOpusCursor('FFmpeg libopus is unavailable')
            self.channels, self.preskip = head[9], int.from_bytes(head[10:12], 'little')
            self.gain = int.from_bytes(head[16:18], 'little', signed=True)
            self.multistream = None
            if head[18] == 0:
                if self.channels > 2:
                    raise RuntimeError('Family 0 supports only mono/stereo')
            elif head[18] == 1 and len(head) >= 21 + self.channels:
                streams, coupled = head[19:21]
                mapping = head[21:21 + self.channels]
                if not 1 <= streams <= 255 or coupled > streams or streams + coupled > 255 or any(
                        value != 255 and value >= streams + coupled for value in mapping):
                    raise RuntimeError('Invalid Opus channel mapping')
                self.multistream = streams, coupled, mapping
            else:
                raise RuntimeError('Unsupported or truncated Opus channel mapping')
            if lib is None:
                try:
                    self.lib, _ = library()
                except (OSError, AttributeError) as error:
                    raise UnsupportedOpusCursor('Bundled libopus is unavailable') from error
            size = (self.lib.opus_decoder_get_size(self.channels) if self.multistream is None
                    else self.lib.opus_multistream_decoder_get_size(*self.multistream[:2]))
            if size <= 0 or (2 * 1024 * 1024) // size < 2:
                raise UnsupportedOpusCursor('Opus decoder snapshots exceed the memory budget')
            codec = codec_for(stream)
            packets = container.demux(stream)
            initial = next(packet for packet in packets if packet.size)
            for side in initial.iter_sidedata():
                if side.data_type == 'skip_samples':
                    raw = bytes(side)
                    if len(raw) < 10:
                        raise RuntimeError('Invalid initial skip_samples side data')
                    # The pinned FFmpeg replaces pending skip only for a
                    # nonzero side-data value; zero preserves OpusHead's skip.
                    incoming = int.from_bytes(raw[:4], 'little', signed=True)
                    if incoming:
                        self.preskip = max(0, incoming)
            first = next(frame for packet in chain((initial,), packets) for frame in codec.decode(packet))
            self.layout = first.layout.name
            if self.multistream is not None:
                # RFC 7845 family 1 uses Vorbis order; FFmpeg frames use speaker order.
                vorbis = (('FC',), ('FL', 'FR'), ('FL', 'FC', 'FR'),
                    ('FL', 'FR', 'BL', 'BR'), ('FL', 'FC', 'FR', 'BL', 'BR'),
                    ('FL', 'FC', 'FR', 'BL', 'BR', 'LFE'),
                    ('FL', 'FC', 'FR', 'SL', 'SR', 'BC', 'LFE'),
                    ('FL', 'FC', 'FR', 'SL', 'SR', 'BL', 'BR', 'LFE'))[self.channels - 1]
                streams, coupled, mapping = self.multistream
                self.multistream = streams, coupled, tuple(mapping[vorbis.index(channel.name)]
                    for channel in first.layout.channels)
            self.origin = first.pts * first.time_base
        self._container = self._decoder = self._packets = None
        self._input = ExitStack()
        self.points = []
        self.end = self.phase = None
        self._total = self.covered_samples = self._next_sample = 0
        self._interval = 24000
        self.opens = self.restores = self.decoded_frames = self.compactions = 0
        self.max_snapshot_bytes = 0
        self.complete = self.closed = False
        self.request_closed = True
        self._checkpoints_enabled = True
        self._previous_key = None

    def _open(self):
        self._container = self._input.enter_context(open_media(self.path))
        stream = self._container.streams[self.stream_index]
        self._packets = self._container.demux(stream)
        self._decoder = Decoder(self.lib, self.channels, gain=self.gain, multistream=self.multistream)
        self._point_limit = min(64, (2 * 1024 * 1024) // self._decoder.size)
        if self._point_limit < 2:
            raise RuntimeError('Opus decoder snapshots exceed the memory budget')
        self._total = 0
        self._skip_remaining = self.preskip
        self.end = self.phase = None
        self.complete = False
        self._previous_key = None
        self.opens += 1

    def _restore(self, point, stopped):
        stream = self._container.streams[self.stream_index]
        anchor = point.anchor
        self._container.seek(anchor.pts - round(Fraction(1, 4) / stream.time_base), stream=stream, backward=True)
        self._packets = self._container.demux(stream)
        for packet in self._packets:
            if stopped():
                # Seek moved the demuxer, but decoder state has not moved yet.
                # Drop live resources so a later request starts from an anchor.
                self._release()
                return None
            if packet.pos == anchor.position and packet.pts == anchor.pts:
                if (hashlib.sha256(bytes(packet)).hexdigest() != anchor.payload_sha256 or
                        packet_metadata(packet, point.skip_remaining) != point.metadata_sha256):
                    return self._restart_without_checkpoints()
                # A laced block or Ogg page can share its physical position.
                # Verify that this timestamp names only one packet in that group.
                following, buffered_bytes = [], 0
                for after in self._packets:
                    if stopped():
                        self._release()
                        return None
                    following.append(after)
                    buffered_bytes += after.size
                    if len(following) > 128 or buffered_bytes > 1024 * 1024:
                        return self._restart_without_checkpoints()
                    if after.size and after.pos == anchor.position and after.pts == anchor.pts:
                        return self._restart_without_checkpoints()
                    if after.size and (after.pos is None or after.pos != anchor.position):
                        break
                self._packets = chain(following, self._packets)
                self._decoder.restore(anchor.state)
                self.end, self.phase = point.previous_end, point.phase
                self._total = anchor.prefix_samples
                self._skip_remaining = point.skip_remaining
                self.complete = False
                self._previous_key = None
                self.restores += 1
                return packet
        return self._restart_without_checkpoints()

    def _restart_without_checkpoints(self):
        """A changed/unlocatable anchor resumes sequentially with the same codec."""
        self.points.clear()
        self._checkpoints_enabled = False
        self._release()
        self._open()
        return next(self._packets, None)

    def frames(self, target, stopped=lambda: False):
        target = Fraction(str(target))
        if self.closed:
            raise RuntimeError('Cursor closed')
        if stopped():
            return
        self.request_closed = False
        pending = None
        try:
            opened = self._container is None
            if opened:
                self._open()
            if self.points:
                point = max((point for point in self.points if point.start <= target),
                            key=lambda point: point.start, default=self.points[0])
                if opened or self.end is not None and target < self.end or point.anchor.prefix_samples > self._total:
                    pending = self._restore(point, stopped)
                    if pending is None:
                        return
            elif self.end is not None and target < self.end:
                self._release()
                self._open()
            while not stopped() and not self.complete:
                if pending is not None:
                    packet, pending = pending, None
                else:
                    packet = next(self._packets, None)
                if packet is None:
                    self.complete = True
                    return
                if not packet.size:
                    continue
                key = (packet.pos, packet.pts)
                if (None in key or packet.pos < 0 or self._previous_key is not None and
                        key <= self._previous_key):
                    # Repeated/nonmonotonic identities cannot name a unique anchor.
                    self.points.clear()
                    self._checkpoints_enabled = False
                self._previous_key = key if None not in key else None
                raw = bytes(packet)
                samples = self.lib.opus_packet_get_nb_samples(raw, len(raw), 48000)
                new = self._total >= self.covered_samples
                previous_skip = self._skip_remaining
                discard = 0
                for side in packet.iter_sidedata():
                    if side.data_type == 'skip_samples':
                        data = bytes(side)
                        if len(data) < 10:
                            raise RuntimeError('Invalid skip_samples side data')
                        incoming = int.from_bytes(data[:4], 'little', signed=True)
                        if incoming:
                            self._skip_remaining = max(0, incoming)
                        discard = int.from_bytes(data[4:8], 'little')
                skip = min(samples, self._skip_remaining)
                # FFmpeg ignores padding larger than the remaining frame.
                if discard > samples - skip:
                    discard = 0
                audible = not packet.is_discard and skip + discard < samples
                state = self._decoder.snapshot() if self._checkpoints_enabled and audible and (not self.points or new and self._total >= self._next_sample) else None
                previous_end, previous_phase = self.end, self.phase
                block = self._decoder.decode(raw, samples)
                self._skip_remaining -= skip
                if not audible:
                    # Decode discarded packets and consume pending skip, but
                    # do not issue PCM or advance the audible sample clock.
                    self._total += samples
                    self.covered_samples = max(self.covered_samples, self._total)
                    self.decoded_frames += 1
                    continue
                pts = packet.pts
                if skip:
                    pts += math.floor(Fraction(skip, 48000) / packet.time_base + Fraction(1, 2))
                # Advance the exact clock without copying skipped PCM into AV
                # frames. The native decoder must still consume every packet.
                frame = SimpleNamespace(pts=pts, time_base=packet.time_base,
                    samples=samples - skip - discard, sample_rate=48000)
                self.end, discontinuity = normalize_audio_timing(frame, self.end)
                if self.phase is None or discontinuity:
                    self.phase = frame.pts
                if state is not None:
                    self.points.append(Checkpoint(Anchor(packet.pos, packet.pts, self._total,
                        hashlib.sha256(raw).hexdigest(), state), previous_end, previous_phase,
                        Fraction(frame.pts, 48000), previous_skip, packet_metadata(packet, previous_skip)))
                    self.max_snapshot_bytes = max(self.max_snapshot_bytes, sum(len(point.anchor.state) for point in self.points))
                    if len(self.points) > 64 or self.max_snapshot_bytes > 2 * 1024 * 1024:
                        raise RuntimeError('Opus checkpoint memory budget exceeded')
                    if len(self.points) == self._point_limit:
                        self.points = self.points[::2]
                        self._interval *= 2
                        self.compactions += 1
                    self._next_sample = self._total + self._interval
                self._total += samples
                self.covered_samples = max(self.covered_samples, self._total)
                self.decoded_frames += 1
                if stopped():
                    return
                if self.end > target:
                    output = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(block[skip:samples - discard].reshape(1, -1)),
                        format='flt', layout=self.layout)
                    output.sample_rate = 48000
                    output.pts, output.time_base = frame.pts, frame.time_base
                    yield output
        except Exception:
            self._release()
            raise
        finally:
            self.request_closed = True

    def _release(self):
        if self._decoder is not None:
            self._decoder.__exit__(None, None, None)
        self._input.close()
        self._container = self._decoder = self._packets = None

    def close(self):
        self._release()
        self.points.clear()
        self.closed = True


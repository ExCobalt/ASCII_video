#!/usr/bin/env bash
# Run in tests/linux-clean.Dockerfile, with /input.mp4 and writable /out mounted.
set -euo pipefail
if command -v python3 >/dev/null || command -v python >/dev/null || command -v ffmpeg >/dev/null; then
    echo 'Clean acceptance image unexpectedly has Python or FFmpeg.' >&2
    exit 1
fi
app=/opt/ascii-video/ascii-video
test -f /input.mp4
mkdir -p /out
"$app" --help > /out/help.txt
"$app" --version > /out/version.txt
"$app" render /input.mp4 --output /out/silent.mp4 --resolution 128x64 \
    --columns 16 --color source --no-audio --overwrite > /out/export.log 2>&1
test -s /out/silent.mp4
for format in mp4/h264 mp4/hevc mp4/av1 webm/vp9 webm/av1; do
    container=${format%/*}
    codec=${format#*/}
    output="/out/$codec.$container"
    "$app" render /input.mp4 --output "$output" --container "$container" --codec "$codec" \
        --resolution 128x64 --columns 16 --color source --dither blue-noise --overwrite >> /out/export.log 2>&1
    test -s "$output"
done
TERM=xterm-256color timeout 30 script -qec \
    "stty rows 24 cols 80; exec $app play /input.mp4 --no-audio --font-scale 1" /out/player.ansi </dev/null >/dev/null
grep -Fq $'\033[?1049h' /out/player.ansi
grep -Fq $'\033[?1049l' /out/player.ansi
# PulseAudio's virtual sink consumes audio at wall-clock speed. ALSA's null
# PCM can consume the prebuffer faster than the decoder can refill it.
test ! -e /dev/snd
pulseaudio -n --daemonize=yes --exit-idle-time=-1 \
    --load='module-native-protocol-unix socket=/tmp/ascii-pulse.sock auth-anonymous=1' \
    --load='module-null-sink sink_name=silent'
printf '%s\n' 'pcm.!default { type pulse hint { show on description "Silent acceptance device" } }' > /tmp/ascii-pulse.conf
PULSE_SERVER=unix:/tmp/ascii-pulse.sock ALSA_CONFIG_PATH=/tmp/ascii-pulse.conf TERM=xterm-256color timeout 30 script -qec \
    "stty rows 24 cols 80; exec $app play /input.mp4 --font-scale 1" /out/player-null-audio.ansi </dev/null >/dev/null
grep -Fq $'\033[?1049l' /out/player-null-audio.ansi
xvfb-run -a -s '-screen 0 1920x1080x24' timeout 90 bash /check-window.sh
printf '%s\n' 'Packaged help/version, software exports, terminal with muted/virtual audio, GUI preview/playback and PNG passed on clean Linux + Xvfb.' \
    'Screenshot needs visual review; a virtual X11 desktop is not physical desktop/audio acceptance.' > /out/result.txt

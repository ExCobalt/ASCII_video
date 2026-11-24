#!/usr/bin/env bash
# Run on a Mac with a graphical login, using a short input containing audio.
set -euo pipefail
driver=$(cd "$(dirname "$0")" && pwd)/check_macos_window.swift
app=$(cd "$1/Contents/MacOS" && pwd)/ASCII-Video
input=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
cd "$3"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH

"$app" --version
for format in mp4/h264 mp4/hevc mp4/av1 webm/vp9 webm/av1; do
    container=${format%/*}
    codec=${format#*/}
    "$app" render "$input" --output "$codec.$container" --container "$container" --codec "$codec" \
        --resolution 128x64 --columns 16 --dither blue-noise --overwrite
    test -s "$codec.$container"
done
TERM=xterm-256color script -q player.ansi /bin/sh -c \
    'stty rows 24 cols 80; "$1" play "$2" --font-scale 1 --no-audio; echo "$?" > "$3"' \
    sh "$app" "$input" "$PWD/player-exit" </dev/null >/dev/null
test "$(cat player-exit)" = 0
grep -Fq $'\033[?1049h' player.ansi
grep -Fq $'\033[?1049l' player.ansi
swift "$driver" "$app"

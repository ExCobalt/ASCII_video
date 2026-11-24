#!/usr/bin/env bash
# Drive the real packaged windows on the clean image's fixed 1920x1080 Xvfb screen.
set -euo pipefail
export LC_ALL=C.UTF-8
/opt/ascii-video/ascii-video gui > /out/gui.log 2>&1 &
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null || true; wait "$app_pid" 2>/dev/null || true' EXIT

main=$(timeout 20 xdotool search --sync --onlyvisible --name '^ASCII Video$')
xdotool windowfocus "$main"
import -window "$main" /out/packaged-main.png
xdotool mousemove --window "$main" 140 26 click 1 type /input.mp4
xdotool mousemove --window "$main" 295 620 click 1
preview=$(timeout 15 xdotool search --sync --onlyvisible --class '^Toplevel$')

# A moving source lets the original-frame pane verify muted playback.
frame_pixels() {
    import -window "$preview" -crop 160x80+100+100 -depth 8 rgb:- | sha256sum
}
sleep 1
import -window "$preview" /out/packaged-preview.png
before=$(frame_pixels)
xdotool mousemove --window "$preview" 535 332 click 1
for _ in $(seq 1 20); do
    [[ "$(frame_pixels)" != "$before" ]] && break
    sleep 0.1
done
[[ "$(frame_pixels)" != "$before" ]]

# Seek, zoom and pan through the same input bindings used on a desktop.
xdotool mousemove --window "$preview" 300 370 click 1
xdotool mousemove --window "$preview" 224 150 click 4 mousedown 1
xdotool mousemove --window "$preview" 260 170 mouseup 1
sleep 1
xdotool mousemove --window "$preview" 374 332 click 1
dialog=$(timeout 15 xdotool search --sync --onlyvisible --class '^TkFDialog$')
xdotool windowfocus "$dialog"
xdotool mousemove --window "$dialog" 200 203 click 1 key Home shift+End
directory=$(mktemp -d /out/preview-XXXXXX)
xdotool type "$directory/frame.png"
xdotool key Return
for _ in $(seq 1 50); do
    [[ -s "$directory/frame.png" ]] && break
    sleep 0.1
done
[[ "$(identify -format '%m %wx%h' "$directory/frame.png")" == 'PNG 1920x1080' ]]
mv "$directory/frame.png" /out/frame.png
rmdir "$directory"

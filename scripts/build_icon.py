"""Create the Windows icon using the project's existing Pillow dependency."""

from pathlib import Path

from PIL import Image, ImageDraw


assets = Path(__file__).resolve().parents[1] / "assets"
image = Image.new("RGBA", (1024, 1024))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((48, 48, 976, 976), radius=180, fill="#101b2b")
draw.rounded_rectangle((112, 112, 912, 912), radius=120,
                       outline="#33465f", width=16)
# Film perforations frame the familiar terminal prompt.
for x in (240, 400, 560, 720):
    for y in (170, 786):
        draw.rounded_rectangle((x, y, x + 64, y + 68), radius=14,
                               fill="#657b96")
draw.line(((260, 352), (444, 512), (260, 672)), fill="#5ff5b7", width=76,
          joint="curve")
draw.rounded_rectangle((518, 624, 774, 690), radius=14, fill="#5ff5b7")
image = image.resize((256, 256), Image.Resampling.LANCZOS)
image.save(assets / "ascii-video.png")
image.save(assets / "ascii-video.ico", sizes=[(16, 16), (24, 24), (32, 32),
                                            (48, 48), (64, 64), (128, 128),
                                            (256, 256)])

"""Draw the app icon: the Instinct spark on a dark rounded square.

python macos/make_icon.py OUT.iconset
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
TOP, BOTTOM = (0x24, 0x34, 0x3C), (0x12, 0x1A, 0x1F)
SPARK = (0xA2, 0xD5, 0xC9)


def spark(center: float, radius: float, waist: float) -> list[tuple[float, float]]:
    """A four-pointed star whose sides curve inwards."""
    points = []
    for step in range(256):
        # A superellipse with exponent < 1 gives the pinched, curved-in sides.
        t = step / 256 * math.tau
        c, s = math.cos(t), math.sin(t)
        x = radius * math.copysign(abs(c) ** (1 / waist), c)
        y = radius * math.copysign(abs(s) ** (1 / waist), s)
        points.append((center + x, center + y))
    return points


def draw() -> Image.Image:
    # macOS icon grid: an 824px rounded square centred in 1024.
    inset, body = 100, 824
    column = Image.new("RGB", (1, SIZE))
    for y in range(SIZE):
        t = y / (SIZE - 1)
        column.putpixel(
            (0, y), tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
        )
    gradient = column.resize((SIZE, SIZE))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (inset, inset, inset + body, inset + body), radius=185, fill=255
    )
    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    icon.paste(gradient, mask=mask)

    star = spark(SIZE / 2, 300, 0.25)
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(star, fill=SPARK + (150,))
    glow = glow.filter(ImageFilter.GaussianBlur(48))
    icon.alpha_composite(Image.composite(glow, Image.new("RGBA", glow.size), mask))
    ImageDraw.Draw(icon).polygon(star, fill=SPARK + (255,))
    return icon


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    icon = draw()
    for size in (16, 32, 128, 256, 512):
        icon.resize((size, size), Image.LANCZOS).save(out / f"icon_{size}x{size}.png")
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            out / f"icon_{size}x{size}@2x.png"
        )


if __name__ == "__main__":
    main()

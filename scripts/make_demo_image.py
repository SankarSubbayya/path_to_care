"""Generate a stylized demo image for the Rajan cellulitis use case.

Why stylized, not photorealistic:
  - Photorealistic medical imagery generation is out of scope here, and
    using a real patient photo would be inappropriate without consent.
  - The triage pipeline accepts ANY image; for end-to-end demo testing the
    important thing is that the model receives a non-empty image with a
    legible caption-style cue. Pair with the canned narrative for a clean
    on-stage demo.

Output:
  docs/figures/demo_cellulitis_foot.png  (1024×1024 PNG, ~50 KB)

Run:
  PYTHONPATH=. .venv/bin/python scripts/make_demo_image.py
"""
from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont


W = H = 1024
OUT = Path("docs/figures/demo_cellulitis_foot.png")

# Editorial palette (matches the UI redesign)
PAPER = (251, 246, 236)
INK = (26, 20, 16)
INK_MUTED = (91, 74, 61)
SKIN_BASE = (208, 175, 152)         # warm Fitzpatrick III-IV tone
SKIN_SHADOW = (155, 122, 100)
ERY_OUTER = (200, 110, 82, 60)      # faint diffuse erythema, alpha
ERY_MID = (190, 70, 50, 130)
ERY_HOT = (148, 36, 22, 200)        # darker red near wound
WOUND = (90, 18, 10)
NAIL_HEAD = (40, 30, 24)
RULE = (214, 201, 180)
CLAY = (177, 74, 50)

def _font(size: int) -> ImageFont.FreeTypeFont:
    # Try a serif for headings and a sans for labels — fall back to default.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _font_sans(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_paper_grain(img: Image.Image) -> None:
    """Add a faint vertical paper-grain texture so the image feels printed."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for x in range(0, w, 22):
        od.line([(x, 0), (x, h)], fill=(0, 0, 0, 5), width=1)
    img.alpha_composite(overlay)


def draw_leg_silhouette(d: ImageDraw.ImageDraw) -> tuple[int, int]:
    """Draw a stylized leg+foot silhouette. Returns the wound pixel coords."""
    # Calf: tapered band coming down from the top
    calf_pts = [
        (350, 60), (560, 60),
        (590, 580), (530, 720),
        (380, 720), (320, 580),
    ]
    d.polygon(calf_pts, fill=SKIN_BASE)
    # Subtle calf shadow on the left
    d.polygon(
        [(350, 60), (400, 60), (400, 720), (380, 720), (320, 580)],
        fill=SKIN_SHADOW,
    )
    # Foot: an L-shaped sweep going right from the ankle
    foot_pts = [
        (380, 720), (530, 720),
        (560, 740), (760, 760), (820, 800), (840, 850),
        (820, 880), (760, 890), (520, 870), (400, 870), (340, 830),
    ]
    d.polygon(foot_pts, fill=SKIN_BASE)
    # Toes
    toe_x = 760
    for i, dy in enumerate([-8, 12, 28, 42, 54]):
        d.ellipse(
            [(toe_x + i * 18, 770 + dy), (toe_x + 18 + i * 18, 800 + dy)],
            fill=SKIN_BASE,
        )
    # Foot shadow at the bottom edge
    d.polygon(
        [(400, 870), (760, 880), (820, 880), (760, 890), (400, 870), (340, 830)],
        fill=SKIN_SHADOW,
    )
    # Puncture wound on the plantar surface (well below the foot)
    wound_x, wound_y = 600, 855
    return wound_x, wound_y


def draw_erythema(base: Image.Image, wound: tuple[int, int]) -> None:
    """Diffuse, irregular red gradient suggesting cellulitis spreading proximally
    from the wound up the calf. Layered alpha blobs + blur."""
    wx, wy = wound
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    # Hot core around the wound
    ld.ellipse([(wx - 80, wy - 40), (wx + 80, wy + 40)], fill=ERY_HOT)
    # Mid red around the foot dorsum
    ld.ellipse([(420, 770), (790, 880)], fill=ERY_MID)
    ld.ellipse([(380, 720), (700, 850)], fill=ERY_MID)
    # Outer faint blush extending up the calf (the lymphangitic streak narrative)
    for i, y in enumerate(range(720, 200, -40)):
        # Each band slightly narrower as we go up, mimicking proximal spread
        narrow = i * 4
        ld.ellipse(
            [(360 + narrow, y - 90), (580 - narrow, y + 90)],
            fill=ERY_OUTER,
        )
    # Streak along medial calf — a "lymphangitic streak"
    streak = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(streak)
    sd.line([(440, 720), (430, 540), (420, 360)], fill=(168, 42, 22, 180), width=14)
    streak = streak.filter(ImageFilter.GaussianBlur(radius=4))
    layer.alpha_composite(streak)

    # Blur whole erythema layer to look like skin redness, not paint
    layer = layer.filter(ImageFilter.GaussianBlur(radius=18))
    base.alpha_composite(layer)


def draw_wound_and_nail(d: ImageDraw.ImageDraw, wound: tuple[int, int]) -> None:
    wx, wy = wound
    # Puncture wound
    d.ellipse([(wx - 12, wy - 12), (wx + 12, wy + 12)], fill=WOUND)
    d.ellipse([(wx - 6, wy - 6), (wx + 6, wy + 6)], fill=(40, 6, 4))
    # Tiny rust ring around it
    d.ellipse([(wx - 22, wy - 22), (wx + 22, wy + 22)], outline=(120, 60, 30), width=2)


def draw_label(d: ImageDraw.ImageDraw, w: int, h: int) -> None:
    serif = _font(34)
    sans_label = _font_sans(20)
    sans_small = _font_sans(16)

    # Top header
    d.text((60, 50), "Demo · Path to Care", font=sans_label, fill=INK_MUTED)
    d.line([(60, 88), (w - 60, 88)], fill=RULE, width=1)

    # Bottom caption block
    d.line([(60, h - 230), (w - 60, h - 230)], fill=RULE, width=1)
    d.text(
        (60, h - 215),
        "Stylized illustration — not a real patient",
        font=sans_small,
        fill=CLAY,
    )
    d.text(
        (60, h - 175),
        "Rural farm worker · Tamil Nadu",
        font=serif,
        fill=INK,
    )
    d.text(
        (60, h - 130),
        "Foot puncture wound (rusty nail, 2 days);",
        font=sans_label,
        fill=INK,
    )
    d.text(
        (60, h - 100),
        "spreading erythema up the calf, fever,",
        font=sans_label,
        fill=INK,
    )
    d.text(
        (60, h - 70),
        "shivering, cannot bear weight",
        font=sans_label,
        fill=INK,
    )

    # Tiny axis-style annotation pointing to the wound
    d.line([(680, 855), (760, 855)], fill=INK, width=1)
    d.text((770, 845), "puncture", font=sans_small, fill=INK)
    d.line([(420, 540), (200, 540)], fill=INK, width=1)
    d.text((60, 530), "lymphangitic streak", font=sans_small, fill=INK)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (W, H), PAPER + (255,))
    d = ImageDraw.Draw(img)

    wound = draw_leg_silhouette(d)
    draw_erythema(img, wound)
    # Re-grab a draw context after alpha-compositing the erythema
    d2 = ImageDraw.Draw(img)
    draw_wound_and_nail(d2, wound)
    draw_label(d2, W, H)
    draw_paper_grain(img)

    img.convert("RGB").save(OUT, format="PNG", optimize=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

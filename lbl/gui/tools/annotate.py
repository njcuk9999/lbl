"""Frames, numbered badges and arrows drawn on a screenshot (Pillow only),
for the testers' guide: tools/tutorial_shots.py draws them on the window's
screenshots, tools/make_guide.py on the pages of the example's report."""
import math
import os

from PIL import Image, ImageDraw, ImageFont

#: the frames, badges and arrows: one vivid colour that the window uses for
#:    nothing, white numbers
MARK = (255, 84, 0)
WIDTH = 1440
FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc")


def font(size):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def annotate(path, marks, origin):
    """Frame, badge and arrow for each mark: (number, box, side), the box in
    page pixels, `origin` the page position of the image's corner; side is
    where the badge goes: left, right, above or below the box."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    big = font(22)
    ox, oy = origin
    for number, box, side in marks:
        if box is None:
            continue
        x, y, w, h = box["x"] - ox, box["y"] - oy, box["width"], box["height"]
        pad = 4
        draw.rounded_rectangle([x - pad, y - pad, x + w + pad, y + h + pad],
                               radius=8, outline=MARK, width=4)
        gap, r = 46, 18
        if side == "left":
            cx, cy, tx, ty = x - pad - gap, y + h / 2, x - pad, y + h / 2
        elif side == "right":
            cx, cy, tx, ty = x + w + pad + gap, y + h / 2, x + w + pad, y + h / 2
        elif side == "above":
            cx, cy, tx, ty = x + min(w / 2, 60), y - pad - gap, x + min(w / 2, 60), y - pad
        else:
            cx, cy, tx, ty = x + min(w / 2, 60), y + h + pad + gap, x + min(w / 2, 60), y + h + pad
        cx = min(max(cx, r + 2), img.width - r - 2)
        cy = min(max(cy, r + 2), img.height - r - 2)
        # the arrow: from the badge's edge to the frame, a head at the frame
        ang = math.atan2(ty - cy, tx - cx)
        sx, sy = cx + r * math.cos(ang), cy + r * math.sin(ang)
        draw.line([sx, sy, tx, ty], fill=MARK, width=4)
        head = 13
        for da in (-0.45, 0.45):
            draw.line([tx, ty, tx - head * math.cos(ang + da),
                       ty - head * math.sin(ang + da)], fill=MARK, width=4)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=MARK,
                     outline=(255, 255, 255), width=3)
        text = str(number)
        tw = draw.textlength(text, font=big)
        draw.text((cx - tw / 2, cy - 13), text, fill=(255, 255, 255), font=big)
    out = os.path.splitext(path)[0] + ".webp"
    img.save(out, "WEBP", quality=84, method=6)
    os.remove(path)
    return out



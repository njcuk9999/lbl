"""The window's background: figures of papers that used LBL, in a mosaic,
blurred. Made once, here, and shipped with the window (static/backdrop_*.jpg
and static/backdrop_credits.json); the window only shows them.

The papers are those whose velocities LBL measured, as read paper by paper
for the References tab (lbl_references.yaml), and the LBL paper itself;
from the LaTeX source of each (arXiv e-print), the figures whose caption
names LBL, then those of its velocities. Only papers under a Creative Commons licence that allows
derivatives are used (a blurred figure is one), each credited.

    python maintenance/gui/paper_backdrop.py <folder with meta/papers.json
        and figs/>

<folder>/meta/papers.json and <folder>/figs/ come from a fetch of the papers'
records and arXiv sources, which is not part of LBL: papers.json holds one
record per paper (arxiv, first, year, title, licence, lbl_mentions, and its
figures, each with its files, relative to figs/, its caption and its kind,
"lbl" when the caption names LBL), and figs/ the figure files.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from PIL import Image, ImageChops, ImageFilter, ImageOps

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "..", "lbl", "gui", "static")
#: the mosaic, in pixels; the page scales it to the window (cover)
WIDTH, HEIGHT = 2400, 1400
#: the rows of figures, and the gap between them [px]
ROWS, GAP = 4, 18
#: the licences used: Creative Commons that allow derivatives
LICENCES = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
            "creativecommons.org/publicdomain/zero/")
#: the page behind the figures, each theme (app.css --bg)
PAGE = dict(light=(233, 238, 246), dark=(8, 17, 31))
#: the dark theme's figures: inverted, hue turned back, black lifted to this
DARK_FLOOR = (13, 22, 40)
#: the blur of the mosaic [px]; the page blurs it a little more
BLUR = 2.2
#: the figures kept, and at most this many of one paper
MOSAIC_FIGURES, PER_PAPER = 40, 2


def render(path):
    """A figure file as an RGB image, its white margins cut"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        image = Image.open(path)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            back = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(back, image)
        image = image.convert("RGB")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "page.png")
            if ext == ".pdf":
                subprocess.run(["pdftoppm", "-r", "110", "-singlefile", "-png",
                                path, out[:-4]], check=True,
                               capture_output=True, timeout=120)
            else:
                subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                                "-dEPSCrop", "-sDEVICE=png16m", "-r110",
                                "-sOutputFile=" + out, path], check=True,
                               capture_output=True, timeout=120)
            image = Image.open(out).convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    box = ImageChops.difference(image, white).convert("L").point(
        lambda v: 255 if v > 12 else 0).getbbox()
    return image.crop(box) if box else None


def dark(image):
    """The figure for the dark theme: inverted, its hues turned back (CSS
    invert(1) hue-rotate(180deg)), its black lifted to the page's"""
    inv = ImageOps.invert(image)
    h, s, v = inv.convert("HSV").split()
    h = h.point(lambda x: (x + 128) % 256)
    turned = Image.merge("HSV", (h, s, v)).convert("RGB")
    return ImageChops.lighter(turned, Image.new("RGB", image.size, DARK_FLOOR))


def choose(papers):
    """(image, credit) of the figures to use: per paper, those whose caption
    names LBL first, then its velocity figures"""
    chosen = yaml.safe_load(open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "lbl_references.yaml")))
    verified = {str(chosen["method"])} | {str(r["arxiv"]) for r in chosen["listed"]
                                          if r.get("arxiv")}
    out = []
    for rec in sorted(papers.values(), key=lambda r: (-(r.get("lbl_mentions") or 0))):
        if str(rec.get("arxiv")) not in verified:
            continue
        licence = rec.get("licence") or ""
        if not any(k in licence for k in LICENCES):
            continue
        figures = sorted(rec.get("figures") or [], key=lambda f: f["kind"] != "lbl")
        taken = 0
        for fig in figures:
            if taken >= PER_PAPER:
                break
            path = os.path.join(FIGS, fig["files"][0])
            try:
                image = render(path)
            except Exception:                              # noqa: BLE001
                continue
            if image is None:
                continue
            ratio = image.width / image.height
            if not 0.7 <= ratio <= 2.8 or min(image.size) < 150:
                continue
            # a plot is mostly white: a photograph or a map is not
            grey = np.asarray(image.convert("L"), float)
            if np.mean(grey > 235) < 0.45:
                continue
            out.append((image, dict(
                first=rec["first"], year=rec["year"], title=rec["title"],
                arxiv=rec["arxiv"], licence=licence, kind=fig["kind"],
                caption=re.sub(r"\\[a-zA-Z]+|[{}$]", "", fig["caption"])[:160])))
            taken += 1
    return out


def layout(images):
    """Justified rows: each row's figures at one height, the row as wide as
    the mosaic, rows added until the mosaic is covered (the last one cut);
    [(index, x, y, w, h)]. Too few figures: they come round again (blurred,
    a figure seen twice does not show)."""
    target = (HEIGHT - (ROWS - 1) * GAP) / ROWS
    out, y, i = [], 0.0, 0
    while y < HEIGHT and images:
        row, width = [], 0.0
        while width - GAP < WIDTH:
            image = images[i % len(images)]
            row.append(i % len(images))
            width += image.width / image.height * target + GAP
            i += 1
        ratios = [images[k].width / images[k].height for k in row]
        h = (WIDTH - GAP * (len(row) - 1)) / sum(ratios)
        x = 0.0
        for k, r in zip(row, ratios):
            out.append((k, int(round(x)), int(round(y)), int(round(r * h)),
                        int(round(h))))
            x += r * h + GAP
        y += h + GAP
    return out


def main(folder):
    global FIGS
    FIGS = os.path.join(folder, "figs")
    papers = json.load(open(os.path.join(folder, "meta", "papers.json")))
    chosen = choose(papers)
    # the papers' order mixed, so that one kind of plot does not fill a row
    rng = np.random.default_rng(2207)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))][:MOSAIC_FIGURES]
    images = [c[0] for c in chosen]
    placed = layout(images)
    used = sorted({i for i, *_ in placed})
    for theme in ("light", "dark"):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), PAGE[theme])
        for i, x, y, w, h in placed:
            tile = images[i].resize((w, h), Image.LANCZOS)
            canvas.paste(dark(tile) if theme == "dark" else tile, (x, y))
        canvas = canvas.filter(ImageFilter.GaussianBlur(BLUR))
        canvas.save(os.path.join(STATIC, "backdrop_%s.jpg" % theme),
                    quality=80, optimize=True, progressive=True)
    credits = [chosen[i][1] for i in used]
    credits.sort(key=lambda c: (c["year"], c["first"]))
    with open(os.path.join(STATIC, "backdrop_credits.json"), "w") as handle:
        json.dump(credits, handle, indent=1, ensure_ascii=False)
    print("%d figures of %d papers in the mosaic, from %d candidates"
          % (len(used), len({c["arxiv"] for c in credits}), len(chosen)))


if __name__ == "__main__":
    main(sys.argv[1])

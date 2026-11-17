#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The "Preliminary" sign of the reports' figures: a red warning triangle with
an exclamation mark, as a road sign has, and the word Preliminary, laid over
the upper right corner of every figure of LBL's report and of PCA2D's.

A report is a look at a run's data, made by a pipeline with its user's
settings, not a figure for a paper; the sign keeps its figures from being
pasted into one as they are. It is laid over the figure's PDF (pypdf), so
that every figure gets the same one wherever it was drawn: LBL's report
pages as they are saved, PCA2D's report at the moment its LaTeX is written
(stamp_tex: every PDF it includes, as a stamped copy beside it).

Created on 2026-11-17

@author: artigau
"""
import functools
import io
import os
import re

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.report.stamp.py'
#: the sign's own file, written beside a report for its introduction (and
#:   never stamped itself)
SIGN_NAME = 'preliminary-sign.pdf'
#: what a stamped copy of a figure is called, beside it
PREFIX = 'stamped-'
#: the red of a road sign
RED = '#C8102E'
#: the sign's size [points]: a triangle with its point up, as a road
#:   warning sign, Preliminary in small letters under the exclamation mark
BORDER = 3.4
HEIGHT = 56.0
TALL = HEIGHT - 2.0 - BORDER
SIDE = 2.0 * TALL / 3.0 ** 0.5
WIDTH = SIDE + BORDER + 2.0
#: the lettering, the first one there: DIN, the road signs' own, then
#:   fonts most systems have (the sign is drawn once and embedded in each
#:   figure)
FONTS = ('DIN Alternate', 'Avenir Next', 'Helvetica Neue', 'Arial',
         'DejaVu Sans')
#: the sign on a figure, against its drawn size (the introduction shows it
#:   whole): half, about 0.4 inch tall
ON_FIGURE = 0.5
#: from the figure's edges [points]
MARGIN = 4.0
#: the most of a figure's width and height the sign takes: a small figure
#:   gets a smaller sign
MOST_WIDTH, MOST_HEIGHT = 0.3, 0.25
#: what the reports say of it, in LaTeX (the sign drawn in the line)
EXPLANATION = (
    r'\textbf{Preliminary.} Every figure of this report carries this red '
    r'warning sign. The report is a look at a run, written '
    r'by a pipeline with the settings it was given, on data that nobody has '
    r'yet checked for any particular purpose (the template, the mask, the '
    r'outliers, the correction and its settings): its figures and numbers '
    r'are for understanding the data, not for a paper. A result meant for '
    r'publication is measured again with the settings it needs, checked, '
    r'and its figures are made from the tables (\texttt{lblrdb}), not '
    r'copied from here.')
#: \includegraphics[options]{file}
_INCLUDE = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})')


# =============================================================================
# Define functions
# =============================================================================
@functools.lru_cache(maxsize=1)
def sign_bytes() -> bytes:
    """The sign, a PDF page of WIDTH x HEIGHT points, transparent (drawn
    once): a triangle, its point up, a thick red border with round corners,
    the figure seen through it, the exclamation mark in it and Preliminary
    in small letters under it, as wide as the triangle is there"""
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle, Polygon
    from matplotlib import patheffects
    from matplotlib.textpath import TextPath

    fig = Figure(figsize=(WIDTH / 72.0, HEIGHT / 72.0))
    FigureCanvasPdf(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, WIDTH)
    ax.set_ylim(0.0, HEIGHT)
    ax.set_axis_off()
    centre = WIDTH / 2.0
    base = 1.0 + BORDER / 2.0
    ax.add_patch(Polygon([(centre - SIDE / 2.0, base),
                          (centre + SIDE / 2.0, base),
                          (centre, base + TALL)], closed=True,
                         facecolor='none', edgecolor=RED, linewidth=BORDER,
                         joinstyle='round'))
    # the inside transparent, the figure seen through it; the word outlined
    #   in white, to be read over a curve
    halo = [patheffects.withStroke(linewidth=1.4, foreground='white')]
    font = FontProperties(family=lettering(), weight='bold')
    # the exclamation mark drawn, as a road sign's is, rather than set in a
    #   font: a bar narrowing downwards, round at both ends, and a round dot
    bottom, top = base + 0.385 * TALL, base + 0.72 * TALL
    wide, narrow = 0.112 * SIDE, 0.07 * SIDE
    ax.add_patch(Polygon([(centre - narrow / 2.0, bottom),
                          (centre + narrow / 2.0, bottom),
                          (centre + wide / 2.0, top),
                          (centre - wide / 2.0, top)], closed=True,
                         facecolor='black', edgecolor='none'))
    for y_mid, radius in ((top, wide / 2.0), (bottom, narrow / 2.0)):
        ax.add_patch(Circle((centre, y_mid), radius, facecolor='black',
                            edgecolor='none'))
    ax.add_patch(Circle((centre, base + 0.265 * TALL), 0.053 * SIDE,
                        facecolor='black', edgecolor='none'))
    # the word: as large as the triangle's inside allows where it sits,
    #   7 points at most (the inside narrows by about 1.2 border widths on
    #   each side along the slanted edges)
    middle = 0.13 * TALL
    unit = TextPath((0.0, 0.0), 'Preliminary', size=1.0,
                    prop=font).get_extents().width
    size = 7.0
    for _ in range(20):
        top = middle + 0.4 * size
        room = SIDE * (1.0 - top / TALL) - 2.6 * BORDER
        if unit * size <= room:
            break
        size *= 0.95
    ax.text(centre, base + middle, 'Preliminary', ha='center', va='center',
            fontsize=size, fontproperties=font, color='black',
            path_effects=halo)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf', transparent=True)
    return buffer.getvalue()


def lettering() -> str:
    """The first of FONTS this system has (asked of matplotlib's list, so
    that the missing ones are not each said in the log)"""
    from matplotlib import font_manager
    there = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in FONTS if name in there), 'DejaVu Sans')


def write_sign(folder: str) -> str:
    """The sign as a file in folder (SIGN_NAME), for a report's text"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, SIGN_NAME)
    with open(path, 'wb') as handle:
        handle.write(sign_bytes())
    return path


def stamp_pdf(source: str, target: str) -> str:
    """
    A copy of a figure's PDF with the sign over the upper right corner of
    every page (target may be source itself)

    :param source: str, the figure
    :param target: str, the stamped copy

    :return: str, target
    """
    from pypdf import PdfReader, PdfWriter, Transformation

    sign = PdfReader(io.BytesIO(sign_bytes())).pages[0]
    width = float(sign.mediabox.width)
    height = float(sign.mediabox.height)
    writer = PdfWriter()
    for page in PdfReader(source).pages:
        box = page.mediabox
        page_width, page_height = float(box.width), float(box.height)
        scale = min(ON_FIGURE, MOST_WIDTH * page_width / width,
                    MOST_HEIGHT * page_height / height)
        left = float(box.left) + page_width - (width + MARGIN) * scale
        bottom = float(box.bottom) + page_height - (height + MARGIN) * scale
        # merged on the writer's own page (pypdf 7 refuses a page that is
        #   not a writer's)
        writer.add_page(page).merge_transformed_page(
            sign, Transformation().scale(scale).translate(left, bottom))
    partial = target + '.part'
    with open(partial, 'wb') as handle:
        writer.write(handle)
    os.replace(partial, target)
    return target


def stamp_tex(text: str, folder: str) -> str:
    """
    A report's LaTeX with every PDF it includes swapped for a stamped copy
    (PREFIX, beside it, made again from the figure each time): the sign on
    every figure, whichever part of the run drew it. A figure that cannot be
    stamped (no pypdf, an unreadable file) is kept as it is.

    :param text: str, the LaTeX
    :param folder: str, the folder it is compiled in

    :return: str, the LaTeX
    """
    made = dict()

    def swap(found):
        name = found.group(2).strip()
        if not name.lower().endswith('.pdf') or \
                os.path.basename(name) in (SIGN_NAME,) or \
                os.path.basename(name).startswith(PREFIX):
            return found.group(0)
        if name not in made:
            source = name if os.path.isabs(name) else os.path.join(folder,
                                                                   name)
            head, tail = name.replace('\\', '/').rpartition('/')[::2]
            stamped = (head + '/' if head else '') + PREFIX + tail
            target = stamped if os.path.isabs(stamped) else os.path.join(
                folder, stamped)
            try:
                stamp_pdf(source, target)
                made[name] = stamped
            except Exception:                                 # noqa: BLE001
                made[name] = name
        return found.group(1) + made[name] + found.group(3)

    return _INCLUDE.sub(swap, text)


def explanation(sign_path: str = SIGN_NAME) -> str:
    """The reports' paragraph on the sign, the sign above it, at the size a
    figure carries it"""
    return ('\\begin{center}\\includegraphics[height=%.1fpt]{%s}'
            '\\end{center}\n%s' % (HEIGHT, sign_path, EXPLANATION))

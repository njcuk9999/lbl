#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The "Preliminary" sign of the reports' figures (lbl.report.stamp): drawn,
laid over the upper right corner of every page of a figure, and every PDF a
report's LaTeX includes swapped for a stamped copy.

Created on 2026-11-17

@author: artigau
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import pytest                                                   # noqa: E402

from lbl.report import stamp                                    # noqa: E402

pypdf = pytest.importorskip('pypdf')
#: the sign's red, as the PDF strokes it (its numbers on lines of their own)
RED_STROKE = re.compile(r'0\.7843\d*\s+0\.0627\d*\s+0\.1803\d*\s+RG')


def _red(page) -> bool:
    return bool(RED_STROKE.search(page.get_contents().get_data().decode(
        'latin-1')))


def _figure(path, pages=1):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for _ in range(pages):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([0, 1], [0, 1])
            pdf.savefig(fig)
            plt.close(fig)
    return path


def test_the_sign():
    page = pypdf.PdfReader(__import__('io').BytesIO(
        stamp.sign_bytes())).pages[0]
    assert round(float(page.mediabox.width), 1) == round(stamp.WIDTH, 1)
    assert round(float(page.mediabox.height), 1) == stamp.HEIGHT
    assert _red(page)
    assert stamp.lettering() in stamp.FONTS


def test_every_page_of_a_figure_gets_it(tmp_path):
    source = _figure(str(tmp_path / 'fig.pdf'), pages=2)
    target = stamp.stamp_pdf(source, str(tmp_path / 'stamped.pdf'))
    before = pypdf.PdfReader(source)
    after = pypdf.PdfReader(target)
    assert len(after.pages) == 2
    for one, two in zip(before.pages, after.pages):
        assert one.mediabox == two.mediabox
        assert not _red(one)
        assert _red(two)
    # in place
    stamp.stamp_pdf(source, source)
    assert _red(pypdf.PdfReader(source).pages[0])


def test_a_report_s_figures_are_swapped_for_stamped_copies(tmp_path):
    (tmp_path / 'figures').mkdir()
    _figure(str(tmp_path / 'figures' / 'a.pdf'))
    stamp.write_sign(str(tmp_path / 'figures'))
    text = ('\\includegraphics[width=\\linewidth,page=2]{figures/a.pdf}\n'
            '\\includegraphics{figures/a.pdf}\n'
            '\\includegraphics[height=34pt]{figures/%s}\n'
            '\\includegraphics{figures/gone.pdf}\n'
            '\\includegraphics{figures/b.png}\n' % stamp.SIGN_NAME)
    out = stamp.stamp_tex(text, str(tmp_path))
    lines = out.splitlines()
    assert lines[0] == ('\\includegraphics[width=\\linewidth,page=2]'
                        '{figures/stamped-a.pdf}')
    assert lines[1] == '\\includegraphics{figures/stamped-a.pdf}'
    # the sign itself, a file that is not there, a picture: as they were
    assert lines[2:] == text.splitlines()[2:]
    assert os.path.exists(tmp_path / 'figures' / 'stamped-a.pdf')
    # made again from the figure, never stamped twice
    assert stamp.stamp_tex(out, str(tmp_path)) == out


def test_the_reports_explain_it():
    text = stamp.explanation('figures/' + stamp.SIGN_NAME)
    assert '\\includegraphics[height=%.1fpt]{figures/%s}' % (
        stamp.HEIGHT, stamp.SIGN_NAME) in text
    assert 'not for a paper' in text

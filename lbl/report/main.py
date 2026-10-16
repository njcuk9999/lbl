"""Assemble the report of one run: a summary table, then one page per step.

Called at the very end, after `compile`, and driven by the wrapper's REPORT
parameter (true by default).

Nothing here may make a run fail. The report is an account of what happened,
not a step of the calculation, so every page that cannot be drawn is skipped
and named on a last page, and a missing LaTeX leaves the figures behind as a
plain PDF.
"""
import os
import shutil
import subprocess
import tempfile
import traceback

from lbl.core import logger

log = logger.Log(filename='')

#: where the reports go, under DATA_DIR
FOLDER = 'reports'

#: the LaTeX frame. Deliberately plain: article, no package that a minimal
#: TeX would not have, so that the report builds on a bare machine.
PREAMBLE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\setlength{\parindent}{0pt}
\begin{document}
"""


def _escape(text):
    """Make a string safe for LaTeX. Object names carry underscores."""
    out = str(text)
    for char in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        out = out.replace(char, '\\' + char)
    return out.replace('~', '\\textasciitilde{}')


def _table(rows):
    """The summary table, as LaTeX."""
    lines = [r'\begin{center}', r'\begin{tabular}{lrl}', r'\toprule',
             r'Quantity & Value & Unit \\', r'\midrule']
    for label, value, unit in rows:
        lines.append('%s & %s & %s \\\\'
                     % (_escape(label), _escape(value), _escape(unit)))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(lines)


def make_report(data_dir, rdb_file, title, science_dir=None, tc_dir=None,
                known_periods=None):
    """Write the report of one run. Returns the path of the PDF, or None.

    :param data_dir: str, the run's DATA_DIR
    :param rdb_file: str, the .rdb that `compile` has just written
    :param title: str, the name of the run, which titles the pages
    :param science_dir: str, the delivered spectra, for the telluric page
    :param tc_dir: str, the cleaned spectra, for the same page
    :param known_periods: periods to mark on the periodogram
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:                                  # noqa: BLE001
        log.warning('no report: matplotlib is not available ({0})'.format(exc))
        return None

    from lbl.report import figures, series, stats

    if not os.path.exists(rdb_file):
        log.warning('no report: {0} is not there'.format(rdb_file))
        return None
    try:
        table = series.read_rdb(rdb_file)
    except Exception as exc:                                  # noqa: BLE001
        log.warning('no report: cannot read {0} ({1})'.format(rdb_file, exc))
        return None

    folder = os.path.join(data_dir, FOLDER)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, '%s.pdf' % title)

    # WHAT THE REPORT TRIES TO DRAW, in order. A page that fails is noted and
    # the report goes on: better an incomplete report than a run that stops
    # on a figure.
    wanted = [
        ('The velocity series',
         lambda: figures.page_sequence(plt, table, title)),
        ('Velocity against BERV',
         lambda: figures.page_berv(plt, table, title)),
        ('Periodogram and spectral window',
         lambda: figures.page_periodogram(plt, table, title, known_periods)),
        ('False inclusion probability, gaussian',
         lambda: figures.page_fip(plt, table, title)),
        ('False inclusion probability, outlier-aware',
         lambda: figures.page_oafip(plt, table, title, known_periods)),
        ('Indicators and temperature projections',
         lambda: figures.page_indicators(plt, table, title))]
    # THE TAPAS SPLINES, so that the telluric pages centre themselves on the
    # strongest line of the species they are about rather than on a window
    # written down once for HARPS.
    spl_water, spl_dry = _tapas(data_dir)
    if series.has_telluric(table) and science_dir and tc_dir:
        wanted.append(('Telluric correction, water column',
                       lambda: figures.page_telluric(plt, table, title,
                                                     science_dir, tc_dir,
                                                     'water', spl_water,
                                                     spl_dry)))
        wanted.append(('Telluric correction, dry column',
                       lambda: figures.page_telluric(plt, table, title,
                                                     science_dir, tc_dir,
                                                     'dry', spl_water,
                                                     spl_dry)))

    work = tempfile.mkdtemp(prefix='lbl_report_')
    try:
        drawn, missing = [], []
        for index, (name, draw) in enumerate(wanted):
            try:
                fig = draw()
            except Exception as exc:                          # noqa: BLE001
                missing.append('%s: %s' % (name, exc))
                log.warning('report page "{0}" failed: {1}'.format(name, exc))
                log.warning(traceback.format_exc(limit=2))
                continue
            if fig is None:
                missing.append('%s: nothing to draw, or koloa not installed'
                               % name)
                continue
            path = os.path.join(work, 'page%02d.pdf' % index)
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            drawn.append((name, path))
        built = _build(work, out, title, rdb_file, stats.summarise(table),
                       drawn, missing)
        if built is None:
            built = _fallback(out, drawn, missing)
        return built
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tapas(data_dir):
    """The TAPAS splines of this run, or (None, None).

    Only used to find where the deepest line of each species sits; a report
    without them falls back to the HARPS values and says so in nothing,
    because the window is then simply the usual one.
    """
    path = os.path.join(data_dir, 'other', 'tapas_lbl.fits')
    if not os.path.exists(path):
        return None, None
    try:
        import numpy as np
        from astropy.io import fits
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        table = fits.getdata(path)
        wave = np.array(table['WAVELENGTH'], dtype=float)
        water = spline(wave, np.array(table['ABSO_WATER'], dtype=float),
                       k=1, ext=3)
        dry = spline(wave, np.array(table['ABSO_OTHERS'], dtype=float),
                     k=1, ext=3)
        return water, dry
    except Exception:                                         # noqa: BLE001
        return None, None


def _build(work, out, title, rdb_file, rows, drawn, missing):
    """Run pdflatex on the assembled document. None if LaTeX is not there."""
    if shutil.which('pdflatex') is None:
        log.warning('pdflatex not found: the report falls back to plain pages')
        return None
    body = [PREAMBLE,
            r'\section*{LBL report: %s}' % _escape(title),
            r'\textit{From %s}' % _escape(os.path.basename(rdb_file)),
            r'\vspace{4mm}', _table(rows)]
    for name, path in drawn:
        body.append(r'\clearpage')
        body.append(r'\subsection*{%s}'
                    % _escape(name[:1].upper() + name[1:]))
        body.append(r'\begin{center}\includegraphics[width=\linewidth,'
                    r'height=0.62\textheight,keepaspectratio]{%s}\end{center}'
                    % os.path.basename(path))
    if missing:
        body.append(r'\clearpage')
        body.append(r'\subsection*{What this report does not show}')
        body.append(r'\begin{itemize}')
        for one in missing:
            body.append(r'\item %s' % _escape(one))
        body.append(r'\end{itemize}')
    body.append(r'\end{document}')

    source = os.path.join(work, 'report.tex')
    with open(source, 'w') as handle:
        handle.write('\n'.join(body))
    for _ in range(2):          # twice, so the page numbers settle
        done = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             'report.tex'], cwd=work, capture_output=True, text=True)
    made = os.path.join(work, 'report.pdf')
    if done.returncode != 0 or not os.path.exists(made):
        log.warning('pdflatex failed; the report falls back to plain pages')
        tail = (done.stdout or '').strip().split('\n')[-6:]
        for line in tail:
            log.warning('    {0}'.format(line))
        return None
    shutil.copyfile(made, out)
    log.general('Report written: {0}'.format(out))
    return out


def _fallback(out, drawn, missing):
    """No LaTeX: bind the figures as they are, so nothing is lost."""
    try:
        from pypdf import PdfWriter
    except Exception:                                         # noqa: BLE001
        try:
            from PyPDF2 import PdfMerger as PdfWriter
        except Exception:                                     # noqa: BLE001
            log.warning('no report: neither pdflatex nor pypdf is available')
            return None
    writer = PdfWriter()
    for _, path in drawn:
        writer.append(path)
    if not drawn:
        return None
    with open(out, 'wb') as handle:
        writer.write(handle)
    log.general('Report written without LaTeX: {0}'.format(out))
    if missing:
        for one in missing:
            log.warning('    not shown: {0}'.format(one))
    return out

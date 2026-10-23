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

import numpy as np

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


def _planets(template_file):
    """The companions whose Keplerian velocities registered the spectra of
    the template (its PLANETS extension, lbl/science/planets.py), or None"""
    if not template_file or not os.path.exists(template_file):
        return None
    try:
        from astropy.io import fits
        with fits.open(template_file) as hdul:
            if 'PLANETS' not in hdul:
                return None
            header = hdul[0].header
            table = hdul['PLANETS'].data
            orbits = []
            for row in table:
                # an epoch not given is NaN in the table, None here
                t0 = float(row['t0']) if row['t0'] == row['t0'] else None
                tp = float(row['tp']) if row['tp'] == row['tp'] else None
                epoch = ('t0 %.5f' % t0 if t0 is not None
                         else 'tp %.5f' % tp)
                orbits.append(dict(name=str(row['name']),
                                   period=float(row['period']),
                                   k=float(row['k']), e=float(row['e']),
                                   omega=float(row['omega']), t0=t0, tp=tp,
                                   epoch=epoch,
                                   reference=str(row['reference'])))
        return dict(star=header.get('PLNSTAR', '?'),
                    catalog=header.get('PLNCAT', '?'), orbits=orbits)
    except Exception as exc:                                  # noqa: BLE001
        log.warning('report: cannot read the orbits of {0} ({1})'.format(
            template_file, exc))
        return None


def _planet_section(planets):
    """The page that says which orbits the template was made on, as LaTeX"""
    lines = [r'\subsection*{Template made on known orbits}',
             'The spectra were registered with their BERV plus the Keplerian '
             'velocity of the star at their time (PLANETS = True), and the '
             'template made in one pass instead of iterating on the '
             'velocities. The velocities reported stay corrected for the '
             'BERV only. Star \\texttt{%s}, orbits from \\texttt{%s}:'
             % (_escape(planets['star']), _escape(planets['catalog'])),
             r'\begin{center}\small', r'\begin{tabular}{lrrrrl}', r'\toprule',
             r'Companion & $P$ (d) & $K$ (m/s) & $e$ & $\omega$ (deg) & '
             r'Epoch (BJD) \\', r'\midrule']
    for orbit in planets['orbits']:
        lines.append('%s & %.6g & %.2f & %.4f & %.1f & %s \\\\'
                     % (_escape(orbit['name']), orbit['period'], orbit['k'],
                        orbit['e'], orbit['omega'], _escape(orbit['epoch'])))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}',
              r'\begin{itemize}']
    for orbit in planets['orbits']:
        lines.append(r'\item %s: %s' % (_escape(orbit['name']),
                                         _escape(orbit['reference'] or '?')))
    lines.append(r'\end{itemize}')
    return '\n'.join(lines)


def make_report(data_dir, rdb_file, title, science_dir=None, tc_dir=None,
                known_periods=None, template_file=None, candidates_of=None):
    """Write the report of one run. Returns the path of the PDF, or None.

    :param data_dir: str, the run's DATA_DIR
    :param rdb_file: str, the .rdb that `compile` has just written
    :param title: str, the name of the run, which titles the pages
    :param science_dir: str, the delivered spectra, for the telluric page
    :param tc_dir: str, the cleaned spectra, for the same page
    :param known_periods: periods to mark on the periodogram (by default
                          those of the orbits the template was made on)
    :param template_file: str, the template of the comparison object, to say
                          whether it was made on known orbits (PLANETS)
    :param candidates_of: str, the object whose transiting planets and TESS
                          candidates (candidates.yaml) have their K fitted
                          together (PLANETS), or None for no such fit
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

    # THE ORBITS the template was made on, if it was (PLANETS)
    planets = _planets(template_file)
    if planets is not None and known_periods is None:
        known_periods = [orbit['period'] for orbit in planets['orbits']]

    folder = os.path.join(data_dir, FOLDER)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, '%s.pdf' % title)

    # WHAT THE REPORT TRIES TO DRAW, in order. A page that fails is noted and
    # the report goes on: better an incomplete report than a run that stops
    # on a figure.
    wanted = [
        ('The velocity series',
         lambda: figures.page_sequence(plt, table, title)),
        ('Iterations to converge',
         lambda: figures.page_iterations(plt, table, title)),
        ('Velocity against BERV',
         lambda: figures.page_berv(plt, table, title))]
    # THE ZERO POINT FROM THE TELLURICS, for an instrument without a
    #   simultaneous reference (TELLDV in the rdb): its series, and the fit
    #   of the exposure of median SNR drawn
    if series.column(table, figures.TELLDV) is not None:
        wanted.append(('Zero point from the tellurics',
                       lambda: figures.page_telluric_zero_point(plt, table,
                                                                title)))
        example = _zero_point_example(table, data_dir, science_dir)
        if example is not None:
            wanted.append(('Zero point from the tellurics, one exposure',
                           lambda: _zero_point_figure(plt, example, data_dir,
                                                      title)))
    # THE TRANSITING PLANETS AND CANDIDATES of the star (candidates.yaml,
    #   PLANETS): their K fitted together, the known orbits held fixed
    fit = _fit_candidates(candidates_of, table, planets)
    if fit is not None:
        known_periods = list(known_periods or []) + [
            float(cand['period']) for cand in fit['candidates']]
    if planets is not None:
        wanted.append(('Velocities folded on the known orbits',
                       lambda: figures.page_phase(plt, table, title, planets)))
    if fit is not None:
        wanted += [
            ('Joint Keplerian fit of the transiting candidates',
             lambda: figures.page_candidate_fit(plt, table, title, fit)),
            ('Velocities folded on the transiting candidates',
             lambda: figures.page_candidate_phase(plt, table, title, fit))]
    # WHAT IS TAKEN OUT for the periodogram and the FIPs: the known orbits
    #   and the fitted candidates, when there are; they are then drawn twice,
    #   as measured and without them, side by side
    if fit is not None:
        residual = figures.subtract_fit(table, fit)
        after = (', known orbits and fitted candidates subtracted'
                 if planets is not None else ', fitted candidates subtracted')
    elif planets is not None:
        residual = figures.subtract_orbits(table, planets)
        after = ', known orbits subtracted'
    else:
        residual, after = None, ''
    if residual is None:
        wanted += [
            ('Periodogram and spectral window',
             lambda: figures.page_periodogram(plt, table, title,
                                              known_periods)),
            ('False inclusion probability, gaussian',
             lambda: figures.page_fip(plt, table, title)),
            ('False inclusion probability, outlier-aware',
             lambda: figures.page_oafip(plt, table, title, known_periods))]
    else:
        wanted += [
            ('Periodogram and spectral window, as measured',
             lambda: figures.page_periodogram(plt, table, title,
                                              known_periods)),
            ('Periodogram and spectral window' + after,
             lambda: figures.page_periodogram(plt, residual, title + after,
                                              known_periods)),
            ('False inclusion probability, gaussian, as measured',
             lambda: figures.page_fip(plt, table, title)),
            ('False inclusion probability, gaussian' + after,
             lambda: figures.page_fip(plt, residual, title + after)),
            ('False inclusion probability, outlier-aware, as measured',
             lambda: figures.page_oafip(plt, table, title, known_periods)),
            ('False inclusion probability, outlier-aware' + after,
             lambda: figures.page_oafip(plt, residual, title + after,
                                        known_periods))]
    wanted.append(('Indicators and temperature projections',
                   lambda: figures.page_indicators(plt, table, title)))
    # THE ACTIVITY LINES, columns of the rdb (lbl.science.activity_lines):
    #   their time series, a page per ACTIVITY_PER_PAGE lines
    lines = figures.activity_lines(table, _activity_labels())
    per_page = figures.ACTIVITY_PER_PAGE
    for first in range(0, len(lines), per_page):
        chunk = lines[first:first + per_page]
        name = 'Activity lines'
        if len(lines) > per_page:
            name += ', %d to %d of %d' % (first + 1, first + len(chunk),
                                          len(lines))
        wanted.append((name, lambda chunk=chunk: figures.page_activity(
            plt, table, chunk, title)))
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
        rows = list(stats.summarise(table))
        iterations = series.column(table, figures.ITERATIONS)
        if iterations is not None and np.isfinite(iterations).any():
            iterations = iterations[np.isfinite(iterations)]
            rows.append(('Iterations to converge',
                         'median %g (%d to %d)'
                         % (np.median(iterations), np.min(iterations),
                            np.max(iterations)), ''))
        rows.append(('Template frame',
                     'BERV + Keplerian, %d orbit(s) of %s'
                     % (len(planets['orbits']), planets['star'])
                     if planets is not None else 'BERV, velocities iterated',
                     ''))
        if fit is not None:
            for cand, k, sk in zip(fit['candidates'], fit['k'], fit['sk']):
                rows.append(('K of %s' % cand.get('name', '?'),
                             '%.2f +- %.2f (%.1f sigma)'
                             % (k, sk, abs(k) / sk if sk > 0 else 0), 'm/s'))
            rows.append(('Jitter of the candidate fit', '%.2f' % fit['jitter'],
                         'm/s'))
        if lines:
            chi2 = [figures.activity_stats(table, key)[2]
                    for key, _, _ in lines]
            top = int(np.nanargmax(chi2)) if np.isfinite(chi2).any() else 0
            rows.append(('Activity lines',
                         '%d measured, most variable %s (chi2_r %.2g)'
                         % (len(lines), lines[top][1], chi2[top]), ''))
        built = _build(work, out, title, rdb_file, rows, drawn, missing,
                       planets, fit)
        if built is None:
            built = _fallback(out, drawn, missing)
        return built
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fit_candidates(object_name, table, planets):
    """The joint fit of the star's transiting planets and TESS candidates
    (candidates.yaml, lbl.report.kepfit), the known orbits of the template
    held fixed; None without object, candidates or velocities to fit"""
    if not object_name:
        return None
    try:
        from lbl.report import kepfit, series
        from lbl.science import planets as kepler
        found = kepler.candidates_for(kepler.star_names(object_name))
        if found is None:
            return None
        known = planets['orbits'] if planets is not None else []
        # a planet whose orbit is known is not fitted again
        taken = {kepler.normalise(orbit.get('name', '')) for orbit in known}
        candidates = [cand for cand in found['candidates']
                      if kepler.normalise(cand.get('name', '')) not in taken]
        if not candidates:
            return None
        result = kepfit.fit(series.column(table, series.TIME) + 2400000.0,
                            series.column(table, series.VELOCITY),
                            series.column(table, series.ERROR),
                            candidates, known)
        result.update(star=found['star'], catalog=found['catalog'])
        for cand, k, sk in zip(candidates, result['k'], result['sk']):
            log.general('report: {0}, P = {1:.6g} d: K = {2:.2f} +- {3:.2f} '
                        'm/s'.format(cand.get('name', '?'),
                                     float(cand['period']), k, sk))
        return result
    except Exception as exc:                                  # noqa: BLE001
        log.warning('report: no fit of the candidates ({0})'.format(exc))
        return None


def _candidate_section(fit):
    """The page that says which candidates were fitted and what K they got,
    as LaTeX"""
    lines = [r'\subsection*{Transiting planets and candidates fitted}',
             'Their K fitted together on the velocities (circular orbits, '
             'period and transit epoch of \\texttt{%s} fixed, one offset, a '
             'jitter of %.2f m/s added to the error bars%s), %d velocities. '
             'Star \\texttt{%s}:'
             % (_escape(os.path.basename(fit['catalog'])), fit['jitter'],
                ', the known orbits held fixed' if fit['known'] else '',
                fit['nfit'], _escape(fit['star'])),
             r'\begin{center}\small', r'\begin{tabular}{llrrrr}',
             r'\toprule',
             r'Name & Status & $P$ (d) & $T_0$ (BJD) & $K$ (m/s) & '
             r'$K/\sigma_K$ \\', r'\midrule']
    for cand, k, sk in zip(fit['candidates'], fit['k'], fit['sk']):
        lines.append('%s & %s & %.6g & %.5f & %.2f $\\pm$ %.2f & %.1f \\\\'
                     % (_escape(cand.get('name', '?')),
                        _escape(cand.get('disposition') or ''),
                        float(cand['period']), float(cand['t0']), k, sk,
                        abs(k) / sk if sk > 0 else 0))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}',
              r'\begin{itemize}']
    for cand in fit['candidates']:
        lines.append(r'\item %s: ephemeris from %s'
                     % (_escape(cand.get('name', '?')),
                        _escape(cand.get('reference') or
                                cand.get('source') or '?')))
    lines.append(r'\end{itemize}')
    return '\n'.join(lines)


def _zero_point_example(table, data_dir, science_dir):
    """The delivered file of the exposure of median SNR, whose telluric fit
    the report draws, or None (the science folder may be the cleaned one,
    <object>_tc, whose files hold no tellurics to fit)"""
    from lbl.report import series
    names = table.get('FILENAME') if hasattr(table, 'get') else None
    if names is None or not science_dir:
        return None
    snr = None
    for key in ('SNR31', 'EXTSN060'):
        snr = series.column(table, key)
        if snr is not None:
            break
    names = [str(name).strip() for name in names]
    pick = (int(np.argsort(snr)[len(snr) // 2]) if snr is not None
            else len(names) // 2)
    name = os.path.basename(names[pick])
    folders = [science_dir]
    if science_dir.rstrip('/').endswith('_tc'):
        folders.insert(0, science_dir.rstrip('/')[:-3])
    for folder in folders:
        for candidate in (name, name + '.fits'):
            path = os.path.join(folder, candidate)
            if os.path.exists(path):
                return path
    return None


def _zero_point_figure(plt, path, data_dir, title):
    """The telluric fit of one exposure (the instrument's own figure)"""
    from lbl.instruments import espadons
    return espadons.telluric_figure(
        plt, path, tapas_file=os.path.join(data_dir, 'models',
                                           'tapas_lbl.fits'), title=title)


def _activity_labels():
    """The usual name of each activity line ({'He 1.08333': 'He I 10830'}),
    from the catalog; empty if it cannot be read (the pages then name the
    lines by their key)"""
    try:
        from lbl.science.activity_lines import load_lines
        return {line['NAME']: line['LABEL'] for line in load_lines()}
    except Exception:                                         # noqa: BLE001
        return dict()


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


def _build(work, out, title, rdb_file, rows, drawn, missing, planets=None,
           fit=None):
    """Run pdflatex on the assembled document. None if LaTeX is not there."""
    if shutil.which('pdflatex') is None:
        log.warning('pdflatex not found: the report falls back to plain pages')
        return None
    body = [PREAMBLE,
            r'\section*{LBL report: %s}' % _escape(title),
            r'\textit{From %s}' % _escape(os.path.basename(rdb_file)),
            r'\vspace{4mm}', _table(rows)]
    if planets is not None:
        body.append(_planet_section(planets))
    if fit is not None:
        body.append(_candidate_section(fit))
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

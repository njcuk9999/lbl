"""The numbers that head the report, in one table.

Kept apart from the figures so that the same numbers can be read by anything
else that wants them, and so that the table and the plots can never disagree.
"""
import datetime
import os

import numpy as np

from lbl.report import series as S


def _now():
    """The moment the report was written, to the minute."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%d %H:%M')


def robust_sigma(values):
    """The robust scatter: 1.4826 times the median absolute deviation.

    Quoted beside the plain rms rather than instead of it. The two do not
    measure the same thing: the MAD only sees the core of the distribution,
    the rms weighs the tails, and a correction that straightens the tails
    while spoiling the core makes them disagree. One outlier is enough to
    make an rms unreadable, so neither alone tells the story.
    """
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size < 2:
        return float('nan')
    return float(1.4826 * np.median(np.abs(good - np.median(good))))


def _span(values):
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size == 0:
        return None
    return float(np.min(good)), float(np.median(good)), float(np.max(good))


def summarise(table):
    """Every number the summary table shows, as (label, value, unit) rows."""
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)

    # WHICH LBL WROTE THIS, and from where: a report that does not say which
    # code produced it cannot be compared with another one six months later.
    rows = []
    try:
        from lbl.core import base
        rows.append(('LBL version', '%s (%s)' % (base.__version__,
                                                 base.__date__), ''))
        rows.append(('LBL path', os.path.dirname(os.path.dirname(
            os.path.abspath(base.__file__))), ''))
    except Exception:                                         # noqa: BLE001
        pass
    rows.append(('Report written', _now(), 'UTC'))
    rows += [('Epochs', '%d' % len(time), ''),
            ('Time span', '%.1f' % (np.max(time) - np.min(time)), 'days'),
            ('Nights', '%d' % len(np.unique(np.floor(time))), ''),
            ('Systemic velocity', '%.1f' % np.median(vrad), 'm/s'),
            ('rms', '%.2f' % np.std(centred, ddof=1), 'm/s'),
            ('Robust scatter (MAD)', '%.2f' % robust_sigma(centred), 'm/s'),
            ('Median error bar', '%.2f' % np.median(svrad), 'm/s')]

    # the ratio that says whether the scatter is explained by the error bars
    median_err = np.median(svrad)
    if np.isfinite(median_err) and median_err > 0:
        rows.append(('Excess over the error bars',
                     '%.2f' % (robust_sigma(centred) / median_err), ''))

    # the nightly scatter, which drops anything that varies within a night
    nights = np.floor(time)
    binned = np.array([np.median(centred[nights == one])
                       for one in np.unique(nights)])
    if binned.size > 2:
        rows.append(('Nightly rms', '%.2f' % np.std(binned, ddof=1), 'm/s'))

    # how far the outliers reach, in units of the robust scatter
    scatter = robust_sigma(centred)
    if np.isfinite(scatter) and scatter > 0:
        reach = np.max(np.abs(centred)) / scatter
        rows.append(('Largest excursion', '%.1f' % reach, 'MAD'))
        rows.append(('Beyond 5 MAD',
                     '%d' % int(np.sum(np.abs(centred) > 5 * scatter)), ''))

    for name, label, unit in ((S.BERV, 'BERV span', 'km/s'),
                              (S.AIRMASS, 'Airmass', '')):
        values = S.column(table, name)
        if values is None:
            continue
        span = _span(values)
        if span is None:
            continue
        if name == S.BERV:
            rows.append((label, '%.1f (%.1f to %.1f)'
                         % (span[2] - span[0], span[0], span[2]), unit))
        else:
            rows.append((label, '%.2f to %.2f (median %.2f)'
                         % (span[0], span[2], span[1]), unit))

    # the telluric exponents, only when the correction ran
    for name, label in ((S.WATER, 'Water exponent'),
                        (S.DRY, 'Dry exponent')):
        values = S.column(table, name)
        if values is None:
            continue
        span = _span(values)
        if span is None:
            continue
        rows.append((label, '%.3f to %.3f (median %.3f)'
                     % (span[0], span[2], span[1]), ''))
    return rows

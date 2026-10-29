"""The report of an LBL run: what it did, in figures.

Written at the very end, after `compile`, one PDF per run in
<DATA_DIR>/reports. Driven by the REPORT parameter of the wrapper, true by
default.

What the report shows:

    the SEQUENCE of the velocities, with their error bars, coloured by the
        star's velocity in the telluric frame (or by the BERV)
    the PERIODOGRAM of the series and that of the window, with the
        alias-aware false inclusion probability (FIP) below them
    the FIP, gaussian and outlier-aware, side by side, because what matters
        is to see whether the answer depends on how the outliers are
        treated, and not only the cleaned answer
    with PLANETS, the velocities folded on the known orbits, and the K of
        the confirmed planets with a published ephemeris fitted on them
    the INDICATORS and the ACTIVITY LINES measured beside the velocity
    the QUALITY OF THE TELLURIC CORRECTION, if it was applied: for the
        best, the median and the worst exposure in water column, then the
        same in dry column

The calculations are LBL's own (lbl.report.periodogram, lbl.report.fip,
lbl.report.stats, lbl.report.kepfit), on numpy and scipy. A page that cannot
be drawn is skipped and named on a last page, and the run never suffers from
it.
"""
from lbl.report.main import make_report

__all__ = ['make_report']

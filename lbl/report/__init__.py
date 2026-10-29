"""The report of an LBL run: what it did, in figures.

Written at the very end, after `compile`, one PDF per run in
<DATA_DIR>/reports. Driven by the REPORT parameter of the wrapper, true by
default.

What the report shows:

    the SEQUENCE of the velocities, per instrument, with its error bars
    the PERIODOGRAM of the series and that of the window
    the outlier-resistant FIP, in its three versions (no cut, soft cut,
        hard cut), because what matters is to see what the cleaning does
        to the answer and not only the cleaned answer
    the QUALITY OF THE TELLURIC CORRECTION, if it was applied: for the
        best, the median and the worst exposure in water column, then the
        same in dry column

THE ROUTINES COME FROM KOLOA (periodograms, FIP, robust cuts) rather than
being rewritten here: they are tested and maintained there. The report is
therefore optional in the strong sense, that is, it skips itself quietly
if koloa is not installed, and the run does not suffer from it.
"""
from lbl.report.main import make_report

__all__ = ['make_report']

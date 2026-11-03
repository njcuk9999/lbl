"""One definition of an observing night for all of LBL
(lbl.core.astro.night_number, NIGHT_START_UT): PCA2D's coadds, its correct
stage and holdout, the reports and the window agree on which exposures are
one night.

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_nights.py
"""
import datetime as dt

import numpy as np
from astropy.table import Table

from lbl.core.astro import NIGHT_START_UT, night_number
from lbl.gui import targets
from lbl.pca2d import diagnostics, lblscan, nights, reconstruct
from lbl.report import series, stats

#: a Chilean night: 23:30 UT on 2024-10-11 and 01:30 and 08:00 UT the next day,
#:    then the next night's first exposure at 23:40 UT
STAMPS = [dt.datetime(2024, 10, 11, 23, 30), dt.datetime(2024, 10, 12, 1, 30),
          dt.datetime(2024, 10, 12, 8, 0), dt.datetime(2024, 10, 12, 23, 40)]


def _jd(stamp):
    return 2451545.0 + (stamp - dt.datetime(2000, 1, 1, 12, 0)).total_seconds() / 86400


JD = np.array([_jd(s) for s in STAMPS])


def test_the_night_changes_at_noon_ut():
    assert NIGHT_START_UT == 12.0
    assert night_number(2460600.999) != night_number(2460601.001)
    # 21:36 UT and 09:36 UT the next morning: one night
    assert night_number(2460600.4) == night_number(2460600.9)
    # an rjd and an MJD + 0.5 give the boundary of the JD
    assert list(night_number(JD - 2400000.0)) == list(night_number(JD) - 2400000)
    assert list(night_number(JD - 2400000.5 + 0.5)) == list(night_number(JD) - 2400000)


def test_the_cube_s_labels_and_the_julian_nights_agree():
    labels = [nights.night_label(s) for s in STAMPS]
    assert labels == ["2024-10-11"] * 3 + ["2024-10-12"]
    numbers = night_number(JD)
    assert len(set(numbers[:3])) == 1 and numbers[3] == numbers[0] + 1


def test_a_night_across_midnight_ut_is_one_night_everywhere():
    coeffs = Table(dict(bjd=JD, b1=[0.1, 0.2, 0.3, 0.4]))
    index = reconstruct.night_index(coeffs)
    assert len(index) == 2
    assert len(set(diagnostics.night_of(JD[:3]))) == 1
    held = diagnostics.holdout_rows(JD, "even")
    assert held[0] == held[1] == held[2] != held[3]
    rjd = JD - 2400000.0
    binned = lblscan.nightly(rjd, np.zeros(4), np.ones(4))
    assert binned.shape[1] == 2
    table = {series.TIME: rjd, series.VELOCITY: np.zeros(4),
             series.ERROR: np.ones(4)}
    rows = {label: text for label, text, _ in stats.summarise(table)}
    assert rows["Nights"] == "2"
    records = {str(i): dict(mjd=float(r - 0.5), instrument="X")
               for i, r in enumerate(rjd)}
    assert targets.summary(records)["nights"] == 2

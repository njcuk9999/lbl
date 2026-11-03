"""The periodogram of PCA2D's report finds a peak whatever the time span
(texreport.periodogram): its frequencies are a number per peak width, not a
fixed number.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_texreport_periodogram.py
"""
import numpy as np
import pytest
from astropy.timeseries import LombScargle

from lbl.pca2d import texreport


@pytest.mark.parametrize("span", [7620.0, 15000.0])
def test_a_strong_peak_is_found_on_a_long_span(span):
    rng = np.random.default_rng(12)
    ratios = []
    for _ in range(12):
        t = np.sort(rng.uniform(0.0, span, 150))
        period = rng.uniform(5.0, 400.0)
        y = 10.0 * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.3))
        y += rng.normal(0.0, 2.0, t.size)
        e = np.full(t.size, 2.0)
        found = texreport.periodogram(t, y, e)
        true_power = float(LombScargle(t, y, e).power(1.0 / period))
        ratios.append(found[3] / true_power)
    # the best power found is the peak's, to the grid's last bit
    assert min(ratios) > 0.95, ratios


def test_a_short_span_keeps_at_least_its_old_grid():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 300.0, 60))
    y = rng.normal(0.0, 1.0, t.size)
    periods = texreport.periodogram(t, y, np.ones(t.size))[0]
    assert periods.size == 6000
    periods = texreport.periodogram(t, y, np.ones(t.size), pmin=0.55)[0]
    assert periods.size == 12000

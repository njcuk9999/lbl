"""lbl.core.math.robust_polyfit: the mask it returns is the one of the fit it
returns, NaN in x is left out, and its threshold is in units of the median
absolute residual. Also the unit of the pressure of air_index."""
import numpy as np

from lbl.core import math as mp


def _data(seed, npts=200):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, npts))
    y = 2.0 + 0.5 * x + rng.normal(0, 0.1, npts)
    bad = rng.random(npts) < 0.1
    y[bad] += rng.normal(0, 3.0, np.sum(bad))
    return x, y


def test_the_mask_is_the_one_of_the_returned_fit():
    for seed in range(100):
        x, y = _data(seed)
        fit, mask = mp.robust_polyfit(x, y, 1, 3)
        assert np.allclose(fit, np.polyfit(x[mask], y[mask], 1), rtol=0,
                           atol=1e-12)
        # every point used is within the threshold of that fit
        res = np.abs(y - np.polyval(fit, x))
        assert np.all(res[mask] < 3 * np.nanmedian(res))


def test_nan_in_x_or_y_is_left_out():
    x, y = _data(1)
    x[5], y[17] = np.nan, np.nan
    fit, mask = mp.robust_polyfit(x, y, 1, 3)
    assert not mask[5] and not mask[17]
    assert abs(fit[0] - 0.5) < 0.05 and abs(fit[1] - 2.0) < 0.05


def test_the_threshold_is_in_median_absolute_residuals():
    """Gaussian residuals cut at 3 median absolute residuals: 2.02 sigma,
    so about 4.3% of the points left out"""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 200000)
    y = rng.normal(0, 1, len(x))
    _, mask = mp.robust_polyfit(x, y, 0, 3)
    assert abs(np.mean(~mask) - 0.043) < 0.004


def test_air_index_pressure_is_in_mm_hg():
    """The density term of the formula is 1 at 15 C and 760 mm Hg"""
    wave = np.array([500.0])
    n_760 = mp.air_index(wave, 15.0, 760.0) - 1
    n_380 = mp.air_index(wave, 15.0, 380.0) - 1
    assert abs(n_380 / n_760 - 0.5) < 1e-3
    # (n - 1) of dry air at 500 nm, 15 C, 1 atm: 2.79e-4
    assert abs(n_760[0] - 2.79e-4) < 0.01e-4

"""The profile fitted to the CCF of the template (lbl.core.math.gauss_fit_e):
its fwhm parameter is the full width at half maximum of its main component,
whatever its shape factor."""
import warnings

import numpy as np
from scipy.optimize import brentq, curve_fit

from lbl.core import math as mp


def _half_width(fwhm, expo, ears=1e9):
    """Half width at half depth of the profile, found numerically"""
    depth0 = 1 - mp.gauss_fit_e(0.0, 0.0, fwhm, 0.5, ears, expo)

    def at_half(x):
        return (1 - mp.gauss_fit_e(x, 0.0, fwhm, 0.5, ears, expo)) - depth0 / 2
    return brentq(at_half, 1e-6 * fwhm, 5 * fwhm)


def test_fwhm_is_the_width_at_half_depth_without_ears():
    for expo in (1.0, 1.42, 2.0, 3.0):
        assert abs(2 * _half_width(5000.0, expo) / 5000.0 - 1) < 1e-10


def test_a_gaussian_fitted_returns_its_fwhm():
    sigma = 2500.0
    x = np.linspace(-30000, 30000, 601)
    y = 1 - 0.4 * np.exp(-0.5 * x ** 2 / sigma ** 2)

    def no_ears(xx, x0, fwhm, amp, expo):
        return mp.gauss_fit_e(xx, x0, fwhm, amp, 1e9, expo)
    with warnings.catch_warnings():
        # a perfect fit has no covariance to estimate
        warnings.simplefilter('ignore')
        coeffs, _ = curve_fit(no_ears, x, y, p0=[100.0, 4000.0, 0.3, 1.8])
    assert abs(coeffs[1] / (mp.fwhm_value() * sigma) - 1) < 1e-6
    assert abs(coeffs[3] - 2) < 1e-6


def test_ears_narrow_the_profile_at_half_depth():
    """The ears, of opposite sign, take about 6% off for ears of 3"""
    ratio = 2 * _half_width(5000.0, 2.0, ears=3.0) / 5000.0
    assert 0.93 < ratio < 0.95

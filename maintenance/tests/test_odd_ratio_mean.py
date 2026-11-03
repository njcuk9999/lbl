"""The weighted mean of Artigau et al. 2022, Appendix B
(lbl.core.math.odd_ratio_mean): the outlier floor it uses by default and its
answer when no point has any weight left, compiled or not."""
import numpy as np

from lbl.core import math as mp


def _python_version():
    """The same function without numba: numba keeps the source in py_func"""
    return mp.odd_ratio_mean.py_func


def test_the_default_floor_is_the_one_of_the_paper():
    """A point 3.5 sigma away: with f0 = 2e-3 it has a weight of about 0.5,
    its Gaussian being about as high as the floor"""
    sigma = 1.0
    gauss = np.exp(-0.5 * 3.5 ** 2)
    weight = (1 - 2e-3) * gauss / ((1 - 2e-3) * gauss + 2e-3)
    assert abs(weight - 0.5) < 0.03
    # many points at zero hold the mean, one at 3.5 sigma pulls it by its
    #   weight: a first-order check of the weight actually used
    value = np.concatenate([np.zeros(10000), [3.5]])
    error = np.full_like(value, sigma)
    mean, _ = mp.odd_ratio_mean(value, error)
    assert abs(mean * 10000 / 3.5 - weight) < 0.01
    # the default and the value given explicitly are the same thing
    assert (mp.odd_ratio_mean(value, error) ==
            mp.odd_ratio_mean(value, error, 2e-3))
    assert (mp.odd_ratio_mean(value, error) !=
            mp.odd_ratio_mean(value, error, 2e-4))


def test_the_compiled_and_the_python_versions_agree():
    rng = np.random.default_rng(22)
    for _ in range(50):
        size = rng.integers(3, 300)
        value = rng.normal(0, 1, size)
        value[rng.random(size) < 0.02] += 30.0
        error = rng.uniform(0.5, 2.0, size)
        error[rng.random(size) < 0.05] = np.nan
        fast = mp.odd_ratio_mean(value, error)
        slow = _python_version()(value, error)
        assert np.allclose(fast, slow, rtol=1e-13, atol=0)


def test_all_the_weights_at_zero_give_nan_and_inf():
    """Two points far enough apart: the median is half way, both are beyond
    the distance where the Gaussian rounds to nothing next to the floor. This
    used to raise ZeroDivisionError once compiled."""
    value, error = np.array([0.0, 80.0]), np.array([1.0, 1.0])
    for func in (mp.odd_ratio_mean, _python_version()):
        with np.errstate(all='ignore'):
            mean, err = func(value, error)
        assert np.isnan(mean)
        assert np.isinf(err)


def test_no_finite_point_and_no_finite_weight_give_nan():
    nans = np.full(4, np.nan)
    assert np.all(np.isnan(mp.odd_ratio_mean(nans, nans)))
    assert np.all(np.isnan(mp.odd_ratio_mean(np.array([1.0, 1.0, 1.0]),
                                             np.zeros(3))))


def test_known_mean_and_error_without_outliers():
    """Close to the plain weighted mean when nothing is an outlier"""
    rng = np.random.default_rng(3)
    value = 12.0 + rng.normal(0, 1, 100000)
    mean, err = mp.odd_ratio_mean(value, np.ones_like(value))
    assert abs(mean - 12.0) < 5 * 1 / np.sqrt(len(value))
    assert abs(err * np.sqrt(len(value)) - 1) < 0.01

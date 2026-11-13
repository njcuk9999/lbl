"""The Gaussian-weighted Savitzky-Golay filter through scattered points
(lbl.core.gaussian_savgol): the value and derivatives of a known function,
the uncertainties it gives against the scatter they have, and its outliers."""
import numpy as np
import pytest

from lbl.core.gaussian_savgol import scatter_savgol


def _function(x):
    """Two Gaussians of width 1 and their first three derivatives"""
    out = np.zeros((len(x), 4))
    for centre, depth in ((-1.0, 0.6), (2.0, 0.3)):
        u = x - centre
        g = depth * np.exp(-0.5 * u ** 2)
        out[:, 0] -= g
        out[:, 1] -= -u * g
        out[:, 2] -= (u ** 2 - 1) * g
        out[:, 3] -= (3 * u - u ** 3) * g
    out[:, 0] += 1.0
    return out


def _cloud(seed, npoint=30000, sigma=0.01):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6, 6, npoint)
    noise = sigma * (1 + rng.uniform(0, 1, npoint))
    y = _function(x)[:, 0] + rng.normal(0, 1, npoint) * noise
    return x, y, noise


def test_the_value_and_derivatives_are_those_of_the_function():
    x, y, noise = _cloud(1)
    grid = np.linspace(-4, 4, 161)
    derivs, errs = scatter_savgol(x, y, grid, 0.5, weights=1 / noise ** 2)
    truth = _function(grid)
    # the noise and the bias of a cubic over +-0.75 on lines of width 1
    for k, tol in enumerate((3e-3, 5e-3, 5e-2, 0.15)):
        assert np.sqrt(np.mean((derivs[:, k] - truth[:, k]) ** 2)) < tol, k
    assert np.all(np.isfinite(errs))


def test_the_uncertainty_is_the_scatter_over_realisations():
    grid = np.linspace(-3, 3, 31)
    values, errors = [], []
    for seed in range(60):
        x, y, noise = _cloud(seed, npoint=6000)
        derivs, errs = scatter_savgol(x, y, grid, 0.5, weights=1 / noise ** 2)
        values.append(derivs)
        errors.append(errs)
    # the value and the first derivative (those the template and the mask use)
    scatter = np.std(values, axis=0)
    ratio = np.median(scatter / np.median(errors, axis=0), axis=0)
    assert ratio[:2] == pytest.approx([1.0, 1.0], abs=0.15)


def test_outliers_are_clipped():
    x, y, noise = _cloud(2)
    grid = np.linspace(-4, 4, 161)
    clean, _ = scatter_savgol(x, y, grid, 0.5, weights=1 / noise ** 2)
    rng = np.random.default_rng(5)
    bad = rng.choice(len(y), len(y) // 100, replace=False)
    y[bad] += 30 * noise[bad] * rng.choice([-1, 1], len(bad))
    dirty, _ = scatter_savgol(x, y, grid, 0.5, weights=1 / noise ** 2)
    assert np.max(np.abs(dirty[:, 0] - clean[:, 0])) < 2e-3


def test_no_value_without_points_on_both_sides():
    # a gap in the points: no extrapolation into it, values on either side
    x, y, noise = _cloud(3)
    keep = (x < 0) | (x > 1)
    grid = np.array([-0.5, 0.2, 0.5, 0.8, 1.5])
    derivs, errs = scatter_savgol(x[keep], y[keep], grid, 0.1,
                                  weights=1 / noise[keep] ** 2)
    assert np.all(np.isfinite(derivs[[0, 4]]))
    assert np.all(np.isnan(derivs[1:4])) and np.all(np.isnan(errs[1:4]))


def test_unsorted_and_invalid_points_are_handled():
    x, y, noise = _cloud(4)
    grid = np.linspace(-2, 2, 21)
    ref, _ = scatter_savgol(x, y, grid, 0.5, weights=1 / noise ** 2)
    y2, w2 = np.array(y), 1 / noise ** 2
    y2[::50] = np.nan
    w2[1::50] = 0.0
    order = np.random.default_rng(0).permutation(len(x))
    out, _ = scatter_savgol(x[order], y2[order], grid, 0.5,
                            weights=w2[order])
    assert np.allclose(out[:, 0], ref[:, 0], atol=2e-3)


def test_no_polynomial_across_a_hole():
    """A stretch with no point (every spectrum blanked there): with max_gap
    and min_balance, no value inside it nor at its edges, where a cubic held
    by the few points beyond it would swing; elsewhere the same values"""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 100.0, 20000))
    y = np.sin(x / 3.0) + rng.normal(0.0, 0.01, x.size)
    hole = (x > 50.0) & (x < 51.2)
    x, y = x[~hole], y[~hole]
    x_out = np.linspace(5.0, 95.0, 1801)
    old, _ = scatter_savgol(x, y, x_out, 1.0, polyorder=3, nderiv=0)
    new, _ = scatter_savgol(x, y, x_out, 1.0, polyorder=3, nderiv=0,
                            max_gap=0.5, min_balance=0.1)
    inside = (x_out > 50.3) & (x_out < 50.9)
    assert np.isfinite(old[inside, 0]).all()        # bridged before
    assert not np.isfinite(new[inside, 0]).any()
    far = np.abs(x_out - 50.6) > 2.0
    assert np.allclose(new[far, 0], old[far, 0])
    good = np.isfinite(new[:, 0])
    assert np.max(np.abs(new[good, 0] - np.sin(x_out[good] / 3.0))) < 0.01

"""A star measured against another star's template: its static mismatch
(metallicity, still in the star's frame) is taken out of the residual
before the rejection tests it, and a telluric residual is not taken with it,
even at a narrow BERV (outliers.star_static)."""
import numpy as np

from lbl.pca2d import outliers


def _rows(deltas, width=400, star_at=200.0, sky_at=150, seed=3):
    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=float)
    n = len(deltas)
    resid = rng.normal(0.0, 1.0, (n, width))
    # the mismatch: a line 4 sigma deep in every exposure, in the star's
    #   frame (star column j is the row's column j + delta)
    for i, d in enumerate(deltas):
        resid[i] += 4.0 * np.exp(-0.5 * ((x - d - star_at) / 2.5) ** 2)
    # a telluric residual, fixed on the detector, where the sky absorbs
    trans = np.ones((n, width))
    trans[:, sky_at - 3:sky_at + 4] = 0.6
    resid[:, sky_at - 2:sky_at + 3] += 5.0
    return resid, trans


def test_the_mismatch_goes_and_the_sky_stays():
    for deltas in (np.linspace(-15, 15, 60), np.linspace(-0.4, 0.4, 60)):
        resid, trans = _rows(deltas)
        tgroup = np.zeros(len(deltas), dtype=int)
        static = outliers.star_static(resid, deltas, tgroup, [True], trans)
        left = outliers.take_static(resid.copy(), static, deltas, tgroup,
                                    [True])
        # the star's line: from 4 sigma in every row to noise
        at_line = np.array([left[i, int(round(200 + d))]
                            for i, d in enumerate(deltas)])
        assert abs(np.mean(at_line)) < 0.5
        # the sky's residual: still there, the median never saw it
        assert np.mean(left[:, 150]) > 4.0


def test_a_star_with_its_own_template_is_untouched():
    deltas = np.linspace(-15, 15, 60)
    resid, trans = _rows(deltas)
    tgroup = np.zeros(len(deltas), dtype=int)
    static = outliers.star_static(resid, deltas, tgroup, [False], trans)
    assert not static.any()
    left = outliers.take_static(resid.copy(), static, deltas, tgroup, [False])
    assert np.array_equal(left, resid)

"""Each observer component kept at a column only where it is significant there.

The correction divides b_n . Q out of every file, and Q was estimated from the
same noisy spectra it corrects: its error at a column is common to every
exposure, a fixed pattern in the observer's frame that lands somewhere else in
the star's at every BERV. Where the component is not significant at a column,
all exposures together, the correction adds that pattern and removes nothing.
The significance is the basis's, not that of one exposure's share of it: a
pattern at half a sigma in every exposure is detected at about half a sigma
times the square root of their number, and it is kept. So each component is
shrunk at each column by the positive-part James-Stein factor of its own
significance there:

    F_i = sum_n w_ni b_n b_n^T        the components' Fisher matrix at column i
    sigma(Q_ji)^2 = [F_i^-1]_jj        every row weighted by its own noise
    s_ji = max(0, 1 - 1/z_ji^2),       z_ji = Q_ji / sigma(Q_ji)

zero where |z| <= 1, close to one where |z| >> 1, and continuous in between.
The factor belongs to the column and is the same for every exposure, so a
pixel's time series stays one series; each exposure still gets its own
amount of correction through its b_n. Per component rather than one factor
per column, since at one column the water can be significant and the oxygen
not. The pixels themselves keep their weight: where the correction goes to
zero, the file keeps its flux as delivered.

Two Savitzky-Golay variants, each over one resolution element (2026-09-11):

  * the significance smoothed, z^2 averaged over the element before the factor
    is taken, so a telluric line is judged over its width rather than pixel
    by pixel; the components themselves are not touched. Order 0, a weighted
    mean: z^2 is positive, and a higher order dips below zero beside a strong
    line;
  * chosen components smoothed, with LBL's template filter, their variance
    reduced by the filter's noise factor before the factor is taken, so a
    resolved line gains significance and the noise stays normalised. For the
    components that are mostly noise; the others may carry pixel-level
    detector structure and are left alone.
"""

import numpy as np

from .resolution import noise_factor, smooth


def component_variance(b, w, chi2_scale=1.0):
    """(J, M) variance of every component at every column: the diagonal of
    the inverse Fisher matrix, every row weighted by its own w, the weights
    scaled by `chi2_scale`. Infinite where no row weights a component."""
    b = np.asarray(b, dtype=float)
    n_comp = b.shape[1]
    n_cols = w.shape[1]
    fisher = np.zeros((n_cols, n_comp, n_comp))
    for j in range(n_comp):
        for k in range(j, n_comp):
            f = (b[:, j] * b[:, k]) @ w / float(chi2_scale)
            fisher[:, j, k] = f
            fisher[:, k, j] = f
    diag = fisher[:, np.arange(n_comp), np.arange(n_comp)]
    ok = np.all(diag > 0, axis=1)
    var = np.full((n_comp, n_cols), np.inf)
    if ok.any():
        sub = fisher[ok]
        # a ridge far below any real curvature, so a column where two
        # components are nearly degenerate still inverts
        sub += 1e-12 * np.trace(sub, axis1=1, axis2=2)[:, None, None] * np.eye(n_comp)
        var[:, ok] = np.linalg.inv(sub)[:, np.arange(n_comp), np.arange(n_comp)].T
    return var


def james_stein(z2):
    """max(0, 1 - 1/z^2), elementwise; zero where z^2 is zero or undefined."""
    z2 = np.where(np.isfinite(z2), z2, 0.0)
    return np.clip(1.0 - 1.0 / np.maximum(z2, 1e-300), 0.0, 1.0)


def shrink_factors(Q, b, w, chi2_scale=1.0, smooth_fwhm=None):
    """(J, M) factors in [0, 1] for the observer components Q (J, M).

    `b` holds the rows' amplitudes (rows, J) and `w` their weights (rows, M),
    one row per cube row. `chi2_scale`, the fit's reduced chi2, scales the
    weights so the errors are the ones the residual shows rather than the
    formal ones. With `smooth_fwhm`, in samples, each component's z^2 is
    averaged over that FWHM before the factor is taken. Columns no row
    weights get zero.
    """
    Q = np.asarray(Q, dtype=float)
    z2 = Q ** 2 / component_variance(b, w, chi2_scale)
    if smooth_fwhm:
        z2 = np.array([smooth(row, smooth_fwhm, polyorder=0) for row in z2])
    return james_stein(z2)


def correction_basis(Q, b, w, chi2_scale=1.0, shrink=True, shrink_smooth=False,
                     smooth_which=(), fwhm=None):
    """(Q_correct, factors): the observer basis the correction divides out.

    What reconstruct.correct_many divides out of the files and what panels 3
    and 6 of the sequence figure show, from one place so the two cannot
    differ. The components in `smooth_which` (0-based) are smoothed and shrunk
    by the significance they then have; the others are shrunk by their own,
    its z^2 averaged over `fwhm` samples with `shrink_smooth`, or left whole
    when `shrink` is off. Column by column apart from those smoothings, so a
    block of columns with a margin wider than the filter gives the columns
    inside it exactly what the whole grid does.
    """
    Q = np.asarray(Q, dtype=float)
    var = component_variance(b, w, chi2_scale)
    out = Q.copy()
    factors = np.ones_like(Q)
    which = list(smooth_which or [])
    if which:
        smoothed, f_smooth = smoothed_components(Q, var, which, fwhm)
        out[which] = smoothed[which]
        factors[which] = f_smooth[which]
    others = [j for j in range(Q.shape[0]) if j not in which]
    if shrink and others:
        z2 = Q[others] ** 2 / var[others]
        if shrink_smooth:
            z2 = np.array([smooth(row, fwhm, polyorder=0) for row in z2])
        factors[others] = james_stein(z2)
        out[others] = factors[others] * Q[others]
    return out, factors


def smoothed_components(Q, var, which, fwhm):
    """Q with the components in `which` (0-based) smoothed by LBL's template
    filter over `fwhm` samples and shrunk by the significance they then have,
    their variance reduced by the filter's noise factor; the others returned
    as they are, unshrunk. Returns (Q_out, factors) with factors 1 on the
    untouched components."""
    Q = np.asarray(Q, dtype=float)
    out = Q.copy()
    factors = np.ones_like(Q)
    reduce = noise_factor(fwhm)
    for j in which:
        smoothed = smooth(Q[j], fwhm)
        factors[j] = james_stein(smoothed ** 2 / (var[j] * reduce))
        out[j] = factors[j] * smoothed
    return out, factors

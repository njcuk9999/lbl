#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Gaussian-weighted Savitzky-Golay filter for points that are not on a grid:
a local polynomial fitted by weighted least squares around each output point,
with Gaussian weights in x, the points' own weights, and a clip of the
outliers on the local median absolute deviation.

It is what the template is made of (lbl.science.template_scatter): the pixels
of all the spectra of a star, each at its own wavelength in the star's frame,
form one cloud of points, and the template and its derivatives are the local
polynomial at each point of the template's grid. No spectrum is ever
resampled, and the derivatives are those of the fitted polynomial.

From templatepy (github.com/eartigau/templatepy, gaussian_savgol.py, MIT,
Etienne Artigau), with the derivatives 0 to n returned in one pass, with
their uncertainties.

@author: artigau
"""
import math

import numpy as np
from numba import njit, prange


# =============================================================================
# Define functions
# =============================================================================
@njit(cache=True)
def _solve(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """mat x = vec by Gaussian elimination with partial pivoting (mat small
    and square); a zero pivot gives a zero coefficient"""
    m = len(vec)
    aug = np.zeros((m, m + 1))
    for i in range(m):
        for j in range(m):
            aug[i, j] = mat[i, j]
        aug[i, m] = vec[i]
    for i in range(m):
        piv = i
        for k in range(i + 1, m):
            if abs(aug[k, i]) > abs(aug[piv, i]):
                piv = k
        if piv != i:
            for j in range(m + 1):
                tmp = aug[i, j]
                aug[i, j] = aug[piv, j]
                aug[piv, j] = tmp
        if abs(aug[i, i]) < 1e-300:
            continue
        for k in range(i + 1, m):
            fac = aug[k, i] / aug[i, i]
            for j in range(i, m + 1):
                aug[k, j] -= fac * aug[i, j]
    out = np.zeros(m)
    for i in range(m - 1, -1, -1):
        if abs(aug[i, i]) < 1e-300:
            continue
        acc = aug[i, m]
        for j in range(i + 1, m):
            acc -= aug[i, j] * out[j]
        out[i] = acc / aug[i, i]
    return out


@njit(cache=True)
def _normal(u: np.ndarray, yy: np.ndarray, ww: np.ndarray, m: int):
    """The normal equations of a weighted polynomial fit in u: V'WV, V'Wy"""
    mat = np.zeros((m, m))
    vec = np.zeros(m)
    powers = np.empty(m)
    for i in range(len(u)):
        w = ww[i]
        if w <= 0:
            continue
        powers[0] = 1.0
        for j in range(1, m):
            powers[j] = powers[j - 1] * u[i]
        for j in range(m):
            vec[j] += w * powers[j] * yy[i]
            for k in range(m):
                mat[j, k] += w * powers[j] * powers[k]
    return mat, vec


@njit(cache=True)
def _poly(coeffs: np.ndarray, u: float) -> float:
    acc = 0.0
    for j in range(len(coeffs) - 1, -1, -1):
        acc = acc * u + coeffs[j]
    return acc


@njit(cache=True)
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weighted median of values (weights >= 0)"""
    idx = np.argsort(values)
    total = 0.0
    for i in range(len(values)):
        total += weights[i]
    if total <= 0:
        return np.nan
    acc = 0.0
    for i in range(len(values)):
        acc += weights[idx[i]]
        if acc >= 0.5 * total:
            return values[idx[i]]
    return values[idx[len(values) - 1]]


@njit(parallel=True, cache=True)
def _scatter_savgol(x, y, w, x_out, sigma_x, radius, polyorder, nderiv,
                    sigma_clip, clip_iter, min_side):
    n_out = len(x_out)
    m = polyorder + 1
    derivs = np.full((n_out, nderiv + 1), np.nan)
    errs = np.full((n_out, nderiv + 1), np.nan)
    for i in prange(n_out):
        x0 = x_out[i]
        left = np.searchsorted(x, x0 - radius)
        right = np.searchsorted(x, x0 + radius)
        n = right - left
        if n < m + 1:
            continue
        # points on both sides: the polynomial is never extrapolated (edges
        #   of an order, a masked telluric band)
        mid = np.searchsorted(x, x0)
        if (mid - left) < min_side or (right - mid) < min_side:
            continue
        # u in units of the radius, for the conditioning of the fit
        u = np.empty(n)
        g = np.empty(n)
        yy = np.empty(n)
        wi = np.empty(n)
        for k in range(n):
            d = x[left + k] - x0
            u[k] = d / radius
            g[k] = math.exp(-0.5 * (d / sigma_x) ** 2)
            yy[k] = y[left + k]
            wi[k] = w[left + k]
        base = g * wi
        cw = base.copy()
        coeffs = np.zeros(m)
        zabs = np.empty(n)
        for it in range(clip_iter + 1):
            mat, vec = _normal(u, yy, cw, m)
            coeffs = _solve(mat, vec)
            if it == clip_iter or sigma_clip <= 0:
                break
            # residuals in units of each point's own error (w = 1/sigma**2),
            #   their robust scale from the median absolute value
            for k in range(n):
                zabs[k] = abs(yy[k] - _poly(coeffs, u[k])) * math.sqrt(wi[k])
            scale = 1.4826 * _weighted_median(zabs, g)
            if not scale > 0:
                break
            ngood = 0
            for k in range(n):
                if zabs[k] > sigma_clip * scale:
                    cw[k] = 0.0
                else:
                    cw[k] = base[k]
                    ngood += 1
            if ngood < m + 1:
                cw = base.copy()
                mat, vec = _normal(u, yy, cw, m)
                coeffs = _solve(mat, vec)
                break
        # the derivatives in x: d^k/dx^k at x0 = k! c_k / radius^k
        for k in range(min(nderiv, polyorder) + 1):
            derivs[i, k] = coeffs[k] * math.gamma(k + 1.0) / radius ** k
        for k in range(polyorder + 1, nderiv + 1):
            derivs[i, k] = 0.0
        # the uncertainties: the sandwich covariance of the weighted fit,
        #   the points' errors scaled by the mean square of their normalised
        #   residuals (the weights w are relative)
        sumg = 0.0
        sumz2 = 0.0
        for k in range(n):
            if cw[k] > 0:
                r = (yy[k] - _poly(coeffs, u[k])) * math.sqrt(wi[k])
                sumz2 += g[k] * r * r
                sumg += g[k]
        if sumg <= 0:
            continue
        s2 = sumz2 / sumg
        mat, vec = _normal(u, yy, cw, m)
        # B = V' diag(cw**2 / w) V
        bmat = np.zeros((m, m))
        powers = np.empty(m)
        for k in range(n):
            if cw[k] <= 0:
                continue
            q = cw[k] * cw[k] / wi[k]
            powers[0] = 1.0
            for j in range(1, m):
                powers[j] = powers[j - 1] * u[k]
            for j in range(m):
                for l in range(m):
                    bmat[j, l] += q * powers[j] * powers[l]
        # of c_k: ek' A^-1 B A^-1 ek, with a = A^-1 ek (A symmetric)
        for k in range(min(nderiv, polyorder) + 1):
            ek = np.zeros(m)
            ek[k] = 1.0
            a = _solve(mat, ek)
            var = 0.0
            for j in range(m):
                for l in range(m):
                    var += a[j] * bmat[j, l] * a[l]
            if var > 0:
                errs[i, k] = math.sqrt(s2 * var) * math.gamma(k + 1.0) / \
                    radius ** k
        for k in range(polyorder + 1, nderiv + 1):
            errs[i, k] = 0.0
    return derivs, errs


def scatter_savgol(x: np.ndarray, y: np.ndarray, x_out: np.ndarray,
                   window_fwhm: float, weights: np.ndarray = None,
                   polyorder: int = 3, nderiv: int = 3,
                   sigma_clip: float = 5.0, clip_iter: int = 3,
                   min_side: int = 2, presorted: bool = False):
    """
    A Gaussian-weighted local polynomial through scattered points, at x_out

    :param x: np.ndarray, the points' abscissae (any order unless presorted)
    :param y: np.ndarray, their values (non-finite values are left out)
    :param x_out: np.ndarray, where the polynomial is evaluated
    :param window_fwhm: float, the FWHM of the Gaussian weight, in units of x;
                        the points within 1.5 FWHM of x_out are used
    :param weights: np.ndarray or None, the points' weights, 1 / sigma**2 up
                    to a constant (None: equal); <= 0 or non-finite left out
    :param polyorder: int, the order of the polynomial
    :param nderiv: int, the highest derivative returned
    :param sigma_clip: float, points further than sigma_clip times the robust
                       scale of the normalised residuals are rejected
                       (<= 0: no clip)
    :param clip_iter: int, the iterations of the clip
    :param min_side: int, the points needed on each side of x_out
    :param presorted: bool, x is already sorted and all points are valid

    :return: (derivs, errs), both (len(x_out), nderiv + 1): the value and its
             derivatives in x (NaN where there are too few points), and their
             uncertainties
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_out = np.asarray(x_out, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if not presorted:
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        order = np.argsort(x[good], kind='stable')
        x = np.ascontiguousarray(x[good][order])
        y = np.ascontiguousarray(y[good][order])
        w = np.ascontiguousarray(w[good][order])
    sigma_x = window_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    radius = 1.5 * window_fwhm
    return _scatter_savgol(x, y, w, x_out, sigma_x, radius, int(polyorder),
                           int(nderiv), float(sigma_clip), int(clip_iter),
                           int(min_side))


# =============================================================================
# End of code
# =============================================================================

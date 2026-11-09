#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Gaussian-weighted Savitzky-Golay filter for points that are not on a grid:
a local polynomial fitted by weighted least squares around each output point,
with Gaussian weights in x, the points' own weights, and a clip of the
outliers on the local scale of the normalised residuals.

It is what the template is made of (lbl.science.template_scatter): the pixels
of all the spectra of a star, each at its own wavelength in the star's frame,
form one cloud of points, and the template and its derivatives are the local
polynomial at each point of the template's grid. No spectrum is ever
resampled, and the derivatives are those of the fitted polynomial.

Each fit is one pass over the points of its window: the normal equations of
a polynomial in u are made of the moments sum(weight * u**p), p = 0 to
2 * polyorder, so these moments (and those of the squared weights, for the
covariance) are all that is summed; the residuals come from the same sums.

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


@njit(parallel=True, cache=True)
def _fit(x, y, w, x_out, sigma_x, radius, polyorder, min_side):
    """The coefficients of the local polynomial in u = (x - x0) / radius at
    each x0 of x_out, their uncertainties, and the mean square of the
    normalised residuals in the window (NaN without enough points); the
    points of weight 0 are left out"""
    n_out = len(x_out)
    m = polyorder + 1
    npow = 2 * m - 1
    coeffs = np.full((n_out, m), np.nan)
    errors = np.full((n_out, m), np.nan)
    msr = np.full(n_out, np.nan)
    inv_var = 1.0 / sigma_x ** 2
    for i in prange(n_out):
        x0 = x_out[i]
        left = np.searchsorted(x, x0 - radius)
        right = np.searchsorted(x, x0 + radius)
        if right - left < m + 1:
            continue
        # the moments: sum g w u^p (normal equations), sum g^2 w u^p
        #   (sandwich covariance), sum g w y u^j, sum g w y^2, sum g
        mom = np.zeros(npow)
        mom2 = np.zeros(npow)
        vec = np.zeros(m)
        syy = 0.0
        sumg = 0.0
        nleft = 0
        nright = 0
        for k in range(left, right):
            wk = w[k]
            if wk <= 0:
                continue
            d = x[k] - x0
            u = d / radius
            g = math.exp(-0.5 * d * d * inv_var)
            gw = g * wk
            g2w = g * gw
            yk = y[k]
            power = 1.0
            for p in range(npow):
                mom[p] += gw * power
                mom2[p] += g2w * power
                if p < m:
                    vec[p] += gw * yk * power
                power *= u
            syy += gw * yk * yk
            sumg += g
            if d < 0:
                nleft += 1
            else:
                nright += 1
        # points on both sides: the polynomial is never extrapolated (edges
        #   of an order, a masked telluric band)
        if nleft + nright < m + 1 or nleft < min_side or nright < min_side:
            continue
        mat = np.empty((m, m))
        bmat = np.empty((m, m))
        for j in range(m):
            for l in range(m):
                mat[j, l] = mom[j + l]
                bmat[j, l] = mom2[j + l]
        c = _solve(mat, vec)
        # sum g w (y - P)^2 = syy - 2 c.vec + c' mat c
        ssr = syy
        for j in range(m):
            ssr -= 2.0 * c[j] * vec[j]
            for l in range(m):
                ssr += c[j] * mat[j, l] * c[l]
        # the points' errors scaled by the mean square of their normalised
        #   residuals (the weights w are relative), weighted by g
        s2 = max(ssr, 0.0) / sumg
        msr[i] = s2
        for j in range(m):
            coeffs[i, j] = c[j]
            # the sandwich covariance of c_j: ej' A^-1 B A^-1 ej
            ej = np.zeros(m)
            ej[j] = 1.0
            a = _solve(mat, ej)
            var = 0.0
            for p in range(m):
                for q in range(m):
                    var += a[p] * bmat[p, q] * a[q]
            if var > 0:
                errors[i, j] = math.sqrt(s2 * var)
    return coeffs, errors, msr


@njit(parallel=True, cache=True)
def _model(x, x_out, coeffs, msr, radius):
    """The local polynomial of the nearest x_out at each x, and the RMS of
    the normalised residuals of its window (NaN beyond the radius or where it
    has none)"""
    n = len(x)
    n_out = len(x_out)
    m = coeffs.shape[1]
    out = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    for k in prange(n):
        idx = np.searchsorted(x_out, x[k])
        if idx >= n_out or (idx > 0 and
                            x[k] - x_out[idx - 1] < x_out[idx] - x[k]):
            idx -= 1
        u = (x[k] - x_out[idx]) / radius
        if abs(u) > 1 or not np.isfinite(coeffs[idx, 0]):
            continue
        acc = 0.0
        for j in range(m - 1, -1, -1):
            acc = acc * u + coeffs[idx, j]
        out[k] = acc
        scale[k] = math.sqrt(msr[idx])
    return out, scale


def scatter_savgol(x: np.ndarray, y: np.ndarray, x_out: np.ndarray,
                   window_fwhm: float, weights: np.ndarray = None,
                   polyorder: int = 3, nderiv: int = 3,
                   sigma_clip: float = 5.0, clip_iter: int = 3,
                   min_side: int = 2, presorted: bool = False):
    """
    A Gaussian-weighted local polynomial through scattered points, at x_out

    :param x: np.ndarray, the points' abscissae (any order unless presorted)
    :param y: np.ndarray, their values (non-finite values are left out)
    :param x_out: np.ndarray, where the polynomial is evaluated (sorted)
    :param window_fwhm: float, the FWHM of the Gaussian weight, in units of x;
                        the points within 1.5 FWHM of x_out are used
    :param weights: np.ndarray or None, the points' weights, 1 / sigma**2 up
                    to a constant (None: equal); <= 0 or non-finite left out
    :param polyorder: int, the order of the polynomial
    :param nderiv: int, the highest derivative returned
    :param sigma_clip: float, points further from the polynomial than
                       sigma_clip times the RMS of the normalised residuals
                       of its window are left out, and the polynomials
                       fitted again (<= 0: no clip). The scale is local, so
                       that where the polynomial cannot follow the points
                       (the cores of narrow lines) they are not clipped
                       more than elsewhere
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
    w_fit = w
    for it in range(clip_iter + 1):
        coeffs, errors, msr = _fit(x, y, w_fit, x_out, sigma_x, radius,
                                   int(polyorder), int(min_side))
        if it == clip_iter or sigma_clip <= 0:
            break
        # the residuals in units of each point's own error, against the RMS
        #   of those of the window of the nearest output point
        model, scale = _model(x, x_out, coeffs, msr, radius)
        zabs = np.abs(y - model) * np.sqrt(w)
        keep = ~(zabs > sigma_clip * scale)
        if np.array_equal(keep, w_fit > 0):
            break
        w_fit = np.where(keep, w, 0.0)
    # the derivatives in x: d^k/dx^k at x0 = k! c_k / radius^k (zero above
    #   the order of the polynomial)
    derivs = np.zeros((len(x_out), nderiv + 1))
    errs = np.zeros((len(x_out), nderiv + 1))
    for k in range(min(nderiv, polyorder) + 1):
        factor = math.factorial(k) / radius ** k
        derivs[:, k] = coeffs[:, k] * factor
        errs[:, k] = errors[:, k] * factor
    invalid = ~np.isfinite(coeffs[:, 0])
    derivs[invalid] = np.nan
    errs[invalid] = np.nan
    return derivs, errs


# =============================================================================
# End of code
# =============================================================================

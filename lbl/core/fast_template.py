#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba kernels for lbl_template (same results as the numpy code they
replace, see lbl.core.npreplica).

Created on 2026-09-18
"""
import numpy as np
from numba import njit

from lbl.core import npreplica


@njit(cache=True, error_model='numpy')
def _np_quantile_select(buf, n, q):
    """
    numpy 'linear' quantile q (fraction) of buf[:n] (no NaN), with the two
    order statistics it needs found by quickselect instead of sorting;
    buf[:n] is reordered
    """
    v = (n - 1) * q
    if v >= n - 1:
        prev = n - 1
        nxt = n - 1
        gamma = v - (-1.0)
    elif v < 0:
        prev = 0
        nxt = 0
        gamma = v - 0.0
    else:
        prev = int(np.floor(v))
        nxt = prev + 1
        gamma = v - prev
    a = npreplica.select(buf, n, prev)
    if nxt == prev:
        b = a
    else:
        # the next value in order is the smallest of those after prev
        b = buf[prev + 1]
        for i in range(prev + 2, n):
            if buf[i] < b:
                b = buf[i]
    diff_b_a = b - a
    if gamma >= 0.5:
        return b - diff_b_a * (1 - gamma)
    return a + diff_b_a * gamma


@njit(cache=True, error_model='numpy')
def roll_ratio_sigma(flux, med, shifts, q_lo, q_hi):
    """
    Velocity scan of lbl_template: for each shift,
    tmp = np.roll(flux, shift) / med and
    (np.nanpercentile(tmp, 100 q_hi) - np.nanpercentile(tmp, 100 q_lo)) / 2

    :param flux: float64 1D array
    :param med: float64 1D array (same length)
    :param shifts: int64 1D array of shifts (pixels)
    :param q_lo: float, low quantile as numpy computes it (16 / 100.)
    :param q_hi: float, high quantile as numpy computes it (84 / 100.)

    :return: float64 array, one value per shift
    """
    n = flux.shape[0]
    out = np.empty(shifts.shape[0])
    buf = np.empty(n)
    for s in range(shifts.shape[0]):
        shift = shifts[s]
        m = 0
        for i in range(n):
            j = (i - shift) % n
            if j < 0:
                j += n
            x = flux[j] / med[i]
            if x == x:
                buf[m] = x
                m += 1
        if m == 0:
            out[s] = np.nan
            continue
        low = _np_quantile_select(buf, m, q_lo)
        high = _np_quantile_select(buf, m, q_hi)
        out[s] = (high - low) / 2.0
    return out


@njit(cache=True, error_model='numpy')
def nanpercentile_rows(cube, qs):
    """
    np.nanpercentile(cube, 100 qs, axis=1) for a 2D float64 cube (qs as the
    fractions numpy computes, p / 100.): shape (len(qs), cube.shape[0])
    """
    nrow, ncol = cube.shape
    out = np.empty((qs.shape[0], nrow))
    buf = np.empty(ncol)
    for r in range(nrow):
        m = 0
        for c in range(ncol):
            x = cube[r, c]
            if x == x:
                buf[m] = x
                m += 1
        if m == 0:
            for iq in range(qs.shape[0]):
                out[iq, r] = np.nan
            continue
        srt = buf[:m]
        srt.sort()
        for iq in range(qs.shape[0]):
            out[iq, r] = npreplica.np_quantile_sorted(srt, m, qs[iq])
    return out

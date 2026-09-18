#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba kernel for lbl.science.general.estimate_noise_model (same results as
the original python loop, see lbl.core.npreplica).

Created on 2026-09-18
"""
import numpy as np
from numba import njit

from lbl.core import math as mp
from lbl.core import npreplica

# the two percentiles of lbl.core.math.estimate_sigma (sigma=1), as the
#   fractions np.nanpercentile computes from them (p / 100)
_P1 = (1 - (1 - mp.normal_fraction(1.0)) / 2) * 100
Q_HI = float(np.true_divide(_P1, np.float64(100)))
Q_LO = float(np.true_divide(100 - _P1, np.float64(100)))


@njit(cache=True, error_model='numpy')
def noise_model_windows(residuals, npoints, q_hi, q_lo):
    """
    The loop of estimate_noise_model for one order: robust sigma
    (estimate_sigma) of the residuals in boxes of npoints pixels centred
    every npoints // 4 pixels; boxes with <= 50% finite values (compared
    to npoints) and zero sigmas are NaN.

    :param residuals: float64 1D array, spectrum - model for this order
    :param npoints: int, the box size in pixels
    :param q_hi: float, upper quantile of estimate_sigma (Q_HI)
    :param q_lo: float, lower quantile of estimate_sigma (Q_LO)

    :return: box centres (int64 array) and sigma in each box
    """
    npix = residuals.shape[0]
    indices = np.arange(0, npix, npoints // 4)
    sigma = np.zeros(indices.shape[0])
    buf = np.empty(npoints + 1)
    for it in range(indices.shape[0]):
        istart = indices[it] - npoints // 2
        iend = indices[it] + npoints // 2
        if istart < 0:
            istart = 0
        if iend > npix:
            iend = npix
        tmp = residuals[istart:iend]
        nvalid = 0
        for i in range(tmp.shape[0]):
            if np.isfinite(tmp[i]):
                nvalid += 1
        frac_valid = nvalid / npoints
        if frac_valid > 0.5:
            if buf.shape[0] < tmp.shape[0]:
                buf = np.empty(tmp.shape[0])
            sigma[it] = npreplica.estimate_sigma(tmp, q_hi, q_lo, buf)
    for it in range(indices.shape[0]):
        if sigma[it] == 0:
            sigma[it] = np.nan
    return indices, sigma

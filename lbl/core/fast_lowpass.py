#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba kernel for the window loop of lbl.core.math.lowpassfilter (same
results as the original python loop, see lbl.core.npreplica).

Created on 2026-09-18
"""
import numpy as np
from numba import njit

from lbl.core import npreplica


@njit(cache=True, error_model='numpy')
def lowpass_windows(input_vect, width):
    """
    Window loop of lowpassfilter: for boxes of `width` pixels every
    width // 4 pixels (starting half a box before the vector), the mean
    pixel position (bottleneck nanmean of the integer positions) and the
    NaN-median of the values (bottleneck nanmedian); boxes with < 3 pixels
    or < 3 finite values are skipped.

    :param input_vect: float64 1D array
    :param width: int, box width in pixels (width // 4 > 0)

    :return: xmed, ymed (float64 arrays)
    """
    nvect = input_vect.shape[0]
    start = -width // 2
    stop = nvect + width // 2
    step = width // 4
    nmax = 0
    if stop > start:
        nmax = (stop - start + step - 1) // step
    xmed = np.empty(nmax)
    ymed = np.empty(nmax)
    buf = np.empty(max(width, 1))
    count = 0
    for iw in range(nmax):
        it = start + iw * step
        low_bound = it
        high_bound = it + width
        if low_bound < 0:
            low_bound = 0
        if high_bound > (nvect - 1):
            high_bound = nvect - 1
        npix = high_bound - low_bound
        if npix < 3:
            continue
        nfinite = 0
        for i in range(low_bound, high_bound):
            if np.isfinite(input_vect[i]):
                nfinite += 1
        if nfinite < 3:
            continue
        # bottleneck nanmean of the (integer) pixel positions
        asum = 0.0
        for i in range(low_bound, high_bound):
            asum += i
        xmed[count] = asum / npix
        # bottleneck nanmedian of the values
        for i in range(npix):
            buf[i] = input_vect[low_bound + i]
        ymed[count] = npreplica.bn_nanmedian(buf, npix)
        count += 1
    return xmed[:count], ymed[:count]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-11-01

@author: cook
"""
import warnings
from typing import Optional, Tuple

import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_template.py'
__STRNAME__ = 'LBL Template'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
InstrumentsList = select.InstrumentsList
InstrumentsType = select.InstrumentsType
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = io.log


# =============================================================================
# Define functions
# =============================================================================
def _s1d_slopevector(params: ParamDict, e2ds: np.ndarray,
                     blaze: np.ndarray) -> np.ndarray:
    """
    Smooth weights going to zero at the edges of each order (and where the
    blaze is below BLAZE_THRESHOLD of its maximum), used by e2ds_to_s1d so
    that the s1d has no discontinuity from one order to the next. Each order
    only depends on its own data.

    :param params: ParamDict, parameter dictionary of constants
    :param e2ds: np.ndarray (2D), the E2DS (not blaze corrected)
    :param blaze: np.ndarray (2D), the blaze

    :return: np.ndarray (2D), the weights (same shape as blaze)
    """
    # get quantities from parameter dictionary of constants
    smooth_size = params['BLAZE_SMOOTH_SIZE']
    blazethres = params['BLAZE_THRESHOLD']
    # get size from e2ds
    nord, npix = e2ds.shape
    # define a kernel that goes from -3 to +3 smooth_sizes of the mask
    xker = np.arange(-smooth_size * 3, smooth_size * 3, 1)
    ker = np.exp(-0.5 * (xker / smooth_size) ** 2)
    # set up the edge vector
    edges = np.ones(npix, dtype=bool)
    # set edges of the image to 0 so that  we get a sloping weight
    edges[:int(3 * smooth_size)] = False
    edges[-int(3 * smooth_size):] = False
    # define the weighting for the edges (slopevector)
    slopevector = np.zeros_like(blaze)
    # for each order find the sloping weight vector
    for order_num in range(nord):
        # get the blaze for this order
        oblaze = np.array(blaze[order_num])
        # find the valid pixels
        cond1 = np.isfinite(oblaze) & np.isfinite(e2ds[order_num])
        with warnings.catch_warnings(record=True) as _:
            cond2 = oblaze > (blazethres * mp.nanmax(oblaze))
        valid = cond1 & cond2 & edges
        # convolve with the edge kernel
        oweight = np.convolve(valid, ker, mode='same')
        # normalise to the maximum
        with warnings.catch_warnings(record=True) as _:
            oweight = oweight - mp.nanmin(oweight)
            oweight = oweight / mp.nanmax(oweight)
        # append to sloping vector storage
        slopevector[order_num] = oweight
    return slopevector


def _s1d_order(order_num: int, wavemap: np.ndarray, se2ds: np.ndarray,
               sblaze: np.ndarray, wavegrid: np.ndarray,
               domain: Optional[Tuple[int, int]] = None):
    """
    Contribution of one order to the s1d of e2ds_to_s1d: the pixels of the
    s1d it covers and the values it adds to the flux and to the weight.

    :param order_num: int, the order
    :param wavemap: np.ndarray (2D), the wave map
    :param se2ds: np.ndarray (2D), e2ds times the slope vector
    :param sblaze: np.ndarray (2D), blaze times the slope vector
    :param wavegrid: np.ndarray (1D), the s1d wave grid (increasing)
    :param domain: optional (start, end) index range of wavegrid: if the
                   order does not add anything there, it is not computed

    :return: None if the order adds nothing, else (start, end, mask, flux,
             weight): wavegrid[start:end][mask] are the s1d points it adds
             flux and weight to
    """
    # get wavelength mask - if there are NaNs in wavemap have to deal with
    #    them (happens at least for polar)
    wavemask = np.isfinite(wavemap[order_num])
    # identify the valid pixels
    valid = np.isfinite(se2ds[order_num]) & np.isfinite(sblaze[order_num])
    valid &= wavemask
    # check that we have at least 5 valid points
    if np.sum(valid) < 5:
        return None
    # get this orders vectors
    owave = wavemap[order_num]
    oe2ds = se2ds[order_num, valid]
    oblaze = sblaze[order_num, valid]
    # check that all points for this order are zero
    if np.sum(owave == 0) != 0:
        # log message about skipping this order
        msg = ('\tOrder {0}: Some points in wavelength '
               'grid are zero. Skipping order.')
        log.info(msg.format(order_num))
        # skip this order
        return None
    # check that the grid increases or decreases in a monotonic way
    diffwave = np.diff(owave[valid])
    # check the signs of wave map gradient
    if np.sign(np.min(diffwave)) != np.sign(np.max(diffwave)):
        msg = ('\tOrder {0}: Wavelength grid curves around. '
               'Skipping order')
        log.info(msg.format(order_num))
        return None
    # can only spline in domain of the wave: wavegrid being increasing,
    #   wavegrid > min & wavegrid < max is the index range start:end
    start = np.searchsorted(wavegrid, mp.nanmin(owave[valid]), side='right')
    end = np.searchsorted(wavegrid, mp.nanmax(owave[valid]), side='left')
    if domain is not None:
        if end <= domain[0] or start >= domain[1]:
            return None
    # create the splines for this order
    spline_sp = mp.iuv_spline(owave[valid], oe2ds, k=5, ext=1)
    spline_bl = mp.iuv_spline(owave[valid], oblaze, k=1, ext=1)
    # valid must be cast as float for splining
    valid_float = valid.astype(float)
    # we mask pixels that are neighbours to a NaN.
    valid_float = np.convolve(valid_float, np.ones(3) / 3.0, mode='same')
    spline_valid = mp.iuv_spline(owave[wavemask], valid_float[wavemask],
                                 k=1, ext=1)
    # finding pixels where we have immediate neighbours that are
    #   considered valid in the spline (to avoid interpolating over large
    #   gaps in validity)
    wave_range = wavegrid[start:end]
    mask = spline_valid(wave_range) > 0.9
    wave_use = wave_range[mask]
    # get splines and add to outputs
    return start, end, mask, spline_sp(wave_use), spline_bl(wave_use)


def e2ds_to_s1d(params: ParamDict, wavemap: np.ndarray, e2ds: np.ndarray,
                blaze: np.ndarray, wavegrid: np.ndarray,
                parity: bool = False,
                domain: Optional[Tuple[int, int]] = None):
    """
    E2DS to S1D function (taken from apero - with adjustments)

    With parity=True, also returns the s1d of the odd orders (1, 3, ...)
    and of the even orders (0, 2, ...), i.e. the same as
    e2ds_to_s1d(params, wavemap[1::2], e2ds[1::2], blaze[1::2], wavegrid)
    and e2ds_to_s1d(params, wavemap[::2], ...), computing each order once.

    :param params: ParamDict, parameter dictionary of constants
    :param wavemap: np.ndarray (2D), the wave map for the E2DS
    :param e2ds: np.ndarray (2D), the E2DS 2D numpy array (norders x npixels)
                 must not be blaze corrected
    :param blaze: np.ndarray (2D), the blaze function of the E2DS - used for
                  weighting orders
    :param wavegrid: np.ndarray (1D), the output s1d wave grid (increasing)
    :param parity: bool, if True also return the odd and even order s1ds
    :param domain: optional (start, end) index range of wavegrid: only the
                   s1d between these indices is needed (the orders that
                   add nothing there are skipped, the rest of the s1d is
                   then incomplete)

    :return: tuple, 1. np.array (1D) the s1d flux, 2. np.array (1D) the weight
             assigned to each order (and the same for the odd and the even
             orders if parity=True)
    """
    # get size from e2ds
    nord, npix = e2ds.shape
    # -------------------------------------------------------------------------
    # define a smooth transition mask at the edges of the image
    # this ensures that the s1d has no discontinuity when going from one order
    # to the next.
    slopevector = _s1d_slopevector(params, e2ds, blaze)
    # multiple the spectrum and blaze by the sloping vector
    sblaze = np.array(blaze) * slopevector
    se2ds = np.array(e2ds) * slopevector
    # -------------------------------------------------------------------------
    # Perform a weighted mean of overlapping orders
    # by performing a spline of both the blaze and the spectrum
    # -------------------------------------------------------------------------
    # outputs: all orders, odd orders, even orders
    nout = 3 if parity else 1
    out_specs = [np.zeros_like(wavegrid) for _ in range(nout)]
    weights = [np.zeros_like(wavegrid) for _ in range(nout)]
    # loop around all orders
    for order_num in range(nord):
        contrib = _s1d_order(order_num, wavemap, se2ds, sblaze, wavegrid,
                             domain=domain)
        if contrib is None:
            continue
        start, end, mask, flux, weight = contrib
        # the outputs this order adds to (in order, as separate calls do)
        if not parity:
            outs = [0]
        elif order_num % 2 == 1:
            outs = [0, 1]
        else:
            outs = [0, 2]
        for iout in outs:
            weights[iout][start:end][mask] += weight
            out_specs[iout][start:end][mask] += flux
    # where out_spec is exactly zero set to NaN
    for iout in range(nout):
        out_specs[iout][out_specs[iout] == 0] = np.nan
    # return properties
    if not parity:
        return out_specs[0], weights[0]
    return (out_specs[0], weights[0], out_specs[1], weights[1],
            out_specs[2], weights[2])


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

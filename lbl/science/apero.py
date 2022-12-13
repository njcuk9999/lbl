#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-11-01

@author: cook
"""
import warnings
from typing import Tuple

import numpy as np

from lbl.core import base
from lbl.core import base_classes
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
log = base_classes.log


# =============================================================================
# Define functions
# =============================================================================
def e2ds_to_s1d(params: ParamDict, wavemap: np.ndarray, e2ds: np.ndarray,
                blaze: np.ndarray, wavegrid: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    E2DS to S1D function (taken from apero - with adjustments)

    :param params: ParamDict, parameter dictionary of constants
    :param wavemap: np.ndarray (2D), the wave map for the E2DS
    :param e2ds: np.ndarray (2D), the E2DS 2D numpy array (norders x npixels)
                 must not be blaze corrected
    :param blaze: np.ndarray (2D), the blaze function of the E2DS - used for
                  weighting orders
    :param wavegrid: np.ndarray (1D), the output s1d wave grid

    :return: tuple, 1. np.array (1D) the s1d flux, 2. np.array (1D) the weight
             assigned to each order
    """
    # get quantities from parameter dictionary of constants
    smooth_size = params['BLAZE_SMOOTH_SIZE']
    blazethres = params['BLAZE_THRESHOLD']
    # -------------------------------------------------------------------------
    # get size from e2ds
    nord, npix = e2ds.shape
    # -------------------------------------------------------------------------
    # define a smooth transition mask at the edges of the image
    # this ensures that the s1d has no discontinuity when going from one order
    # to the next. We define a scale for this mask
    # smoothing scale
    # -------------------------------------------------------------------------
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

    # multiple the spectrum and blaze by the sloping vector
    sblaze = np.array(blaze) * slopevector
    se2ds = np.array(e2ds) * slopevector
    # -------------------------------------------------------------------------
    # Perform a weighted mean of overlapping orders
    # by performing a spline of both the blaze and the spectrum
    # -------------------------------------------------------------------------
    out_spec = np.zeros_like(wavegrid)
    # out_spec_err = np.zeros_like(wavegrid)
    weight = np.zeros_like(wavegrid)
    # loop around all orders
    for order_num in range(nord):
        # get wavelength mask - if there are NaNs in wavemap have to deal with
        #    them (happens at least for polar)
        wavemask = np.isfinite(wavemap[order_num])
        # identify the valid pixels
        valid = np.isfinite(se2ds[order_num]) & np.isfinite(sblaze[order_num])
        valid &= wavemask
        # if we have no valid points we need to skip
        if np.sum(valid) == 0:
            continue
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
            continue
        # check that the grid increases or decreases in a monotonic way
        gradwave = np.gradient(owave)
        # check the signs of wave map gradient
        if np.sign(np.min(gradwave)) != np.sign(np.max(gradwave)):
            msg = ('\tOrder {0}: Wavelength grid curves around. '
                   'Skipping order')
            log.info(msg.format(order_num))
            continue
        # create the splines for this order
        spline_sp = mp.iuv_spline(owave[valid], oe2ds, k=5, ext=1)
        spline_bl = mp.iuv_spline(owave[valid], oblaze, k=1, ext=1)
        # valid must be cast as float for splining
        valid_float = valid.astype(float)
        # we mask pixels that are neighbours to a NaN.
        valid_float = np.convolve(valid_float, np.ones(3) / 3.0, mode='same')
        spline_valid = mp.iuv_spline(owave[wavemask], valid_float[wavemask],
                                     k=1, ext=1)
        # can only spline in domain of the wave
        useful_range = (wavegrid > mp.nanmin(owave[valid]))
        useful_range &= (wavegrid < mp.nanmax(owave[valid]))
        # finding pixels where we have immediate neighbours that are
        #   considered valid in the spline (to avoid interpolating over large
        #   gaps in validity)
        maskvalid = np.zeros_like(wavegrid, dtype=bool)
        maskvalid[useful_range] = spline_valid(wavegrid[useful_range]) > 0.9
        useful_range &= maskvalid
        # get splines and add to outputs
        weight[useful_range] += spline_bl(wavegrid[useful_range])
        out_spec[useful_range] += spline_sp(wavegrid[useful_range])

    # where out_spec is exactly zero set to NaN
    out_spec[out_spec == 0] = np.nan
    # return properties
    return out_spec, weight


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

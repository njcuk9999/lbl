#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-16

@author: cook
"""
import copy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUVSpline
from typing import Tuple, Union

# try to import bottleneck module
# noinspection PyBroadException
try:
    import bottleneck as bn

    HAS_BOTTLENECK = True
except Exception as e:
    HAS_BOTTLENECK = False
# try to import numba module
# noinspection PyBroadException
try:
    from numba import jit

    HAS_NUMBA = True
except Exception as _:
    jit = None
    HAS_NUMBA = False

from astropy.io import fits
from astropy import constants
from astropy.table import Table
import numpy as np
from typing import Dict, List, Tuple, Union

from lbl.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.math.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get speed of light
speed_of_light_ms = constants.c.value


# =============================================================================
# Define nan functions
# =============================================================================
def nanargmax(a: Union[list, np.ndarray],
              axis: Union[None, int, Tuple[int]] = None
              ) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanargmax depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.

    :type a: np.ndarray
    :type axis: int

    :return: the argument maximum of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK:
        # return bottleneck function
        return bn.nanargmax(a, axis=axis)
    else:
        # return numpy function
        return np.nanargmax(a, axis=axis)


def nanargmin(a: Union[list, np.ndarray],
              axis: Union[None, int, Tuple[int]] = None
              ) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanargmin depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.

    :type a: np.ndarray
    :type axis: int

    :return: the argument minimum of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK:
        # return bottleneck function
        return bn.nanargmin(a, axis=axis)
    else:
        # return numpy function
        return np.nanargmin(a, axis=axis)


def nanmax(a: Union[list, np.ndarray],
           axis: Union[None, int, Tuple[int]] = None,
           **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanmax depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the maximum of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.nanmax(a, axis=axis)
    else:
        # return numpy function
        return np.nanmax(a, axis=axis, **kwargs)


def nanmin(a: Union[list, np.ndarray],
           axis: Union[None, int, Tuple[int]] = None,
           **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanmin depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the minimum of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.nanmin(a, axis=axis)
    else:
        # return numpy function
        return np.nanmin(a, axis=axis, **kwargs)


def nanmean(a: Union[list, np.ndarray],
            axis: Union[None, int, Tuple[int]] = None,
            **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanmean depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the mean of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.nanmean(a, axis=axis)
    else:
        # return numpy function
        return np.nanmean(a, axis=axis, **kwargs)


def nanmedian(a: Union[list, np.ndarray],
              axis: Union[None, int, Tuple[int]] = None,
              **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanmedian depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the median of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.nanmedian(a, axis=axis)
    else:
        # return numpy function
        return np.nanmedian(a, axis=axis, **kwargs)


def nanstd(a: Union[list, np.ndarray],
           axis: Union[None, int, Tuple[int]] = None, ddof: int = 0,
           **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nanstd depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param ddof: int, optional. Means Delta Degrees of Freedom. The divisor
                 used in calculations is ``N - ddof``, where ``N`` represents
                 the number of non-NaN elements. By default `ddof` is zero.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int
    :type ddof: int

    :return: the standard deviation of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.nanstd(a, axis=axis, ddof=ddof)
    else:
        # return numpy function
        return np.nanstd(a, axis=axis, ddof=ddof, **kwargs)


def nansum(a: Union[list, np.ndarray],
           axis: Union[None, int, Tuple[int]] = None,
           **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of nansum depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the sum of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # make sure a is an array
        a1 = np.array(a)
        # bottle neck return in type given for bool array this is not
        #  what we want
        if a1.dtype == bool:
            a1 = a1.astype(int)
        # return bottleneck function
        return bn.nansum(a1, axis=axis)
    else:
        # return numpy function
        return np.nansum(a, axis=axis, **kwargs)


def median(a: Union[list, np.ndarray],
           axis: Union[None, int, Tuple[int]] = None,
           **kwargs) -> Union[int, float, np.ndarray]:
    """
    Bottleneck or numpy implementation of median depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param axis: {int, None}, optional, Axis along which the median is computed.
                 The default (axis=None) is to compute the median of the
                 flattened array.
    :param kwargs: keyword arguments passed to numpy function only

    :type a: np.ndarray
    :type axis: int

    :return: the median of array `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK and len(kwargs) == 0:
        # return bottleneck function
        return bn.median(a, axis=axis)
    else:
        # return numpy function
        return np.median(a, axis=axis, **kwargs)


# =============================================================================
# Define general math functions
# =============================================================================
class NanSpline:
    def __init__(self, emsg: str, x: Union[np.ndarray, None] = None,
                 y: Union[np.ndarray, None] = None, **kwargs):
        """
        This is used in place of scipy.interpolateInterpolatedUnivariateSpline
        (Any spline following this will return all NaNs)

        :param emsg: str, the error that means we have to use the NanSpline
        """
        self.emsg = str(emsg)
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.kwargs = copy.deepcopy(kwargs)

    def __repr__(self) -> str:
        """
        String representation of the class

        :return: string representation
        """
        return 'NanSpline: \n' + self.emsg

    def __str__(self) -> str:
        """
        String representation of the class

        :return: string representation
        """
        return self.__repr__()

    def __call__(self, x: np.ndarray, nu: int = 0,
                 ext: Union[int, None] = None) -> np.ndarray:
        """
        Return a vector of NaNs (this means the spline failed due to
        less points

        This is used in place of scipy.interpolateInterpolatedUnivariateSpline

        Parameters
        ----------
        x : array_like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: x can be unordered but the
            evaluation is more efficient if x is (partially) ordered.
        nu  : int
            The order of derivative of the spline to compute.
        ext : int
            Controls the value returned for elements of ``x`` not in the
            interval defined by the knot sequence.

            * if ext=0 or 'extrapolate', return the extrapolated value.
            * if ext=1 or 'zeros', return 0
            * if ext=2 or 'raise', raise a ValueError
            * if ext=3 or 'const', return the boundary value.

            The default value is 0, passed from the initialization of
            UnivariateSpline.

        """
        return np.repeat(np.nan, len(x))


def iuv_spline(x: np.ndarray, y: np.ndarray, **kwargs
               ) -> Union[IUVSpline, NanSpline]:
    """
    Do an Interpolated Univariate Spline taking into account NaNs (with masks)

    (from scipy.interpolate import InterpolatedUnivariateSpline)

    :param x: the x values of the input to Interpolated Univariate Spline
    :param y: the y values of the input ot Interpolated Univariate Spline

    :param kwargs: passed to scipy.interpolate.InterpolatedUnivariateSpline
    :return: spline instance (from InterpolatedUnivariateSpline(x, y, **kwargs))
    """
    # copy x and y
    x, y = np.array(x), np.array(y)
    # find all NaN values
    nanmask = ~np.isfinite(y)
    # deal with dimensions error (on k)
    #   otherwise get   dfitpack.error: (m>k) failed for hidden m
    if kwargs.get('k', None) is not None:
        # deal with to few parameters in x
        if len(x[~nanmask]) < (kwargs['k'] + 1):
            # raise exception if len(x) is bad
            emsg = ('IUV Spline len(x) < k+1 '
                    '\n\tk={0}\n\tlen(x) = {1}'
                    '\n\tx={2}\n\ty={3}')
            eargs = [kwargs['k'], len(x), str(x)[:70], str(y)[:70]]
            # return a nan spline
            return NanSpline(emsg.format(*eargs), x=x, y=y, **kwargs)
        if len(y[~nanmask]) < (kwargs['k'] + 1):
            # raise exception if len(x) is bad
            emsg = ('IUV Spline len(y) < k+1 '
                    '\n\tk={0}\n\tlen(y) = {1}'
                    '\n\tx={2}\n\ty={3}')
            eargs = [kwargs['k'], len(y), str(x)[:70], str(y)[:70]]
            # return a nan spline
            return NanSpline(emsg.format(*eargs), x=x, y=y, **kwargs)

    if np.sum(~nanmask) < 2:
        y = np.repeat(np.nan, len(x))
    elif np.sum(~nanmask) == 0:
        y = np.repeat(np.nan, len(x))
    else:
        # replace all NaN's with linear interpolation
        badspline = IUVSpline(x[~nanmask], y[~nanmask], k=1, ext=1)
        y[nanmask] = badspline(x[nanmask])
    # return spline
    return IUVSpline(x, y, **kwargs)


def lowpassfilter(input_vect: np.ndarray, width: int = 101) -> np.ndarray:
    """
    Computes a low-pass filter of an input vector.

    This is done while properly handling NaN values, but at the same time
    being reasonably fast.

    Algorithm:

    provide an input vector of an arbitrary length and compute a running NaN
    median over a box of a given length (width value). The running median is
    NOT computed at every pixel but at steps of 1/4th of the width value.
    This provides a vector of points where the nan-median has been computed
    (ymed) and mean position along the input vector (xmed) of valid (non-NaN)
    pixels. This xmed/ymed combination is then used in a spline to recover a
    vector for all pixel positions within the input vector.

    When there are no valid pixel in a 'width' domain, the value is skipped
    in the creation of xmed and ymed, and the domain is splined over.

    :param input_vect: numpy 1D vector, vector to low pass
    :param width: int, width (box size) of the low pass filter

    :return:
    """
    # indices along input vector
    index = np.arange(len(input_vect))
    # placeholders for x and y position along vector
    xmed = []
    ymed = []
    # loop through the lenght of the input vector
    for it in np.arange(-width // 2, len(input_vect) + width // 2, width // 4):
        # if we are at the start or end of vector, we go 'off the edge' and
        # define a box that goes beyond it. It will lead to an effectively
        # smaller 'width' value, but will provide a consistent result at edges.
        low_bound = it
        high_bound = it + int(width)
        # deal with lower bounds out of bounds --> set to zero
        if low_bound < 0:
            low_bound = 0
        # deal with upper bounds out of bounds --> set to max
        if high_bound > (len(input_vect) - 1):
            high_bound = (len(input_vect) - 1)
        # get the pixel bounds
        pixval = index[low_bound:high_bound]
        # do not low pass if not enough points
        if len(pixval) < 3:
            continue
        # if no finite value, skip
        if np.max(np.isfinite(input_vect[pixval])) == 0:
            continue
        # mean position along vector and NaN median value of
        # points at those positions
        xmed.append(np.nanmean(pixval))
        ymed.append(np.nanmedian(input_vect[pixval]))
    # convert to arrays
    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)
    # we need at least 3 valid points to return a
    # low-passed vector.
    if len(xmed) < 3:
        return np.zeros_like(input_vect) + np.nan
    # low pass with a mean
    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2
    # splining the vector
    spline = InterpolatedUnivariateSpline(xmed, ymed, k=1, ext=3)
    lowpass = spline(np.arange(len(input_vect)))
    # return the low pass filtered input vector
    return lowpass


def doppler_shift(wavegrid: np.ndarray, velocity: float) -> np.ndarray:
    """
    Apply a doppler shift

    :param wave: wave grid to shift
    :param velocity: float, velocity expressed in m/s

    :return: np.ndarray, the updated wave grid
    """
    # relativistic calculation (1 - v/c)
    part1 = 1 - velocity / speed_of_light_ms
    # relativistic calculation (1 + v/c)
    part2 = 1 + velocity / speed_of_light_ms
    # return updated wave grid
    return wavegrid * np.sqrt(part1 / part2)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

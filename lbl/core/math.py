#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-16

@author: cook
"""
import copy
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from astropy import constants
from scipy import optimize
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline as IUVSpline
from scipy.special import erf

from lbl.core import base
from lbl.core import base_classes

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

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.math.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get speed of light
speed_of_light_ms = constants.c.value
speed_of_light = speed_of_light_ms / 1000.0


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
# Numba JIT functionality
# =============================================================================
# must catch if we do not have the jit decorator and define our own
if not HAS_NUMBA:
    def jit(**options):
        # don't use options but they are required to match jit definition
        _ = options

        # define decorator
        def decorator(func):
            # define wrapper
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # return wrapper
            return wrapper

        # return decorator
        return decorator


# Set "nopython" mode for best performance, equivalent to @nji
@jit(nopython=True)
def odd_ratio_mean(value: np.ndarray, error: np.ndarray,
                   odd_ratio: float = 2e-4, nmax: int = 10,
                   conv_cut=1e-2) -> Tuple[float, float]:
    """
    Provide values and corresponding errors and compute a weighted mean

    :param value: np.array (1D), value array
    :param error: np.array (1D), uncertainties for value array
    :param odd_ratio: float, the probability that the point is bad
                    Recommended value in Artigau et al. 2021 : f0 = 0.002
    :param nmax: int, maximum number of iterations to pass through
    :param conv_cut: float, the convergence cut criteria - how precise we have
                     to get

    :return: tuple, 1. the weighted mean, 2. the error on weighted mean
    """
    # deal with NaNs in value or error
    keep = np.isfinite(value) & np.isfinite(error)
    # deal with no finite values
    if np.sum(keep) == 0:
        return np.nan, np.nan
    # remove NaNs from arrays
    value, error = value[keep], error[keep]
    # work out some values to speed up loop
    error2 = error ** 2
    # placeholders for the "while" below
    guess_prev = np.inf
    # the 'guess' must be started as close as we possibly can to the actual
    # value. Starting beyond ~3 sigma (or whatever the odd_ratio implies)
    # would lead to the rejection of pretty much all points and would
    # completely mess the convergence of the loop
    guess = np.nanmedian(value)
    bulk_error = 1.0
    ite = 0
    # loop around until we do all required iterations
    while (np.abs(guess - guess_prev) / bulk_error > conv_cut) and (ite < nmax):
        # store the previous guess
        guess_prev = float(guess)
        # model points as gaussian weighted by likelihood of being a valid point
        # nearly but not exactly one for low-sigma values
        gfit = (1 - odd_ratio) * np.exp(-0.5 * ((value - guess) ** 2 / error2))
        # find the probability that a point is bad
        odd_bad = odd_ratio / (gfit + odd_ratio)
        # find the probability that a point is good
        odd_good = 1 - odd_bad
        # calculate the weights based on the probability of being good
        weights = odd_good / error2
        # update the guess based on the weights
        if np.sum(np.isfinite(weights)) == 0:
            guess = np.nan
        else:
            guess = np.nansum(value * weights) / np.nansum(weights)
            # work out the bulk error
            bulk_error = np.sqrt(1.0 / np.nansum(odd_good / error2))
        # keep track of the number of iterations
        ite += 1
    # return the guess and bulk error
    return guess, bulk_error


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


def lowpassfilter(input_vect: np.ndarray, width: int = 101,
                  k: int = 2) -> np.ndarray:
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
    :param k: int, order of the spline used (passed to IUVSpline)

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
        if np.sum(np.isfinite(input_vect[pixval])) < 3:
            continue
        # mean position along vector and NaN median value of
        # points at those positions
        xmed.append(nanmean(pixval))
        ymed.append(nanmedian(input_vect[pixval]))
    # convert to arrays
    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)
    # we need at least 3 valid points to return a
    # low-passed vector.
    if len(xmed) < 3:
        return np.full_like(input_vect, np.nan)
    # low pass with a mean
    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2
    # splining the vector
    spline = iuv_spline(xmed, ymed, k=k, ext=3)
    lowpass = spline(np.arange(len(input_vect)))
    # return the low pass filtered input vector
    return lowpass


def doppler_shift(wavegrid: np.ndarray, velocity: float) -> np.ndarray:
    """
    Apply a doppler shift

    :param wavegrid: wave grid to shift
    :param velocity: float, velocity expressed in m/s

    :return: np.ndarray, the updated wave grid
    """
    # relativistic calculation (1 - v/c)
    part1 = 1 - (velocity / speed_of_light_ms)
    # relativistic calculation (1 + v/c)
    part2 = 1 + (velocity / speed_of_light_ms)
    # return updated wave grid
    return wavegrid * np.sqrt(part1 / part2)


def gauss_function(x: Union[float, np.ndarray], a: float, x0: float,
                   sigma: float, dc: float) -> Union[float, np.ndarray]:
    """
    A standard 1D gaussian function (for fitting against)

    :param x: numpy array (1D), the x data points
    :param a: float, the amplitude
    :param x0: float, the mean position of the gaussian
    :param sigma: float, the standard deviation (FWHM) of the gaussian
    :param dc: float, the constant level below the gaussian

    :return gauss: numpy array (1D), size = len(x), the output gaussian
    """
    # return gauss function
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + dc


def gauss_fit_s(x: Union[float, np.ndarray], x0: float, sigma: float,
                a: float, zp: float, slope: float) -> Union[float, np.ndarray]:
    """
    Gaussian fit with a slope

    :param x: numpy array (1D), the x values for the gauss fit
    :param x0: float, the mean position
    :param sigma: float, the FWHM
    :param a: float, the amplitude
    :param zp: float, the dc level
    :param slope: float, the float (x-x0) * slope

    :return: np.ndarray - the gaussian value with slope correction
    """
    # calculate gaussian
    gauss = a * np.exp(-0.5 * (x - x0) ** 2 / (sigma ** 2)) + zp
    correction = (x - x0) * slope
    return gauss + correction


def gauss_fit_e(x: Union[float, np.ndarray], x0: float, fwhm: float,
                amp: float, ears: float, expo: float) -> Union[float, np.ndarray]:
    """
    Gaussian fit with a shape factor and ears

    :param x: numpy array (1D), the x values for the gauss fit
    :param x0: float, the mean position
    :param fwhm: float, the FWHM
    :param amp: float, the amplitude
    :param ears: float, the ears (wings of the gaussian)
    :param expo: float, the shape factor

    :return: np.ndarray - the gaussian value
    """
    # gaussian FWHM = 2*sqrt(2*ln(2))*sigma
    ew = fwhm/(2*(2*np.log(2))**(1/expo))

    g1 = amp * np.exp(-np.abs(x - x0) ** expo / ew ** expo)
    g2 = (amp / ears ** 2) * np.exp(-np.abs(x - x0) ** expo / (ew * 2) ** expo)

    return 1 - g1 + g2


def fwhm(sigma: Union[float, np.ndarray] = 1.0) -> Union[float, np.ndarray]:
    """
    Get the Full-width-half-maximum value from the sigma value (~2.3548)

    :param sigma: float, the sigma, default value is 1.0 (normalised gaussian)
    :return: 2 * sqrt(2 * log(2)) * sigma = 2.3548200450309493 * sigma
    """
    # return fwdm
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def normal_fraction(sigma: Union[float, np.ndarray] = 1.0
                    ) -> Union[float, np.ndarray]:
    """
    Return the expected fraction of population inside a range
    (Assuming data is normally distributed)

    :param sigma: the number of sigma away from the median to be
    :return:
    """
    # return error function
    return erf(sigma / np.sqrt(2.0))


def estimate_sigma(tmp: np.ndarray, sigma=1.0) -> float:
    """
    Return a robust estimate of N sigma away from the mean

    :param tmp: np.array (1D) - the data to estimate N sigma of
    :param sigma: float, the number of sigma to calculate for

    :return: the 1 sigma value
    """
    # make sure we don't have all nan values
    if np.sum(np.isfinite(tmp)) == 0:
        return np.nan
    # get formal definition of N sigma
    sig1 = normal_fraction(sigma)
    # get the 1 sigma as a percentile
    p1 = (1 - (1 - sig1) / 2) * 100
    # work out the lower and upper percentiles for 1 sigma
    upper = np.nanpercentile(tmp, p1)
    lower = np.nanpercentile(tmp, 100 - p1)
    # return the mean of these two bounds
    return (upper - lower) / 2.0


def curve_fit(*args, funcname: Union[str, None] = None, **kwargs):
    """
    Wrapper around curve_fit to catch a curve_fit error

    :param args: args passed to curve_fit
    :param funcname: str or None, if set this is the function that called
                     curve_fit
    :param kwargs: kwargs passed to curve_fit
    :return: return of curve_fit
    """
    # deal with no funcname defined
    if funcname is None:
        funcname = __NAME__ + '.curve_fit()'
    # try to run curve fit
    try:
        with warnings.catch_warnings(record=True) as _:
            return optimize.curve_fit(*args, **kwargs)
    # deal with exception
    except Exception as err:
        p0 = kwargs.get('p0', 'Not Set')
        x = kwargs.get('xdata', [])
        y = kwargs.get('ydata', [])
        f = kwargs.get('f', None)
        error = '{0}: {1}'.format(type(err), str(err))
        # get error message
        emsg = 'CurveFitException'
        emsg += '\n\tFunction = {0}'.format(funcname)
        emsg += '\n\tP0: {0}'.format(p0)
        emsg += '\n\t{0}'.format(error)
        # raise an LBL CurveFit Exception
        raise base_classes.LblCurveFitException(emsg, x=x, y=y, f=f, p0=p0,
                                                func=funcname, error=error)


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int,
                   nsigcut: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    A robust polyfit (iterating on the residuals) until nsigma is below the
    nsigcut threshold. Takes care of NaNs before fitting

    :param x: np.ndarray, the x array to pass to np.polyval
    :param y: np.ndarray, the y array to pass to np.polyval
    :param degree: int, the degree of polynomial fit passed to np.polyval
    :param nsigcut: float, the threshold sigma required to return result
    :return:
    """
    # set up mask
    keep = np.isfinite(y)
    # set the nsigmax to infinite
    nsigmax = np.inf
    # set the fit as unset at first
    fit = None
    # while sigma is greater than sigma cut keep fitting
    while nsigmax > nsigcut:
        # calculate the polynomial fit (of the non-NaNs)
        fit = np.polyfit(x[keep], y[keep], degree)
        # calculate the residuals of the polynomial fit
        res = y - np.polyval(fit, x)
        # work out the new sigma values
        sig = np.nanmedian(np.abs(res))
        if sig == 0:
            nsig = np.zeros_like(res)
            nsig[res != 0] = np.inf
        else:
            nsig = np.abs(res) / sig
        # work out the maximum sigma
        nsigmax = np.max(nsig[keep])
        # re-work out the keep criteria
        keep = nsig < nsigcut
    # return the fit and the mask of good values
    return np.array(fit), np.array(keep)


def medfilt_1d(a: Union[list, np.ndarray],
               window: Union[None, int] = None) -> np.ndarray:
    """
    Bottleneck or scipy.signal implementation of medfilt depending on imports

    :param a: numpy array, Input array. If `a` is not an array, a conversion
              is attempted.
    :param window: int, The number of elements in the moving window.

    :type a: np.ndarray
    :type window: int

    :return: the 1D median filtered array of `a` (int, float or np.ndarray)
    """
    # check bottleneck functionality
    if HAS_BOTTLENECK:
        # get half window size
        half_window = window // 2
        # need to shift
        a1 = np.append(a, [np.nan] * half_window)
        # median filter (via bottleneck function)
        y = bn.move_median(a1, window=window, min_count=half_window)
        # return shifted bottleneck function
        return y[half_window:]
    else:
        # return scipy function
        return signal.medfilt(a, kernel_size=window)


def air_index(wavelength: Union[np.ndarray, float], temp: float = 15.0,
              pressure: float = 760.0) -> Union[np.ndarray, float]:
    """
    transform the wavelength in vacuum on the air
    wavelength in nm, t=temperature in C, p=pressure in millibar

    From EA via Romain Allart (For HARPS specifically)

    :param wavelength: np.array - vector of wavelengths
    :param temp: float, air temperature in degrees C
    :param pressure: float, air pressure in millibar

    :return: depending on input a np.array of refractive indexes or a single
             refractive index
    """
    part1 = 1e-6 * pressure * (1 + (1.049 - 0.0157 * temp) * 1e-6 * pressure)
    part2 = 720.883 * (1 + 0.003661 * temp)
    part3 = 64.328 + 29498.1 / (146 - (1e3 / wavelength) ** 2)
    part4 = 255.4 / (41 - (1e3 / wavelength) ** 2)
    # get the refractive index
    n = ((part1 / part2) * (part3 + part4)) + 1

    return n


def val_cheby(coeffs: np.ndarray, xvector: Union[np.ndarray, int, float],
              domain: List[float]) -> Union[np.ndarray, int, float]:
    """
    Using the output of fit_cheby calculate the fit to x  (i.e. y(x))
    where y(x) = T0(x) + T1(x) + ... Tn(x)

    :param coeffs: output from fit_cheby
    :param xvector: x value for the y values with fit
    :param domain: domain to be transformed to -1 -- 1. This is important to
    keep the components orthogonal. For SPIRou orders, the default is 0--4088.
    You *must* use the same domain when getting values with fit_cheby
    :return: corresponding y values to the x inputs
    """
    # transform to a -1 to 1 domain
    domain_cheby = 2 * (xvector - domain[0]) / (domain[1] - domain[0]) - 1
    # fit values using the domain and coefficients
    yvector = np.polynomial.chebyshev.chebval(domain_cheby, coeffs)
    # return y vector
    return yvector


def rot_broad(wvl: np.ndarray, flux: np.ndarray, epsilon: float, vsini: float,
              eff_wvl: Optional[float] = None) -> np.ndarray:
    """
    **********************************************************************
    ***** THIS FUNCTION IS COPIED FROM PyAstronomy/pyasl/rotBroad.py *****
    ***** AND MODIFIED TO USE THE SAME CONVENTIONS AS THE LBL CODE.  *****
    ***** AND AVOID DEPENDENCIES ON OTHER PyAstronomy MODULES.       *****
    **********************************************************************

    Apply rotational broadening using a single broadening kernel.
    The effect of rotational broadening on the spectrum is
    wavelength dependent, because the Doppler shift depends
    on wavelength. This function neglects this dependence, which
    is weak if the wavelength range is not too large.
    .. note:: numpy.convolve is used to carry out the convolution
              and "mode = same" is used. Therefore, the output
              will be of the same size as the input, but it
              will show edge effects.
    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The flux
    epsilon : float
        Linear limb-darkening coefficient
    vsini : float
        Projected rotational velocity in km/s.
    eff_wvl : float, optional
        The wavelength at which the broadening
        kernel is evaluated. If not specified,
        the mean wavelength of the input will be
        used.
    Returns
    -------
    Broadened spectrum : array
        The rotationally broadened output spectrum.
    """
    # Wavelength binsize
    dwl = wvl[1] - wvl[0]
    # deal with no effective wavelength
    if eff_wvl is None:
        eff_wvl = np.mean(wvl)
    # The number of bins needed to create the broadening kernel
    binn_half = int(np.floor(((vsini / speed_of_light) * eff_wvl / dwl))) + 1
    gwvl = (np.arange(4*binn_half) - 2*binn_half) * dwl + eff_wvl
    # Create the broadening kernel
    dl = gwvl - eff_wvl
    # -------------------------------------------------------------------------
    # this bit is from _Gdl.gdl
    #    Calculates the broadening profile.
    # -------------------------------------------------------------------------
    # set vc
    vc = vsini / speed_of_light
    # set eps (make sure it is a float)
    eps = float(epsilon)
    # calculate the max vc
    dlmax = vc * eff_wvl
    # generate the c1 and c2 parameters
    c1 = 2 * (1 - eps) / (np.pi * dlmax * (1-eps/3))
    c2 = eps / (2 * dlmax * (1 - eps/3))
    # storage for the output
    bprof = np.zeros(len(dl))
    # Calculate the broadening profile
    xvec = dl / dlmax
    indi0 = np.where(np.abs(xvec) < 1.0)[0]
    bprof[indi0] = c1 * np.sqrt(1 - xvec[indi0]**2) + c2 * (1 - xvec[indi0]**2)
    # Correct the normalization for numeric accuracy
    # The integral of the function is normalized, however, especially in the
    # case of mild broadening (compared to the wavelength resolution), the
    # discrete  broadening profile may no longer be normalized, which leads to
    # a shift of the output spectrum, if not accounted for.
    bprof /= (np.sum(bprof) * dwl)
    # -------------------------------------------------------------------------
    # Remove the zero entries
    indi = np.where(bprof > 0.0)[0]
    bprof = bprof[indi]
    # -------------------------------------------------------------------------
    result = np.convolve(flux, bprof, mode="same") * dwl
    return result


def bin_by_time(longitude: float, time_value: Union[np.ndarray ,float],
                day_frac: float = 0) -> Union[np.ndarray, float]:
    """
    Bin a time by the local time of the site to a specific point in the day
    (by day_frac where 0 = midnight before observation, 0.5 = noon, and 1.0 =
    midnight after observation)

    :param params: ParamDict, the parameter dictionary of constants, containing
                   at least OBS_LONG, the longitude of the observatory in
                   degrees
    :param time_value: astropy.Time, the time to bin (in UTC)
    :param day_frac: float, the fraction of the day to bin to (0 = midnight
                     before observation, 0.5 = noon, and 1.0 = midnight after
    :return:
    """
    # calculate the bin_time for this site (as a fraction of a day)
    local_bin_time = ((-longitude + 360) / 360 + day_frac) % 1
    # get the binned time for time_value
    binned_time_value = np.round(time_value - local_bin_time).astype(int)
    # need local binned time
    local_binned_time_value = binned_time_value + local_bin_time
    # return the binned time
    return local_binned_time_value


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

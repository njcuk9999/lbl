#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General science functions and algorithms

Created on 2021-03-16

@author: cook
"""
from astropy.io import fits
from astropy import constants
from astropy import units as uu
from astropy.table import Table
import numpy as np
import os
from scipy import stats
from typing import Any, Dict, List, Tuple, Union
import warnings
import wget

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default
from lbl.instruments import select
from lbl.science import plot

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.general.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get time from base
Time = base.AstropyTime
TimeDelta = base.AstropyTimeDelta
# get classes
Instrument = default.Instrument
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
InstrumentsType = select.InstrumentsType
# get speed of light
speed_of_light_ms = constants.c.value
speed_of_light_kms = constants.c.value / 1000.0


# =============================================================================
# Define compute functions
# =============================================================================
def make_ref_dict(inst: InstrumentsType, reftable_file: str,
                  reftable_exists: bool, science_files: List[str],
                  mask_file: str) -> Dict[str, np.ndarray]:
    """
    Make the reference table dictionary

    :param inst: Instrument instance
    :param reftable_file: str, the ref table filename
    :param reftable_exists: bool, if True file exists and we load it
                            otherwise we create it
    :param science_files: list of absolute paths to science files
    :param mask_file: absolute path to mask file

    :return:
    """
    # get parameter dictionary of constants
    params = inst.params
    # storage for ref dictionary
    ref_dict = dict()
    # load the mask
    mask_table = inst.load_mask(mask_file)
    # -------------------------------------------------------------------------
    # deal with loading from file
    if reftable_exists:
        # load ref table from disk
        table = Table.read(reftable_file, format=params['REF_TABLE_FMT'])
        # copy columns
        ref_dict['ORDER'] = np.array(table['ORDER'])
        ref_dict['WAVE_START'] = np.array(table['WAVE_START'])
        ref_dict['WAVE_END'] = np.array(table['WAVE_END'])
        ref_dict['WEIGHT_LINE'] = np.array(table['WEIGHT_LINE'])
        ref_dict['XPIX'] = np.array(table['XPIX'])
        ref_dict['LINE_SNR'] = np.array(table['LINE_SNR'])
        ref_dict['LINE_DEPTH'] = np.array(table['LINE_DEPTH'])
        ref_dict['LOCAL_FLUX'] = np.array(table['LOCAL_FLUX'])
        # ratio of expected VS actual RMS in difference of model vs line
        ref_dict['RMSRATIO'] = np.array(table['RMSRATIO'])
        # effective number of pixels in line
        ref_dict['NPIXLINE'] = np.array(table['NPIXLINE'])
        # mean line position in pixel space
        ref_dict['MEANXPIX'] = np.array(table['MEANXPIX'])
        # blaze value compared to peak for that order
        ref_dict['MEANBLAZE'] = np.array(table['MEANBLAZE'])
        # amp continuum
        ref_dict['AMP_CONTINUUM'] = np.array(table['AMP_CONTINUUM'])
        # Considering the number of pixels, expected and actual RMS,
        #     this is the likelihood that the line is acually valid from a
        #     Chi2 test point of view
        ref_dict['CHI2'] = np.array(table['CHI2'])
        # probability of valid considering the chi2 CDF for the number of DOF
        ref_dict['CHI2_VALID_CDF'] = np.array(table['CHI2_VALID_CDF'])
        # close table
        del table
    # -------------------------------------------------------------------------
    # deal with creating table
    else:
        # load wave solution from first science file
        wavegrid = inst.get_wave_solution(science_files[0])
        # storage for vectors
        order, wave_start, wave_end, weight_line, xpix = [], [], [], [], []
        line_snr, line_depth, local_flux = [], [], []
        # loop around orders
        for order_num in range(wavegrid.shape[0]):
            # get the min max wavelengths for this order
            min_wave = np.min(wavegrid[order_num])
            max_wave = np.max(wavegrid[order_num])
            # build a mask for mask lines in this order
            good = mask_table['ll_mask_s'] > min_wave
            good &= mask_table['ll_mask_s'] < max_wave
            # if we have values then add to arrays
            if np.sum(good) > 0:
                # add an order flag
                order += list(np.repeat(order_num, np.sum(good) - 1))
                # get the wave starts
                wave_start += list(mask_table['ll_mask_s'][good][:-1])
                # get the wave ends
                wave_end += list(mask_table['ll_mask_s'][good][1:])
                # get the weights of the lines (only used to get systemic
                # velocity as a starting point)
                weight_line += list(mask_table['w_mask'][good][:-1])
                # spline x pixels using wave grid
                xgrid = np.arange(len(wavegrid[order_num]))
                xspline = mp.iuv_spline(wavegrid[order_num], xgrid)
                # get the x pixel vector for mask
                xpix += list(xspline(mask_table['ll_mask_s'][good][:-1]))
                # get the line snr
                line_snr += list(mask_table['line_snr'][good][:-1])
                line_depth += list(mask_table['depth'][good][:-1])
                local_flux += list(mask_table['local_flux'][good][:-1])
        # make xpix a numpy array
        xpix = np.array(xpix)
        # add to reference dictionary
        ref_dict['ORDER'] = np.array(order)
        ref_dict['WAVE_START'] = np.array(wave_start)
        ref_dict['WAVE_END'] = np.array(wave_end)
        ref_dict['WEIGHT_LINE'] = np.array(weight_line)
        ref_dict['XPIX'] = xpix
        ref_dict['LINE_SNR'] = np.array(line_snr)
        ref_dict['LINE_DEPTH'] = np.array(line_depth)
        ref_dict['LOCAL_FLUX'] = np.array(local_flux)
        # ratio of expected VS actual RMS in difference of model vs line
        ref_dict['RMSRATIO'] = np.zeros_like(xpix, dtype=float)
        # effective number of pixels in line
        ref_dict['NPIXLINE'] = np.zeros_like(xpix, dtype=int)
        # mean line position in pixel space
        ref_dict['MEANXPIX'] = np.zeros_like(xpix, dtype=float)
        # blaze value compared to peak for that order
        ref_dict['MEANBLAZE'] = np.zeros_like(xpix, dtype=float)
        # amp continuum
        ref_dict['AMP_CONTINUUM'] = np.zeros_like(xpix, dtype=float)
        # Considering the number of pixels, expected and actual RMS,
        #     this is the likelihood that the line is acually valid from a
        #     Chi2 test point of view
        ref_dict['CHI2'] = np.zeros_like(xpix, dtype=float)
        # probability of valid considering the chi2 CDF for the number of DOF
        ref_dict['CHI2_VALID_CDF'] = np.zeros_like(xpix, dtype=float)

        # ---------------------------------------------------------------------
        # convert ref_dict to table (for saving to disk
        ref_table = Table()
        for key in ref_dict.keys():
            ref_table[key] = np.array(ref_dict[key])
        # log writing
        log.general('Writing ref table {0}'.format(reftable_file))
        # write to file
        io.write_table(reftable_file, ref_table,
                       fmt=inst.params['REF_TABLE_FMT'])

    # -------------------------------------------------------------------------
    # return table (either loaded from file or constructed from mask +
    #               wave solution)
    return ref_dict


def get_velo_scale(wave_vector: np.ndarray, hp_width: float) -> int:
    """
    Calculate the velocity scale give a wave vector and a hp width

    :param wave_vector: np.ndarray, the wave vector
    :param hp_width: float, the hp width

    :return: int, the velocity scale in pixels
    """
    # work out the velocity scale
    dwave = np.gradient(wave_vector)
    dvelo = 1 / mp.nanmedian((wave_vector / dwave) / speed_of_light_ms)
    # velocity pixel scale (to nearest pixel)
    width = int(hp_width / dvelo)
    # make sure pixel width is odd
    if width % 2 == 0:
        width += 1
    # return  velocity scale
    return width


def spline_template(inst: InstrumentsType, template_file: str,
                    systemic_vel: float) -> Dict[str, mp.IUVSpline]:
    """
    Calculate all the template splines (for later use)

    :param inst: Instrument instance
    :param template_file: str, the absolute path to the template file
    :param systemic_vel: float, the systemic velocity
    :return:
    """
    # log that we are producing all template splines
    msg = 'Defining all the template splines required later'
    log.general(msg)
    # get the pixel hp_width [needs to be in m/s]
    hp_width = inst.params['HP_WIDTH'] * 1000
    # load the template
    template_table = inst.load_template(template_file)
    # get properties from template table
    twave = np.array(template_table['wavelength'])
    tflux = np.array(template_table['flux'])
    tflux0 = np.array(template_table['flux'])
    # -------------------------------------------------------------------------
    # work out the velocity scale
    width = get_velo_scale(twave, hp_width)
    # -------------------------------------------------------------------------
    # we high-pass on a scale of ~101 pixels in the e2ds
    tflux -= mp.lowpassfilter(tflux, width=width)
    # -------------------------------------------------------------------------
    # get the gradient of the log of the wave
    grad_log_wave = np.gradient(np.log(twave))
    # get the derivative of the flux
    dflux = np.gradient(tflux) / grad_log_wave / speed_of_light_ms
    # get the 2nd derivative of the flux
    d2flux = np.gradient(dflux) / grad_log_wave / speed_of_light_ms
    # get the 3rd derivative of the flux
    d3flux = np.gradient(d2flux) / grad_log_wave / speed_of_light_ms
    # -------------------------------------------------------------------------
    # we create the spline of the template to be used everywhere later
    valid = np.isfinite(tflux) & np.isfinite(dflux)
    valid &= np.isfinite(d2flux) & np.isfinite(d3flux)
    # -------------------------------------------------------------------------
    # doppler shift wave grid with respect to systemic velocity
    ntwave = mp.doppler_shift(twave[valid], -systemic_vel)
    # -------------------------------------------------------------------------
    # storage for splines
    sps = dict()
    # template removed from its systemic velocity, the spline is redefined
    #   for good
    k_order = 5
    # flux, 1st and 2nd derivatives
    sps['spline0'] = mp.iuv_spline(ntwave, tflux0[valid], k=k_order, ext=1)
    sps['spline'] = mp.iuv_spline(ntwave, tflux[valid], k=k_order, ext=1)
    sps['dspline'] = mp.iuv_spline(ntwave, dflux[valid], k=k_order, ext=1)
    sps['d2spline'] = mp.iuv_spline(ntwave, d2flux[valid], k=k_order, ext=1)
    sps['d3spline'] = mp.iuv_spline(ntwave, d3flux[valid], k=k_order, ext=1)
    # -------------------------------------------------------------------------
    # we create a mask to know if the splined point  is valid
    tmask = np.isfinite(tflux).astype(float)
    ntwave1 = mp.doppler_shift(twave, -systemic_vel)
    sps['spline_mask'] = mp.iuv_spline(ntwave1, tmask, k=1, ext=1)
    # -------------------------------------------------------------------------
    # return splines
    return sps


def get_velocity_step(wave_vector: np.ndarray, rounding: bool = True) -> float:
    """
    Return the velocity step in m/s

    :param wave_vector: np.array, the wave grid
    :param rounding: bool, if True applies rounding (to the nearest 250 m/s)

    :return: float, the grid step in m/s
    """
    # get the gradient of the wave length solution
    # deal with 1D wave vector
    if len(wave_vector.shape) == 1:
        dwave = np.gradient(wave_vector)
    # deal with 2D wave vector
    else:
        dwave = np.gradient(wave_vector, axis=1)
    # get the velocity step [in km/s]
    velostep = mp.nanmedian(dwave / wave_vector) * speed_of_light_ms / 1e3
    # if we are not rounding then we return the full velocity step
    if not rounding:
        # return grid step in m/s
        return velostep * 1000
    # grid step in a convenient fraction of 1 km/s
    grid_step = 1e3 * np.floor(velostep * 2) / 4
    if grid_step == 0:
        grid_step = 250.0
    # return grid step in m/s
    return grid_step


def get_magic_grid(wave0: float, wave1: float, dv_grid: float = 500):
    """
    magic grid is a standard way of representing a wavelength vector it is set
    so that each element is exactly dv_grid step in velocity. If you shift
    your velocity, then you have a simple translation of this vector.

    :param wave0: first wavelength element
    :param wave1: second wavelength element
    :param dv_grid: grid size in m/s
    :return:
    """
    # default for the function is 500 m/s
    # the arithmetic is a but confusing here, you first find how many
    # elements you have on your grid, then pass it to an exponential
    # the first element is exactely wave0, the last element is NOT
    # exactly wave1, but is very close and is set to get your exact
    # step in velocity
    # get the length of the magic vector
    logwaveratio = np.log(wave1 / wave0)
    len_magic = int(np.ceil(logwaveratio * speed_of_light_ms / dv_grid))
    # get the positions for "magic length"
    plen_magic = np.arange(len_magic)
    # define the magic grid to use in ccf
    magic_grid = np.exp((plen_magic / len_magic) * logwaveratio) * wave0
    # return the magic grid
    return magic_grid


def rough_ccf_rv(inst: InstrumentsType, wavegrid: np.ndarray,
                 sci_data: np.ndarray, wave_mask: np.ndarray,
                 weight_line: np.ndarray, kind: str,
                 line_snr: np.ndarray) -> Tuple[float, float]:
    """
    Perform a rough CCF calculation of the science data

    :param inst: Instrument instance
    :param wavegrid: wave grid (shape shape as sci_data)
    :param sci_data: spectrum
    :param wave_mask: list of wavelength centers of mask lines
    :param weight_line: list of weights for each mask line
    :param kind: the kind of file we are doing the ccf on (for logging)
    :param line_snr: list of snr for each line

    :return: tuple, 1. systemic velocity estimate, 2. ccf ewidth
    """
    func_name = __NAME__ + '.rough_ccf_rv()'
    # -------------------------------------------------------------------------
    # get parameters
    rv_min = inst.params['ROUGH_CCF_MIN_RV']
    rv_max = inst.params['ROUGH_CCF_MAX_RV']
    rv_ewid_guess = inst.params['ROUGH_CCF_EWIDTH_GUESS']
    snr_min = inst.params['MASK_SNR_MIN']
    # -------------------------------------------------------------------------
    # if we have a 2D array make it 1D (but ignore overlapping regions)
    if wavegrid.shape[0] > 1:
        # 2D mask for making 2D --> 1D
        mask = np.ones_like(wavegrid, dtype=bool)
        # only include wavelengths for this order that don't overlap with
        #   the previous order
        for order_num in range(1, wavegrid.shape[0]):
            # elements where wavegrid doesn't overlap with previous
            porder = wavegrid[order_num] > wavegrid[order_num - 1, ::-1]
            # add to mask
            mask[order_num] &= porder
        # only include wavelengths for this order that don't overlap with the
        #   next order
        for order_num in range(0, wavegrid.shape[0] - 1):
            # elements where wavegrid doesn't overlap with next order
            norder = wavegrid[order_num] < wavegrid[order_num + 1, ::-1]
            # add to mask
            mask[order_num] &= norder
        # make sure no NaNs present
        mask &= np.isfinite(sci_data)
        # make the sci_data and wave grid are 1d
        wavegrid2 = wavegrid[mask]
        sci_data2 = sci_data[mask]
    # else we have a 1D array so just need to make sure no NaNs present
    else:
        # make sure no NaNs present
        mask = np.isfinite(sci_data)
        # apply mask to wavegrid and sci data
        wavegrid2 = wavegrid[mask]
        sci_data2 = sci_data[mask]
    # -------------------------------------------------------------------------
    # Make a magic grid to use in the CCF
    # -------------------------------------------------------------------------
    # spline the science data
    spline_sp = mp.iuv_spline(wavegrid2, sci_data2, k=1, ext=1)
    # min wavelength in domain
    wave0 = float(np.nanmin(wavegrid2))
    # maxwavelength in domain
    wave1 = float(np.nanmax(wavegrid2))
    # velocity step in m/s
    # med_rv = np.nanmedian(wavegrid2 / np.gradient(wavegrid2))
    # rv_step = speed_of_light_ms / med_rv
    # work out a valid velocity step in m/s
    grid_step = get_velocity_step(wavegrid2)
    # get the magic wave grid
    magic_grid = get_magic_grid(wave0, wave1, dv_grid=grid_step)
    # spline the magic grid
    magic_spline = spline_sp(magic_grid)
    # define a spline across the magic grid
    index_spline = mp.iuv_spline(magic_grid, np.arange(len(magic_grid)))
    # we find the position along the magic grid for the CCF lines
    index_mask = np.array(index_spline(wave_mask) + 0.5, dtype=int)

    # -------------------------------------------------------------------------
    # perform the CCF
    # -------------------------------------------------------------------------
    # define the steps from min to max (in rv steps) - in pixels
    istep = np.arange(int(rv_min / grid_step), int(rv_max / grid_step))
    # define the dv grid from the initial steps in pixels * rv step
    dvgrid = istep * grid_step
    # set up the CCF vector for all rv elements
    ccf_vector = np.zeros(len(istep))
    # define a mask that only keeps certain index values
    keep_line = index_mask > (rv_max / grid_step) + 2
    keep_line &= index_mask < len(magic_spline) - (rv_max / grid_step) - 2
    keep_line &= line_snr > snr_min
    # only keep the indices and weights within the keep line mask
    index_mask = index_mask[keep_line]
    weight_line = weight_line[keep_line]
    # now loop around the dv elements
    for dv_element in range(len(istep)):
        # get the ccf for each index (and multiply by weight of the line
        ccf_indices = magic_spline[index_mask - istep[dv_element]] * weight_line
        # ccf vector at this dv element is the sum of these ccf values
        ccf_vector[dv_element] = mp.nansum(ccf_indices)

    # high-pass the CCF just to be really sure that we are finding a true CCF
    # peak and not a spurious excursion in the low-frequencies
    rfit, _ = mp.robust_polyfit(dvgrid, ccf_vector, 2, 5)
    ccf_vector -= np.polyval(rfit, dvgrid)

    # -------------------------------------------------------------------------
    # fit the CCF
    # -------------------------------------------------------------------------
    # get the position of the maximum CCF
    ccfmax = np.argmax(ccf_vector)
    # guess the amplitude (minus the dc level)
    ccf_dc = mp.nanmedian(ccf_vector)
    ccf_amp = ccf_vector[ccfmax] - ccf_dc
    # construct a guess of a guassian fit to the CCF
    #    guess[0]: float, the mean position
    #    guess[1]: float, the ewidth
    #    guess[2]: float, the amplitude
    #    guess[3]: float, the dc level
    #    guess[4]: float, the float (x-x0) * slope
    guess = [dvgrid[ccfmax], rv_ewid_guess, ccf_amp, ccf_dc, 0.0]
    # set specific func name for curve fit errors
    sfuncname = '{0}.KIND={1}'.format(func_name, kind)
    # push into curve fit
    gcoeffs, pcov = mp.curve_fit(mp.gauss_fit_s, dvgrid, ccf_vector, p0=guess,
                                 funcname=sfuncname)
    # record the systemic velocity and the FWHM
    systemic_velocity = gcoeffs[0]
    ccf_ewidth = abs(gcoeffs[1])
    # fit ccf
    ccf_fit = mp.gauss_fit_s(dvgrid, *gcoeffs)
    # log ccf velocity
    msg = '\t\tCCF velocity ({0}) = {1:.2f} m/s'
    margs = [kind, -systemic_velocity]
    log.general(msg.format(*margs))
    msg = '\t\tCCF FWHM ({0}) = {1:.1f} m/s'
    margs = [kind, mp.fwhm() * ccf_ewidth]
    log.general(msg.format(*margs))
    # -------------------------------------------------------------------------
    # debug plot
    # -------------------------------------------------------------------------
    plot.compute_plot_ccf(inst, dvgrid, ccf_vector, ccf_fit, gcoeffs)
    # -------------------------------------------------------------------------
    # return the systemic velocity and the ewidth
    return systemic_velocity, ccf_ewidth


def get_scaling_ratio(spectrum1: np.ndarray,
                      spectrum2: np.ndarray) -> float:
    """
    Get the scaling ratio that minimizes least-square between two spectra
    with a

    :param spectrum1: np.ndarray, the first spectrum
    :param spectrum2: np.ndarray, the second spectrum

    :return: float, the scaling ratio between spectrum 1 and 2
    """
    # calculate the spectra squared (used a few times)
    spectrum1_2 = spectrum1 ** 2
    spectrum2_2 = spectrum2 ** 2

    # find good values (non NaN)
    good = np.isfinite(spectrum1) & np.isfinite(spectrum2)

    with warnings.catch_warnings(record=True) as _:
        # do not include points 5 sigma away from spectrum 1
        good &= np.abs(spectrum1) < mp.estimate_sigma(spectrum1) * 5
        # do not include points 5 sigma away from spectrum 2
        good &= np.abs(spectrum2) < mp.estimate_sigma(spectrum2) * 5

    # There may be no valid pixel in the order, if that's the case, we set
    # the amplitude to NaN
    if np.sum(good) == 0:
        return np.nan
    # first estimate of amplitude sqrt(ratio of squares)
    ratio = mp.nansum(spectrum1_2[good]) / mp.nansum(spectrum2_2[good])
    amp = np.sqrt(ratio)
    # loop around iteratively
    for iteration in range(5):
        # get residuals between spectrum 1 and spectrum 2 * amplitude
        residuals = spectrum1 - (amp * spectrum2)
        # get the sigma of the residuals
        sigma_res = mp.estimate_sigma(residuals)
        # re-calculate good mask
        good = np.isfinite(spectrum1) & np.isfinite(spectrum2)
        with warnings.catch_warnings(record=True) as _:
            good &= np.abs(residuals / sigma_res) < 3
        # calculate amp scale
        part1 = mp.nansum(residuals[good] * spectrum2[good])
        part2 = mp.nansum(spectrum1_2[good])
        part3 = mp.nansum(spectrum2_2[good])
        scale = part1 / np.sqrt(part2 * part3)
        # ratio this off the amplitude
        amp = amp / (1 - scale)
    # return the scaling ratio (amp)
    return amp


def estimate_noise_model(spectrum: np.ndarray, wavegrid: np.ndarray,
                         model: np.ndarray, hpwidth: float) -> np.ndarray:
    """
    Estimate the noise on spectrum given the model

    :param spectrum: np.ndarray, the spectrum
    :param wavegrid: np.ndarray, the wave grid for the spectrum
    :param model: np.ndarray, the model
    :param npoints: int, the number of points to spline across

    :return: np.ndarray, the rms vector for this spectrum give the model
    """
    # storage for output rms
    rms = np.zeros_like(spectrum)
    # loop around each order and estimate noise model
    for order_num in range(spectrum.shape[0]):
        # get the wavelength for this order
        waveord = wavegrid[order_num]
        # calculate the number of points for the sliding error rms
        npoints = get_velo_scale(waveord, hpwidth)
        # get the residuals between science and model
        residuals = spectrum[order_num] - model[order_num]
        # get the pixels along the model to spline at (box centers)
        indices = np.arange(0, model.shape[1], npoints//4)
        # store the sigmas
        sigma = np.zeros_like(indices, dtype=float)
        # loop around each pixel and work out sigma value
        for it in range(len(indices)):
            # get start and end values for this box
            istart = indices[it] - npoints//2
            iend = indices[it] + npoints//2
            # fix boundary problems
            if istart < 0:
                istart = 0
            if iend > model.shape[1]:
                iend = model.shape[1]
            # work out the sigma of this box
            sigma[it] = mp.estimate_sigma(residuals[istart: iend])
            # set any zero values to NaN
            sigma[sigma == 0] = np.nan
        # mask all NaN values
        good = np.isfinite(sigma)
        # if we have enough points calculate the rms
        if np.sum(good) > 2:
            # get the spline across all indices
            rms_spline = mp.iuv_spline(indices[good], sigma[good], k=1, ext=3)
            # apply the spline to the model positions
            rms[order_num] = rms_spline(np.arange(model.shape[1]))
        # else we don't have a noise model
        else:
            # we fill the rms with NaNs for each pixel
            rms[order_num] = np.full(model.shape[1], fill_value=np.nan)
    # return rms
    return rms


def bouchy_equation_line(vector: np.ndarray, diff_vector: np.ndarray,
                         mean_rms: np.ndarray) -> Tuple[float, float]:
    """
    Apply the Bouchy 2001 equation to a vector for the diff

    :param vector: np.ndarray, the vector
    :param diff_vector: np.ndarray, the difference between model and vector
                             i.e. diff = (vector - model) * weights
    :param mean_rms: np.ndarray, the mean rms for this line

    :return: tuple, 1. float, the Bouchy line value, 2. float, the rms of the
             Bouchy line value
    """
    with warnings.catch_warnings(record=True) as _:
        # work out the rms
        rms_pix = mean_rms / vector
        # work out the RV error
        rms_value = 1 / np.sqrt(np.sum(1 / rms_pix ** 2))
        # feed the line
        # nansum can break here - subtle: must be a sum
        #   nansum --> 0 / 0  [breaks]   sum --> nan / nan [works]
        value = np.sum(diff_vector * vector) / np.sum(vector ** 2)
    # return the value and rms of the value
    return value, rms_value


def compute_rv(inst: InstrumentsType, sci_iteration: int,
               sci_data: np.ndarray, sci_hdr: fits.Header,
               splines: Dict[str, Any], ref_table: Dict[str, Any],
               blaze: np.ndarray, systemic_all: np.ndarray,
               mjdate_all: np.ndarray, ccf_ewidth: Union[float, None] = None,
               reset_rv: bool = True, model_velocity: float = np.inf,
               science_file: str = '') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute the RV using a line-by-line analysis

    :param inst: Instrument instance
    :param sci_iteration: int, the science files iteration number
    :param sci_data: np.ndarray, the science data
    :param sci_hdr: fits.Header, the science header
    :param splines: Dict, the dictionary of template splines
    :param ref_table: Dict, the reference table
    :param blaze: np.ndarray, the blaze to correct the science data
    :param systemic_all: np.ndarray, the systemic velocity storage for all
                         science file (filled in function for this iteration)
    :param mjdate_all: np.ndarray, the mid exposure time in mjd for all
                       science files (filled in function for this iteration)
    :param ccf_ewidth: None or float, the ccf_ewidth to use (if set)
    :param reset_rv: bool, whether convergence failed in previous
                               iteration (first iteration always False)
    :param model_velocity: float, inf if not set but if set doesn't recalculate
                           model velocity
    :param science_file: str, the science file name (required for some
                         instruments)

    :return: tuple, 1. the reference table dict, 2. the output dictionary
    """
    # -------------------------------------------------------------------------
    # get parameters from inst.params
    # -------------------------------------------------------------------------
    # get the noise model switch
    use_noise_model = inst.params['USE_NOISE_MODEL']
    # get the pixel hp_width [needs to be in m/s]
    hp_width = inst.params['HP_WIDTH'] * 1000
    # get the object science name
    object_science = inst.params['OBJECT_SCIENCE']
    # get the mid exposure header key
    mid_exp_time_key = inst.params['KW_MID_EXP_TIME']
    # get the number of iterations to do to compute the rv
    compute_rv_n_iters = inst.params['COMPUTE_RV_N_ITERATIONS']
    # get plot order
    model_plot_orders = inst.params['COMPUTE_MODEL_PLOT_ORDERS']
    # get the minimum line width (in pixels) to consider line valid
    min_line_width = inst.params['COMPUTE_LINE_MIN_PIX_WIDTH']
    # get the threshold in sigma on nsig (dv / dvrms) to keep valid
    nsig_threshold = inst.params['COMPUTE_LINE_NSIG_THRES']
    # define the fraction of the bulk error the rv mean must be above
    #    for compute rv to have converged
    converge_thres = inst.params['COMPUTE_RV_BULK_ERROR_CONVERGENCE']
    # define the maximum number of iterations deemed to lead to a good RV
    max_good_num_iters = inst.params['COMPUTE_RV_MAX_N_GOOD_ITERS']
    # define the number of sigma to clip based on the rms away from the model
    #   (sigma clips science data)
    rms_sigclip_thres = inst.params['COMPUTE_RMS_SIGCLIP_THRES']
    # get readout noise
    readout_noise = inst.params['READ_OUT_NOISE']
    # -------------------------------------------------------------------------
    # deal with noise model
    if not use_noise_model:
        # TODO: get a noise model
        # rms = get_noise_model(science_files)
        rms = np.zeros_like(sci_data)
    else:
        rms = np.sqrt(np.abs(sci_data) + readout_noise ** 2)
    # -------------------------------------------------------------------------
    # copy science data
    sci_data0 = np.array(sci_data)
    # get the mid exposure time in MJD
    mjdate = io.get_hkey(sci_hdr, mid_exp_time_key)
    # -------------------------------------------------------------------------
    # get the wave grid for this science data
    # -------------------------------------------------------------------------
    # instrument specific wave solution --> use instrument method
    wavegrid = inst.get_wave_solution(data=sci_data, header=sci_hdr,
                                      science_filename=science_file)
    # loop around orders
    for order_num in range(sci_data.shape[0]):
        # work out the velocity scale
        width = get_velo_scale(wavegrid[order_num], hp_width)
        # we high-pass on a scale of ~101 pixels in the e2ds
        sci_data[order_num] -= mp.lowpassfilter(sci_data[order_num],
                                                width=width)
    # -------------------------------------------------------------------------
    # get BERV
    # -------------------------------------------------------------------------
    # instrument specific berv --> use instrument method
    berv = inst.get_berv(sci_hdr)

    # weighting of one everywhere
    ccf_weight = np.ones_like(ref_table['WEIGHT_LINE'], dtype=float)

    # -------------------------------------------------------------------------
    # Systemic velocity estimate
    # -------------------------------------------------------------------------
    # deal with first estimate of RV / CCF equivalent width
    if reset_rv:
        # if we are not using calibration file
        if not inst.flag_calib(sci_hdr):
            # calculate the rough CCF RV estimate
            sys_rv, ewidth = rough_ccf_rv(inst, wavegrid, sci_data,
                                          ref_table['WAVE_START'], ccf_weight,
                                          kind='science',
                                          line_snr=ref_table['LINE_SNR'])
            # if ccf width is not set then set it and log message
            if ccf_ewidth is None:
                ccf_ewidth = float(ewidth)
                # log ccf ewidth
                msg = '\t\tCCF e-width = {0:.2f} m/s'
                margs = [ccf_ewidth]
                log.general(msg.format(*margs))
        else:
            sys_rv, ccf_ewidth = 0, 0
    # for FP files
    else:
        # use the systemic velocity from closest date
        closest = np.argmin(mjdate - mjdate_all)
        # get the closest system rv to this observation
        sys_rv = systemic_all[closest] + berv
        # log the systemic velocity and the berv
        msg = '\tUsing Systemic velocity: {0:.4f} m/s  BERV: {1:.4f} m/s'
        margs = [-systemic_all[closest], -berv]
        log.general(msg.format(*margs))
        # log using the sys_Rv
        msg = '\tSystemic rv + berv={0:.4f} m/s from MJD={1}'
        margs = [-sys_rv, mjdate_all[closest]]
        log.general(msg.format(*margs))
    # -------------------------------------------------------------------------
    # iteration loop
    # -------------------------------------------------------------------------
    # a keep mask - for keep good mask lines
    mask_keep = np.ones_like(ref_table['ORDER'], dtype=bool)
    # store number of iterations required to converge
    num_to_converge = 0
    # set up models to spline onto
    model = np.zeros_like(sci_data)
    model0 = np.zeros_like(sci_data)
    dmodel = np.zeros_like(sci_data)
    d2model = np.zeros_like(sci_data)
    d3model = np.zeros_like(sci_data)
    # get the splines out of the spline dictionary
    spline0 = splines['spline0']
    spline = splines['spline']
    dspline = splines['dspline']
    d2spline = splines['d2spline']
    d3spline = splines['d3spline']
    spline_mask = splines['spline_mask']
    # set up storage for the dv, d2v, d3v and corresponding rms values
    #    fill with NaNs
    dv = np.full(len(ref_table['WAVE_START']), np.nan)
    d2v = np.full(len(ref_table['WAVE_START']), np.nan)
    d3v = np.full(len(ref_table['WAVE_START']), np.nan)
    sdv = np.full(len(ref_table['WAVE_START']), np.nan)
    sd2v = np.full(len(ref_table['WAVE_START']), np.nan)
    sd3v = np.full(len(ref_table['WAVE_START']), np.nan)
    # stoarge for final rv values
    rv_final = np.full(len(ref_table['WAVE_START']), np.nan)
    # storage for plotting
    plot_dict = dict()
    # get zero time
    zero_time = Time.now()
    # storage of rmsratio
    stddev_nsig = np.nan
    # loop around iterations
    for iteration in range(compute_rv_n_iters):
        # log iteration
        log.general('\t' + '-' * 50)
        log.general('\tIteration {0}'.format(iteration + 1))
        log.general('\t' + '-' * 50)
        # add to the number of iterations used to converge
        num_to_converge += 1
        # get start time
        start_time = Time.now()
        # ---------------------------------------------------------------------
        # update model, dmodel, d2model, d3model
        # ---------------------------------------------------------------------
        # get model offset
        if np.isfinite(model_velocity):
            model_offset = float(model_velocity)
        else:
            model_offset = 0
        # loop around each order and update model, dmodel, d2model, d3model
        for order_num in range(sci_data.shape[0]):
            # doppler shifted wave grid for this order
            shift = -sys_rv - model_offset
            wave_ord = mp.doppler_shift(wavegrid[order_num], shift)
            # get the blaze for this order
            blaze_ord = blaze[order_num]
            # get the low-frequency component out
            model_mask = np.ones_like(model[order_num])
            # add the spline mask values to model_mask (spline mask is 0 or 1)
            smask = spline_mask(wave_ord) < 0.99
            # set spline mask splined values to NaN
            model_mask[smask] = np.nan
            # RV shift the spline and correct for blaze and add model mask
            model[order_num] = spline(wave_ord) * blaze_ord * model_mask
            # work out the ratio between spectrum and model
            amp = get_scaling_ratio(sci_data[order_num], model[order_num])
            # apply this scaling ratio to the model
            model[order_num] = model[order_num] * amp
            # if this is the first iteration update model0
            if iteration == 0:
                # spline the original template and apply blaze
                model0[order_num] = spline0(wave_ord) * blaze_ord
                # get the median for the model and original spectrum
                med_model0 = mp.nanmedian(model0[order_num])
                med_sci_data_0 = mp.nanmedian(sci_data0[order_num])
                # normalize by the median
                with warnings.catch_warnings(record=True):
                    model0[order_num] = model0[order_num] / med_model0
                # multiply by the median of the original spectrum
                with warnings.catch_warnings(record=True):
                    model0[order_num] = model0[order_num] * med_sci_data_0
            # update the other splines
            dmodel[order_num] = dspline(wave_ord) * blaze_ord * amp
            d2model[order_num] = d2spline(wave_ord) * blaze_ord * amp
            d3model[order_num] = d3spline(wave_ord) * blaze_ord * amp
        # ---------------------------------------------------------------------
        # estimate rms
        # ---------------------------------------------------------------------
        # if we are not using a noise model - estimate the noise
        if not use_noise_model:
            rms = estimate_noise_model(sci_data, wavegrid, model, hp_width)
            # work out the number of sigma away from the model
            nsig = (sci_data - model) / rms
            # mask for nsigma
            sigmask = np.abs(nsig) > rms_sigclip_thres
            # apply sigma clip to the science data
            sci_data[sigmask] = np.nan
        else:
            for order_num in range(sci_data.shape[0]):
                # work out normalised residual
                tmp = (sci_data[order_num] - model[order_num]) / rms[order_num]
                # robust estimate of sigma value
                scale = mp.estimate_sigma(tmp)
                # update the rms based on the sigma scale value
                rms[order_num] *= scale
        # ---------------------------------------------------------------------
        # work out dv line-by-line
        # ---------------------------------------------------------------------
        # get orders
        orders = ref_table['ORDER']
        # keep track of which order we are looking at
        current_order = None
        # set these for use/update later
        nwavegrid = mp.doppler_shift(wavegrid, -sys_rv)
        # get splines between shifted wave grid and pixel grid
        wave2pixlist = []
        xpix = np.arange(model.shape[1])
        for order_num in range(wavegrid.shape[0]):
            wave2pixlist.append(mp.iuv_spline(nwavegrid[order_num], xpix))
        # ---------------------------------------------------------------------
        # debug plot dictionary for plotting later
        if iteration == 0:
            plot_dict['WAVEGRID'] = nwavegrid
            plot_dict['MODEL'] = model
            plot_dict['PLOT_ORDERS'] = model_plot_orders
            plot_dict['LINE_ORDERS'] = []
            plot_dict['WW_ORD_LINE'] = []
            plot_dict['SPEC_ORD_LINE'] = []
        # calculate the CCF for the model (if we don't have model_velocity)
        if not np.isfinite(model_velocity):
            rkwargs = dict(kind='model', line_snr=ref_table['LINE_SNR'])
            sys_model_rv, ewidth_model = rough_ccf_rv(inst, wavegrid, model,
                                                      ref_table['WAVE_START'],
                                                      ccf_weight, **rkwargs)
            model_velocity = -sys_model_rv + sys_rv
            log.general('\tModel velocity {:.2f} m/s'.format(model_velocity))
            # we don't want to continue this run if we have model_velocity
            continue
        # ---------------------------------------------------------------------
        # loop through all lines
        for line_it in range(0, len(orders)):
            # get the order number for this line
            order_num = orders[line_it]
            # -----------------------------------------------------------------
            # if line has been flagged as bad (in all but the first iteration)
            #   skip this line
            if (iteration != 1) and not (mask_keep[line_it]):
                continue
            # -----------------------------------------------------------------
            # if this is a new order the get residuals for this order
            if order_num != current_order:
                # update current order
                current_order = int(order_num)
            # get this orders values
            ww_ord = nwavegrid[order_num]
            sci_ord = sci_data[order_num]
            wave2pix = wave2pixlist[order_num]
            rms_ord = rms[order_num]
            model_ord = model[order_num]
            dmodel_ord = dmodel[order_num]
            d2model_ord = d2model[order_num]
            d3model_ord = d3model[order_num]
            blaze_ord = blaze[order_num]
            # -----------------------------------------------------------------
            # get the start and end wavelengths and pixels for this line
            wave_start = ref_table['WAVE_START'][line_it]
            wave_end = ref_table['WAVE_END'][line_it]
            x_start, x_end = wave2pix([wave_start, wave_end])
            # round pixel positions to nearest pixel
            x_start, x_end = int(np.floor(x_start)), int(np.ceil(x_end))
            # -----------------------------------------------------------------
            # boundary conditions
            if (x_end - x_start) < min_line_width:
                mask_keep[line_it] = False
                continue
            if x_start < 0:
                mask_keep[line_it] = False
                continue
            if x_end > len(ww_ord) - 2:
                mask_keep[line_it] = False
                continue
            # -----------------------------------------------------------------
            # get weights at the edge of the domain. Pixels inside have a
            # weight of 1, at the edge, it's proportional to the overlap
            weight_mask = np.ones(x_end - x_start + 1)
            # deal with overlapping pixels (before start)
            if ww_ord[x_start] < wave_start:
                refdiff = ww_ord[x_start + 1] - wave_start
                wavediff = ww_ord[x_start + 1] - ww_ord[x_start]
                weight_mask[0] = refdiff / wavediff
            # deal with overlapping pixels (after end)
            if ww_ord[x_end + 1] > wave_end:
                refdiff = ww_ord[x_end] - wave_end
                wavediff = ww_ord[x_end + 1] - ww_ord[x_end]
                weight_mask[-1] = 1 - (refdiff / wavediff)
            # get the x pixels
            xpix = np.arange(x_start, len(weight_mask) + x_start)
            # get mean xpix and mean blaze for line
            mean_xpix = mp.nansum(weight_mask * xpix) / mp.nansum(weight_mask)
            mean_blaze = blaze_ord[(x_start + x_end) // 2]
            # push mean xpix and mean blaze into ref table
            ref_table['MEANXPIX'][line_it] = mean_xpix
            ref_table['MEANBLAZE'][line_it] = mean_blaze
            # -----------------------------------------------------------------
            # add to the plots dictionary (for plotting later)
            if iteration == 1:
                plot_dict['LINE_ORDERS'] += [order_num]
                plot_dict['WW_ORD_LINE'] += [ww_ord[x_start:x_end + 1]]
                plot_dict['SPEC_ORD_LINE'] += [sci_ord[x_start:x_end + 1]]
            # -----------------------------------------------------------------
            # derivative of the segment
            d_seg = dmodel_ord[x_start: x_end + 1] * weight_mask
            # keep track of second and third derivatives
            d2_seg = d2model_ord[x_start: x_end + 1] * weight_mask
            d3_seg = d3model_ord[x_start: x_end + 1] * weight_mask
            # residual of the segment
            # TODO -> investigate data type problem
            # data type should work in the sum below. Does
            # not happen with SPIRou data
            sci_seg = sci_ord[x_start:x_end + 1]
            model_seg = model_ord[x_start:x_end + 1]
            # diff_seg = (sci_seg - model_seg) * weight_mask
            # work out the sum of the weights of the weight mask
            sum_weight_mask = np.sum(weight_mask)
            # TODO -> should be an option to subtraction or not the
            # TODO -> mean line flux
            denominator = np.nansum(model_seg ** 2 * weight_mask ** 2)

            if (sum_weight_mask != 0) and (denominator != 0):
                # subtract off normalized science sum
                scisum = mp.nansum(sci_seg * weight_mask)
                sci_seg = sci_seg - (scisum / sum_weight_mask)
                # subtract off normalized model sum
                modsum = mp.nansum(model_seg * weight_mask)
                model_seg = model_seg - (modsum / sum_weight_mask)

            diff_seg = (sci_seg - model_seg) * weight_mask
            # work out the sum of the rms
            sum_rms = np.sum(rms_ord[x_start: x_end + 1] * weight_mask)
            # work out the mean rms
            mean_rms = sum_rms / sum_weight_mask
            # -----------------------------------------------------------------
            # work out the 1st derivative
            #    From bouchy 2001 equation, RV error for each pixel
            # -----------------------------------------------------------------
            bout = bouchy_equation_line(d_seg, diff_seg, mean_rms)
            dv[line_it], sdv[line_it] = bout
            # -----------------------------------------------------------------
            # work out the 2nd derivative
            #    From bouchy 2001 equation, RV error for each pixel
            # -----------------------------------------------------------------
            bout = bouchy_equation_line(d2_seg, diff_seg, mean_rms)
            d2v[line_it], sd2v[line_it] = bout
            # -----------------------------------------------------------------
            # work out the 3rd derivative
            #    From bouchy 2001 equation, RV error for each pixel
            # -----------------------------------------------------------------
            bout = bouchy_equation_line(d3_seg, diff_seg, mean_rms)
            d3v[line_it], sd3v[line_it] = bout
            # -----------------------------------------------------------------
            # ratio of expected VS actual RMS in difference of model vs line
            ref_table['RMSRATIO'][line_it] = mp.nanstd(diff_seg) / mean_rms
            # effective number of pixels in line
            ref_table['NPIXLINE'][line_it] = len(diff_seg)
            # Considering the number of pixels, expected and actual RMS, this
            #   is the likelihood that the line is actually valid from chi2
            #   point of view
            ref_table['CHI2'][line_it] = mp.nansum((diff_seg / mean_rms) ** 2)
        # ---------------------------------------------------------------------
        # calculate the number of sigmas measured vs predicted

        # ---------------------------------------------------------------------
        # get the best etimate of the velocity and update sline
        rv_mean, bulk_error = mp.odd_ratio_mean(dv, sdv)

        # update rv_mean value with the response curve from template.
        #    An rv_mean value is always under-estimated in absolute value but
        #    the slope is 1 for absolute values close to zero.
        # TODO -> put these values in a look-up table somewhere!!!
        # ... no, not that smart after all
        # rv_mean = erfinv(rv_mean / 2250) * 2500
        nsig = (dv - rv_mean) / sdv
        # remove nans
        nsig = nsig[np.isfinite(nsig)]
        # remove sigma outliers
        nsig = nsig[np.abs(nsig) < nsig_threshold]
        # get the sigma of nsig
        stddev_nsig = mp.estimate_sigma(nsig)
        # log the value
        msg = '\t\tstdev_meas/stdev_pred = {0:.2f}'
        margs = [stddev_nsig]
        log.general(msg.format(*margs))
        # ---------------------------------------------------------------------
        # get final rv value
        rv_final = np.array(dv + sys_rv - berv)
        # add mean rv to sys_rv
        sys_rv = sys_rv + rv_mean
        # get end time
        end_time = Time.now()
        # get duration
        duration = (end_time - start_time).to(uu.s).value
        # log stats to screen
        msgs = []
        msgs += ['Iteration {0}: bulk error: {1:.2f} m/s rv = {2:.2f} m/s']
        msgs += ['Iteration duration: {3:.4f}']
        msgs += ['RV = {4:.2f} m/s, sigma = {1:.2f} m/s']
        margs = [iteration, bulk_error, -sys_rv, duration, rv_mean,
                 bulk_error]
        # loop around messages and add to log
        for msg in msgs:
            log.general('\t\t' + msg.format(*margs))
        # ---------------------------------------------------------------------
        # do a convergence check
        if np.abs(rv_mean) < (converge_thres * bulk_error):
            # break here
            break
    # -------------------------------------------------------------------------
    # line plot
    # -------------------------------------------------------------------------
    plot.compute_line_plot(inst, plot_dict)
    # -------------------------------------------------------------------------
    # update reference table
    # -------------------------------------------------------------------------
    # express to have sign fine relative to convention
    ref_table['dv'] = -rv_final
    ref_table['sdv'] = sdv
    # adding to the fits table the 2nd derivative projection
    ref_table['d2v'] = d2v
    ref_table['sd2v'] = sd2v
    # adding to the fits table the 3rd derivative projection
    ref_table['d3v'] = d3v
    ref_table['sd3v'] = sd3v
    # calculate the chi2 cdf
    chi2_cdf = 1 - stats.chi2.cdf(ref_table['CHI2'], ref_table['NPIXLINE'])
    ref_table['CHI2_VALID_CDF'] = chi2_cdf
    # -------------------------------------------------------------------------
    # update _all arrays
    # -------------------------------------------------------------------------
    # update the systemic velocity array
    systemic_all[sci_iteration] = sys_rv - berv
    # update the mjd date array
    mjdate_all[sci_iteration] = mjdate
    # end iterations
    log.general('\t' + '-' * 50)
    # -------------------------------------------------------------------------
    # Update convergence
    # -------------------------------------------------------------------------
    if num_to_converge >= max_good_num_iters:
        # flag that we need to take a completely new rv measurement
        reset_rv = True
        # log that rv did not converge
        wmsg = ('This RV is (probably) bad (iterations = {0}). '
                'Next step we will measure it with a CCF')
        wargs = [num_to_converge]
        log.warning(wmsg.format(*wargs))

    else:
        # make sure we are not taking a completely new rv measurement
        reset_rv = False
        # log that rv converged
        msg = 'Compute RV converged in {0} steps'
        margs = [num_to_converge]
        log.general(msg.format(*margs))
    # -------------------------------------------------------------------------
    # Log total time
    total_time = (Time.now() - zero_time).to(uu.s).value
    # -------------------------------------------------------------------------
    # save outputs to dictionary
    outputs = dict()
    outputs['SYSTEMIC_ALL'] = systemic_all
    outputs['MJDATE_ALL'] = mjdate_all
    outputs['RESET_RV'] = reset_rv
    outputs['NUM_ITERATIONS'] = num_to_converge
    outputs['SYSTEMIC_VELOCITY'] = sys_rv
    outputs['RMSRATIO'] = stddev_nsig
    outputs['CCF_EW'] = ccf_ewidth
    outputs['HP_WIDTH'] = hp_width
    outputs['TOTAL_DURATION'] = total_time
    outputs['MODEL_VELOCITY'] = model_velocity
    # -------------------------------------------------------------------------
    # return reference table and outputs
    return ref_table, outputs


def smart_timing(durations: List[float], left: int) -> Tuple[float, float, str]:
    """
    Calculate the mean time taken per iteration, the standard deviation in
    time taken per iteration and a time left string (smart)

    :param durations: List of floats, the durations we already have
    :param left: int, the number of iterations left

    :return: tuple, 1. the mean time of iterations, 2. the std of the iterations
             3. an estimate of the time left as a string (HH:MM:SS)
    """
    # deal with not enough stats to work out values
    if len(durations) < 2:
        return np.nan, np.nan, ''
    # work out the mean time
    mean_time = mp.nanmean(durations)
    # work out the std time
    std_time = mp.nanstd(durations)
    # get time delta
    timedelta = TimeDelta(mean_time * left * uu.s)
    # get in hh:mm:ss format
    time_left = str(timedelta.to_datetime())
    # return values
    return mean_time, std_time, time_left


# =============================================================================
# Define compil functions
# =============================================================================
def make_rdb_table(inst: InstrumentsType, rdbfile: str,
                   lblrvfiles: np.ndarray, plot_dir: str) -> Table:
    """
    Make the primary rdb table (row per observation)

    :param inst: Instrument instance
    :param rdbfile: str, the rdb file absolute path (only used to save plot)
    :param lblrvfiles: np.ndarray, array of strings, the absolute path to each
                       LBL RV file
    :param plot_dir: str, the absolute path to the directory to save plot
                     to (if plot commands are set to True)

    :return: astropy.table.Table, the RDB table (row per observation)
    """
    # set function name
    func_name = __NAME__ + '.make_rdb_table()'
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # -------------------------------------------------------------------------
    # get parameters
    # -------------------------------------------------------------------------
    # get limits
    wave_min = inst.params['COMPIL_WAVE_MIN']
    wave_max = inst.params['COMPIL_WAVE_MAX']
    max_pix_wid = inst.params['COMPIL_MAX_PIXEL_WIDTH']
    obj_sci = inst.params['OBJECT_SCIENCE']
    ccf_ew_fp = inst.params['COMPIL_FP_EWID']
    reference_wavelength = inst.params['COMPIL_SLOPE_REF_WAVE']
    # get the ccf e-width column name
    ccf_ew_col = inst.params['KW_CCF_EW']
    # get the header keys to add to rdb_table
    header_keys, fp_flags = inst.rdb_columns()
    # get base names
    lblrvbasenames = []
    for lblrvfile in lblrvfiles:
        lblrvbasenames.append(os.path.basename(lblrvfile))
    # -------------------------------------------------------------------------
    # output rdb column set up
    # -------------------------------------------------------------------------
    # storage for rdb table dictionary
    rdb_dict = dict()
    # add columns
    rdb_dict['rjd'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['vrad'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['svrad'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['d2v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['sd2v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['d3v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['sd3v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['local_file_name'] = np.array(lblrvbasenames)
    # time for matplotlib
    rdb_dict['plot_date'] = np.zeros_like(lblrvfiles, dtype=float)
    # dW from paper (equation 11)
    rdb_dict['dW'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['sdW'] = np.zeros_like(lblrvfiles, dtype=float)
    # for DACE, derived from d2v
    rdb_dict['fwhm'] = np.zeros_like(lblrvfiles, dtype=float)
    # for DACE, derived from d2v
    rdb_dict['sig_fwhm'] = np.zeros_like(lblrvfiles, dtype=float)
    # velocity at a reference wavelength fitting for the chromatic slope
    rdb_dict['vrad_achromatic'] = np.zeros_like(lblrvfiles, dtype=float)
    # error on achromatic velocity
    rdb_dict['svrad_achromatic'] = np.zeros_like(lblrvfiles, dtype=float)
    # chromatic slope
    rdb_dict['vrad_chromatic_slope'] = np.zeros_like(lblrvfiles, dtype=float)
    # error in chromatic slope
    rdb_dict['svrad_chromatic_slope'] = np.zeros_like(lblrvfiles, dtype=float)
    # get filename column
    rdb_dict['FILENAME'] = [[]] * len(lblrvfiles)
    # add header keys
    for hdr_key in header_keys:
        # empty elements in a list for each key to fill
        rdb_dict[hdr_key] = [[]] * len(lblrvfiles)

    # -------------------------------------------------------------------------
    # open first file to set up good mask
    # -------------------------------------------------------------------------
    # load table and header
    rvtable0, rvhdr0 = inst.load_lblrv_file(lblrvfiles[0])
    # flag a calibration file
    flag_calib = inst.flag_calib(rvhdr0)
    # do not consider lines below wave_min limit
    good = rvtable0['WAVE_START'] > wave_min
    # do not consider lines above wave_max limit
    good &= rvtable0['WAVE_START'] < wave_max
    # do not consider lines wider than max_pix_wid limit
    good &= rvtable0['NPIXLINE'] < max_pix_wid
    # remove good from rv table
    rvtable0 = rvtable0[good]
    # size of arrays
    nby, nbx = len(lblrvfiles), np.sum(good)
    # set up rv and dvrms
    rvs, dvrms = np.zeros([nby, nbx]), np.zeros([nby, nbx])

    # -------------------------------------------------------------------------
    # work out for cumulative plot for first file
    # -------------------------------------------------------------------------
    # median velocity for first file rv table
    med_velo = mp.nanmedian(rvtable0['dv'])
    # work out the best lines (less than 5 sigma)
    best_mask = rvtable0['sdv'] < np.nanpercentile(rvtable0['sdv'], 5.0)
    # list for plot storage
    vrange_all, pdf_all, pdf_fit_all = [], [], []

    # -------------------------------------------------------------------------
    # Loop around lbl rv files
    # -------------------------------------------------------------------------
    # log progress
    log.info('Producing LBL RDB 1 table')
    # loop around lbl rv files
    for row in tqdm(range(len(lblrvfiles))):
        # ---------------------------------------------------------------------
        # get lbl rv file table and header
        # ---------------------------------------------------------------------
        # load table and header
        rvtable, rvhdr = inst.load_lblrv_file(lblrvfiles[row])
        # fix header (instrument specific)
        rvhdr = inst.fix_lblrv_header(rvhdr)
        # fill rjd value
        rdb_dict['rjd'][row] = inst.get_rjd_value(rvhdr)
        # fill in plot date
        rdb_dict['plot_date'][row] = inst.get_plot_date(rvhdr)
        # ---------------------------------------------------------------------
        # fill in filename
        # ---------------------------------------------------------------------
        rdb_dict['FILENAME'][row] = os.path.basename(lblrvfiles[row])
        # ---------------------------------------------------------------------
        # fill in header keys
        # ---------------------------------------------------------------------
        for ikey, key in enumerate(header_keys):
            # deal with FP flags
            if obj_sci == 'FP' and fp_flags[ikey]:
                rdb_dict[key][row] = np.nan
            # if we have key add the value
            elif key in rvhdr:
                rdb_dict[key][row] = rvhdr[key]
            # else print a warning and add a NaN
            else:
                wmsg = 'Key {0} not present in file {1}'
                wargs = [key, lblrvfiles[row]]
                log.warning(wmsg.format(*wargs))
                rdb_dict[key][row] = np.nan
        # ---------------------------------------------------------------------
        # Read all lines for this file and load into arrays
        # ---------------------------------------------------------------------
        # if we don't have a calibration we set the rvs and dvrms from rv table
        if not flag_calib:
            rvs[row] = rvtable[good]['dv']
            dvrms[row] = rvtable[good]['sdv']
        # else we calcualte it using odd ratio mean
        else:
            cal_rv = np.array(rvtable[good]['dv'], dtype=float)
            cal_dvrms = np.array(rvtable[good]['sdv'], dtype=float)
            # estimate using odd ratio mean
            cal_guess, cal_bulk_error = mp.odd_ratio_mean(cal_rv, cal_dvrms)
            # push into rdb_dict
            rdb_dict['vrad'][row] = cal_guess
            rdb_dict['svrad'][row] = cal_bulk_error
        # get the d2v, sd2v, d3v and sd3v values from table
        d2v = np.array(rvtable[good]['d2v'], dtype=float)
        sd2v = np.array(rvtable[good]['sd2v'], dtype=float)
        d3v = np.array(rvtable[good]['d3v'], dtype=float)
        sd3v = np.array(rvtable[good]['sd3v'], dtype=float)
        # use the odd mean ratio to calculate d2v and sd2v
        d2v_guess, d2v_bulk_error = mp.odd_ratio_mean(d2v, sd2v)
        # push into rdb_dict
        rdb_dict['d2v'][row] = d2v_guess
        rdb_dict['sd2v'][row] = d2v_bulk_error
        # use the odd mean ratio to calculate d3v and sd3v
        d3v_guess, d3v_bulk_error = mp.odd_ratio_mean(d3v, sd3v)
        # push into rdb_dict
        rdb_dict['d3v'][row] = d3v_guess
        rdb_dict['sd3v'][row] = d3v_bulk_error
        # ---------------------------------------------------------------------
        # if we don't have a calibration add plot values
        if not flag_calib:
            # plot specific math
            xlim = [med_velo - 5000, med_velo + 5000]
            # get velocity range
            vrange = np.arange(xlim[0], xlim[1], 50.0)
            # storage for probability density function
            pdf = np.zeros_like(vrange, dtype=float)
            # mask the rv and dvrms by best_mask
            best_rv = rvs[row][best_mask]
            best_dvrms = dvrms[row][best_mask]
            # track finite values
            finite_mask = np.isfinite(best_rv) & np.isfinite(best_dvrms)
            # loop around each line
            for line_it in range(len(best_rv)):
                # only deal with finite masks
                if finite_mask[line_it]:
                    # get exponent
                    part = (vrange - best_rv[line_it]) / best_dvrms[line_it]
                    # calculate pdf weights
                    pdf_weight = np.exp(-0.5 * part ** 2)
                    pdf_weight = pdf_weight / best_dvrms[line_it]
                    # add to pdf for each vrange
                    pdf = pdf + pdf_weight
            # fit the probability density function
            guess = [med_velo, 500.0, np.max(pdf), 0.0, 0.0]
            # set specific func name for curve fit errors
            sfuncname = '{0}.RDB1-ROW[{1}]'.format(func_name, row)
            # try to compute curve fit
            try:
                pdf_coeffs, _ = mp.curve_fit(mp.gauss_fit_s, vrange, pdf,
                                             p0=guess, funcname=sfuncname)
                # fit pdf function
                pdf_fit = mp.gauss_fit_s(vrange, *pdf_coeffs)
            except base_classes.LblCurveFitException as e:
                wmsg = 'CurveFit exception - skipping file'
                wmsg += '\n\tFile = {0}'.format(lblrvfiles[row])
                wmsg += '\n\tP0 = {0}'.format(e.p0)
                wmsg += '\n\tFunction = {0}'.format(e.func)
                wmsg += '\n\tError: {0}'.format(e.error)
                log.warning(wmsg)
                # do not add this file
                continue
            # append values to plot lists
            vrange_all.append(vrange)
            pdf_all.append(pdf)
            pdf_fit_all.append(pdf_fit)
    # -------------------------------------------------------------------------
    # cumulative plot
    # -------------------------------------------------------------------------
    # plot if not a calibration
    if not flag_calib:
        # construct plot name
        plot_name = os.path.basename(rdbfile.replace('.rdb', '')) + '_cumul.pdf'
        plot_path = os.path.join(plot_dir, plot_name)
        # plot the cumulative plot
        plot.compil_cumulative_plot(inst, vrange_all, pdf_all,
                                    pdf_fit_all, plot_path)
    # ---------------------------------------------------------------------
    # First guess at vrad
    # ---------------------------------------------------------------------
    # if we don't have a calibration need to guess vrad
    if not flag_calib:
        # log progress
        msg = 'Forcing a stdev of 1 for all lines'
        log.general(msg)
        msg = 'Constructing a per-epoch mean velocity'
        log.general(msg)
        # loop around lines and generate a first guess for vrad and svrad
        for line_it in tqdm(range(rvs.shape[0])):
            # calculate the number of sigma away from median
            # diff = rvs[line_it] - mp.nanmedian(rvs[line_it])
            # nsig = diff / dvrms[line_it]
            # force a std dev of 1
            # TODO : have the normalization to sigma as an option
            # dvrms[line_it] = dvrms[line_it] #* mp.estimate_sigma(nsig)
            # use the odd ratio mean to guess vrad and svrad
            orout1 = mp.odd_ratio_mean(rvs[line_it], dvrms[line_it])
            # add to output table
            rdb_dict['vrad'][line_it] = orout1[0]
            rdb_dict['svrad'][line_it] = orout1[1]
        # ---------------------------------------------------------------------
        # Model per epoch
        # ---------------------------------------------------------------------
        # de-biasing line - matrix that contains a replicated 2d version
        #    of the per-epoch mean
        # rv_per_epoch_model = np.repeat(rdb_dict['vrad'], rvs.shape[1])
        # rv_per_epoch_model = rv_per_epoch_model.reshape(rvs.shape)
        # ---------------------------------------------------------------------
        # line-by-line mean position
        # ---------------------------------------------------------------------
        # storage for line mean / error
        per_line_mean = np.zeros(rvs.shape[1])
        per_line_error = np.zeros(rvs.shape[1])
        # log progress
        log.info('Producing line-by-line mean positions')
        # compute the per-line bias
        for line_it in tqdm(range(len(per_line_mean))):
            # get the difference and error for each line
            diff1 = rvs[:, line_it] - np.nanmedian(rvs[:, line_it])
            # here we avoid having a shallow copy
            err1 = np.array(dvrms[:, line_it])
            # try to guess the odd ratio mean
            # noinspection PyBroadException
            try:
                # corr_rms = mp.estimate_sigma(diff1 / err1)
                # if corr_rms<.2:
                #    print(corr_rms,line_it)
                # err1*=corr_rms # updating the per-line RMS, also updates dvrms[:,line_it]
                guess2, bulk_error2 = mp.odd_ratio_mean(diff1, err1)
                per_line_mean[line_it] = guess2
                per_line_error[line_it] = bulk_error2
            # if odd ratio mean fails push NaNs into arrays
            except Exception as _:
                per_line_mean[line_it] = np.nan
                per_line_mean[line_it] = np.nan

        # normalize the per-line mean to zero
        guess3, bulk_error3 = mp.odd_ratio_mean(per_line_mean, per_line_error)
        per_line_mean = per_line_mean - guess3

        # ---------------------------------------------------------------------
        # Model per line
        # ---------------------------------------------------------------------
        # construct a 2d model of line biases
        rv_per_line_model = np.tile(per_line_mean, rvs.shape[0])
        rv_per_line_model = rv_per_line_model.reshape(rvs.shape)
    else:
        rv_per_line_model = np.zeros(rvs.shape)

    # print progress
    log.info('Computing chromatic slope and per-bandpass statistics')
    # zero filled array
    lblrv_zeros = np.zeros_like(lblrvfiles, dtype=float)
    # ---------------------------------------------------------------------
    # Update table with vrad/svrad, per epoch values and fwhm/sig_fwhm
    # ---------------------------------------------------------------------
    for row in tqdm(range(len(lblrvfiles))):
        # if we have a calibration load the lbl rv file
        if flag_calib:
            rvtable, rvhdr = inst.load_lblrv_file(lblrvfiles[row])
            residuals = np.array(rvtable['dv'])
            # get the error
            err = np.array(rvtable['sdv'])
            rvs_row = residuals
        else:
            # get the residuals of the rvs to the rv per line model
            rvs_row = rvs[row]
            residuals = rvs[row] - rv_per_line_model[row]
            err = dvrms[row]
            # recompute the guess at the vrad / svrad
            guess4, bulk_error4 = mp.odd_ratio_mean(residuals, err)
            rdb_dict['vrad'][row] = guess4
            rdb_dict['svrad'][row] = bulk_error4

        # ---------------------------------------------------------------------
        # fit a slope to the rv
        # ---------------------------------------------------------------------
        # probability that the point is drawn from the distribution described
        #     by uncertainties. We assume that outliers have 1e-4 likelihood
        #     and a flat prior
        prob_good = np.ones_like(rvs_row)

        # keep track of velocity at previous step to see if fit converges
        prev_velo = np.inf
        # initializing values
        # velocity at reference wavelength m/s/um
        achromatic_velo, sig_achromatic_velo = np.nan, np.nan
        # chromatic slope of velocity
        chromatic_slope, sig_chromatic_slope = np.nan, np.nan
        # store count
        itr_count = 1
        # loop around 10 iterations (but break on convergence)
        while itr_count <= 10:
            # find finite points
            good = np.isfinite(err) & np.isfinite(rvs_row)
            # get valid wave and subtract reference wavelength
            valid_wave = rvtable0['WAVE_START'][good] - reference_wavelength
            # get valid rv an dv drms
            valid_rv = rvs_row[good]
            valid_dvrms = err[good]
            # get weights for the np.polyval fit. Do *not* use the square of
            # error bars for weight (see polyfit help)

            # TODO -> add a threshold for lines that have suspiciously small
            # TODO -> errors compared to the bulk of error bar values.
            # TODO -> only done if some errors are equal to zero.
            if np.min(valid_dvrms) == 0:
                threshold = np.nanpercentile(valid_dvrms, 1)
                suspicious = valid_dvrms < threshold
                valid_dvrms[suspicious] = threshold

            weight_lin_fit = prob_good[good] / valid_dvrms
            # fitting the lines with weights
            # noinspection PyTupleAssignmentBalance
            scoeffs, scov = np.polyfit(valid_wave, valid_rv, 1,
                                       w=weight_lin_fit, cov=True)
            # work out the fitted vector
            sfit = np.polyval(scoeffs, valid_wave)
            # work out how many sigma off we are from the fit
            nsig = (valid_rv - sfit) / valid_dvrms
            # work out the likelihood of the line being a valid point
            gpart = np.exp(-0.5 * nsig ** 2)
            prob_good[good] = (1 + 1.0e-4) * gpart / (1.0e-4 + gpart)
            # get the slope and velocity from the fit
            achromatic_velo = scoeffs[1]
            # expressed in m/s/um, hence the *1000 to go from nm (wave grid)
            chromatic_slope = scoeffs[0] * 1000
            # errors from covariance matrix
            sig_chromatic_slope = np.sqrt(scov[0, 0]) * 1000
            sig_achromatic_velo = np.sqrt(scov[1, 1])
            # see whether we need another iteration
            if np.abs(achromatic_velo - prev_velo) < 0.1 * sig_achromatic_velo:
                # if within 10% of errors break
                break
            # else we update the previous velocity
            else:
                prev_velo = float(achromatic_velo)
                # add to the count (for logging)
                itr_count += 1
        # ---------------------------------------------------------------------
        # deal with having a FP (CCF_EW_ROW doesn't really make sense)
        # get the ccf_ew
        # if an FP, then ccf_ew_row is zero and it is set to the default FP value
        # We do not check against the OBJECT name as some user may have more complex
        # naming schemes and/or stars may have FP within the string sequence
        # (ALFPER ... or who knows!)

        ccf_ew_row = rdb_dict[ccf_ew_col][row]
        # TODO -> make this smarter if we have LFC or HC
        if ccf_ew_row == 0:
            ccf_ew_row = float(ccf_ew_fp)

        # work out the fwhm (1 sigma * sigma value)
        d2v = rdb_dict['d2v'][row]
        sd2v = rdb_dict['sd2v'][row]
        # ---------------------------------------------------------------------
        # this is the dW in the paper
        #   see equation 11 in paper dW = FWHM * dFWHM
        dw = d2v * 8 * np.log(2)
        sdw = sd2v * 8 * np.log(2)
        rdb_dict['dW'][row] = dw
        rdb_dict['sdW'][row] = sdw
        # ---------------------------------------------------------------------
        # calculate the mean full width half max
        mean_fwhm = mp.fwhm() * ccf_ew_row
        # fwhm_row = mp.fwhm() * (ccf_ew_row + per_epoch_d2v / ccf_ew_row)
        fwhm_row = mean_fwhm + dw / mean_fwhm
        sig_fwhm_row = sdw / mean_fwhm
        # ---------------------------------------------------------------------
        # update rdb table
        rdb_dict['fwhm'][row] = fwhm_row
        rdb_dict['sig_fwhm'][row] = sig_fwhm_row
        rdb_dict['vrad_achromatic'][row] = achromatic_velo
        rdb_dict['svrad_achromatic'][row] = sig_achromatic_velo
        rdb_dict['vrad_chromatic_slope'][row] = chromatic_slope
        rdb_dict['svrad_chromatic_slope'][row] = sig_chromatic_slope

        # ---------------------------------------------------------------------
        # Per-band per region RV measurements
        # ---------------------------------------------------------------------
        # get the instrument specific binned parameters
        binned_dict = inst.get_binned_parameters()
        # get info from binned dictionary
        bands = binned_dict['bands']
        blue_end = binned_dict['blue_end']
        red_end = binned_dict['red_end']
        region_names = binned_dict['region_names']
        region_low = binned_dict['region_low']
        region_high = binned_dict['region_high']
        # get the shape of the binned parameters
        # bshape = (len(lblrvfiles), len(bands), len(region_names))
        # make a rv and error matrix based on these binned params
        # rvs_matrix = np.full(bshape, np.nan)
        # err_matrix = np.full(bshape, np.nan)
        # loop around files
        # for row in range(len(lblrvfiles)):
        # get the residuals and dvrms for this rv file
        if flag_calib:
            # rvs[row] - rv_per_line_model[row]
            tmp_rv = np.array(rvs_row, dtype=float)
            # dvrms[row]
            tmp_err = np.array(err, dtype=float)
        else:
            tmp_rv = rvs[row] - rv_per_line_model[row]
            tmp_err = dvrms[row]

        # loop around the bands
        for iband in range(len(bands)):
            # make a mask based on the band (can use rvtable0 as wave start
            #   is the same for all rvtables)
            band_mask = rvtable0['WAVE_START'] > blue_end[iband]
            band_mask &= rvtable0['WAVE_START'] < red_end[iband]
            # loop around the regions
            for iregion in range(len(region_names)):
                # mask based on region
                region_mask = rvtable0['XPIX'] > region_low[iregion]
                region_mask &= rvtable0['XPIX'] < region_high[iregion]
                # -------------------------------------------------------------
                # get combined mask for band and region
                comb_mask = band_mask & region_mask
                # -------------------------------------------------------------
                # get the finite points
                finite_mask = np.isfinite(tmp_err[comb_mask])
                finite_mask &= np.isfinite(tmp_rv[comb_mask])
                # deal with not having enough values (half of total being
                #    non finite)
                if np.sum(finite_mask) < np.sum(comb_mask) / 2:
                    continue
                # deal with not having enough points in general (min = 5)
                if np.sum(comb_mask) < 5:
                    continue
                # -------------------------------------------------------------
                # make a guess on the vrad and svrad via odd ratio mean
                guess7, bulk_error7 = mp.odd_ratio_mean(tmp_rv[comb_mask],
                                                        tmp_err[comb_mask])
                # -------------------------------------------------------------

                cargs = [bands[iband], region_names[iregion]]
                vrad_colname = 'vrad_{0}{1}'.format(*cargs)
                svrad_colname = 'svrad_{0}{1}'.format(*cargs)

                # add new column if not present
                if vrad_colname not in rdb_dict:
                    rdb_dict[vrad_colname] = lblrv_zeros.copy()
                if svrad_colname not in rdb_dict:
                    rdb_dict[svrad_colname] = lblrv_zeros.copy()

                rdb_dict[vrad_colname][row] = guess7
                rdb_dict[svrad_colname][row] = bulk_error7
    # ---------------------------------------------------------------------
    # convert rdb_dict to table
    # ---------------------------------------------------------------------
    rdb_table = Table()
    # loop around columns
    for colname in rdb_dict.keys():
        # add to table
        rdb_table[colname] = rdb_dict[colname]
    # ---------------------------------------------------------------------
    # return rdb table
    return rdb_table


def make_rdb_table2(inst: InstrumentsType, rdb_table: Table) -> Table:
    """
    Combine the rdb table per observation into a table per epoch

    :param inst: Instrument instance
    :param rdb_table: astropy.table.Table, the RDB table (row per observation)

    :return: astropy.table.Table, the RDB table (row per epoch)
    """
    # set function name
    _ = __NAME__ + '.make_rdb_table2()'
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # get the epoch groupings and epoch values
    epoch_groups, epoch_values = inst.get_epoch_groups(rdb_table)
    # -------------------------------------------------------------------------
    # create dictionary storage for epochs
    rdb_dict2 = dict()
    # copy columns from rdb_table (as empty lists)
    for colname in rdb_table.colnames:
        rdb_dict2[colname] = []
    # -------------------------------------------------------------------------
    # Determine columns that use weighted mean
    vrad_colnames = []
    svrad_colnames = []
    # get vrad and svrad columns
    for colname in rdb_table.colnames:
        if colname.startswith('vrad'):
            vrad_colnames.append(colname)
        if colname.startswith('svrad'):
            svrad_colnames.append(colname)
    # determine the weighted mean
    wmean_pairs = dict(zip(vrad_colnames, svrad_colnames))
    wmean_pairs['d2v'] = 'sd2v'
    wmean_pairs['d3v'] = 'sd3v'
    wmean_pairs['fwhm'] = 'sig_fwhm'
    # -------------------------------------------------------------------------
    # log progress
    log.info('Producing LBL RDB 2 table')
    # loop around unique dates
    for idate in tqdm(range(len(epoch_groups))):
        # get the date of this iteration
        epoch = epoch_groups[idate]
        # find all observations for this date
        epoch_mask = epoch_values == epoch
        # get masked table for this epoch (only rows for this epoch)
        itable = rdb_table[epoch_mask]
        # loop around all keys in rdb_table and populate rdb_dict
        for colname in rdb_table.colnames:
            # -----------------------------------------------------------------
            # if column requires wmean, combine value and error
            if colname in wmean_pairs:
                # get value and error for this udate
                vals = itable[colname]
                errs = itable[wmean_pairs[colname]]
                # get error^2
                errs2 = errs ** 2
                # deal with all nans
                if np.sum(np.isfinite(errs2)) == 0:
                    value = np.nan
                    err_value = np.nan
                else:
                    # get 1/error^2
                    value = mp.nansum(vals / errs2) / mp.nansum(1 / errs2)
                    err_value = np.sqrt(1 / mp.nansum(1 / errs2))
                    # push into table
                rdb_dict2[colname].append(value)
                rdb_dict2[wmean_pairs[colname]].append(err_value)
            # -----------------------------------------------------------------
            # if not vrad or svrad then try to mean the column or if not
            #   just take the first value
            elif colname not in wmean_pairs.values():
                # try to produce the mean of rdb table
                # noinspection PyBroadException
                try:
                    rdb_dict2[colname].append(np.mean(itable[colname]))
                except Exception as _:
                    rdb_dict2[colname].append(itable[colname][0])
    # ---------------------------------------------------------------------
    # convert rdb_dict2 to table
    # ---------------------------------------------------------------------
    rdb_table2 = Table()
    # loop around columns
    for colname in rdb_dict2.keys():
        # add to table
        rdb_table2[colname] = rdb_dict2[colname]
    # ---------------------------------------------------------------------
    # return rdb table
    return rdb_table2


def make_drift_table(inst: InstrumentsType, rdb_table: Table) -> Table:
    """
    Make the drift table

    :param inst: Instrument instance
    :param rdb_table: astropy.table.Table, the RDB table (row per epoch)

    :return: astropy.table.Table, the RDB table per epoch corrected for the
             calibration file
    """
    # set function name
    _ = __NAME__ + '.make_drift_table()'
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # get wave file key
    kw_wavefile = inst.params['KW_WAVEFILE']
    # get type key
    type_key = inst.params['KW_REF_KEY']
    # get FP reference string
    ref_list = inst.params['FP_REF_LIST']
    std_list = inst.params['FP_STD_LIST']
    # -------------------------------------------------------------------------
    # storage for output table
    rdb_dict3 = dict()
    # fill columns with empty lists similar to rdb_table2
    for colname in rdb_table.colnames:
        rdb_dict3[colname] = []
    # -------------------------------------------------------------------------
    # get unique wave files
    uwaves = np.unique(rdb_table[kw_wavefile])
    # log progress
    log.info('Producing LBL drift table')
    # loop around unique wave files
    for uwavefile in tqdm(uwaves):
        # find all entries that match this wave file
        wave_mask = rdb_table[kw_wavefile] == uwavefile
        # get table for these entries
        itable = rdb_table[wave_mask]
        # get a list of filenames
        types = itable[type_key]
        # ---------------------------------------------------------------------
        # assume we don't have a reference
        ref_present = False
        ref = Table()
        # loop around filenames and look for reference file
        for row in range(len(types)):
            # type must be in reference list of types
            if types[row] in ref_list:
                # instrumental condition
                if inst.drift_condition(itable[row]):
                    ref_present = True
                    ref = itable[row]
                    break
        # ---------------------------------------------------------------------
        # if we have a reference present correct the file
        if ref_present:
            # loop around the wave files of this type
            for row in range(len(types)):
                # -------------------------------------------------------------
                # if observation is in the standard list of observations
                if types[row] in std_list:
                    # loop around column names
                    for colname in itable.colnames:
                        # -----------------------------------------------------
                        # if column is vrad correct for reference
                        if colname.startswith('vrad'):
                            # get reference value
                            refvrad = ref[colname]
                            # get rv value
                            vrad = itable[colname][row]
                            # correct value
                            vrad_comb = vrad - refvrad
                            # add to dictionary
                            rdb_dict3[colname].append(vrad_comb)
                        # -----------------------------------------------------
                        # if column is svrad correct for reference
                        elif colname.startswith('svrad'):
                            # get reference value
                            refsvrad2 = ref[colname] ** 2
                            # get rv value
                            svrad2 = itable[colname][row] ** 2
                            # correct value
                            svrad_comb = np.sqrt(svrad2 + refsvrad2)
                            # add to dictionary
                            rdb_dict3[colname].append(svrad_comb)
                        # -----------------------------------------------------
                        # else we have a non vrad / svrad column - add as is
                        else:
                            rdb_dict3[colname].append(itable[colname][row])
                # -------------------------------------------------------------
                # else we have a reference file - just add it as is
                else:
                    # loop around column names
                    for colname in itable.colnames:
                        rdb_dict3[colname].append(itable[colname][row])
        # ---------------------------------------------------------------------
        # else we don't have a reference file present --> set to NaN
        else:
            # loop around the wave files of this type
            for row in range(len(types)):
                # loop around column names
                for colname in itable.colnames:
                    # ---------------------------------------------------------
                    # if column is vrad correct for reference
                    if colname.startswith('vrad'):
                        # correct value
                        rdb_dict3[colname].append(np.nan)
                    # ---------------------------------------------------------
                    # if column is svrad correct for reference
                    elif colname.startswith('svrad'):
                        # correct value
                        rdb_dict3[colname].append(np.nan)
                    # ---------------------------------------------------------
                    # else we have a non vrad / svrad column - add as is
                    else:
                        rdb_dict3[colname].append(itable[colname][row])
    # ---------------------------------------------------------------------
    # convert rdb_dict3 to table
    # ---------------------------------------------------------------------
    rdb_table3 = Table()
    # loop around columns
    for colname in rdb_dict3.keys():
        # add to table
        rdb_table3[colname] = rdb_dict3[colname]
    # ---------------------------------------------------------------------
    # return rdb table
    return rdb_table3


def correct_rdb_drift(inst: InstrumentsType, rdb_table: Table,
                      drift_table: Table) -> Table:
    """
    Correct RDB table for drifts (where entry exists in both drift table
    and in rdb_table (based on KW_FILENAME keyword) when entry does not exist
    vrad and svrad are set to NaN

    :param inst: Instrument instance
    :param rdb_table: astropy.table.Table - the RDB 1 table per observation
    :param drift_table: astropy.table.Table - the drift table per observation

    :return: astropy.table.Table - the rdb per observation corrected for drift
             where a drift exists (else vrad and svrad are NaN)
    """
    # set function name
    _ = __NAME__ + '.make_drift_table()'
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # -------------------------------------------------------------------------
    # storage for output table
    rdb_dict4 = dict()
    # fill columns with empty lists similar to rdb_table2
    for colname in rdb_table.colnames:
        rdb_dict4[colname] = []
    # -------------------------------------------------------------------------
    # get the time for cross-matching between the drift file and science
    timestamp = rdb_table['rjd']

    # log progress
    log.info('Producing LBL RDB drift corrected table')
    # loop around the wave files of this type
    for row in tqdm(range(len(timestamp))):
        # create a mask of all files that match in drift file
        file_mask = timestamp[row] == drift_table['rjd']
        # ---------------------------------------------------------------------
        # deal with no files present - cannot correct drift
        if np.sum(file_mask) == 0:
            # loop around all columns
            for colname in rdb_dict4.keys():
                # -------------------------------------------------------------
                # if column is vrad correct for reference
                if colname.startswith('vrad'):
                    # correct value
                    rdb_dict4[colname].append(np.nan)
                # -------------------------------------------------------------
                # if column is svrad correct for reference
                elif colname.startswith('svrad'):
                    # correct value
                    rdb_dict4[colname].append(np.nan)
                # -------------------------------------------------------------
                # else we have a non vrad / svrad column - add as is
                else:
                    rdb_dict4[colname].append(rdb_table[colname][row])
        # ---------------------------------------------------------------------
        # else we have file(s) - use the first
        else:
            # get position in the drift table
            pos = np.where(file_mask)[0][0]
            # loop around all columns
            for colname in rdb_dict4.keys():
                # -------------------------------------------------------------
                # if column is vrad correct for reference
                if colname.startswith('vrad'):
                    # get vrad drift
                    vrad_drift = drift_table[colname][pos]
                    # correct vrad
                    vrad_corr = rdb_table[colname][row] - vrad_drift
                    # correct value
                    rdb_dict4[colname].append(vrad_corr)
                # -------------------------------------------------------------
                # if column is svrad correct for reference
                elif colname.startswith('svrad'):
                    # get svrad drift
                    svrad_drift_2 = drift_table[colname][pos] ** 2
                    # get value
                    svrad_value_2 = rdb_table[colname][row] ** 2
                    # correct svrad
                    svrad_corr = np.sqrt(svrad_value_2 + svrad_drift_2)
                    # correct value
                    rdb_dict4[colname].append(svrad_corr)
                # -------------------------------------------------------------
                # else we have a non vrad / svrad column - add as is
                else:
                    rdb_dict4[colname].append(rdb_table[colname][row])
    # ---------------------------------------------------------------------
    # convert rdb_dict3 to table
    # ---------------------------------------------------------------------
    rdb_table4 = Table()
    # loop around columns
    for colname in rdb_dict4.keys():
        # add to table
        rdb_table4[colname] = rdb_dict4[colname]
    # ---------------------------------------------------------------------
    # return rdb table
    return rdb_table4


# =============================================================================
# Template and Mask functions
# =============================================================================
def get_stellar_models(inst: InstrumentsType, model_dir: str
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the stellar models required for this instrument / observation
    (based on params and the get_stellar_model_format_dict from the instrument)

    :param inst: instrument class instance
    :param model_dir: str, the model directory path

    :return: tuple, 1. np.ndarray: the model wave map and 2. np.ndarray: the
             model spectrum to be used
    """
    # get parameters from instrument class
    params = inst.params
    # -------------------------------------------------------------------------
    # get stellar model wave url
    wave_url = params['STELLAR_WAVE_URL']
    # get stellar model wave file
    wavefile = params['STELLAR_WAVE_FILE']
    # get stellar model url
    model_url = params['STELLAR_MODEL_URL']
    # get stellar model file
    modelfile = params['STELLAR_MODEL_FILE']
    # -------------------------------------------------------------------------
    # get the output path for the wave file
    wavefile_outdir = str(model_dir)
    # get the outpth path for the model files
    modelfile_outdir = str(model_dir)
    # -------------------------------------------------------------------------
    # get instrument specific format dictionary
    fkwargs = inst.get_stellar_model_format_dict(params)
    # -------------------------------------------------------------------------
    # get the full download links
    wavefile_url = (wave_url + wavefile).format(**fkwargs)
    modelfile_url = (model_url + modelfile).format(**fkwargs)
    # -------------------------------------------------------------------------
    # get the full output paths (including files)
    wavefile_outfile = os.path.join(wavefile_outdir, wavefile)
    model_outfile = os.path.join(modelfile_outdir, modelfile)
    # get the full wave output file and model output file
    out_wavefile = wavefile_outfile.format(**fkwargs)
    out_model = model_outfile.format(**fkwargs)
    # -------------------------------------------------------------------------
    # check if we already have the wave file
    if not os.path.exists(out_wavefile):
        # print that we are downloading wave file
        msg = 'Downloading model wave file. \n\t{0}'
        log.general(msg.format(wavefile_url))
        # get wave file (now format cards)
        wget.download(wavefile_url, out=out_wavefile)
    else:
        # print that we are downloading wave file
        msg = 'Found model wave file. \n\t{0}'
        log.general(msg.format(out_wavefile))
    # -------------------------------------------------------------------------
    # check if we already have the model
    if not os.path.exists(out_model):
        # print that we are downloading model file
        msg = 'Downloading model spectrum. \n\t{0}'
        log.general(msg.format(modelfile_url))
        # get wave file (now format cards)
        wget.download(modelfile_url, out=out_model)
    else:
        # print that we are downloading model file
        msg = 'Found model spectrum. \n\t{0}'
        log.general(msg.format(out_model))
    # -------------------------------------------------------------------------
    # load the model wave file and the model spectrum
    m_wavemap, _ = io.load_fits(out_wavefile)
    m_spectrum, _ = io.load_fits(out_model)
    # -------------------------------------------------------------------------
    # model wavemap is in Angstrom -> convert to nm
    m_wavemap = m_wavemap / 10.0
    # -------------------------------------------------------------------------
    # return the wavefile and model file to be used
    return m_wavemap, m_spectrum


def find_mask_lines(inst: InstrumentsType, template_table: Table) -> Table:
    """
    Get the mask lines from the template data

    :param inst:
    :param template_table:
    :return:
    """
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # get the wave and flux vectors for the tempalte
    t_wave = np.array(template_table['wavelength'])
    t_flux = np.array(template_table['flux'])
    with warnings.catch_warnings(record=True) as _:
        t_snr = t_flux / template_table['rms']
        # remove infinite values
        t_snr[np.isinf(t_snr)] = np.nan
    # -------------------------------------------------------------------------
    # smooth the spectrum to avoid lines that coincide with small-scale noise
    #   excursion
    t_flux_tmp = np.zeros_like(t_flux)
    for offset in range(-2, 3):
        t_flux_tmp += np.roll(t_flux, offset)
    # copy over original vector
    t_flux = np.array(t_flux_tmp)
    # -------------------------------------------------------------------------
    # find the first and second derivative of the flux
    dflux = np.gradient(t_flux)
    ddflux = np.gradient(dflux)
    # -------------------------------------------------------------------------
    # lines are regions there is a sign change in the derivative of the flux
    #   we also have some checks for NaNs
    cond1 = np.sign(dflux[1:]) != np.sign(dflux[:-1])
    cond2 = np.isfinite(ddflux[1:])
    cond3 = np.isfinite(dflux[1:])
    cond4 = np.isfinite(dflux[:-1])
    # get the position where all these condition are True
    line = np.where(cond1 & cond2 & cond3 & cond4)[0]
    # -------------------------------------------------------------------------
    # create vectors for the outputs
    # start of a line
    ll_mask_s = np.zeros_like(line, dtype=float)
    # end of a line
    ll_mask_e = np.zeros_like(line, dtype=float)
    # depth of line relative to continuum
    depth = np.zeros_like(line, dtype=float)
    # -------------------------------------------------------------------------
    # the weight is the second derivative of the flux. The sharper the line,
    # the more weight we give it
    # weight of the line is not used in LBL but nice to document
    w_mask = ddflux[line]
    f_mask = t_flux[line]
    snr_mask = t_snr[line]
    # -------------------------------------------------------------------------
    # find the bits of continuum on either side of line and find depth
    #    relative to that
    with warnings.catch_warnings(record=True) as _:
        depth[1:-1] = 1 - f_mask[1:-1] / (0.5 * (f_mask[0:-2] + f_mask[2:]))
    # -------------------------------------------------------------------------
    # print progress
    log.general('Finding mask lines')
    # loop around each line and populate the start and ends of the line
    for it in tqdm(range(len(line))):
        # get pixel start and end
        start, end = line[it], line[it] + 2
        # ---------------------------------------------------------------------
        # we perform a linear interpolation to find the exact wavelength
        # where the derivatives goes to zero
        coeffs = np.polyfit(dflux[start:end], t_wave[start:end], 1)
        # we only want the center
        wave_cent = coeffs[1]
        # ---------------------------------------------------------------------
        # set the start equal to the center of the line
        ll_mask_s[it] = wave_cent
        # ---------------------------------------------------------------------
        # set the end equal to the center of the line
        ll_mask_e[it] = wave_cent
    # -------------------------------------------------------------------------
    # store in a table for on going use
    table = Table()
    table['ll_mask_s'] = ll_mask_s
    table['ll_mask_e'] = ll_mask_e
    table['w_mask'] = w_mask
    table['value'] = f_mask
    table['depth'] = depth
    table['line_snr'] = abs(depth * snr_mask)
    # return the mask table
    return table


def mask_systemic_velocity(inst: InstrumentsType, line_table: Table,
                           m_wavemap: np.ndarray,
                           m_spectrum: np.ndarray) -> float:
    """
    Measure the systemic velocity of a mask compared to a model

    :param inst: the instrument class
    :param line_table: astropy table, the table of lines
    :param m_wavemap: np.ndarray, the 1D model wave map
    :param m_spectrum: np.ndarray, the 1D model flux spectrum

    :return: float, the systemic velocity
    """
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # create a spline of the model spectrum
    smodel = mp.iuv_spline(m_wavemap, m_spectrum)
    # get the center of the lines
    wave_cent = 0.5 * (line_table['ll_mask_s'] + line_table['ll_mask_e'])
    # get a dv range to search across
    dvs = np.arange(-200, 200 + 0.5, 0.5)
    # work out the negative mask
    neg_mask = line_table['w_mask'] > 0
    # get the negative masked vectors
    weight_tmp = line_table['w_mask'][neg_mask]
    wave_tmp = wave_cent[neg_mask]
    # -------------------------------------------------------------------------
    # print progress
    log.general('Calculating CCF for each DV element')
    # storage for the CCF outputs
    ccf = np.zeros_like(dvs)
    # calculate the CCF for each dv element
    for it in tqdm(range(len(dvs))):
        # get the wave grid at this rv element
        wave_it = mp.doppler_shift(wave_tmp, dvs[it] * 1000)
        # get the model on this wave grid
        model_it = smodel(wave_it)
        # calculate the ccf (just the sum of weights * model)
        ccf[it] = np.sum(weight_tmp * model_it)
    # robust polyfit on the ccf
    ccf_coeffs = mp.robust_polyfit(dvs, ccf, 3, 5)
    # remove gradients in the ccf
    ccf = ccf / np.polyval(ccf_coeffs[0], dvs)
    # -------------------------------------------------------------------------
    # find the exact position of the minimum
    pix_min = np.argmin(ccf)
    pstart, pend = pix_min - 1, pix_min + 2
    # fit a quadratic
    fit_coeffs = np.polyfit(dvs[pstart:pend], ccf[pstart:pend], 2)
    # work out the systemic velocity
    sys_vel = -0.5 * fit_coeffs[1] / fit_coeffs[0]
    # display system velocity
    msg = 'System velocity for {0} is {1:.3f} km/s'
    log.general(msg.format(inst.params['OBJECT_TEMPLATE'], sys_vel))
    # -------------------------------------------------------------------------
    # ccf plot
    plot.mask_plot_ccf(inst, dvs, ccf, sys_vel)
    # -------------------------------------------------------------------------
    # return the systemic velocity
    return sys_vel


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

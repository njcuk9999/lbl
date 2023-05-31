#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-12-13

@author: cook
"""
import os
from typing import Any, Dict, Tuple, Union
import shutil

import numpy as np
from astropy import constants
from astropy.table import Table

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default
from lbl.instruments import select
from lbl.science import general
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
# spline typing
SplineReturn = Union[mp.IUVSpline, mp.NanSpline]


# =============================================================================
# Define functions
# =============================================================================
def get_tapas_lbl(inst: InstrumentsType, modeldir: str,
                  extname: str) -> Table:
    """
    Get the tapas lbl file (either from disk or from online)

    :param inst: Instrument Type instance - the instrument class
    :param modeldir: str, the model directory
    :param extname: str, the extension name to load

    :return: astropy.table.Table - the Table for extname
    """
    # get parameters from inst
    tapas_file = inst.params['TELLUCLEAN_TAPAS_FILE']
    data_dir = inst.params['DATA_DIR']
    # get data directory
    data_dir = io.check_directory(data_dir)
    # make other directory
    tapas_path = io.make_dir(data_dir, 'other', 'Other')
    # construct tapas in/out file path
    tapas_inpath = os.path.join(modeldir, tapas_file)
    tapas_outpath = os.path.join(tapas_path, tapas_file)
    # copy tapas file to tapas
    if os.path.exists(tapas_outpath):
        pass
    elif os.path.exists(tapas_inpath):
        # copy over tapas file
        shutil.copy(tapas_inpath, tapas_outpath)
    else:
        emsg = 'Cannot find tapas file: {0}'
        raise LblException(emsg.format(tapas_inpath))
    # load table
    tapas_lbl = io.load_table(tapas_outpath, extname=extname)
    # return the table
    return tapas_lbl


def get_tapas_spl(inst: InstrumentsType, model_dir: str
                  ) -> Tuple[SplineReturn, SplineReturn]:
    """
    Read table with TAPAS model for 6 molecules.

    :param inst: Instrument Type instance - the instrument class
    :param model_dir: str, the model directory
    :return:
    """

    # get parameters from params
    tapas_dv = inst.params['TELLUCLEAN_DV0']
    # -------------------------------------------------------------------------
    # get tapas file
    tmp_tapas = get_tapas_lbl(inst, model_dir, 'ABSOSPEC')
    # -------------------------------------------------------------------------
    # extract out the wave solution, water and others columns from table
    tapas_wave = tmp_tapas['WAVELENGTH']
    trans_others = tmp_tapas['ABSO_OTHERS']
    trans_water = tmp_tapas['ABSO_WATER']
    # -------------------------------------------------------------------------
    # doppler shift wave solution by tapas_dv
    if tapas_dv != 0.0:
        tapas_wave_shifted = mp.doppler_shift(tapas_wave, tapas_dv * 1000)
    else:
        tapas_wave_shifted = np.array(tapas_wave)
    # define spline function for both, optionally shift the grid
    spl_others = mp.iuv_spline(tapas_wave_shifted, trans_others, k=1, ext=3)
    spl_water = mp.iuv_spline(tapas_wave_shifted, trans_water, k=1, ext=3)
    # -------------------------------------------------------------------------
    # return spl_others and spl_water
    return spl_others, spl_water


def load_tellu_masks(inst, model_dir: str) -> Tuple[Table, Table]:
    """
    Load telluric masks for others and water content

    :param inst: Instrument Type instance - the instrument class
    :param model_dir: str, the model directory

    :return: tuple, 1. the others mask, 2. the water mask
    """
    # get parameters from inst
    mask_domain_lower = inst.params['TELLUCLEAN_MASK_DOMAIN_LOWER']
    mask_domain_upper = inst.params['TELLUCLEAN_MASK_DOMAIN_UPPER']
    # -------------------------------------------------------------------------
    # read others file
    table_others = get_tapas_lbl(inst, model_dir, 'CCFOTHER')
    # remove domain that is out of bounds
    keep_others = table_others['ll_mask_s'] > mask_domain_lower
    keep_others &= table_others['ll_mask_s'] < mask_domain_upper
    # apply mask to others table
    table_others = table_others[keep_others]
    # -------------------------------------------------------------------------
    # read water file
    table_water = get_tapas_lbl(inst, model_dir, 'CCFWATER')
    # remove domain that is out of bounds
    keep_others = table_water['ll_mask_s'] > mask_domain_lower
    keep_others &= table_water['ll_mask_s'] < mask_domain_upper
    # apply mask to others table
    table_water = table_water[keep_others]
    # -------------------------------------------------------------------------
    # return tables for the other mask and water mask
    return table_others, table_water


def get_abso_sp(wave_vector: np.ndarray, expo_others: float, expo_water: float,
                spl_others: SplineReturn, spl_water: SplineReturn,
                kwidth: float = 4.95, ex_gau: float = 2.20,
                dv_abso: float = 0.0, wave0: float = 965, wave1: float = 2500,
                dv_grid: float = 1.0) -> np.ndarray:
    """
    Return the absorption spectrum from exponents describing water and 'others'
    in absorption

    :param wave_vector: np.array (1D) - the wave grid onto which spectrum is
                        splined
    :param expo_others: float, the optical depth of all species other than
                        water
    :param expo_water: float, the optical depth of water
    :param spl_others: Spline, of the others mask
    :param spl_water: Spline, of the water mask
    :param kwidth: float, gaussian width of kernel
    :param ex_gau: float, exponent of the gaussian (a value of 2 is gaussian, a
                   value > 2 is boxy)
    :param dv_abso: float, velocity of absorption in km/s
    :param wave0: float, first wavelength element of the magic grid
    :param wave1: float, second wavelength element of the magic grid
    :param dv_grid: float, grid size in km/s

    :return: np.ndarray, the transmission absorption spectrum
    """

    # if zero is given as both exponents return a flat vector
    if (expo_others == 0) and (expo_water == 0):
        return np.ones_like(wave_vector)
    # -------------------------------------------------------------------------
    # define the convolution kernel for the model. This shape factor can be
    #   modified if required.
    wkernel = kwidth / mp.fwhm()
    # definition of the convoolution kernel x grid, defined over 4 FWHM
    kernal_width = int(kwidth * 4)
    # get the kernel vector
    kernel_vec = np.arange(-kernal_width, kernal_width + 1, 0.5, dtype=float)
    # normalize the kernel
    kernel = np.exp(-0.5 * np.abs(kernel_vec / wkernel) ** ex_gau)
    kernel = kernel[kernel > 1.0e-6 * np.max(kernel)]
    kernel = kernel / np.sum(kernel)
    # -------------------------------------------------------------------------
    # create a magic grid onto which we spline our transmission, same as for
    #   s1d_v creation in APERO
    magic_grid = general.get_magic_grid(wave0, wave1, dv_grid * 1000)
    # -------------------------------------------------------------------------
    # get the water and other masks
    sp_others = spl_others(magic_grid)
    sp_water = spl_water(magic_grid)
    # -------------------------------------------------------------------------
    # for numerical stability, we may have values very slightly below 0 from
    #    the spline above, negative values don't work with fractional exponents
    sp_others[sp_others < 0] = 0.0
    sp_water[sp_water < 0] = 0.0
    # -------------------------------------------------------------------------
    # applying optical depths
    trans_others = sp_others ** expo_others
    trans_water = sp_water ** expo_water
    # getting the full absorption at full resolution
    trans = trans_others * trans_water
    # convolving after product (to avoid the infamous commutativity problem
    trans_convolved = np.convolve(trans, kernel, mode='same')
    # -------------------------------------------------------------------------
    # shift magic grid by the velocity of absorption
    magic_grid_shift = mp.doppler_shift(magic_grid, 1000 * dv_abso)
    # spline the transmission onto this shift grid
    magic_spline = mp.iuv_spline(magic_grid_shift, trans_convolved)
    # -------------------------------------------------------------------------

    # deal with flux being a 2D array
    if len(wave_vector.shape) == 2:
        trans_out = np.zeros(wave_vector.shape)
        # loop around each order and spline
        for order_num in range(wave_vector.shape[0]):
            trans_out[order_num] = magic_spline(wave_vector[order_num])
    # else we have a 1D array --> spline full vector at once
    else:
        trans_out = np.array(magic_spline(wave_vector))
    # -------------------------------------------------------------------------
    # return transmission
    return trans_out


def correct_tellu(inst: InstrumentsType, template_file: str,
                  e2ds_params: Dict[str, Any],
                  spl_others: SplineReturn,
                  spl_water: SplineReturn,
                  model_dir: str) -> Dict[str, Any]:
    """
    Pass an e2ds dictionary and return the telluric-corredted data.

    This is a rough model fit and we will need to perform residual correction
    on top of it (if possible)

    Will fit both water and all dry components of the absoprtion separately

    :param inst: InstrumentType instance, the class for this instrument
    :param template_file: str, the path to the template file
    :param e2ds_params: dict, the e2ds dictionary of parameters
    :param spl_others: spline, the spline of other absorbers spectrum
    :param spl_water: splien, the spline of water absorbers spectrum
    :param model_dir: str, the path to the model directory

    :return: dictionary the updated e2ds_params with telluric corrected data in
    """
    # get parameters from inst
    force_airmass = inst.params['TELLUCLEAN_FORCE_AIRMASS']
    ccf_scan_range = inst.params['TELLUCLEAN_CCF_SCAN_RANGE']
    max_iterations = inst.params['TELLUCLEAN_MAX_ITERATIONS']
    kwidth = inst.params['TELLUCLEAN_KERNEL_WID']
    ex_gau = inst.params['TELLUCLEAN_GAUSSIAN_SHAPE']
    wave0 = inst.params['TELLUCLEAN_WAVE_LOWER']
    wave1 = inst.params['TELLUCLEAN_WAVE_UPPER']
    trans_threshold = inst.params['TELLUCLEAN_TRANSMISSION_THRESHOLD']
    sigma_threshold = inst.params['TELLUCLEAN_SIGMA_THRESHOLD']
    recenter_ccf = inst.params['TELLUCLEAN_RECENTER_CCF']
    recenter_ccf_fit_others = inst.params['TELLUCLEAN_RECENTER_CCF_FIT_OTHERS']
    default_water_abso = inst.params['TELLUCLEAN_DEFAULT_WATER_ABSO']
    others_bounds_lower = inst.params['TELLUCLEAN_OTHERS_BOUNDS_LOWER']
    others_bounds_upper = inst.params['TELLUCLEAN_OTHERS_BOUNDS_UPPER']
    water_bounds_lower = inst.params['TELLUCLEAN_WATER_BOUNDS_LOWER']
    water_bounds_upper = inst.params['TELLUCLEAN_WATER_BOUNDS_UPPER']
    conv_limit = inst.params['TELLUCLEAN_CONVERGENCE_LIMIT']
    # get the e2ds flux from e2ds parameters
    e2ds_flux = np.array(e2ds_params['flux'])
    e2ds_ini_flux = np.array(e2ds_params['flux'])
    wavemap = e2ds_params['wavelength']
    airmass = e2ds_params['AIRMASS']
    objname = e2ds_params['OBJECT']
    berv = e2ds_params['BERV']
    # -------------------------------------------------------------------------
    # Load the template (if it exists and if we want to use it)
    # -------------------------------------------------------------------------
    # if we don't have template we use an array of ones
    if not inst.params['TELLUCLEAN_USE_TEMPLATE']:
        # template flag
        template_flag = False
        # proxy for template spline
        template_spl = None
    elif not os.path.exists(template_file):
        # template flag
        template_flag = False
        # proxy for template spline
        template_spl = None
    else:
        # template flag
        template_flag = True
        # load the template
        template_table = inst.load_template(template_file)
        # get wave and flux for the template
        wave_template = template_table['wavelength']
        flux_template = template_table['flux']
        # mask out nans
        keep_mask = np.isfinite(flux_template)
        wave_template = wave_template[keep_mask]
        flux_template = flux_template[keep_mask]
        # get the template spline
        template_spl = mp.iuv_spline(wave_template, flux_template, k=1, ext=3)

    # -------------------------------------------------------------------------
    # Deal with overlapping orders in e2ds and trim the overlapping domain
    #   between orders
    # -------------------------------------------------------------------------
    keep = np.ones(wavemap.shape, dtype=bool)

    # Question: why greater than 3?
    if e2ds_flux.shape[0] > 3:
        # loop around order and make sure all wavelengths kept are between
        #   the previous and next orders wave maps
        for order_num in range(1, wavemap.shape[0] - 1):
            keep[order_num] = wavemap[order_num] > wavemap[order_num - 1][::-1]
            keep[order_num] &= wavemap[order_num] < wavemap[order_num + 1][::-1]
        # first and last orders are rejected
        keep[0] = False
        keep[-1] = False

    # apply mask in 1D to wave solution and spectrum
    keep = keep.ravel()
    # keep non-overlapping bits
    wave_vector = wavemap.ravel()[keep]
    sp_vector = e2ds_flux.ravel()[keep]
    # -------------------------------------------------------------------------
    # shift template onto our wavelength grid
    # -------------------------------------------------------------------------
    # if we have a template apply the berv offset
    if template_flag:
        # apply wavelength to spline
        template_wave = mp.doppler_shift(wave_vector, -berv)
        # apply wavelength to spline
        template_vector = template_spl(template_wave)
    # else our template is just ones
    else:
        template_vector = np.ones_like(wave_vector)
    # -------------------------------------------------------------------------
    # remove NaNs for numerical stability
    # -------------------------------------------------------------------------
    # removing nans and setting to zero biases a bit the CCF but this should be
    #   okay after we converge
    sp_vector[np.isnan(sp_vector)] = 0
    sp_vector[sp_vector < 0] = 0
    # -------------------------------------------------------------------------
    # Measure the exponents
    # -------------------------------------------------------------------------
    # first guess at velocity of absorption is 0 km/s
    dv_abso = 0.0
    # set initial velocity measurements for water and others to zero
    dv_water = 0.0
    dv_others = 0.0
    # -------------------------------------------------------------------------
    # load the masks
    table_others, table_water = load_tellu_masks(inst, model_dir)
    # -------------------------------------------------------------------------
    # start with no correction of abso to get the ccf
    # we start at zero to get a velocity measurement even if we may force to
    # airmass
    expo_water = 0
    expo_others = 0
    # keep track of consecutive exponents and test convergence
    expo_water_prev = np.inf
    expo_others_prev = np.inf
    # set the gradient of exponents to be large initially
    dexpo = np.inf
    # storage for amps and exponents
    amp_water_list = []
    amp_others_list = []
    expos_water = []
    expos_others = []
    # scanning range for the ccf computation
    ddvec = np.arange(-ccf_scan_range, ccf_scan_range + 1)  # + dv_abso

    # storage for the CCFs
    ccf_others = np.zeros_like(ddvec, dtype=float)
    ccf_water = np.zeros_like(ddvec, dtype=float)

    # get a wave solution for the tranmission shifted by the berv if required
    wave_trans = np.array(wave_vector)
    # -------------------------------------------------------------------------
    # counter for the number of iterations (we will stop after a certain number)
    #   if it hasn't converged
    iteration = 0
    # storage for plotting
    plt_ddvecs, plt_ccf_waters, plt_ccf_others = dict(), dict(), dict()
    sp_tmp, trans = np.zeros_like(sp_vector), np.zeros_like(sp_vector)
    # loop around until criteria met or maximum iterations met
    while (dexpo > conv_limit) and (iteration < max_iterations):
        # ---------------------------------------------------------------------
        # get the absorption spectrum from exponents describing water and
        #     'others' in absorption
        trans = get_abso_sp(wave_trans, expo_others, expo_water,
                            spl_others, spl_water, kwidth=kwidth, ex_gau=ex_gau,
                            dv_abso=dv_abso, wave0=wave0, wave1=wave1)
        # ---------------------------------------------------------------------
        # remove the transmission from the spectrum
        sp_tmp = sp_vector / trans
        # remove the template from the spectrum
        sp_tmp = sp_tmp / template_vector

        # flag any NaN pixels
        valid = np.isfinite(sp_tmp)
        # remove any poor transmission
        valid &= (trans > np.exp(trans_threshold))
        # ---------------------------------------------------------------------
        # apply some cuts to very discrepant points. These will be set to zero
        #   not to bias the CCF too mch
        sigma_cut = np.nanmedian(np.abs(sp_tmp)) * sigma_threshold
        # remove non finite pixels
        sp_tmp[~np.isfinite(sp_tmp)] = 0.0
        # remove outliers
        sp_tmp[sp_tmp > sigma_cut] = 0.0
        # remove negative pixels
        sp_tmp[sp_tmp < 0] = np.nan
        # ---------------------------------------------------------------------
        # compute the CCFs for others and water
        # ---------------------------------------------------------------------
        # get the CCF of the test spectrum
        ccf_spl = mp.iuv_spline(wave_vector[valid], sp_tmp[valid], k=1, ext=1)
        # storage for all CCF values for others and water
        all_others = np.zeros([len(table_others), len(ddvec)])
        all_water = np.zeros([len(table_water), len(ddvec)])
        # loop around our CCF elements vector and calculate the CCF
        #    we compute ccf_others all the tiem, even when forcing the airmass
        #    just to look at its structure and potential residuals
        for it in range(len(ddvec)):
            # shift the wavelengths by ccf element
            wave_other1 = table_others['ll_mask_s']
            wave_other2 = mp.doppler_shift(wave_other1, ddvec[it] * 1000)
            # calculate spline at this wavelength shift and multiply by mask
            tmp_other = ccf_spl(wave_other2) * table_others['w_mask']
            # set all zero values to NaN
            tmp_other[tmp_other == 0] = np.nan
            # push into CCF vector
            all_others[:, it] = tmp_other
            # -----------------------------------------------------------------
            # shift the wavelengths by ccf element
            wave_water1 = table_water['ll_mask_s']
            wave_water2 = mp.doppler_shift(wave_water1, ddvec[it] * 1000)
            # calculate spline at this wavelength shift and multiply by mask
            tmp_water = ccf_spl(wave_water2) * table_water['w_mask']
            # set all zero values to NaN
            tmp_water[tmp_water == 0] = np.nan
            # push into CCF vector
            all_water[:, it] = tmp_water
        # CCF by definition is the sum of all spectra points
        ccf_water = np.nansum(all_water, axis=0)
        ccf_others = np.nansum(all_others, axis=0)
        # ---------------------------------------------------------------------
        # remove a polynomial fit (remove continuum of the CCF) for water
        water_coeffs, _ = mp.robust_polyfit(ddvec, ccf_water, 2, 3)
        ccf_water = ccf_water - np.polyval(water_coeffs, ddvec)
        # remove a polynomial fit (remove continuum of the CCF) for water
        others_coeffs, _ = mp.robust_polyfit(ddvec, ccf_others, 2, 3)
        ccf_others = ccf_others - np.polyval(others_coeffs, ddvec)
        # ---------------------------------------------------------------------
        # subtract the median of the CCF outside the core of the gaussian.
        #     We take this to be the 'external' part of of the scan range
        out_mask = np.abs(ddvec) > ccf_scan_range / 2
        ccf_water = ccf_water - np.nanmedian(ccf_water[out_mask])
        if not force_airmass:
            ccf_others = ccf_others - np.nanmedian(ccf_others[out_mask])
        # ---------------------------------------------------------------------
        # we measure absorption velocity by fitting a gaussian to the
        #     absorption profile. This updates the dv_abso value for the
        #     next steps.
        if recenter_ccf:
            # deal with first iteration
            if iteration == 0:
                # set up our curve fit guess for water
                guess = [ddvec[np.argmin(ccf_water), 4, np.nanmin(ccf_water)]]
                # fit water
                popt, pcov = mp.curve_fit(mp.gauss_function, ddvec,
                                          ccf_water, p0=guess)
                # velocity of the water is the mean position of the gaussian
                dv_water = popt[1]
                # fit others (if required)
                if recenter_ccf_fit_others:
                    # set up our curve fit guess for others
                    guess = [0, 4, np.nanmin(ccf_others)]
                    # fit water
                    popt, pcov = mp.curve_fit(mp.gauss_function, ddvec,
                                              ccf_others, p0=guess)
                    # velocity of others is the mean position of the gaussian
                    dv_others = popt[1]
                # else set to the same velocity as water
                else:
                    dv_others = float(dv_water)
                # mean of water and others
                dv_abso = 0.5 * (dv_others + dv_water)
        # ---------------------------------------------------------------------
        # define middle of ccf
        middle_mask = np.abs(ddvec - dv_abso) < kwidth
        # get the amplitude of the middle of the CCF
        amp_water = np.nansum(ccf_water[middle_mask])
        amp_others = np.nansum(ccf_others[middle_mask])
        # push into storage
        amp_water_list.append(amp_water)
        amp_others_list.append(amp_others)
        # store exponents
        expos_others.append(expo_others)
        expos_water.append(expo_water)
        # ---------------------------------------------------------------------
        # for first iteration the next exponents to be used for others is the
        #   airmass (in all cases) and for water is the default typical value
        if iteration == 0:
            # set the exponent of others to airmass
            expo_others = float(airmass)
            # set the exponent of otherse to default water absorption
            expo_water = float(default_water_abso)
            # keep track of the convergence params
            expo_water_prev = expo_water
            expo_others_prev = expo_others
            # store vectors for plotting
            plt_ddvecs[iteration] = np.array(ddvec)
            plt_ccf_waters[iteration] = ccf_water
            plt_ccf_others[iteration] = ccf_others
            # add to iteration number
            iteration += 1
            # continue to next iteration
            continue
        # ---------------------------------------------------------------------
        # for the 2 to 5 iterations we fit a line and find the point where
        #   the amplitude would be zero
        # for more than 5 iterations we get smarter and fit a 2nd order
        # polynomial
        # ---------------------------------------------------------------------
        # get the current amp_waters and amp_others
        c_amp_water = np.array(amp_water_list)
        c_amp_others = np.array(amp_others_list)
        c_expos_water = np.array(expos_water)
        c_expos_others = np.array(expos_others)

        # simple linear fit for iterations <= 5
        if iteration <= 5:
            # for others
            sortmask_others = np.argsort(np.abs(c_amp_others))
            fit_others = np.polyfit(c_amp_others[sortmask_others[:4]],
                                    c_expos_others[sortmask_others[:4]], 1)
            # for water
            sortmask_water = np.argsort(np.abs(c_amp_water))
            fit_water = np.polyfit(c_amp_water[sortmask_water[:4]],
                                   c_expos_water[sortmask_water[:4]], 1)
        # otherwise fit a 2nd order polynomial
        else:
            fit_others = np.polyfit(c_amp_others, c_expos_others, 1)
            fit_water = np.polyfit(c_amp_water, c_expos_water, 1)
        # ---------------------------------------------------------------------
        # deal with the next guesse at expo others
        # ---------------------------------------------------------------------
        # if we force the airmass the next guess is always the airmass
        if force_airmass:
            expo_others = float(airmass)
        # otherwise we use the fit
        else:
            expo_others = float(fit_others[1])
        # deal with out of bounds expo_others
        if expo_others < others_bounds_lower:
            expo_others = others_bounds_lower
            # TODO: could be used as a QC check
        elif expo_others > others_bounds_upper:
            expo_others = others_bounds_upper
            # TODO: could be used as a QC check
        # ---------------------------------------------------------------------
        # deal with the next guesse at expo others
        # ---------------------------------------------------------------------
        # find best guess for water exponent
        expo_water = float(fit_water[1])
        # deal with out of bounds expo_water
        if expo_water < water_bounds_lower:
            expo_water = water_bounds_lower
            # TODO: could be used as a QC check
        elif expo_water > water_bounds_upper:
            expo_water = water_bounds_upper
            # TODO: could be used as a QC check
        # ---------------------------------------------------------------------
        # calculate difference between this iteration and previous iteration
        expo_water_diff = np.abs(expo_water_prev - expo_water)
        expo_others_diff = np.abs(expo_others_prev - expo_others)

        # have we converged yet?
        if force_airmass:
            dexpo = float(expo_water_diff)

        else:
            dexpo = np.sqrt(expo_water_diff ** 2 + expo_others_diff ** 2)
        # ---------------------------------------------------------------------
        # print out this iterations values
        if not force_airmass:
            msg = ('{0}:\twater expo={1:.4f}, dry expo={2:.4f}, '
                   'airmass={3:.4f}, template={4}  dexpo={5:.4f}')
        else:
            msg = ('{0}:\twater expo={1:.4f}, dry expo={2:.4f}, '
                   'force=True, template={4}  dexpo={5:.4f}')
        # args
        margs = [iteration, expo_water, expo_others, airmass, template_flag,
                 dexpo]
        # log message
        log.general(msg.format(*margs))

        # keep track of the convergence params
        expo_water_prev = float(expo_water)
        expo_others_prev = float(expo_others)
        # store vectors for plotting
        plt_ddvecs[iteration] = np.array(ddvec)
        plt_ccf_waters[iteration] = ccf_water
        plt_ccf_others[iteration] = ccf_others
        # add to iteration number
        iteration += 1
    # -------------------------------------------------------------------------
    # plot the ccf vectors for each iteration
    plot.ccf_vector_plot(inst, plt_ddvecs, plt_ccf_waters, plt_ccf_others,
                         objname)
    # plot the corrected spectrum
    plot.tellu_corr_plot(inst, wave_vector, sp_vector, trans,
                         template_vector, template_flag, objname)
    # -------------------------------------------------------------------------
    # re-get wave grid for transmission (without trimming)
    wave_trans = np.array(wavemap)
    # -------------------------------------------------------------------------
    # get the final absorption spectrum to be used on the science data.
    #    No trimming done on the wave grid
    abso_e2ds = get_abso_sp(wave_trans, expo_others, expo_water,
                            spl_others, spl_water, kwidth=kwidth, ex_gau=ex_gau,
                            dv_abso=dv_abso, wave0=wave0, wave1=wave1)
    # -------------------------------------------------------------------------
    # all absorption deeper than exp(-1) ~ 30% is considered too deep to
    #     be corrected. We set values there to NaN
    thres_mask = abso_e2ds < np.exp(trans_threshold)
    # set masked values to NaN
    abso_e2ds[thres_mask] = np.nan
    # -------------------------------------------------------------------------
    # correct input e2ds image
    corrected_e2ds = e2ds_ini_flux / abso_e2ds
    # set any infinite values to NaNs
    corrected_e2ds[~np.isfinite(e2ds_ini_flux)] = np.nan
    # -------------------------------------------------------------------------
    # push back into e2ds array
    e2ds_params['pre_cleaned_flux'] = corrected_e2ds
    e2ds_params['pre_cleaned_mask'] = thres_mask
    e2ds_params['pre_cleaned_abso'] = abso_e2ds
    # add the exponents of water and others
    e2ds_params['pre_cleaned_exponent_water'] = expo_water
    e2ds_params['pre_cleaned_exponent_others'] = expo_others
    # add the velocities of water and others
    e2ds_params['pre_cleaned_dv_water'] = dv_water
    e2ds_params['pre_cleaned_dv_others'] = dv_others
    # calculate the derivative of the ccfs powers
    dccf_water = np.gradient(ccf_water)
    dccf_others = np.gradient(ccf_others)
    e2ds_params['pre_cleaned_power_water'] = np.nansum(dccf_water ** 2)
    e2ds_params['pre_cleaned_power_others'] = np.nansum(dccf_others ** 2)
    # -------------------------------------------------------------------------
    # return the e2ds parameters
    return e2ds_params


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

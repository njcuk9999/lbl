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
from lbl.science import tellu_derivative

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
log = io.log
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
    # make other directory (quietly: this runs two to three times per file)
    tapas_path = io.make_dir(data_dir, 'other', 'Other', verbose=False)
    # construct tapas in/out file path
    tapas_inpath = io.model_path(modeldir, 'tellurics', tapas_file)
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
    wkernel = kwidth / mp.fwhm_value()
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
    kwidth = inst.params['TELLUCLEAN_KERNEL_WID']
    ex_gau = inst.params['TELLUCLEAN_GAUSSIAN_SHAPE']
    wave0 = inst.params['TELLUCLEAN_WAVE_LOWER']
    wave1 = inst.params['TELLUCLEAN_WAVE_UPPER']
    trans_threshold = inst.params['TELLUCLEAN_TRANSMISSION_THRESHOLD']
    # get the e2ds flux from e2ds parameters
    e2ds_flux = np.array(e2ds_params['flux'])
    e2ds_ini_flux = np.array(e2ds_params['flux'])
    wavemap = e2ds_params['wavelength']
    airmass = e2ds_params['AIRMASS']
    objname = e2ds_params['OBJECT']
    berv = e2ds_params['BERV']
    # le BERV dont la TRANSMISSION doit etre decalee, qui n'est pas celui des
    # vitesses: zero quand la grille est deja dans le repere de l'observateur
    berv_trans = e2ds_params.get('BERV_TRANS', 0.0)
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

        if not inst.params['USE_SAVGOL_TEMPLATE']:
            flux_template = np.array(template_table['flux'])
        else:
            flux_template = np.array(template_table['flux_savgol_d0'])

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
    sp_vector[np.isnan(sp_vector)] = 0
    sp_vector[sp_vector < 0] = 0
    # -------------------------------------------------------------------------
    # the transmission is evaluated where the ground sees it
    #
    # TAPAS DESCRIBES THE ATMOSPHERE, SO IT LIVES IN THE OBSERVER FRAME. When
    # the wave solution handed to us is already barycentric, as it is for the
    # ESO S2D files of HARPS, HARPS-N and ESPRESSO, the transmission has to be
    # evaluated at the wavelengths the ground sees, not the ones the barycentre
    # does. get_berv is zero for those classes, which is the right answer for
    # the velocities and the reason this shift was never applied; the BERV of
    # the header is what the atmosphere moved by.
    #
    # Measured on HD 10700 (HARPS, 2026-10-15), O2 B band 686-690 nm: with the
    # barycentric grid the model correlates with the spectrum at -0.035, i.e.
    # not at all, and the cleaning divided by at most 2.7 where the model asks
    # for 7.5. Shifted here, the correlation is 0.973 and the residual offset
    # against the BERV goes from a slope of -1.0003 to 0.0000.
    #
    # doppler_shift(wave, +berv) maps barycentric wavelengths onto the ones the
    # observer sees: at a BERV of +23.74 km/s it moves 688 nm by -23.74 km/s.
    wave_trans = np.array(wave_vector)
    if berv_trans:
        wave_trans = mp.doppler_shift(wave_trans, berv_trans)
    # the absorption is not shifted in velocity: the model is evaluated in the
    #   ground frame above, which is where the atmosphere is
    dv_abso = 0.0
    # -------------------------------------------------------------------------
    # THE EXPONENTS, BY THE DERIVATIVE METHOD, FOR EVERY INSTRUMENT
    #
    # The derivative method of telluric_fit (tellu_tools.optimize_exponents,
    # called by predict_abso.py) keeps the exponents that make the CORRECTED
    # spectrum smoothest, i.e. that minimise the weighted scatter of its
    # pixel-to-pixel gradient on the telluric windows. A wrong exponent leaves
    # a derivative-shaped residual at every telluric line; stellar lines
    # contribute the same whatever the exponent, so the criterion depends on
    # the tellurics alone (lbl.science.tellu_derivative).
    #
    # IT REPLACES THE CCF LOOP for good (2026-10-27), which cross-correlated
    # the spectrum with TAPAS line masks and extrapolated each exponent from a
    # polynomial fitted on its first iterations. That extrapolation often
    # failed: on 100 HARPS exposures of HD 10700 (2026-10-15), 37% ended at
    # water = 0.0100 and 64% at dry = 0.0500, its lower bounds, where there is
    # always water above a telescope and an airmass of 1.08 asks for about
    # 1.1; and it degraded the spectrum (gradient scatter in the 593-595 nm
    # water band: raw 344.8, the loop 359.6, the derivative 304.1). It ran
    # only on the instruments that had not been switched to the derivative
    # (TELLUCLEAN_DERIVATIVE_WATER): every instrument now cleans the same way.
    #
    # Water is free per file, the water column changing too fast for a
    # nightly model; the dry absorbers are free too (fixed to the airmass they
    # impose a systematic 4% error, La Silla sitting above part of the column
    # TAPAS assumes), unless TELLUCLEAN_FORCE_AIRMASS fixes them to it.
    #
    # AND THE FIT SEES THE SPECTRUM DIVIDED BY THE TEMPLATE: that is APERO's
    # logic, and it is where the constraint comes from. template_vector is
    # ones without a template (the first, blind pass of lbl_wrap).
    sp_for_fit = sp_vector / template_vector
    # THE NOISE FOLLOWS THE DIVISIONS: photon noise of the observed flux,
    #   sqrt(F), carried through the division by the template here and by
    #   the transmission in the fit (tellu_derivative.objective)
    noise_for_fit = np.sqrt(np.clip(sp_vector, 0, None)) / template_vector
    try:
        expo_water, expo_others = tellu_derivative.fit_exponents(
            wave_trans, sp_for_fit, spl_water, spl_others, float(airmass),
            resolution=inst.params['APPROX_RESOLUTION'],
            free_dry=not force_airmass, noise=noise_for_fit)
    except Exception as exc:
        # nothing to fit on (no telluric window in this spectrum's range):
        #   the derivative's own fallback, marked as such
        expo_water = 0.35 + tellu_derivative.FALLBACK_MARK
        expo_others = float(airmass)
        log.warning('\ttelluric exponents not fitted ({0}): H2O {1:.4f}, '
                    'dry at the airmass {2:.4f}, template={3}'
                    ''.format(exc, expo_water, expo_others, template_flag))
    else:
        # LE JOURNAL DOIT DIRE SI C'EST UN REPLI. A quatre decimales, la
        # marque de 1e-7 est invisible, et une pose au repli s'affiche
        # "0.3500" exactement comme une pose qui converge vers 0.35: j'ai
        # compte zero repli sur un lot qui en avait treize (2026-10-16).
        # ONE LINE PER SPECTRUM, with the exponents applied (its header's
        #   KW_TAU_H2O and KW_TAU_OTHERS) and which pass it is (template)
        if tellu_derivative.is_fallback(expo_water):
            log.general('\tH2O exponent: FALLBACK {0:.4f} (the derivative '
                        'cost curve had nothing to go on), dry {1:.4f}, '
                        'template={2}'.format(expo_water, expo_others,
                                              template_flag))
        else:
            log.general('\ttelluric exponents by the derivative method: '
                        'H2O {0:.4f}, dry {1:.4f} (airmass {2:.4f}), '
                        'template={3}'.format(expo_water, expo_others,
                                              float(airmass), template_flag))
    # the transmission at these exponents, for the plot
    trans = get_abso_sp(wave_trans, expo_others, expo_water,
                        spl_others, spl_water, kwidth=kwidth, ex_gau=ex_gau,
                        dv_abso=dv_abso, wave0=wave0, wave1=wave1)
    # plot the corrected spectrum
    plot.tellu_corr_plot(inst, wave_vector, sp_vector, trans,
                         template_vector, template_flag, objname)
    # -------------------------------------------------------------------------
    # re-get wave grid for transmission (without trimming)
    #
    # AND SHIFT IT THE SAME WAY, for the same reason as above: this is the
    # transmission that is actually divided out of the science data, so a
    # grid left in the barycentric frame here would undo the fit done in the
    # observer frame. The loop above was corrected without this line once
    # (2026-10-15) and nothing changed in the written files, which is how the
    # omission showed up.
    wave_trans = np.array(wavemap)
    if berv_trans:
        wave_trans = mp.doppler_shift(wave_trans, berv_trans)
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

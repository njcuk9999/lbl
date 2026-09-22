#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General science functions and algorithms

Created on 2021-03-16

@author: cook
"""
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wget
from astropy import constants
from astropy import units as uu
from astropy.table import Table
from scipy import stats
from scipy.stats import pearsonr

from lbl.core import astro
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import fastmath
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
LblLowCCFSNR = base_classes.LblLowCCFSNR
log = io.log
InstrumentsType = select.InstrumentsType
# get speed of light
speed_of_light_ms = constants.c.value
speed_of_light_kms = constants.c.value / 1000.0
# the two percentiles of mp.estimate_sigma (sigma=1), as fractions computed
#   the way np.nanpercentile does (p / 100)
_P1 = (1 - (1 - mp.normal_fraction(1.0)) / 2) * 100
ESTIMATE_SIGMA_Q_HI = float(np.true_divide(_P1, np.float64(100)))
ESTIMATE_SIGMA_Q_LO = float(np.true_divide(100 - _P1, np.float64(100)))


# =============================================================================
# Define compute functions
# =============================================================================
def multi_wait(inst: InstrumentsType, reftable_exists: bool,
               reftable_file: str) -> bool:
    """
    Only used if iterations > 0 and total > 0 in inst.params

    In this case we need the first iteration to continue as normal and all
    other iterations to wait for the file to exist

    :param inst: Instrument instance
    :param reftable_exists: bool, whether the reftable currently exists
    :param reftable_file: str, the reftable filename

    :return: reftable_exists, bool, the updated flag whether reftable exists
    """
    # first case if reftable_exists is True then we have no problem in any
    #   mode
    if reftable_exists:
        return True
    # get the iteration and total parameters
    iteration = inst.params['ITERATION']
    total = inst.params['TOTAL']
    # get the maximum time out in seconds
    max_timeout = 300
    # if either iteration or total is below zero we are not in multi-iteration
    #   mode - so return the current value of reftable_exists
    if iteration < 0 or total < 0:
        return False
    # if iteration == 0 then we are in the first iteration and we can continue
    if iteration == 0:
        return False
    # start the count
    count = 0
    # else we have to wait for iteration zero to make the ref table
    while not os.path.exists(reftable_file):
        # every 10 iterations print a message
        if count % 10 == 0:
            msg = ('Iteration {0}/{1}: Waiting for reftable to be created by'
                   'iteration 0: Count = {2}')
            margs = [iteration, total, count]
            log.general(msg.format(*margs))
        # sleep for 1 second
        time.sleep(1)
        # add one to count
        count += 1
        # if count is greater than max_timeout we have timed out
        if count > max_timeout:
            # raise an error
            emsg = ('Timeout waiting for reftable to be created by iteration '
                    '0')
            raise LblException(emsg)
    # lets wait a little longer to make sure the file is written completely
    if count > 0:
        time.sleep(5)
    # return
    return True


def filter_science_files(inst: InstrumentsType,
                         science_files: List[str]) -> List[str]:
    """
    If we are in multi-mode filter the files by the iteration number and
    total number of iterations.

    If inst.params['ITERATION'] or inst.params['TOTAL'] is less than zero
    then we are not in multi-iteration mode and we return all the files.

    :param inst: Instrument instance
    :param science_files: list of strings, the science files to put into a
                          sub group
    :return:
    """
    # get the iteration and total parameters
    iteration = inst.params['ITERATION']
    total = inst.params['TOTAL']
    # if either iteration or total is below zero we are not in multi-iteration
    #   mode - so return all the files
    if iteration < 0 or total < 0:
        return science_files
    # return the subset of files
    return science_files[iteration::total]


def make_ref_dict(inst: InstrumentsType, reftable_file: str,
                  reftable_exists: bool, science_files: List[str],
                  mask_file: str, calib_dir: str) -> Dict[str, np.ndarray]:
    """
    Make the reference table dictionary

    :param inst: Instrument instance
    :param reftable_file: str, the ref table filename
    :param reftable_exists: bool, if True file exists and we load it
                            otherwise we create it
    :param science_files: list of absolute paths to science files
    :param mask_file: absolute path to mask file
    :param calib_dir: str, the calibration directory

    :return:
    """
    # deal with waiting for reftable to exist when in multi-iteration mode
    reftable_exists = multi_wait(inst, reftable_exists, reftable_file)
    # get parameter dictionary of constants
    params = inst.params
    # storage for ref dictionary
    ref_dict = dict()
    # load the mask
    mask_table = inst.load_mask(mask_file)
    # -------------------------------------------------------------------------
    # deal with loading from file
    if reftable_exists:
        # log writing
        log.general('Reading existing ref table {0}'.format(reftable_file))
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
        wavegrid = inst.get_sample_wave_grid(calib_dir, science_files[0])
        # storage for vectors
        order, wave_start, wave_end, weight_line, xpix = [], [], [], [], []
        line_snr, line_depth, local_flux = [], [], []
        # loop around orders
        for order_num in range(wavegrid.shape[0]):
            # get the min max wavelengths for this order
            min_wave = np.nanmin(wavegrid[order_num])
            max_wave = np.nanmax(wavegrid[order_num])
            # build a mask for mask lines in this order
            good = mask_table['ll_mask_s'] > min_wave
            good &= mask_table['ll_mask_s'] < max_wave
            # if we pass a 'full' mask, then we only keep local maxima
            # only valid for science frames
            if params['DATA_TYPE'] == 'SCIENCE':
                good &= mask_table['w_mask'] < 0

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
                local_flux += list(mask_table['value'][good][:-1])
        # make xpix a numpy array
        xpix = np.array(xpix)
        # add to reference dictionary
        ref_dict['ORDER'] = np.array(order)
        ref_dict['WAVE_START'] = np.array(wave_start)
        ref_dict['WAVE_END'] = np.array(wave_end)
        ref_dict['WEIGHT_LINE'] = np.array(weight_line)
        ref_dict['XPIX'] = xpix
        with warnings.catch_warnings(record=True) as _:
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
    # deal with empty ref table (shouldn't happen)
    if len(ref_dict['ORDER']) == 0:
        emsg = 'Reference table is empty - something went wrong'
        raise LblException(emsg)
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
                    systemic_vel: float, models_dir: str
                    ) -> Dict[str, mp.IUVSpline]:
    """
    Calculate all the template splines (for later use)

    :param inst: Instrument instance
    :param template_file: str, the absolute path to the template file
    :param systemic_vel: float, the systemic velocity
    :param models_dir: str, the absolute path to the models directory

    :return:
    """
    # log that we are producing all template splines
    msg = 'Defining all the template splines required later'
    log.general(msg)
    # get the pixel hp_width [needs to be in m/s]
    hp_width = inst.params['HP_WIDTH'] * 1000
    # load the template
    margs = [template_file]
    msg = 'Loading template file {}'
    log.general(msg.format(*margs))
    template_table = inst.load_template(template_file)
    # get properties from template table
    twave = np.array(template_table['wavelength'])
    # -------------------------------------------------------------------------
    if not inst.params['USE_SAVGOL_TEMPLATE']:
        tflux = np.array(template_table['flux'])
        # deal with the odd/even templates
        if 'flux_odd' in template_table.colnames:
            # this captures the odd/even asymetry in resolution along orders
            # we have 3 templates, one averaing the odd, one the even and one
            # having all the flux from the entire ordes
            tflux_odd = np.array(template_table['flux_odd'])
            tflux_even = np.array(template_table['flux_even'])
        else:
            tflux_odd, tflux_even = np.array([]), np.array([])
    else:
        tflux = np.array(template_table['flux_savgol_d0'])
        tflux_odd = np.array(template_table['flux_odd_savgol_d0'])
        tflux_even = np.array(template_table['flux_even_savgol_d0'])
    # -------------------------------------------------------------------------
    # some masking to avoid errors later
    # if we have the n_valid flag in the table, then we must have at least
    # 1/2 of points valid
    if 'n_valid' in template_table.colnames:
        msg = 'We reject points with n_valid < 1/2 max(n_valid)'
        log.general(msg)
        # these come out of the APERO pipeline
        # TODO -> get n_valid in the template code
        bad = template_table['n_valid'] < np.max(template_table['n_valid']) / 2
        tflux[bad] = np.nan

    # flag completely spurious data in the template. Negative values
    # and values > 10x the median are rejected
    flag_spurious_sig = inst.params['COMPUTE_FLAG_SPURIOUS_SIG']
    bad = (tflux < 0) | (tflux / np.nanmedian(tflux) > flag_spurious_sig)
    tflux[bad] = np.nan
    # -------------------------------------------------------------------------
    # Deal with Rotational broadening
    # -------------------------------------------------------------------------
    if isinstance(inst.params['ROTBROAD'], tuple):
        # Apply rotational broadening if inst.params['ROTBROAD'] is
        # passed as an argument.
        epsilon = inst.params['ROTBROAD'][0]
        # TODO check that its 0-1 for epsilon
        # TODO check that its 1-100 for vsini to avoid m/s rather than km/s
        vsini = inst.params['ROTBROAD'][1]
        # Calculate the step in the wavelength grid
        step = (twave / np.gradient(twave))
        sample_sampling = np.nanmedian(step / constants.c.to(uu.km / uu.s).value)
        # calculate the pixel value for the broadening (the delta function
        #     position)
        npix = np.ceil(vsini / sample_sampling).astype(int)
        # get the wave grid of the template that we are going to use
        twave_tmp = np.linspace(twave[len(twave) // 2 - npix],
                                twave[len(twave) // 2 + npix],
                                2 * npix + 1)
        # set up the delta value and populate the delta function at the
        # point of interest
        delta_tmp = np.zeros_like(twave_tmp)
        delta_tmp[npix] = 1
        # run the rotation kernel
        rot_kernel = mp.rot_broad(twave_tmp, delta_tmp, epsilon, vsini)
        # display a message that we have rotationally broadened the spectrum
        msg = 'Rotational broadening epsilon = {}, vsini = {} km/s'
        log.general(msg.format(*inst.params['ROTBROAD']))
        # push the rotational broadening onto the flux using the rotational
        #  broadening kernel
        tflux2 = np.zeros_like(tflux)
        for ioff in range(len(delta_tmp)):
            tflux2 += np.roll(tflux, ioff - npix) * rot_kernel[ioff]
        # set this to the flux we would have had from before
        tflux = tflux2
    # -------------------------------------------------------------------------
    # get the gradient of the log of the wave
    grad_log_wave = np.gradient(np.log(twave))
    # work out the glw_c (grad_log_wave / speed_of_light_ms)
    glw_c = grad_log_wave * speed_of_light_ms

    if not inst.params['USE_SAVGOL_TEMPLATE']:
        # get the derivative of the flux
        dflux = np.gradient(tflux) / glw_c
        # get the 2nd derivative of the flux
        d2flux = np.gradient(dflux) / glw_c
        # get the 3rd derivative of the flux
        d3flux = np.gradient(d2flux) / glw_c
        # deal with the odd/even templates
        if 'flux_odd' in template_table.colnames:
            # get the derivative of the flux
            dflux_odd = np.gradient(tflux_odd) / glw_c
            dflux_even = np.gradient(tflux_even) / glw_c
            # get the 2nd derivative of the flux
            d2flux_odd = np.gradient(dflux_odd) / glw_c
            d2flux_even = np.gradient(dflux_even) / glw_c
            # get the 3rd derivative of the flux
            d3flux_odd = np.gradient(d2flux_odd) / glw_c
            d3flux_even = np.gradient(d2flux_even) / glw_c
        else:
            dflux_odd, dflux_even = np.array([]), np.array([])
            d2flux_odd, d2flux_even = np.array([]), np.array([])
            d3flux_odd, d3flux_even = np.array([]), np.array([])
    else:
        def savgol_deriv(ttable: Table, flux_key: str, deriv: int = 0):
            """
            Temporary function to get derivatives from savgol columns
            """
            key = '{0}_savgol_d{1}'.format(flux_key, deriv)
            dd = np.array(ttable[key]) / (glw_c ** (deriv))
            return dd
        # get the derivative of the flux
        dflux = savgol_deriv(template_table, 'flux', 1)
        # get the 2nd derivative of the flux
        d2flux = savgol_deriv(template_table, 'flux', 2)
        # get the 3rd derivative of the flux
        d3flux = savgol_deriv(template_table, 'flux', 3)
        # get the derivative of the flux
        dflux_odd = savgol_deriv(template_table, 'flux_odd', 1)
        # get the 2nd derivative of the flux
        d2flux_odd = savgol_deriv(template_table, 'flux_odd', 2)
        # get the 3rd derivative of the flux
        d3flux_odd = savgol_deriv(template_table, 'flux_odd', 3)
        # get the derivative of the flux
        dflux_even = savgol_deriv(template_table, 'flux_even', 1)
        # get the 2nd derivative of the flux
        d2flux_even = savgol_deriv(template_table, 'flux_even', 2)
        # get the 3rd derivative of the flux
        d3flux_even = savgol_deriv(template_table, 'flux_even', 3)
    # -------------------------------------------------------------------------
    # we create the spline of the template to be used everywhere later
    valid = np.isfinite(tflux) & np.isfinite(dflux)
    valid &= np.isfinite(d2flux) & np.isfinite(d3flux)
    # -------------------------------------------------------------------------
    # doppler shift wave grid with respect to systemic velocity
    ntwave = mp.doppler_shift(twave[valid], systemic_vel)
    # -------------------------------------------------------------------------
    # storage for splines
    sps = dict()
    # template removed from its systemic velocity, the spline is redefined
    #   for good
    k_order = 2
    # flux, 1st and 2nd derivatives
    sps['spline'] = mp.iuv_spline(ntwave, tflux[valid], k=k_order, ext=1)
    # deal with the odd/even templates
    if 'flux_odd' in template_table.colnames:
        # keeping track of odd/even derivatives
        sps['spline_odd'] = mp.iuv_spline(ntwave, tflux_odd[valid],
                                          k=k_order, ext=1)
        sps['spline_even'] = mp.iuv_spline(ntwave, tflux_even[valid],
                                           k=k_order, ext=1)
        sps['dspline_odd'] = mp.iuv_spline(ntwave, dflux_odd[valid],
                                           k=k_order, ext=1)
        sps['dspline_even'] = mp.iuv_spline(ntwave, dflux_even[valid],
                                            k=k_order, ext=1)
        sps['d2spline_odd'] = mp.iuv_spline(ntwave, d2flux_odd[valid],
                                            k=k_order, ext=1)
        sps['d2spline_even'] = mp.iuv_spline(ntwave, d2flux_even[valid],
                                             k=k_order, ext=1)
        sps['d3spline_odd'] = mp.iuv_spline(ntwave, d3flux_odd[valid],
                                            k=k_order, ext=1)
        sps['d3spline_even'] = mp.iuv_spline(ntwave, d3flux_even[valid],
                                             k=k_order, ext=1)
    else:
        sps['dspline'] = mp.iuv_spline(ntwave, dflux[valid],
                                       k=k_order, ext=1)
        sps['d2spline'] = mp.iuv_spline(ntwave, d2flux[valid],
                                        k=k_order, ext=1)
        sps['d3spline'] = mp.iuv_spline(ntwave, d3flux[valid],
                                        k=k_order, ext=1)

    # deal with residual projection tables
    resproj_keys = []
    if isinstance(inst.params['RESPROJ_TABLES'], dict):
        resproj_keys = list(inst.params['RESPROJ_TABLES'].keys())
        # loop around table
        for key in inst.params['RESPROJ_TABLES'].keys():
            # log that we are constructing the spline
            msg = 'We construct the spline for the "{}" response'
            log.general(msg.format(key))
            # construct the path for the spline
            spline_file = inst.params['RESPROJ_TABLES'][key]
            spline_path = os.path.join(models_dir, spline_file)
            # make sure file exists
            if not os.path.exists(spline_path):
                emsg = 'RESPROJ Table {0} does not exist'
                raise LblException(emsg.format(key))
            # load the table
            tbl_res = Table.read(spline_path)
            # give an error if the table does not have the correct columns
            if 'wavelength' not in tbl_res.colnames:
                emsg = 'RESPROJ Table {0} must have a "wavelength" column'
                raise LblException(emsg.format(key))
            if 'fractional_gradient' not in tbl_res.colnames:
                emsg = ('RESPROJ Table {0} must have a "fractional_gradient" '
                        ' column')
                raise LblException(emsg.format(key))
            # read the wavelength and fraction gradient columns
            wave_res = np.array(tbl_res['wavelength'])
            sp_res = np.array(tbl_res['fractional_gradient'])
            # make sure we don't have nans for the spline
            valid_res = np.isfinite(sp_res)

            tmp_spline = mp.iuv_spline(wave_res[valid_res], sp_res[valid_res],
                                       k=1, ext=1)
            # create a smooth edge condition for points past edges, otherwise
            # the k_order spline may diverge. This is done with k=1 to create
            # a linear interpolation as an edge condition
            sp_res[~valid_res] = tmp_spline(wave_res[~valid_res])

            # spline and add to spline dictionary. No need for masking so we
            # can use a k value larger than 1
            sps[key] = mp.iuv_spline(wave_res, sp_res, k=k_order, ext=1)
    # -------------------------------------------------------------------------
    # we create a mask to know if the splined point  is valid
    tmask = np.isfinite(tflux).astype(float)
    ntwave1 = mp.doppler_shift(twave, systemic_vel)
    sps['spline_mask'] = mp.iuv_spline(ntwave1, tmask, k=1, ext=1)
    # deal with odd/even templates
    if 'flux_odd' in template_table.colnames:
        tmask_odd = np.isfinite(tflux_odd).astype(float)
        tmask_even = np.isfinite(tflux_even).astype(float)
        sps['spline_mask_odd'] = mp.iuv_spline(ntwave1, tmask_odd,
                                               k=1, ext=1)
        sps['spline_mask_even'] = mp.iuv_spline(ntwave1, tmask_even,
                                                k=1, ext=1)
    # -------------------------------------------------------------------------
    # evaluate the splines on a window of knots around the requested points
    #   (same values, without FITPACK's linear search along all the knots)
    for key in list(sps.keys()):
        if isinstance(sps[key], mp.IUVSpline):
            sps[key] = mp.KnotWindowSpline(sps[key])
    # groups of splines evaluated together at the same wavelengths in
    #   compute_rv: the template and its derivatives (same knots), the masks
    #   and the residual projection splines
    if 'flux_odd' in template_table.colnames:
        for suffix in ['_odd', '_even']:
            names = ['spline', 'dspline', 'd2spline', 'd3spline']
            sps['group' + suffix] = mp.SplineGroup(
                [sps[name + suffix] for name in names])
    else:
        names = ['spline', 'dspline', 'd2spline', 'd3spline']
        sps['group'] = mp.SplineGroup([sps[name] for name in names])
    for key in list(sps.keys()):
        if key.startswith('spline_mask') or key in resproj_keys:
            sps['group_' + key] = mp.SplineGroup([sps[key]])
    # -------------------------------------------------------------------------
    # return splines
    return sps


def get_systemic_vel_props(inst: InstrumentsType, template_file: str,
                           mask_file: str) -> Dict[str, Any]:
    # set the function name
    func_name = __NAME__ + '.get_systemic_vel_props()'
    # print progress
    log.general('Calculating systemic velocity properties')
    # output return in a dictionary
    props = dict()
    # check if there is an extension with the RV for all epochs
    # noinspection PyBroadException
    props['SCI_TABLE'] = io.load_table(template_file, extname='SCI_TABLE',
                                       required=False)
    if props['SCI_TABLE'] is None:
        msg = 'No RV table found'
        log.general(msg)

    # get the systemic velocity for mask
    props['MASK_SYS_VEL'] = inst.get_mask_systemic_vel(mask_file)
    # deal with non science files
    if inst.params['DATA_TYPE'] != 'SCIENCE':
        # add values to props
        props['VSYS'] = 0
        props['FWHM'] = np.nan
        props['CONTRAST'] = np.nan
        props['SNR'] = np.nan
        props['EARS'] = np.nan
        props['EXPO'] = np.nan
        return props
    # get the object name
    sci_objname = inst.params['OBJECT_SCIENCE']
    template_objname = inst.params['OBJECT_COMPARISON']
    rv_min = inst.params['ROUGH_CCF_MIN_RV']
    rv_max = inst.params['ROUGH_CCF_MAX_RV']
    rv_step = inst.params['ROUGH_CCF_STEP_RV']
    rv_filter_size = inst.params['ROUGH_CCF_FILTER_SIZE']
    # ---------------------------------------------------------------------
    # load the correct mask file
    mask_table = inst.load_mask(mask_file)
    # ---------------------------------------------------------------------
    # get ccf regions for instrument
    ccf_regions = dict()
    for ccf_region in inst.params['CCF_CLEAN_BANDS']:
        ccf_regions[ccf_region] = astro.ccf_regions[ccf_region]
    # ---------------------------------------------------------------------
    # get the template file for the science object
    sci_template_file = template_file.replace(template_objname, sci_objname)
    # load the science template
    msg = 'Loading science template file {}'
    margs = [sci_template_file]
    log.general(msg.format(*margs))
    sci_template_table = inst.load_template(sci_template_file)
    # ---------------------------------------------------------------------
    # mask of best regions (expected to be clean)
    keep = np.zeros(len(mask_table)).astype(bool)
    # loop around regions and create a mask of the mask_table
    for ccf_region in ccf_regions:
        # get the min and max wavelengths for this region
        wavemin, wavemax = ccf_regions[ccf_region]
        # mask the mask_table
        good = mask_table['ll_mask_s'] > wavemin
        good &= mask_table['ll_mask_s'] < wavemax
        # we are using the FULL mask therefore we keep only local
        # maxima
        good &= mask_table['w_mask'] > 0

        # apply mask to keep
        keep[good] = True
    # mask the mask table
    mask_table = mask_table[keep]
    # ---------------------------------------------------------------------
    # get the wavelength from the template
    wavegrid = np.array(sci_template_table['wavelength'])
    # get the flux from the template
    fluxgrid = np.array(sci_template_table['flux'])
    # fluxgrid = medfilt(fluxgrid, 3)
    # ---------------------------------------------------------------------
    # spline the science temmplate
    good = np.isfinite(fluxgrid)
    # Use k=1 to avoid ringing at edges if 'good' has holes.
    sps = mp.iuv_spline(wavegrid[good], fluxgrid[good], k=1, ext=1)
    # ---------------------------------------------------------------------
    # define the ccf dv grid
    dv = np.arange(rv_min, rv_max + rv_step, rv_step)
    # storage for the ccf
    ccf_vector = np.zeros_like(dv)
    # compute the ccf
    for it in range(len(dv)):
        # shift the mask by dv for this iteration
        pos = mp.doppler_shift(mask_table['ll_mask_s'], -dv[it])
        # calculate the ccf for this iteration
        ccf_vector[it] = mp.nansum(mask_table['depth'] * sps(pos))
    # CCF can be normalized to its median as we have only used
    # features in absorption rather than the 'full'
    ccf_vector /= np.nanmedian(ccf_vector)

    # ---------------------------------------------------------------------
    # construct a guess at the ccf fit
    # default FWHM is 6 km/s as this is about the typical width of slowly
    # rotating M dwarfs
    coeffs = [dv[np.argmin(ccf_vector)], 6000,
              1 - np.nanmin(ccf_vector), 3, 2]
    lowf = None
    # do this twice for better fit
    for iteration in range(2):
        # get the estimate of the fit
        estimate = mp.gauss_fit_e(dv, *coeffs)
        # find the continuum by subtraing the estimate from the ccf
        lowf = mp.lowpassfilter(ccf_vector - estimate,
                                int(rv_filter_size / rv_step))
        # try to compute curve fit
        try:
            # we were idiots, the lowf should be subtracted, not divided
            # ... there are days like that.
            coeffs, _ = mp.curve_fit(mp.gauss_fit_e, dv, ccf_vector - lowf,
                                     p0=coeffs)
        except base_classes.LblCurveFitException as e:
            emsg = 'CurveFit exception in rough CCF'
            emsg += '\n\tIteration = {0}'.format(iteration)
            emsg += '\n\tP0 = {0}'.format(e.p0)
            emsg += '\n\tFunction = {0}'.format(func_name)
            emsg += '\n\tError: {0}'.format(e.error)
            raise LblException(emsg)
    # ---------------------------------------------------------------------
    # work out rms
    gfit = mp.gauss_fit_e(dv, *coeffs)
    rms = mp.estimate_sigma(ccf_vector - lowf - gfit)
    # ---------------------------------------------------------------------
    # print out values
    msg = 'Rough CCF fit:'
    msg += '\n\tVsys = {0:.4f}'.format(coeffs[0])
    msg += '\n\tFWHM = {0:.4f}'.format(coeffs[1])
    msg += '\n\tCONTRAST = {0:.4f}'.format(coeffs[2])
    msg += '\n\tSNR = {0:.4f}'.format(coeffs[2] / rms)
    msg += '\n\tEARS = {0:.4f}'.format(coeffs[3])
    msg += '\n\tEXPO = {0:.4f}'.format(coeffs[4])
    # ---------------------------------------------------------------------
    # add values to props
    props['VSYS'] = coeffs[0]
    props['FWHM'] = coeffs[1]
    props['CONTRAST'] = coeffs[2]
    props['SNR'] = coeffs[2] / rms
    props['EARS'] = coeffs[3]
    props['EXPO'] = coeffs[4]
    # ---------------------------------------------------------------------
    # plot of the CCF and its residual
    plot.compute_plot_sysvel(inst, dv, ccf_vector - lowf, gfit, props)
    # ---------------------------------------------------------------------
    # return the props
    return props


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

    :param wave0: float, first wavelength element
    :param wave1: float, second wavelength element
    :param dv_grid: float, grid size in m/s
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
    len_magic = int(np.floor(logwaveratio * speed_of_light_ms / dv_grid))
    wave1 = np.exp(len_magic * dv_grid / speed_of_light_ms) * wave0
    # redefining wave1 to have a round number of velocity bins
    logwaveratio = np.log(wave1 / wave0)

    # get the positions for "magic length"
    plen_magic = np.arange(len_magic)
    # define the magic grid to use in ccf
    magic_grid = np.exp((plen_magic / len_magic) * logwaveratio) * wave0
    # return the magic grid
    return magic_grid


def rough_ccf_rv(inst: InstrumentsType, wavegrid: np.ndarray,
                 sci_data: np.ndarray, wave_mask: np.ndarray,
                 weight_line: np.ndarray, kind: str) -> Tuple[float, float]:
    """
    Perform a rough CCF calculation of the science data

    :param inst: Instrument instance
    :param wavegrid: wave grid (shape shape as sci_data)
    :param sci_data: spectrum
    :param wave_mask: list of wavelength centers of mask lines
    :param weight_line: list of weights for each mask line
    :param kind: the kind of file we are doing the ccf on (for logging)

    :return: tuple, 1. systemic velocity estimate, 2. ccf ewidth
    """
    func_name = __NAME__ + '.rough_ccf_rv()'
    # -------------------------------------------------------------------------
    # get parameters
    rv_min = inst.params['ROUGH_CCF_MIN_RV']
    rv_max = inst.params['ROUGH_CCF_MAX_RV']
    rv_ewid_guess = inst.params['ROUGH_CCF_EWIDTH_GUESS']
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
    # set the data below zero to NaNs
    sci_data2[sci_data2 < 0] = np.nan
    # set the data above 5 sigma to NaNs
    sci_data2[sci_data2 > 5 * np.nanmedian(sci_data2)] = np.nan

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
    # high-pass of CCF expressed in pixels, not km/s
    ccf_hp_scale = inst.params['COMPUTE_CCF_HP_SCALE']
    ccf_hp_width_pix = int(ccf_hp_scale / (grid_step / 1000))
    ccf_vector -= mp.lowpassfilter(ccf_vector, ccf_hp_width_pix)

    # -------------------------------------------------------------------------
    # fit the CCF
    # -------------------------------------------------------------------------
    # get the position of the maximum CCF
    ccfmin = np.argmin(ccf_vector)
    # guess the amplitude (minus the dc level)
    ccf_dc = mp.nanmedian(ccf_vector)
    ccf_amp = ccf_vector[ccfmin] - ccf_dc
    # construct a guess of a guassian fit to the CCF
    #    guess[0]: float, the mean position
    #    guess[1]: float, the ewidth
    #    guess[2]: float, the amplitude
    #    guess[3]: float, the dc level
    #    guess[4]: float, the float (x-x0) * slope
    guess = [dvgrid[ccfmin], rv_ewid_guess, ccf_amp, ccf_dc, 0.0]
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
    margs = [kind, mp.fwhm_value() * ccf_ewidth]
    log.general(msg.format(*margs))
    msg = '\t\tCCF contrast = {0:.2f}'
    margs = [gcoeffs[2]]
    log.general(msg.format(*margs))

    ccf_snr = -gcoeffs[2] / mp.estimate_sigma(ccf_vector)
    msg = '\t\tCCF SNR = {0:.2f}'
    margs = [ccf_snr]
    log.general(msg.format(*margs))
    # -------------------------------------------------------------------------
    # debug plot
    # -------------------------------------------------------------------------
    plot.compute_plot_ccf(inst, dvgrid, ccf_vector, ccf_fit, gcoeffs)
    # -------------------------------------------------------------------------
    # CCF SNR is too low for this to valid - raise an exception
    if ccf_snr < inst.params['CCF_SNR_MIN']:
        emsg = 'CCF SNR must be > {0:.2f}, it is {1:.2f}'
        eargs = [inst.params['CCF_SNR_MIN'], ccf_snr]
        raise LblLowCCFSNR(emsg.format(*eargs))
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
                         model: np.ndarray,
                         noise_sampling_width: float,
                         npoints_orders: Optional[List[int]] = None,
                         nstep: int = 4) -> np.ndarray:
    """
    Estimate the noise on spectrum given the model

    :param spectrum: np.ndarray, the spectrum
    :param wavegrid: np.ndarray, the wave grid for the spectrum
    :param model: np.ndarray, the model
    :param noise_sampling_width: float, the width of the window used to sample
                                   the noise.
    :param npoints_orders: optional, the number of points of the sliding
                           window for each order (get_velo_scale of each
                           order of wavegrid), if already known
    :param nstep: int, number of windows per window width (the windows
                  are npoints // nstep apart; 4 is the original sampling)

    :return: np.ndarray, the rms vector for this spectrum give the model
    """
    # storage for output rms
    rms = np.zeros_like(spectrum)
    # loop around each order and estimate noise model
    for order_num in range(spectrum.shape[0]):
        # get the wavelength for this order
        waveord = wavegrid[order_num]
        # deal with all-nan waveord
        if np.sum(np.isfinite(waveord)) < 5:
            rms[order_num] = np.full(model.shape[1], fill_value=np.nan)
            continue
        # calculate the number of points for the sliding error rms
        if npoints_orders is None:
            npoints = get_velo_scale(waveord, noise_sampling_width)
        else:
            npoints = npoints_orders[order_num]
        # get the residuals between science and model
        residuals = spectrum[order_num] - model[order_num]
        # robust sigma in boxes of npoints pixels, every npoints // 4 pixels
        #   (box centers = indices). Boxes with <= 50% valid points (compared
        #   to npoints, also at the edges) and zero sigmas are NaN.
        indices, sigma = fastmath.noise_model_windows(
            np.ascontiguousarray(residuals, dtype=float), int(npoints),
            ESTIMATE_SIGMA_Q_HI, ESTIMATE_SIGMA_Q_LO, int(nstep))
        # mask all NaN values
        good = np.isfinite(sigma)
        # if we have enough points calculate the rms
        if np.sum(good) > 2:
            # get the spline across all indices
            rms_values = mp.iuv_spline_eval(indices[good], sigma[good],
                                            np.arange(model.shape[1]),
                                            k=1, ext=1)
            # apply the spline to the model positions
            rms[order_num] = rms_values
        # else we don't have a noise model
        else:
            # we fill the rms with NaNs for each pixel
            rms[order_num] = np.full(model.shape[1], fill_value=np.nan)
    rms[rms == 0] = np.nan
    # return rms
    return rms


def _spline_group(splines: Dict[str, Any], key: str,
                  members: List[Any]) -> mp.SplineGroup:
    """
    The spline group splines[key] (made by spline_template), or a new group
    of the member splines if the dictionary does not have it
    """
    if key in splines:
        return splines[key]
    return mp.SplineGroup(members)


def _native(array: np.ndarray) -> np.ndarray:
    """
    Native byte order version of an array (same dtype precision, same
    values); FITS data are big-endian. No copy if already native.
    """
    array = np.asarray(array)
    return np.ascontiguousarray(array,
                                dtype=array.dtype.newbyteorder('='))


def _f64(array: np.ndarray) -> np.ndarray:
    """
    Native-endian, C-contiguous float64 version of an array (FITS data are
    big-endian, which numba does not take). No copy if already the case.
    """
    return np.ascontiguousarray(array, dtype=np.float64)


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
               sci_data: np.ndarray, sci_hdr: io.LBLHeader,
               splines: Dict[str, Any], ref_table: Dict[str, Any],
               blaze: np.ndarray, systemic_props: Dict[str, Any],
               systemic_all: np.ndarray,
               mjdate_all: np.ndarray, model_velocity: float = np.inf,
               science_file: str = '', mask_file: str = ''
               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute the RV using a line-by-line analysis

    :param inst: Instrument instance
    :param sci_iteration: int, the science files iteration number
    :param sci_data: np.ndarray, the science data
    :param sci_hdr: io.LBLHeader, the science header
    :param splines: Dict, the dictionary of template splines
    :param ref_table: Dict, the reference table
    :param blaze: np.ndarray, the blaze to correct the science data
    :param systemic_props: dictionary containing the systemic velocity
                           properties (calculated from the science template)
    :param systemic_all: np.ndarray, the systemic velocity for all science files
    :param mjdate_all: np.ndarray, the mid exposure time in mjd for all
                       science files (filled in function for this iteration)
    :param model_velocity: float, inf if not set but if set doesn't recalculate
                           model velocity
    :param science_file: str, the science file name (required for some
                         instruments)
    :param mask_file: str, the mask file name (required for science data types)

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
    # object_science = inst.params['OBJECT_SCIENCE']
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
    # flag that we have residual projection tables
    resproj_flag = isinstance(inst.params['RESPROJ_TABLES'], dict)
    # get the size of running window sample = noise
    noise_sampling_width = inst.params['NOISE_SAMPLING_WIDTH']
    # -------------------------------------------------------------------------
    # deal with max number of iterations higher than computer_rv_n_iters
    if max_good_num_iters > compute_rv_n_iters:
        emsg = ('Max number of good iterations '
                '(COMPUTE_RV_MAX_N_GOOD_ITERS={0}) is greater than the '
                'total number of iterations (COMPUTER_RV_N_ITERATIONS={1})')
        eargs = [max_good_num_iters, compute_rv_n_iters]
        raise LblException(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # deal with noise model
    if not use_noise_model:
        # TODO: get a noise model
        # rms = get_noise_model(science_files)
        rms = np.zeros_like(sci_data)
    else:
        rms = np.sqrt(np.abs(sci_data) + readout_noise ** 2)
    # -------------------------------------------------------------------------
    # work on a native byte order copy of the science data (FITS data are
    #   big-endian): same values, no conversion in every operation below
    sci_data = _native(sci_data)
    # copy science data
    sci_data0 = np.array(sci_data)
    # get the mid exposure time in MJD
    mjdate = sci_hdr.get_hkey(mid_exp_time_key, dtype=float)
    # -------------------------------------------------------------------------
    # get the wave grid for this science data
    # -------------------------------------------------------------------------
    # instrument specific wave solution --> use instrument method
    wavegrid = inst.get_wave_solution(data=sci_data, header=sci_hdr,
                                      science_filename=science_file)
    wavegrid = _native(wavegrid)
    # window size of the noise model in each order (depends on wavegrid only)
    noise_npoints = []
    for order_num in range(wavegrid.shape[0]):
        if np.sum(np.isfinite(wavegrid[order_num])) < 5:
            noise_npoints.append(None)
        else:
            noise_npoints.append(get_velo_scale(wavegrid[order_num],
                                                noise_sampling_width))
    # loop around orders
    # for order_num in range(sci_data.shape[0]):
    #    # work out the velocity scale
    #    width = get_velo_scale(wavegrid[order_num], hp_width)
    #    # we high-pass on a scale of ~101 pixels in the e2ds
    #    sci_data[order_num] -= mp.lowpassfilter(sci_data[order_num],
    #                                            width=width)
    # -------------------------------------------------------------------------
    # get BERV
    # -------------------------------------------------------------------------
    # instrument specific berv --> use instrument method
    berv = inst.get_berv(sci_hdr)
    inst.params['BERV'] = berv

    width = np.zeros(sci_data.shape[0], dtype=int)
    for order_num in range(sci_data.shape[0]):  # TODO change back to hp_width
        # within each order, we determine the median width of lines and
        # we define the filtering size as 5x that value. This avoids the
        # hp_width parameter. We don't expect that derivatives of lines
        # have a significant impact on that scale length
        lines_in_order = np.where((ref_table['ORDER'] == order_num))[0]

        if len(lines_in_order) < 3:
            # this order is bad if you have less than 3 lines
            continue
        starts = ref_table['WAVE_END'][lines_in_order]
        ends = ref_table['WAVE_START'][lines_in_order]

        med_line_width = np.nanmedian(starts / ends - 1) * speed_of_light_ms
        hp_width = 5 * med_line_width
        width[order_num] = get_velo_scale(wavegrid[order_num], hp_width)

    # -------------------------------------------------------------------------
    # Systemic velocity estimate
    # -------------------------------------------------------------------------
    if inst.params['DATA_TYPE'] == 'SCIENCE':
        # get the velocity estimate from the template (if possible)
        velo_estimate = get_velo_estimate(systemic_props, mjdate)
        # deal with first estimate of RV / CCF equivalent width
        vtot = velo_estimate + systemic_props['VSYS']
        sys_rv, ccf_fwhm = berv - vtot, systemic_props['FWHM']
        ccf_ewidth = ccf_fwhm / mp.fwhm_value()
    else:
        # for FP files
        sys_rv, ccf_ewidth = 0, 0

    # if reset_rv:
    #     # if we are not using calibration file
    #     if inst.params['DATA_TYPE'] == 'SCIENCE':
    #         # calculate the rough CCF RV estimate
    #         mask_table = inst.load_mask(mask_file)
    #         # get the mask parameters`
    #         wave_mask = np.array(mask_table['ll_mask_s'], dtype=float)
    #         ccf_weight = np.array(mask_table['w_mask'], dtype=float)
    #
    #         sys_rv, ewidth = rough_ccf_rv(inst, wavegrid, sci_data,
    #                                       wave_mask, ccf_weight,
    #                                       kind='science')
    #         # if ccf width is not set then set it and log message
    #         if ccf_ewidth is None:
    #             ccf_ewidth = float(ewidth)
    #             # log ccf ewidth
    #             msg = '\t\tCCF e-width = {0:.2f} m/s'
    #             margs = [ccf_ewidth]
    #             log.general(msg.format(*margs))
    #     else:
    #         sys_rv, ccf_ewidth = 0, 0
    # # for FP files
    # else:
    #     # use the systemic velocity from closest date
    #     closest = np.argmin(mjdate - mjdate_all)
    #     # get the closest system rv to this observation
    #     sys_rv = systemic_all[closest] + berv
    #     # log the systemic velocity and the berv
    #     msg = '\tUsing Systemic velocity: {0:.4f} m/s  BERV: {1:.4f} m/s'
    #     margs = [-systemic_all[closest], -berv]
    #     log.general(msg.format(*margs))
    #     # log using the sys_Rv
    #     msg = '\tSystemic rv + berv={0:.4f} m/s from MJD={1}'
    #     margs = [-sys_rv, mjdate_all[closest]]
    #     log.general(msg.format(*margs))

    # -------------------------------------------------------------------------
    # iteration loop
    # -------------------------------------------------------------------------
    # a keep mask - for keep good mask lines
    mask_keep = np.ones_like(ref_table['ORDER'], dtype=bool)
    # lines whose MEANXPIX / MEANBLAZE and RMSRATIO / NPIXLINE / CHI2 values
    #   in ref_table are set by this file (the others keep the values of a
    #   previous file)
    lines_xpix_set = np.zeros(len(ref_table['ORDER']), dtype=bool)
    lines_stats_set = np.zeros(len(ref_table['ORDER']), dtype=bool)
    # line indices in each order and line edges (for the line-by-line loop)
    line_orders = np.asarray(ref_table['ORDER'], dtype=np.int64)
    order_lines = [np.where(line_orders == order_num)[0]
                   for order_num in range(sci_data.shape[0])]
    line_wave_start = _f64(ref_table['WAVE_START'])
    line_wave_end = _f64(ref_table['WAVE_END'])
    # store number of iterations required to converge
    num_to_converge = 0
    # set up models to spline onto
    model = np.zeros_like(sci_data)
    model0 = np.zeros_like(sci_data)
    dmodel = np.zeros_like(sci_data)
    d2model = np.zeros_like(sci_data)
    d3model = np.zeros_like(sci_data)

    # deal with the projection models
    proj_model = dict()
    # only need to fill this if we have residual projection tables
    if resproj_flag:
        # loop around residual projection tables
        for key in inst.params['RESPROJ_TABLES']:
            # each projection model has a model, proj and sproj
            #  we fill them with empty arrays to start with
            proj_model[key] = dict()
            wave_len = len(ref_table['WAVE_START'])
            proj_model[key]['model'] = np.zeros_like(sci_data)
            proj_model[key]['proj'] = np.full(wave_len, np.nan)
            proj_model[key]['sproj'] = np.full(wave_len, np.nan)

    # correction of the model from the low-passed ratio of science to model
    ratio = np.zeros_like(sci_data)
    b_ratio = np.zeros_like(sci_data)
    norm = np.zeros_like(sci_data)
    # get the splines out of the spline dictionary
    if 'spline_odd' not in splines:
        spline = splines['spline'], splines['spline']
        dspline = splines['dspline'], splines['dspline']
        d2spline = splines['d2spline'], splines['d2spline']
        d3spline = splines['d3spline'], splines['d3spline']
        spline_mask = [splines['spline_mask'], splines['spline_mask']]
        log.general('\tThe template has no left/right asymetry')
    else:
        spline = splines['spline_odd'], splines['spline_even']
        dspline = splines['dspline_odd'], splines['dspline_even']
        d2spline = splines['d2spline_odd'], splines['d2spline_even']
        d3spline = splines['d3spline_odd'], splines['d3spline_even']
        spline_mask = [splines['spline_mask_odd'], splines['spline_mask_even']]
        log.general('\tThe template has left/right asymetry handled '
                    'with odd/even splines')
    # the same splines grouped for a joint evaluation (template and its
    #   derivatives share their knots): same values as calling them
    tgroup = (_spline_group(splines, 'group_odd', [spline[0], dspline[0],
                                                   d2spline[0], d3spline[0]]),
              _spline_group(splines, 'group_even', [spline[1], dspline[1],
                                                    d2spline[1], d3spline[1]]))
    if 'spline_odd' not in splines:
        tgroup = (_spline_group(splines, 'group', [spline[0], dspline[0],
                                                   d2spline[0], d3spline[0]]),) * 2
    mgroup = (_spline_group(splines, 'group_spline_mask_odd', [spline_mask[0]]),
              _spline_group(splines, 'group_spline_mask_even', [spline_mask[1]]))
    if 'spline_odd' not in splines:
        mgroup = (_spline_group(splines, 'group_spline_mask',
                                [spline_mask[0]]),) * 2

    # set up storage for the dv, d2v, d3v and corresponding rms values
    #    fill with NaNs
    dv = np.full(len(ref_table['WAVE_START']), np.nan)
    d0v = np.full(len(ref_table['WAVE_START']), np.nan)
    d2v = np.full(len(ref_table['WAVE_START']), np.nan)
    d3v = np.full(len(ref_table['WAVE_START']), np.nan)
    sdv = np.full(len(ref_table['WAVE_START']), np.nan)
    sd0v = np.full(len(ref_table['WAVE_START']), np.nan)
    sd2v = np.full(len(ref_table['WAVE_START']), np.nan)
    sd3v = np.full(len(ref_table['WAVE_START']), np.nan)
    # keep track of the fraction of each line that is valid
    frac_line_valid = np.zeros(len(ref_table['WAVE_START']))

    # stoarge for final rv values
    rv_final = np.full(len(ref_table['WAVE_START']), np.nan)
    # storage for plotting
    plot_dict = dict()
    # get zero time
    zero_time = Time.now()
    # storage of rmsratio
    stddev_nsig = np.nan
    # once we reached the threshold for convergenve, we flip this flag for
    # one last iteration
    flag_last_iter = False
    # starting values for condition on the last_iteration = True/False
    rv_mean = np.inf
    bulk_error = 1.0
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
            # flag whether an order is even or odd
            is_even = (order_num % 2) == 0
            # flags orders that have <3 lines
            # do not spend time on this order computing anything
            if width[order_num] == 0:
                continue
            # doppler shifted wave grid for this order
            shift = -sys_rv - model_offset
            wave_ord = mp.doppler_shift(wavegrid[order_num], shift)
            # get the blaze for this order
            blaze_ord = blaze[order_num]
            # template spline and derivatives at these wavelengths (d2 and
            #   d3 only needed on the last iteration)
            nspl = 4 if flag_last_iter else 2
            tvals = tgroup[is_even](wave_ord, nspl)
            # get the low-frequency component out
            model_mask = np.ones_like(model[order_num])
            # add the spline mask values to model_mask (spline mask is 0 or 1)
            smask = mgroup[is_even](wave_ord)[0] < 0.99
            # set spline mask splined values to NaN
            model_mask[smask] = np.nan
            # RV shift the spline and correct for blaze and add model mask
            ord_model = tvals[0] * blaze_ord * model_mask
            # push into mask
            model[order_num] = ord_model
            # we are so close in RV with RV_mean<10*sigma that there is no need
            # to do the low-pass filtering again
            if iteration == 0:
                do_hp = True
                # nsig_rv_mean = np.inf
            else:
                nsig_rv_mean = np.abs(rv_mean) / bulk_error
                if nsig_rv_mean > 10:
                    do_hp = True  # too far from 0, do the high-pass filtering
                else:
                    do_hp = False

            if do_hp:
                part1 = sci_data[order_num]
                part2 = model[order_num]
                part2[part2 == 0] = np.nan
                # low pass filter
                with warnings.catch_warnings(record=True):
                    ratio_full = part1 / part2
                    lp_wid = int(width[order_num])
                    ratio[order_num] = mp.lowpassfilter(ratio_full, lp_wid, k=3)
                if order_num == 0:
                    log.general('\t\tLow-pass match of model to science ratio')

            model[order_num] *= ratio[order_num]

            # if this is the first iteration update model0
            if iteration == 0:
                # normalization of the local pseudo continuum level of the unblazed
                # model
                model0[order_num] = tvals[0]
                model0[order_num][model0[order_num] == 0] = np.nan
                lp_wid = model0.shape[1] // 2 # half the blaze length
                norm[order_num] = mp.lowpassfilter(model0[order_num], lp_wid, k=2)

                # spline the original template and apply blaze
                model0[order_num] *= blaze_ord
                # get the good values for the median
                valid = np.isfinite(model0[order_num])
                valid &= np.isfinite(sci_data[order_num])

                # normalize by the median
                with warnings.catch_warnings(record=True):
                    # get the sci data nad model for this order and valid points
                    rpart1 = sci_data0[order_num][valid]
                    rpart2 = model0[order_num][valid]
                    # science to model ratio
                    ratio_sci = mp.nanmedian(rpart1 / rpart2)
                    model0[order_num] = model0[order_num] * ratio_sci
            # -----------------------------------------------------------------
            # update the other splines
            # track ratio if relevant
            b_ratio[order_num] = blaze_ord * ratio[order_num]
            dmodel[order_num] = tvals[1] * b_ratio[order_num]
            # only do the d2 and d3 stuff if on last iteration
            if flag_last_iter:
                d2model_ord = tvals[2] * b_ratio[order_num]
                d3model_ord = tvals[3] * b_ratio[order_num]
                d2model[order_num] = d2model_ord
                d3model[order_num] = d3model_ord
                # deal with residual projection tables if required
                if resproj_flag:
                    # loop around residual project tables
                    for key in inst.params['RESPROJ_TABLES']:
                        # calculate spline
                        # To keep the physical units of the gradient files
                        # (computed on pseudo-continuum normalised spectra),
                        # we need to multiply by the local continuum level
                        rp_spline = _spline_group(
                            splines, 'group_' + key, [splines[key]])(
                            wave_ord)[0]  # * b_ratio[order_num] / norm[order_num]
                        # The models are always expressed in terms of the
                        # original spectrum
                        # rblaze = np.nanmedian(sci_data0[order_num] / blaze_ord)
                        # rp_spline *= (blaze_ord * rblaze)
                        # add to projection model
                        proj_model[key]['model'][order_num] = rp_spline

        # ---------------------------------------------------------------------
        # estimate rms
        # ---------------------------------------------------------------------
        # if we are not using a noise model - estimate the noise
        if not use_noise_model:

            rms = estimate_noise_model(sci_data, wavegrid, model,
                                       noise_sampling_width,
                                       npoints_orders=noise_npoints,
                                       nstep=inst.params['NOISE_SAMPLING_NSTEP'])
            # work out the number of sigma away from the model
            nsig = (sci_data - model) / rms
            # mask for nsigma
            with warnings.catch_warnings(record=True) as _:
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
                # TODO keep track of the stats on 'scale' as it should be 1
                # TODO for the photon-noise limited case.
                rms[order_num] *= scale
        # ---------------------------------------------------------------------
        # work out dv line-by-line
        # ---------------------------------------------------------------------
        # get orders
        orders = line_orders
        # set these for use/update later
        nwavegrid = mp.doppler_shift(wavegrid, -sys_rv)
        # the wave -> pixel splines of the orders (evaluated at the line
        #   edges below); orders with too few finite points give NaNs
        xpix = np.arange(model.shape[1])
        wave2pix_ok = np.ones(wavegrid.shape[0], dtype=bool)
        for order_num in range(wavegrid.shape[0]):
            # deal with too few finite points in wavegrid
            if np.sum(np.isfinite(nwavegrid[order_num])) < 5:
                emsg = ('Order {0} has too few finite points (<5) in '
                        'wavegrid to spline')
                log.warning(emsg.format(order_num))
                wave2pix_ok[order_num] = False
        # ---------------------------------------------------------------------
        # debug plot dictionary for plotting later
        if iteration == 0:
            plot_dict['WAVEGRID'] = nwavegrid
            plot_dict['MODEL'] = model
            plot_dict['PLOT_ORDERS'] = model_plot_orders
            plot_dict['LINE_ORDERS'] = []
            plot_dict['WW_ORD_LINE'] = []
            plot_dict['SPEC_ORD_LINE'] = []
            plot_dict['MODEL_ORD_LINE'] = []
        # calculate the CCF for the model (if we don't have model_velocity)
        if not np.isfinite(model_velocity):
            mask_table = inst.load_mask(mask_file)
            wave_mask = np.array(mask_table['ll_mask_s'], dtype=float)
            ccf_weight = np.array(mask_table['depth'], dtype=float)
            mask_snr = np.array(mask_table['line_snr'], dtype=float)
            # if SNR of line is less than 3 we don't use it
            ccf_weight[mask_snr < 3] = 0
            # absolute depths cannot be greater than 1
            ccf_weight[abs(ccf_weight) > 1] = 0
            # calculate the rough CCF RV estimate
            sys_model_rv, ewidth_model = rough_ccf_rv(inst, wavegrid, model,
                                                      wave_mask, ccf_weight,
                                                      kind='model')

            model_velocity = sys_model_rv - sys_rv
            log.general('\tModel velocity {:.2f} m/s'.format(model_velocity))
            # we don't want to continue this run if we have model_velocity
            continue
        # ---------------------------------------------------------------------
        # loop through all lines
        #   (numba kernel with the same arithmetic as the per-line python
        #    code it replaces: see lbl.core.fastmath.line_loop)
        # lines visited in this iteration (lines flagged as bad are skipped
        #   in all but the second iteration)
        if iteration != 1:
            visit = np.array(mask_keep)
        else:
            visit = np.ones(len(orders), dtype=bool)
        # pixel position of the line edges: spline between wavelength and
        #   pixel position of each order (one spline per order)
        x_start_f = np.full(len(orders), np.nan)
        x_end_f = np.full(len(orders), np.nan)
        for order_num in range(len(order_lines)):
            in_order = order_lines[order_num]
            if len(in_order) == 0 or not wave2pix_ok[order_num]:
                continue
            edges = np.concatenate([line_wave_start[in_order],
                                    line_wave_end[in_order]])
            pix_edges = mp.iuv_spline_eval(nwavegrid[order_num], xpix, edges)
            x_start_f[in_order] = pix_edges[:len(in_order)]
            x_end_f[in_order] = pix_edges[len(in_order):]
        # the per-line code did int(np.floor(x)): it fails on a non-finite x
        if not np.all(np.isfinite(x_start_f[visit]) &
                      np.isfinite(x_end_f[visit])):
            raise ValueError('cannot convert float NaN to integer')
        x_start_i = np.zeros(len(orders), dtype=np.int64)
        x_end_i = np.zeros(len(orders), dtype=np.int64)
        x_start_i[visit] = np.floor(x_start_f[visit]).astype(np.int64)
        x_end_i[visit] = np.floor(x_end_f[visit]).astype(np.int64)
        # residual projection models (only used on the last iteration)
        if flag_last_iter and resproj_flag:
            resproj_keys = list(inst.params['RESPROJ_TABLES'].keys())
            proj_models = np.array([proj_model[key]['model']
                                    for key in resproj_keys], dtype=float)
            proj_all = np.array([proj_model[key]['proj']
                                 for key in resproj_keys], dtype=float)
            sproj_all = np.array([proj_model[key]['sproj']
                                  for key in resproj_keys], dtype=float)
        else:
            resproj_keys = []
            proj_models = np.zeros((0, 1, 1))
            proj_all = np.zeros((0, len(orders)))
            sproj_all = np.zeros((0, len(orders)))
        # lines that passed the boundary conditions (for the plot)
        passed_bounds = np.zeros(len(orders), dtype=bool)
        # lines whose RMSRATIO / NPIXLINE / CHI2 are set (last iteration)
        stats_updated = np.zeros(len(orders), dtype=bool)
        # run the line-by-line loop
        fastmath.line_loop(iteration, flag_last_iter, orders,
                           line_wave_start, line_wave_end,
                           x_start_i, x_end_i, mask_keep,
                           _f64(nwavegrid), _f64(sci_data), _f64(rms),
                           _f64(model), _f64(dmodel), _f64(d2model),
                           _f64(d3model), _f64(blaze), _f64(b_ratio),
                           _f64(norm), proj_models,
                           min_line_width, dv, sdv, d0v, sd0v, d2v, sd2v,
                           d3v, sd3v, frac_line_valid,
                           ref_table['MEANXPIX'], ref_table['MEANBLAZE'],
                           ref_table['RMSRATIO'], ref_table['NPIXLINE'],
                           ref_table['CHI2'], proj_all, sproj_all,
                           passed_bounds, stats_updated)
        # keep track of the reference table values set for this file
        lines_xpix_set |= passed_bounds
        lines_stats_set |= stats_updated
        # push the residual projections back
        for ikey, key in enumerate(resproj_keys):
            proj_model[key]['proj'] = proj_all[ikey]
            proj_model[key]['sproj'] = sproj_all[ikey]
        # add to the plots dictionary (for plotting later)
        if iteration == 1:
            for line_it in np.where(passed_bounds)[0]:
                order_num = orders[line_it]
                x_start, x_end = x_start_i[line_it], x_end_i[line_it]
                plot_dict['LINE_ORDERS'] += [order_num]
                plot_dict['WW_ORD_LINE'] += [
                    nwavegrid[order_num][x_start:x_end + 1]]
                plot_dict['SPEC_ORD_LINE'] += [
                    sci_data[order_num][x_start:x_end + 1]]
                plot_dict['MODEL_ORD_LINE'] += [
                    model[order_num][x_start:x_end + 1]]
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
        rv_final = np.array(dv + sys_rv - berv + model_offset)
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
        if flag_last_iter:
            # break here
            break
        # do a convergence check
        if np.abs(rv_mean) < (converge_thres * bulk_error):
            msg = '\t\tConverged RV = {0:.2f} m/s, sigma = {1:.2f} m/s'
            margs = [rv_mean, bulk_error]
            log.general(msg.format(*margs))

            # One more iteration
            flag_last_iter = True

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
    # adding to the fits table the 0th derivative projection
    # this is a contrast of the line. This is 1+ the change in depth
    ref_table['contrast'] = 1 + d0v
    ref_table['sig_contrast'] = sd0v
    # adding to the fits table the 2nd derivative projection
    ref_table['d2v'] = d2v
    ref_table['sd2v'] = sd2v
    # adding to the fits table the 3rd derivative projection
    ref_table['d3v'] = d3v
    ref_table['sd3v'] = sd3v
    # calculate the chi2 cdf
    chi2_cdf = 1 - stats.chi2.cdf(ref_table['CHI2'], ref_table['NPIXLINE'])
    ref_table['CHI2_VALID_CDF'] = chi2_cdf
    # fraction of each lines that is a valid pixel when computing the dv
    ref_table['FRAC_LINE_VALID'] = frac_line_valid
    # deal with residual projection tables if required
    if resproj_flag:
        # loop around residual project tables
        for key in inst.params['RESPROJ_TABLES']:
            # add to the reference table
            ref_table[key] = proj_model[key]['proj']
            ref_table['s' + key] = proj_model[key]['sproj']
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
        wmsg = 'This RV is (probably) bad (iterations = {0}). '
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
    outputs['LINES_XPIX_SET'] = lines_xpix_set
    outputs['LINES_STATS_SET'] = lines_stats_set
    # -------------------------------------------------------------------------
    # return reference table and outputs
    return ref_table, outputs


def get_velo_estimate(systemic_props: Dict[str, Any],
                      mjdate: np.ndarray) -> float:
    """
    Get the velocity estimate (if using a LBL generated template)
    otherwise we set this to zero

    :param systemic_props: dict, containing the SCI_TABLE from the loaded
                           template
    :param mjdate: np.ndarray, the mid exposure times of each file

    :return: float, the velocity estimate of the star (from the template)
    """
    if systemic_props['SCI_TABLE'] is None:
        # warn user that non-lbl template used
        msg = ('No SCI_TABLE table found in template - '
               'we recommend using only LBL templates. '
               'Settings velocity estimate to zero')
        log.warning(msg)
        # set velocity estimate to zero
        return 0.0
    # -------------------------------------------------------------------------
    # get the science table from template
    sci_table = systemic_props['SCI_TABLE']
    # -------------------------------------------------------------------------
    # deal with first estimate of RV / CCF equivalent width
    # exptimes in days
    # if KW_EXPTIME is null this will raise a ValueError
    try:
        expday = np.array(sci_table['KW_EXPTIME']).astype(float) / 86400
    except Exception as _:
        # warn user that KW_EXPTIME is not valid
        msg = ('KW_EXPTIME not valid - seeting velocity estimate to zero')
        log.warning(msg)
        return 0.0
    # -------------------------------------------------------------------------
    # times used to match the systemic velocity
    # to the observation time
    mid_times = np.array(sci_table['KW_MID_EXP_TIME'].astype(float))
    mjd_table_low = mid_times - expday / 2.0
    mjd_table_high = mid_times + expday / 2.0
    valid = (mjd_table_low < mjdate) & (mjdate < mjd_table_high)
    # -------------------------------------------------------------------------
    if np.sum(valid) == 0:
        # this could occur if we pass a friend template (no measure on
        # that day) or if we add a new file but do not recompute the
        # template. The last mjd in SCI_TABLE is before the observation.
        return 0.0
    # -------------------------------------------------------------------------
    # we do it if we have more than one value to get an estimate
    # for the starting point of the LBL iteration. If we have only
    # one value, this just returns a single numpy value that is not
    # a list. It avoids passing a list to sys_rv.
    return np.mean(sci_table['VSYS'][valid])


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


def add_to_rv_headers(headers: Dict[str, List[Any]],
                      comments: Dict[str, str],
                      header: Any) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """
    Add to rv headers + comments (for merging later)

    :param headers: dictionary of lists, all header keys for all files
    :param comments: dictionary of strs, all comments for each header key
    :param header: fits.Header, the header to add

    :return: the updated headers and comments dictionaries
    """
    # loop around header keys
    for key in header:
        if key not in headers:
            headers[key] = [header[key]]
        else:
            headers[key].append(header[key])
        if key not in comments:
            comments[key] = header.comments[key]

    return headers, comments


def combine_rv_headers(headers: Dict[str, List[Any]],
                      comments: Dict[str, str]) -> Dict[str, Tuple[Any, str]]:
    """
    Combine rv headers into a single header

    :param headers: dictionary of lists, all header keys for all files
    :param comments: dictionary of strs, comments for each header key

    :return: dictionary, combined header with (value, comment) tuples
    """
    # storage for combined headers
    c_header = dict()
    # loop around header keys
    for key in headers:
        # ---------------------------------------------------------------------
        # get the values from the headers
        values = headers[key]
        comment = comments[key]
        # ---------------------------------------------------------------------
        if len(key) > 8 and key[:8] != 'HIERARCH':
            key = 'HIERARCH ' + key
        # ---------------------------------------------------------------------
        # don't add this key if we have no values (shouldn't happen)
        if len(values) == 0:
            continue
        # ---------------------------------------------------------------------
        # check if all values are the same
        all_same = all(x == values[0] for x in values)
        # if all values are the same, this is easy, just use the first
        if all_same:
            # we have to avoid too long header keys
            if len(key) + len(str(values[0])) > 75:
                continue
            c_header[key] = (values[0], comment)
            # and continue
            continue
        # ---------------------------------------------------------------------
        # otherwise, we need to combine depending on type
        if all(isinstance(x, (int, float)) for x in values):
            # numerical values - take mean and std
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            num = len(values)
            # deal with NaNs and infinite values
            if not np.isfinite(mean_val):
                mean_val = 'NaN'
            if not np.isfinite(std_val):
                std_val = 'NaN'
            # we have to avoid too long header keys
            if len(key) + len(str(mean_val)) > 75:
                continue
            if len(key) + len(str(std_val)) > 75:
                continue

            key1 = f'HIERARCH CAVG {key}'
            key2 = f'HIERARCH CSTD {key}'
            key3 = f'HIERARCH CNUM {key}'
            c_header[key1] = (mean_val, comment)
            c_header[key2] = (std_val, comment)
            c_header[key3] = (num, comment)
        # otherwise we are dealing with strings - just use the last
        else:
            c_header[key] = (values[-1], comment)
    # -------------------------------------------------------------------------
    # return the combined header
    return c_header

# =============================================================================
# Define compil functions
# =============================================================================
def make_rdb_table(inst: InstrumentsType, rdbfile: str,
                   lblrvfiles: np.ndarray, plot_dir: str
                   ) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, str]]]:
    """
    Make the primary rdb table (row per observation)

    :param inst: Instrument instance
    :param rdbfile: str, the rdb file absolute path (only used to save plot)
    :param lblrvfiles: np.ndarray, array of strings, the absolute path to each
                       LBL RV file
    :param plot_dir: str, the absolute path to the directory to save plot
                     to (if plot commands are set to True)

    :return: rdb dict for fits/rdb file data
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
    # get fraction of time a line as been measured
    frac_time_meas = inst.params['COMPIL_FRAC_TIME_MEAS']
    # obj_sci = inst.params['OBJECT_SCIENCE']
    ccf_ew_fp = inst.params['COMPIL_FP_EWID']
    # get the threshold of pearson-r likelihood of correlation with BERV
    cut_pearsonr = inst.params['COMPIL_CUT_PEARSONR']
    # reference wavelength for slope intercept
    reference_wavelength = inst.params['COMPIL_SLOPE_REF_WAVE']
    # get the ccf e-width column name
    ccf_ew_col = inst.params['KW_CCF_EW']
    # flag whether we have residual projection tables
    resproj_flag = isinstance(inst.params['RESPROJ_TABLES'], dict)
    # Force the per - line dispersion to match uncertainties. In other words,
    # the per-line (vrad-median(vrad))/svrad will be forced to a median value
    # of 1 if True. This causes a degradation of performances by 5-10% for
    # SPIRou but makes the svrad more representative of the expected dispersion
    # in the timeseries.
    force_sigma_per_line = inst.params['COMPIL_FORCE_SIGMA_PER_LINE']

    # get the header keys to add to rdb_table
    header_keys, fp_flags = inst.rdb_columns()
    # get base names
    lblrvbasenames = []
    for lblrvfile in lblrvfiles:
        lblrvbasenames.append(os.path.basename(lblrvfile))
    # -------------------------------------------------------------------------
    # output rdb column set up
    # -------------------------------------------------------------------------
    # storage for the header keys
    rdb_headers, rdb_comments = dict(), dict()
    # storage for rdb table dictionary
    rdb_dict: Dict[str, Any] = dict()
    # add columns (dv, sdv, d2v, sd2v, d3v, sd3v, rjd, vrad, svrad)
    rdb_dict['rjd'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['vrad'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['svrad'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['d2v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['sd2v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['d3v'] = np.zeros_like(lblrvfiles, dtype=float)
    rdb_dict['sd3v'] = np.zeros_like(lblrvfiles, dtype=float)
    # deal with residual projection table
    if resproj_flag:
        # loop around keys in resproj tables
        for key in inst.params['RESPROJ_TABLES']:
            rdb_dict[key] = np.zeros_like(lblrvfiles, dtype=float)
            rdb_dict['s' + key] = np.zeros_like(lblrvfiles, dtype=float)
    # define the local file name
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
    # mean line contrast
    rdb_dict['contrast'] = np.zeros_like(lblrvfiles, dtype=float)
    # error on mean line contrast
    rdb_dict['sig_contrast'] = np.zeros_like(lblrvfiles, dtype=float)
    # velocity at a reference wavelength fitting for the chromatic slope
    rdb_dict['vrad_achromatic'] = np.zeros_like(lblrvfiles, dtype=float)
    # error on achromatic velocity
    rdb_dict['svrad_achromatic'] = np.zeros_like(lblrvfiles, dtype=float)
    # chromatic slope
    rdb_dict['vrad_chromatic_slope'] = np.zeros_like(lblrvfiles, dtype=float)
    # error in chromatic slope
    rdb_dict['svrad_chromatic_slope'] = np.zeros_like(lblrvfiles, dtype=float)
    # wavelength dependency of velocity following Zechmeister et al. 2018 (CRX)
    rdb_dict['CRX'] = np.zeros_like(lblrvfiles, dtype=float)
    # error on CRX
    rdb_dict['sCRX'] = np.zeros_like(lblrvfiles, dtype=float)
    # get filename column
    rdb_dict['FILENAME'] = [[]] * len(lblrvfiles)
    # add header keys
    for hdr_key in header_keys:
        # ignore header keys that are None
        if hdr_key is None:
            continue
        # empty elements in a list for each key to fill
        rdb_dict[hdr_key] = [[]] * len(lblrvfiles)
    # add version
    rdb_dict['LBL_VERSION'] = [base.__version__] * len(lblrvfiles)
    # add version date
    rdb_dict['LBL_VDATE'] = [base.__date__] * len(lblrvfiles)
    # add date now (isot format)
    rdb_dict['LBL_PDATE'] = [base.__now__.isot] * len(lblrvfiles)
    # add local science directory
    rdb_dict['LBL_SCI_DIR'] = [[]] * len(lblrvfiles)
    # -------------------------------------------------------------------------
    # open first file to set up good mask
    # -------------------------------------------------------------------------
    # load table and header
    rvtable0, rvhdr0 = inst.load_lblrv_file(str(lblrvfiles[0]))
    # get columns of rvtable0
    wavestart0 = np.array(rvtable0['WAVE_START'])
    npixline0 = np.array(rvtable0['NPIXLINE'])
    # flag a calibration file
    flag_calib = inst.params['DATA_TYPE'] != 'SCIENCE'
    # do not consider lines below wave_min limit
    good = wavestart0 > wave_min
    # do not consider lines above wave_max limit
    good &= wavestart0 < wave_max
    # do not consider lines wider than max_pix_wid limit
    good &= npixline0 < max_pix_wid
    # remove good from rv table
    rvtable0 = rvtable0[good]
    ref_good_pix = np.array(good)  # if a calibration, we'll need this later
    # get columns of rvtable0
    dv0 = np.array(rvtable0['dv'])
    sdv0 = np.array(rvtable0['sdv'])
    # size of arrays
    nby, nbx = len(lblrvfiles), np.sum(good)
    # set up rv and dvrms
    wave_arr = np.zeros([nby, nbx])
    dv_arr, sdv_arr = np.zeros([nby, nbx]), np.zeros([nby, nbx])
    d2v_arr, sd2v_arr = np.zeros([nby, nbx]), np.zeros([nby, nbx])
    d3v_arr, sd3v_arr = np.zeros([nby, nbx]), np.zeros([nby, nbx])
    contrast_arr, scontrast_arr = np.zeros([nby, nbx]), np.zeros([nby, nbx])

    # projection model for the rdb_dict
    proj_model = dict()
    # deal with residual projection table
    if resproj_flag:
        # loop around keys in residual projection tables
        for key in inst.params['RESPROJ_TABLES']:
            # each projection model as a proj_arr and sproj_arr
            proj_model[key] = dict()
            proj_model[key]['proj_arr'] = np.zeros([nby, nbx])
            proj_model[key]['sproj_arr'] = np.zeros([nby, nbx])
    # store the berv
    berv = np.zeros(len(lblrvfiles))
    # -------------------------------------------------------------------------
    # work out for cumulative plot for first file
    # -------------------------------------------------------------------------
    # median velocity for first file rv table
    med_velo = mp.nanmedian(dv0)
    # work out the best lines (less than 5 sigma)
    best_mask = sdv0 < np.nanpercentile(sdv0, 5.0)
    # list for plot storage
    vrange_all, pdf_all, pdf_fit_all = [], [], []

    # -------------------------------------------------------------------------
    # Loop around lbl rv files
    # -------------------------------------------------------------------------
    # log progress
    log.info('Producing LBL RDB 1 table')
    # store any missing keys
    missing_keys = dict()
    # loop around lbl rv files
    for row in tqdm(range(len(lblrvfiles))):
        # ---------------------------------------------------------------------
        # get lbl rv file table and header
        # ---------------------------------------------------------------------
        # load table and header
        rvtable, rvhdr = inst.load_lblrv_file(lblrvfiles[row])
        # store the berv value for each file
        berv[row] = inst.get_berv(rvhdr)
        # fix header (instrument specific)
        rvhdr = inst.fix_lblrv_header(rvhdr)
        # fill rjd value
        rdb_dict['rjd'][row] = inst.get_rjd_value(rvhdr)
        # fill in plot date
        rdb_dict['plot_date'][row] = inst.get_plot_date(rvhdr)
        # add keys to rdb_headers
        rdb_headers, rdb_comments = add_to_rv_headers(rdb_headers, rdb_comments,
                                                      rvhdr)
        # ---------------------------------------------------------------------
        # fill in filename
        # ---------------------------------------------------------------------
        rdb_dict['FILENAME'][row] = os.path.basename(lblrvfiles[row])
        rdb_dict['LBL_SCI_DIR'][row] = os.path.dirname(lblrvfiles[row])
        # ---------------------------------------------------------------------
        # fill in header keys
        # ---------------------------------------------------------------------
        # loop around header keys
        for ikey, key in enumerate(header_keys):
            # ignore keys that are none
            if key is None:
                continue
            # make sure key is a string
            key = str(key)
            # deal with FP flags
            if flag_calib and fp_flags[ikey]:
                rdb_dict[key][row] = np.nan
            # if we have key add the value
            elif key in rvhdr:
                rdb_dict[key][row] = rvhdr.get_hkey(key)
            # else print a warning and add a NaN
            else:
                # add to missing keys dict counter
                if key in missing_keys:
                    missing_keys[key] += 1
                else:
                    missing_keys[key] = 1
                # set value to NaN
                rdb_dict[key][row] = np.nan
        # ---------------------------------------------------------------------
        # Read all lines for this file and load into arrays
        # ---------------------------------------------------------------------
        # if we don't have a calibration we set the rvs and dvrms from rv table
        if not flag_calib:
            dv_arr[row] = rvtable['dv'][good]
            sdv_arr[row] = rvtable['sdv'][good]
        # else we calculate it using odd ratio mean
        else:
            cal_rv = np.array(rvtable['dv'][good], dtype=float)
            cal_dvrms = np.array(rvtable['sdv'][good], dtype=float)
            # estimate using odd ratio mean
            cal_guess, cal_bulk_error = mp.odd_ratio_mean(cal_rv, cal_dvrms)
            # push into rdb_dict
            rdb_dict['vrad'][row] = cal_guess
            rdb_dict['svrad'][row] = cal_bulk_error
        # deal with residual projection tables
        if resproj_flag:
            # loop around keys in residual projection tables
            for key in inst.params['RESPROJ_TABLES']:
                # check that key is in rvtable
                if key not in rvtable.colnames:
                    # log an error
                    emsg = ('RESPROJ Table key "{0}" not in rvtable.'
                            'Please do not populat the RESPROJ_TABLES with '
                            'key {0}')
                    raise LblException(emsg.format(key))
                # copy the rvtable array for this residual projection
                arr = np.array(rvtable[key][good], dtype=float)
                # copy the rvtable error array for this residual projection
                sarr = np.array(rvtable['s' + key][good], dtype=float)
                # get the guess and bulk error
                val_guess, val_bulk_error = mp.odd_ratio_mean(arr, sarr)
                # push into the rdb dictioanry
                rdb_dict[key][row] = val_guess
                rdb_dict['s' + key][row] = val_bulk_error
        # get the d2v, sd2v, d3v and sd3v values from table
        wave_vec = np.array(rvtable['WAVE_START'][good], dtype=float)
        contrast = np.array(rvtable['contrast'][good], dtype=float)
        scontrast = np.array(rvtable['sig_contrast'][good], dtype=float)
        d2v = np.array(rvtable['d2v'][good], dtype=float)
        sd2v = np.array(rvtable['sd2v'][good], dtype=float)
        d3v = np.array(rvtable['d3v'][good], dtype=float)
        sd3v = np.array(rvtable['sd3v'][good], dtype=float)
        # push these values into array (for saving images later)
        wave_arr[row] = wave_vec
        d2v_arr[row], sd2v_arr[row] = d2v, sd2v
        d3v_arr[row], sd3v_arr[row] = d3v, sd3v
        contrast_arr[row], scontrast_arr[row] = contrast, scontrast
        # use the odd mean ratio to calculate d2v and sd2v
        contrast_guess, contrast_bulk_error = mp.odd_ratio_mean(contrast,
                                                                scontrast)
        # push into rdb_dict
        rdb_dict['contrast'][row] = contrast_guess
        rdb_dict['sig_contrast'][row] = contrast_bulk_error

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
        # if we don't have a calibration add plot values (only used by the
        #   cumulative plot)
        cumul_plot = inst.params['PLOT'] and inst.params['PLOT_COMPIL_CUMUL']
        if not flag_calib and cumul_plot:
            # plot specific domain. We use the median velocity of the file
            # for the plot domain. +-15 km/s to get an idea of residuals.
            xlim = [med_velo - 15000, med_velo + 15000]
            # get velocity range
            vrange = np.arange(xlim[0], xlim[1], 50.0)
            # storage for probability density function
            pdf = np.zeros_like(vrange, dtype=float)
            # mask the rv and dvrms by best_mask
            best_rv = dv_arr[row][best_mask]
            best_dvrms = sdv_arr[row][best_mask]
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
    # display missing keys
    if len(missing_keys) > 0:
        wmsg = ('The following header keys were not present in some files. '
                'These values will be set to NaN in the rdb table.')
        log.warning(wmsg)
        # loop around missing keys
        for missing_key in missing_keys:
            wmsg = '\t{0}: missing in {1} file(s)'
            log.warning(wmsg.format(missing_key, missing_keys[missing_key]))

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
        msg = 'Constructing a per-epoch mean velocity'
        log.general(msg)
        # loop around files and generate a first guess for vrad and svrad
        for file_it in tqdm(range(dv_arr.shape[0])):
            # use the odd ratio mean to guess vrad and svrad
            orout1 = mp.odd_ratio_mean(dv_arr[file_it], sdv_arr[file_it])
            # add to output table
            rdb_dict['vrad'][file_it] = orout1[0]
            rdb_dict['svrad'][file_it] = orout1[1]
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
        per_line_mean = np.zeros(dv_arr.shape[1])
        per_line_error = np.zeros(dv_arr.shape[1])
        # log progress
        log.info('Producing line-by-line mean positions')

        if force_sigma_per_line:
            msg = 'Forcing a stdev of 1 for all lines'
            log.general(msg)
        else:
            msg = 'We do not force the mean(svrad) to match per line RMS'
            log.general(msg)

        # fraction of valid measurements for given line
        frac_valid = np.nanmean(np.isfinite(dv_arr), axis=0)

        # for each line, likelihood that correlation with BERV
        # is statistically significant. Not correlation with BERV
        # gives a flat 0 to 1 distribution. Some correlation with
        # BERV will lead to pile-up at very small values. With a
        # p=1e-4 value, we would reject 1 or 2 lines for a typical
        # target in the absence of a True correlation.
        prob_pearsonr = np.zeros(len(per_line_mean))
        # compute the per-line bias
        for line_it in tqdm(range(len(per_line_mean))):
            # We should have a threshold in the fraction of 'valid' times the
            #     line has been measured
            if frac_valid[line_it] < frac_time_meas:
                per_line_mean[line_it] = np.nan
                per_line_error[line_it] = np.nan
                continue
            # find valid velocity measurements
            good = np.isfinite(dv_arr[:, line_it])
            per_line_diff = (dv_arr[:, line_it] - rdb_dict['vrad'])
            # skip if less that 3 valid values for this line
            if np.sum(good) < 3:
                if dv_arr.shape[0] < 5:
                    per_line_mean[line_it] = np.nanmean(dv_arr[:, line_it])
                    per_line_error[line_it] = np.nanstd(dv_arr[:, line_it])/np.sqrt(np.sum(good))
                else:
                    per_line_mean[line_it] = np.nan
                    per_line_error[line_it] = np.nan
                continue

            if force_sigma_per_line:
                # forcing sigma of 1 the mean uncertainty per line. The
                #    uncertainties are therefore representative of the distribution
                #    for any given line. We keep the spectrum-to-spectrum
                #    differences and force the correct 1-sigma width in the
                #    distribution.
                plratio = per_line_diff / sdv_arr[:, line_it]
                sig1, sig2 = np.nanpercentile(plratio, [16, 84])
                sigma = (sig2 - sig1) / 2
                sdv_arr[:, line_it] = sdv_arr[:, line_it] * sigma

            # first output of pearsonr is the correlation itself.
            # this is meaningless if we don't know the number of points
            # that came in. The second term is the likelihood that the
            # correlation is due to a statistical fluctuation (it significance)
            # and can be used to flag lines that suspiciously correlate
            # with BERV.
            with warnings.catch_warnings(record=True) as _:
                pout = pearsonr(berv[good], per_line_diff[good])
            prob_pearsonr[line_it] = pout[1]

            if prob_pearsonr[line_it] < cut_pearsonr:
                per_line_mean[line_it] = np.nan
                per_line_error[line_it] = np.nan
                continue
            # get the difference and error for each line
            with warnings.catch_warnings(record=True) as _:
                diff1 = dv_arr[:, line_it] - rdb_dict['vrad']
                diff1 -= np.nanmedian(diff1)
            # here we avoid having a shallow copy
            err1 = np.array(sdv_arr[:, line_it])
            # try to guess the odd ratio mean
            # noinspection PyBroadException
            try:
                # also updates dvrms[:,line_it]
                guess2, bulk_error2 = mp.odd_ratio_mean(diff1, err1)
                per_line_mean[line_it] = guess2
                per_line_error[line_it] = bulk_error2
            # if odd ratio mean fails push NaNs into arrays
            except Exception as _:
                per_line_mean[line_it] = np.nan
                per_line_error[line_it] = np.nan

        # normalize the per-line mean to zero
        guess3, bulk_error3 = mp.odd_ratio_mean(per_line_mean, per_line_error)
        per_line_mean = per_line_mean - guess3
        nsig = np.abs(per_line_mean / mp.estimate_sigma(per_line_mean))
        per_line_mean[nsig > 100] = np.nan

        # ---------------------------------------------------------------------
        # Model per line
        # ---------------------------------------------------------------------
        # construct a 2d model of line biases
        rv_per_line_model = np.tile(per_line_mean, dv_arr.shape[0])
        rv_per_line_model = rv_per_line_model.reshape(dv_arr.shape)
    else:
        rv_per_line_model = np.zeros(dv_arr.shape)

    # -------------------------------------------------------------------------
    # save all rv data to a fits file
    rdb_fitsdata = dict()
    rdb_fitsdata['WAVE'] = wave_arr
    rdb_fitsdata['DV'] = dv_arr
    rdb_fitsdata['SDV'] = sdv_arr
    rdb_fitsdata['D2V'] = d2v_arr
    rdb_fitsdata['SD2V'] = sd2v_arr
    rdb_fitsdata['D3V'] = d3v_arr
    rdb_fitsdata['SD3V'] = sd3v_arr
    rdb_fitsdata['RDB0'] = rvtable0
    # -------------------------------------------------------------------------
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
            rvtable = rvtable[ref_good_pix]  # to match dvarr size
            residuals = np.array(rvtable['dv'])
            # get the error
            err = np.array(rvtable['sdv'])
            rvs_row = residuals
        else:
            # get the residuals of the rvs to the rv per line model
            rvs_row = dv_arr[row]
            residuals = dv_arr[row] - rv_per_line_model[row]
            err = sdv_arr[row]
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
        # crx value
        crx, scrx = np.nan, np.nan
        # store count
        itr_count = 1
        # loop around 10 iterations (but break on convergence)
        while itr_count <= 10:
            # find finite points
            good = np.isfinite(err) & np.isfinite(rvs_row)
            good &= np.isfinite(rv_per_line_model[0])
            # find very large outliers in the per-line velocities
            nsig = np.abs(rvs_row - np.nanmedian(rvs_row)) / err
            good &= (nsig < 10)
            good &= (err < 100 * np.nanmedian(err))
            # get valid wave and subtract reference wavelength
            valid_wave = rvtable0['WAVE_START'][good] - reference_wavelength
            valid_log_wave = np.log(rvtable0['WAVE_START'][good])
            # get valid rv an dv drms
            valid_rv = (rvs_row - rv_per_line_model[0])[good]
            valid_dvrms = err[good]
            # get weights for the np.polyval fit. Do *not* use the square of
            # error bars for weight (see polyfit help)

            # TODO -> add a threshold for lines that have suspiciously small
            # TODO -> errors compared to the bulk of error bar values.
            # TODO -> only done if some errors are equal to zero.
            if np.min(valid_dvrms) == 0:
                # we probably want a slightly lower threshold that the 1%-ile,
                #    the valid limit is typically lower
                threshold = np.nanpercentile(valid_dvrms, 1) / 2
                suspicious = valid_dvrms < threshold
                valid_dvrms[suspicious] = threshold

            weight_lin_fit = prob_good[good] / valid_dvrms
            # fitting the lines with weights
            # noinspection PyTupleAssignmentBalance
            scoeffs, scov = np.polyfit(valid_wave, valid_rv, 1,
                                       w=weight_lin_fit, cov=True)
            # fitting the lines with weights (in CRX frame)
            scoeffs_crx, scov_crx = np.polyfit(valid_log_wave, valid_rv, 1,
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
            # get the crx values
            crx = scoeffs_crx[0]
            scrx = np.sqrt(scov_crx[0, 0])
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
        mean_fwhm = mp.fwhm_value() * ccf_ew_row
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
        rdb_dict['CRX'][row] = crx
        rdb_dict['sCRX'][row] = scrx
        # ---------------------------------------------------------------------
        # Per-band per region RV measurements
        # ---------------------------------------------------------------------
        # get the instrument specific binned parameters
        binned_dict = inst.get_binned_parameters()
        binned_dict = inst.get_uniform_binned_parameters(binned_dict)
        # get info from binned dictionary
        bands = binned_dict['bands']
        blue_end = binned_dict['blue_end']
        red_end = binned_dict['red_end']
        region_names = binned_dict['region_names']
        region_low = binned_dict['region_low']
        region_high = binned_dict['region_high']
        use_regions = binned_dict['use_regions']
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
            tmp_rv = dv_arr[row] - rv_per_line_model[row]
            tmp_err = sdv_arr[row]
        # ---------------------------------------------------------------------
        # populate columns with zeros (if not present)
        # ---------------------------------------------------------------------
        # loop around the bands
        for iband in range(len(bands)):
            # decide whether to use regions
            if use_regions[iband]:
                band_regions = list(region_names)
            # if not use just the full domain
            else:
                band_regions = ['']
            # loop around the regions
            for iregion in range(len(band_regions)):
                cargs = [bands[iband], band_regions[iregion]]
                vrad_colname = 'vrad_{0}{1}'.format(*cargs)
                svrad_colname = 'svrad_{0}{1}'.format(*cargs)
                # add new column if not present
                if vrad_colname not in rdb_dict:
                    rdb_dict[vrad_colname] = lblrv_zeros.copy()
                if svrad_colname not in rdb_dict:
                    rdb_dict[svrad_colname] = lblrv_zeros.copy()
        # ---------------------------------------------------------------------
        # loop around the bands
        for iband in range(len(bands)):
            # make a mask based on the band (can use rvtable0 as wave start
            #   is the same for all rvtables)
            band_mask = rvtable0['WAVE_START'] > blue_end[iband]
            band_mask &= rvtable0['WAVE_START'] < red_end[iband]
            # decide whether to use regions
            if use_regions[iband]:
                band_regions = list(region_names)
                band_region_low = list(region_low)
                band_region_high = list(region_high)
            # if not use just the full domain
            else:
                band_regions = ['']
                band_region_low = [-np.inf]
                band_region_high = [np.inf]
            # loop around the regions
            for iregion in range(len(band_regions)):
                # mask based on region
                region_mask = rvtable0['XPIX'] > band_region_low[iregion]
                region_mask &= rvtable0['XPIX'] < band_region_high[iregion]
                # -------------------------------------------------------------
                # get combined mask for band and region
                comb_mask = band_mask & region_mask
                # -------------------------------------------------------------
                # get the finite points
                finite_mask = np.isfinite(tmp_err[comb_mask])
                finite_mask &= np.isfinite(tmp_rv[comb_mask])
                # deal with not having enough values (half of total being
                #    non finite)
                if np.sum(finite_mask) < np.sum(comb_mask) / 10:
                    continue
                # deal with not having enough points in general (min = 5)
                if np.sum(comb_mask) < 5:
                    continue
                # -------------------------------------------------------------
                # make a guess on the vrad and svrad via odd ratio mean
                guess7, bulk_error7 = mp.odd_ratio_mean(tmp_rv[comb_mask],
                                                        tmp_err[comb_mask])
                # -------------------------------------------------------------
                # get column names to add these bands/regions
                cargs = [bands[iband], band_regions[iregion]]
                vrad_colname = 'vrad_{0}{1}'.format(*cargs)
                svrad_colname = 'svrad_{0}{1}'.format(*cargs)
                # add to rdb_dict
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
    # add to output dictionary
    rdb_fitsdata['RDB'] = rdb_table
    # ---------------------------------------------------------------------
    # merge all rv headers into a single header
    rdb_header = combine_rv_headers(rdb_headers, rdb_comments)
    # ---------------------------------------------------------------------
    # return rdb table
    return rdb_fitsdata, rdb_header


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
    # determine the weighted mean columns
    wmean_pairs = dict()
    # search for columns that are the same as another column but just with an
    #   added 's' on front
    for colname in rdb_table.colnames:
        if colname.startswith('s') and colname[1:] in rdb_table.colnames:
            if colname[1:] not in wmean_pairs:
                wmean_pairs[colname[1:]] = colname
    # add any column pairs that don't conform to this
    wmean_pairs['fwhm'] = 'sig_fwhm'
    wmean_pairs['contrast'] = 'sig_contrast'
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
                    with warnings.catch_warnings(record=True) as _:
                        value = mp.nansum(vals / errs2) / mp.nansum(1 / errs2)
                        err_value = np.sqrt(1 / mp.nansum(1 / errs2))
                    # push into table
                rdb_dict2[colname].append(value)
                rdb_dict2[wmean_pairs[colname]].append(err_value)
            # -----------------------------------------------------------------
            # if not vrad or svrad then try to mean the column or if not
            #   just take the first value
            elif colname not in wmean_pairs.values():
                # do not try to do a mean on strings (its stupid)
                if isinstance(itable[colname][0], str):
                    rdb_dict2[colname].append(itable[colname][0])
                    continue
                # deal with masked columns
                if hasattr(itable[colname], 'mask'):
                    if itable[colname].mask[0]:
                        rdb_dict2[colname].append(np.nan)
                        continue
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
    # store rows which do not have reference file
    no_ref_files = []
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
            # record no ref present
            no_ref_files.append(uwavefile)
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
    # report on number of files without reference
    wmsg = 'Number of references not present = {0}/{1}'
    log.warning(wmsg.format(len(no_ref_files), len(uwaves)))
    # loop around files with no reference
    for no_ref_file in no_ref_files:
        log.warning('\t- {0}'.format(no_ref_file))
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
    # round time stamp to 8 decimal places (to avoid rounding errors)
    timestamp = np.round(timestamp, 8)
    # round the drift_table rjd to 8 decimal places (to avoid rounding errors)
    drift_times = np.round(drift_table['rjd'], 8)
    # -------------------------------------------------------------------------
    # log progress
    log.info('Producing LBL RDB drift corrected table')
    # store vrad = NaN rows
    no_value = 0
    nan_vrad = dict()
    # loop around the wave files of this type
    for row in tqdm(range(len(timestamp))):
        # create a mask of all files that match in drift file
        file_mask = timestamp[row] == drift_times
        # ---------------------------------------------------------------------
        # deal with no files present - cannot correct drift
        if np.sum(file_mask) == 0:
            # keep track of the number of NaNs
            no_value += 1
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
                    # keep track of the number of NaNs
                    if np.isnan(vrad_drift):
                        if colname in nan_vrad:
                            nan_vrad[colname] += 1
                        else:
                            nan_vrad[colname] = 1
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
    # report on no values
    if no_value > 0:
        wmsg = '{0}/{1} row(s) have no drift value (set to Nan)'
        log.warning(wmsg.format(no_value, len(rdb_table)))
    # report on nan values in vrad columns
    for colname in nan_vrad:
        wmsg = '{0}/{1} rows(s) of {2} have NaN drift values'
        log.warning(wmsg.format(nan_vrad[colname], len(rdb_table), colname))
    # ---------------------------------------------------------------------
    # convert rdb_dict4 to table
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
def check_data_type(data_type: Optional[str]):
    """
    Check the data type is correct - must be SCIENCE or FP or LFC

    param data_type: str or None, the data type set by user

    :return: None, just raises error if incorrect
    """
    if data_type in [None, 'None', '', 'Null']:
        emsg = ('DATA_TYPE must be SCIENCE or FP or LFC. '
                'DATA_TYPE currently unset')
        raise base_classes.LblException(emsg)

    if data_type not in ['SCIENCE', 'FP', 'LFC']:
        emsg = ('DATA_TYPE must be SCIENCE or FP or LFC. '
                'DATA_TYPE = {0}').format(data_type)
        raise base_classes.LblException(emsg)


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
    m_wavemap = io.load_fits(out_wavefile)
    m_spectrum = io.load_fits(out_model)
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

    # -------------------------------------------------------------------------
    if not inst.params['USE_SAVGOL_TEMPLATE']:
        t_flux = np.array(template_table['flux'])
        # smooth the spectrum to avoid lines that coincide with small-scale noise
        #   excursion
        t_flux_tmp = np.zeros_like(t_flux)
        for offset in range(-2, 3):
            t_flux_tmp += np.roll(t_flux, offset)
        # copy over original vector
        t_flux = np.array(t_flux_tmp)
        # find the first and second derivative of the flux
        dflux = np.gradient(t_flux)
        ddflux = np.gradient(dflux)
    else:
        t_flux = np.array(template_table['flux_savgol_d0'])
        dflux = np.array(template_table['flux_savgol_d1'])
        ddflux = np.array(template_table['flux_savgol_d2'])
    # -------------------------------------------------------------------------
    with warnings.catch_warnings(record=True) as _:
        t_snr = t_flux / template_table['rms']
        # remove infinite values
        t_snr[np.isinf(t_snr)] = np.nan
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
    table['ll_mask_s'] = np.array(ll_mask_s)
    table['ll_mask_e'] = np.array(ll_mask_e)
    table['w_mask'] = np.array(w_mask)
    table['value'] = np.array(f_mask)
    table['depth'] = np.array(depth)
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
    log.general(msg.format(inst.params['OBJECT_COMPARISON'], sys_vel))
    # -------------------------------------------------------------------------
    # ccf plot
    plot.mask_plot_ccf(inst, dvs, ccf, sys_vel)
    # -------------------------------------------------------------------------
    # return the systemic velocity
    return sys_vel


def find_model_mask_lines(inst: InstrumentsType, m_wavemap: np.ndarray,
                          m_spectrum: np.ndarray) -> Table:
    """
    Get the mask lines from the stellar model (MASK_FROM_MODEL): the model
    is convolved to MASK_MODEL_RESOLUTION, resampled on a constant velocity
    grid and its lines are found as for a template (find_mask_lines), between
    MASK_MODEL_WAVE_MIN and MASK_MODEL_WAVE_MAX. The lines are in the rest
    frame of the model.

    :param inst: the instrument class
    :param m_wavemap: np.ndarray, the 1D model wave map [nm]
    :param m_spectrum: np.ndarray, the 1D model flux spectrum

    :return: astropy table, the table of lines
    """
    # get parameters from instrument class
    params = inst.params
    resolution = params['MASK_MODEL_RESOLUTION']
    wavemin = params['MASK_MODEL_WAVE_MIN']
    wavemax = params['MASK_MODEL_WAVE_MAX']
    # resolution element (FWHM) in m/s
    fwhm = speed_of_light_ms / resolution
    # grid step: 6 points per resolution element
    dv_grid = fwhm / 6
    # domain padded by 1000 km/s so that the edges of the convolution and of
    #   the derivatives are outside the mask
    wave0 = mp.doppler_shift(wavemin, 1000e3)
    wave1 = mp.doppler_shift(wavemax, -1000e3)
    if wave0 < np.min(m_wavemap) or wave1 > np.max(m_wavemap):
        emsg = ('Stellar model ({0:.1f} to {1:.1f} nm) does not cover the '
                'model mask domain (MASK_MODEL_WAVE_MIN={2}, '
                'MASK_MODEL_WAVE_MAX={3})')
        margs = [np.min(m_wavemap), np.max(m_wavemap), wavemin, wavemax]
        raise LblException(emsg.format(*margs))
    # print progress
    msg = 'Convolving the stellar model to R={0:.0f} ({1:.1f} to {2:.1f} nm)'
    log.general(msg.format(resolution, wavemin, wavemax))
    # the model is interpolated on a 4 times finer constant velocity grid
    #   and convolved there. It is converted to photons (the model is an
    #   energy flux, the templates are in photons)
    fine_grid = get_magic_grid(wave0, wave1, dv_grid=dv_grid / 4)
    keep = (m_wavemap > 0.99 * wave0) & (m_wavemap < 1.01 * wave1)
    fine_flux = np.interp(fine_grid, m_wavemap[keep],
                          m_spectrum[keep] * m_wavemap[keep])
    # gaussian kernel of the resolution element (in fine grid points)
    sigma = fwhm / mp.fwhm_value() / (dv_grid / 4)
    xkernel = np.arange(-int(np.ceil(5 * sigma)), int(np.ceil(5 * sigma)) + 1)
    kernel = np.exp(-0.5 * (xkernel / sigma) ** 2)
    fine_flux = np.convolve(fine_flux, kernel / np.sum(kernel), mode='same')
    # model on the constant velocity grid of the mask (normalized flux)
    grid = get_magic_grid(wave0, wave1, dv_grid=dv_grid)
    flux = np.interp(grid, fine_grid, fine_flux)
    flux = flux / np.median(flux)
    # -------------------------------------------------------------------------
    # model as a template table: the noise-free model gets a nominal SNR per
    #   pixel and the savgol derivatives are computed at the resolution of
    #   the model (as lbl_template does at the instrument resolution)
    model_table = Table()
    model_table['wavelength'] = grid
    model_table['flux'] = flux
    model_table['rms'] = flux / params['MASK_MODEL_SNR']
    savgol_fluxes = inst.calculate_savgol_template(dv_grid=dv_grid,
                                                   flux_dict=dict(flux=flux),
                                                   approx_res=resolution)
    for key in savgol_fluxes:
        model_table[key] = savgol_fluxes[key]
    # find the lines as for a template
    line_table = find_mask_lines(inst, model_table)
    # keep the lines within the mask domain
    keep = line_table['ll_mask_s'] > wavemin
    keep &= line_table['ll_mask_s'] < wavemax
    # keep the lines deeper than MASK_MODEL_MIN_DEPTH (the noise-free model
    #   has many shallow features that are not seen in the data)
    keep &= np.abs(line_table['depth']) > params['MASK_MODEL_MIN_DEPTH']
    # return the mask table
    return line_table[keep]


def model_mask_header(inst: InstrumentsType) -> io.LBLHeader:
    """
    Header keys of a mask built from the stellar model: the model file, the
    resolution, the wavelength domain and the minimum line depth

    :param inst: the instrument class

    :return: io.LBLHeader, the header keys
    """
    # get parameters from instrument class
    params = inst.params
    # the stellar model file used
    fkwargs = inst.get_stellar_model_format_dict(params)
    model_file = params['STELLAR_MODEL_FILE'].format(**fkwargs)
    # push into header
    hdict = io.LBLHeader()
    hdict = inst.set_hkey(hdict, 'KW_MASK_MODEL', model_file)
    hdict = inst.set_hkey(hdict, 'KW_MASK_MODEL_RES',
                          float(params['MASK_MODEL_RESOLUTION']))
    hdict = inst.set_hkey(hdict, 'KW_MASK_MODEL_WMIN',
                          float(params['MASK_MODEL_WAVE_MIN']))
    hdict = inst.set_hkey(hdict, 'KW_MASK_MODEL_WMAX',
                          float(params['MASK_MODEL_WAVE_MAX']))
    hdict = inst.set_hkey(hdict, 'KW_MASK_MODEL_MINDEPTH',
                          float(params['MASK_MODEL_MIN_DEPTH']))
    # return the header keys
    return hdict


def check_model_mask(inst: InstrumentsType, mask_file: str):
    """
    Warn if an existing mask built from the stellar model was built with a
    different model (e.g. log g), resolution, wavelength domain or minimum
    line depth: the mask name only contains the model temperature

    :param inst: the instrument class
    :param mask_file: str, the mask file

    :return: None, logs a warning for each difference
    """
    # load the mask header
    mask_hdr = inst.load_header(mask_file, kind='mask fits file')
    # compare with the current model mask settings
    expected = model_mask_header(inst)
    for key in expected:
        value = mask_hdr[key] if key in mask_hdr else None
        if value != expected[key]:
            msg = ('Mask {0} was built with {1}={2} (current settings: {3}). '
                   'Set --overwrite to recalculate mask')
            log.warning(msg.format(mask_file, key, value, expected[key]))


def template_systemic_velocity(inst: InstrumentsType, template_file: str,
                               models_dir: str) -> float:
    """
    Systemic velocity of a template relative to the stellar model, measured
    as lbl_mask does for the masks built from the template (lines of the
    template correlated with the model). Used with the masks built from the
    model (in the rest frame of the model, shared by all objects)

    :param inst: the instrument class
    :param template_file: str, the template file
    :param models_dir: str, the model directory

    :return: float, the systemic velocity in km/s
    """
    # load the template
    template_table = inst.load_template(template_file)
    # get the stellar model
    m_wavemap, m_spectrum = get_stellar_models(inst, models_dir)
    # find the lines of the template
    line_table = find_mask_lines(inst, template_table)
    # return the systemic velocity of the template lines against the model
    return mask_systemic_velocity(inst, line_table, m_wavemap, m_spectrum)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

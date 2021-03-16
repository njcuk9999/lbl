#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-16

@author: cook
"""
from astropy.io import fits
from astropy import constants
from astropy.table import Table
import numpy as np
from typing import Dict, List, Tuple, Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.general.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = default.Instrument
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
# get speed of light
speed_of_light_ms = constants.c.value


# =============================================================================
# Define functions
# =============================================================================
def make_ref_dict(inst: Instrument, reftable_file: str,
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
        # ratio of expected VS actual RMS in difference of model vs line
        ref_dict['RMSRATIO'] = np.array(table['RMSRATIO'])
        # effective number of pixels in line
        ref_dict['NPIXLINE'] = np.array(table['NPIXLINE'])
        # mean line position in pixel space
        ref_dict['MEANPIX'] = np.array(table['MEANPIX'])
        # blaze value compared to peak for that order
        ref_dict['MEANBLAZE'] = np.array(table['MEANBLAZE'])
        # amp continuum
        ref_dict['AMP_CONTINUUM'] = np.array(table['AMP_CONTIUNUUM'])
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
        # Question: always the same wave length solution ?
        # Question: Should this wave solution come from calib ?
        # load wave solution from first science file
        wavegrid = inst.get_wave_solution(science_files[0])
        # storage for vectors
        order, wave_start, wave_end, weight_line, xpix = [], [], [], [], []
        # loop around orders
        for order_num in range(wavegrid.shape[0]):
            # get the min max wavelengths for this order
            min_wave = np.min(wavegrid[order_num])
            max_wave = np.min(wavegrid[order_num])
            # build a mask for mask lines in this order
            good = mask_table['ll_mask_s'] > min_wave
            good &= mask_table['ll_mask_s'] < max_wave
            # if we have values then add to arrays
            if np.sum(good) > 0:
                # add an order flag
                order += list(np.repeat(order_num, np.sum(good)))
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
        # make xpix a numpy array
        xpix = np.array(xpix)
        # add to reference dictionary
        ref_dict['ORDER'] = np.array(order)
        ref_dict['WAVE_START'] = np.array(wave_start)
        ref_dict['WAVE_END'] = np.array(wave_end)
        ref_dict['WEIGHT_LINE'] = np.array(weight_line)
        ref_dict['XPIX'] = xpix
        # ratio of expected VS actual RMS in difference of model vs line
        ref_dict['RMSRATIO'] = np.zeros_like(xpix, dtype=float)
        # effective number of pixels in line
        ref_dict['NPIXLINE'] = np.zeros_like(xpix,dtype = int)
        # mean line position in pixel space
        ref_dict['MEANPIX'] = np.zeros_like(xpix,dtype = float)
        # blaze value compared to peak for that order
        ref_dict['MEANBLAZE'] = np.zeros_like(xpix,dtype = float)
        # amp continuum
        ref_dict['AMP_CONTINUUM'] = np.zeros_like(xpix,dtype = float)
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
        log.logger.info('Writing ref table {0}'.format(reftable_file))
        # write to file
        io.write_table(reftable_file, ref_table)

    # -------------------------------------------------------------------------
    # return table (either loaded from file or constructed from mask +
    #               wave solution)
    return ref_dict


def spline_template(inst: Instrument, template_file: str,
                    systemic_vel: float) -> Dict[str, mp.IUVSpline]:
    """
    Calculate all the template splines (for later use)

    :param inst: Instrument instance
    :param template_file: str, the absolute path to the template file
    :param systemic_vel: float, the systemic velocity
    :return:
    """
    # get the pixel hp_width
    hp_width = inst.params['HP_WIDTH']
    # load the template
    template_table = inst.load_template(template_file)
    # get properties from template table
    twave = np.array(template_table['wave'])
    tflux = np.array(template_table['flux'])
    tflux0 = np.array(template_table['flux'])
    # -------------------------------------------------------------------------
    # work out the velocity scale
    dwave = np.gradient(twave)
    dvelo = 1 / mp.nanmedian(twave / dwave) / speed_of_light_ms
    # velocity pixel scale (to nearest pixel)
    width = int(hp_width / dvelo)
    # make sure pixel width is odd
    if width % 2 == 0:
        width += 1
    # -------------------------------------------------------------------------
    # we high-pass on a scale of ~101 pixels in the e2ds
    tflux -= mp.lowpassfilter(tflux, width=width)
    # -------------------------------------------------------------------------
    # get the gradient of the log of the wave
    grad_log_wave = np.gradient(np.log(twave))
    # get the derivative of the flux
    dflux = np.gradient(tflux) / grad_log_wave / speed_of_light_ms
    # get the 2nd derivative of the flux
    ddflux = np.gradient(dflux) / grad_log_wave / speed_of_light_ms
    # get the 3rd derivative of the flux
    dddflux = np.gradient(ddflux) / grad_log_wave / speed_of_light_ms
    # -------------------------------------------------------------------------
    # we create the spline of the template to be used everywhere later
    valid = np.isfinite(tflux) & np.isfinite(dflux)
    valid &= np.isfinite(ddflux) & np.isfinite(dddflux)
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
    sps['ddspline'] = mp.iuv_spline(ntwave, ddflux[valid], k=k_order, ext=1)
    sps['dddspline'] = mp.iuv_spline(ntwave, dddflux[valid], k=k_order, ext=1)
    # -------------------------------------------------------------------------
    # we create a mask to know if the splined point  is valid
    tmask = np.isfinite(tflux).astype(float)
    sps['spline_mask'] = mp.iuv_spline(ntwave, tmask, k=1, ext=1)
    # -------------------------------------------------------------------------
    # return splines
    return sps





# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

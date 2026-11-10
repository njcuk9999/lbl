#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
import os
import traceback
import warnings
from functools import partial
from typing import List

import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import provenance
from lbl.core import fastmath
from lbl.core import math as mp
from lbl.core import vetted
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import apero
from lbl.science import general
from lbl.science import planets
from lbl.science import template_scatter

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
# add arguments (must be in parameters.py)
ARGS_TEMPLATE = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'TEMPLATE_SUBDIR', 'SCIENCE_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_COMPARISON', 'BLAZE_FILE', 'BLAZE_CORRECTED',
    # other
    'VERBOSE', 'PROGRAM',
]

DESCRIPTION_TEMPLATE = 'Use this code to create the LBL template'


# =============================================================================
# Define functions
# =============================================================================
def berv_bins(berv: np.ndarray, nbin: int) -> np.ndarray:
    """
    The bin of each file: bins of equal size, ordered by BERV. The bin follows
    the rank of the file's BERV, so that files with nearly the same BERV share
    a bin and a telluric residual stays at one BERV in the median over bins.

    :param berv: np.ndarray, the BERV of each file
    :param nbin: int, the number of bins

    :return: np.ndarray of int, the bin of each file, 0 to nbin - 1
    """
    # argsort gives the files in order of BERV; its argsort is each file's rank
    rank = np.argsort(np.argsort(berv))
    return np.array(nbin * (rank / len(berv)), dtype=int)


def lowpass_points(hp_width_kms: float, grid_step_ms: float) -> int:
    """
    The width of a low pass filter in points of a grid uniform in velocity

    :param hp_width_kms: float, the width in km/s (HP_WIDTH)
    :param grid_step_ms: float, the step of the grid in m/s

    :return: int, the width in points of the grid
    """
    return int(np.round(float(hp_width_kms) * 1000 / float(grid_step_ms)))


def file_level(flux: np.ndarray, weight: np.ndarray) -> float:
    """
    The flux level of one file's s1d: its median flux per unit weight (1 if
    it has no valid point or a level that is not positive)

    :param flux: np.ndarray, the s1d flux (NaN where the file has no data)
    :param weight: np.ndarray, the s1d weight

    :return: float, the level
    """
    good = np.isfinite(flux) & np.isfinite(weight) & (weight > 0)
    if not np.any(good):
        return 1.0
    level = float(np.median(flux[good] / weight[good]))
    if not np.isfinite(level) or level <= 0:
        return 1.0
    return level


def add_to_bin(flux_bin: np.ndarray, weight_bin: np.ndarray,
               norm_bin: np.ndarray, flux: np.ndarray, weight: np.ndarray,
               level: float):
    """
    Add one file's s1d to its bin (in place). A point the file does not have
    is left out for this file only: neither its flux nor its weight is
    added there, and the other files of the bin still make that point.

    The bin is flux_bin / norm_bin, the mean of flux / (weight * level) over
    the files that have the point, weighted by weight * level. The level
    (file_level) is what makes it a mean: with weight alone, a point that
    only some files have would move to their flux level and leave a step
    in the bin where a brighter or fainter file is missing. weight_bin is the
    sum of the weights, for the threshold on the weights.

    :param flux_bin: np.ndarray, the bin's summed flux
    :param weight_bin: np.ndarray, the bin's summed weight
    :param norm_bin: np.ndarray, the bin's summed weight times level
    :param flux: np.ndarray, the file's s1d flux (NaN where it has no data)
    :param weight: np.ndarray, the file's s1d weight
    :param level: float, the file's flux level (file_level)
    """
    valid = np.isfinite(flux) & np.isfinite(weight)
    flux_bin[valid] += flux[valid]
    weight_bin[valid] += weight[valid]
    norm_bin[valid] += weight[valid] * level


def main(**kwargs):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param kwargs: kwargs to parse to instrument - anything in params can be
                   parsed (overwrites instrumental and default parameters)
    :return:
    """
    # deal with parsing arguments
    args = select.parse_args(ARGS_TEMPLATE, kwargs, DESCRIPTION_TEMPLATE)
    # load instrument
    inst = select.load_instrument(args, plogger=log)
    # get data directory
    data_dir = io.check_directory(inst.params['DATA_DIR'])
    # move log file (now we have data directory)
    lbl_misc.move_log(data_dir, __NAME__)
    # print splash
    lbl_misc.splash(name=__STRNAME__, instrument=inst.name,
                    params=args, plogger=log)
    # run __main__
    try:
        namespace = __main__(inst)
    except LblException as e:
        tbmsg = traceback.format_exc()
        emsg = '{0}\n\nTraceback:\n{1}'
        eargs = [e.message, tbmsg]
        raise LblException(emsg.format(*eargs), verbose=False) from e
    except Exception as e:
        tbmsg = traceback.format_exc()
        emsg = 'Unexpected {0} error: {1}: {2}\n\nTraceback:\n{3}'
        eargs = [__NAME__, type(e), str(e), tbmsg]
        raise LblException(emsg.format(*eargs)) from e
    # end code
    lbl_misc.end(__NAME__, plogger=log)
    # return local namespace
    return namespace


def __main__(inst: InstrumentsType, **kwargs):
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or inst.params is None:
        # deal with parsing arguments
        args = select.parse_args(ARGS_TEMPLATE, kwargs, DESCRIPTION_TEMPLATE)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg

    # must run the template on BOTH the science and comparison objects
    # if science and comparison are the same we just run one template
    if inst.params['OBJECT_SCIENCE'] == inst.params['OBJECT_COMPARISON']:
        objnames = [str(inst.params['OBJECT_SCIENCE'])]
        objkinds = ['science']
    # if comparison is not set, we just run one template on the science object
    elif inst.params['OBJECT_COMPARISON'] in [None, '', 'None', 'Null']:
        inst.params['OBJECT_COMPARISON'] = str(inst.params['OBJECT_SCIENCE'])
        objnames = [str(inst.params['OBJECT_SCIENCE'])]
        objkinds = ['science']
    # otherwise we run both
    else:
        objnames = [str(inst.params['OBJECT_SCIENCE']),
                    str(inst.params['OBJECT_COMPARISON'])]
        objkinds = ['science', 'comparison']
    # loop around object names
    for objname, objkind in zip(objnames, objkinds):
        # print a printout to show which object with are making a template for
        # only if we have more than one templates to loop around
        if len(objnames) > 1:
            log.info('*'*50)
            lmsg = 'Running template for {0} object: {1}'.format(objkind, objname)
            log.info(lmsg)
            log.info('*' * 50)
        # run the full template code
        with io.fast_fits_config():
            run_template(inst, objname, objkind)
    # return all local variables (for debug)
    return locals()


def _load_template_header(inst, filename: str):
    """
    Science header for the template loop: only the keys the template uses
    if the instrument knows them (same values), else the full header
    """
    keys = inst.template_header_keys()
    if keys is None:
        return inst.load_header(filename)
    return inst.load_header_keys(filename, keys)


def _load_template_science(inst, filename: str):
    """
    Science data and header for the template loop: only the header keys the
    template uses if the instrument knows them (same values), else the full
    header
    """
    keys = inst.template_header_keys()
    if keys is None:
        return inst.load_science_file(filename)
    return inst.load_science_file_keys(filename, keys)


def _nanpercentile_rows(cube: np.ndarray, percents: List[float]
                        ) -> np.ndarray:
    """
    np.nanpercentile(cube, percents, axis=1), with a numba kernel for 2D
    float64 cubes (same values)
    """
    if cube.ndim == 2 and cube.dtype == np.float64:
        # the fractions np.nanpercentile uses
        qs = np.true_divide(np.asarray(percents), np.float64(100))
        return fastmath.nanpercentile_rows(np.ascontiguousarray(cube), qs)
    return np.nanpercentile(cube, percents, axis=1)


def _bervbin_template(inst, science_files, blaze, calib_dir, wavegrid,
                      grid_step_magic, hp_width, rv0, drv0, berv, nbin,
                      planets_used, flat_rv, n_passes, tqdm):
    """
    The template as the median of the spectra resampled to s1d (e2ds_to_s1d)
    and stacked in nbin bins of BERV (TEMPLATE_METHOD = 'BERVBIN'), the
    velocities iterated on in up to n_passes passes

    :return: dict(p50, p50_odd, p50_even, rms, rms_odd, rms_even,
             sci_table, rv0, berv)
    """
    # the velocity corrections are updated in place: a copy
    drv0 = np.array(drv0)
    # storage of the science table
    med_spec_hp = 1
    # we distribute the files in bins of equal size ordered by berv
    # Files with nearly the same berv will be in the same bin
    ibin = berv_bins(berv, nbin)
    # storage of the science table - reset
    sci_table = dict()
    # iterate 5 times
    for ite in range(5):
        # create a cube that contains one line for each file
        flux_cube = np.zeros([len(wavegrid), nbin])
        # we also keep track of odd/even order contributions
        odd_cube = np.zeros([len(wavegrid), nbin])
        even_cube = np.zeros([len(wavegrid), nbin])

        # weight cube to account for order overlap
        weight_cube = np.zeros([len(wavegrid), nbin])
        # same odd/even with the weights
        odd_weight_cube = np.zeros([len(wavegrid), nbin])
        even_weight_cube = np.zeros([len(wavegrid), nbin])
        # the weights times each file's flux level, that the fluxes are
        #   divided by (add_to_bin)
        norm_cube = np.zeros([len(wavegrid), nbin])
        odd_norm_cube = np.zeros([len(wavegrid), nbin])
        even_norm_cube = np.zeros([len(wavegrid), nbin])

        # print iteration loop
        msg = 'Iteration {0} of {1}'
        log.info(msg.format(ite + 1, n_passes))
        # science table
        sci_table = dict()
        # all bervs
        berv = np.zeros_like(science_files, dtype=float)
        # loop around files, a progress bar for the files (nothing else is
        #   printed in the loop, the shifts are summed up after it)
        desc = 'E2DS->S1D, iteration {0} of {1}'.format(ite + 1, n_passes)
        for it, filename in enumerate(tqdm(science_files, desc=desc,
                                           unit='file')):
            # select the first science file as a reference file
            sci_image, sci_hdr = _load_template_science(inst, filename)

            # get wave solution for reference file
            sci_wave = inst.get_wave_solution(filename, sci_image, sci_hdr)

            # load blaze (just ones if not needed)
            if blaze is None:
                bargs = [filename, sci_image, sci_hdr, calib_dir]
                bout = inst.load_blaze_from_science(*bargs, normalize=False)
                blazeimage, blaze_flag = bout
            # test for all ones (no blaze)
            elif np.sum(blaze.ravel()) == len(blaze.ravel()):
                blaze_flag = True
                blazeimage = np.array(blaze)
            else:
                blaze_flag = False
                blazeimage = np.array(blaze)
            # deal with not having blaze (for s1d weighting)
            if blaze_flag:
                sci_image, blazeimage = inst.no_blaze_corr(sci_image, sci_wave)
            # get the berv
            berv[it] = inst.get_berv(sci_hdr)
            # populate science table
            sci_table = inst.populate_sci_table(filename, sci_table, sci_hdr,
                                                berv=berv[it])

            # apply berv if required, with the star's velocity, the two
            #   composed relativistically
            if berv[it] != 0.0 or planets_used is not None:
                sci_wave = mp.doppler_shift(
                    sci_wave, mp.add_velocities(-berv[it], rv0[it]))
            # set exactly zeros to NaNs
            sci_image[sci_image == 0] = np.nan
            # compute s1d from e2ds (all orders, odd orders, even orders)
            if ite == 0:
                s1d_all = apero.e2ds_to_s1d(inst.params, sci_wave, sci_image,
                                            blazeimage, wavegrid, parity=True)
            # if this is not the first iteration we recompute the wave grid
            #   given what we know from the previous loop
            if ite != 0:
                # if we are not at ite ==0, we scan to find an RV offset to
                # register all spectra
                dv = np.arange(-50, 50)
                sig = np.zeros_like(dv, dtype=float)
                # We use the central point of the slope, which is always
                # selected as to be the cleanest, with most RV content, and
                # with the highest S/N. We use a narrow domain around the point
                # for the optimisation of template RV shift.
                cut_low = inst.params['COMPIL_SLOPE_REF_WAVE'] * 0.95
                cut_high = inst.params['COMPIL_SLOPE_REF_WAVE'] * 1.05
                rms_domain = (wavegrid > cut_low) * (wavegrid < cut_high)
                # s1d from e2ds, only needed in rms_domain (orders that do
                #   not reach it are skipped)
                rms_index = np.where(rms_domain)[0]
                s1d_flux, _ = apero.e2ds_to_s1d(inst.params, sci_wave,
                                                sci_image, blazeimage,
                                                wavegrid,
                                                domain=(rms_index[0],
                                                        rms_index[-1] + 1))
                # domain that is at the center of the detector
                s1d_flux_tmp = s1d_flux[rms_domain]
                # we remove the low frequency content
                s1d_flux_tmp /= mp.lowpassfilter(s1d_flux_tmp, hp_width)
                # get the med spec for this domain
                med_spec_hp_domain = med_spec_hp[rms_domain]
                # -------------------------------------------------------------
                # perform a sigma clip of the edges (to avoid dummy edge
                #    effects)
                #   for each shift: np.nanpercentile of
                #   np.roll(s1d_flux_tmp, shift) / med_spec_hp_domain
                #   at 16 and 84 (numba kernel, same values)
                qs = np.true_divide(np.asarray([16, 84]), np.float64(100))
                sig[:] = fastmath.roll_ratio_sigma(
                    np.ascontiguousarray(s1d_flux_tmp, dtype=np.float64),
                    np.ascontiguousarray(med_spec_hp_domain,
                                         dtype=np.float64),
                    dv.astype(np.int64), qs[0], qs[1])
                sig /= np.nanmedian(sig)
                imin = np.argmin(sig)
                # just to avoid dummy edge effects
                if imin == 0:
                    imin = 1
                if imin == len(sig) - 1:
                    imin = len(sig) - 2
                # -------------------------------------------------------------
                # fit the dv to get the approximate RV shift
                fit = np.polyfit(dv[imin - 1:imin + 2], sig[imin - 1:imin + 2], 2)
                drv0[it] = -fit[1] / (2 * fit[0]) * grid_step_magic

                # subtract off this rv shift from our rv
                rv0[it] -= drv0[it]
                # get wave solution for reference file.
                sci_wave = inst.get_wave_solution(filename, sci_image, sci_hdr)
                sci_wave = mp.doppler_shift(
                    sci_wave, mp.add_velocities(-berv[it], rv0[it]))

                # compute s1d from e2ds (all, odd and even orders)
                # with an updated wavelength grid
                s1d_all = apero.e2ds_to_s1d(inst.params, sci_wave, sci_image,
                                            blazeimage, wavegrid, parity=True)

            # odd and even orders see the updated wavelength grid if we are
            #   not at ite ==0
            s1d_flux, s1d_weight = s1d_all[0], s1d_all[1]
            s1d_flux_odd, s1d_weight_odd = s1d_all[2], s1d_all[3]
            s1d_flux_even, s1d_weight_even = s1d_all[4], s1d_all[5]

            # push into arrays: a point this file does not have (NaN) is
            #   left out of its bin for this file only (add_to_bin), all
            #   orders, odd orders and even orders with the level of the
            #   file's s1d of all orders
            level = file_level(s1d_flux, s1d_weight)
            add_to_bin(flux_cube[:, ibin[it]], weight_cube[:, ibin[it]],
                       norm_cube[:, ibin[it]], s1d_flux, s1d_weight, level)
            add_to_bin(odd_cube[:, ibin[it]], odd_weight_cube[:, ibin[it]],
                       odd_norm_cube[:, ibin[it]], s1d_flux_odd,
                       s1d_weight_odd, level)
            add_to_bin(even_cube[:, ibin[it]], even_weight_cube[:, ibin[it]],
                       even_norm_cube[:, ibin[it]], s1d_flux_even,
                       s1d_weight_even, level)

        # -------------------------------------------------------------------------
        # Step 6. Creation of the template
        # -------------------------------------------------------------------------
        # get bad weight threshold from params
        min_weight = inst.params['TEMPLATE_WEIGHT_MIN']
        cubes = ((flux_cube, weight_cube, norm_cube),
                 (odd_cube, odd_weight_cube, odd_norm_cube),
                 (even_cube, even_weight_cube, even_norm_cube))
        for cube, wcube, ncube in cubes:
            # points are not valid where no file of the bin has them (weight
            #   zero), where the flux is exactly zero, or where the summed
            #   weight is under TEMPLATE_WEIGHT_MIN
            bad_domain = (wcube == 0) | (cube == 0) | (wcube < min_weight)
            cube[bad_domain] = np.nan
            ncube[bad_domain] = 1
        # ---------------------------------------------------------------------
        # print progress
        log.general('Calculating template')
        # divide by the weights times the files' levels: the weighted mean of
        #   the files of each bin that have the point (add_to_bin), which also
        #   corrects for overlapping orders
        flux_cube = flux_cube / norm_cube
        odd_cube = odd_cube / odd_norm_cube
        even_cube = even_cube / even_norm_cube

        # normalize each slice of the cube by its median
        for it in tqdm(range(flux_cube.shape[1])):
            med_norm = np.nanmedian(flux_cube[:, it])
            flux_cube[:, it] = flux_cube[:, it] / med_norm
            odd_cube[:, it] = odd_cube[:, it] / med_norm
            even_cube[:, it] = even_cube[:, it] / med_norm

        # copy
        flux_cube0 = np.array(flux_cube)
        # -------------------------------------------------------------------------
        # applying low pass filter
        log.general('\tApplying low pass filter to cube')
        # deal with science
        if inst.params['DATA_TYPE'] == 'SCIENCE':
            with warnings.catch_warnings(record=True) as _:
                # calculate the median of the big cube
                med_spec = mp.nanmedian(flux_cube, axis=1)
                med_spec_hp = med_spec / mp.lowpassfilter(med_spec, hp_width)

                # iterate until low frequency gone
                for sci_it in tqdm(range(flux_cube.shape[1])):
                    # remove the stellar features
                    ratio = flux_cube[:, sci_it] / med_spec
                    # apply median filtered ratio (low frequency removal)
                    lowpass = mp.lowpassfilter(ratio, hp_width)
                    flux_cube[:, sci_it] /= lowpass
                    # deal with left and right. Same lowpass for both
                    odd_cube[:, sci_it] /= lowpass
                    even_cube[:, sci_it] /= lowpass

        else:
            with warnings.catch_warnings(record=True) as _:
                # calculate the median of the big cube
                med_spec = mp.nanmedian(flux_cube, axis=1)
                med_spec_hp = med_spec / mp.lowpassfilter(med_spec, hp_width)
                # mask to keep only FP peaks and avoid dividing
                # two small values (minima between lines in median and
                # individual spectrum) when computing the lowpass
                peaks = med_spec > mp.lowpassfilter(med_spec, hp_width)
                # iterate until low frequency gone
                for sci_it in tqdm(range(flux_cube.shape[1])):
                    # remove the stellar features
                    ratio = flux_cube[:, sci_it] / med_spec
                    ratio[~peaks] = np.nan

                    # apply median filtered ratio (low frequency removal)
                    lowpass = mp.lowpassfilter(ratio, hp_width)
                    flux_cube[:, sci_it] /= lowpass
                    # deal with left and right. Same lowpass for both
                    odd_cube[:, sci_it] /= lowpass
                    even_cube[:, sci_it] /= lowpass

        if ite != 0:
            # determine the sigma of RV shifts between templates
            n1, p1 = np.nanpercentile(drv0, [16, 84])
            sig = (p1 - n1) / 2.0
            msg = 'Looping on files; relative RV shift RMS {:.1f}m/s'
            log.general(msg.format(sig))
            # if an RMS < 100m/s (or whatever we set as a value)
            # , we are happy, we exit the loop on relative RV shifts
            if sig < inst.params['MAX_CONVERGENCE_TEMPLATE_RV']:
                break
        # with the companions' velocities the spectra are already where they
        #   belong: one pass, no iteration on the velocities
        if planets_used is not None:
            log.general('PLANETS: template made in one pass, on BERV + the '
                        'Keplerian velocity of each spectrum')
            break
        # with a constant velocity too (SMALL_RV_SHIFT)
        if flat_rv:
            log.general('SMALL_RV_SHIFT: template made in one pass, on BERV '
                        'alone')
            break

    sci_table['VSYS'] = rv0
    if planets_used is not None:
        # the Keplerian velocity of each spectrum, kept in the template's
        #   table of spectra (VKEP)
        sci_table['VKEP'] = np.array(planets_used['velocity'])

    # -------------------------------------------------------------------------
    # bin cube by BERV (to give equal weighting to epochs)
    # -------------------------------------------------------------------------
    # get minimum number of berv bins
    nmin_bervbin = inst.params['BERVBIN_MIN_ENTRIES']
    # get the size of the berv bins
    bervbin_size = inst.params['BERVBIN_SIZE']

    # -------------------------------------------------------------------------
    # get the median and +/- 1 sigma values for the cube
    # -------------------------------------------------------------------------
    log.general('Calculate 16th, 50th and 84th percentiles')
    with warnings.catch_warnings(record=True) as _:
        # to get statistics on the ber-bin rms, we need more than 3
        # bervbins
        log.general('computation done per-berv bin')
        log.general('\t- computation on flux cube')
        p16, p50, p84 = _nanpercentile_rows(flux_cube, [16, 50, 84])
        # same for left and right
        log.general('\t- computation on odd cube')
        p16_odd, p50_odd, p84_odd = _nanpercentile_rows(odd_cube, [16, 50, 84])
        log.general('\t- computation on even cube')
        p16_even, p50_even, p84_even = _nanpercentile_rows(even_cube, [16, 50, 84])
        # calculate the rms of each wavelength element
        rms = (p84 - p16) / 2
        # same for left and right
        rms_odd = (p84_odd - p16_odd) / 2
        rms_even = (p84_even - p16_even) / 2

    # -------------------------------------------------------------------------
    # TODO: Andd local SNR part from EA
    # Local SNR (infinite where the rms is zero)
    with warnings.catch_warnings(record=True) as _:
        snr_odd = p50_odd / rms_odd
        snr_even = p50_even / rms_even
    # Perform a low pass filter on the SNR
    snr_odd = mp.lowpassfilter(snr_odd, hp_width)
    snr_even = mp.lowpassfilter(snr_even, hp_width)
    # We reject domains that are below and SNR = 10
    # TODO -> makes this a global parameter
    # The minimal SNR required for a pixel considered to be valid
    # - only for science data
    if inst.params['DATA_TYPE'] == 'SCIENCE':
        # We determine the SNR from the -1 to +1 sigma equivalent distribution
        # of the input spectra
        snr_threshold = inst.params['TEMPLATE_SNR_THRES']
        low_snr_odd = snr_odd < snr_threshold
        p50_odd[low_snr_odd] = np.nan
        rms_odd[low_snr_odd] = np.nan
        low_snr_even = snr_even < snr_threshold
        p50_even[low_snr_even] = np.nan
        rms_even[low_snr_even] = np.nan
    return dict(p50=p50, p50_odd=p50_odd, p50_even=p50_even, rms=rms,
                rms_odd=rms_odd, rms_even=rms_even, sci_table=sci_table,
                rv0=rv0, berv=berv, flux_cube=flux_cube, ibin=ibin)


def run_template(inst, objname: str, objkind: str):
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # must force object science to object template
    inst.params['OBJECT_SCIENCE'] = str(objname)
    # check data type
    general.check_data_type(inst.params['DATA_TYPE'])
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst)
    template_dir, science_dir = dparams['TEMPLATE_DIR'], dparams['SCIENCE_DIR']
    calib_dir = dparams['CALIB_DIR']
    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # template filename
    template_file = inst.template_file(template_dir, objkind, required=False)
    # a vetted template is the depot's, never made again from spectra (it
    #   has none here): template_file put it there
    if vetted.is_vetted(objname):
        log.general('{0} is a vetted template: {1}'.format(objname,
                                                           template_file))
        return locals()

    # science filenames
    science_files = inst.science_files(science_dir)
    # -------------------------------------------------------------------------
    # Step 3: Check if template exists
    # -------------------------------------------------------------------------
    # the token of the spectra the template is made from (provenance): a
    #   template on disk made from other spectra (more were added, some
    #   removed) is made again. One made before tokens existed is kept
    #   unless a spectrum is newer than it, and is given its token
    # the method of the template (BSD, BERV Spectral Drizzle, for science
    #   spectra; the FP and the other calibrations are made with BERVBIN).
    #   SCATTER is the old name of BSD
    template_method = str(inst.params['TEMPLATE_METHOD']).upper()
    if template_method == 'SCATTER':
        template_method = 'BSD'
    if template_method not in ('BSD', 'BERVBIN'):
        raise LblException('TEMPLATE_METHOD must be BSD or BERVBIN, '
                           'not {0}'.format(template_method))
    if template_method == 'BSD' and inst.params['DATA_TYPE'] != 'SCIENCE':
        template_method = 'BERVBIN'
    # a BSD template is not one made from the same files by BERVBIN (the
    #   token keeps the method's old name: a template made before the
    #   rename is still the same)
    if template_method == 'BSD':
        inputs_token = provenance.token(
            science_files, extra=dict(method='SCATTER', version=1))
    else:
        inputs_token = provenance.token(science_files)
    exists = os.path.exists(template_file)
    if exists and provenance.recorded(template_file) is None and \
            not provenance.older_than(template_file, science_files):
        provenance.record(template_file, inputs_token, len(science_files))
    current = provenance.is_current(template_file, inputs_token)
    if exists and current and not inst.params['OVERWRITE']:
        # log that mask exist
        msg = 'Template {0} exists. Skipping template creation. '
        log.warning(msg.format(template_file))
        log.warning('Set --overwrite to recalculate mask')
        # return here
        return locals()
    elif exists and inst.params['OVERWRITE']:
        log.general(f'--overwrite=True. Recalculating template {template_file}')
    elif exists:
        log.general(f'Template {template_file} was made from other spectra '
                    f'than the {len(science_files)} there now. Calculating '
                    f'template.')
    else:
        log.general(f'Could not find {template_file}. Calculating template.')

    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames (after checking if template exists)
    # -------------------------------------------------------------------------
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    # load blaze file if set
    if blaze_file is not None:
        blaze = inst.load_blaze(blaze_file, science_file=str(science_files[0]),
                                normalize=False)
    else:
        blaze = None

    # -------------------------------------------------------------------------
    # Step 4: Deal with reference file (first file)
    # -------------------------------------------------------------------------
    # may need to filter out calibrations
    science_files = inst.filter_files(science_files)
    # select the most recent science file as a reference file
    refimage, refhdr = inst.load_science_file(science_files[-1])
    # get wave solution for reference file
    refwave = inst.get_wave_solution(science_files[-1], refimage, refhdr)
    # get domain coverage
    wavemin = inst.params['COMPIL_WAVE_MIN']
    wavemax = inst.params['COMPIL_WAVE_MAX']
    # work out a valid velocity step in m/s
    grid_step_magic = general.get_velocity_step(refwave)
    # grid scale for the template
    wavegrid = general.get_magic_grid(wave0=wavemin, wave1=wavemax,
                                      dv_grid=grid_step_magic)
    # the width of the low pass filters, HP_WIDTH [km/s] in points of this
    #   grid, the one they are applied on
    hp_width = lowpass_points(inst.params['HP_WIDTH'], grid_step_magic)
    # -------------------------------------------------------------------------
    # Step 5: Loop around each file and load into cube
    # -------------------------------------------------------------------------

    # We do not want to put to overwhelm the memory, so we construct the
    # flux cube with a hierarchcal binning. We first determine the number of
    # files and and we have too many, we stack every Nth file into a smaller
    # cube .

    # We first determine the number of files and we have too many, we stack
    # every Nth file into a smaller cube
    nmaxbins = inst.params['TEMPLATE_MEDBINMAX']
    nbin = np.min([nmaxbins, len(science_files)])
    # create a cube that contains one line for each file
    flux_cube = np.zeros([len(wavegrid), nbin])
    # we also keep track of odd/even order contributions
    odd_cube = np.zeros([len(wavegrid), nbin])
    even_cube = np.zeros([len(wavegrid), nbin])

    # weight cube to account for order overlap
    weight_cube = np.zeros([len(wavegrid), nbin])
    # same odd/even with the weights
    odd_weight_cube = np.zeros([len(wavegrid), nbin])
    even_weight_cube = np.zeros([len(wavegrid), nbin])
    # storage for the rv and drv0
    rv0 = np.zeros_like(science_files, dtype=float)
    drv0 = np.zeros_like(science_files, dtype=float)

    berv = np.zeros_like(science_files, dtype=float)
    # the time of each spectrum [BJD], for the companions' velocities
    bjd = np.zeros_like(science_files, dtype=float)
    use_planets = (bool(inst.params['PLANETS'])
                   and inst.params['DATA_TYPE'] == 'SCIENCE')
    for sci_it in range(len(science_files)):
        sci_hdr = _load_template_header(inst, science_files[sci_it])
        berv[sci_it] = inst.get_berv(sci_hdr)
        if use_planets:
            # RJD = JD - 2400000
            try:
                bjd[sci_it] = inst.get_rjd_value(sci_hdr) + 2400000.0
            except Exception as exc:
                msg = ('PLANETS: no time for {0} ({1}): the known '
                       'orbits are not used')
                log.warning(msg.format(science_files[sci_it], exc))
                use_planets = False
    # -------------------------------------------------------------------------
    # PLANETS: the known companions of the star. If the catalogs have their
    #   orbits, every spectrum is registered with its BERV and their velocity
    #   at its time, and the template is made in one pass; if not (None), the
    #   velocities are iterated on as below
    planets_used = None
    if use_planets:
        header_names = [refhdr.get(inst.params.get('KW_OBJNAME', 'OBJECT')),
                        refhdr.get('DRSOBJN')]
        planets_used = planets.orbits_for(
            planets.star_names(objname, header_names),
            inst.params['DATA_DIR'])
    if planets_used is not None:
        log.info('PLANETS: the template is made on BERV + the Keplerian '
                 'velocity of each spectrum, in one pass')
        rv0 = planets.star_velocity(bjd, planets_used['orbits'])
        planets_used['velocity'] = np.array(rv0)
        msg = ('PLANETS: velocity of the star from {0:.1f} to {1:.1f} m/s '
               'over the {2} spectra')
        log.info(msg.format(np.min(rv0), np.max(rv0), len(rv0)))
    # SMALL_RV_SHIFT: the star's velocity taken as constant, the template
    #   made in one pass on BERV alone (unless its orbits are known above);
    #   False iterates on the velocities below
    flat_rv = (planets_used is None and bool(inst.params['SMALL_RV_SHIFT'])
               and inst.params['DATA_TYPE'] == 'SCIENCE')
    if flat_rv:
        log.info('SMALL_RV_SHIFT: the velocity of the star is taken as '
                 'constant: the template is made on BERV alone, in one pass '
                 '(SMALL_RV_SHIFT = False iterates on the velocities)')
    # the passes: one when the velocities are known (or taken as
    #   constant), up to 5 otherwise
    n_passes = 1 if (planets_used is not None or flat_rv) else 5
    # the template: local polynomials through the pixels of all the spectra,
    #   none resampled (BSD), or the median of the spectra resampled and
    #   stacked in bins of BERV (BERVBIN, also for the FP)
    if template_method == 'BSD':
        built = template_scatter.build(
            inst, science_files, partial(_load_template_science, inst), blaze,
            calib_dir, wavegrid, berv, rv0, iterate_rv=(n_passes > 1),
            tqdm=tqdm)
        p50, p50_odd, p50_even = (built['flux'], built['flux_odd'],
                                  built['flux_even'])
        eflux, eflux_odd, eflux_even = (built['eflux'], built['eflux_odd'],
                                        built['eflux_even'])
        # the rms columns the mask reads as BERVBIN's: template_uncertainty
        #   takes sqrt(pi / 2) rms / sqrt(nbin) for the uncertainty of the
        #   template, which is eflux here
        to_rms = np.sqrt(nbin) / np.sqrt(np.pi / 2)
        rms, rms_odd, rms_even = (eflux * to_rms, eflux_odd * to_rms,
                                  eflux_even * to_rms)
        sci_table, rv0 = built['sci_table'], built['rv0']
        savgol_fluxes = built['savgol_fluxes']
        # the points of the odd and even templates under TEMPLATE_SNR_THRES
        #   (their SNR low passed), as BERVBIN
        snr_threshold = inst.params['TEMPLATE_SNR_THRES']
        for key, flux_p, rms_p in (('flux_odd', p50_odd, rms_odd),
                                   ('flux_even', p50_even, rms_even)):
            with warnings.catch_warnings(record=True) as _:
                snr = mp.lowpassfilter(flux_p / rms_p, hp_width)
                low_snr = ~(snr >= snr_threshold)
            flux_p[low_snr] = np.nan
            rms_p[low_snr] = np.nan
            for deriv in range(4):
                savgol_fluxes['{0}_savgol_d{1}'.format(key, deriv)][low_snr] = \
                    np.nan
        sci_table['VSYS'] = rv0
        if planets_used is not None:
            sci_table['VKEP'] = np.array(planets_used['velocity'])
    else:
        built = _bervbin_template(inst, science_files, blaze, calib_dir,
                                  wavegrid, grid_step_magic, hp_width, rv0,
                                  drv0, berv, nbin, planets_used, flat_rv,
                                  n_passes, tqdm)
        p50, p50_odd, p50_even = (built['p50'], built['p50_odd'],
                                  built['p50_even'])
        rms, rms_odd, rms_even = (built['rms'], built['rms_odd'],
                                  built['rms_even'])
        eflux, eflux_odd, eflux_even = rms, rms_odd, rms_even
        sci_table, rv0, berv = (built['sci_table'], built['rv0'],
                                built['berv'])
        savgol_fluxes = None
    # -------------------------------------------------------------------------
    # other parameters for the header
    nfiles = len(science_files)
    # TODO -- use the same as in APERO with the smart weight in BERV
    template_coverage = len(np.unique(berv // 1000))  # in km/s
    total_nobs_berv = len(np.unique(berv // 1000))  # in m/s

    # -------------------------------------------------------------------------
    # Step 7. check quality of data
    # -------------------------------------------------------------------------
    inst.check_quality_nan(refwave, wavegrid, [p50], 'flux')
    inst.check_quality_nan(refwave, wavegrid, [p50_odd, p50_even],
                           'flux_odd_even')

    # -------------------------------------------------------------------------
    # Step 8. Calculate Savitzky-Golay filtered template for better handling
    #         of higher derivatives
    # -------------------------------------------------------------------------
    # (BSD has them already: the derivatives of its local polynomials)
    if savgol_fluxes is None:
        savgol_fluxes = inst.calculate_savgol_template(
            dv_grid=grid_step_magic,
            flux_dict=dict(flux=p50, flux_odd=p50_odd, flux_even=p50_even))
    # set a key to tell us the type of template created
    if len(savgol_fluxes) > 0:
        if template_method == 'BSD':
            template_type = 'LBL_BSD'
        else:
            template_type = 'LBL_SAVGOL'
        # repeat quality control checks for savgol fluxes
        for col in savgol_fluxes:
            # skip odd cols (we will deal with odd/even in even cases) and the
            #   uncertainties
            if 'odd' in col or col.startswith('eflux'):
                continue
            # deal with even (odd + even)
            elif 'even' in col:
                svec = [savgol_fluxes[col.replace('even', 'odd')],
                        savgol_fluxes[col]]
                inst.check_quality_nan(refwave, wavegrid, svec,
                                       col.replace('even', 'odd_even'))
            # deal with full orders
            else:
                inst.check_quality_nan(refwave, wavegrid, [savgol_fluxes[col]],
                                       col)
    else:
        template_type = 'LBL_NON_SAVGOL'

    # -------------------------------------------------------------------------
    # Step 9. check quality of data
    # -------------------------------------------------------------------------
    # we check that at least 50% of the data is valid (not NaN) in the template
    frac_valid = np.ones(refwave.shape[0])
    frac_valid_odd_even = np.ones(refwave.shape[0])
    frac_valid_savgol = np.ones(refwave.shape[0])

    # need to get back "orders"
    for ordernum in range(refwave.shape[0]):
        wavemin = np.nanmin(refwave[ordernum, :])
        wavemax = np.nanmax(refwave[ordernum, :])

        wavemask = (wavegrid > wavemin) & (wavegrid < wavemax)

        n_valid = np.sum(np.isfinite(p50[wavemask]))
        n_total = np.sum(wavemask)
        # an order outside the template's grid: no samples, no fraction
        if not n_total:
            frac_valid[ordernum] = np.nan
            frac_valid_odd_even[ordernum] = np.nan
            continue
        frac_valid[ordernum] = n_valid / n_total

        # For odd/even we add the number of odd + even
        # since odd orders have even orders set to nan and vice versa.
        n_valid_even = np.sum(np.isfinite(p50_even[wavemask]))
        n_valid_odd = np.sum(np.isfinite(p50_odd[wavemask]))
        frac_valid_odd_even[ordernum] = (n_valid_even + n_valid_odd) / n_total



    # -------------------------------------------------------------------------
    # Step 9. Write template
    # -------------------------------------------------------------------------
    # get props
    props = dict(wavelength=wavegrid, flux=p50, eflux=eflux, rms=rms,
                 flux_odd=p50_odd, eflux_odd=eflux_odd, flux_even=p50_even,
                 eflux_even=eflux_even, rms_odd=rms_odd,
                 rms_even=rms_even, template_coverage=template_coverage,
                 total_nobs_berv=total_nobs_berv, template_nobs=nfiles,
                 savgol_fluxes=savgol_fluxes, template_type=template_type,
                 planets=planets_used, small_rv_shift=flat_rv)
    # write table
    inst.write_template(template_file, props, refhdr, sci_table)
    # and what it was made from (provenance)
    provenance.record(template_file, inputs_token, len(science_files))

    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    # do not remove this line
    logmsg = log.get_cache()
    # return
    return locals()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

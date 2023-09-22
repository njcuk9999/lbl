#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
import os
import warnings

import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import apero
from lbl.science import general

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
# add arguments (must be in parameters.py)
ARGS_TEMPLATE = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'TEMPLATE_SUBDIR', 'SCIENCE_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE'
    # other
                      'VERBOSE', 'PROGRAM',
]

DESCRIPTION_TEMPLATE = 'Use this code to create the LBL template'


# =============================================================================
# Define functions
# =============================================================================
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
        raise LblException(e.message, verbose=False)
    except Exception as e:
        emsg = 'Unexpected {0} error: {1}: {2}'
        eargs = [__NAME__, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
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
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # must force object science to object template
    inst.params['OBJECT_SCIENCE'] = str(inst.params['OBJECT_TEMPLATE'])
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
    template_file = inst.template_file(template_dir, required=False)
    # science filenames
    science_files = inst.science_files(science_dir)
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    # load blaze file if set
    if blaze_file is not None:
        blaze = inst.load_blaze(blaze_file, science_file=science_files[0],
                                normalize=False)
    else:
        blaze = None
    # -------------------------------------------------------------------------
    # Step 3: Check if mask exists
    # -------------------------------------------------------------------------
    if os.path.exists(template_file) and not inst.params['OVERWRITE']:
        # log that mask exist
        msg = 'Template {0} exists. Skipping template creation. '
        log.warning(msg.format(template_file))
        log.warning('Set --overwrite to recalculate mask')
        # return here
        return locals()
    elif os.path.exists(template_file) and inst.params['OVERWRITE']:
        log.general(f'--overwrite=True. Recalculating template {template_file}')
    else:
        log.general(f'Could not find {template_file}. Calculating template.')
    # -------------------------------------------------------------------------
    # Step 4: Deal with reference file (first file)
    # -------------------------------------------------------------------------
    # may need to filter out calibrations
    science_files = inst.filter_files(science_files)
    # select the first science file as a reference file
    refimage, refhdr = inst.load_science_file(science_files[0])
    # get wave solution for reference file
    refwave = inst.get_wave_solution(science_files[0], refimage, refhdr)
    # get domain coverage
    wavemin = inst.params['COMPIL_WAVE_MIN']
    wavemax = inst.params['COMPIL_WAVE_MAX']
    # work out a valid velocity step in m/s
    grid_step_magic = general.get_velocity_step(refwave)
    # grid scale for the template
    wavegrid = general.get_magic_grid(wave0=wavemin, wave1=wavemax,
                                      dv_grid=grid_step_magic)
    # -------------------------------------------------------------------------
    # Step 5: Loop around each file and load into cube
    # -------------------------------------------------------------------------
    # create a cube that contains one line for each file
    flux_cube = np.zeros([len(wavegrid), len(science_files)])
    # weight cube to account for order overlap
    weight_cube = np.zeros([len(wavegrid), len(science_files)])
    # science table
    sci_table = dict()
    # all bervs
    berv = np.zeros_like(science_files, dtype=float)
    # loop around files
    for it, filename in enumerate(science_files):
        # print progress
        msg = 'Processing E2DS->S1D for file {0} of {1}'
        margs = [it + 1, len(science_files)]
        log.general(msg.format(*margs))
        # select the first science file as a reference file
        sci_image, sci_hdr = inst.load_science_file(filename)
        # get wave solution for reference file
        sci_wave = inst.get_wave_solution(filename, sci_image, sci_hdr)
        # load blaze (just ones if not needed)
        if blaze is None:
            bargs = [filename, sci_image, sci_hdr, calib_dir]
            bout = inst.load_blaze_from_science(*bargs, normalize=False)
            blazeimage, blaze_flag = bout
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
        # apply berv if required
        if berv[it] != 0.0:
            sci_wave = mp.doppler_shift(sci_wave, -berv[it])
        # set exactly zeros to NaNs
        sci_image[sci_image == 0] = np.nan
        # compute s1d from e2ds
        s1d_flux, s1d_weight = apero.e2ds_to_s1d(inst.params, sci_wave,
                                                 sci_image, blazeimage,
                                                 wavegrid)
        # push into arrays
        flux_cube[:, it] = s1d_flux
        weight_cube[:, it] = s1d_weight
    # -------------------------------------------------------------------------
    # Step 6. Creation of the template
    # -------------------------------------------------------------------------
    # points are not valid where weight is zero or flux_cube is exactly zero
    bad_domain = (weight_cube == 0) | (flux_cube == 0)
    # set the bad fluxes to NaN
    flux_cube[bad_domain] = np.nan
    # set the weighting of bad pixels to 1
    weight_cube[bad_domain] = 1
    # -------------------------------------------------------------------------
    # print progress
    log.general('Calculating template')
    # divide by the weights (to correct for overlapping orders)
    flux_cube = flux_cube / weight_cube
    # normalize each slice of the cube by its median
    for it in tqdm(range(len(science_files))):
        flux_cube[:, it] = flux_cube[:, it] / np.nanmedian(flux_cube[:, it])

    # copy
    flux_cube0 = np.array(flux_cube)
    # get the pixel hp_width [needs to be in m/s]
    grid_step_original = general.get_velocity_step(refwave, rounding=False)

    hp_width = int(np.round(inst.params['HP_WIDTH'] * 1000 / grid_step_original))
    # -------------------------------------------------------------------------
    # applying low pass filter
    log.general('\tApplying low pass filter to cube')
    # deal with science
    if inst.params['DATA_TYPE'] == 'SCIENCE':
        with warnings.catch_warnings(record=True) as _:
            # calculate the median of the big cube
            median = mp.nanmedian(flux_cube, axis=1)
            # iterate until low frequency gone
            for sci_it in tqdm(range(flux_cube.shape[1])):
                # remove the stellar features
                ratio = flux_cube[:, sci_it] / median
                # apply median filtered ratio (low frequency removal)
                lowpass = mp.lowpassfilter(ratio, hp_width)
                flux_cube[:, sci_it] /= lowpass
    else:
        with warnings.catch_warnings(record=True) as _:
            # calculate the median of the big cube
            median = mp.nanmedian(flux_cube, axis=1)
            # mask to keep only FP peaks and avoid dividing
            # two small values (minima between lines in median and
            # individual spectrum) when computing the lowpass
            peaks = median > mp.lowpassfilter(median, hp_width)
            # iterate until low frequency gone
            for sci_it in tqdm(range(flux_cube.shape[1])):
                # remove the stellar features
                ratio = flux_cube[:, sci_it] / median
                ratio[~peaks] = np.nan

                # apply median filtered ratio (low frequency removal)
                lowpass = mp.lowpassfilter(ratio, hp_width)
                flux_cube[:, sci_it] /= lowpass

    # -------------------------------------------------------------------------
    # bin cube by BERV (to give equal weighting to epochs)
    # -------------------------------------------------------------------------
    # get minimum number of berv bins
    nmin_bervbin = inst.params['BERVBIN_MIN_ENTRIES']
    # get the size of the berv bins
    bervbin_size = inst.params['BERVBIN_SIZE']
    # only for science data
    if inst.params['DATA_TYPE'] == 'SCIENCE':
        # get the berv bin centers
        bervbins = berv // bervbin_size
        # find unique berv bins
        ubervbins = np.unique(bervbins)
        # storage the number of observations per berv bin
        nobs_bervbin = np.zeros_like(ubervbins, dtype=int)
        # get a flux cube for the binned by berv data
        fcube_shape = [flux_cube.shape[0], len(ubervbins)]
        flux_cube_bervbin = np.full(fcube_shape, np.nan)
        # loop around unique berv bings and merge entries via median
        for it, bervbin in enumerate(ubervbins):
            # get mask for those observations in berv bin
            good = bervbins == bervbin
            # count the number of observation in this berv bin
            n_obs = np.sum(good)
            # log progress message
            msg = 'Computing BERV bin {0} of {1}, n files = {2}'
            margs = [it + 1, len(ubervbins), n_obs]
            log.general(msg.format(*margs))
            # deal with minimum number of observations allowed
            if n_obs < nmin_bervbin:
                continue
            # add to the number of observations used
            nobs_bervbin[it] = n_obs
            # combine all observations in bin with median
            with warnings.catch_warnings(record=True) as _:
                bervmed = np.nanmedian(flux_cube[:, good], axis=1)
                flux_cube_bervbin[:, it] = bervmed
        # calculate the number of observations used and berv bins used
        nfiles = np.sum(nobs_bervbin)
        total_nobs_berv = np.sum(nobs_bervbin != 0)
        # calculate the number of observations and the berv coverage
        template_coverage = total_nobs_berv * grid_step_original / 1000
    # else deal with non-science cases
    else:
        flux_cube_bervbin = flux_cube
        # calculate the number of observations used
        total_nobs_berv = 0
        # calculate the number of observations and the berv coverage
        template_coverage = 0
        # we use all files
        nfiles = len(science_files)
        # Set uberv bins
        ubervbins = []

    # -------------------------------------------------------------------------
    # get the median and +/- 1 sigma values for the cube
    # -------------------------------------------------------------------------
    log.general('Calculate 16th, 50th and 84th percentiles')
    with warnings.catch_warnings(record=True) as _:
        if len(ubervbins) > 3:
            # to get statistics on the ber-bin rms, we need more than 3
            # bervbins
            log.general('computation done per-berv bin')
            p16, p50, p84 = np.nanpercentile(flux_cube_bervbin, [16, 50, 84],
                                             axis=1)
        else:
            # if too few berv bins, we take stats on whole cube rather than
            #    per-bervbin
            log.general('computation done per-file, not per-berv bin')
            p16, p50, p84 = np.nanpercentile(flux_cube, [16, 50, 84],
                                             axis=1)
        # calculate the rms of each wavelength element
        rms = (p84 - p16) / 2

    # -------------------------------------------------------------------------
    # Step 7. Write template
    # -------------------------------------------------------------------------
    # get props
    props = dict(wavelength=wavegrid, flux=p50, eflux=rms, rms=rms,
                 template_coverage=template_coverage,
                 total_nobs_berv=total_nobs_berv, template_nobs=nfiles)
    # write table
    inst.write_template(template_file, props, refhdr, sci_table)

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

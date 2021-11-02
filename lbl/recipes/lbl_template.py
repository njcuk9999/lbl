#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
import warnings

import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select
from lbl.science import general
from lbl.resources import lbl_misc
from lbl.science import apero


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
ARGS_TEMPLATE = [
                # core
                'INSTRUMENT', 'CONFIG_FILE',
                # directory
                'DATA_DIR', 'TEMPLATE_SUBDIR', 'SCIENCE_SUBDIR',
                # science
                'OBJECT_SCIENCE', 'OBJECT_TEMPLATE'
                # other
                'VERBOSE', 'PROGRAM',
                ]
# TODO: Etienne - Fill out
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
    inst = select.load_instrument(args, logger=log)
    # get data directory
    data_dir = io.check_directory(inst.params['DATA_DIR'])
    # move log file (now we have data directory)
    lbl_misc.move_log(data_dir, __NAME__)
    # print splash
    lbl_misc.splash(name=__STRNAME__, instrument=inst.name,
                    cmdargs=inst.params['COMMAND_LINE_ARGS'], logger=log)
    # run __main__
    try:
        namespace = __main__(inst)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected lbl_compute error: {0}: {1}'
        eargs = [type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    # end code
    lbl_misc.end(__NAME__, logger=log)
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
        blaze = inst.load_blaze(blaze_file)
    else:
        blaze = None
    # -------------------------------------------------------------------------
    # Step 3: Deal with reference file (first file)
    # -------------------------------------------------------------------------
    # select the first science file as a reference file
    refimage, refhdr =  inst.load_science(science_files[0])
    # get wave solution for reference file
    refwave = inst.get_wave_solution(science_files[0], refimage, refhdr)
    # get domain coverage
    wavemin, wavemax = float(np.nanmin(refwave)), float(np.nanmax(refwave))
    # work out a valid velocity step in m/s
    grid_step = general.get_velocity_step(refwave)
    # grid scale for the template
    wavegrid = general.get_magic_grid(wave0=wavemin, wave1=wavemax,
                                      dv_grid=grid_step)
    # -------------------------------------------------------------------------
    # Step 4: Loop around each file and load into cube
    # -------------------------------------------------------------------------
    # create a cube that contains one line for each file
    flux_cube = np.zeros([len(wavegrid), len(science_files)])
    # weight cube to account for order overlap
    weight_cube = np.zeros([len(wavegrid), len(science_files)])
    # science table
    sci_table = dict()
    # loop around files
    for it, filename in enumerate(science_files):
        # print progress
        msg = 'Processing E2DS->S1D for file {0} of {1}'
        margs = [it + 1, len(science_files)]
        log.general(msg.format(*margs))
        # select the first science file as a reference file
        sci_image, sci_hdr = inst.load_science(filename)
        # get wave solution for reference file
        sci_wave = inst.get_wave_solution(filename, sci_image, sci_hdr)
        # load blaze (just ones if not needed)
        if blaze is None:
            blaze = inst.load_blaze_from_science(sci_image, sci_hdr, calib_dir)
        # get the berv
        berv = inst.get_berv(sci_hdr)
        # populate science table
        sci_table = inst.populate_sci_table(filename, sci_table, sci_hdr,
                                            berv=berv)
        # apply berv if required
        if berv != 0.0:
            sci_wave = mp.doppler_shift(sci_wave, berv)
        # compute s1d from e2ds
        s1d_flux, s1d_weight = apero.e2ds_to_s1d(inst.params, sci_wave,
                                                 sci_image, blaze, wavegrid)
        # push into arrays
        flux_cube[:, it] = s1d_flux
        weight_cube[:, it] = s1d_weight
    # -------------------------------------------------------------------------
    # Creation of the template
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

    # get the median and +/- 1 sigma values for the cube
    log.general('Calculate 16th, 50th and 84th percentiles')
    with warnings.catch_warnings(record=True) as _:
        p16, p50, p84 = np.nanpercentile(flux_cube, [16, 50, 84], axis=1)
        # calculate the rms of each wavelength element
        rms = (p84 - p16) / 2

    # -------------------------------------------------------------------------
    # Write template
    # -------------------------------------------------------------------------
    # get props
    props = dict(wavelength=wavegrid, flux=p50, eflux=rms, rms=rms)
    # write table
    inst.write_template(template_file, props, refhdr, sci_table)

    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import plot
from lbl.science import general

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_blaze.py'
__STRNAME__ = 'LBL Blaze'
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
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'BLAZE_FILE', 'BLAZE_CORRECTED',
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
    # get the pixel hp_width [needs to be in m/s]
    hp_width = inst.params['HP_WIDTH'] * 1000
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst)
    template_dir, science_dir = dparams['TEMPLATE_DIR'], dparams['SCIENCE_DIR']
    calib_dir = dparams['CALIB_DIR']

    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames (after checking if template exists)
    # -------------------------------------------------------------------------
    # science filenames
    science_files = inst.science_files(science_dir)
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    # load blaze file if set
    if blaze_file is not None:
        blaze = inst.load_blaze(blaze_file, science_file=str(science_files[0]),
                                normalize=False)
    else:
        blaze = None

    # -------------------------------------------------------------------------
    # Step 3: Deal with reference file (first file)
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
    # load blaze (just ones if not needed)
    if blaze is None:
        bargs = [science_files[0], refimage, refhdr, calib_dir]
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
        sci_image, blazeimage = inst.no_blaze_corr(refimage, refwave)

    # get the blaze parameters (may be instrument specific)
    nth_deg, bdomain = inst.norm_blaze_params()
    # do the blaze correction
    sout = mp.smart_blaze_norm(refwave, blazeimage, nth_deg, bdomain, full=True)
    # --------------------------------------------------------------------------
    # force plots on
    inst.params['PLOT'] = True
    # plot smart blaze
    plot.smart_nblaze_plot(inst, *sout, nth_deg=nth_deg)

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

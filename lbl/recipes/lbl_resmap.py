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
__NAME__ = 'lbl_resmap.py'
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
        blaze = inst.load_blaze(blaze_file, science_file=str(science_files[0]),
                                normalize=True)
    else:
        blaze = None
    # -------------------------------------------------------------------------
    # Etienne's code here
    # -------------------------------------------------------------------------

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

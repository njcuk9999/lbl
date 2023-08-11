#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_mask

Build mask for the template for use in LBL compute/compile

Created on 2021-08-24

@author: cook
"""
import os

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.resources import lbl_misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_reset.py'
__STRNAME__ = 'LBL Reset'
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
ARGS_MASK = [  # core
    'INSTRUMENT', 'CONFIG_FILE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
    'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
]

DESCRIPTION_MASK = 'Use this code to clean out any LBL data from DATA_DIR'


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
    args = select.parse_args(ARGS_MASK, kwargs, DESCRIPTION_MASK)
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
        args = select.parse_args(ARGS_MASK, kwargs, DESCRIPTION_MASK)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # -------------------------------------------------------------------------
    # Step 1: Set up data directories
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst, skip_obj=True)
    # -------------------------------------------------------------------------
    # Step 2: clean directories
    # -------------------------------------------------------------------------
    # clean _tc from science directories (tellu cleaned files)
    io.clean_directory(dparams['SCIENCE_DIR'], dir_suffix='_tc')
    # clean reftable directory
    io.clean_directory(dparams['LBLRT_DIR'])
    # clean rdb directory
    io.clean_directory(dparams['LBL_RDB_DIR'])
    # clean rv directory
    io.clean_directory(dparams['LBLRV_ALL'])
    # clean mask directory
    io.clean_directory(dparams['MASK_DIR'])
    # clean template directory
    io.clean_directory(dparams['TEMPLATE_DIR'])
    # clean plot directory
    io.clean_directory(dparams['PLOT_DIR'])
    # clean calib directory
    io.clean_directory(dparams['CALIB_DIR'],
                       include_files=inst.params['SAMPLE_WAVE_GRID_FILE'])
    # clean log directory
    logpath = os.path.dirname(base_classes.log.filepath)
    io.clean_directory(logpath)
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

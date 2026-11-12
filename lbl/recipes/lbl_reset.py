#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_reset

Remove the products of LBL from DATA_DIR (the spectra it derived, tables,
velocities, masks, templates, plots, the sample wave grid)

Created on 2021-08-24

@author: cook
"""
import os
import traceback

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
log = io.log
# add arguments (must be in parameters.py)
ARGS_MASK = [  # core
    'INSTRUMENT', 'CONFIG_FILE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
    'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
]

DESCRIPTION_MASK = 'Use this code to clean out any LBL data from DATA_DIR'
# the science folders LBL derives from an object: <object>_tc (telluric
#   cleaning), <object>_pca2d and <object>_tc_pca2d (PCA2D, with an optional
#   _<name> after it), <object>_residuals (PCA2D's save_residuals, the same)
DERIVED_SCIENCE = (r'(_tc|_pca2d(_[0-9A-Za-z._+-]+)?'
                   r'|_residuals(_[0-9A-Za-z._+-]+)?)$')


# =============================================================================
# Define functions
# =============================================================================
def reset_plan(data_dir: str, params) -> list:
    """
    What the reset empties in DATA_DIR, in the order it does it:
    [(what, folder, keyword arguments of io.clean_directory)]. The recipe
    and the window (its Cleanup tab says what a reset deletes before it is
    done, io.files_to_clean) both take it from here

    :param data_dir: str, LBL's DATA_DIR
    :param params: the instrument's parameters (its subfolders, its sample
                   wave grid file)

    :return: list of (str, str, dict)
    """
    def join(key):
        return os.path.join(data_dir, params[key])

    return [
        # the science folders LBL derived (telluric cleaning, PCA2D), never
        #   the spectra
        ('derived', join('SCIENCE_SUBDIR'), dict(dir_regex=DERIVED_SCIENCE)),
        ('reftable', join('LBLREFTAB_SUBDIR'), dict()),
        ('rdb', join('LBLRDB_SUBDIR'), dict()),
        ('lblrv', join('LBLRV_SUBDIR'), dict()),
        ('masks', join('MASK_SUBDIR'), dict()),
        ('templates', join('TEMPLATE_SUBDIR'), dict()),
        ('plots', os.path.join(data_dir, 'plots'), dict()),
        # of calib/, the sample wave grid alone (the blazes stay)
        ('wavegrid', join('CALIB_SUBDIR'),
         dict(include_files=[params['SAMPLE_WAVE_GRID_FILE']])),
    ]


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
    # the derived science folders, reftable, rdb, lblrv, masks, templates,
    #   plots, the sample wave grid (reset_plan)
    for _what, folder, kwargs in reset_plan(dparams['DATA_DIR'],
                                            inst.params):
        io.clean_directory(folder, **kwargs)
    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    return locals()


def run():
    """
    Main function for running the setup from the command line
    (avoids returning locals)

    :return:
    """
    # run the main function
    _ = main()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

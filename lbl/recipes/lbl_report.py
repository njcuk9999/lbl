#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_report

Write the report of what is in the data of this object: a pdf (built with
LaTeX) and a tar archive of its figures, one file each, ready for a paper.
This runs at the very end of the processing, once the rdb files are there.

Created on 2026-09-24

@author: artigau
"""
import os

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import report

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_report.py'
__STRNAME__ = 'LBL Report'
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
ARGS_REPORT = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'SCIENCE_SUBDIR',
    'LBLRV_SUBDIR', 'LBLRDB_SUBDIR', 'REPORT_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_COMPARISON',
    # the report itself
    'REPORT_PERIOD_MIN', 'REPORT_PERIOD_MAX', 'REPORT_FIP_PRIOR',
    'REPORT_MAX_RIVER_FILES', 'REPORT_RIVER_WIDTH', 'REPORT_EXOPLANET_EU',
    'REPORT_MAX_LINE_FILES', 'REPORT_CACHE_DAYS',
    # other
    'VERBOSE', 'PROGRAM',
]

DESCRIPTION_REPORT = 'Use this code to write the report of an LBL run'


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
    args = select.parse_args(ARGS_REPORT, kwargs, DESCRIPTION_REPORT)
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
    """
    The main recipe function - all code dealing with recipe functionality
    should go here

    :param inst: Instrument instance
    :param kwargs: kwargs to parse to instrument (only use if inst is None)

    :return: all variables in local namespace
    """
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or inst.params is None:
        # deal with parsing arguments
        args = select.parse_args(ARGS_REPORT, kwargs, DESCRIPTION_REPORT)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst)
    # -------------------------------------------------------------------------
    # Step 2: Write the report (the pdf and the tar archive of the figures)
    # -------------------------------------------------------------------------
    outdir = report.make_report(inst, dparams)
    # print where it is
    log.info('Report written to {0}'.format(outdir))
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

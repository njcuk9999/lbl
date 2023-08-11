#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.resources import lbl_misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_compute.py'
__STRNAME__ = 'LBL Compute'
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
ARGS_COMPUTE = [  # core
    'INSTRUMENT', 'CONFIG_FILE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
    'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'INPUT_FILE', 'TEMPLATE_FILE',
    'BLAZE_FILE', 'HP_WIDTH', 'USE_NOISE_MODEL',
    # plotting
    'PLOT', 'PLOT_COMPUTE_CCF', 'PLOT_COMPUTE_LINES',
    # other
    'SKIP_DONE', 'VERBOSE', 'PROGRAM',
]
# TODO: Etienne - Fill out
DESCRIPTION_COMPUTE = 'Use this code to compute the LBL rv'


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
    args = select.parse_args(ARGS_COMPUTE, kwargs, DESCRIPTION_COMPUTE)
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
    return locals()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

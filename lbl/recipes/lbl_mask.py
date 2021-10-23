#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_mask

Build mask for the template for use in LBL compute/compile

Created on 2021-08-24

@author: cook
"""
import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.science import general
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
ARGS_MASK = [# core
             'INSTRUMENT', 'CONFIG_FILE',
             # directory
             'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR',
             # science
             'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'OBJECT_TEFF',
             'OBJECT_LOGG', 'OBJECT_FEH', 'OBJECT_Z', 'OBJECT_ALPHA',
             # other
             'VERBOSE', 'PROGRAM',
            ]
# TODO: Etienne - Fill out
DESCRIPTION_MASK = 'Use this code to calculate the LBL mask'


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
        args = select.parse_args(ARGS_MASK, kwargs, DESCRIPTION_MASK)
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
    mask_dir, template_dir = dparams['MASK_DIR'], dparams['TEMPLATE_DIR']
    models_dir = dparams['MODELS_DIR']
    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # mask filename
    mask_file = inst.mask_file(mask_dir, required=False)
    # template filename
    template_file = inst.template_file(template_dir)
    # get template file
    template_image, template_hdr = inst.load_template(template_file,
                                                      get_hdr=True)
    # see if the template is a calibration template
    flag_calib = inst.flag_calib(template_hdr)
    # -------------------------------------------------------------------------
    # Step 3: Get the Goettingen Phoenix models
    # -------------------------------------------------------------------------
    general.get_stellar_models(inst, )

    # -------------------------------------------------------------------------
    # Step 4: Find correct model for this template
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Step 5: Define a line mask for the model
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Step 6: Work out systemic velocity for the template
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Step 7: Write masks to file
    # -------------------------------------------------------------------------

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

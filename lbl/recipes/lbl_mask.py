#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_mask

Build mask for the template for use in LBL compute/compile

Created on 2021-08-24

@author: cook
"""
import numpy as np
import os

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select
from lbl.science import general
from lbl.resources import lbl_misc


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_mask.py'
__STRNAME__ = 'LBL Mask'
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
ARGS_MASK = [
             # core
             'INSTRUMENT', 'CONFIG_FILE',
             # directory
             'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR',
             # science
             'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'OBJECT_TEFF',
             'OBJECT_LOGG', 'OBJECT_Z', 'OBJECT_ALPHA',
             # plotting
             'PLOT', 'PLOT_MASK_CCF',
             # other
             'OVERWRITE', 'VERBOSE', 'PROGRAM',
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
    inst = select.load_instrument(args, plogger=log)
    # get data directory
    data_dir = io.check_directory(inst.params['DATA_DIR'])
    # move log file (now we have data directory)
    lbl_misc.move_log(data_dir, __NAME__)
    # print splash
    lbl_misc.splash(name=__STRNAME__, instrument=inst.name,
                    cmdargs=inst.params['COMMAND_LINE_ARGS'], plogger=log)
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
    # must force object science to object template
    inst.params['OBJECT_SCIENCE'] = str(inst.params['OBJECT_TEMPLATE'])
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
    template_table, template_hdr = inst.load_template(template_file,
                                                      get_hdr=True)
    # see if the template is a calibration template
    flag_calib = inst.flag_calib(template_hdr)

    # -------------------------------------------------------------------------
    # Step 3: Check if mask exists
    # -------------------------------------------------------------------------
    if os.path.exists(mask_file) and not inst.params['OVERWRITE']:
        # log that mask exist
        msg = 'Mask {0} exists. Skipping mask creation. '
        log.warning(msg.format(mask_file))
        log.warning('Set --overwrite to recalculate mask')
        # return here
        return locals()

    # -------------------------------------------------------------------------
    # Step 4: Find correct Goettingen Phoenix models and get them if not
    #         present - only done for non calibration files
    # -------------------------------------------------------------------------
    if not flag_calib:
        m_wavemap, m_spectrum = general.get_stellar_models(inst, models_dir)
    else:
        m_wavemap, m_spectrum = None, None

    # -------------------------------------------------------------------------
    # Step 5: Find the lines (regions of sign change in the derivative)
    # -------------------------------------------------------------------------
    line_table = general.find_mask_lines(inst, template_table)

    # -------------------------------------------------------------------------
    # Step 6: Work out systemic velocity for the template
    # -------------------------------------------------------------------------
    if not flag_calib:
        # work out the systemic velocity in km/s
        sys_vel = general.mask_systemic_velocity(inst, line_table, m_wavemap,
                                                 m_spectrum)
        # update line_table for the systemic velocity
        line_table['ll_mask_s'] = mp.doppler_shift(line_table['ll_mask_s'],
                                                   1000 * sys_vel)
        line_table['ll_mask_e'] = mp.doppler_shift(line_table['ll_mask_e'],
                                                   1000 * sys_vel)
    else:
        sys_vel = 0.0

    # -------------------------------------------------------------------------
    # Step 7: Write masks to file
    # -------------------------------------------------------------------------
    # get the positive and negative masks
    neg_mask = line_table['w_mask'] > 0
    pos_mask = line_table['w_mask'] < 0
    # now normalize the weights
    norm = np.nanmean(np.abs(line_table['w_mask']))
    line_table['w_mask'] = line_table['w_mask'] / norm
    # write mask to file
    inst.write_mask(mask_file, line_table, pos_mask, neg_mask, sys_vel,
                    template_hdr)

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

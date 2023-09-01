#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
import sys

from lbl.core import base
from lbl.core import base_classes
from lbl.recipes import lbl_compile
from lbl.recipes import lbl_compute
from lbl.recipes import lbl_mask
from lbl.recipes import lbl_telluclean
from lbl.recipes import lbl_template
from lbl.resources import lbl_misc

__NAME__ = 'lbl_mask.py'
__STRNAME__ = 'LBL Mask'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__

DESCRIPTION_MASK = 'Use this code to wrap around lbl'

# define keys to remove from run params
REMOVE_KEYS = [  # core
    'INSTRUMENT', 'DATA_DIR', 'DATA_TYPES', 'DATA_SOURCE',
    # science keys
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'OBJECT_TEFF',
    # run keys
    'RUN_LBL_TELLUCLEAN', 'RUN_LBL_TEMPLATE', 'RUN_LBL_MASK',
    'RUN_LBL_COMPUTE', 'RUN_LBL_COMPILE',
    # skip keys
    'SKIP_LBL_TELLUCLEAN', 'SKIP_LBL_TEMPLATE', 'SKIP_LBL_MASK',
    'SKIP_LBL_COMPUTE', 'SKIP_LBL_COMPILE',
    # general keys already used
    'SKIP_DONE', 'OVERWRITE', 'TELLUCLEAN_USE_TEMPLATE']

# Define the default values
DEFAULTS = dict()
DEFAULTS['RUN_LBL_TELLUCLEAN'] = False
DEFAULTS['RUN_LBL_TEMPLATE'] = False
DEFAULTS['RUN_LBL_MASK'] = False
DEFAULTS['RUN_LBL_COMPUTE'] = False
DEFAULTS['RUN_LBL_COMPILE'] = False
DEFAULTS['SKIP_LBL_TEMPLATE'] = False
DEFAULTS['SKIP_LBL_MASK'] = False
DEFAULTS['SKIP_LBL_COMPUTE'] = False
DEFAULTS['SKIP_LBL_COMPILE'] = False


# =============================================================================
# Define functions
# =============================================================================
def main(runparams: dict):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param runparams: dict, parameters to pass to lbl recipes

    :return:
    """
    # reset the sys.argv (arguments from command line aren't used)
    sys.argv = [__NAME__]
    # get key parameters
    instrument = lbl_misc.check_runparams(runparams, 'INSTRUMENT')
    data_dir = lbl_misc.check_runparams(runparams, 'DATA_DIR')
    data_source = lbl_misc.check_runparams(runparams, 'DATA_SOURCE')
    data_types = lbl_misc.check_runparams(runparams, 'DATA_TYPES')
    object_sciences = lbl_misc.check_runparams(runparams, 'OBJECT_SCIENCE')
    object_templates = lbl_misc.check_runparams(runparams, 'OBJECT_TEMPLATE')
    object_teffs = lbl_misc.check_runparams(runparams, 'OBJECT_TEFF')
    # -------------------------------------------------------------------------
    # push other keyword arguments into keyword arguments dictionary
    keyword_args = dict()
    for key in runparams:
        if key not in REMOVE_KEYS:
            keyword_args[key] = runparams[key]
    # make sure we have defaults if key not in runparams
    for key in DEFAULTS:
        if key not in runparams:
            runparams[key] = DEFAULTS[key]
    # -------------------------------------------------------------------------
    # sanity checks on runparams (certain things should not be set together)
    if runparams['RUN_LBL_MASK'] and 'MASK_FILE' in runparams:
        if runparams['MASK_FILE'] not in [None, 'None', '', 'Null']:
            emsg = ('LBL_WRAP ERROR: Cannot have RUN_LBL_MASK=True and '
                    'MASK_FILE={0} (Must be unset)')
            raise base_classes.LblException(emsg.format(runparams['MASK_FILE']))
    # -------------------------------------------------------------------------
    # loop around all files
    for num in range(len(object_sciences)):
        # get this iterations values
        data_type = data_types[num]
        object_science = object_sciences[num]
        object_template = object_templates[num]
        object_teff = object_teffs[num]
        # ---------------------------------------------------------------------
        # run all pre-cleaning steps
        if runparams['RUN_LBL_TELLUCLEAN'] and data_type == 'SCIENCE':
            # run telluric cleaning (without template)
            lbl_telluclean.main(instrument=instrument, data_dir=data_dir,
                                data_source=data_source,
                                data_type=data_type,
                                object_science=object_science,
                                object_template=object_template,
                                skip_done=False,
                                telluclean_use_template=False,
                                **keyword_args)
            # update template name
            if not object_template.endswith('_tc'):
                object_template = object_template + '_tc'
            # make the template (if not present)
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science + '_tc',
                              object_template=object_template,
                              overwrite=True,
                              **keyword_args)
            # re-run tellu clean with uncorrected science data now using our
            #  template (made from cleaned science data)
            lbl_telluclean.main(instrument=instrument, data_dir=data_dir,
                                data_source=data_source,
                                data_type=data_type,
                                object_science=object_science,
                                object_template=object_template,
                                skip_done=False,
                                telluclean_use_template=True,
                                **keyword_args)
            # update object name
            if not object_science.endswith('_tc'):
                object_science = object_science + '_tc'

        # ---------------------------------------------------------------------
        # make the template (if not present)
        if runparams['RUN_LBL_TEMPLATE']:
            # Must produce the template for the science data and the template
            #   we use a set to do this (only runs once if they are the same)
            for _obj_template in {object_science, object_template}:
                lbl_template.main(instrument=instrument, data_dir=data_dir,
                                  data_source=data_source,
                                  data_type=data_type,
                                  object_science=object_science,
                                  object_template=_obj_template,
                                  overwrite=not runparams['SKIP_LBL_TEMPLATE'],
                                  **keyword_args)
        # ---------------------------------------------------------------------
        # make the mask (if not present)
        if runparams['RUN_LBL_MASK']:
            # Must produce the mask for the science data and the template
            #   we use a set to do this (only runs once if they are the same)
            for _obj_template in {object_science, object_template}:
                lbl_mask.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science,
                              object_template=_obj_template,
                              object_teff=object_teff,
                              overwrite=not runparams['SKIP_LBL_MASK'],
                              **keyword_args)
        # ---------------------------------------------------------------------
        # # make the noise model (if not present)
        # if runparams['RUN_LBL_NOISE']:
        #     lbl_noise(instrument=instrument, data_dir=data_dir,
        #               object_science=object_science,
        #               object_template=object_template,
        #               **keyword_args)
        # ---------------------------------------------------------------------
        # run the compute code
        if runparams['RUN_LBL_COMPUTE']:
            lbl_compute.main(instrument=instrument, data_dir=data_dir,
                             data_source=data_source,
                             data_type=data_type,
                             object_science=object_science,
                             object_template=object_template,
                             skip_done=runparams['SKIP_LBL_COMPUTE'],
                             **keyword_args)
        # ---------------------------------------------------------------------
        # run the compile code
        if runparams['RUN_LBL_COMPILE']:
            lbl_compile.main(instrument=instrument, data_dir=data_dir,
                             data_source=data_source,
                             data_type=data_type,
                             object_science=object_science,
                             object_template=object_template,
                             skip_done=runparams['SKIP_LBL_COMPILE'],
                             **keyword_args)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set up parameters
    rparams = dict()
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_DIR'] = '/data/spirou/data/lbl/'
    # science criteria
    rparams['DATA_TYPES'] = ['FP', 'SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['FP', 'GL699']
    rparams['OBJECT_TEMPLATE'] = ['FP', 'GL699']
    rparams['OBJECT_TEFF'] = [300, 3224]
    # what to run
    rparams['RUN_LBL_TELLUCLEAN'] = True
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    # whether to skip done files
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # run main
    main(rparams)

# =============================================================================
# End of code
# =============================================================================

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
from lbl.core import io
from lbl.recipes import lbl_compile
from lbl.recipes import lbl_compute
from lbl.recipes import lbl_mask
from lbl.recipes import lbl_telluclean
from lbl.recipes import lbl_template
from lbl.recipes import lbl_reset
from lbl.resources import lbl_misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_wrap.py'
__STRNAME__ = 'LBL Wrapper'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Description of recipe
DESCRIPTION_MASK = 'Use this code to wrap around lbl'
# get the logger
log = io.log
# define keys to remove from run params
REMOVE_KEYS = [  # core
    'INSTRUMENT', 'DATA_DIR', 'DATA_TYPES', 'DATA_SOURCE',
    # science keys
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'OBJECT_TEFF',
    'BLAZE_CORRECTED', 'BLAZE_FILE',
    # run keys
    'RUN_LBL_RESET',
    'RUN_LBL_TELLUCLEAN', 'RUN_LBL_TEMPLATE', 'RUN_LBL_MASK',
    'RUN_LBL_COMPUTE', 'RUN_LBL_COMPILE',
    # skip keys
    'SKIP_LBL_TELLUCLEAN', 'SKIP_LBL_TEMPLATE', 'SKIP_LBL_MASK',
    'SKIP_LBL_COMPUTE', 'SKIP_LBL_COMPILE',
    # general keys already used
    'SKIP_DONE', 'OVERWRITE', 'TELLUCLEAN_USE_TEMPLATE']

# Define the default values
DEFAULTS = dict()
DEFAULTS['RUN_LBL_RESET'] = False
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
    blaze_corrs = lbl_misc.check_runparams(runparams, 'BLAZE_CORRECTED',
                                           required=False)
    blaze_files = lbl_misc.check_runparams(runparams, 'BLAZE_FILE',
                                           required=False)
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
    # mark the expected length if a list
    olen = len(object_sciences)
    # loop around all files
    for num in range(olen):
        # get the science target
        object_science = object_sciences[num]
        # print wrapper splash
        lbl_misc.splash(name=__STRNAME__, instrument=instrument,
                        plogger=log)
        # print iteration we are running
        msg = 'Running [{0}] iteration {1}/{2}'
        margs = [object_science, num + 1, olen]
        log.info(msg.format(*margs))
        # wrap check args
        wkargs = dict(iteration=num, length=olen)
        # get this iterations values (and check if they are a list of matching
        #    length to object_sciences) or just a single value
        data_type = lbl_misc.wraplistcheck(data_types,
                                           'DATA_TYPES', **wkargs)
        object_template = lbl_misc.wraplistcheck(object_templates,
                                                 'OBJECT_TEMPLATE', **wkargs)
        object_teff = lbl_misc.wraplistcheck(object_teffs,
                                             'OBJECT_TEFF', **wkargs)
        blaze_corr = lbl_misc.wraplistcheck(blaze_corrs,
                                            'BLAZE_CORRECTED', **wkargs)
        blaze_file = lbl_misc.wraplistcheck(blaze_files,
                                            'BLAZE_FILE', **wkargs)
        # ---------------------------------------------------------------------
        if runparams['RUN_LBL_RESET']:
            lbl_reset.main(instrument=instrument, data_dir=data_dir,
                           data_source=data_source, **keyword_args)
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
                                blaze_corrected=blaze_corr,
                                blaze_file=blaze_file,
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
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
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
                                blaze_corrected=blaze_corr,
                                blaze_file=blaze_file,
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
                                  blaze_corrected=blaze_corr,
                                  blaze_file=blaze_file,
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
                             blaze_corrected=blaze_corr,
                             blaze_file=blaze_file,
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from astropy.table import Table

from lbl.core import base
from lbl import lbl_compute
from lbl import lbl_compil
from lbl import lbl_template
from lbl import lbl_mask
from lbl import lbl_noise
from lbl import lbl_telluclean


__NAME__ = 'lbl_mask.py'
__STRNAME__ = 'LBL Mask'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__

DESCRIPTION_MASK = 'Use this code to wrap around lbl'

# define keys to remove from run params
REMOVE_KEYS = ['INSTRUMENT', 'DATA_DIR', '']

# =============================================================================
# Define functions
# =============================================================================
def main(runparams: dict):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param kwargs: kwargs to parse to instrument - anything in params can be
                   parsed (overwrites instrumental and default parameters)
    :return:
    """
    # get key parameters
    instrument = runparams['INSTRUMENT']
    data_dir = runparams['DATA_DIR']
    data_types = runparams['DATA_TYPES']
    object_sciences = runparams['OBJECT_SCIENCE']
    object_templates = runparams['OBJECT_TEMPLATE']
    object_teffs = runparams['OBJECT_TEFF']


    # loop around all files
    for num in range(len(object_sciences)):
        # get this iterations values
        data_type = data_types[num]
        object_science = object_sciences[num]
        object_template = object_templates[num]
        object_teff = object_teffs[num]

        # run all pre-cleaning steps
        if runparams['RUN_LBL_TELLUCLEAN'] and data_type == 'SCIENCE':
            # run telluric cleaning (without template)
            lbl_telluclean(instrument=instrument, data_dir=data_dir,
                           object_science=object_science,
                           object_template=object_template,
                           skip_done=runparams['SKIP_LBL_TELLUCLEAN'],
                           telluclean_use_template=False)
            # make the template (if not present)
            lbl_template(instrument=instrument, data_dir=data_dir,
                         object_science=object_science,
                         object_template=object_template,
                         overwrite=~runparams['SKIP_LBL_TELLUCLEAN'])
            lbl_telluclean(instrument=instrument, data_dir=data_dir,
                           object_science=object_science,
                           object_template=object_template,
                           skip_done=runparams['SKIP_LBL_TELLUCLEAN'],
                           telluclean_use_template=True)
        # make the template (if not present)
        if runparams['RUN_LBL_TEMPLATE']:
            lbl_template(instrument=instrument, data_dir=data_dir,
                         object_science=object_science,
                         object_template=object_template,
                         overwrite=~runparams['SKIP_LBL_TEMPLATE'])
        # make the mask (if not present)
        if runparams['RUN_LBL_MASK']:
            lbl_mask(instrument=instrument, data_dir=data_dir,
                     object_science=object_science,
                     object_template=object_template,
                     object_teff=object_teff,
                     overwrite=~runparams['SKIP_LBL_MASK'])
        # # make the noise model (if not present)
        # if runparams['RUN_LBL_NOISE']:
        #     lbl_noise(instrument=instrument, data_dir=data_dir,
        #               object_science=object_science,
        #               object_template=object_template)
        # run the compute code
        if runparams['RUN_LBL_COMPUTE']:
            lbl_compute(instrument=instrument, data_dir=data_dir,
                        object_science=object_science,
                        object_template=object_template,
                        skip_done=runparams['SKIP_LBL_COMPUTE'])
        # run the compile code
        if runparams['RUN_LBL_COMPILE']:
            lbl_compil(instrument=instrument, data_dir=data_dir,
                       object_science=object_science,
                       object_template=object_template,
                       skip_done=runparams['SKIP_LBL_COMPILE'])


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
    rparams['SKIP_LBL_TELLUCLEAN'] = False
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # run main
    main(rparams)


# =============================================================================
# End of code
# =============================================================================

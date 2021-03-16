#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Default parameters (non-instrument specific)

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.parameters.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# set up parameter dictionary
params = base_classes.ParamDict()

# =============================================================================
# Define general parameters
# =============================================================================

# add default params
params.set(key='CONFIG_FILE', value=None, source=__NAME__,
           desc='Config file for user settings',
           arg='--config', dtype=str)

# add main data directory (structure assumed below)
params.set(key='DATA_DIR', value=None, source=__NAME__,
           desc='Main data directory',
           arg='--datadir', dtype=str)

# add instrument
params.set(key='INSTRUMENT', value=None, source=__NAME__,
           desc='The instrument to use',
           arg='--instrument', dtype=str)

# Define whether to do plots
params.set(key='PLOTS', value=False, source=__NAME__,
           desc='Whether to do plots for the compute function',
           arg='--plot', dtype=bool)

# Define whether to do debug plots
params.set(key='DEBUG_PLOTS', value=False, source=__NAME__,
           desc='Whether to do debug plots for the compute function',
           arg='--debugplot', dtype=bool)

# Define whether to skip done files
params.set(key='SKIP_DONE', value=True, source=__NAME__,
           desc='Whether to skip done files',
           arg='--skip', dtype=bool)

# =============================================================================
# Define compute parameters
# =============================================================================
# The object name for the compute function
params.set(key='OBJECT_SCIENCE', value=None, source=__NAME__,
           desc='The object name for the compute function',
           arg='--obj_sci', dtype=str)

# The object name to use for the template
params.set(key='OBJECT_TEMPLATE', value=None, source=__NAME__,
           desc='The object name to use for the template',
           arg='--obj_template', dtype=str)

# Define blaze file
params.set(key='BLAZE_FILE', value=None, source=__NAME__,
           desc='Blaze file to use',
           arg='--blaze', dtype=str)

# Template file to use (if not defined will try to find template for OBJECT
params.set(key='TEMPLATE_FILE', value=None, source=__NAME__,
           desc='Template file to use (if not defined will try to find'
                ' template for OBJECT',
           arg='--template', dtype=str)

# define the input files
params.set(key='INPUT_FILE', value=None, source=__NAME__,
           desc='The input file express to use (i.e. *e2dsff*AB.fits)',
           arg='--input_file', dtype=str)

# Define ref table format
params.set(key='REF_TABLE_FMT', value='csv', source=__NAME__,
           desc='Ref table format (i.e. csv)')

# define the HP width
params.set(key='HP_WIDTH', value=None, source=__NAME__,
           desc='The HP width')

# define the SNR cut off threshold
params.set(key='SNR_THRESHOLD', value=None, source=__NAME__,
           desc='The SNR cut off threshold')


# =============================================================================
# Define header keys
# =============================================================================
# Wave coefficients header key
params.set(key='KW_WAVECOEFFS', value=None, source=__NAME__,
           desc='Wave coefficients header key')

# define wave num orders key in header
params.set(key='KW_WAVEORDN', value=None, source=__NAME__,
           desc='wave num orders key in header')

# define wave degree key in header
params.set(key='KW_WAVEDEGN', value=None, source=__NAME__,
           desc='wave degree key in header')

# define snr keyword
params.set(key='KW_SNR', value=None, source=__NAME__,
           desc='snr key in header')


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

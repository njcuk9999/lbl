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

# =============================================================================
# Define compute parameters
# =============================================================================
# The object name for the compute function
params.set(key='OBJECT', value=None, source=__NAME__,
           desc='The object name for the compute function',
           arg='--obj', dtype=str)

# Define whether to do plots
params.set(key='PLOTS', value=False, source=__NAME__,
           desc='Whether to do plots for the compute function',
           arg='--plot', dtype=bool)

# Define whether to do debug plots
params.set(key='DEBUG_PLOTS', value=False, source=__NAME__,
           desc='Whether to do debug plots for the compute function',
           arg='--debugplot', dtype=bool)

# Define blaze file
params.set(key='BLAZE_FILE', value=False, source=__NAME__,
           desc='Blaze file to use',
           arg='--blaze', dtype=str)

# Template file to use (if not defined will try to find template for OBJECT
params.set(key='TEMPLATE_FILE', value=False, source=__NAME__,
           desc='Template file to use (if not defined will try to find'
                ' template for OBJECT',
           arg='--template', dtype=str)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

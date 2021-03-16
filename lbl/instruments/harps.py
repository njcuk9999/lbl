#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Harps instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
class Harps(base_classes.Instrument):
    def __init__(self, params):
        # set parameters for instrument
        self.params = params
        # call to super function
        super().__init__(self.params['INSTRUMENT'])


# get copy of params
params = parameters.params.copy()

# update properties based on spirou
params['INSTRUMENT'] = 'HARPS'

# push into Instrument class
harps_inst = Harps(params)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

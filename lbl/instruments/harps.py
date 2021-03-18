#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Harps instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.harps.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = default.Instrument
LblException = base_classes.LblException
log = base_classes.log


# =============================================================================
# Define functions
# =============================================================================
class Harps(Instrument):
    def __init__(self, params: base_classes.ParamDict):
        # call to super function
        super().__init__('HARPS')
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()

    # -------------------------------------------------------------------------
    # SPIROU SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    def param_override(self):
        """
        Parameter override for HARPS parameters
        (update default params)

        :return: None - updates self.params
        """
        pass

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

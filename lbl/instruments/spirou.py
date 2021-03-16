#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPIRou instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define Spirou class
# =============================================================================
class Spirou(base_classes.Instrument):
    def __init__(self, params: base_classes.ParamDict):
        # call to super function
        super().__init__('SPIROU')
        # set parameters for instrument
        self.params = params
        # override params
        self.override()

    def override(self):
        # set function name
        func_name = __NAME__ + '.Spirou.override()'
        # set parameters to update
        self.params.set('INSTRUMENT', 'SPIROU', source=func_name)




# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

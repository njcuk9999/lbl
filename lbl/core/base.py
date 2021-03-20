#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base functionality

Created on 2021-03-15

@author: cook
"""
from astropy.time import Time, TimeDelta
import os

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base.py'
__version__ = '0.0.001'
__date__ = '2021-03-15'
__authors__ = 'Neil Cook, Etienne Artigau'

# currently supported instruments
INSTRUMENTS = ['SPIROU']

# log variables
LOG_FILE = os.path.join(os.path.expanduser('~'), 'lbl.log')
LOG_FORMAT = '%(asctime)s %(message)s'

# astropy time is slow the first time - get it done now and do not re-import
__now__ = Time.now()
AstropyTime = Time
AstropyTimeDelta = TimeDelta

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

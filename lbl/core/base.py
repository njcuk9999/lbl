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
__NAME__: str = 'base.py'
__version__: str = '0.0.008'
__date__: str = '2021-04-19'
__authors__: str = 'Neil Cook, Etienne Artigau'
__package__: str = 'lbl'

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

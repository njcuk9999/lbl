#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base functionality

Created on 2021-03-15

@author: cook
"""
import os

from astropy.time import Time, TimeDelta

# =============================================================================
# Define variables
# =============================================================================
__NAME__: str = 'base.py'
__version__: str = '0.63.003'
__date__: str = '2024-02-02'
__authors__: str = ('Neil Cook, Etienne Artigau, Charles Cadieux, Thomas Vandal,'
                    'Ryan Cloutier, Pierre Larue')
__package__: str = 'lbl'

# currently supported instruments
INSTRUMENTS = ['SPIROU', 'HARPS', 'ESPRESSO', 'CARMENES', 'NIRPS_HA',
               'HARPSN', 'MARNOOX', 'SOPHIE']

# log variables
LOG_FILE = os.path.join(os.path.expanduser('~'), 'lbl.log')
LOG_FORMAT = '%(asctime)s %(message)s'
# astropy time is slow the first time - get it done now and do not re-import
__now__ = Time.now()
AstropyTime = Time
AstropyTimeDelta = TimeDelta


# must define the tqdm module as we need to turn it off in certain circumstances
def tqdm_module(use_tqdm: bool = True, verbose: int = 2):
    """
    Get the tqdm module in on or off mode

    :return: function, the tqdm method (or class with a call)
    """

    # this will replace tqdm with the return of the first arg
    def _tqdm(*args, **kwargs):
        _ = kwargs
        return args[0]

    # if we want to use tqdm then use it
    if use_tqdm and verbose == 2:
        from tqdm import tqdm as _tqdm
    # return the tqdm function (or a placeholder that does nothing)
    return _tqdm


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

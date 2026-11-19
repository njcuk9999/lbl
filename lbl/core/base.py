#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base functionality

Created on 2021-03-15

@author: cook
"""
import os
from pathlib import Path

from astropy.time import Time, TimeDelta

# =============================================================================
# Define variables
# =============================================================================
__package_name__: str = 'lbl'
__PATH__ = Path(__file__).parent.parent
with open(__PATH__.joinpath('version.txt'), 'r') as vfile:
    vtext = vfile.readlines()
__NAME__: str = 'base.py'
__version__ = vtext[0].strip()
__date__ = vtext[1].strip()
__authors__: str = ('Neil Cook, Etienne Artigau, Charles Cadieux, '
                    'Thomas Vandal, Ryan Cloutier, Pierre Larue')
__package__: str = 'lbl'

# currently supported instruments
INSTRUMENTS = ['SPIROU', 'HARPS', 'ESPRESSO', 'CARMENES', 'NIRPS_HA',
               'NIRPS_HE', 'HARPSN', 'MAROONX', 'SOPHIE', 'CORALIE',
               'EXPRES', 'NEID', 'ESPADONS']

# add generic instrument last
INSTRUMENTS += ['Generic']

# log variables
TIME_NOW = str(Time.now().unix).replace('.', '_')
LOG_FILE = os.path.join(os.path.expanduser('~'), 'lbl',
                        'lbl_{0}.log'.format(TIME_NOW))
LOG_FORMAT = '%(asctime)s %(message)s'
# make sure log path exists
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))

# astropy time is slow the first time - get it done now and do not re-import
__now__ = Time.now()
AstropyTime = Time
AstropyTimeDelta = TimeDelta


def portable_name(name: str) -> str:
    """
    A file name every disk can hold: the characters Windows (and exFAT)
    refuse in a name, < > : " | ? * and the control characters, as '-'.
    ESO names its calibrations by the time they were taken
    (r.NIRPS.2023-04-11T12:56:58.825_BLAZE_A.fits), and the spectra's
    headers name them so: the window writes such a file under this name,
    and Instrument.blaze_path looks for it under both

    :param name: str, a file name (separators are left as they are)

    :return: str, the same name with only characters every disk takes
    """
    return ''.join('-' if char in '<>:"|?*' or ord(char) < 32 else char
                   for char in str(name))


#: HARPS's two fibres, by their modes (HIERARCH ESO INS MODE): HAM and the
#:    high-efficiency EGGS, each alone or observed together with NIRPS
#:    ('+NIRPS'). The two have their own resolution and velocity zero point
HARPS_MODES = (('HARPS', 'HARPS+NIRPS'), ('EGGS', 'EGGS+NIRPS'))


def harps_eggs(mode) -> bool:
    """
    Whether a HARPS mode (HIERARCH ESO INS MODE) is the EGGS fibre's:
    'EGGS', or 'EGGS+NIRPS' when HARPS observes together with NIRPS

    :param mode: the header's value (None when it has none)

    :return: bool, True for EGGS, False for HAM or no mode
    """
    return str(mode or '').strip().upper().split('+')[0] == 'EGGS'


#: CORALIE's eras as DACE names them (CORALIE98, CORALIE07, CORALIE14,
#:    CORALIE24) and the date each starts [MJD], in the gaps between their
#:    last and first exposures on DACE (1998-11-04 to 2007-04-08, 2007-06-02
#:    to 2014-11-04, 2014-11-13 to 2024-10-15 at 06:00 UT, then from 21:18
#:    UT). The instrument changed at each (CORALIE98 has 68 orders, the
#:    later eras 69): one template must not mix them
CORALIE_ERAS = (('98', -1.0e9), ('07', 54230.0), ('14', 56970.0),
                ('24', 60599.5))


def coralie_era(mjd) -> str:
    """
    CORALIE's era of an exposure by its date

    :param mjd: float, the date of the exposure [MJD; a JD is taken too]

    :return: str, '98', '07', '14' or '24'
    """
    mjd = float(mjd)
    if mjd > 2400000.5:
        mjd -= 2400000.5
    return [era for era, start in CORALIE_ERAS if mjd >= start][-1]


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

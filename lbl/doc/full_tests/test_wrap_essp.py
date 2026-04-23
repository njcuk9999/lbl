#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL using SPIROU (APERO mode)

Created on 2023-06-08
Last updated 2023-06-08

@author: cook
"""
import argparse
import os

from lbl import base
from lbl import lbl_wrap

# =============================================================================
# Define variables
# =============================================================================
# Define path containing test files
#    requires: all_lbl_tests.tar  (contact Etienne Artigau for this file)
TEST_PATH = '/scratch3/lbl/data/test/'

# define which instruments to test (using functions in this module)
INSTRUMENTS = [
               'harpsn_essp',
               'harps_essp',
               'expres_essp',
               'neid_essp'
               ]

# define global params to override
GLOBAL = dict()
GLOBAL['PLOT'] = False

# reset all data before running
GLOBAL['RUN_LBL_RESET'] = True
# Dictionary of table name for the file used in the projection against the
#     derivative. Key is to output column name that will propagate into the
#     final RDB table and the value is the filename of the table. The table
#     must follow a number of characteristics explained on the LBL website.
GLOBAL['RESPROJ_TABLES'] = dict()
#GLOBAL['RESPROJ_TABLES']['DTEMP3000'] = 'temperature_gradient_3000.fits'
# GLOBAL['RESPROJ_TABLES']['DTEMP3500'] = 'temperature_gradient_3500.fits'
# GLOBAL['RESPROJ_TABLES']['DTEMP4000'] = 'temperature_gradient_4000.fits'
# GLOBAL['RESPROJ_TABLES']['DTEMP4500'] = 'temperature_gradient_4500.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP5000'] = 'temperature_gradient_5000.fits'
# GLOBAL['RESPROJ_TABLES']['DTEMP5500'] = 'temperature_gradient_5500.fits'
# GLOBAL['RESPROJ_TABLES']['DTEMP6000'] = 'temperature_gradient_6000.fits'


# =============================================================================
# Define functions
# =============================================================================
def harpsn_essp():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPSN'
    rparams['DATA_SOURCE'] = 'ESSP'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPSN_ESSP')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['SUN']
    rparams['OBJECT_COMPARISON'] = ['SUN']
    rparams['OBJECT_TEFF'] = [5775]
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = False
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # return parameters
    return rparams


def harps_essp():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'ESSP'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPS_ESSP')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['SUN']
    rparams['OBJECT_COMPARISON'] = ['SUN']
    rparams['OBJECT_TEFF'] = [5775]
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = False
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # return parameters
    return rparams


def expres_essp():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'EXPRES'
    rparams['DATA_SOURCE'] = 'ESSP'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'EXPRES_ESSP')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['SUN']
    rparams['OBJECT_COMPARISON'] = ['SUN']
    rparams['OBJECT_TEFF'] = [5775]
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = False
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # return parameters
    return rparams


def neid_essp():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'NEID'
    rparams['DATA_SOURCE'] = 'ESSP'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'NEID_ESSP')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['SUN']
    rparams['OBJECT_COMPARISON'] = ['SUN']
    rparams['OBJECT_TEFF'] = [5775]
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = False
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # return parameters
    return rparams


# =============================================================================
# Define main script to loop through instruments
# =============================================================================
def get_args():
    """
    Define allowed command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='Run LBL tests')
    parser.add_argument('--instruments', type=str, default=None,
                        help='Instrument(s) to run (comma separated list)',
                        choices=INSTRUMENTS)
    # add test path
    parser.add_argument('--testpath', type=str, default=TEST_PATH,
                        help='Path to test data')
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return args


def main():
    # get command line arguments
    args = get_args()
    if args.instruments is None:
        instruments = INSTRUMENTS
    else:
        instruments = args.instruments.split(',')
    # deal with overriding test path
    if os.path.exists(args.testpath):
        global TEST_PATH
        TEST_PATH = args.testpath
    # loop around instruments
    for instrument in instruments:
        # get rparams
        try:
            rparams = eval(instrument)()
        except Exception as _:
            msg = 'No Instrument definition for {0}. Skipping'
            print(msg.format(instrument))
            continue
        # make sure we have instrument available
        if rparams['INSTRUMENT'] not in base.INSTRUMENTS:
            msg = 'Instrument {0} not available. Skipping'
            print(msg.format(rparams['INSTRUMENT']))
            continue
        # make sure we have instrument
        if not os.path.exists(rparams['DATA_DIR']):
            msg = 'Instrument directory for {0} missing. Skipping'
            print(msg.format(instrument))
            continue
        # ---------------------------------------------------------------------
        # Run the wrapper code using the above settings
        # ---------------------------------------------------------------------
        # override global params
        for key in GLOBAL:
            rparams[key] = GLOBAL[key]
        # run main
        lbl_wrap(rparams)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    main()

# =============================================================================
# End of code
# =============================================================================

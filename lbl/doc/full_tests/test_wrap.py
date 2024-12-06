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
INSTRUMENTS = ['carmenes_vis',
               'espresso',
               'harps_orig', 'harps_eso',
               'harpsn_orig', 'harpsn_eso',
               'nirps_ha_apero', 'nirps_he_apero',
               'nirps_ha_eso', 'nirps_he_eso',
               'spirou_apero', 'spirou_cadc',
               'maroonx_b', 'maroonx_r',
               'sophie',
               'coralie'
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
GLOBAL['RESPROJ_TABLES']['DTEMP3000'] = 'temperature_gradient_3000.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP3500'] = 'temperature_gradient_3500.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP4000'] = 'temperature_gradient_4000.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP4500'] = 'temperature_gradient_4500.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP5000'] = 'temperature_gradient_5000.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP5500'] = 'temperature_gradient_5500.fits'
GLOBAL['RESPROJ_TABLES']['DTEMP6000'] = 'temperature_gradient_6000.fits'


# =============================================================================
# Define functions
# =============================================================================
def carmenes_vis():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'CARMENES'
    rparams['DATA_SOURCE'] = 'None'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'CARMENES-vis')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['K2-18']
    rparams['OBJECT_TEMPLATE'] = ['K2-18']
    rparams['OBJECT_TEFF'] = [3500]
    rparams['BLAZE_FILE'] = None
    rparams['BLAZE_CORRECTED'] = True
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def espresso():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'ESPRESSO'
    rparams['DATA_SOURCE'] = 'None'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'ESPRESSO')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['LHS1140']
    rparams['OBJECT_TEMPLATE'] = ['LHS1140']
    rparams['OBJECT_TEFF'] = [3216]
    rparams['BLAZE_FILE'] = 'M.ESPRESSO.2020-09-03T23_38_07.710.fits'
    rparams['BLAZE_CORRECTED'] = True
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def harps_orig():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'ORIG'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPS_ORIG')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = 'HARPS.2014-09-02T21_06_48.529_blaze_A.fits'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def harps_eso():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'ESO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPS_ESO')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['GJ682']
    rparams['OBJECT_TEMPLATE'] = ['GJ682']
    rparams['OBJECT_TEFF'] = [3349]
    rparams['BLAZE_CORRECTED'] = True
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def nirps_ha_apero():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'NIRPS_HA'
    rparams['DATA_SOURCE'] = 'APERO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'NIRPS-HA-apero')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = 'E967E354D8_pp_blaze_A.fits'
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


def nirps_he_apero():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'NIRPS_HE'
    rparams['DATA_SOURCE'] = 'APERO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'NIRPS-HE-apero')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = '5F59EED6FE_pp_blaze_A.fits'
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


def nirps_ha_eso():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'NIRPS_HA'
    rparams['DATA_SOURCE'] = 'ESO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'NIRPS-HA-eso')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = 'r.NIRPS.2023-03-05T103313.641_BLAZE_A.fits'
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


def nirps_he_eso():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'NIRPS_HE'
    rparams['DATA_SOURCE'] = 'ESO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'NIRPS-HE-eso')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = 'r.NIRPS.2023-01-22T144532.460_BLAZE_A.fits'
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


def spirou_apero():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_SOURCE'] = 'APERO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'SPIROU-apero')
    rparams['DATA_TYPES'] = ['FP', 'SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['FP', 'GL699']
    rparams['OBJECT_TEMPLATE'] = ['FP', 'GL699']
    rparams['OBJECT_TEFF'] = [300, 3224]
    rparams['BLAZE_FILE'] = 'F8018F48F0_pp_blaze_AB.fits'
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


def spirou_cadc():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_SOURCE'] = 'CADC'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'SPIROU-cadc')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['GL699']
    rparams['OBJECT_TEMPLATE'] = ['GL699']
    rparams['OBJECT_TEFF'] = [3224]
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


def harpsn_orig():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPSN'
    rparams['DATA_SOURCE'] = 'ORIG'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPSN_ORIG')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['HARPSN_DRS3.7_TOI1266']
    rparams['OBJECT_TEMPLATE'] = ['HARPSN_DRS3.7_TOI1266']
    rparams['OBJECT_TEFF'] = [5668]
    rparams['BLAZE_FILE'] = 'HARPN.2022-05-12T05-15-56.213_blaze_A.fits'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def harpsn_eso():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPSN'
    rparams['DATA_SOURCE'] = 'ESO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPSN_ESO')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['HARPSN_DRS2.3.5_TOI1266']
    rparams['OBJECT_TEMPLATE'] = ['HARPSN_DRS2.3.5_TOI1266']
    rparams['OBJECT_TEFF'] = [5668]
    rparams['BLAZE_CORRECTED'] = True
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def maroonx_b():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'MAROONX'
    rparams['DATA_SOURCE'] = 'BLUE'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'MAROONX_b')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['GJ486']
    rparams['OBJECT_TEMPLATE'] = ['GJ486']
    rparams['OBJECT_TEFF'] = [3400]
    rparams['BLAZE_FILE'] = '20200603T13_masterflat_backgroundsubtracted_FFFFF_x_0000.hd5'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def maroonx_r():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'MAROONX'
    rparams['DATA_SOURCE'] = 'RED'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'MAROONX_r')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['GJ486']
    rparams['OBJECT_TEMPLATE'] = ['GJ486']
    rparams['OBJECT_TEFF'] = [3400]
    rparams['BLAZE_FILE'] = '20200603T13_masterflat_backgroundsubtracted_FFFFF_x_0000.hd5'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def sophie():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'SOPHIE'
    rparams['DATA_SOURCE'] = 'None'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'SOPHIE')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['Gl873']
    rparams['OBJECT_TEMPLATE'] = ['Gl873']
    rparams['OBJECT_TEFF'] = [3228]
    rparams['BLAZE_FILE'] = 'SOPHIE.2021-08-31T15-29-01.650_blaze_A.fits'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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


def coralie():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'CORALIE'
    rparams['DATA_SOURCE'] = 'None'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'CORALIE')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['HD114082']
    rparams['OBJECT_TEMPLATE'] = ['HD114082']
    rparams['OBJECT_TEFF'] = [6600]
    rparams['BLAZE_FILE'] = 'CORALIE.2022-02-04T22:01:57.000_blaze_A.fits'
    rparams['BLAZE_CORRECTED'] = False
    # what to run and skip if already on disk
    rparams['RUN_LBL_TELLUCLEAN'] = True
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL using SPIROU (APERO mode)

Created on 2023-06-08
Last updated 2023-06-08

@author: cook
"""
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
               'harps',
               'harpsn_v2', 'harpsn_v3',
               'nirps_ha_apero', 'nirps_he_apero',
               'spirou_apero', 'spirou_cadc',
               'maroonx_b', 'maroonx_r']


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
    rparams['BLAZE_FILE'] = 'carmenes_dummy_blaze.fits'
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


def harps():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'None'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPS')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['PROXIMA']
    rparams['OBJECT_TEMPLATE'] = ['PROXIMA']
    rparams['OBJECT_TEFF'] = [2810]
    rparams['BLAZE_FILE'] = 'HARPS.2014-09-02T21_06_48.529_blaze_A.fits'
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


# TODO: Add NIRPS-HA-ESO

# TODO: Add NIRPS-HE-ESO


def spirou_apero():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_SOURCE'] = 'APERO'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'SPIROU-apero')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['GL699']
    rparams['OBJECT_TEMPLATE'] = ['GL699']
    rparams['OBJECT_TEFF'] = [3224]
    rparams['BLAZE_FILE'] = 'F8018F48F0_pp_blaze_AB.fits'
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


# TODO: Add spirou_cadc


def harpsn_v2():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'v2'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPSN_v2')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['HARPSN_DRS2.3.5_TOI1266']
    rparams['OBJECT_TEMPLATE'] = ['HARPSN_DRS2.3.5_TOI1266']
    rparams['OBJECT_TEFF'] = [5668]
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


def harpsn_v3():
    # set up parameters
    rparams = dict()
    # LBL parameters
    rparams['INSTRUMENT'] = 'HARPS'
    rparams['DATA_SOURCE'] = 'v3'
    rparams['DATA_DIR'] = os.path.join(TEST_PATH, 'HARPSN_v3')
    rparams['DATA_TYPES'] = ['SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['HARPSN_DRS3.7_TOI1266']
    rparams['OBJECT_TEMPLATE'] = ['HARPSN_DRS3.7_TOI1266']
    rparams['OBJECT_TEFF'] = [5668]
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


def main():
    # loop around instruments
    for instrument in INSTRUMENTS:
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

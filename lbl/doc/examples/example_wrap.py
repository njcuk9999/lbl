#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for HARPS (Proxima-tc)

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import lbl_wrap


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set up parameters
    rparams = dict()
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_DIR'] = '/data/spirou/data/lbl/'
    # science criteria
    rparams['DATA_TYPES'] = ['FP', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE',
                             'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE',
                             'SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['FP', 'GJ1002', 'GJ1286', 'GJ1289', 'GL15A',
                                 'GL411', 'GL412A', 'GL687', 'GL699', 'GL905']
    rparams['OBJECT_TEMPLATE'] = ['FP', 'GJ1002', 'GJ1286', 'GJ1289', 'GL15A',
                                  'GL411', 'GL412A', 'GL687', 'GL699', 'GL905']
    rparams['OBJECT_TEFF'] = [300, 2900, 2900, 3250, 3603, 3550, 3549, 3420,
                              3224, 2930]
    # what to run
    rparams['RUN_LBL_TELLUCLEAN'] = True
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    # whether to skip done files
    rparams['SKIP_LBL_TELLUCLEAN'] = False
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # run main
    lbl_wrap(rparams)


# =============================================================================
# End of code
# =============================================================================

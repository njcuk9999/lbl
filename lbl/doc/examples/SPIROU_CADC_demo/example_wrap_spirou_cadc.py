#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL using SPIROU (CADC mode)

Created on 2022-09-26
Last updated 2022-09-26

@author: artigau, cook
"""
from lbl import lbl_wrap


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set up parameters
    rparams = dict()
    # -------------------------------------------------------------------------
    # LBL parameters
    # -------------------------------------------------------------------------
    # You may also add any constant here to override the default value
    #     (see README for details) - this is NOT recommended for non developers
    #   Note this may have undesired affects as these parameters apply globally
    #     for all LBL recipes
    # -------------------------------------------------------------------------
    # This is the instrument name
    #   Currently supported instruments are SPIROU, HARPS, ESPRESSO, CARMENES
    #                                       NIRPS_HE, NIRPS_HA, MAROONX
    rparams['INSTRUMENT'] = 'SPIROU'
    #   Data source must be as follows:
    #       SPIROU: APERO or CADC
    #       NIRPS_HA: APERO or ESO
    #       NIRPS_HE: APERO or ESO
    #       CARMENES: None
    #       ESPRESSO: None
    #       HARPS: None
    #       MAROONX: RED or BLUE
    rparams['DATA_SOURCE'] = 'CADC'
    # The data directory where all data is stored under - this should be an
    #    absolute path
    rparams['DATA_DIR'] = '/data/spirou/data/lbl/'
    # The input file string (including wildcards) - if not set will use all
    #   files in the science directory (for this object name)
    # rparams['INPUT_FILE'] = '2*e2dsff_AB.fits'

    # -------------------------------------------------------------------------
    # special parameters only for SPIROU CADC
    # -------------------------------------------------------------------------
    # Must set the fiber type (AB, A, B, C) as this is control by extname
    rparams['FORCE_FIBER'] = 'AB'
    # Must set the flux column (controlled by extname)
    #   - Can be Flux (for e.fits and t.fits)
    #   - Can be Pol or StokesI or Null1 or Null2 (for p.fits)
    rparams['FLUX_EXTENSION_NAME'] = 'Flux'

    # -------------------------------------------------------------------------
    # science criteria
    # -------------------------------------------------------------------------
    # The data type (either SCIENCE or FP or LFC)
    rparams['DATA_TYPES'] = ['FP', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE',
                             'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE',
                             'SCIENCE']
    # The object name (this is the directory name under the /science/
    #    sub-directory and thus does not have to be the name in the header
    rparams['OBJECT_SCIENCE'] = ['FP', 'GJ1002', 'GJ1286', 'GJ1289', 'GL15A',
                                 'GL411', 'GL412A', 'GL687', 'GL699', 'GL905']
    # This is the template that will be used or created (depending on what is
    #   run)
    rparams['OBJECT_TEMPLATE'] = ['FP', 'GJ1002', 'GJ1286', 'GJ1289', 'GL15A',
                                  'GL411', 'GL412A', 'GL687', 'GL699', 'GL905']
    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    rparams['OBJECT_TEFF'] = [300, 2900, 2900, 3250, 3603, 3550, 3549, 3420,
                              3224, 2930]
    # -------------------------------------------------------------------------
    # what to run and skip if already on disk
    # -------------------------------------------------------------------------
    # Whether to run the telluric cleaning process (NOT recommended for data
    #   that has better telluric cleaning i.e. SPIROU using APERO)
    rparams['RUN_LBL_TELLUCLEAN'] = False
    # Whether to create templates from the data in the science directory
    #   If a template has been supplied from elsewhere this set is NOT required
    rparams['RUN_LBL_TEMPLATE'] = True
    # Whether to create a mask using the template created or supplied
    rparams['RUN_LBL_MASK'] = True
    # Whether to run the LBL compute step - which computes the line by line
    #   for each observation
    rparams['RUN_LBL_COMPUTE'] = True
    # Whether to run the LBL compile step - which compiles the rdb file and
    #   deals with outlier rejection
    rparams['RUN_LBL_COMPILE'] = True
    # whether to skip observations if a file is already on disk (useful when
    #   adding a few new files) there is one for each RUN_XXX step
    #   - Note cannot skip tellu clean
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # -------------------------------------------------------------------------
    # LBL settings
    # -------------------------------------------------------------------------
    # You can change any setting in parameters (or override those changed
    #   by specific instruments) here
    # -------------------------------------------------------------------------
    # Advanced settings
    #   Do not use without contacting the LBL developers
    # -------------------------------------------------------------------------
    # Dictionary of table name for the file used in the projection against the
    #     derivative. Key is to output column name that will propagate into the
    #     final RDB table and the value is the filename of the table. The table
    #     must follow a number of characteristics explained on the LBL website.
    # rparams['RESPROJ_TABLES'] = []]

    # Rotational velocity parameters, should be a list of two values, one being
    #     the epsilon and the other one being the vsini in km/s as defined in the
    #     PyAstronomy.pyasl.rotBroad function
    # rparams['ROTBROAD'] = []]

    # turn on plots
    rparams['PLOT'] = False

    # -------------------------------------------------------------------------
    # Run the wrapper code using the above settings
    # -------------------------------------------------------------------------
    # run main
    lbl_wrap(rparams)


# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrap file for LBL recipes

Created on 2024-07-08 14:09:38.491

@author: Neil Cook, Etienne Artigau, Charles Cadieux, Thomas Vandal, Ryan Cloutier, Pierre Larue

user: cook
lbl version: 0.63.009
lbl date: 2024-07-08
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
    #   Currently supported instruments are 
	#		SPIROU
	#		HARPS
	#		ESPRESSO
	#		CARMENES
	#		NIRPS_HA
	#		NIRPS_HE
	#		HARPSN
	#		MAROONX
	#		SOPHIE
    rparams['INSTRUMENT'] = '{INSTRUMENT}'
    #   Data source must be as follows: 
	#		SPIROU: APERO or CADC
	#		NIRPS_HA: APERO or CADC or ESO
	#		NIRPS_HE: APERO or CADC or ESO
	#		HARPS: ORIG or ESO
	#		CARMENES: None
	#		ESPRESSO: None
	#		HARPSN: ORIG or ESO
	#		MAROONX: RED or BLUE
	#		SOPHIE: None
    rparams['DATA_SOURCE'] = '{DATA_SOURCE}}'
    # The data directory where all data is stored under - this should be an
    #    absolute path
    rparams['DATA_DIR'] = '{DATA_DIR}'
    # The input file string (including wildcards) - if not set will use all
    #   files in the science directory (for this object name)
    # rparams['INPUT_FILE'] = '*'
    # Override the blaze filename
    #      (if not set will use the default for instrument)
    # rparams['BLAZE_FILE'] = 'blaze.fits'
    # -------------------------------------------------------------------------
    # science criteria
    # -------------------------------------------------------------------------
    # The data type (either SCIENCE or FP or LFC)
    rparams['DATA_TYPES'] = ['SCIENCE']
    # The object name (this is the directory name under the /science/
    #    sub-directory and thus does not have to be the name in the header
    rparams['OBJECT_SCIENCE'] = ['OBJECT_SCIENCE_NAME']
    # This is the comparison template that will be used or created
    #   (depending on what is run)
    rparams['OBJECT_COMPARISON'] = ['OBJECT_COMPARISON_NAME']
    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    rparams['OBJECT_TEFF'] = [3000]
    # -------------------------------------------------------------------------
    # what to run and skip if already on disk
    # -------------------------------------------------------------------------
    # Whether to run the telluric cleaning process (NOT recommended for data
    #   that has better telluric cleaning i.e. SPIROU using APERO)
    rparams['RUN_LBL_TELLUCLEAN'] = True
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
_LBL_COMPILE}
    # -------------------------------------------------------------------------
    # PCA2D precleaning (optional, off unless uncommented)
    # -------------------------------------------------------------------------
    # Fits every exposure as the star in its own frame plus a few components
    #   in the star's frame (n_star) and in the observer's frame (n_earth),
    #   and divides the observer's part (residual tellurics, airglow,
    #   detector) out of the spectra before the velocities. LBL measures the
    #   delivered object, writes the corrected one to science/<object>_pca2d,
    #   and measures that; the report then holds both series, the river plots
    #   of the correction and LBL's own pages. With RUN_LBL_TELLUCLEAN it
    #   comes after the telluric cleaning (<object>_tc_pca2d).
    #   Every key, and a worked example: lbl/doc/PCA2D.md and
    #   lbl/doc/example_wrap_pca2d.py
    # rparams['PCA2D'] = dict(n_star=1, n_earth=3)
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
    # rparams['RESPROJ_TABLES'] = []

    # Rotational velocity parameters, should be a list of two values, one being
    #     the epsilon and the other one being the vsini in km/s as defined in the
    #     PyAstronomy.pyasl.rotBroad function
    # rparams['ROTBROAD'] = []

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

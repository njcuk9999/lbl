#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Default parameters (non-instrument specific)

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.parameters.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# set up parameter dictionary
params = base_classes.ParamDict()

# =============================================================================
# Define general parameters
# =============================================================================

# add default params - can be None
params.set(key='CONFIG_FILE', value=None, source=__NAME__,
           desc='Config file for user settings',
           arg='--config', dtype=str)

# add main data directory (structure assumed below)
params.set(key='DATA_DIR', value=None, source=__NAME__,
           desc='Main data directory',
           arg='--datadir', dtype=str, not_none=True)

# add masks sub directory
params.set(key='MASK_SUBDIR', value='masks', source=__NAME__,
           desc='mask sub directory', arg='--maskdir', dtype=str)

# add template sub directory
params.set(key='TEMPLATE_SUBDIR', value='templates', source=__NAME__,
           desc='template sub directory', arg='--templatedir', dtype=str)

# add calib sub directory
params.set(key='CALIB_SUBDIR', value='calib', source=__NAME__,
           desc='calib sub directory', arg='--calibdir', dtype=str)

# add science sub directory
params.set(key='SCIENCE_SUBDIR', value='science', source=__NAME__,
           desc='science sub directory', arg='--scidir', dtype=str)

# add lblrv sub directory
params.set(key='LBLRV_SUBDIR', value='lblrv', source=__NAME__,
           desc='LBL RV sub directory', arg='--lblrvdir', dtype=str)

# add lblreftable sub directory
params.set(key='LBLREFTAB_SUBDIR', value='lblreftable', source=__NAME__,
           desc='LBL ref table sub directory', arg='--lblreftabdir', dtype=str)

# add lblrdb sub directory
params.set(key='LBLRDB_SUBDIR', value='lblrdb', source=__NAME__,
           desc='LBL RDB sub directory', arg='--lblrdbdir', dtype=str)

# add instrument
params.set(key='INSTRUMENT', value=None, source=__NAME__,
           desc='The instrument to use',
           arg='--instrument', dtype=str, not_none=True)

# Define whether to do plots
params.set(key='PLOT', value=False, source=__NAME__,
           desc='Whether to do plots for the compute function',
           arg='--plot', dtype=bool)

# Define whether to do debug plots
params.set(key='DEBUG_PLOT', value=False, source=__NAME__,
           desc='Whether to do debug plots for the compute function',
           arg='--debugplot', dtype=bool)

# Define whether to skip done files
params.set(key='SKIP_DONE', value=True, source=__NAME__,
           desc='Whether to skip done files',
           arg='--skip', dtype=bool)

# =============================================================================
# Define compute parameters
# =============================================================================
# The object name for the compute function
params.set(key='OBJECT_SCIENCE', value=None, source=__NAME__,
           desc='The object name for the compute function',
           arg='--obj_sci', dtype=str, not_none=True)

# The object name to use for the template
params.set(key='OBJECT_TEMPLATE', value=None, source=__NAME__,
           desc='The object name to use for the template',
           arg='--obj_template', dtype=str, not_none=True)

# Define blaze file - can be None
params.set(key='BLAZE_FILE', value=None, source=__NAME__,
           desc='Blaze file to use',
           arg='--blaze', dtype=str)

# Template file to use (if not defined will try to find template for OBJECT)
#   - can be None
params.set(key='TEMPLATE_FILE', value=None, source=__NAME__,
           desc='Template file to use (if not defined will try to find'
                ' template for OBJECT',
           arg='--template', dtype=str)

# define the input files
params.set(key='INPUT_FILE', value=None, source=__NAME__,
           desc='The input file express to use (i.e. *e2dsff*AB.fits)',
           arg='--input_file', dtype=str,
           not_none=True)

# Define ref table format
params.set(key='REF_TABLE_FMT', value='csv', source=__NAME__,
           desc='Ref table format (i.e. csv)')

# define the High pass width [km/s]
params.set(key='HP_WIDTH', value=None, source=__NAME__,
           desc='The HP width', not_none=True)

# define the SNR cut off threshold
params.set(key='SNR_THRESHOLD', value=None, source=__NAME__,
           desc='The SNR cut off threshold', not_none=True)

# define switch whether to use noise model
params.set(key='USE_NOISE_MODEL', value=False, source=__NAME__,
           desc='Switch whether to use noise model or not for the rms')

# define the rough CCF rv minimum limit in m/s
params.set(key='ROUGH_CCF_MIN_RV', value=-3e5, source=__NAME__,
           desc='The rough CCF rv minimum limit in m/s')

# define the rough CCF rv maximum limit in m/s
params.set(key='ROUGH_CCF_MAX_RV', value=3e5, source=__NAME__,
           desc='The rough CCF rv maximum limit in m/s')

# define the rough CCF rv step in m/s
params.set(key='ROUGH_CCF_RV_STEP', value=500, source=__NAME__,
           desc='The rought CCF rv step in m/s')

# define the rough CCF ewidth guess for fit in m/s
params.set(key='ROUGH_CCF_EWIDTH_GUESS', value=2000, source=__NAME__,
           desc='The rough CCF ewidth guess for fit in m/s')

# define the number of iterations to do to converge during compute rv
params.set(key='COMPUTE_RV_N_ITERATIONS', value=10, source=__NAME__,
           desc='The number of iterations to do to converge during compute RV')

# define the plot order for compute rv model plot
params.set(key='COMPUTE_MODEL_PLOT_ORDERS', value=None, source=__NAME__,
           desc='The plot orders for compute rv model plot', not_none=True)

# define the minimum line width (in pixels) to consider line valid
params.set(key='COMPUTE_LINE_MIN_PIX_WIDTH', value=5, source=__NAME__,
           desc='The minimum line width (in pixels) to consider line valid')

# define the threshold in sigma on nsig (dv / dvrms) to keep valid
params.set(key='COMPUTE_LINE_NSIG_THRES', value=8, source=__NAME__,
           desc='The threshold in sigma on nsig (dv / dvrms) to keep valid')

# define the fraction of the bulk error the rv mean must be above for compute
#   rv to have converged
params.set(key='COMPUTE_RV_BULK_ERROR_CONVERGENCE', value=0.2, source=__NAME__,
           desc='fraction of the bulk error the rv mean must be above for '
                'compute rv to have converged')

# define the maximum number of iterations deemed to lead to a good RV
params.set(key='COMPUTE_RV_MAX_N_GOOD_ITERS', value=8, source=__NAME__,
           desc='The maximum number of iterations deemed to lead to a good RV')


# =============================================================================
# Define header keys
# =============================================================================
# Wave coefficients header key
params.set(key='KW_WAVECOEFFS', value=None, source=__NAME__,
           desc='Wave coefficients header key', not_none=True)

# define wave num orders key in header
params.set(key='KW_WAVEORDN', value=None, source=__NAME__,
           desc='wave num orders key in header', not_none=True)

# define wave degree key in header
params.set(key='KW_WAVEDEGN', value=None, source=__NAME__,
           desc='wave degree key in header', not_none=True)

# define the key that gives the mid exposure time in MJD
params.set(key='KW_MID_EXP_TIME', value=None, source=__NAME__,
           desc='mid exposure time in MJD', not_none=True)

# define snr keyword
params.set(key='KW_SNR', value=None, source=__NAME__,
           desc='snr key in header', not_none=True)

# define the BERV keyword
params.set(key='KW_BERV', value=None, source=__NAME__,
           desc='the barycentric correction keyword', not_none=True)

# define the Blaze calibration file
params.set(key='KW_BLAZE_FILE', value=None, source=__NAME__,
           desc='The Blaze calibration file', not_none=True)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

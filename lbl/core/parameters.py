#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Default parameters (non-instrument specific)

Created on 2021-03-15

@author: cook
"""
from typing import Any, Tuple

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
           desc='Config file for user settings (absolute path)',
           arg='--config', dtype=str)

# add main data directory (structure assumed below)
params.set(key='DATA_DIR', value=None, source=__NAME__,
           desc='Main data directory (absolute path)',
           arg='--datadir', dtype=str, not_none=True)

# add masks sub directory (relative to data directory)
params.set(key='MASK_SUBDIR', value='masks', source=__NAME__,
           desc='mask sub directory (relative to data directory)',
           arg='--maskdir', dtype=str)

# add template sub directory (relative to data directory)
params.set(key='TEMPLATE_SUBDIR', value='templates', source=__NAME__,
           desc='template sub directory (relative to data directory)',
           arg='--templatedir', dtype=str)

# add calib sub directory (relative to data directory)
params.set(key='CALIB_SUBDIR', value='calib', source=__NAME__,
           desc='calib sub directory (relative to data directory)',
           arg='--calibdir', dtype=str)

# add science sub directory (relative to data directory)
params.set(key='SCIENCE_SUBDIR', value='science', source=__NAME__,
           desc='science sub directory (relative to data directory)',
           arg='--scidir', dtype=str)

# add lblrv sub directory (relative to data directory)
params.set(key='LBLRV_SUBDIR', value='lblrv', source=__NAME__,
           desc='LBL RV sub directory (relative to data directory)',
           arg='--lblrvdir', dtype=str)

# add lblreftable sub directory (relative to data directory)
params.set(key='LBLREFTAB_SUBDIR', value='lblreftable', source=__NAME__,
           desc='LBL ref table sub directory (relative to data directory)',
           arg='--lblreftabdir', dtype=str)

# add lblrdb sub directory (relative to data directory)
params.set(key='LBLRDB_SUBDIR', value='lblrdb', source=__NAME__,
           desc='LBL RDB sub directory (relative to data directory)',
           arg='--lblrdbdir', dtype=str)

# add instrument
params.set(key='INSTRUMENT', value=None, source=__NAME__,
           desc='The instrument to use', options=base.INSTRUMENTS,
           arg='--instrument', dtype=str, not_none=True)

# add data source
params.set(key='DATA_SOURCE', value='None', source=__NAME__,
           desc='The data source to use',
           arg='--data_source', dtype=str, not_none=True)

# add instrument earth location (for use in astropy.coordinates.EarthLocation)
params.set(key='EARTH_LOCATION', value=None, source=__NAME__,
           desc='The instrument earth location (for use in '
                'astropy.coordinates.EarthLocation)',
           dtype=str, not_none=True)

# Define whether to skip done files
params.set(key='SKIP_DONE', value=False, source=__NAME__,
           desc='Whether to skip done files',
           arg='--skip_done', dtype=bool)

# Define whether to skip done files
params.set(key='OVERWRITE', value=False, source=__NAME__,
           desc='Whether to overwrite files that already exist',
           arg='--overwrite', dtype=bool)

# Define the verbosity level
params.set(key='VERBOSE', value=2, source=__NAME__, options=[0, 1, 2],
           desc='Verbosity 0=only warnings/errors, 1=info/warnings/errors,'
                '2=general/info/warning/errors  (default is 2)',
           arg='--verbose', dtype=int)

# Define whether to use tqdm (if verbose = 2)
params.set(key='USE_TQDM', value=True, source=__NAME__, dtype=int,
           desc='Whether to use tqdm module in loops (only for verbose=2)')

# Define whether to add program id to the logging message
params.set(key='PROGRAM', value=None, source=__NAME__, dtype=str,
           desc='Whether to add program id to the logging message',
           arg='--program')

# Define whether to write RDB fits files
params.set(key='WRITE_RDB_FITS', value=True, source=__NAME__,
           dtype=bool, desc='whether to write RDB fits files')

# Define fiber of files (required for some modes before reading header
#    - not used in many modes as we can read header of input files)
params.set(key='FORCE_FIBER', value=None, source=__NAME__, dtype=str,
           desc='Fiber of files (required for some modes before reading '
                'header - not used in many modes as we can read header of '
                'input files)')

# Define the flux extension name (required for some modes)
params.set(key='FLUX_EXTENSION_NAME', value=None, source=__NAME__, dtype=str,
           desc='the flux extension name (required for some modes)')

# Define which iteration we are running (for multiprocessing)
#    -1 means no multiprocessing
params.set(key='ITERATION', value=-1, source=__NAME__, dtype=int,
              desc='which iteration we are running (for multiprocessing) '
                    '-1 means no multiprocessing',
           arg='--iteration')

# Define the total number of iterations (for multiprocessing)
#     -1 means no multiprocessing
params.set(key='TOTAL', value=-1, source=__NAME__, dtype=int,
           desc='the total number of iterations (for multiprocessing) '
                '-1 means no multiprocessing',
           arg='--total')

# =============================================================================
# Define common parameters (between compute / compil)
# =============================================================================
# The object name for the compute function
params.set(key='OBJECT_SCIENCE', value=None, source=__NAME__,
           desc='The object name for the compute function',
           arg='--obj_sci', dtype=str, not_none=True)

# The object name to use for the template
params.set(key='OBJECT_TEMPLATE', value=None, source=__NAME__,
           desc='The object name to use for the template '
                '(If None set to OBJECT_SCIENCE)',
           arg='--obj_template', dtype=str)

# Set the data type (science, FP or LFC)
params.set(key='DATA_TYPE', value=None, source=__NAME__,
           desc='Set the data type (science, FP or LFC)', dtype=str,
           arg='--data_type', not_none=True)

# define the mask type (pos, neg, full)
params.set(key='SCIENCE_MASK_TYPE', value='full',
           desc='the mask type (pos, neg, full)', dtype=str)

# define the mask type (pos, neg, full)
params.set(key='FP_MASK_TYPE', value='neg',
           desc='the mask type (pos, neg, full)', dtype=str)

# define the mask type (pos, neg, full)
params.set(key='LFC_MASK_TYPE', value='neg',
           desc='the mask type (pos, neg, full)', dtype=str)

# =============================================================================
# Define compute parameters
# =============================================================================
# Define blaze file - can be None
params.set(key='BLAZE_FILE', value=None, source=__NAME__,
           desc='Blaze file to use (must be present in the CALIB directory)',
           arg='--blaze', dtype=str)

# Template file to use (if not defined will try to find template for OBJECT)
#   - can be None
params.set(key='TEMPLATE_FILE', value=None, source=__NAME__,
           desc='Template file to use (if not defined will try to find'
                ' template for OBJECT_TEMPLATE) must be present in the'
                'TEMPLATES directory',
           arg='--template', dtype=str)

# define the input files
params.set(key='INPUT_FILE', value=None, source=__NAME__,
           desc='The input file expression to use (i.e. *e2dsff*AB.fits)',
           arg='--input_file', dtype=str,
           not_none=True)

# override the mask to be used
params.set(key='MASK_FILE', value='None', source=__NAME__,
           desc='Override the mask to be used (within mask dir or full path)',
           arg='--mask_file', dtype=str)

# Define ref table format
params.set(key='REF_TABLE_FMT', value='csv', source=__NAME__,
           desc='Ref table format (i.e. csv)')

# define the High pass width [km/s]
params.set(key='HP_WIDTH', value=None, source=__NAME__,
           desc='The High pass width [km/s]', not_none=True)

# define the SNR cut off threshold
params.set(key='SNR_THRESHOLD', value=None, source=__NAME__,
           desc='The SNR cut off threshold', not_none=True)

# define switch whether to use noise model for RMS calculation
params.set(key='USE_NOISE_MODEL', value=False, source=__NAME__,
           desc='Switch whether to use noise model or not for the RMS '
                'calculation', options=[True, False])

# define the running size (in m/s) for the RMS calculation, default = 200km/s
# hould be big enough to be representative of the noise but
# small enough to be representative of variations through
# the domain. For typical pRV instruments, can be kept as a gobal value, no need
# to change it. If we ever go with a much coarser sampling, this value will need
# to be adjusted.
params.set(key='NOISE_SAMPLING_WIDTH', value=2e5, source=__NAME__,
           desc='Size of the running window (in m/s) for the RMS calculation.')

# define the rough CCF rv minimum limit in m/s
params.set(key='ROUGH_CCF_MIN_RV', value=-3e5, source=__NAME__,
           desc='The rough CCF rv minimum limit in m/s')

# define the rough CCF rv maximum limit in m/s
params.set(key='ROUGH_CCF_MAX_RV', value=3e5, source=__NAME__,
           desc='The rough CCF rv maximum limit in m/s')

# define the rough CCF step size in m/s
params.set(key='ROUGH_CCF_STEP_RV', value=500, source=__NAME__,
           desc='The rough CCF step size in m/s')

# define the rough CCF filter size in m/s
params.set(key='ROUGH_CCF_FILTER_SIZE', value=100000, source=__NAME__,
           desc='The rough CCF filter size in m/s')

# define which bands to use for the clean CCF (see astro.ccf_regions)
params.set(key='CCF_CLEAN_BANDS', value=None, source=__NAME__,
           desc='which bands to use for the clean CCF (see astro.ccf_regions) ',
           not_none=True)

# define the minimum SNR for the rough CCF. Below that, the CCF is not
# considered to be reliable
params.set(key='CCF_SNR_MIN', value=7, source=__NAME__,
           desc='Minimum SNR of CCF for starting point '
                'velocity to be considered valid')

# define the rough CCF ewidth guess for fit in m/s
params.set(key='ROUGH_CCF_EWIDTH_GUESS', value=2000, source=__NAME__,
           desc='The rough CCF ewidth guess for fit in m/s')

# define the number of iterations to do to converge during compute rv
params.set(key='COMPUTE_RV_N_ITERATIONS', value=20, source=__NAME__,
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
params.set(key='COMPUTE_RV_BULK_ERROR_CONVERGENCE', value=1.0, source=__NAME__,
           desc='fraction of the bulk error the rv mean must be above for '
                'compute rv to perform one more iteration')

# define the maximum number of iterations deemed to lead to a good RV
params.set(key='COMPUTE_RV_MAX_N_GOOD_ITERS', value=15, source=__NAME__,
           desc='The maximum number of iterations deemed to lead to a good RV')

# define the number of sigma to clip based on the rms away from the model
#   (sigma clips science data)
params.set(key='COMPUTE_RMS_SIGCLIP_THRES', value=5, source=__NAME__,
           desc='define the number of sigma to clip based on the rms away '
                'from the model (sigma clips science data)')

# scale of the high-passing of the CCF in rough-ccf
# should be a few stellar FWHM. Expressed in km/s
params.set(key='COMPUTE_CCF_HP_SCALE', value=30, source=__NAME__,
           desc='scale of the high-passing of the CCF in rough-ccf '
                'should be a few stellar FWHM. Expressed in km/s')

# define the name of the sample wave grid file (saved to the calib dir)
params.set(key='SAMPLE_WAVE_GRID_FILE', value=None, not_none=True,
           source=__NAME__,
           desc=('define the name of the sample wave grid file '
                 '(saved to the calib dir)'))

# Dictionary of table name for the file used in the projection against the
#     derivative. Key is to output column name that will propagate into the
#     final RDB table and the value is the filename of the table. The table
#     must follow a number of characteristics explained on the LBL website.
params.set(key='RESPROJ_TABLES', value=None, source=__NAME__,
           desc=('Dictionary of table name for the file used in the '
                 'projection against the derivative. Key is to output '
                 'column name that will propagate into the finale RDB '
                 'table and the value is the filename of the table. The '
                 'table must follow a number of characteristics explained '
                 'on the LBL  website.'), not_none=True)

# Rotational velocity parameters, should be a list of two values, one being
#     the epsilon and the other one being the vsini in km/s as defined in the
#     PyAstronomy.pyasl.rotBroad function
params.set(key='ROTBROAD', value=None, source=__NAME__,
           desc=('Rotational velocity parameters, should be a list of two '
                 'values, one being the epsilon and the other one being the '
                 'vsini in km/s as defined in the PyAstronomy.pyasl.rotBroad'
                 'function.'), not_none=True)

# =============================================================================
# Define compil parameters
# =============================================================================
# define the suffix to give the rdb files
params.set(key='RDB_SUFFIX', value='', source=__NAME__,
           desc='The suffix to give the rdb files',
           arg='--suffix_rdb', dtype=str)

# define the plot order for the compute rv model plot
params.set('COMPUTE_MODEL_PLOT_ORDERS', value=None, source=__NAME__,
           desc='The plot order for the compute rv model plot'
                'this can be an integer of a list of integers')

# define the compil minimum wavelength allowed for lines [nm]
params.set('COMPIL_WAVE_MIN', None, source=__NAME__,
           desc='The compil minimum wavelength allowed for lines [nm]',
           not_none=True)

# define the compil maximum wavelength allowed for lines [nm]
params.set('COMPIL_WAVE_MAX', None, source=__NAME__,
           desc='The compil maximum wavelength allowed for lines [nm]',
           not_none=True)

# define the maximum pixel width allowed for lines [pixels]
params.set('COMPIL_MAX_PIXEL_WIDTH', None, source=__NAME__,
           desc='The maximum pixel width allowed for lines [pixels]',
           not_none=True)

# Max likelihood of correlation with BERV to use line
params.set('COMPIL_CUT_PEARSONR', -1, source=__NAME__,
           desc='Max likelihood of correlation with BERV to use line')

# define the CCF e-width to use for FP files
params.set('COMPIL_FP_EWID', None, source=__NAME__,
           desc='define the CCF e-width to use for FP files',
           not_none=True)

# define whether to add the magic "binned wavelength" bands rv
params.set('COMPIL_ADD_UNIFORM_WAVEBIN', False, source=__NAME__,
           desc='define whether to add the magic "binned wavelength" bands rv')

# define the number of bins used in the magic "binned wavelength" bands
params.set('COMPIL_NUM_UNIFORM_WAVEBIN', 15, source=__NAME__,
           desc='define the number of bins used in the magic "binned '
                'wavelength" bands')

# define the first band (from get_binned_parameters) to plot (band1)
params.set('COMPILE_BINNED_BAND1', None, source=__NAME__,
           desc='The first band (from get_binned_parameters) to plot (band1)',
           not_none=True)

# define the second band (from get_binned_parameters) to plot (band2)
#    this is used for colour   band2 - band3
params.set('COMPILE_BINNED_BAND2', None, source=__NAME__,
           desc='The second band (from get_binned_parameters) to plot (band2) '
                'this is used for colour (band2 - band3)',
           not_none=True)

# define the third band (from get_binned_parameters) to plot (band3)
#    this is used for colour   band2 - band3
params.set('COMPILE_BINNED_BAND3', None, source=__NAME__,
           desc='The third band (from get_binned_parameters) to plot (band3) '
                'this is used for colour (band2 - band3)',
           not_none=True)

# define the reference wavelength used in the slope fitting  in nm
params.set('COMPIL_SLOPE_REF_WAVE', None, source=__NAME__,
           desc='define the reference wavelength used in the slope '
                'fitting in nm',
           not_none=True)

# define a threshold based on the fraction on time a line as been measured
params.set('COMPIL_FRAC_TIME_MEAS', 0.1, source=__NAME__,
           desc='a threshold based on the fraction on time a line as '
                'been measured')

# define a threshold based on the fraction on time a line as been measured
params.set('COMPIL_FORCE_SIGMA_PER_LINE', False, source=__NAME__,
           desc='Force the per-line dispersion to match uncertainties. In other'
                'words, the per-line (vrad-median(vrad))/svrad will be forced '
                'to a median value of 1 if True. This causes a degradation of '
                'performances by 5-10% for SPIRou but makes the svrad more '
                'representative of the expected dispersion in the timeseries.')

# define the FP reference string that defines that an FP observation was
#    a reference (calibration) file - should be a list of strings
params.set('FP_REF_LIST', None, source=__NAME__,
           desc='define the FP reference string that defines that an FP '
                'observation was a reference (calibration) file - should be a '
                'list of strings',
           not_none=True)

# define the FP standard string that defines that an FP observation was NOT
#    a reference file - should be a list of strings
params.set('FP_STD_LIST', None, source=__NAME__,
           desc='# define the FP standard string that defines that an FP '
                'observation was NOT a reference file - should be a list of '
                'strings',
           not_none=True)

# define readout noise per instrument (assumes ~5e- and 10 pixels)
params.set('READ_OUT_NOISE', None, source=__NAME__,
           desc='define readout noise per instrument (assumes ~5e- and '
                '10 pixels)',
           not_none=True)

# =============================================================================
# Define plot parameters
# =============================================================================
# Define whether to do any plots
params.set(key='PLOT', value=False, source=__NAME__,
           desc='Whether to do plots for the compute function',
           arg='--plot', dtype=bool)

# Define whether to do the compute ccf plot
params.set(key='PLOT_COMPUTE_CCF', value=True, source=__NAME__,
           desc='Whether to do the compute ccf plot',
           arg='--plotccf', dtype=bool)

# Define whether to do the compute sysvel plot
params.set(key='PLOT_COMPUTE_SYSVEL', value=True, source=__NAME__,
           desc='Whether to do the compute sysvel plot',
           arg='--plotsysvel', dtype=bool)

# Define whether to do the compute line plot
params.set(key='PLOT_COMPUTE_LINES', value=True, source=__NAME__,
           desc='Whether to do the compute line plot',
           arg='--plotline', dtype=bool)

# Define whether to do the compil cumulative plot
params.set(key='PLOT_COMPIL_CUMUL', value=True, source=__NAME__,
           desc='Whether to do the compil cumulative plot',
           arg='--plotcumul', dtype=bool)

# Define whether to do the compil binned plot
params.set(key='PLOT_COMPIL_BINNED', value=True, source=__NAME__,
           desc='Whether to do the compil binned plot',
           arg='--plotbinned', dtype=bool)

# Define whether to do the mask ccf plot
params.set(key='PLOT_MASK_CCF', value=True, source=__NAME__,
           desc='whether to do the mask ccf plot',
           arg='--plotmaskccf', dtype=bool)

# Define whether to do the ccf vector plot
params.set(key='PLOT_CCF_VECTOR_PLOT', value=True, source=__NAME__,
           desc='whether to do the ccf vector plot',
           arg='--plotccfvec', dtype=bool)

# Define whether to do the tellu correction plot
params.set(key='PLOT_TELLU_CORR_PLOT', value=True, source=__NAME__,
           desc='whether to do the tellu correction plot',
           arg='--plottcorr', dtype=bool)

# =============================================================================
# Define template and mask parameters
# =============================================================================
# Define the wave url for the stellar models
params.set(key='STELLAR_WAVE_URL', value=None, source=__NAME__,
           desc='the wave url for the stellar models',
           dtype=str, not_none=True)

# Define the wave file for the stellar models (using wget)
params.set(key='STELLAR_WAVE_FILE', value=None, source=__NAME__,
           desc='the wave file for the stellar models (using wget)',
           dtype=str, not_none=True)

# Define the stellar model url
params.set(key='STELLAR_MODEL_URL', value=None, source=__NAME__,
           desc='the stellar model url',
           dtype=str, not_none=True)

# Define the minimum allowed SNR in a pixel to add it to the mask
params.set(key='MASK_SNR_MIN', value=None, source=__NAME__,
           desc='the minimum allowed SNR in a pixel to add it to the mask',
           dtype=float, not_none=True)

# Define the stellar model file name (using wget, with appriopriate format
#     cards)
params.set(key='STELLAR_MODEL_FILE', value=None, source=__NAME__,
           desc='Define the stellar model file name (using wget, with '
                'appriopriate format cards)',
           dtype=str, not_none=True)

# Define the object temperature (stellar model)
params.set(key='OBJECT_TEFF', value=None, source=__NAME__,
           desc='the object temperature (stellar model)', arg='--obj_temp',
           dtype=float)

# Define the object surface gravity (log g) (stellar model)
params.set(key='OBJECT_LOGG', value=None, source=__NAME__,
           desc='the object surface gravity (log g) (stellar model)',
           arg='--obj_logg', dtype=float, not_none=True)

# Define the object Z (stellar model)
params.set(key='OBJECT_Z', value=None, source=__NAME__,
           desc='the object Z (stellar model)',
           arg='--obj_z', dtype=float, not_none=True)

# Define the object alpha (stellar model)
params.set(key='OBJECT_ALPHA', value=None, source=__NAME__,
           desc='the object alpha (stellar model)',
           arg='--obj_alpha', dtype=float, not_none=True)

# blaze smoothing size (s1d template)
params.set(key='BLAZE_SMOOTH_SIZE', value=None, source=__NAME__,
           desc='blaze smoothing size (s1d template)', dtype=float,
           not_none=True)

# blaze threshold (s1d template)
params.set(key='BLAZE_THRESHOLD', value=None, source=__NAME__,
           desc='blaze threshold (s1d template)', dtype=float, not_none=True)

# define the earliest allowed file used for template construction
params.set(key='TEMPLATE_MJDSTART', value=None, source=__NAME__,
           desc='the earliest allowed FP calibration used for template '
                'construction (None for unset)', dtype=float)

# define the latest allowed file used for template construction
params.set(key='TEMPLATE_MJDEND', value=None, source=__NAME__,
           desc='the latest allowed FP calibration used for template '
                'construction (None for unset)', dtype=float)

# define the minimum number of observations required for a template berv bin
params.set('BERVBIN_MIN_ENTRIES', value=3, source=__NAME__,
           desc='the minimum number of observations required for a '
                'template berv bin', dtype=int)

# define the berv bin size in m/s
params.set('BERVBIN_SIZE', value=3000, source=__NAME__,
           desc='define the berv bin size in m/s')

# define the wave solution polynomial type (Chebyshev or numpy)
params.set('WAVE_POLY_TYPE', value='numpy', source=__NAME__,
           desc='define the wave solution polynomial type '
                '(Chebyshev or numpy)')

# =============================================================================
# Define telluric tellu-cleaning parameters
# =============================================================================
# define whether to do the tellu-clean
params.set(key='DO_TELLUCLEAN', value=None, source=__NAME__,
           desc='whether to do the tellu-clean', not_none=True)

# define whether to use template in tellu-cleaning
params.set(key='TELLUCLEAN_USE_TEMPLATE', value=True, source=__NAME__,
           desc='whether to use template in tellu-cleaning')

# define the default model repo url
params.set(key='MODEL_REPO_URL',
           value='https://www.astro.umontreal.ca/~artigau/lbl/models',
           source=__NAME__,
           desc='define the default model repo url', not_none=True)

# define the model files
MODEL_FILES = dict()
# TODO put in per-instrument profiles
MODEL_FILES['Mdwarf Temperature Gradient Table'] = 'Mdwarf_temp_gradient.fits'
MODEL_FILES['Mdwarf Mask [HARPS]'] = 'mdwarf_harps.fits'
MODEL_FILES['Mdwarf Mask [SOPHIE]'] = 'mdwarf_harps.fits'
MODEL_FILES['Mdwarf Mask [NIRPS-HA]'] = 'mdwarf_nirps_ha.fits'
MODEL_FILES['Mdwarf Mask [NIRPS-HE]'] = 'mdwarf_nirps_he.fits'
MODEL_FILES['Mdwarf Mask [SPIROU]'] = 'mdwarf_spirou.fits'
MODEL_FILES['Tapas file'] = 'tapas_lbl.fits'

# define a dictionary of model files to be downloaded from the MODEL_REPO_URL
params.set(key='MODEL_FILES', value=MODEL_FILES, source=__NAME__,
           desc='define a dictionary of model files to be downloaded from '
                'the MODEL_REPO_URL', not_none=True)

# define the default mask filename
params.set(key='DEFAULT_MASK_FILE', value=None, source=__NAME__,
           desc='define the default mask filename', not_none=True)

# define the tapas file
params.set(key='TELLUCLEAN_TAPAS_FILE', value='tapas_lbl.fits',
           source=__NAME__, desc='define the tapas file')

# define the dv offset for tellu-cleaning in km/s
params.set(key='TELLUCLEAN_DV0', value=None, source=__NAME__, not_none=True,
           desc='the dv offset for tellu-cleaning in km/s')

# Define the lower wave limit for the absorber spectrum masks in nm
params.set(key='TELLUCLEAN_MASK_DOMAIN_LOWER', value=None, source=__NAME__,
           desc='the lower wave limit for the absorber spectrum masks in nm',
           not_none=True)

# Define the upper wave limit for the absorber spectrum masks in nm
params.set(key='TELLUCLEAN_MASK_DOMAIN_UPPER', value=None, source=__NAME__,
           desc='the upper wave limit for the absorber spectrum masks in nm',
           not_none=True)

# Define whether to force using airmass from header
params.set(key='TELLUCLEAN_FORCE_AIRMASS', value=None, source=__NAME__,
           desc='whether to force using airmass from header', not_none=True)

# Define the CCF scan range in km/s
params.set(key='TELLUCLEAN_CCF_SCAN_RANGE', value=None, source=__NAME__,
           desc='the CCF scan range in km/s', not_none=True)

# Define the maximum number of iterations for the tellu-cleaning loop
params.set(key='TELLUCLEAN_MAX_ITERATIONS', value=20, source=__NAME__,
           desc='the maximum number of iterations for the tellu-cleaning loop',
           not_none=True)

# Define the kernel width in pixels
params.set(key='TELLUCLEAN_KERNEL_WID', value=None, source=__NAME__,
           desc='the kernel width in pixels', not_none=True)

# Define the gaussian shape (2=pure gaussian, >2=boxy)
params.set(key='TELLUCLEAN_GAUSSIAN_SHAPE', value=None, source=__NAME__,
           desc='the gaussian shape (2=pure gaussian, >2=boxy', not_none=True)

# Define the wave grid lower wavelength limit in nm
params.set(key='TELLUCLEAN_WAVE_LOWER', value=None, source=__NAME__,
           desc='the wave grid lower wavelength limit in nm', not_none=True)

# Define the wave griv upper wavelength limit
params.set(key='TELLUCLEAN_WAVE_UPPER', value=None, source=__NAME__,
           desc='the wave grid upper wavelength limit in nm', not_none=True)

# Define the transmission threshold exp(-1) at which tellurics are uncorrectable
params.set(key='TELLUCLEAN_TRANSMISSION_THRESHOLD', value=-1, source=__NAME__,
           desc='the transmission threshold exp(-1) at which tellurics '
                'are uncorrectable', not_none=True)

# Define the sigma cut threshold above which pixels are removed from fit
params.set(key='TELLUCLEAN_SIGMA_THRESHOLD', value=10, source=__NAME__,
           desc='the sigma cut threshold above which pixels are removed '
                'from fit', not_none=True)

# Define whether to recenter the CCF on the first iteration
params.set(key='TELLUCLEAN_RECENTER_CCF', value=None, source=__NAME__,
           desc='whether to recenter the CCF on the first iteration',
           not_none=True)

# Define whether to recenter the CCF of others on the first iteration
params.set(key='TELLUCLEAN_RECENTER_CCF_FIT_OTHERS', value=None, source=__NAME__,
           desc='whether to recenter the CCF on the first iteration',
           not_none=True)

# Define the default water absorption to use
params.set(key='TELLUCLEAN_DEFAULT_WATER_ABSO', value=None, source=__NAME__,
           desc='the default water absorption to use', not_none=True)

# Define the lower limit on valid exponent of water absorbers
params.set(key='TELLUCLEAN_WATER_BOUNDS_LOWER', value=None, source=__NAME__,
           desc='the lower limit on valid exponent of water absorbers',
           not_none=True)

# Define the upper limit on valid exponent of water absorbers
params.set(key='TELLUCLEAN_WATER_BOUNDS_UPPER', value=None, source=__NAME__,
           desc='the upper limit on valid exponent of water absorbers',
           not_none=True)

# Define the lower limit on valid exponent of other absorbers
params.set(key='TELLUCLEAN_OTHERS_BOUNDS_LOWER', value=None, source=__NAME__,
           desc='the lower limit on valid exponent of other absorbers',
           not_none=True)

# Define the upper limit on valid exponent of other absorbers
params.set(key='TELLUCLEAN_OTHERS_BOUNDS_UPPER', value=None, source=__NAME__,
           desc='the upper limit on valid exponent of other absorbers',
           not_none=True)

# Define the default convergence limit for the telluric pre-clean
params.set(key='TELLUCLEAN_CONVERGENCE_LIMIT', value=1.0e-3, source=__NAME__,
           desc='the default convergence limit for the telluric pre-clean')

# =============================================================================
# Define other parameters
# =============================================================================
# define some storage of command line arguments used
params.set(key='COMMAND_LINE_ARGS', value=None, source=__NAME__,
           desc='storage of command line arguments used')

# =============================================================================
# Define header keys
# =============================================================================
# Wave coefficients header key
params.set(key='KW_WAVECOEFFS', value=None, source=__NAME__,
           desc='Wave coefficients header key')

# define wave num orders key in header
params.set(key='KW_WAVEORDN', value=None, source=__NAME__,
           desc='wave num orders key in header')

# define wave degree key in header
params.set(key='KW_WAVEDEGN', value=None, source=__NAME__,
           desc='wave degree key in header')

# define the key that gives the mid exposure time in MJD
params.set(key='KW_MID_EXP_TIME', value=None, source=__NAME__,
           desc='mid exposure time in MJD', not_none=True)

# define snr keyword
params.set(key='KW_SNR', value=None, source=__NAME__,
           desc='snr key in header', not_none=True)

# define the BERV keyword
params.set(key='KW_BERV', value=None, source=__NAME__,
           desc='the barycentric correction keyword', not_none=True,
           fp_flag=True)

# define the Blaze calibration file
params.set(key='KW_BLAZE_FILE', value=None, source=__NAME__,
           desc='The Blaze calibration file', not_none=True)

# define the number of iterations
params.set(key='KW_NITERATIONS', value='ITE_RV', source=__NAME__,
           desc='the number of iterations',
           comment='Num iterations to reach sigma accuracy')

# define the number of iterations
params.set(key='KW_RESET_RV', value='RESET_RV', source=__NAME__,
           desc='Num iterations larger than 10',
           comment='Probably bad RV')

# define the systemic velocity in m/s
params.set(key='KW_SYSTEMIC_VELO', value='SYSTVELO', source=__NAME__,
           desc='the systemic velocity in m/s',
           comment='systemic velocity in m/s')

# define the rms to photon noise ratio
params.set(key='KW_RMS_RATIO', value='RMSRATIO', source=__NAME__,
           desc='the rms to photon noise ratio',
           comment='RMS vs photon noise')

# define the e-width of LBL CCF
params.set(key='KW_CCF_EW', value='CCF_EW', source=__NAME__,
           desc='the e-width of LBL CCF',
           comment='e-width of LBL CCF in m/s')

# define the high-pass LBL width [km/s]
params.set(key='KW_HP_WIDTH', value='HP_WIDTH', source=__NAME__,
           desc='the high-pass LBL width [km/s]',
           comment='high-pass LBL width in km/s')

# define the LBL version
params.set(key='KW_VERSION', value='LBL_VERS', source=__NAME__,
           desc='the LBL version',
           comment='LBL code version')

# define the LBL date
params.set(key='KW_VDATE', value='LBLVDATE', source=__NAME__,
           desc='the LBL version',
           comment='LBL version date')

# define the process date
params.set(key='KW_PDATE', value='LBLPDATE', source=__NAME__,
           desc='the LBL processed date',
           comment='LBL processed date')

# define the lbl instrument was used
params.set(key='KW_INSTRUMENT', value='LBLINSTR', source=__NAME__,
           desc='the LBL processed date',
           comment='LBL instrument used')

# define the start time of the observation key
params.set(key='KW_MJDATE', value=None, source=__NAME__, not_none=False,
           desc='the start time of the observation')

# define the exposure time of the observation
params.set(key='KW_EXPTIME', value=None, source=__NAME__, not_none=False,
           desc='the exposure time of the observation')

# define the airmass of the observation
params.set(key='KW_AIRMASS', value=None, source=__NAME__, not_none=False,
           desc='the airmass of the observation', fp_flag=True)

# define the human date of the observation
params.set(key='KW_DATE', value=None, source=__NAME__, not_none=False,
           desc='the human date of the observation')

# define the tau_h20 of the observation
params.set(key='KW_TAU_H2O', value=None, source=__NAME__, not_none=False,
           desc='the tau_h20 of the observation', fp_flag=True)

# define the tau_other of the observation
params.set(key='KW_TAU_OTHERS', value=None, source=__NAME__, not_none=False,
           desc='the tau_other of the observation', fp_flag=True)

# define the DPRTYPE of the observation
params.set(key='KW_DPRTYPE', value=None, source=__NAME__, not_none=False,
           desc='the DPRTYPE of the observation')

# define the output type of the file
params.set(key='KW_OUTPUT', value=None, source=__NAME__, not_none=False,
           desc='the output type of the file')

# define the drs object name
params.set(key='KW_DRSOBJN', value=None, source=__NAME__, not_none=False,
           desc='the drs object name')

# define the FIBER of the observation
params.set(key='KW_FIBER', value=None, source=__NAME__, not_none=False,
           desc='define the FIBER of the observation')

# define the observation time (mjd) of the wave solution
params.set(key='KW_WAVETIME', value=None, source=__NAME__, not_none=False,
           desc='the observation time (mjd) of the wave solution')

# define the filename of the wave solution
params.set(key='KW_WAVEFILE', value=None, source=__NAME__, not_none=False,
           desc='the filename of the wave solution')

# define the telluric TELLUCLEAN velocity of water absorbers
params.set(key='KW_TLPDVH2O', value=None, source=__NAME__, not_none=False,
           desc='the telluric TELLUCLEAN velocity of water absorbers',
           fp_flag=True)

# define the telluric TELLUCLEAN velocity of other absorbers
params.set(key='KW_TLPDVOTR', value=None, source=__NAME__, not_none=False,
           desc='the telluric TELLUCLEAN velocity of other absorbers',
           fp_flag=True)

# define the wave solution calibration filename
params.set(key='KW_CDBWAVE', value=None, source=__NAME__, not_none=False,
           desc='the wave solution used')

# define the original object name
params.set(key='KW_OBJNAME', value=None, source=__NAME__, not_none=False,
           desc='the original object name')

# define the rhomb 1 predefined position
params.set(key='KW_RHOMB1', value=None, source=__NAME__, not_none=False,
           desc='the rhomb 1 predefined position')

# define the rhomb 2 predefined position
params.set(key='KW_RHOMB2', value=None, source=__NAME__, not_none=False,
           desc='the rhomb 2 predefined position')

# define the calib-reference density
params.set(key='KW_CDEN_P', value=None, source=__NAME__, not_none=False,
           desc='the calib-reference density')

# define the FP Internal Temp: FPBody(deg C)
params.set(key='KW_FPI_TEMP', value=None, source=__NAME__, not_none=False,
           desc='the FP Internal Temp: FPBody(deg C)')

# define the FP External Temp: FPBody(deg C)
params.set(key='KW_FPE_TEMP', value=None, source=__NAME__, not_none=False,
           desc='the FP External Temp: FPBody(deg C)')

# define the SNR goal per pixel per frame
params.set(key='KW_SNRGOAL', value=None, source=__NAME__, not_none=False,
           desc='the SNR goal per pixel per frame')

# define the SNR in chosen order
params.set(key='KW_EXT_SNR', value=None, source=__NAME__, not_none=False,
           desc='the SNR in chosen order')

# define the barycentric julian date
params.set(key='KW_BJD', value=None, source=__NAME__, not_none=False,
           desc='The barycentric julian date', fp_flag=True)

# define the shape code dx value
params.set(key='KW_SHAPE_DX', value=None, source=__NAME__, not_none=False,
           desc='The shape code dx value')

# define the shape code dy value
params.set(key='KW_SHAPE_DY', value=None, source=__NAME__, not_none=False,
           desc='The shape code dy value')

# define the shape code A value
params.set(key='KW_SHAPE_A', value=None, source=__NAME__, not_none=False,
           desc='The shape code A value')

# define the shape code B value
params.set(key='KW_SHAPE_B', value=None, source=__NAME__, not_none=False,
           desc='The shape code B value')

# define the shape code C value
params.set(key='KW_SHAPE_C', value=None, source=__NAME__, not_none=False,
           desc='The shape code C value')

# define the shape code D value
params.set(key='KW_SHAPE_D', value=None, source=__NAME__, not_none=False,
           desc='The shape code D value')

# define the header key for FP internal temp [deg C]
params.set(key='KW_FP_INT_T', value=None, source=__NAME__, not_none=False,
           desc='the header key for FP internal temp [deg C]')

# define the header key for FP internal pressue [mbar]
params.set(key='KW_FP_INT_P', value=None, source=__NAME__, not_none=False,
           desc='the header key for FP internal pressue [mbar]')

# define the reference header key (must also be in rdb table) to
#    distinguish FP calibration files from FP simultaneous files
params.set(key='KW_REF_KEY', value=None, source=__NAME__, not_none=True,
           desc='define the reference header key (must also be in rdb table) '
                'to distinguish FP calibration files from FP simultaneous '
                'files')

# define the temperature of the object
params.set(key='KW_TEMPERATURE', value=None, source=__NAME__, not_none=True,
           desc='the temperature of the object')

# Template/model velocity from CCF
params.set(key='KW_MODELVEL', value=None, source=__NAME__, not_none=True,
           desc='Template/model velocity from CCF',
           comment='Template velo. from CCF [m/s]')

# Number of template files used in template
params.set(key='KW_NTFILES', value='LBLNTMPL', source=__NAME__, not_none=True,
           desc='Number of files used in template construction',
           comment='Number of files used in template construction')

# define the berv coverage of a template
params.set(key='KW_TEMPLATE_COVERAGE', value='LBLTCOVR', source=__NAME__,
           not_none=True, desc='define the berv coverage of a template',
           comment='Template BERV coverage in km/s')

# define the number of template berv bins
params.set(key='KW_TEMPLATE_BERVBINS', value='LBLTBRVB', source=__NAME__,
           not_none=True, desc='define the number of template berv bins',
           comment='Number of template BERV bins')

# define the instrumental drift key
params.set(key='KW_INST_DRIFT', value=None, source=__NAME__, not_none=False,
           desc='define the instrumental drift key word in m/s',
           comment='Instrumental drift in m/s')


# =============================================================================
# Header conversion dictionary
# =============================================================================
# Note these are only to be used when we need to translate from something
#   other than a fits.Header - i.e. a pandas.hd5 file where keys can be
#   anything and not 8 character strings to be used in headers
# Note sometimes we need to convert these, so we do this here by defining a
#   custom function
# -----------------------------------------------------------------------------
# Define function to convert the Instrument_Drift key
def instr_drift(okey, nkey, value) -> Tuple[Any, str]:
    try:
        value = value.split('m/s')[0].strip()
        value = float(value)
        return float(value), 'Instrumental drift in m/s'
    except Exception as _:
        emsg = 'Cannot translate value {0} to {1}. Value={2}'
        eargs = [okey, nkey, value]
        raise base_classes.LblException(emsg.format(*eargs))


# Define function to convert JD to MJD
def jd_to_mjd(okey, nkey, value) -> Tuple[Any, str]:
    try:
        # calcualte the mjd value
        mjdvalue = base.AstropyTime(value, format='jd').mjd
        comment = 'MJD from {0}'.format(okey)
        # push back into header dictionary
        return mjdvalue, comment
    except Exception as _:
        emsg = 'Cannot translate value {0} to {1}. Value={2}'
        eargs = [okey, nkey, value]
        raise base_classes.LblException(emsg.format(*eargs))


# Define function to convert MJD to human readable
def jd_to_human(okey, nkey, value) -> Tuple[Any, str]:
    try:
        # calcualte the mjd value
        humanvalue = base.AstropyTime(value, format='jd').iso
        comment = 'Human readable date from {0}'.format(okey)
        # push back into header dictionary
        return humanvalue, comment
    except Exception as _:
        emsg = 'Cannot translate value {0} to {1}. Value={2}'
        eargs = [okey, nkey, value]
        raise base_classes.LblException(emsg.format(*eargs))


# Define function to sky DPRTYPE
def set_dprtype(okey, nkey, value) -> Tuple[Any, str]:
    _ = okey, nkey, value
    return 'OBJ_SKY', 'DPRTYPE (set to OBJ_SKY manually)'


# Define keys to copy (currently for all instruments)
htrans = base_classes.HeaderTranslate()
htrans.add('Instrument_Drift', 'INSDRIFT', func=instr_drift)
htrans.add('JD_UTC_FLUXWEIGHTED_FRD', 'MJDFWFRD', func=jd_to_mjd)
htrans.add('JD_UTC_FLUXWEIGHTED_PC', 'MJDFWPC', func=jd_to_mjd)
htrans.add('JD_UTC_MIDPOINT', 'MJDMID', func=jd_to_mjd)
htrans.add('JD_UTC_START', 'MJSTART', func=jd_to_mjd)
htrans.add('BERV_FLUXWEIGHTED_FRD', 'BERV')
htrans.add('JD_UTC_FLUXWEIGHTED_FRD', 'DATE-OBS', func=jd_to_human)
htrans.add('Instrument_Drift', 'DPRTYPE', func=set_dprtype)
htrans.add('MAROONX TELESCOPE AIRMASS', 'AIRMASS')
htrans.add('BERV_SIMBAD_TARGET', 'OBJNAME')


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

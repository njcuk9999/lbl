#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOPHIE instrument class here: instrument specific settings

Created on 2023-06-21

@author: p. larue
"""
import glob
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from lbl.core import astro
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default
from lbl.instruments import essp


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.neid.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get time from base
Time = base.AstropyTime
# get classes
Instrument = default.Instrument
LblException = base_classes.LblException
log = io.log


# =============================================================================
# Define SOPHIE class
# =============================================================================
class Neid(Instrument):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict):
        # call to super function
        super().__init__('NEID')
        # extra parameters (specific to instrument)
        self.default_template_name = 'LBL_Template_{0}_neid.fits'
        self.default_mask_name = 'LBL_Mask_{obj}_{mtype}_neid.fits'
        self.default_sample_wave_name = 'sample_wave_grid_neid.fits'
        # define wave limits in nm
        self.wavemin = 379.66
        self.wavemax = 822.36
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)

    # -------------------------------------------------------------------------
    # SPIROU SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    def param_override(self):
        """
        Parameter override for SOPHIE parameters
        (update default params)

        :return: None - updates self.params
        """
        emsg = 'NEID not implemented yet'
        raise NotImplemented(emsg)


# =============================================================================
# Define NEID ESSP class
# =============================================================================
class Neid_ESSP(essp.ESSP):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict, name: str = None):
        # get the name
        if name is None:
            name = 'NEID_ESSP'
        # call to super function
        super().__init__(params, args, name)
        # extra parameters (specific to instrument)
        self.default_template_name = 'LBL_Template_{0}_neid_essp.fits'
        self.default_mask_name = 'LBL_Mask_{obj}_{mtype}_neid_essp.fits'
        self.default_sample_wave_name = 'sample_wave_grid_neid_essp.fits'
        # define wave limits in nm
        self.wavemin = 387.515
        self.wavemax = 691.128
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)

    def param_override(self):
        """
        Parameter override for NEID_ESSP parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.Neid_ESSP.param_override()'
        # first run the inherited method
        super().param_override()
        # ---------------------------------------------------------------------
        # set parameters to update
        # ---------------------------------------------------------------------
        # set parameters to update
        self.param_set('INSTRUMENT', 'EXPRES', source=func_name)
        # add instrument earth location
        #    (for use in astropy.coordinates.EarthLocation)
        self.param_set('EARTH_LOCATION', 'lowell')
        # define the default science input files
        self.param_set('INPUT_FILE', '*.fits', source=func_name)
        # The input science data are blaze corrected
        self.param_set('BLAZE_CORRECTED', False, source=func_name)
        # define the mask table format
        self.param_set('REF_TABLE_FMT', 'csv', source=func_name)
        # define the mask type
        self.param_set('SCIENCE_MASK_TYPE', 'full', source=func_name)
        self.param_set('FP_MASK_TYPE', 'neg', source=func_name)
        self.param_set('LFC_MASK_TYPE', 'neg', source=func_name)
        # define the default mask url and filename
        self.param_set('DEFAULT_MASK_FILE', source=func_name,
                        value='mdwarf_harps.fits')
        # define the High pass width in km/s
        self.param_set('HP_WIDTH', 500, source=func_name)
        # approximate mean resolution in lambda/dlambda
        self.param_set('APPROX_RESOLUTION', 150000, source=func_name)
        # define the SNR cut off threshold
        # Question: HARPS value?
        self.param_set('SNR_THRESHOLD', 10, source=func_name)
        # define which bands to use for the clean CCF (see astro.ccf_regions)
        self.param_set('CCF_CLEAN_BANDS', ['r'], source=func_name)
        # define the plot order for the compute rv model plot
        self.param_set('COMPUTE_MODEL_PLOT_ORDERS', [50], source=func_name)
        # define the compil minimum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MIN', self.wavemin, source=func_name)
        # define the compil maximum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MAX', self.wavemax, source=func_name)
        # define the maximum pixel width allowed for lines [pixels]
        self.param_set('COMPIL_MAX_PIXEL_WIDTH', 50, source=func_name)
        # define min likelihood of correlation with BERV
        self.param_set('COMPIL_CUT_PEARSONR', -1, source=func_name)
        # define the CCF e-width to use for FP files
        # Question: HARPS value?
        self.param_set('COMPIL_FP_EWID', 5.0, source=func_name)
        # define whether to add the magic "binned wavelength" bands rv
        self.param_set('COMPIL_ADD_UNIFORM_WAVEBIN', True)
        # define the number of bins used in the magic "binned wavelength" bands
        self.param_set('COMPIL_NUM_UNIFORM_WAVEBIN', 15)
        # define the first band (from get_binned_parameters) to plot (band1)
        self.param_set('COMPILE_BINNED_BAND1', 'r', source=func_name)
        # define the second band (from get_binned_parameters) to plot (band2)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND2', 'g', source=func_name)
        # define the third band (from get_binned_parameters) to plot (band3)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND3', 'r', source=func_name)
        # define the reference wavelength used in the slope fitting in nm
        self.param_set('COMPIL_SLOPE_REF_WAVE', 550, source=func_name)
        # define the name of the sample wave grid file (saved to the calib dir)
        self.param_set('SAMPLE_WAVE_GRID_FILE', self.default_sample_wave_name,
                       source=func_name)
        # define the FP reference string that defines that an FP observation was
        #    a reference (calibration) file - should be a list of strings
        # Question: Check DRP TYPE for STAR,FP file
        # TODO verify DPR TYPE in SOPHIE headers for FPs
        self.param_set('FP_REF_LIST', ['STAR,WAVE,FP'], source=func_name)
        # define the FP standard string that defines that an FP observation
        #    was NOT a reference file - should be a list of strings
        # Question: Check DRP TYPE for STAR,FP file
        # TODO verify DPR TYPE in SOPHIE headers for STAR+FPs
        self.param_set('FP_STD_LIST', ['STAR,WAVE,FP'], source=func_name)
        # define readout noise per instrument (assumes ~5e- and 10 pixels)
        self.param_set('READ_OUT_NOISE', 15, source=func_name)
        # Define the wave url for the stellar models
        self.param_set('STELLAR_WAVE_URL', source=func_name,
                        value='ftp://phoenix.astro.physik.uni-goettingen.de/'
                              'HiResFITS/')
        # Define the wave file for the stellar models (using wget)
        self.param_set('STELLAR_WAVE_FILE', source=func_name,
                        value='WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        # Define the stellar model url
        self.param_set('STELLAR_MODEL_URL', source=func_name,
                        value='ftp://phoenix.astro.physik.uni-goettingen.de/'
                              'HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
                              '{ZSTR}{ASTR}/')
        # Define the minimum allowed SNR in a pixel to add it to the mask
        self.param_set('MASK_SNR_MIN', value=5, source=func_name)
        # Define the stellar model file name (using wget, with appropriate
        #     format  cards)
        self.param_set('STELLAR_MODEL_FILE', source=func_name,
                        value='lte{TEFF}-{LOGG}-{ZVALUE}{ASTR}'
                              '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
        # Define the object surface gravity (log g) (stellar model)
        self.param_set('OBJECT_LOGG', value=4.5, source=func_name)
        # Define the object Z (stellar model)
        self.param_set('OBJECT_Z', value=0.0, source=func_name)
        # Define the object alpha (stellar model)
        self.param_set('OBJECT_ALPHA', value=0.0, source=func_name)
        # blaze smoothing size (s1d template)
        self.param_set('BLAZE_SMOOTH_SIZE', value=20, source=func_name)
        # blaze threshold (s1d template)
        self.param_set('BLAZE_THRESHOLD', value=0.2, source=func_name)
        # define the size of the berv bins in m/s
        self.param_set('BERVBIN_SIZE', value=3000)
        # ---------------------------------------------------------------------
        # define whether to do the tellu-clean
        self.param_set('DO_TELLUCLEAN', value=True, source=func_name)
        # define the dv offset for tellu-cleaning in km/s
        self.param_set('TELLUCLEAN_DV0', value=0, source=func_name)
        # Define the lower wave limit for the absorber spectrum masks in nm
        self.param_set('TELLUCLEAN_MASK_DOMAIN_LOWER', value=500,
                        source=func_name)
        # Define the upper wave limit for the absorber spectrum masks in nm
        self.param_set('TELLUCLEAN_MASK_DOMAIN_UPPER', value=700,
                        source=func_name)
        # Define whether to force using airmass from header
        self.param_set('TELLUCLEAN_FORCE_AIRMASS', value=False,
                        source=func_name)
        # Define the CCF scan range in km/s
        self.param_set('TELLUCLEAN_CCF_SCAN_RANGE', value=50,
                        source=func_name)
        # Define the maximum number of iterations for the tellu-cleaning loop
        self.param_set('TELLUCLEAN_MAX_ITERATIONS', value=20, source=func_name)
        # Define the kernel width in pixels
        self.param_set('TELLUCLEAN_KERNEL_WID', value=1.4, source=func_name)
        # Define the gaussian shape (2=pure gaussian, >2=boxy)
        self.param_set('TELLUCLEAN_GAUSSIAN_SHAPE', value=2.2,
                        source=func_name)
        # Define the wave grid lower wavelength limit in nm
        self.param_set('TELLUCLEAN_WAVE_LOWER', value=350, source=func_name)
        # Define the wave griv upper wavelength limit
        self.param_set('TELLUCLEAN_WAVE_UPPER', value=750, source=func_name)
        # Define the transmission threshold exp(-1) at which tellurics are
        #     uncorrectable
        self.param_set('TELLUCLEAN_TRANSMISSION_THRESHOLD', value=-1,
                        source=func_name)
        # Define the sigma cut threshold above which pixels are removed from fit
        self.param_set('TELLUCLEAN_SIGMA_THRESHOLD', value=10,
                        source=func_name)
        # Define whether to recenter the CCF on the first iteration
        self.param_set('TELLUCLEAN_RECENTER_CCF', value=False,
                        source=func_name)
        # Define whether to recenter the CCF of others on the first iteration
        self.param_set('TELLUCLEAN_RECENTER_CCF_FIT_OTHERS', value=False,
                        source=func_name)
        # Define the default water absorption to use
        self.param_set('TELLUCLEAN_DEFAULT_WATER_ABSO', value=0.5,
                        source=func_name)
        # Define the lower limit on valid exponent of water absorbers
        self.param_set('TELLUCLEAN_WATER_BOUNDS_LOWER', value=0.01,
                        source=func_name)
        # Define the upper limit on valid exponent of water absorbers
        self.param_set('TELLUCLEAN_WATER_BOUNDS_UPPER', value=15,
                        source=func_name)
        # Define the lower limit on valid exponent of other absorbers
        self.param_set('TELLUCLEAN_OTHERS_BOUNDS_LOWER', value=0.05,
                        source=func_name)
        # Define the upper limit on valid exponent of other absorbers
        self.param_set('TELLUCLEAN_OTHERS_BOUNDS_UPPER', value=15,
                        source=func_name)
        # ---------------------------------------------------------------------
        # Parameters for the template construction
        # ---------------------------------------------------------------------
        # max number of bins for the median of the template. Avoids handling
        # too many spectra at once.
        self.param_set('TEMPLATE_MEDBINMAX', 19, source=func_name)
        # maximum RMS between the template and the median of the template
        # to accept the median of the template as a good template. If above
        # we iterate once more. Expressed in m/s
        self.param_set('MAX_CONVERGENCE_TEMPLATE_RV', 100, source=func_name)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

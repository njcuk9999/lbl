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

from lbl.core import astro
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.sophie.py'
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
class Sophie(Instrument):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict):
        # call to super function
        super().__init__('SOPHIE')
        # extra parameters (specific to instrument)
        self.default_template_name = 'Template_{0}_SOPHIE.fits'
        # define wave limits in nm
        self.wavemin = 387.2
        self.wavemax = 694.3
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
        # set function name
        func_name = __NAME__ + '.Sophie.override()'
        # set parameters to update
        self.param_set('INSTRUMENT', 'SOPHIE', source=func_name)
        # add instrument earth location
        #    (for use in astropy.coordinates.EarthLocation)
        self.param_set('EARTH_LOCATION', 'OHP')
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
        # define the SNR cut off threshold
        # Question: HARPS value?
        self.param_set('SNR_THRESHOLD', 10, source=func_name)
        # define which bands to use for the clean CCF (see astro.ccf_regions)
        self.param_set('CCF_CLEAN_BANDS', ['r'], source=func_name)
        # define the plot order for the compute rv model plot
        self.param_set('COMPUTE_MODEL_PLOT_ORDERS', [36], source=func_name)
        # define the compil minimum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MIN', 387.2, source=func_name)
        # define the compil maximum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MAX', 694.3, source=func_name)
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
        self.param_set('SAMPLE_WAVE_GRID_FILE',
                        'sample_wave_grid_sophie.fits', source=func_name)
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

        # ---------------------------------------------------------------------
        # Header keywords
        # ---------------------------------------------------------------------
        # define wave coeff key in header
        self.param_set('KW_WAVECOEFFS', 'HIERARCH OHP DRS CAL TH COEFF LL{0}',
                        source=func_name)
        # define wave num orders key in header
        self.param_set('KW_WAVEORDN', 'HIERARCH OHP DRS CAL TH ORDER NBR',
                        source=func_name)
        # define wave degree key in header
        self.param_set('KW_WAVEDEGN', 'HIERARCH OHP DRS CAL TH DEG LL',
                        source=func_name)
        # define the key that gives the mid exposure time in MJD
        self.param_set('KW_MID_EXP_TIME', 'HIERARCH OHP DRS BJD',
                        source=func_name)
        # define the start time of the observation
        self.param_set('KW_MJDATE', 'HIERARCH OHP DRS BJD', source=func_name)
        # define snr keyword
        self.param_set('KW_SNR', 'HIERARCH OHP DRS SPE EXT SN36',
                        source=func_name)
        # define berv keyword
        self.param_set('KW_BERV', 'HIERARCH OHP DRS BERV', source=func_name)
        # define the Blaze calibration file
        self.param_set('KW_BLAZE_FILE', 'HIERARCH OHP DRS BLAZE FILE',
                        source=func_name)
        # define the exposure time of the observation
        self.param_set('KW_EXPTIME', 'HIERARCH OHP CCD DIT',
                        source=func_name)
        # define the airmass of the observation
        self.param_set('KW_AIRMASS', 'HIERARCH OHP TEL AIRM START',
                        source=func_name)
        # define the human date of the observation
        self.param_set('KW_DATE', 'DATE', source=func_name)
        # define the tau_h20 of the observation
        self.param_set('KW_TAU_H2O', 'TLPEH2O', source=func_name)
        # define the tau_other of the observation
        self.param_set('KW_TAU_OTHERS', 'TLPEOTR', source=func_name)
        # define the DPRTYPE of the observation
        self.param_set('KW_DPRTYPE', 'HIERARCH OHP DPR TYPE',
                        source=func_name)
        # define the filename of the wave solution
        self.param_set('KW_WAVEFILE', 'HIERARCH OHP DRS CAL TH FILE',
                        source=func_name)
        # define the original object name
        self.param_set('KW_OBJNAME', 'HIERARCH OHP TARG NAME',
                        source=func_name)
        # define the SNR goal per pixel per frame (can not exist - will be
        #   set to zero)
        self.param_set('KW_SNRGOAL', 'SNRGOAL', source=func_name)
        # define the SNR in chosen order
        self.param_set('KW_EXT_SNR', 'HIERARCH OHP DRS CAL EXT SN36',
                        source=func_name)
        # define the barycentric julian date
        self.param_set('KW_BJD', 'HIERARCH OHP DRS BJD', source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.param_set('KW_REF_KEY', 'HIERARCH OHP DPR TYPE', source=func_name)
        # velocity of template from CCF
        self.param_set('KW_MODELVEL', 'MODELVEL', source=func_name)
        # the temperature of the object
        # TODO: how do we get the temperature for HARPS? / SOPHIE ?
        self.param_set('KW_TEMPERATURE', None, source=func_name)

    # -------------------------------------------------------------------------
    # SOPHIE SPECIFIC METHODS
    # -------------------------------------------------------------------------
    def load_header(self, filename: str, kind: str = 'fits file',
                    extnum: int = 0, extname: str = None) -> io.LBLHeader:
        """
        Load a header into a dictionary (may not be a fits file)
        We must push this to a dictinoary as not all instrument confirm to
        a fits header

        :param filename: str, the filename to load
        :param kind: str, the kind of file we are loading
        :param extnum: int, the extension number to load
        :param extname: str, the extension name to load
        :return:
        """
        # get header
        hdr = io.load_header(filename, kind, extnum, extname)
        # return the LBL Header class
        return io.LBLHeader.from_fits(hdr, filename)

    def mask_file(self, model_directory: str, mask_directory: str,
                  required: bool = True) -> str:
        """
        Make the absolute path for the mask file

        :param model_directory: str, the directory the model is located at
        :param mask_directory: str, the directory the mask should be copied to
        :param required: bool, if True checks that file exists on disk

        :return: absolute path to mask file
        """
        # copy the default mask file to the mask directory
        self.copy_default_mask(model_directory, mask_directory,
                               self.params['DEFAULT_MASK_FILE'])
        # get data type
        data_type = self.params['DATA_TYPE']
        # get type of mask
        mask_type = self.params['{0}_MASK_TYPE'.format(data_type)]
        # deal with no object
        if self.params['MASK_FILE'] not in [None, 'None', '']:
            # define base name
            basename = self.params['MASK_FILE']
            # if basename is full path use this
            if os.path.exists(basename):
                abspath = str(basename)
            else:
                # get absolute path
                abspath = os.path.join(mask_directory, basename)
        elif self.params['OBJECT_TEMPLATE'] is None:
            raise LblException('OBJECT_TEMPLATE name must be defined')
        else:
            objname = self.params['OBJECT_TEMPLATE']
            # define base name
            basename = '{0}_{1}.fits'.format(objname, mask_type)
            # get absolute path
            abspath = os.path.join(mask_directory, basename)
        # check that this file exists
        if required:
            io.check_file_exists(abspath, 'mask')
        # return absolute path
        return abspath

    def template_file(self, directory: str, required: bool = True) -> str:
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at
        :param required: bool, if True checks that file exists on disk

        :return: absolute path to template file
        """
        # deal with no object template
        self._set_object_template()
        # set template name
        objname = self.params['OBJECT_TEMPLATE']
        # get template file
        if self.params['TEMPLATE_FILE'] is None:
            basename = self.default_template_name.format(objname)
        else:
            basename = self.params['TEMPLATE_FILE']
        # get absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        if required:
            io.check_file_exists(abspath, 'template')
        # return absolute path
        return abspath

    def blaze_file(self, directory: str) -> Union[str, None]:
        """
        Make the absolute path for the blaze file if set in params

        :param directory: str, the directory the file is located at

        :return: absolute path to blaze file or None (if not set)
        """
        if self.params['BLAZE_FILE'] is None:
            return None
        # set base name
        basename = self.params['BLAZE_FILE']
        # get absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        io.check_file_exists(abspath, 'blaze')
        # return absolute path
        return abspath

    def load_blaze(self, filename: str, science_file: Optional[str] = None,
                   normalize: bool = True) -> Union[np.ndarray, None]:
        """
        Load a blaze file

        :param filename: str, absolute path to filename
        :param science_file: str, a science file (to load the wave solution
                             from) we expect this science file wave solution
                             to be the wave solution required for the blaze
        :param normalize: bool, if True normalized the blaze per order

        :return: data (np.ndarray) or None
        """
        # deal with already flagged as corrected
        if self.params['BLAZE_CORRECTED']:
            return None
        # if we have a file defined use it
        if filename is not None:
            blaze = io.load_fits(filename, kind='blaze fits file')
            # deal with normalizing per order
            if normalize:
                # get the blaze parameters (may be instrument specific)
                nth_deg, bdomain = self.norm_blaze_params()
                # require the wave grid
                wavegrid = self.get_wave_solution(science_file)
                # normalizse the blaze
                blaze = mp.smart_blaze_norm(wavegrid, blaze, nth_deg, bdomain)
            # return blaze
            return blaze
        else:
            return None

    def get_mask_systemic_vel(self, mask_file: str) -> float:
        """
        Get the systemic velocity in m/s of the mask

        :param mask_file: the absolute path to the mask file

        :return: float, systemic velocity in m/s
        """
        # get systemic velocity key
        sysvelkey = self.params['KW_SYSTEMIC_VELO']
        # load the mask header
        mask_hdr = self.load_header(mask_file, kind='mask fits file')
        # get info on template systvel for splining correctly
        systemic_vel = mask_hdr.get_hkey(sysvelkey, dtype=float)
        # return systemic velocity in m/s
        return systemic_vel

    def science_files(self, directory: str) -> np.ndarray:
        """
        List the absolute paths of all science files

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            objname = self.params['OBJECT_SCIENCE']
        # deal the input file string
        if self.params['INPUT_FILE'] is None:
            raise LblException('INPUT_FILE must be defined')
        # check that the object sub-directory exists
        abspath = io.make_dir(directory, objname, 'Science object')
        # set up basename
        basename = os.path.basename(self.params['INPUT_FILE'])
        # add to abspath
        abspath = os.path.join(abspath, basename)
        # look for files
        files = glob.glob(abspath)
        # deal with no files found
        if len(files) == 0:
            emsg = 'No science objects found for {0}. Search string={1}'
            eargs = [objname, abspath]
            raise LblException(emsg.format(*eargs))
        else:
            # sort files
            files = np.sort(files)
            # return numpy array of files
            return files

    def sort_science_files(self, science_files: List[str]) -> List[str]:
        """
        Sort science files (instrument specific)

        :param science_files: list of strings - list of science files

        :return: list of strings - sorted list of science files
        """
        times = []
        # loop around science files
        for science_file in science_files:
            # load header
            sci_hdr = self.load_header(science_file)
            # get mid exposure time
            mid_exp_time = sci_hdr.get_hkey(self.params['KW_MID_EXP_TIME'],
                                            science_file, dtype=float)
            # get time
            times.append(mid_exp_time)
        # get sort mask
        sortmask = np.argsort(times)
        # apply sort mask
        science_files = np.array(science_files)[sortmask]
        # return sorted files
        return list(science_files)

    def load_blaze_from_science(self, science_file: str,
                                sci_image: np.ndarray,
                                sci_hdr: io.LBLHeader,
                                calib_directory: str,
                                normalize: bool = True
                                ) -> Tuple[np.ndarray, bool]:
        """
        Load the blaze file using a science file header

        :param science_file: str, the science file
        :param sci_image: np.array - the science image (if we don't have a
                          blaze, we need this for the shape of the blaze)
        :param sci_hdr: io.LBLHeader - the science file header
        :param calib_directory: str, the directory containing calibration files
                                (i.e. containing the blaze files)
        :param normalize: bool, if True normalized the blaze per order

        :return: the blaze and a flag whether blaze is set to ones (science
                 image already blaze corrected)
        """
        # deal with blaze already corrected
        if self.params['BLAZE_CORRECTED']:
            # blaze corrected
            return np.ones_like(sci_image), True
        # get blaze file from science header
        blaze_file = sci_hdr.get_hkey(self.params['KW_BLAZE_FILE'])
        # construct absolute path
        abspath = os.path.join(calib_directory, blaze_file)
        # check that this file exists
        io.check_file_exists(abspath, 'blaze')
        # read blaze file (data and header)
        blaze = io.load_fits(abspath, kind='blaze fits file')
        # deal with normalizing per order
        if normalize:
            # get the blaze parameters (may be instrument specific)
            nth_deg, bdomain = self.norm_blaze_params()
            # require the wave grid
            wavegrid = self.get_wave_solution(science_file, sci_image, sci_hdr)
            # normalizse the blaze
            blaze = mp.smart_blaze_norm(wavegrid, blaze, nth_deg, bdomain)
        # return blaze
        return blaze, False

    def get_wave_solution(self, science_filename: Optional[str] = None,
                          data: Optional[np.ndarray] = None,
                          header: Optional[io.LBLHeader] = None
                          ) -> np.ndarray:
        """
        Get a wave solution from a file (for HARPS this is from the header)
        :param science_filename: str, the absolute path to the file - for
                                 spirou this is a file with the wave solution
                                 in the header
        :param header: io.LBLHeader, this is the header to use (if not given
                       requires filename to be set to load header)
        :param data: np.ndarray, this must be set along with header (if not
                     give we require filename to be set to load data)

        :return: np.ndarray, the wave map. Shape = (num orders x num pixels)
        """
        # get header keys
        kw_wavecoeffs = self.params['KW_WAVECOEFFS']
        kw_waveordn = self.params['KW_WAVEORDN']
        kw_wavedegn = self.params['KW_WAVEDEGN']
        # ---------------------------------------------------------------------
        # get header
        if header is None or data is None:
            sci_data = io.load_fits(science_filename, 'wave fits file')
            sci_hdr = self.load_header(science_filename, 'wave fits file')
        else:
            sci_data, sci_hdr = data, header
        # ---------------------------------------------------------------------
        # get the data shape
        nby, nbx = sci_data.shape
        # get xpix
        xpix = np.arange(nbx)
        # ---------------------------------------------------------------------
        # get wave order from header
        waveordn = sci_hdr.get_hkey(kw_waveordn, science_filename, dtype=int)
        wavedegn = sci_hdr.get_hkey(kw_wavedegn, science_filename, dtype=int)
        # get the wave 2d list
        wavecoeffs = sci_hdr.get_hkey_2d(key=kw_wavecoeffs,
                                         dim1=waveordn, dim2=wavedegn + 1,
                                         filename=science_filename)
        # ---------------------------------------------------------------------
        # convert to wave map
        wavemap = np.zeros([waveordn, nbx])
        for order_num in range(waveordn):
            wavemap[order_num] = np.polyval(wavecoeffs[order_num][::-1], xpix)
        # ---------------------------------------------------------------------
        # HARPS wave solution is in Angstrom - convert to nm for consistency
        wavemap = wavemap / 10.0
        # ---------------------------------------------------------------------
        # HARPS wave solution is in air - convert to vacuum
        n_index = mp.air_index(wavemap)
        wavemap = wavemap * n_index
        # ---------------------------------------------------------------------
        # return wave solution map
        return wavemap

    def load_bad_hdr_keys(self) -> Tuple[list, Any]:
        """
        Load the bad values and bad key for HARPS -- not used currently

        :return: tuple, 1. the list of bad values, 2. the bad key in
                 a file header to check against bad values
        """
        # currently no bad keys for HARPS
        # return an empty list and bad_hdr_key = None
        return [], None

    def get_berv(self, sci_hdr: io.LBLHeader) -> float:
        """
        Get the Barycenteric correction for the RV in m/s

        :param sci_hdr: io.LBLHeader, the science header

        :return:
        """
        # get BERV header key
        hdr_key = self.params['KW_BERV']
        # BERV depends on whether object is FP or not
        if 'FP' not in self.params['OBJECT_SCIENCE']:
            berv = sci_hdr.get_hkey(hdr_key, dtype=float) * 1000
        else:
            berv = 0.0
        # return the berv measurement (in m/s)
        return berv

    def populate_sci_table(self, filename: str, tdict: dict,
                           sci_hdr: io.LBLHeader, berv: float = 0.0) -> dict:
        """
        Populate the science table

        :param filename: str, the filename of the science image
        :param tdict: dictionary, the storage dictionary for science table
                      can be empty or have previous rows to append to
        :param sci_hdr: fits Header, the header of the science image
        :param berv: float, the berv value to add to storage dictionary

        :return: dict, a dictionary table of the science parameters
        """
        # these are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_DATE', 'KW_DPRTYPE', 'KW_OBJNAME', 'KW_EXT_SNR']
        # add the filename
        tdict = self.add_dict_list_value(tdict, 'FILENAME', filename)
        # loop around header keys
        for drs_key in drs_keys:
            # if key is in params we can add the value to keys
            if drs_key in self.params:
                key = self.params[drs_key]
            else:
                key = str(drs_key)
            # get value from header
            value = sci_hdr.get(key, 'NULL')
            # add to tdict
            tdict = self.add_dict_list_value(tdict, drs_key, value)
        # add the berv separately
        tdict = self.add_dict_list_value(tdict, 'BERV', berv)
        # return updated storage dictionary
        return tdict

    def rdb_columns(self) -> Tuple[np.ndarray, List[bool]]:
        """
        Define the fits header columns names to add to the RDB file
        These should be references to keys in params

        :return: tuple, 1. np.array of strings (the keys), 2. list of bools
                 the flags whether these keys should be used with FP files
        """
        # there are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_AIRMASS', 'KW_DATE', 'KW_BERV', 'KW_DPRTYPE',
                    'KW_TAU_H2O', 'KW_TAU_OTHERS', 'KW_NITERATIONS',
                    'KW_RESET_RV',
                    'KW_SYSTEMIC_VELO', 'KW_WAVEFILE', 'KW_OBJNAME',
                    'KW_EXT_SNR', 'KW_BJD', 'KW_CCF_EW']
        # convert to actual keys (not references to keys)
        keys = []
        fp_flags = []
        for drs_key in drs_keys:
            # initial set fp flag to False
            fp_flag = False
            # ignore keys that are None
            if drs_key is None:
                continue
            # if key is in params we can add the value to keys
            if drs_key in self.params:
                keys.append(self.params[drs_key])
                # we can also look for fp flag - this is either True or False
                #    if True we skip this key for FP files - default is False
                #    (i.e. not to skip)
                instance = self.params.instances[drs_key]
                if instance is not None:
                    if instance.fp_flag is not None:
                        fp_flag = instance.fp_flag
            else:
                keys.append(drs_key)
            # append fp flags
            fp_flags.append(fp_flag)
        # return a numpy array
        return np.array(keys), fp_flags

    def fix_lblrv_header(self, header: io.LBLHeader) -> io.LBLHeader:
        """
        Fix the LBL RV header

        :param header: io.LBLHeader, the LBL RV fits file header

        :return: io.LBLHeader, the updated LBL RV fits file header
        """
        # get keys from params
        kw_snrgoal = self.params['KW_SNRGOAL']
        kw_ccf_ew = self.params['KW_CCF_EW']
        # ---------------------------------------------------------------------
        # because FP files don't have an SNR goal
        if kw_snrgoal not in header:
            header[kw_snrgoal] = 0
        # ---------------------------------------------------------------------
        # deal with not having CCF_EW
        # TODO: this is template specific
        if kw_ccf_ew not in header:
            header[kw_ccf_ew] = 5.5 / mp.fwhm_value() * 1000
        # ---------------------------------------------------------------------
        # return header
        return header

    def get_rjd_value(self, header: io.LBLHeader) -> float:

        """
        Get the rjd either from KW_MID_EXP_TIME or KW_BJD
        time returned is in MJD (not JD)

        :param header: io.LBLHeader - the LBL rv header
        :return:
        """
        # get keys from params
        kw_mjdmid = self.params['KW_MID_EXP_TIME']
        kw_bjd = self.params['KW_BJD']
        # get mjdmid and bjd
        mid_exp_time = header.get_hkey(kw_mjdmid, dtype=float)
        bjd = header.get_hkey(kw_bjd, dtype=float)
        if isinstance(bjd, str) or np.isnan(bjd):
            # return RJD = MJD + 0.5
            return float(mid_exp_time) + 0.5
        else:
            # convert bjd to mjd
            bjd_mjd = Time(bjd, format='jd').mjd
            # return RJD = MJD + 0.5
            return float(bjd_mjd) + 0.5

    def get_plot_date(self, header: io.LBLHeader):
        """
        Get the matplotlib plotting date

        :param header: io.LBLHeader - the LBL rv header

        :return: float, the plot date
        """
        # get mjdate key
        kw_mjdate = self.params['KW_MJDATE']
        # get mjdate
        mjdate = header.get_hkey(kw_mjdate, dtype=float)
        # convert to plot date and take off JD?
        plot_date = Time(mjdate, format='mjd').plot_date
        # return float plot date
        return float(plot_date)

    def get_binned_parameters(self) -> Dict[str, list]:
        """
        Defines a "binning dictionary" splitting up the array by:

        Each binning dimension has [str names, start value, end value]

        - bands  (in wavelength)
            [bands / blue_end / red_end]

        - cross order regions (in pixels)
            [region_names / region_low / region_high]

        :return: dict, the binned dictionary
        """
        # ---------------------------------------------------------------------
        # define regions, and blue/red band ends
        bout = astro.choose_bands(astro.bands, self.wavemin, self.wavemax)
        bands, blue_end, red_end, use_regions = bout
        # ---------------------------------------------------------------------
        # define the region names (suffices)
        region_names = ['', '_0-2044', '_2044-4088']
        # lower x pixel bin point [pixels]
        region_low = [0, 0, 2048]
        # upper x pixel bin point [pixels]
        region_high = [4096, 2048, 4096]
        # ---------------------------------------------------------------------
        # return all this information (in a dictionary)
        binned = dict()
        binned['bands'] = list(bands)
        binned['blue_end'] = list(blue_end)
        binned['red_end'] = list(red_end)
        binned['region_names'] = list(region_names)
        binned['region_low'] = list(region_low)
        binned['region_high'] = list(region_high)
        binned['use_regions'] = list(use_regions)
        # ---------------------------------------------------------------------
        # return this binning dictionary
        return binned


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

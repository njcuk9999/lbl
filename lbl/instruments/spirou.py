#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPIRou instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from astropy.table import Table

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.spirou.py'
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
# Define Spirou class
# =============================================================================
class Spirou(Instrument):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict, name: str = None):
        # get the name
        if name is None:
            name = 'SPIROU'
        # call to super function
        super().__init__(name)
        # define wave limits in nm
        self.wavemin = 955.793
        self.wavemax = 2515.771
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)

    # -------------------------------------------------------------------------
    # INSTRUMENT SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    def param_override(self):
        """
        Parameter override for SPIRou parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.Spirou.override()'
        # set parameters to update
        self.param_set('INSTRUMENT', 'SPIROU', source=func_name)
        # add instrument earth location
        #    (for use in astropy.coordinates.EarthLocation)
        self.param_set('EARTH_LOCATION', 'Canada-France-Hawaii Telescope')
        # define the default science input files
        self.param_set('INPUT_FILE', '*.fits', source=func_name)
        # The input science data are blaze corrected
        self.param_set('BLAZE_CORRECTED', False, source=func_name)
        # define the mask table format
        self.param_set('REF_TABLE_FMT', 'csv', source=func_name)
        # define the mask type
        # Note that we use 'full' but only keep local maxima
        # this is used to improve the CCF for fainter targets
        self.param_set('SCIENCE_MASK_TYPE', 'full', source=func_name)
        self.param_set('FP_MASK_TYPE', 'neg', source=func_name)
        self.param_set('LFC_MASK_TYPE', 'neg', source=func_name)
        # define the default mask url and filename
        self.param_set('DEFAULT_MASK_FILE', source=func_name,
                        value='mdwarf_spirou.fits')
        # define the High pass width in km/s
        self.param_set('HP_WIDTH', 500, source=func_name)
        # define the SNR cut off threshold
        self.param_set('SNR_THRESHOLD', 10, source=func_name)
        # define which bands to use for the clean CCF (see astro.ccf_regions)
        self.param_set('CCF_CLEAN_BANDS', ['h', 'k'], source=func_name)
        # define the plot order for the compute rv model plot
        self.param_set('COMPUTE_MODEL_PLOT_ORDERS', [35], source=func_name)
        # define the compil minimum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MIN', 900, source=func_name)
        # define the compil maximum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MAX', 2500, source=func_name)
        # define the maximum pixel width allowed for lines [pixels]
        self.param_set('COMPIL_MAX_PIXEL_WIDTH', 50, source=func_name)
        # define min likelihood of correlation with BERV
        self.param_set('COMPIL_CUT_PEARSONR', -1, source=func_name)
        # define the CCF e-width to use for FP files
        self.param_set('COMPIL_FP_EWID', 5.0, source=func_name)
        # define whether to add the magic "binned wavelength" bands rv
        self.param_set('COMPIL_ADD_UNIFORM_WAVEBIN', True)
        # define the number of bins used in the magic "binned wavelength" bands
        self.param_set('COMPIL_NUM_UNIFORM_WAVEBIN', 25)
        # define the first band (from get_binned_parameters) to plot (band1)
        self.param_set('COMPILE_BINNED_BAND1', 'H', source=func_name)
        # define the second band (from get_binned_parameters) to plot (band2)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND2', 'J', source=func_name)
        # define the third band (from get_binned_parameters) to plot (band3)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND3', 'H', source=func_name)
        # define the reference wavelength used in the slope fitting in nm
        self.param_set('COMPIL_SLOPE_REF_WAVE', 1600, source=func_name)
        # define the name of the sample wave grid file (saved to the calib dir)
        self.param_set('SAMPLE_WAVE_GRID_FILE',
                        'sample_wave_grid_spirou.fits', source=func_name)
        # define the FP reference string that defines that an FP observation was
        #    a reference (calibration) file - should be a list of strings
        self.param_set('FP_REF_LIST', ['FP_FP'], source=func_name)
        # define the FP standard string that defines that an FP observation
        #    was NOT a reference file - should be a list of strings
        self.param_set('FP_STD_LIST', ['OBJ_FP', 'POLAR_FP'], source=func_name)
        # define readout noise per instrument (assumes ~5e- and 10 pixels)
        self.param_set('READ_OUT_NOISE', 30, source=func_name)
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
        # define the earliest allowed file used for template construction
        self.param_set('TEMPLATE_MJDSTART', value=None)
        # define the latest allowed file used for template construction
        self.param_set('TEMPLATE_MJDEND', value=None)
        # define the size of the berv bins in m/s
        self.param_set('BERVBIN_SIZE', value=3000)
        # define whether to do the tellu-clean
        self.param_set('DO_TELLUCLEAN', value=False, source=func_name)
        # define the wave solution polynomial type (Chebyshev or numpy)
        self.param_set('WAVE_POLY_TYPE', value='Chebyshev', source=func_name)
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
        self.param_set('KW_WAVECOEFFS', 'WAVE{0:04d}', source=func_name)
        # define wave num orders key in header
        self.param_set('KW_WAVEORDN', 'WAVEORDN', source=func_name)
        # define wave degree key in header
        self.param_set('KW_WAVEDEGN', 'WAVEDEGN', source=func_name)
        # define the key that gives the mid exposure time in MJD
        self.param_set('KW_MID_EXP_TIME', 'MJDMID', source=func_name)
        # define snr keyword
        self.param_set('KW_SNR', 'EXTSN035', source=func_name)
        # define berv keyword
        self.param_set('KW_BERV', 'BERV', source=func_name)
        # define the Blaze calibration file
        self.param_set('KW_BLAZE_FILE', 'CDBBLAZE', source=func_name)
        # define the start time of the observation
        self.param_set('KW_MJDATE', 'MJDATE', source=func_name)
        # define the exposure time of the observation
        self.param_set('KW_EXPTIME', 'EXPTIME', source=func_name)
        # define the airmass of the observation
        self.param_set('KW_AIRMASS', 'AIRMASS', source=func_name)
        # define the human date of the observation
        self.param_set('KW_DATE', 'DATE-OBS', source=func_name)
        # define the tau_h20 of the observation
        self.param_set('KW_TAU_H2O', 'TLPEH2O', source=func_name)
        # define the tau_other of the observation
        self.param_set('KW_TAU_OTHERS', 'TLPEOTR', source=func_name)
        # define the DPRTYPE of the observation
        self.param_set('KW_DPRTYPE', 'DPRTYPE', source=func_name)
        # define the output type of the file
        self.param_set('KW_OUTPUT', 'DRSOUTID', source=func_name)
        # define the drs object name
        self.param_set('KW_DRSOBJN', 'DRSOBJN', source=func_name)
        # define the fiber of the observation,
        self.param_set('KW_FIBER', 'FIBER', source=func_name)
        # define the observation time (mjd) of the wave solution
        self.param_set('KW_WAVETIME', 'WAVETIME', source=func_name)
        # define the filename of the wave solution
        self.param_set('KW_WAVEFILE', 'WAVEFILE', source=func_name)
        # define the telluric TELLUCLEAN velocity of water absorbers
        self.param_set('KW_TLPDVH2O', 'TLPDVH2O', source=func_name)
        # define the telluric TELLUCLEAN velocity of other absorbers
        self.param_set('KW_TLPDVOTR', 'TLPDVOTR', source=func_name)
        # define the wave solution calibration filename
        self.param_set('KW_CDBWAVE', 'CDBWAVE', source=func_name)
        # define the original object name
        self.param_set('KW_OBJNAME', 'OBJECT', source=func_name)
        # define the rhomb 1 predefined position
        self.param_set('KW_RHOMB1', 'SBRHB1_P', source=func_name)
        # define the rhomb 2 predefined position
        self.param_set('KW_RHOMB2', 'SBRHB2_P', source=func_name)
        # define the calib-reference density
        self.param_set('KW_CDEN_P', 'SBCDEN_P', source=func_name)
        # define the FP Internal Temp: FPBody(deg C)
        self.param_set('KW_FPI_TEMP', 'SBCFPI_T', source=func_name)
        # define the FP External Temp: FPBody(deg C)
        self.param_set('KW_FPE_TEMP', 'SBCFPE_T', source=func_name)
        # define the SNR goal per pixel per frame (can not exist - will be
        #   set to zero)
        self.param_set('KW_SNRGOAL', 'SNRGOAL', source=func_name)
        # define the SNR in chosen order
        self.param_set('KW_EXT_SNR', 'EXTSN035', source=func_name)
        # define the barycentric julian date
        self.param_set('KW_BJD', 'BJD', source=func_name)
        # define the shape code dx value
        self.param_set('KW_SHAPE_DX', 'SHAPE_DX', source=func_name)
        # define the shape code dy value
        self.param_set('KW_SHAPE_DY', 'SHAPE_DY', source=func_name)
        # define the shape code A value
        self.param_set('KW_SHAPE_A', 'SHAPE_A', source=func_name)
        # define the shape code B value
        self.param_set('KW_SHAPE_B', 'SHAPE_B', source=func_name)
        # define the shape code C value
        self.param_set('KW_SHAPE_C', 'SHAPE_C', source=func_name)
        # define the shape code D value
        self.param_set('KW_SHAPE_D', 'SHAPE_D', source=func_name)
        # define the header key for FP internal temp [deg C]
        self.param_set('KW_FP_INT_T', 'SBCFPI_T', source=func_name)
        # define the header key for FP internal pressue [mbar]
        self.param_set('KW_FP_INT_P', 'SBCFPB_P', source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.param_set('KW_REF_KEY', 'DPRTYPE', source=func_name)
        # the temperature of the object
        self.param_set('KW_TEMPERATURE', 'OBJTEMP', source=func_name)
        # velocity of template from CCF
        self.param_set('KW_MODELVEL', 'MODELVEL', source=func_name)

    # -------------------------------------------------------------------------
    # INSTRUMENT SPECIFIC METHODS
    # -------------------------------------------------------------------------
    def load_header(self, filename: str, kind: str = 'fits file',
                    extnum: Optional[int] = None,
                    extname: str = None) -> io.LBLHeader:
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

    def get_extname(self, kind: str) -> Optional[str]:
        """
        Get the extension name based on the kind and params['FIBER'] key

        :param kind: str, the kind of extension we need (Flux, Blaze, Wave)

        :return: str, the extension name
        """
        # we always use first extension
        return None

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
            basename = 'Template_s1dv_{0}_sc1d_v_file_AB.fits'.format(objname)
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

    def load_science_header(self, science_file: str) -> io.LBLHeader:
        """
        Load science file header

        :param science_file: str, the science file header

        :return: fits header, the loaded header
        """
        return self.load_header(science_file)

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
            sci_hdr = self.load_science_header(science_file)
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

        :param science_file: str, the science file header (not used)
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

    def get_wave_solution(self, science_filename: Union[str, None] = None,
                          data: Union[np.ndarray, None] = None,
                          header: Union[io.LBLHeader, None] = None
                          ) -> np.ndarray:
        """
        Get a wave solution from a file (for SPIROU this is from the header)
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
        poly_type = self.params['WAVE_POLY_TYPE']
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
            # we can have two type of polynomial type
            #  TODO: in future should only be chebyshev
            if poly_type == 'Chebyshev':
                wavemap[order_num] = mp.val_cheby(wavecoeffs[order_num], xpix,
                                                  domain=[0, nbx])
            else:
                wavemap[order_num] = np.polyval(wavecoeffs[order_num][::-1],
                                                xpix)
        # ---------------------------------------------------------------------
        # return wave solution map
        return wavemap

    def drift_condition(self, table_row: Table.Row) -> bool:
        """
        Extra drift condition on a column to identify the correct reference
        file

        :param table_row: astropy.table.row.Row - the row of the table
                          to check against

        :return: True if reference file, False else-wise
        """
        # get the correct prefix for FILENAME
        filename = str(table_row['FILENAME'])
        filename = filename.replace('_FP_FP_lbl.fits', '')
        # get the correct prefix for WAVEFILE
        wavefile = table_row['WAVEFILE']

        ext = '_{0}.fits'.format('C')
        wavefile = wavefile.replace('_wave_night' + ext, '')
        wavefile = wavefile.replace('_wavesol_ref' + ext, '')
        # return test statement
        return filename == wavefile

    def load_bad_hdr_keys(self) -> Tuple[np.ndarray, str]:
        """
        Load the bad values and bad key for spirou

        :return: tuple, 1. the list of bad values, 2. the bad key in
                 a file header to check against bad values
        """
        # set up googlesheet parameters
        url_base = ('https://docs.google.com/spreadsheets/d/'
                    '{}/gviz/tq?tqx=out:csv&sheet={}')
        sheet_id = '1gvMp1nHmEcKCUpxsTxkx-5m115mLuQIGHhxJCyVoZCM'
        worksheet = 0
        bad_odo_url = url_base.format(sheet_id, worksheet)
        # log progress
        msg = 'Loading bad odometer codes from spreadsheet'
        log.general(msg)
        # fetch data
        data = requests.get(bad_odo_url)
        tbl = Table.read(data.text, format='ascii')
        # get bad keys
        bad_values = np.array(tbl['ODOMETER']).astype(str)
        # define bad file header key
        bad_key = 'EXPNUM'
        # log number of bad values loaded
        msg = '\t{0} bad values loaded'
        margs = [len(bad_values)]
        log.general(msg.format(*margs))
        # return the
        return bad_values, bad_key

    def get_berv(self, sci_hdr: io.LBLHeader) -> float:
        """
        Get the Barycenteric correction for the RV in m/s

        :param sci_hdr: io.LBLHeader, the science header

        :return:
        """
        # get BERV header key
        hdr_key = self.params['KW_BERV']
        # get BERV (if not a calibration)
        if not self.params['DATA_TYPE'] != 'SCIENCE':
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

    def filter_files(self, science_files: List[str]) -> List[str]:
        """
        Filter calibrations - no simutaenous calibrations

        :param science_files: list of science filenames

        :return: list of str, the filters calibration filenames
        """
        # get tqdm
        tqdm = base.tqdm_module(self.params['USE_TQDM'], log.console_verbosity)
        # get mjd start and end
        start = self.params['TEMPLATE_MJDSTART']
        end = self.params['TEMPLATE_MJDEND']
        # if we have a science observation and start and end are None we don't
        #   need to filter
        mcond1 = self.params['DATA_TYPE'] == 'SCIENCE'
        mcond2 = start in [None, 'None', '']
        mcond3 = end in [None, 'None', '']
        # return all files if this is the case
        if mcond2 and mcond3 and mcond1:
            return science_files
        # filtering files
        log.general('Filtering {0} files...'.format(self.params['DATA_TYPE']))
        # select the first science file as a reference file
        refimage, refhdr = self.load_science_file(science_files[0])
        ref_fibertype = self.get_dpr_fibtype(refhdr)
        # storage
        keep_files = []
        # loop around science files
        for science_file in tqdm(science_files):
            # load science file header
            sci_hdr = self.load_science_header(science_file)
            # find out if we have a calibration
            if not mcond1:
                # get dprtype in each fiber
                sci_fiber = self.get_dpr_fibtype(sci_hdr, fiber='AB')
                ref_fiber = self.get_dpr_fibtype(sci_hdr, fiber='C')
                # i.e. FP_FP or LFC_LFC
                cond1 = sci_fiber == ref_fiber
            else:
                cond1 = True
            # -----------------------------------------------------------------
            # must check time frame (if present) for FP
            cond2, cond3 = True, True
            # get mjdmid
            mjdmid = sci_hdr[self.params['KW_MID_EXP_TIME']]
            # check start time for FP calibration
            if not mcond2:
                cond2 = mjdmid >= start
            # check end time for FP calibration
            if not mcond3:
                cond3 = mjdmid <= end
            # -----------------------------------------------------------------
            # if all conditions are met keep files
            if cond1 and cond2 and cond3:
                keep_files.append(science_file)
            # -----------------------------------------------------------------
        # if we have no files break here
        if len(keep_files) == 0:
            # deal with calibration
            if not mcond1:
                emsg = ('Object is classified as a calibration however none of'
                        'the files provided have:')
                emsg += '\n\tDPRTYPE={0}_{0}'
            # deal with science
            else:
                emsg = ('Object is classified as science however none of the'
                        'files provide have:')
            # add info about the range of files (if FP files)
            if not mcond2:
                emsg += '\n\tMJDMID>{1}'
            if not mcond3:
                emsg += '\n\tMJDMID<{2}'
            # tell the user what they can do to fix this - we don't want emails
            emsg += ('\nPlease define a template and do not run the '
                     'template code or add some valid files')
            # get error arguments
            eargs = [ref_fibertype, start, end]
            # raise exception - we need some files to make a template!
            raise base_classes.LblException(emsg.format(*eargs))
        else:
            # deal with calibration
            if mcond1:
                msg = ('Object is classified as a calibration. Found {1} files'
                       ' with {0}_{0}, ignoring {2} other files')
            # deal with science
            else:
                msg = ('Object is classified as science. Found {1} files,'
                       'ignoring {2} other files')
            margs = [ref_fibertype, len(keep_files),
                     len(science_files) - len(keep_files)]
            log.info(msg.format(*margs))
        # return only files with DPRTYPE same in both fibers
        return keep_files

    def get_dpr_fibtype(self, hdr: io.LBLHeader,
                        fiber: Optional[str] = None) -> str:

        # get dprtype
        dprtype = hdr.get_hkey(self.params['KW_DPRTYPE'])
        # deal with getting fiber
        if fiber is None:
            fiber = hdr.get_hkey(self.params['KW_FIBER'])
        # split fiber
        dprfibtypes = dprtype.split('_')
        # get fiber type
        if fiber in ['AB', 'A', 'B']:
            return dprfibtypes[0]
        else:
            return dprfibtypes[1]

    def rdb_columns(self) -> Tuple[np.ndarray, List[bool]]:
        """
        Define the fits header columns names to add to the RDB file
        These should be references to keys in params

        :return: tuple, 1. np.array of strings (the keys), 2. list of bools
                 the flags whether these keys should be used with FP files
        """
        # these are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_AIRMASS', 'KW_DATE',
                    'KW_BERV', 'KW_TAU_H2O', 'KW_TAU_OTHERS',
                    'KW_DPRTYPE', 'KW_NITERATIONS', 'KW_RESET_RV',
                    'KW_SYSTEMIC_VELO', 'KW_WAVETIME', 'KW_WAVEFILE',
                    'KW_TLPDVH2O', 'KW_TLPDVOTR', 'KW_CDBWAVE', 'KW_OBJNAME',
                    'KW_RHOMB1', 'KW_RHOMB2', 'KW_CDEN_P', 'KW_SNRGOAL',
                    'KW_EXT_SNR', 'KW_BJD', 'KW_SHAPE_DX', 'KW_SHAPE_DY',
                    'KW_SHAPE_A', 'KW_SHAPE_B', 'KW_SHAPE_C', 'KW_SHAPE_D',
                    'KW_CCF_EW', 'KW_FP_INT_T', 'KW_FP_INT_P', 'KW_FPI_TEMP',
                    'KW_FPE_TEMP']
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
        # define the band names
        bands = ['Y', 'gapYJ', 'J', 'gapJH', 'H', 'gapHK', 'K', 'redK']
        # define the blue end of each band [nm]
        blue_end = [900.0, 1113.400, 1153.586, 1354.422, 1462.897, 1808.544,
                    1957.792, 2343.105]
        # define the red end of each band [nm]
        red_end = [1113.4, 1153.586, 1354.422, 1462.897, 1808.544, 1957.792,
                   2343.105, 2500.000]
        # define whether we should use regions for each band
        use_regions = [True, True, True, True, True, True,
                       True, True]
        # ---------------------------------------------------------------------
        # define the region names (suffices)
        region_names = ['', '_0-2044', '_2044-4088', '_1532-2556']
        # lower x pixel bin point [pixels]
        region_low = [0, 0, 2044, 1532]
        # upper x pixel bin point [pixels]
        region_high = [4088, 2044, 4088, 2556]
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

    def find_inputs(self):
        """
        Find the input files for an instrument and copy them to the correct
        places

        :return:
        """
        # get parameters
        params = self.params
        # get tqdm
        tqdm = base.tqdm_module(params['USE_TQDM'], log.console_verbosity)
        # get data directory
        datadir = io.check_directory(params['DATA_DIR'])
        # --------------------------------------------------------------------
        # ask user for path
        found = False
        upath = ''
        # loop until user path exists
        while not found:
            upath = input('\nEnter raw path to search for files:\t')
            if os.path.exists(upath):
                found = True
            else:
                log.warning('\nError: Path does not exist. Please try again.')
        # convert to Path
        upath = Path(upath)
        # search raw path for files
        files = np.array(list(upath.rglob('*.fits')))
        # --------------------------------------------------------------------
        # locate files
        # --------------------------------------------------------------------
        # print progress
        log.general('Locating sym FP files')
        # find sym FP e2dsff files
        suffix = '_pp_e2dsff_C.fits'
        symfp_keys = dict()
        symfp_keys[params['KW_DPRTYPE']] = ['POLAR_FP', 'OBJ_FP']
        symfp_keys[params['KW_OUTPUT']] = ['EXT_E2DS_FF']
        symfp_files = io.find_files(files, suffix=suffix, hkeys=symfp_keys)
        # print number found
        log.general('\tFound {0} sym FP files'.format(len(symfp_files)))
        # remove these from files
        files = list(np.array(files)[~np.in1d(files, symfp_files)])
        # --------------------------------------------------------------------
        # print progress
        log.general('Locating FP_FP files')
        # find FP_FP e2dsff files
        fpfp_keys = dict()
        contains = '_pp_e2dsff_AB.fits'
        fpfp_keys[params['KW_DPRTYPE']] = ['FP_FP']
        fpfp_keys[params['KW_OUTPUT']] = ['EXT_E2DS_FF']
        fpfp_files = io.find_files(files, contains=contains, hkeys=fpfp_keys)
        # print number found
        log.general('\tFound {0} FP_FP files'.format(len(fpfp_files)))
        # remove these from files
        files = list(np.array(files)[~np.in1d(files, fpfp_files)])
        # --------------------------------------------------------------------
        # print progress
        log.general('Locating Science files')
        # find science tcorr files
        suffix = 'o_pp_e2dsff_tcorr_AB.fits'
        sci_keys = dict()
        sci_keys[params['KW_DPRTYPE']] = ['POLAR_FP', 'OBJ_FP',
                                          'POLAR_DARK', 'OBJ_DARK']
        sci_keys[params['KW_DRSOBJN']] = [params['OBJECT_SCIENCE']]
        sci_keys[params['KW_OUTPUT']] = ['TELLU_OBJ']
        science_files = io.find_files(files, suffix=suffix, hkeys=sci_keys)
        # print number found
        log.general('\tFound {0} Science files'.format(len(science_files)))
        # remove these from files
        files = list(np.array(files)[~np.in1d(files, science_files)])

        # --------------------------------------------------------------------
        # print progress
        log.general('Locating Template files')
        # find template files
        if params['TEMPLATE_FILE'] in ['None', '', None]:
            suffix = 'Template_s1dv_{0}_sc1d_v_file_AB.fits'
            suffix = suffix.format(params['OBJECT_TEMPLATE'])
        else:
            suffix = params['TEMPLATE_FILE']
        temp_files = io.find_files(files, suffix=suffix)
        # print number found
        log.general('\tFound {0} Template files'.format(len(temp_files)))
        # remove these from files
        files = list(np.array(files)[~np.in1d(files, temp_files)])
        # --------------------------------------------------------------------
        # print progress
        log.general('Locating Mask files')
        # find mask files
        suffix = '_full.fits'
        mask_files = io.find_files(files, suffix=suffix)
        # print number found
        log.general('\tFound {0} Mask files'.format(len(temp_files)))
        # remove these from files
        files = list(np.array(files)[~np.in1d(files, mask_files)])
        # --------------------------------------------------------------------
        # storage of blaze files
        blaze_files = []
        # loop around fibers
        for fiber in ['AB', 'C']:
            # print progress
            log.general('Locating Blaze {0} files'.format(fiber))
            # find blaze files
            if params['BLAZE_FILE'] in ['None', '', None]:
                suffix = '_blaze_{0}.fits'.format(fiber)
                blaze_keys = dict()
                blaze_keys[params['KW_OUTPUT']] = ['FF_BLAZE']
            else:
                suffix = params['BLAZE_FILE']
                blaze_keys = None
            blaze_files += io.find_files(files, suffix=suffix, hkeys=blaze_keys)
        # print number found
        log.general('\tFound {0} Blaze files'.format(len(blaze_files)))
        # --------------------------------------------------------------------
        # copy files
        # --------------------------------------------------------------------
        # get directories
        science_dir = io.make_dir(datadir, params['SCIENCE_SUBDIR'], 'Science',
                                  subdir=params['OBJECT_SCIENCE'])
        fp_dir = io.make_dir(datadir, params['SCIENCE_SUBDIR'], 'Science',
                             subdir='FP')
        template_dir = io.make_dir(datadir, params['TEMPLATE_SUBDIR'],
                                   'Templates')
        mask_dir = io.make_dir(datadir, params['MASK_SUBDIR'], 'Mask')
        calib_dir = io.make_dir(datadir, params['CALIB_SUBDIR'], 'Calib')
        # --------------------------------------------------------------------
        # copy science
        # --------------------------------------------------------------------
        # log progress
        log.general('Copying science files')
        # loop around science files
        for science_file in tqdm(science_files):
            # get infile
            infile = str(science_file)
            outfile = os.path.join(science_dir, os.path.basename(infile))
            # copy
            shutil.copy(infile, outfile)
        # --------------------------------------------------------------------
        # copy fps
        # --------------------------------------------------------------------
        # log progress
        log.general('Copying FP files')
        # loop around science files
        for fp_file in tqdm(symfp_files + fpfp_files):
            # get infile
            infile = str(fp_file)
            outfile = os.path.join(fp_dir, os.path.basename(infile))
            # copy
            shutil.copy(infile, outfile)
        # --------------------------------------------------------------------
        # copy template files
        # --------------------------------------------------------------------
        # log progress
        log.general('Copying template files')
        # loop around science files
        for temp_file in tqdm(temp_files):
            # get infile
            infile = str(temp_file)
            outfile = os.path.join(template_dir, os.path.basename(infile))
            # copy
            shutil.copy(infile, outfile)
        # --------------------------------------------------------------------
        # copy mask files
        # --------------------------------------------------------------------
        # log progress
        log.general('Copying mask files')
        # loop around science files
        for mask_file in tqdm(mask_files):
            # get infile
            infile = str(mask_file)
            outfile = os.path.join(mask_dir, os.path.basename(infile))
            # copy
            shutil.copy(infile, outfile)
        # --------------------------------------------------------------------
        # copy blaze files
        # --------------------------------------------------------------------
        # log progress
        log.general('Copying blaze files')
        # loop around science files
        for blaze_file in tqdm(blaze_files):
            # get infile
            infile = str(blaze_file)
            outfile = os.path.join(calib_dir, os.path.basename(infile))
            # copy
            shutil.copy(infile, outfile)


# =============================================================================
# Define Spirou CADC class
# =============================================================================
class SpirouCADC(Spirou):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict, name: str = None):
        # get the name
        if name is None:
            name = 'SPIROU_CADC'
        # call to super function
        super().__init__(params, args, name)
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)

    def param_override(self):
        """
        Parameter override for NIRPS_HA ESO parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.SPIROU_CADC.override()'
        # first run the inherited method
        super().param_override()

        # Fiber must be set for SPIROU CADC
        self.param_set('FORCE_FIBER', 'AB', source=func_name)
        # Set FLUX_EXTENSION_NAME
        #   - Can be Flux (for e.fits and t.fits)
        #   - Can be Pol or StokesI or Null1 or Null2 (for p.fits)
        self.param_set('FLUX_EXTENSION_NAME', 'Flux', source=func_name)

    # -------------------------------------------------------------------------
    # SPIROU SPECIFIC METHODS
    # -------------------------------------------------------------------------
    def get_extname(self, kind: str) -> str:
        """
        Get the extension name based on the kind and params['FIBER'] key

        :param kind: str, the kind of extension we need (Flux, Blaze, Wave)

        :return: str, the extension name
        """
        # Fiber must be set for SPIROU CADC
        if 'FORCE_FIBER' not in self.params:
            emsg = 'Keyword FORCE_FIBER must be set for SPIROU CADC mode'
            raise base_classes.LblException(emsg)
        # get fiber from params
        fiber = self.params['FORCE_FIBER']
        # return the extension name
        return f'{kind}{fiber}'

    def load_science_file(self, science_file: str
                          ) -> Tuple[np.ndarray, io.LBLHeader]:
        """
        Load science data and header

        :param science_file: str, the filename to load
        :return:
        """
        # Fiber must be set for SPIROU CADC
        if 'FLUX_EXTENSION_NAME' not in self.params:
            emsg = ('Keyword FLUX_EXTENSION_NAME must be set for '
                    'SPIROU CADC mode')
            base_classes.LblException(emsg)
        if self.params['FLUX_EXTENSION_NAME'] is None:
            emsg = ('Keyword FLUX_EXTENSION_NAME must be set for '
                    'SPIROU CADC mode')
            base_classes.LblException(emsg)
        # flux extname kind
        flux_extname = self.params['FLUX_EXTENSION_NAME']
        # full extension name
        extname = self.get_extname(flux_extname)
        # load the first extension of each
        sci_data = io.load_fits(science_file, kind='science Flux extension',
                                extname=extname)
        sci_hdr = self.load_header(science_file, kind='science Flux extension',
                                   extname=extname)
        # return data and header
        return sci_data, sci_hdr

    def blaze_file(self, directory: str) -> Union[str, None]:
        """
        Make the absolute path for the blaze file if set in params

        :param directory: str, the directory the file is located at

        :return: absolute path to blaze file or None (if not set)
        """
        # Should always be taken from t.fits extension
        #   but there is a blaze (so should not be None)
        return ''

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
        # loaded from science file --> filename not required
        _ = filename
        # deal with already flagged as corrected
        if self.params['BLAZE_CORRECTED']:
            return None
        # load blaze
        blaze = io.load_fits(science_file, kind='blaze fits extension',
                             extname=self.get_extname('Blaze'))
        # deal with normalizing per order
        if normalize:
            # get the blaze parameters (may be instrument specific)
            nth_deg, bdomain = self.norm_blaze_params()
            # require the wave grid
            wavegrid = self.get_wave_solution(science_file)
            # normalize the blaze
            blaze = mp.smart_blaze_norm(wavegrid, blaze, nth_deg, bdomain)
        # return blaze
        return blaze

    def load_header(self, filename: str, kind: str = 'fits file',
                    extnum: Optional[int] = None,
                    extname: str = None) -> io.LBLHeader:
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
        if extnum is None and extname is None:
            extname = self.get_extname('Flux')
        # get header
        hdr = io.load_header(filename, kind, extnum, extname)
        # return the LBL Header class
        return io.LBLHeader.from_fits(hdr, filename)

    def load_science_header(self, science_file: str) -> io.LBLHeader:
        """
        Load science file header

        :param science_file: str, the science file header

        :return: fits header, the loaded header
        """
        return self.load_header(science_file, extname=self.get_extname('Flux'))

    def load_blaze_from_science(self, science_file: str,
                                sci_image: np.ndarray,
                                sci_hdr: io.LBLHeader,
                                calib_directory: str,
                                normalize: bool = True
                                ) -> Tuple[np.ndarray, bool]:
        """
        Load the blaze file using a science file header

        :param science_file: str, the science file header
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
            emsg = ('BLAZE_CORRECTED=True, this is not possible for {0} '
                    ' science file packaged is not blaze corrected.')
            raise base_classes.LblException(emsg.format(self.name))
        # we always load CADC blaze from extension
        # we always load CADC blaze from extension
        # sci_image, sci_hdr and calib_directory are not used
        _ = sci_image, sci_hdr, calib_directory
        # this function becomes the same as load_blaze
        return self.load_blaze('', science_file, normalize), False

    def get_wave_solution(self, science_filename: Optional[str] = None,
                          data: Optional[np.ndarray] = None,
                          header: Optional[io.LBLHeader] = None
                          ) -> np.ndarray:
        """
        Get a wave solution from a file (for SPIROU this is from the header)
        :param science_filename: str, the absolute path to the file - for
                                 spirou this is a file with the wave solution
                                 in the header
        :param header: io.LBLHeader, this is the header to use (if not given
                       requires filename to be set to load header)
        :param data: np.ndarray, this must be set along with header (if not
                     give we require filename to be set to load data)

        :return: np.ndarray, the wave map. Shape = (num orders x num pixels)
        """
        # we load wavelength solution from extension
        # so we do not use data and header
        _ = data, header
        # load wavemap
        wavemap = io.load_fits(science_filename, 'wave fits extension',
                               extname=self.get_extname('Wave'))
        # return wave solution map
        return wavemap


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

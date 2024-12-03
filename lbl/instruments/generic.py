#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPIRou instrument class here: instrument specific settings

Created on 2021-05-27

@author: cook
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

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.generic.py'
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
class Generic(Instrument):
    def __init__(self, params: base_classes.ParamDict):
        # call to super function
        super().__init__('Generic')
        # extra parameters (specific to instrument)
        self.default_template_name = 'Template_{0}_Generic.fits'
        # define wave limits in nm
        self.wavemin = 377.189
        self.wavemax = 790.788
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()

    # -------------------------------------------------------------------------
    # INSTRUMENT SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    def generic_validate(self, key: str) -> Any:
        """
        We must validate the keys for a generic instrument as these are not
        being set by default

        :param key: str, the key to validate

        :raises LblException: if key is not set
        :return: Any, the value in params
        """
        if self.params[key] is None:
            emsg = ('Key {0} must be defined when using Generic Instrument'
                    '').format(key)
            raise LblException(emsg)
        # set the parameter
        self.params.set(key, self.params[key], source='USER[Generic]')
        return self.params[key]

    def param_override(self):
        """
        Parameter override for SPIRou parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.Generic.override()'
        # update instrument
        value = self.generic_validate('GENERIC_INSTRUMENT')
        self.param_set('INSTRUMENT', value, source=func_name)
        # set a new template name
        newtname = self.default_template_name.replace('Generic', value)
        # update data source
        value = self.generic_validate('GENERIC_DATA_SOURCE')
        self.param_set('DATA_SOURCE', value, source=func_name)
        # ---------------------------------------------------------------------
        # update the template name
        self.default_template_name = newtname
        # ---------------------------------------------------------------------
        # generic wave min and max
        value = self.generic_validate('GENERIC_WAVEMIN')
        self.wavemin = value
        value = self.generic_validate('GENERIC_WAVEMAX')
        self.wavemax = value
        # ---------------------------------------------------------------------
        # Now override parameters that would be overwritten by a normal
        # instrument
        # They are set to None in the wrap script so we make sure the user
        # has set these
        # ---------------------------------------------------------------------
        # add instrument earth location
        #    (for use in astropy.coordinates.EarthLocation)
        self.generic_validate('EARTH_LOCATION')
        # define the High pass width in km/s
        self.generic_validate('HP_WIDTH')
        # define the SNR cut off threshold
        self.generic_validate('SNR_THRESHOLD')
        # define which bands to use for the clean CCF (see astro.ccf_regions)
        self.generic_validate('CCF_CLEAN_BANDS')
        # define the plot order for the compute rv model plot
        self.generic_validate('COMPUTE_MODEL_PLOT_ORDERS')
        # define the compil minimum wavelength allowed for lines [nm]
        self.generic_validate('COMPIL_WAVE_MIN')
        # define the compil maximum wavelength allowed for lines [nm]
        self.generic_validate('COMPIL_WAVE_MAX')
        # define the maximum pixel width allowed for lines [pixels]
        self.generic_validate('COMPIL_MAX_PIXEL_WIDTH')
        # define min likelihood of correlation with BERV
        self.generic_validate('COMPIL_CUT_PEARSONR')
        # define the CCF e-width to use for FP files
        self.generic_validate('COMPIL_FP_EWID')
        # define whether to add the magic "binned wavelength" bands rv
        self.generic_validate('COMPIL_ADD_UNIFORM_WAVEBIN')
        # define the number of bins used in the magic "binned wavelength" bands
        self.generic_validate('COMPIL_NUM_UNIFORM_WAVEBIN')
        # define the first band (from get_binned_parameters) to plot (band1)
        #    this is used for colour   band2 - band3
        self.generic_validate('COMPILE_BINNED_BAND1')
        # define the second band (from get_binned_parameters) to plot (band2)
        #    this is used for colour   band2 - band3
        self.generic_validate('COMPILE_BINNED_BAND2')
        # define the third band (from get_binned_parameters) to plot (band3)
        #    this is used for colour   band2 - band3
        self.generic_validate('COMPILE_BINNED_BAND3')
        # define the reference wavelength used in the slope fitting in nm
        self.generic_validate('COMPIL_SLOPE_REF_WAVE')
        # Set FLUX_EXTENSION_NAME
        self.param_set('FLUX_EXTENSION_NAME', 'FLUX', source=func_name)
        # define the name of the sample wave grid file (saved to the calib dir)
        self.param_set('SAMPLE_WAVE_GRID_FILE',
                        'sample_wave_grid.fits', source=func_name)
        # define the FP reference string that defines that an FP observation was
        #    a reference (calibration) file - should be a list of strings
        # Question: Check DRP TYPE for STAR,FP file
        self.param_set('FP_REF_LIST', ['STAR,WAVE,FP'], source=func_name)
        # define the FP standard string that defines that an FP observation
        #    was NOT a reference file - should be a list of strings
        # Question: Check DRP TYPE for STAR,FP file
        self.param_set('FP_STD_LIST', ['STAR,WAVE,FP'], source=func_name)
        # define readout noise per instrument (assumes ~5e- and 10 pixels)
        self.generic_validate('READ_OUT_NOISE')
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
        self.generic_validate('MASK_SNR_MIN')
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
        self.generic_validate('BLAZE_SMOOTH_SIZE')
        # blaze threshold (s1d template)
        self.generic_validate('BLAZE_THRESHOLD')
        # define the size of the berv bins in m/s
        self.generic_validate('BERVBIN_SIZE')
        # ---------------------------------------------------------------------
        # define whether to do the tellu-clean
        self.generic_validate('DO_TELLUCLEAN')
        # define the dv offset for tellu-cleaning in km/s
        self.generic_validate('TELLUCLEAN_DV0')
        # Define the lower wave limit for the absorber spectrum masks in nm
        self.generic_validate('TELLUCLEAN_MASK_DOMAIN_LOWER')
        # Define the upper wave limit for the absorber spectrum masks in nm
        self.generic_validate('TELLUCLEAN_MASK_DOMAIN_UPPER')
        # Define whether to force using airmass from header
        self.generic_validate('TELLUCLEAN_FORCE_AIRMASS')
        # Define the CCF scan range in km/s
        self.generic_validate('TELLUCLEAN_CCF_SCAN_RANGE')
        # Define the maximum number of iterations for the tellu-cleaning loop
        self.generic_validate('TELLUCLEAN_MAX_ITERATIONS')
        # Define the kernel width in pixels
        self.generic_validate('TELLUCLEAN_KERNEL_WID')
        # Define the gaussian shape (2=pure gaussian, >2=boxy)
        self.generic_validate('TELLUCLEAN_GAUSSIAN_SHAPE')
        # Define the wave grid lower wavelength limit in nm
        self.generic_validate('TELLUCLEAN_WAVE_LOWER')
        # Define the wave griv upper wavelength limit
        self.generic_validate('TELLUCLEAN_WAVE_UPPER')
        # Define the transmission threshold exp(-1) at which tellurics are
        #     uncorrectable
        self.generic_validate('TELLUCLEAN_TRANSMISSION_THRESHOLD')
        # Define the sigma cut threshold above which pixels are removed from fit
        self.generic_validate('TELLUCLEAN_SIGMA_THRESHOLD')
        # Define whether to recenter the CCF on the first iteration
        self.generic_validate('TELLUCLEAN_RECENTER_CCF')
        # Define whether to recenter the CCF of others on the first iteration
        self.generic_validate('TELLUCLEAN_RECENTER_CCF_FIT_OTHERS')
        # Define the default water absorption to use
        self.generic_validate('TELLUCLEAN_DEFAULT_WATER_ABSO')
        # Define the lower limit on valid exponent of water absorbers
        self.generic_validate('TELLUCLEAN_WATER_BOUNDS_LOWER')
        # Define the upper limit on valid exponent of water absorbers
        self.generic_validate('TELLUCLEAN_WATER_BOUNDS_UPPER')
        # Define the lower limit on valid exponent of other absorbers
        self.generic_validate('TELLUCLEAN_OTHERS_BOUNDS_LOWER')
        # Define the upper limit on valid exponent of other absorbers
        self.generic_validate('TELLUCLEAN_OTHERS_BOUNDS_UPPER')
        # ---------------------------------------------------------------------
        # Parameters for the template construction
        # ---------------------------------------------------------------------
        # max number of bins for the median of the template. Avoids handling
        # too many spectra at once.
        self.generic_validate('TEMPLATE_MEDBINMAX')
        # maximum RMS between the template and the median of the template
        # to accept the median of the template as a good template. If above
        # we iterate once more. Expressed in m/s
        self.generic_validate('MAX_CONVERGENCE_TEMPLATE_RV')
        # ---------------------------------------------------------------------
        # Header keywords
        # ---------------------------------------------------------------------
        # define the key that gives the mid exposure time in MJD
        self.param_set('KW_MID_EXP_TIME', 'BJD', source=func_name)
        # define the start time of the observation
        self.param_set('KW_MJDATE', 'BJD', source=func_name)
        # define snr keyword
        self.param_set('KW_SNR', 'SNR', source=func_name)
        # define berv keyword
        self.param_set('KW_BERV', 'BERV', source=func_name)
        # # define the Blaze calibration file
        self.param_set('KW_BLAZE_FILE', 'NONE', source=func_name)
        # define the exposure time of the observation
        self.param_set('KW_EXPTIME', 'EXPTIME', source=func_name)
        # define the airmass of the observation
        self.param_set('KW_AIRMASS', 'AIRMASS', source=func_name)
        # define the human date of the observation
        self.param_set('KW_DATE', 'DATE', source=func_name)
        # define the tau_h20 of the observation
        self.param_set('KW_TAU_H2O', 'TLPEH2O', source=func_name)
        # define the tau_other of the observation
        self.param_set('KW_TAU_OTHERS', 'TLPEOTR', source=func_name)
        # define the DPRTYPE of the observation
        self.param_set('KW_DPRTYPE', 'DPRTYPE', source=func_name)
        # define the filename of the wave solution
        self.param_set('KW_WAVEFILE', 'NONE', source=func_name)
        # define the original object name
        self.param_set('KW_OBJNAME', 'OBJNAME', source=func_name)
        # define the SNR goal per pixel per frame (can not exist - will be
        #   set to zero)
        self.param_set('KW_SNRGOAL', 'NONE', source=func_name)
        # define the SNR in chosen order
        self.param_set('KW_EXT_SNR', 'EXT_SNR', source=func_name)
        # define the barycentric julian date
        self.param_set('KW_BJD', 'BJD', source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.param_set('KW_REF_KEY', 'DPRTYPE', source=func_name)
        # velocity of template from CCF
        self.param_set('KW_MODELVEL', 'MODELVEL', source=func_name)
        # the temperature of the object
        self.param_set('KW_TEMPERATURE', None, source=func_name)

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
        hdr = io.load_header(filename, kind, 0, None)
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

        # have to check for blaze extension
        with fits.open(science_file) as hdul:
            if 'BLAZE' not in hdul:
                emsg = ('Blaze extension not found in {0} for {1}. '
                        '\n\tEither data format is wrong or '
                        'BLAZE_CORRECTED needs to be set to True')
                eargs = [science_file, self.params['INSTRUMENT']]
                raise LblException(emsg.format(*eargs))

        # load blaze
        blaze = io.load_fits(science_file, kind='blaze fits extension',
                             extname='BLAZE')
        # deal with normalizing per order
        if normalize:
            # normalize blaze per order
            for order_num in range(blaze.shape[0]):
                # normalize by the 90% percentile
                norm = np.nanpercentile(blaze[order_num], 90)
                # apply to blaze
                blaze[order_num] = blaze[order_num] / norm
        # return blaze
        return blaze

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
            # get time
            times.append(sci_hdr[self.params['KW_MID_EXP_TIME']])
        # get sort mask
        sortmask = np.argsort(times)
        # apply sort mask
        science_files = np.array(science_files)[sortmask]
        # return sorted files
        return list(science_files)

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
                    'Generic mode')
            base_classes.LblException(emsg)
        if self.params['FLUX_EXTENSION_NAME'] is None:
            emsg = ('Keyword FLUX_EXTENSION_NAME must be set for '
                    'Generic mode')
            base_classes.LblException(emsg)
        # full extension name
        extname = 'FLUX'
        # load the first extension of each
        sci_data = io.load_fits(science_file, kind='science Flux extension',
                                extname=extname)
        sci_hdr = self.load_header(science_file, kind='science Flux extension',
                                   extnum=0)
        # return data and header
        return sci_data, sci_hdr

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
            # blaze corrected
            return np.ones_like(sci_image), True
        # load blaze
        blaze = io.load_fits(science_file, kind='blaze fits extension',
                             extname='BLAZE')
        # deal with normalizing per order
        if normalize:
            # normalize blaze per order
            for order_num in range(blaze.shape[0]):
                # normalize by the 90% percentile
                norm = np.nanpercentile(blaze[order_num], 90)
                # apply to blaze
                blaze[order_num] = blaze[order_num] / norm
        # return blaze
        return blaze, False

    def no_blaze_corr(self, sci_image: np.ndarray,
                      sci_wave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        If we do not have a blaze we need to create an artificial one so that
        the s1d has a proper weighting

        :param sci_image: the science image (will be unblazed corrected)
        :param sci_wave: the wavelength solution for the science image

        :return: Tuple, 1. the unblazed science_image, 2. the artifical blaze
        """
        # get the wave centers for each order
        wave_cen = sci_wave[:, sci_wave.shape[1] // 2]
        # espresso has 2 orders per 'true' order so have to take every other
        #   wave element
        wave_cen = wave_cen[::2]
        # find the 'diffraction' order for a given 'on-detector' order
        dpeak = wave_cen / (wave_cen - np.roll(wave_cen, 1))
        dfit, _ = mp.robust_polyfit(1 / wave_cen, dpeak, 1, 3)
        # ---------------------------------------------------------------------
        # use the fit to get the blaze assuming a sinc**2 profile.
        # The minima of a given order corresponds to the position of the
        # consecutive orders
        # ---------------------------------------------------------------------
        # storage for the calculated blaze
        blaze = np.zeros(sci_wave.shape)
        # loop around each order
        for order_num in range(sci_wave.shape[0]):
            # get the wave grid for this order
            owave = sci_wave[order_num]
            # get the center of this order (with a small offset to avoid
            #  a division by zero in the sinc at phase = 0
            owave_cen = owave[len(owave) // 2] + 1e-6
            # calculate the period of this order
            period = owave_cen / np.polyval(dfit, 1 / owave)
            # calculate the phase of the sinc**2
            phase = np.pi * (owave - owave_cen) / period
            # assume the sinc profile. There is a factor 2 difference in the
            #   phase as the sinc is squared. sin**2 has a period that is a
            #   factor of 2 shorter than the sin
            blaze[order_num] = (np.sin(phase) / phase) ** 2
        # the blaze is not expressed as a flux density but the science
        # spectrum is. We match the two
        gradwave = np.gradient(sci_wave, axis=1)
        for order_num in range(blaze.shape[0]):
            gradwave[order_num] /= np.nanmedian(gradwave[order_num])
        # un-correct the science image
        sci_image = (sci_image / gradwave) * blaze
        # return un-corrected science image and the calculated blaze
        return sci_image, blaze

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
                               extname='WAVELENGTH')
        # return wave solution map
        return wavemap

    def load_bad_hdr_keys(self) -> Tuple[list, Any]:
        """
        Load the bad values and bad key for Espresso -- not used currently

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
        drs_keys = ['KW_MJDATE', 'KW_EXPTIME', 'KW_DATE', 'KW_DPRTYPE',
                    'KW_OBJNAME', 'KW_EXT_SNR']
        # add the filename
        tdict = self.add_dict_list_value(tdict, 'FILENAME', filename)
        # loop around header keys
        for drs_key in drs_keys:
            # if key is in params we can add the value to keys
            if drs_key in self.params:
                key = self.params[drs_key]
            else:
                key = str(drs_key)

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
        # these are defined in params
        drs_keys = ['KW_MJDATE', 'KW_EXPTIME',
                    'KW_AIRMASS', 'KW_DATE', 'KW_BERV', 'KW_DPRTYPE',
                    'KW_TAU_H2O', 'KW_TAU_OTHERS' 'KW_NITERATIONS',
                    'KW_RESET_RV', 'KW_SYSTEMIC_VELO', 'KW_OBJNAME',
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
                # need to deal with keys that define multiple drs keys
                #   in this case use the original drs_key name
                key = self.params[drs_key]
                if isinstance(key, str):
                    keys.append(key)
                else:
                    keys.append(drs_key)
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

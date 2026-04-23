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

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.expres.py'
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
class ESSP(Instrument):
    """
    DO NOT USE DIRECTLY please use one of the instruments that inherits this

    This is just for the common functionality of the ESSP pipeline
    """
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict, name: str = None):
        """
        :param params:
        :param args:
        :param name:
        """
        # get the name
        if name is None:
            name = 'ESSP'
        # call to super function
        super().__init__(name)
        # extra parameters (specific to instrument)
        self.default_template_name = None
        self.default_mask_name = None
        self.default_sample_wave_name = None
        # define wave limits in nm
        self.wavemin = None
        self.wavemax = None
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)

    # -------------------------------------------------------------------------
    # ESSP SPECIFIC PARAMETERS --> set in instrument
    # -------------------------------------------------------------------------
    def param_override(self):
        """
        Parameter override for ESSP parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.ESSP.param_override()'
        # ---------------------------------------------------------------------
        # Header keywords
        # ---------------------------------------------------------------------
        # define the key that gives the mid exposure time in MJD
        self.param_set('KW_MID_EXP_TIME', 'MJD_UTC',
                        source=func_name)
        # define the start time of the observation
        self.param_set('KW_MJDATE', 'MJD_UTC', source=func_name)
        # define snr keyword
        self.param_set('KW_SNR', None, source=func_name)
        # define berv keyword
        self.param_set('KW_BERV', 'BERV', source=func_name)
        # define the Blaze calibration file
        self.param_set('KW_BLAZE_FILE', None, source=func_name)
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
        self.param_set('KW_DPRTYPE', None, source=func_name)
        # define the filename of the wave solution  ## SUSPECT
        self.param_set('KW_WAVEFILE', None, source=func_name)
        # define the original object name
        self.param_set('KW_OBJNAME', None, source=func_name)
        # define the SNR goal per pixel per frame (can not exist - will be
        #   set to zero)
        # TODO -> no equivalent in ESPRESSO
        self.param_set('KW_SNRGOAL', 'NONE', source=func_name)
        # define the SNR in chosen order
        self.param_set('KW_EXT_SNR', None, source=func_name)
        # define the barycentric julian date
        self.param_set('KW_BJD', None, source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.param_set('KW_REF_KEY', None, source=func_name)
        # velocity of template from CCF
        # self.param_set('KW_MODELVEL', 'HIERARCH TNG QC CCF RV',
        #                source=func_name)
        self.param_set('KW_MODELVEL', 'MODELVEL', source=func_name)
        # the temperature of the object
        # TODO: how do we get the temperature for ESPRESSO?
        self.param_set('KW_TEMPERATURE', None, source=func_name)




    # -------------------------------------------------------------------------
    # ESSP SPECIFIC METHODS
    # -------------------------------------------------------------------------
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
        elif self.params['OBJECT_COMPARISON'] is None:
            raise LblException('OBJECT_COMPARISON name must be defined')
        else:
            objname = self.params['OBJECT_COMPARISON']
            # define base name
            basename = self.default_mask_name.format(obj=objname,
                                                     mtype=mask_type)
            # get absolute path
            abspath = os.path.join(mask_directory, basename)
        # check that this file exists
        if required:
            io.check_file_exists(abspath, 'mask')
        # return absolute path
        return abspath

    def blaze_file(self, directory: str) -> Union[str, None]:
        """
        Make the absolute path for the blaze file if set in params

        :param directory: str, the directory the file is located at

        :return: absolute path to blaze file or None (if not set)
        """
        # Should always be taken from .fits extension
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
                             extname='blaze')
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

    def load_science_file(self, science_file: str
                          ) -> Tuple[np.ndarray, io.LBLHeader]:
        """
        Load a science exposure

        Note data should be a 2D array (even if data is 1D)
        Treat 1D data as a single order?

        :param science_file: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (io.LBLHeader)
        """
        # load the first extension of each
        sci_data = io.load_fits(science_file, kind='science fits file',
                                extname='FLUX')
        sci_hdr = self.load_header(science_file, kind='science fits file')
        # return data and header
        return sci_data, sci_hdr

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
            # noinspection PyTypeChecker
            mid_exp_time = float(sci_hdr[self.params['KW_MID_EXP_TIME']])
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
        blaze = io.load_fits(abspath, kind='blaze fits file', extname='BLAZE')
        # require the wave grid
        wavegrid = self.get_wave_solution(science_file, sci_image,
                                          sci_hdr)
        # blaze is not pixel blaze (spectral density blaze)
        blaze = blaze * np.gradient(wavegrid, axis=1)
        # deal with normalizing per order
        if normalize:
            # get the blaze parameters (may be instrument specific)
            nth_deg, bdomain = self.norm_blaze_params()
            # normalizse the blaze
            blaze = mp.smart_blaze_norm(wavegrid, blaze, nth_deg, bdomain)
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
        # un-correct the science image
        sci_image = sci_image * blaze
        # return un-corrected science image and the calculated blaze
        return sci_image, blaze

    def get_wave_solution(self, science_filename: Optional[str] = None,
                          data: Optional[np.ndarray] = None,
                          header: Optional[io.LBLHeader] = None
                          ) -> np.ndarray:
        """
        Get a wave solution from a file
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
        # wave solution is in angstrom --> nm
        wavemap = wavemap / 10.0
        # return wave solution map
        return wavemap

    def load_bad_hdr_keys(self) -> Tuple[list, Any]:
        """
        Load the bad values and bad key-- not used currently

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
        if self.params['OBJECT_SCIENCE'] in ['FP', 'LFC', 'SUN']:
            berv = 0.0
        elif hdr_key is None:
            berv = 0.0
        else:
            berv = sci_hdr.get_hkey(hdr_key, dtype=float) * 1000
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
                    'KW_DATE']
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
                    'KW_AIRMASS', 'KW_DATE', 'KW_BERV',
                    'KW_TAU_H2O', 'KW_TAU_OTHERS', 'KW_NITERATIONS',
                    'KW_RESET_RV',
                    'KW_SYSTEMIC_VELO', 'KW_WAVEFILE', 'KW_CCF_EW']
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
        bjd = header.get_hkey(kw_bjd, required=False)
        if bjd is None or isinstance(bjd, str) or np.isnan(bjd):
            try:
                # return RJD = MJD + 0.5
                return float(mid_exp_time) + 0.5
            except Exception:
                pass
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

    def write_tellu_cleaned(self, write_tellu_file: str, props: dict,
                            sci_hdict: io.LBLHeader,
                            science_filename: Optional[str] = None):
        """
        Write the write_tellu_file to disk

        :param write_tellu_file: str, the file and path to write to
        :param props: dictionnary output from the TELLUCLEANed code
        :param sci_hdict: fits Header, an input file header to copy the header
                          from to the new template file
        :param science_filename: str, the science filename (not used for
                                 default)
        :return:
        """
        _ = science_filename
        # convert hdict to header
        sci_hdr = sci_hdict.to_fits()
        # populate primary header
        header = fits.Header()
        # copy header from reference header
        header = io.copy_header(header, sci_hdr)
        # add custom keys
        header = self.set_hkey(header, 'KW_VERSION', __version__)
        header = self.set_hkey(header, 'KW_VDATE', __date__)
        header = self.set_hkey(header, 'KW_PDATE', Time.now().iso)
        header = self.set_hkey(header, 'KW_INSTRUMENT',
                               self.params['INSTRUMENT'])
        # set the LBL output data type
        header = self.set_hkey(header, 'KW_OUTPUT', 'LBL_TELLU_CLEAN')
        # set the LBL input object object name
        header = self.set_hkey(header, 'KW_LBL_OBJNAME',
                               self.params['OBJECT_SCIENCE'].strip())
        # set the LBL input template object name
        header = self.set_hkey(header, 'KW_LBL_TMPNAME',
                               self.params['OBJECT_COMPARISON'].strip())
        # add telluric key words
        header = self.set_hkey(header, 'KW_TAU_H2O',
                               props['pre_cleaned_exponent_water'])
        header = self.set_hkey(header, 'KW_TAU_OTHERS',
                               props['pre_cleaned_exponent_others'])
        # set image as pre_cleaned_flux
        image = props['pre_cleaned_flux']
        # we need to get the data array from the fits file
        data_array = io.load_fits(props['FILENAME'],
                                  kind='science fits file',
                                  extname='optimal')
        # we push the image into the data arrays "spectrum" column
        data_array['spectrum'] = image
        # adding extensions that are not the flux after telluric correction
        #   (error propagation, wavelength grid)
        datalist = [None, data_array]
        headerlist = [header, None]
        datatypelist = [None, 'table']
        # open hdulist
        with fits.open(props['FILENAME']) as hdulist:
            # add the header for extension 1
            if len(hdulist) > 1:
                headerlist[1] = hdulist[1].header
            # loop around and add other extensions
            for hdu in hdulist[2:]:
                datalist.append(hdu.data)
                headerlist.append(hdu.header)
                if isinstance(hdu, fits.hdu.image.ImageHDU):
                    datatypelist.append('image')
                else:
                    datatypelist.append('table')
        # ---------------------------------------------------------------------
        # change the file name
        write_tellu_file = self.modify_tellu_filename(write_tellu_file)
        # ---------------------------------------------------------------------
        # Save template to disk
        log.general('Saving tellu-cleaned file: {0}'.format(write_tellu_file))
        # ---------------------------------------------------------------------
        # write to file
        io.write_fits(write_tellu_file, data=datalist,
                      header=headerlist, dtype=datatypelist)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

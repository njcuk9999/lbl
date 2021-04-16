#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPIRou instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
from astropy.table import Table
from astropy.io import fits
import glob
import numpy as np
import os
import requests
from typing import Dict, Tuple, Union

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
log = base_classes.log


# =============================================================================
# Define Spirou class
# =============================================================================
class Spirou(Instrument):
    def __init__(self, params: base_classes.ParamDict):
        # call to super function
        super().__init__('SPIROU')
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()

    # -------------------------------------------------------------------------
    # SPIROU SPECIFIC PARAMETERS
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
        self.params.set('INSTRUMENT', 'SPIROU', source=func_name)
        # define the default science input files
        self.params.set('INPUT_FILE', '*e2dsff*AB.fits', source=func_name)
        # define the mask
        self.params.set('REF_TABLE_FMT', 'csv', source=func_name)
        # define the High pass width in km/s
        self.params.set('HP_WIDTH', 223, source=func_name)
        # define the SNR cut off threshold
        self.params.set('SNR_THRESHOLD', 10, source=func_name)
        # define the plot order for the compute rv model plot
        self.params.set('COMPUTE_MODEL_PLOT_ORDERS', [35], source=func_name)
        # define the compil minimum wavelength allowed for lines [nm]
        self.params.set('COMPIL_WAVE_MIN', 900, source=func_name)
        # define the compil maximum wavelength allowed for lines [nm]
        self.params.set('COMPIL_WAVE_MAX', 2500, source=func_name)
        # define the maximum pixel width allowed for lines [pixels]
        self.params.set('COMPIL_MAX_PIXEL_WIDTH', 50, source=func_name)
        # define the first band (from get_binned_parameters) to plot (band1)
        self.params.set('COMPILE_BINNED_BAND1', 'H', source=func_name)
        # define the second band (from get_binned_parameters) to plot (band2)
        #    this is used for colour   band2 - band3
        self.params.set('COMPILE_BINNED_BAND2', 'J', source=func_name)
        # define the third band (from get_binned_parameters) to plot (band3)
        #    this is used for colour   band2 - band3
        self.params.set('COMPILE_BINNED_BAND3', 'H', source=func_name)
        # define the FP reference string that defines that an FP observation was
        #    a reference (calibration) file - should be a list of strings
        self.params.set('FP_REF_LIST', ['FP_FP'], source=func_name)
        # define the FP standard string that defines that an FP observation
        #    was NOT a reference file - should be a list of strings
        self.params.set('FP_STD_LIST', ['OBJ_FP', 'POLAR_FP'], source=func_name)

        # ---------------------------------------------------------------------
        # Header keywords
        # ---------------------------------------------------------------------
        # define wave coeff key in header
        self.params.set('KW_WAVECOEFFS', 'WAVE{0:04d}', source=func_name)
        # define wave num orders key in header
        self.params.set('KW_WAVEORDN', 'WAVEORDN', source=func_name)
        # define wave degree key in header
        self.params.set('KW_WAVEDEGN', 'WAVEDEGN', source=func_name)
        # define the key that gives the mid exposure time in MJD
        self.params.set('KW_MID_EXP_TIME', 'MJDMID', source=func_name)
        # define snr keyword
        self.params.set('KW_SNR', 'EXTSN035', source=func_name)
        # define berv keyword
        self.params.set('KW_BERV', 'BERV', source=func_name)
        # define the Blaze calibration file
        self.params.set('KW_BLAZE_FILE', 'CDBBLAZE', source=func_name)
        # define the start time of the observation
        self.params.set('KW_MJDATE', 'MJDATE', source=func_name)
        # define the exposure time of the observation
        self.params.set('KW_EXPTIME', 'EXPTIME', source=func_name)
        # define the airmass of the observation
        self.params.set('KW_AIRMASS', 'AIRMASS', source=func_name)
        # define the filename of the observation
        self.params.set('KW_FILENAME', 'FILENAME', source=func_name)
        # define the human date of the observation
        self.params.set('KW_DATE', 'DATE-OBS', source=func_name)
        # define the tau_h20 of the observation
        self.params.set('KW_TAU_H2O', 'TAU_H2O', source=func_name)
        # define the tau_other of the observation
        self.params.set('KW_TAU_OTHERS', 'TAU_OTHE', source=func_name)
        # define the DPRTYPE of the observation
        self.params.set('KW_DPRTYPE', 'DPRTYPE', source=func_name)
        # define the observation time (mjd) of the wave solution
        self.params.set('KW_WAVETIME', 'WAVETIME', source=func_name)
        # define the filename of the wave solution
        self.params.set('KW_WAVEFILE', 'WAVEFILE', source=func_name)
        # define the telluric preclean velocity of water absorbers
        self.params.set('KW_TLPDVH2O', 'TLPDVH2O', source=func_name)
        # define the telluric preclean velocity of other absorbers
        self.params.set('KW_TLPDVOTR', 'TLPDVOTR', source=func_name)
        # define the wave solution calibration filename
        self.params.set('KW_CDBWAVE', 'CDBWAVE', source=func_name)
        # define the original object name
        self.params.set('KW_OBJNAME', 'OBJNAME', source=func_name)
        # define the rhomb 1 predefined position
        self.params.set('KW_RHOMB1', 'SBRHB1_P', source=func_name)
        # define the rhomb 2 predefined position
        self.params.set('KW_RHOMB2', 'SBRHB2_P', source=func_name)
        # define the calib-reference density
        self.params.set('KW_CDEN_P', 'SBCDEN_P', source=func_name)
         # define the SNR goal per pixel per frame
        self.params.set('KW_SNRGOAL', 'SNRGOAL', source=func_name)
        # define the SNR in chosen order
        self.params.set('KW_EXT_SNR', 'EXTSN035', source=func_name)
        # define the barycentric julian date
        self.params.set('KW_BJD', 'BJD', source=func_name)
        # define the shape code dx value
        self.params.set('KW_SHAPE_DX', 'SHAPE_DX', source=func_name)
        # define the shape code dy value
        self.params.set('KW_SHAPE_DY', 'SHAPE_DY', source=func_name)
        # define the shape code A value
        self.params.set('KW_SHAPE_A', 'SHAPE_A', source=func_name)
        # define the shape code B value
        self.params.set('KW_SHAPE_B', 'SHAPE_B', source=func_name)
        # define the shape code C value
        self.params.set('KW_SHAPE_C', 'SHAPE_C', source=func_name)
        # define the shape code D value
        self.params.set('KW_SHAPE_D', 'SHAPE_D', source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.params.set('KW_REF_KEY', 'DPRTYPE', source=func_name)
        # velocity of template from CCF
        self.params.set('KW_MODELVEL', 'MODELVEL', source=func_name)

    # -------------------------------------------------------------------------
    # SPIROU SPECIFIC METHODS
    # -------------------------------------------------------------------------
    def mask_file(self, directory: str) -> str:
        """
        Make the absolute path for the mask file

        :param directory: str, the directory the file is located at

        :return: absolute path to mask file
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            objname = self.params['OBJECT_SCIENCE']
        # define base name
        basename = '{0}_pos.fits'.format(objname)
        # get absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        io.check_file_exists(abspath)
        # return absolute path
        return abspath

    def template_file(self, directory: str) -> str:
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        # deal with no object template
        self._set_object_template()
        # set template name
        objname = self.params['OBJECT_TEMPLATE']
        # get template file
        if self.params['TEMPLATE_FILE'] is None:
            basename = 'Template_s1d_{0}_sc1d_v_file_AB.fits'.format(objname)
        else:
            basename = self.params['TEMPLATE_FILE']
        # get absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        io.check_file_exists(abspath)
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
        io.check_file_exists(abspath)
        # return absolute path
        return abspath

    def load_blaze(self, filename: str) -> Union[np.ndarray, None]:
        """
        Load a blaze file

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self
        if filename is not None:
            blaze, _ = io.load_fits(filename, kind='blaze fits file')
            # normalize blaze per order
            for order_num in range(blaze.shape[0]):
                # normalize by the 90% percentile
                norm = np.nanpercentile(blaze[order_num], 90)
                # apply to blaze
                blaze[order_num] = blaze[order_num] / norm
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
        # load the mask header
        mask_hdr = io.load_header(mask_file, kind='mask fits file')
        # get info on template systvel for splining correctly
        systemic_vel = -1000 * io.get_hkey(mask_hdr, 'SYSTVEL')
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
        basename = self.params['INPUT_FILE']
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

    def load_blaze_from_science(self, sci_hdr: fits.Header,
                                calib_directory: str) -> np.ndarray:
        """
        Load the blaze file using a science file header

        :param sci_hdr: fits.Header - the science file header
        :param calib_directory: str, the directory containing calibration files
                                (i.e. containing the blaze files)
        :return: None
        """
        # get blaze file from science header
        blaze_file = io.get_hkey(sci_hdr, self.params['KW_BLAZE_FILE'])
        # construct absolute path
        abspath = os.path.join(calib_directory, blaze_file)
        # check that this file exists
        io.check_file_exists(abspath)
        # read blaze file (data and header)
        blaze, _ = io.load_fits(abspath, kind='blaze fits file')
        # normalize blaze per order
        for order_num in range(blaze.shape[0]):
            # normalize by the 90% percentile
            norm = np.nanpercentile(blaze[order_num], 90)
            # apply to blaze
            blaze[order_num] = blaze[order_num] / norm
        # return blaze
        return blaze

    def get_wave_solution(self, science_filename: Union[str, None] = None,
                          data: Union[np.ndarray, None] = None,
                          header: Union[fits.Header, None] = None
                          ) -> np.ndarray:
        """
        Get a wave solution from a file (for SPIROU this is from the header)
        :param science_filename: str, the absolute path to the file - for
                                 spirou this is a file with the wave solution
                                 in the header
        :param header: fits.Header, this is the header to use (if not given
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
            sci_data, sci_hdr = io.load_fits(science_filename,
                                             'wave fits file')
        else:
            sci_data, sci_hdr = data, header
        # ---------------------------------------------------------------------
        # get the data shape
        nby, nbx = sci_data.shape
        # get xpix
        xpix = np.arange(nbx)
        # ---------------------------------------------------------------------
        # get wave order from header
        waveordn = io.get_hkey(sci_hdr, kw_waveordn, science_filename)
        wavedegn = io.get_hkey(sci_hdr, kw_wavedegn, science_filename)
        # get the wave 2d list
        wavecoeffs = io.get_hkey_2d(sci_hdr, key=kw_wavecoeffs,
                                    dim1=waveordn, dim2=wavedegn + 1,
                                    filename=science_filename)
        # ---------------------------------------------------------------------
        # convert to wave map
        wavemap = np.zeros([waveordn, nbx])
        for order_num in range(waveordn):
            wavemap[order_num] = np.polyval(wavecoeffs[order_num][::-1], xpix)
        # ---------------------------------------------------------------------
        # return wave solution map
        return wavemap

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

    def get_berv(self, sci_hdr: fits.Header) -> float:
        """
        Get the Barycenteric correction for the RV

        :param sci_hdr: fits.Header, the science header

        :return:
        """
        # get BERV header key
        hdr_key = self.params['KW_BERV']
        # BERV depends on whether object is FP or not
        if 'FP' not in self.params['OBJECT_SCIENCE']:
            berv = io.get_hkey(sci_hdr, hdr_key) * 1000
        else:
            berv = 0.0
        # return the berv measurement (in m/s)
        return berv

    def rdb_columns(self):
        """
        Define the fits header columns names to add to the RDB file
        These should be references to keys in params

        :return:
        """
        # there are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_AIRMASS', 'KW_FILENAME', 'KW_DATE',
                    'KW_BERV', 'KW_TAU_H2O', 'KW_TAU_OTHERS',
                    'KW_DPRTYPE', 'KW_NITERATIONS',
                    'KW_SYSTEMIC_VELO', 'KW_WAVETIME', 'KW_WAVEFILE',
                    'KW_TLPDVH2O', 'KW_TLPDVOTR', 'KW_CDBWAVE', 'KW_OBJNAME',
                    'KW_RHOMB1', 'KW_RHOMB2', 'KW_CDEN_P', 'KW_SNRGOAL',
                    'KW_EXT_SNR', 'KW_BJD', 'KW_SHAPE_DX', 'KW_SHAPE_DY',
                    'KW_SHAPE_A', 'KW_SHAPE_B', 'KW_SHAPE_C', 'KW_SHAPE_D',
                    'KW_CCF_EW']
        # convert to actual keys (not references to keys)
        keys = []
        for drs_key in drs_keys:
            if drs_key in self.params:
                keys.append(self.params[drs_key])
            else:
                keys.append(drs_key)
        # return a numpy array
        return np.array(keys)

    def fix_lblrv_header(self, header: fits.Header) -> fits.Header:
        """
        Fix the LBL RV header

        :param header: fits.Header, the LBL RV fits file header

        :return: fits.Header, the updated LBL RV fits file header
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
            header[kw_ccf_ew] = 5.5 / mp.fwhm() * 1000
        # ---------------------------------------------------------------------
        # return header
        return header

    def get_rjd_value(self, header: fits.Header) -> float:

        """
        Get the rjd either from KW_MID_EXP_TIME or KW_BJD
        time returned is in MJD (not JD)

        :param header: fits.Header - the LBL rv header
        :return:
        """
        # get keys from params
        kw_mjdmid = self.params['KW_MID_EXP_TIME']
        kw_bjd = self.params['KW_BJD']
        # get mjdmid and bjd
        mid_exp_time = io.get_hkey(header, kw_mjdmid)
        bjd = io.get_hkey(header, kw_bjd)
        if isinstance(bjd, str):
            return float(mid_exp_time)
        else:
            # convert bjd to mjd
            bjd_mjd = Time(bjd, format='jd').mjd
            # return bjd in mjd
            return float(bjd_mjd)

    def get_plot_date(self, header: fits.Header):
        """
        Get the matplotlib plotting date

        :param header: fits.Header - the LBL rv header

        :return: float, the plot date
        """
        # get mjdate key
        kw_mjdate = self.params['KW_MJDATE']
        # get mjdate
        mjdate = io.get_hkey(header, kw_mjdate)
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
        binned['bands'] = bands
        binned['blue_end'] = blue_end
        binned['red_end'] = red_end
        binned['region_names'] = region_names
        binned['region_low'] = region_low
        binned['region_high'] = region_high
        # ---------------------------------------------------------------------
        # return this binning dictionary
        return binned

    def get_epoch_groups(self, rdb_table: Table
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given instrument this is how we define epochs
        returns the epoch groupings, and the value of the epoch for each
        row in rdb_table

        :param rdb_table: astropy.table.Table - the rdb table (source of epoch
                          information)

        :return: tuple, 1. the epoch groupings, 2. the value of the epoch for
                 each row of the rdb_table
        """
        # get the date col from params
        kw_date = self.params['KW_DATE']
        # get unique dates (per epoch)
        epoch_groups = np.unique(rdb_table[kw_date])
        # get the epoch values for each row of rdb_table
        epoch_values = np.array(rdb_table[kw_date])
        # return the epoch groupings and epoch values
        return epoch_groups, epoch_values


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

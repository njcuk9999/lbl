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
from typing import Any, Dict, List, Tuple, Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
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
        # deal with no object
        if self.params['OBJECT_TEMPLATE'] is None:
            raise LblException('OBJECT_TEMPLATE name must be defined')
        else:
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
        _, mask_hdr = io.load_fits(mask_file, kind='mask fits file')
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

    def ref_table_file(self, directory: str) -> Tuple[Union[str, None], bool]:
        """
        Make the absolute path for the ref_table file (if it exists)

        :param directory: str, the directory the file is located at

        :return: absolute path to ref_table file (if it exists) or None
        """
        # deal with no object
        if self.params['OBJECT_TEMPLATE'] is None:
            raise LblException('OBJECT_TEMPLATE name must be defined')
        else:
            objname = self.params['OBJECT_TEMPLATE']
        # set base name
        basename = 'ref_table_{0}.csv'.format(objname)
        # get absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        if not io.check_file_exists(abspath, required=False):
            # ref_table does not exist --> return None
            return abspath, False
        else:
            # return absolute path
            return abspath, True

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

    def get_lblrv_file(self, science_filename: str, directory: str
                       ) -> Tuple[Union[str, None], bool]:
        """
        Construct the LBL RV file name and check whether it exists

        :param science_filename: str, the science filename
        :param directory: str, the directory name for lbl rv files

        :return: tuple, 1. the lbl rv filename, 2. whether file exists on disk
        """
        # deal with no object
        if self.params['OBJECT_TEMPLATE'] is None:
            raise LblException('OBJECT_TEMPLATE name must be defined')
        else:
            tobjname = self.params['OBJECT_TEMPLATE']
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            sobjname = self.params['OBJECT_SCIENCE']
        # get science file basename
        science_basename = os.path.basename(science_filename).split('.fits')[0]
        # construct base name
        bargs = [science_basename, sobjname, tobjname]
        basename = '{0}_{1}_{2}_lbl.fits'.format(*bargs)
        # construct absolute path
        abspath = os.path.join(directory, basename)
        # check that this file exists
        if not io.check_file_exists(abspath, required=False):
            # ref_table does not exist --> return None
            return abspath, False
        else:
            # return absolute path
            return abspath, True

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
        log.logger.info(msg)
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
        log.logger.info(msg.format(*margs))
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
        # Question: should we deal with BERV = np.nan?
        if 'FP' not in self.params['OBJECT_SCIENCE']:
            berv = io.get_hkey(sci_hdr, hdr_key) * 1000
        else:
            berv = 0.0
        # return the berv measurement (in m/s)
        return berv

    def write_ref_table(self, ref_table: Dict[str, Any],
                        ref_filename: str, header: fits.Header,
                        outputs: Dict[str, Any]):
        """
        Write the reference table to file "filename"

        :param ref_table: dict, the reference table dictionary
        :param ref_filename: str, the reference table absolute path
        :param header: fits.Header, the header to write to primary extension
        :param outputs: dict, a dictionary of outputs from compute_rv function

        :return: None - write file
        """
        # ---------------------------------------------------------------------
        # add keys to header
        # ---------------------------------------------------------------------
        # add number of iterations
        header['ITE_RV'] = (outputs['NUM_ITERATIONS'],
                            'Num iterat to reach sigma accuracy')
        # add systemic velocity in m/s
        header['SYSTVELO'] = (outputs['SYSTEMIC_VELOCITY'],
                              'Systemic velocity in m/s')
        # add rms to photon noise ratio
        header['RMSRATIO'] = (outputs['RMSRATIO'], 'RMS vs photon noise')
        # add e-width of LBL CCF
        header['CCF_EW'] = (outputs['CCF_EW'], 'e-width of LBL CCF in m/s')
        # add the high-pass LBL width [km/s]
        header['HP_WIDTH'] = (outputs['HP_WIDTH'],
                              'high-pass LBL width in km/s')
        # add LBL version
        header['LBL_VERS'] = (__version__, 'LBL code version')
        # add LBL date
        header['LBLVDATE'] = (__date__, 'LBL version date')
        # add process date
        header['LBLPDATE'] = (Time.now().fits, 'LBL processed date')
        # ---------------------------------------------------------------------
        # Convert ref table dictionary to table
        # ---------------------------------------------------------------------
        table = Table()
        for col in ref_table:
            table[col] = np.array(ref_table[col])
        # ---------------------------------------------------------------------
        # save to fits file
        # ---------------------------------------------------------------------
        # log saving of file
        msg = 'Writing reference table to: {0}'
        margs = [ref_filename]
        log.logger.info(msg.format(*margs))
        # write to disk
        io.write_fits(ref_filename, data=[None, table],
                      header=[header, None], dtype=[None, 'table'])


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

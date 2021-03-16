#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPIRou instrument class here: instrument specific settings

Created on 2021-03-15

@author: cook
"""
from astropy.table import Table
import glob
import numpy as np
import os
import requests
from typing import List, Tuple, Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import default
from lbl.science import general


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = default.Instrument
LblException = base_classes.LblException

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
        # set function name
        func_name = __NAME__ + '.Spirou.override()'
        # set parameters to update
        self.params.set('INSTRUMENT', 'SPIROU', source=func_name)
        # define the default science input files
        self.params.set('INPUT_FILE', '*e2dsff*AB.fits', source=func_name)
        # define the mask
        self.params.set('REF_TABLE_FMT', 'csv', source=func_name)
        # define the HP width
        self.params.set('HP_WIDTH', 223, source=func_name)
        # define the SNR cut off threshold
        self.params.set('SNR_THRESHOLD', 10, source=func_name)
        # define wave coeff key in header
        self.params.set('KW_WAVECOEFFS', 'WAVE{0:04d}', source=func_name)
        # define wave num orders key in header
        self.params.set('KW_WAVEORDN', 'WAVEORDN', source=func_name)
        # define wave degree key in header
        self.params.set('KW_WAVEDEGN', 'WAVEDEGN', source=func_name)
        # define snr keyword
        self.params.set('KW_SNR', 'EXTSN035', source=func_name)


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

    def get_wave_solution(self, filename: str) -> np.ndarray:
        """
        Get a wave solution from a file (for SPIROU this is from the header)
        :param filename: str, the absolute path to the file - for spirou this
                         is a file with the wave solution in the header
        :return:
        """
        # load the science spectrum
        sci_data, sci_hdr = self.load_science(filename)

        # get header keys
        kw_wavecoeffs = self.params['KW_WAVECOEFFS']
        kw_waveordn = self.params['KW_WAVEORDN']
        kw_wavedegn = self.params['KW_WAVEDEGN']

        # ---------------------------------------------------------------------
        # get header
        data, header = io.load_fits(filename, 'wave fits file')
        # ---------------------------------------------------------------------
        # get the data shape
        nby, nbx = data.shape
        # get xpix
        xpix = np.arange(nbx)
        # ---------------------------------------------------------------------
        # get wave order from header
        waveordn = io.get_hkey(header, kw_waveordn, filename)
        wavedegn = io.get_hkey(header, kw_wavedegn, filename)
        # get the wave 2d list
        wavecoeffs = io.get_hkey_2d(header, key=kw_wavecoeffs,
                                    dim1=waveordn, dim2=wavedegn + 1,
                                    filename=filename)
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

    def load_bad_hdr_keys(self) -> Tuple[List[str], str]:
        """
        Load the bad values and bad key for spirou

        :return: tuple, 1. the list of bad values, 2. the bad key in
                 a file header to check against bad values
        """
        # set up googlesheet parameters
        URL_BASE = ('https://docs.google.com/spreadsheets/d/'
                    '{}/gviz/tq?tqx=out:csv&sheet={}')
        SHEET_ID = '1gvMp1nHmEcKCUpxsTxkx-5m115mLuQIGHhxJCyVoZCM'
        WORKSHEET = 0
        BAD_ODO_URL = URL_BASE.format(SHEET_ID, WORKSHEET)
        # fetch data
        data = requests.get(BAD_ODO_URL)
        tbl = Table.read(data.text, format='ascii')
        # get bad keys
        bad_values = tbl['ODOMETER'].astype(str)
        # define bad file header key
        bad_key = 'EXPNUM'
        # return the
        return bad_values, bad_key




# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selection of instrument functions

Created on 2021-03-15

@author: cook
"""
from astropy.io import fits
from astropy.table import Table
import numpy as np
from typing import Any, Dict, Tuple, Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.default.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# load classes
ParamDict = base_classes.ParamDict


# =============================================================================
# Define classes
# =============================================================================
class Instrument:
    params = ParamDict()

    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return 'Instrument[{0}]'.format(self.name)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _not_implemented(method):
        emsg = 'Must implement {0} in specific instrument class'
        raise NotImplemented(emsg.format(method))

    def mask_file(self, directory):
        """
        Make the absolute path for the mask file

        :param directory: str, the directory the file is located at

        :return: absolute path to mask file
        """
        _ = directory
        raise self._not_implemented('mask_file')

    def load_mask(self, filename: str) -> Table:
        """
        Load a mask

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self, filename
        return io.load_table(filename, kind='mask table')

    def template_file(self, directory: str):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = self, directory
        raise self._not_implemented('template_file')

    def load_template(self, filename: str) -> Table:
        """
        Load a template

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self, filename
        return io.load_table(filename, kind='template fits file')

    def blaze_file(self, directory: str):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = self, directory
        raise self._not_implemented('blaze_file')

    # complex blaze return
    BlazeReturn = Union[Tuple[np.ndarray, fits.Header], None]

    def load_blaze(self, filename: str):
        """
        Load a blaze file

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self, filename
        raise self._not_implemented('load_blaze')

    def get_mask_systemic_vel(self, mask_file: str) -> float:
        """
        Get the systemic velocity in m/s of the mask

        :param mask_file: the absolute path to the mask file

        :return: float, systemic velocity in m/s
        """
        raise self._not_implemented('get_mask_systemic_vel')

    def science_files(self, directory: str):
        """
        List the absolute paths of all science files

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = directory
        raise self._not_implemented('science_files')

    def load_blaze_from_science(self, sci_hdr: fits.Header,
                                calib_directory: str):
        """
        Load the blaze file using a science file header

        :param sci_hdr: fits.Header - the science file header
        :param calib_directory: str, the directory containing calibration files
                                (i.e. containing the blaze files)

        :return: None
        """
        _ = sci_hdr
        raise self._not_implemented('science_files')

    def load_science(self, filename: str) -> Tuple[np.ndarray, fits.Header]:
        """
        Load a science exposure

        Note data should be a 2D array (even if data is 1D)
        Treat 1D data as a single order?

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self
        return io.load_fits(filename, kind='science fits file')

    def ref_table_file(self, directory: str):
        """
        Make the absolute path for the ref_table file

        :param directory: str, the directory the file is located at

        :return: absolute path to ref_table file
        """
        _ = directory
        raise self._not_implemented('ref_table_file')

    def get_wave_solution(self, science_filename: Union[str, None] = None,
                          data: Union[np.ndarray, None] = None,
                          header: Union[fits.Header, None] = None):
        """
        Get a wave solution from a file

        :param science_filename: str, the absolute path to the file - for
                                 spirou this is a file with the wave solution
                                 in the header
        :param header: fits.Header, this is the header to use (if not given
                       requires filename to be set to load header)
        :param data: np.ndarray, this must be set along with header (if not
                     give we require filename to be set to load data)

        :return: np.ndarray, the wave map. Shape = (num orders x num pixels)
        """
        _ = science_filename, data, header
        raise self._not_implemented('get_wave_solution')

    def get_lblrv_file(self, science_filename: str, directory: str):
        """
        Construct the LBL RV file name and check whether it exists

        :param science_filename: str, the absolute path to the file
        :param directory: str, the directory to find lblrv file in

        :return:
        """
        _ = science_filename, directory
        raise self._not_implemented('get_wave_solution')

    def load_bad_hdr_keys(self):
        """
        Load the bad values and bad key for spirou

        :return: tuple, 1. the list of bad values, 2. the bad key in
                 a file header to check against bad values
        """
        raise self._not_implemented('load_bad_hdr_keys')

    def get_berv(self, sci_hdr: fits.Header):
        """
        Get the Barycenteric correction for the RV

        :param sci_hdr: fits.Header, the science header

        :return:
        """
        _ = sci_hdr
        raise self._not_implemented('get_berv')

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
        _ = ref_table, ref_filename, header, outputs
        raise self._not_implemented('get_berv')


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

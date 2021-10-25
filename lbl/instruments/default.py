#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selection of instrument functions

Created on 2021-03-15

@author: cook
"""
from astropy.io import fits
from astropy.table import Table
import glob
import numpy as np
import os
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
# get time from base
Time = base.AstropyTime
# load classes
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log


# =============================================================================
# Define classes
# =============================================================================
class Instrument:
    params: ParamDict = ParamDict()

    def __init__(self, name: str):
        """
        Default Instrument class - this should be inherited by an instrument
        class - not used by itself

        :param name: str, the name of the Instrument
        """
        self.name = name

    def __str__(self) -> str:
        return 'Instrument[{0}]'.format(self.name)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _not_implemented(method):
        """
        This class has methods that must be implemented in an Instrument
        subclass - raise a NotImplemented exception

        :param method: str, the method that needs to be implemented
        :return:
        """
        emsg = 'Must implement {0} in specific instrument class'
        raise NotImplemented(emsg.format(method))

    # -------------------------------------------------------------------------
    # Common instrument methods (should be overridden if instrument requires)
    # -------------------------------------------------------------------------
    def load_mask(self, filename: str) -> Table:
        """
        Load a mask

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self, filename
        return io.load_table(filename, kind='mask table')

    def load_template(self, filename: str, get_hdr: bool = False) -> Table:
        """
        Load a template

        :param filename: str, absolute path to filename
        :param get_hdr: bool, whether to get the head or not

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        _ = self, filename
        return io.load_table(filename, kind='template fits file',
                             get_hdr=get_hdr)

    def set_hkey(self, header: fits.Header, key: str, value: Any,
                 comment: Union[str, None] = None) -> fits.Header:
        """
        Set a header key (looking for key in params) and set it and its comment
        as necessary

        :param header:
        :param key:
        :param value:
        :param comment:
        :return:
        """

        # look for key in params
        if key in self.params:
            key = self.params[key]
            if key in self.params.instances:
                # get comment from Const.comment
                comment = self.params.instances[key].comment

        # assign value to header
        if comment is None:
            header[key] = (value, '')
        else:
            header[key] = (value, comment)
        # return header
        return header

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

    def ref_table_file(self, directory: str) -> Tuple[Union[str, None], bool]:
        """
        Make the absolute path for the ref_table file (if it exists)

        :param directory: str, the directory the file is located at

        :return: absolute path to ref_table file (if it exists) or None
        """
        # deal with no object template
        self._set_object_template()
        # set object name
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

    def get_lblrv_file(self, science_filename: str, directory: str
                       ) -> Tuple[Union[str, None], bool]:
        """
        Construct the LBL RV file name and check whether it exists

        :param science_filename: str, the science filename
        :param directory: str, the directory name for lbl rv files

        :return: tuple, 1. the lbl rv filename, 2. whether file exists on disk
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            sobjname = self.params['OBJECT_SCIENCE']
        # deal with no object template
        self._set_object_template()
        # set object template
        tobjname = self.params['OBJECT_TEMPLATE']
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

    def write_lblrv_table(self, ref_table: Dict[str, Any],
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
        header = self.set_hkey(header, 'KW_NITERATIONS',
                               value=outputs['NUM_ITERATIONS'])
        # add systemic velocity in m/s
        header = self.set_hkey(header, 'KW_SYSTEMIC_VELO',
                               value=outputs['SYSTEMIC_VELOCITY'])
        # add rms to photon noise ratio
        header = self.set_hkey(header, 'KW_RMS_RATIO',
                               value=outputs['RMSRATIO'])
        # add e-width of LBL CCF
        header = self.set_hkey(header, 'KW_CCF_EW', value=outputs['CCF_EW'])
        # add the high-pass LBL width [km/s]
        header = self.set_hkey(header, 'KW_HP_WIDTH', value=outputs['HP_WIDTH'])
        # add LBL version
        header = self.set_hkey(header, 'KW_VERSION', value=__version__)
        # add LBL date
        header = self.set_hkey(header, 'KW_VDATE', value=__date__)
        # add process date
        header = self.set_hkey(header, 'KW_PDATE', value=Time.now().fits)
        # add which lbl instrument was used
        header = self.set_hkey(header, 'KW_INSTRUMENT', value=self.name)
        # add the template velocity from CCF
        header = self.set_hkey(header, 'KW_MODELVEL',
                               value=outputs['MODEL_VELOCITY'])
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
        log.general(msg.format(*margs))
        # write to disk
        io.write_fits(ref_filename, data=[None, table],
                      header=[header, None], dtype=[None, 'table'])

    def get_lblrv_files(self, directory: str) -> np.ndarray:
        """
        Get all lbl rv files from directory for this object_science and
        object_template

        :param directory: str, the lbl rv directory absolute path

        :return: list of strs, the lbl rv files for this object_science and
                 object_template
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            sobjname = self.params['OBJECT_SCIENCE']
        # deal with no object template
        self._set_object_template()
        # set object template
        tobjname = self.params['OBJECT_TEMPLATE']
        # construct base name
        bargs = ['*', sobjname, tobjname]
        basename = '{0}_{1}_{2}_lbl.fits'.format(*bargs)
        # get absolute path
        abspath = os.path.join(directory, basename)
        # find all files
        files = glob.glob(abspath)
        # sort files in alphabetical order
        files = np.array(files)[np.argsort(files)]
        # return files
        return files

    def load_lblrv_file(self, filename: str) -> Tuple[Table, fits.Header]:
        """
        Load an LBL RV file

        :param filename: str, the LBL RV filename

        :return: tuple, 1. the LBL RV table, 2. the LBL RV header
        """
        _ = self
        # get fits bin table as astropy table
        table = io.load_table(filename, kind='lbl rv fits table')
        # get fits header
        header = io.load_header(filename, kind='lbl rv fits table')
        # retuurn table and header
        return table, header

    def get_lblrdb_files(self, directory: str
                         ) -> Tuple[str, str, str, str, str]:
        """
        Construct the LBL RDB absolute path and filenames

        :return:
        """
        # deal with no template set
        self._set_object_template()
        # construct base filename
        outargs = [self.params['OBJECT_SCIENCE'],
                   self.params['OBJECT_TEMPLATE'],
                   self.params['RDB_SUFFIX']]
        outname1 = 'lbl_{0}_{1}{2}.rdb'.format(*outargs)
        outname2 = 'lbl2_{0}_{1}{2}.rdb'.format(*outargs)
        outname3 = 'lbl_{0}_{1}{2}_drift.rdb'.format(*outargs)
        outname4 = 'lbl2_{0}_{1}{2}_drift.rdb'.format(*outargs)
        outname5 = 'drift.rdb'
        # construct absolute paths
        outpath1 = os.path.join(directory, outname1)
        outpath2 = os.path.join(directory, outname2)
        outpath3 = os.path.join(directory, outname3)
        outpath4 = os.path.join(directory, outname4)
        outpath5 = os.path.join(directory, outname5)
        # return outpath 1 + 2
        return outpath1, outpath2, outpath3, outpath4, outpath5

    def load_lblrdb_file(self, filename: str) -> Table:
        """
        Load the LBL rdb file

        :param filename: str, the LBL rdb file to load

        :return: Table, the LBL rdb astropy table
        """
        _ = self
        # load table
        table = io.load_table(filename, kind='LBL rdb fits table', fmt='rdb')
        # return table
        return table

    def science_template_subdir(self) -> str:
        """
        Create the object science / object template sub directory

        :return: str, the sub directory named by object science and object
                 template
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            sobjname = self.params['OBJECT_SCIENCE']
        # deal with no object template
        self._set_object_template()
        # set object template
        tobjname = self.params['OBJECT_TEMPLATE']
        # return sub directory
        return '{0}_{1}'.format(sobjname, tobjname)

    def _set_object_template(self):
        """
        Check that if OBJECT_TEMPLATE is not set, if it is not set
        then set it to OBJECT_SCIENCE

        :return: None - updates OBJECT_TEMPLATE if not set
        """
        # set function name
        func_name = __NAME__ + '.Spirou._set_object_template()'
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            objname = self.params['OBJECT_SCIENCE']
        # deal with no object
        if self.params['OBJECT_TEMPLATE'] is None:
            self.params.set('OBJECT_TEMPLATE', value=objname, source=func_name)

    def write_template(self, template_file: str, props: dict,
                       sci_hdr: fits.Header, sci_table: dict):
        """
        Write the template file to disk

        :param template_file: str, the file and path to write to
        :param props: dict, the template columns
        :param sci_table: dict, the science table in dictionary form
        :return:
        """
        # construct a new hdu list
        hdulist = fits.HDUList()
        # populate primary header
        header = fits.Header()
        # copy header from reference header
        header = io.copy_header(header, sci_hdr)
        # add custom keys
        header = self.set_hkey(header, 'KW_NTFILES', len(sci_table['FILENAME']))
        header = self.set_hkey(header, 'KW_VERSION', __version__)
        header = self.set_hkey(header, 'KW_VDATE', __date__)
        header = self.set_hkey(header, 'KW_PDATE', Time.now().iso)
        header = self.set_hkey(header, 'KW_INSTRUMENT',
                               self.params['INSTRUMENT'])
        # ---------------------------------------------------------------------
        # create main table
        table1 = Table()
        table1['wavelength'] = props['wavelength']
        table1['flux'] = props['flux']
        table1['eflux'] = props['eflux']
        table1['rms'] = props['rms']
        # ---------------------------------------------------------------------
        # construct table 2 - the science list
        table2 = Table()
        for key in sci_table:
            table2[key] = sci_table[key]
        # ---------------------------------------------------------------------
        # Save template to disk
        log.general('Saving template to file: {0}'.format(template_file))
        # ---------------------------------------------------------------------
        # write to file
        io.write_fits(template_file, data=[None, table1, table2],
                      header=[header, None, None],
                      dtype=[None, 'table', 'table'])

    def write_mask(self, mask_file: str, line_table: Table,
                   pos_mask: np.ndarray, neg_mask: np.ndarray,
                   sys_vel: float, template_hdr: fits.Header):
        """
        Write the mask (in lbl_mask) to disk

        :param mask_file: str, the mask path and default filename
        :param line_table: astropy table, the line table to add
        :param pos_mask: np.array, the positive weights mask
        :param neg_mask: np.array, the negative weights mask
        :param sys_vel: float, the systemic velocity for the object
        :param template_hdr: fits.Header, the template header (to be copied
                             to the template)

        :return:
        """
        # get the mask type for the default mask name
        mask_type = self.params['MASK_TYPE']
        # set up the three outputs
        masks = [pos_mask, neg_mask, np.ones_like(pos_mask, dtype=bool)]
        extensions = ['pos', 'neg', 'full']
        # ---------------------------------------------------------------------
        # loop around each file
        for it in range(len(masks)):
            # get mask
            mask = masks[it]
            # get filename
            new_mask_file = mask_file.replace(mask_type, extensions[it])
            # set up primary HDU
            header = fits.Header()
            # copy header from reference header
            header = io.copy_header(header, template_hdr)
            # add keys
            header = self.set_hkey(header, 'KW_SYSTEMIC_VELO', sys_vel * 1000)
            header = self.set_hkey(header, 'KW_VERSION', __version__)
            header = self.set_hkey(header, 'KW_VDATE', __date__)
            header = self.set_hkey(header, 'KW_PDATE', Time.now().iso)
            header = self.set_hkey(header, 'KW_INSTRUMENT',
                                   self.params['INSTRUMENT'])
            # log writing
            msg = 'Writing mask file to disk: {0}'
            log.general(msg.format(new_mask_file))
            # write to file
            io.write_fits(new_mask_file, data=[None, line_table[mask]],
                          header=[header, None],
                          dtype=[None, 'table'])

    # -------------------------------------------------------------------------
    # Methods that MUST be overridden by the child instrument class
    # -------------------------------------------------------------------------
    def mask_file(self, directory: str, required: bool = True):
        """
        Make the absolute path for the mask file

        :param directory: str, the directory the file is located at

        :return: absolute path to mask file
        """
        _ = directory
        raise self._not_implemented('mask_file')

    def template_file(self, directory: str, required: bool = True):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = self, directory
        raise self._not_implemented('template_file')

    def blaze_file(self, directory: str):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = self, directory
        raise self._not_implemented('blaze_file')

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

    def drift_condition(self, table_row: Table.Row):
        """
        Extra drift condition on a column to identify the correct reference
        file

        :param table_row: astropy.table.row.Row - the row of the table
                          to check against

        :return: True if reference file, False else-wise
        """
        _ = table_row
        raise self._not_implemented('drift_condition')

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

    def populate_sci_table(self, filename: str, tdict: dict,
                           sci_hdr: fits.Header, berv: float = 0.0):
        """
        Populate the science table

        :param sci_hdr:
        :param berv:
        :return:
        """
        _ = filename, tdict, sci_hdr, berv
        raise self._not_implemented('get_berv')

    def flag_calib(self, sci_hdr: fits.Header):
        """
        Flag a file as a calibration file

        :return:
        """
        _ = self, fits.Header
        raise self._not_implemented('flag_calib')

    def rdb_columns(self):
        """
        Define the fits header columns names to add to the RDB file

        :return:
        """
        _ = self
        raise self._not_implemented('rdb_columns')

    def fix_lblrv_header(self, header: fits.Header):
        """
        Fix the LBL RV header

        :param header: fits.Header, the LBL RV fits file header

        :return: fits.Header, the updated LBL RV fits file header
        """
        _ = header
        raise self._not_implemented('fix_lblrv_header')

    def get_rjd_value(self, header: fits.Header):

        """
        Get the rjd either from KW_MID_EXP_TIME or KW_BJD
        time returned is in MJD (not JD)

        :param header: fits.Header - the LBL rv header
        :return:
        """
        _ = header
        raise self._not_implemented('get_rjd_value')

    def get_plot_date(self, header: fits.Header):
        """
        Get the matplotlib plotting date

        :param header: fits.Header - the LBL rv header

        :return: float, the plot date
        """
        _ = header
        raise self._not_implemented('get_plot_date')

    def get_binned_parameters(self):
        """
        Defines a "binning dictionary" splitting up the array by:

        - bands  (in wavelength)  [bands / blue_end / red_end]

        - cross order (in pixels) [xbin_names / xbin_low / xbin_high]

        :return: dict, the binned dictionary
        """
        raise self._not_implemented('get_binned_parameters')

    def get_epoch_groups(self, rdb_table: Table):
        """
        For a given instrument this is how we define epochs
        returns the epoch groupings, and the value of the epoch for each
        row in rdb_table

        :param rdb_table: astropy.table.Table - the rdb table (source of epoch
                          information)

        :return: tuple, 1. the epoch groupings, 2. the value of the epoch for
                 each row of the rdb_table
        """
        _ = rdb_table
        # return the epoch groupings and epoch values
        raise self._not_implemented('get_epoch_groups')

    def find_inputs(self):
        """
        Find the input files for an instrument and copy them to the correct
        places

        :return:
        """
        # return the epoch groupings and epoch values
        raise self._not_implemented('find_inputs')

    def add_dict_list_value(self, store: Dict[str, Any], key: str,
                             value: Any) -> Dict[str, list]:
        """
        Add a value to a dictionary store

        :param store: dict, the storage dictionary to store lists in
        :param key: str, the key to add/update dictionary
        :param value: Any, the value to add to the dictionary key list

        :return: dict, the updated store
        """
        # if we don't have key add it
        if key not in store:
            store[key] = [value]
        # if we do append the list
        else:
            store[key].append(value)
        # return the dictionary
        return store

    def get_stellar_model_format_dict(self, params: base_classes.ParamDict
                                      ) -> dict:
        """
        Get the format dictionary for the stellar model URLS from
        the supplied header

        default uses the phoenix models from Goettigen

        :param params: ParamDict, the parameter dictionary of constants for
                       this instrument

        :return:
        """
        # set up format dictionary
        fdict = dict()
        # ---------------------------------------------------------------------
        # define the temperature range of the grids
        teff_min = 3000
        teff_max = 6000
        teff_step = 100
        teff_range = np.arange(teff_min, teff_max + teff_step, teff_step)
        # ---------------------------------------------------------------------
        # define a z range of the grids
        z_min = 0.0
        z_max = 0.0
        z_step = 1.0
        z_range = np.arange(z_min, z_max + z_step, z_step)
        # ---------------------------------------------------------------------
        # define a log g range of the grids
        logg_min = 0.0
        logg_max = 6.0
        logg_step = 0.5
        logg_range = np.arange(logg_min, logg_max + logg_step, logg_step)
        # ---------------------------------------------------------------------
        # define a alpha range of the grids
        alpha_min = 0.0
        alpha_max = 0.0
        alpha_step = 0.2
        alpha_range = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
        # ---------------------------------------------------------------------
        # get default teff
        if params['OBJECT_TEFF'] in ['None', '', None]:
            log.error('Teff is require. Please add OBJECT_TEFF to config')
            input_teff = np.inf
        else:
            input_teff = params['OBJECT_TEFF']
        # need to convert this to a closest teff
        teff = _get_closest(input_teff, teff_range)
        # ---------------------------------------------------------------------
        # get the closest log g
        logg = _get_closest(params['OBJECT_LOGG'], logg_range)
        # ---------------------------------------------------------------------
        # get the closest Fe/H
        zvalue = _get_closest(params['OBJECT_Z'], z_range)
        # ---------------------------------------------------------------------
        # get the  closest Fe/H
        alpha = _get_closest(params['OBJECT_ALPHA'], alpha_range)
        # ---------------------------------------------------------------------
        # add to format dictionary
        fdict['TEFF'] = '{0:05d}'.format(int(teff))
        fdict['LOGG'] = '{0:.2f}'.format(logg)
        fdict['ZVALUE'] = '{0:.1f}'.format(zvalue)
        if zvalue == 0:
            fdict['ZSTR'] = 'Z-0.0'
        else:
            fdict['ZSTR'] = 'Z{0:+.1f}'.format(zvalue)
        fdict['AVALUE'] = alpha
        if alpha != 0:
            fdict['ASTR'] = '.Alpha={0:+.2f}'.format(alpha)
        else:
            fdict['ASTR'] = ''
        # return the format dictionary
        return fdict


# =============================================================================
# worker functions
# =============================================================================
def _get_closest(value: float, values: np.ndarray):
    """
    Find the closest value to an array of values

    :param value: float, the value to find the closest of
    :param values: np.ndarray, the values in which to choose a closest value
    :return:
    """
    # find the position of the closest
    pos = np.argmin(abs(value - values))
    # return the closest value
    return values[pos]


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

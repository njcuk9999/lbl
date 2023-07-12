#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selection of instrument functions

Created on 2021-03-15

@author: cook
"""
import glob
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import EarthLocation

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp

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
        # extra parameters (specific to instrument) may not be required
        #   for all instruments but all should be here to know they are used
        #   by at least one child class
        self.orders: Optional[List[int]] = None
        self.norders: Optional[int] = None
        self.npixel: Optional[int] = None
        self.default_template_name: Optional[str] = None
        # extension of the science files
        self.science_ext = '.fits'
        # hd5 file definitions
        self.header_storekey: Optional[str] = None
        self.blaze_storekey: Optional[str] = None
        self.sci_storekey: Optional[str] = None
        self.sci_subkey: Optional[str] = None
        self.wave_storekey: Optional[str] = None
        self.sci_extension: Optional[int] = None
        self.blaze_extension: Optional[int] = None
        self.tcorr_extension: Optional[str] = None
        self.valid_suffices: Optional[List[str]] = None
        # define wave limits in nm
        self.wavemin: Optional[float] = None
        self.wavemax: Optional[float] = None

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

    def load_mask(self, filename: str) -> Table:
        """
        Load a mask

        :param filename: str, absolute path to filename

        :return: Table, the mask table
        """
        _ = self, filename
        # print progress
        log.general('Loaded mask table: {0}'.format(filename))
        # return mask table
        return io.load_table(filename, kind='mask table')

    def load_template(self, filename: str, get_hdr: bool = False
                      ) -> Union[Table, Tuple[Table, io.LBLHeader]]:
        """
        Load a template

        :param filename: str, absolute path to filename
        :param get_hdr: bool, whether to get the head or not

        :return: tuple, data (np.ndarray) and header (io.LBLHeader)
        """
        _ = self, filename
        return io.load_table(filename, kind='template fits file',
                             get_hdr=get_hdr)

    def set_hkey(self, header: Union[io.LBLHeader, fits.Header],
                 key: str, value: Any, comment: Union[str, None] = None
                 ) -> Union[io.LBLHeader, fits.Header]:
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

    def ref_table_file(self, directory: str,
                       mask_file: str) -> Tuple[Union[str, None], bool]:
        """
        Make the absolute path for the ref_table file (if it exists)

        :param directory: str, the directory the file is located at
        :param mask_file: str, the mask file path

        :return: absolute path to ref_table file (if it exists) or None
        """
        # deal with no object template
        self._set_object_template()
        # set object name
        mask_name = os.path.basename(mask_file).replace('.fits', '')
        # set base name
        basename = 'ref_table_{0}.csv'.format(mask_name)
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
        science_basename = os.path.basename(science_filename)
        science_basename = science_basename.split(self.science_ext)[0]
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
                          ref_filename: str, header_dict: io.LBLHeader,
                          outputs: Dict[str, Any]):
        """
        Write the reference table to file "filename"

        :param ref_table: dict, the reference table dictionary
        :param ref_filename: str, the reference table absolute path
        :param header_dict: io.LBLHeader, the header to write to primary
                            extension
        :param outputs: dict, a dictionary of outputs from compute_rv function

        :return: None - write file
        """
        header = header_dict.to_fits()
        # ---------------------------------------------------------------------
        # add keys to header
        # ---------------------------------------------------------------------
        # add number of iterations
        header = self.set_hkey(header, 'KW_NITERATIONS',
                               value=outputs['NUM_ITERATIONS'])
        # add quality flag
        header = self.set_hkey(header, 'KW_RESET_RV',
                               value=int(outputs['RESET_RV']))
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

    def load_lblrv_file(self, filename: str) -> Tuple[Table, io.LBLHeader]:
        """
        Load an LBL RV file

        :param filename: str, the LBL RV filename

        :return: tuple, 1. the LBL RV table, 2. the LBL RV header
        """
        _ = self
        # get fits bin table as astropy table
        table = io.load_table(filename, kind='lbl rv fits table')
        # get fits header
        header = self.load_header(filename, kind='lbl rv fits table')
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

    def write_rdb_fits(self, filename: str, rdb_data: Dict[str, Any]):
        """
        Write the rdb fits file to disk

        :param filename: str, the filename to save to (it extension is .rdb
                         changes to .fits)
        :param rdb_data: dict, the rdb data to add to fits file

        :return: None, writes fits file "filename"
        """
        # remove the rdb and add fits
        filename = filename.replace('.rdb', '.fits')
        # populate primary header
        header0 = fits.Header()
        # add custom keys
        header0 = self.set_hkey(header0, 'KW_VERSION', __version__)
        header0 = self.set_hkey(header0, 'KW_VDATE', __date__)
        header0 = self.set_hkey(header0, 'KW_PDATE', Time.now().iso)
        header0 = self.set_hkey(header0, 'KW_INSTRUMENT',
                                self.params['INSTRUMENT'])
        # construct the parameter table
        param_table = self.params.param_table()
        # set up data extensions
        datalist = [None, rdb_data['WAVE'],
                    rdb_data['DV'], rdb_data['SDV'],
                    rdb_data['D2V'], rdb_data['SD2V'],
                    rdb_data['D3V'], rdb_data['SD3V'],
                    rdb_data['RDB0'], rdb_data['RDB'],
                    param_table]
        headerlist = [header0, None, None, None,
                      None, None, None, None,
                      None, None, None]
        datatypelist = [None, 'image', 'image', 'image',
                        'image', 'image', 'image', 'image',
                        'table', 'table', 'table']
        name_list = [None, 'WAVE', 'DV', 'SDV',
                     'D2V', 'SD2V', 'D3V', 'SD3V',
                     'RDB0', 'RDB', 'PTABLE']
        # ---------------------------------------------------------------------
        # Save template to disk
        log.general('Saving tellu-cleaned file: {0}'.format(filename))
        # ---------------------------------------------------------------------
        # write to file
        io.write_fits(filename, data=datalist, header=headerlist,
                      dtype=datatypelist, names=name_list)

    def write_template(self, template_file: str, props: dict,
                       sci_hdict: io.LBLHeader, sci_table: dict):
        """
        Write the template file to disk

        :param template_file: str, the file and path to write to
        :param props: dict, the template columns
        :param sci_hdict: fits Header, an input file header to copy the header
                          from to the new template file
        :param sci_table: dict, the science table in dictionary form
        :return:
        """
        # convert hdict to fits header
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
        header = self.set_hkey(header, 'KW_TEMPLATE_COVERAGE',
                               value=props['template_coverage'])
        header = self.set_hkey(header, 'KW_TEMPLATE_BERVBINS',
                               value=props['total_nobs_berv'])
        header = self.set_hkey(header, 'KW_NTFILES', props['template_nobs'])
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
        header = self.set_hkey(header, 'KW_TAU_H2O',
                               props['pre_cleaned_exponent_water'])
        header = self.set_hkey(header, 'KW_TAU_OTHERS',
                               props['pre_cleaned_exponent_others'])
        # set image as pre_cleaned_flux
        image = props['pre_cleaned_flux']
        # adding extensions that are not the flux after telluric correction
        #   (error propagation, wavelength grid)
        datalist = [None, image]
        headerlist = [header, None]
        datatypelist = [None, 'image']
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

    def write_mask(self, mask_file: str, line_table: Table,
                   pos_mask: np.ndarray, neg_mask: np.ndarray,
                   sys_vel: float, template_hdict: io.LBLHeader):
        """
        Write the mask (in lbl_mask) to disk

        :param mask_file: str, the mask path and default filename
        :param line_table: astropy table, the line table to add
        :param pos_mask: np.array, the positive weights mask
        :param neg_mask: np.array, the negative weights mask
        :param sys_vel: float, the systemic velocity for the object
        :param template_hdict: io.LBLHeader, the template header (to be copied
                               to the template)

        :return:
        """
        # get data type
        data_type = self.params['DATA_TYPE']
        # get type of mask
        mask_type = self.params['{0}_MASK_TYPE'.format(data_type)]
        # set up the three outputs
        masks = [pos_mask, neg_mask, np.ones_like(pos_mask, dtype=bool)]
        extensions = ['pos', 'neg', 'full']
        # get the template header
        template_hdr = template_hdict.to_fits()
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

    @staticmethod
    def get_model_files(directory: str, url: str, model_dict: Dict[str, str]):
        """
        Chec/Get the model files from the model repository

        :param directory: str, the model directory to copies files to
        :param url: str, the url to get the files from
        :param model_dict: Dict[str, str] - the dictionary of files to get
                           from the model directory
        :return:
        """
        # loop around model files
        for key in model_dict:
            # get the filename
            filename = model_dict[key]
            # if url is not set or filename is not set return here - we have
            #   no default mask
            if url is None or filename is None:
                return
            # construct path to mask file
            model_file = os.path.join(directory, filename)
            # update url to include file
            fileurl = '{0}/{1}'.format(url, filename)
            # get file from url
            io.get_urlfile(fileurl, key, model_file, required=False)

    @staticmethod
    def copy_default_mask(model_directory: str, mask_directory: str,
                          filename: str):
        """
        Copy the default mask for this instrument to the mask directory

        :param model_directory: str, the model directory
        :param mask_directory: str, the mask directory
        :param filename: str, the filename to copy
        :return:
        """
        # set filename
        if filename is None:
            return
        # define input and output path
        in_path = os.path.join(model_directory, filename)
        out_path = os.path.join(mask_directory, filename)
        # deal with file already existing
        if os.path.exists(out_path):
            return
        # only copy if we have a file to copy
        if os.path.exists(in_path):
            shutil.copy(in_path, out_path)

    # -------------------------------------------------------------------------
    # Methods that MUST be overridden by the child instrument class
    # -------------------------------------------------------------------------
    def mask_file(self, model_directory: str, mask_directory: str,
                  required: bool = True):
        """
        Make the absolute path for the mask file

        :param model_directory: str, the directory the model is located at
        :param mask_directory: str, the directory the mask should be copied to
        :param required: bool, if True checks that file exists on disk

        :return: absolute path to mask file
        """
        _ = model_directory, mask_directory
        raise self._not_implemented('mask_file')

    def template_file(self, directory: str, required: bool = True):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at
        :param required: bool, if True checks that file exists on disk

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

    def load_blaze(self, filename: str, science_file: Optional[str] = None,
                   normalize: bool = True):
        """
        Load a blaze file

        :param filename: str, absolute path to filename
        :param science_file: str, a science file (to load the wave solution
                             from) we expect this science file wave solution
                             to be the wave solution required for the blaze
        :param normalize: bool, if True normalized the blaze per order

        :return: data (np.ndarray) or None
        """
        _ = self, filename, normalize
        raise self._not_implemented('load_blaze')

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
        sci_data = io.load_fits(science_file, kind='science fits file')
        sci_hdr = self.load_header(science_file, kind='science fits file')
        # return data and header
        return sci_data, sci_hdr

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

    def sort_science_files(self, science_files: List[str]) -> List[str]:
        """
        Sort science files (instrument specific)

        :param science_files: list of strings - list of science files

        :return: list of strings - sorted list of science files
        """
        # default, don't sort just return
        return science_files

    def load_blaze_from_science(self, science_file: str,
                                sci_image: np.ndarray,
                                sci_hdr: io.LBLHeader, calib_directory: str,
                                normalize: bool = True):
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
        _ = sci_image, sci_hdr, calib_directory, normalize
        raise self._not_implemented('science_files')

    def no_blaze_corr(self, sci_image: np.ndarray, sci_wave: np.ndarray):
        """
        If we do not have a blaze we need to create an artificial one so that
        the s1d has a proper weighting

        :param sci_image: the science image (will be unblazed corrected)
        :param sci_wave: the wavelength solution for the science image

        :return: Tuple, 1. the unblazed science_image, 2. the artifical blaze
        """
        _ = sci_image, sci_wave
        raise self._not_implemented('no_blaze_corr')

    def get_wave_solution(self, science_filename: Optional[str] = None,
                          data: Optional[np.ndarray] = None,
                          header: Optional[io.LBLHeader] = None):
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
        _ = science_filename, data, header
        raise self._not_implemented('get_wave_solution')

    def get_sample_wave_grid(self, calib_dir: str, science_file: str):
        """
        Get the sample wave grid
        """
        # get the sample wave grid filename
        sample_wavegrid_file = self.params['SAMPLE_WAVE_GRID_FILE']
        # get the wave grid path
        save_wavegrid_path = os.path.join(calib_dir, sample_wavegrid_file)
        # check if wave grid exists - if it does just load it
        if os.path.exists(save_wavegrid_path):
            # print loading
            msg = 'Loading sample wavegrid: {0}'
            log.general(msg.format(save_wavegrid_path))
            # return the wave grid
            wavegrid = io.load_fits(save_wavegrid_path)
            return wavegrid
        # else get it from the wave solution
        else:
            # print loading
            msg = 'Creating sample wavegrid from {0}'
            log.general(msg.format(science_file))
            # get wave grid from the supplied filename
            wavegrid = self.get_wave_solution(science_file)
            header_dict = self.load_header(science_file)
            # convert header to fits
            header = header_dict.to_fits()
            # write the file to the calibration directory
            io.write_fits(save_wavegrid_path, data=[None, wavegrid],
                          header=[header, None], dtype=[None, 'image'])
            # return the wave grid
            return wavegrid

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

    def get_berv(self, sci_hdr: io.LBLHeader):
        """
        Get the Barycenteric correction for the RV

        :param sci_hdr: io.LBLHeader, the science header

        :return:
        """
        _ = sci_hdr
        raise self._not_implemented('get_berv')

    def populate_sci_table(self, filename: str, tdict: dict,
                           sci_hdr: io.LBLHeader, berv: float = 0.0):
        """
        Populate the science table

        :param filename: str, the filename of the science image
        :param tdict: dictionary, the storage dictionary for science table
                      can be empty or have previous rows to append to
        :param sci_hdr: fits Header, the header of the science image
        :param berv: float, the berv value to add to storage dictionary

        :return: dict, a dictionary table of the science parameters
        """
        _ = filename, tdict, sci_hdr, berv
        raise self._not_implemented('get_berv')

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
        mcond2 = start in [None, 'None', '']
        mcond3 = end in [None, 'None', '']
        # return all files if this is the case
        if mcond2 and mcond3:
            return science_files
        # filtering files
        log.general('Filtering {0} files...'.format(self.params['DATA_TYPE']))
        # storage
        keep_files = []
        # loop around science files
        for science_file in tqdm(science_files):
            # load science file header
            sci_hdr = self.load_header(science_file)
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
            if cond2 and cond3:
                keep_files.append(science_file)
            # -----------------------------------------------------------------
        # if we have no files break here
        if len(keep_files) == 0:
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
            eargs = [start, end]
            # raise exception - we need some files to make a template!
            raise base_classes.LblException(emsg.format(*eargs))
        else:
            msg = ('Object is classified as science. Found {1} files'
                   ' ignoring {2} other files')
            margs = [len(keep_files),
                     len(science_files) - len(keep_files)]
            log.info(msg.format(*margs))
        # return only files with DPRTYPE same in both fibers
        return keep_files

    def rdb_columns(self):
        """
        Define the fits header columns names to add to the RDB file

        :return:
        """
        _ = self
        raise self._not_implemented('rdb_columns')

    def fix_lblrv_header(self, header: io.LBLHeader):
        """
        Fix the LBL RV header

        :param header: io.LBLHeader, the LBL RV fits file header

        :return: io.LBLHeader, the updated LBL RV fits file header
        """
        _ = header
        raise self._not_implemented('fix_lblrv_header')

    def get_rjd_value(self, header: io.LBLHeader):

        """
        Get the rjd either from KW_MID_EXP_TIME or KW_BJD
        time returned is in MJD (not JD)

        :param header: io.LBLHeader - the LBL rv header
        :return:
        """
        _ = header
        raise self._not_implemented('get_rjd_value')

    def get_plot_date(self, header: io.LBLHeader):
        """
        Get the matplotlib plotting date

        :param header: io.LBLHeader - the LBL rv header

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

    def modify_tellu_filename(self, filename: str) -> str:
        """
        Modifier for the telluric filename (default is to do nothing)

        :param filename: str, the filename to modify

        :return: str, the modified filename
        """
        return filename

    def get_uniform_binned_parameters(self, binned: Dict[str, list]
                                      ) -> Dict[str, list]:
        """
        Define "magic" binned regions from starting wavelength to end wavelength
        (defined by COMPIL_WAVE_MIN and COMPIL_WAVE_MAX)

        These are binned by wavelength into COMPIL_NUM_MAGIC_BANDS number of
        bins

        If COMPIL_ADD_MAGIC_BANDS is False this function does not add any
        magic bins

        :param binned: dict, the binned dictionary from get_binned_parameters
        :return: dict, the updated binned directory (if COMPIL_ADD_MAGIC_BANDS)
        """
        # get the pre-defined start and end wavelenghts
        wave0 = self.params['COMPIL_WAVE_MIN']
        wave1 = self.params['COMPIL_WAVE_MAX']
        # whether to use magic bins
        use_magic = self.params['COMPIL_ADD_UNIFORM_WAVEBIN']
        # get the number of bins to use
        nbins = self.params['COMPIL_NUM_UNIFORM_WAVEBIN']
        # ---------------------------------------------------------------------
        # if we aren't using magic bins return here
        if not use_magic:
            return binned
        # ---------------------------------------------------------------------
        # work out the
        logwaveratio = np.log(wave1 / wave0)
        # redefining wave1 to have a round number of velocity bins
        # get the positions for "magic length"
        plen_magic = np.arange(nbins + 1)
        # define the magic grid to use in ccf
        magic_grid = np.exp((plen_magic / nbins) * logwaveratio) * wave0
        # work out the mean magic grid positions
        mean_magic_grid = 0.5 * (magic_grid[:-1] + magic_grid[1:])
        # loop around mean magic grid (one shorter than magic grid)
        for v_it, vel in enumerate(mean_magic_grid):
            # contstruct band name
            band_name = '{0}nm'.format(int(vel))
            # append the band name
            binned['bands'].append(band_name)
            # blue end is the nth element in magic grid
            binned['blue_end'].append(magic_grid[v_it])
            # red end is the nth+1 element in magic grid
            binned['red_end'].append(magic_grid[v_it + 1])
            # make sure we do not use regions for magic binned parameters
            binned['use_regions'].append(False)
        # finally return the updated binned dictionary
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
        kw_mjd = self.params['KW_MJDATE']
        # get the earth location parameter
        earth_location = self.params['EARTH_LOCATION']
        # deal with not having EARTH_LOCATION (backwards compatibility)
        if earth_location is None:
            emsg = ('EARTH_LOCATION is not defined in the instrument ')
            raise base_classes.LblException(emsg)
        # try to get location of site from astropy
        try:
            loc = EarthLocation.of_site(earth_location)
        except Exception as e:
            emsg = ('Problem with EARTH_LOCATION {0} in '
                    'astropy.coordinates.EarthLocation\n\tError {1}: {2}')
            eargs = [earth_location, type(e), str(e)]
            raise base_classes.LblException(emsg.format(*eargs))
        # get local time at midnight
        epoch_values = mp.bin_by_time(loc.lon.value, rdb_table[kw_mjd],
                                      day_frac=0)
        # get the unique epoch groups
        epoch_groups = np.unique(epoch_values)
        # return the epoch groupings and epoch values
        return epoch_groups, epoch_values

    def find_inputs(self):
        """
        Find the input files for an instrument and copy them to the correct
        places

        :return:
        """
        # return the epoch groupings and epoch values
        raise self._not_implemented('find_inputs')

    @staticmethod
    def add_dict_list_value(store: Dict[str, Any], key: str,
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

    @staticmethod
    def get_stellar_model_format_dict(params: base_classes.ParamDict) -> dict:
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
            emsg = 'Teff is require. Please add OBJECT_TEFF to config'
            raise LblException(emsg)
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

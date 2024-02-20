#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import copy
import itertools
import os
import warnings
from collections import UserDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import wget
from astropy.io import fits
from astropy.table import Table

from lbl.core import base
from lbl.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
LblException = base_classes.LblException
log = base_classes.log
# set forbidden header keys (not to be copied)
FORBIDDEN_KEYS = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                  'EXTEND', 'COMMENT', 'CRVAL1', 'CRPIX1', 'CDELT1',
                  'CRVAL2', 'CRPIX2', 'CDELT2', 'BSCALE', 'BZERO',
                  'PHOT_IM', 'FRAC_OBJ', 'FRAC_SKY', 'FRAC_BB',
                  'NEXTEND', '', 'HISTORY', 'XTENSION']


# =============================================================================
# Define classes
# =============================================================================
class LBLHeader(UserDict):
    def __init__(self, *arg, **kw):
        """
        Construct the case insensitive dictionary class
        :param arg: arguments passed to dict
        :param kw: keyword arguments passed to dict
        """
        # set function name
        # _ = display_func('__init__', __NAME__, self.class_name)
        # super from dict
        super(LBLHeader, self).__init__(*arg, **kw)
        # add storage for comments
        self.comments = dict()
        # add a filename (could be None)
        self.filename = None

    def __getitem__(self, key: str) -> object:
        """
        Method used to get the value of an item using "key"
        used as x.__getitem__(y) <==> x[y]
        where key is case insensitive

        :param key: string, the key for the value returned (case insensitive)

        :type key: str

        :return value: object, the value stored at position "key"
        """
        # deal with hierarchal keys
        if key not in self.data and key.startswith('HIERARCH '):
            return self.data[key[len('HIERARCH '):]]
        # return from supers dictionary storage
        return self.data[key]

    def __setitem__(self, key: str, value: Any, comment: str = None):
        """
        Sets an item wrapper for self[key] = value
        :param key: string, the key to set for the parameter
        :param value: object, the object to set (as in dictionary) for the
                      parameter

        :type key: str
        :type value: object

        :return: None
        """
        # deal with tuple (value, comment)
        if isinstance(value, tuple) and len(value) == 2:
            value, comment = value
        # deal with setting / updating comments
        if comment is not None:
            self.comments[key] = comment
        elif key not in self.comments:
            self.comments[key] = ''
        # then do the normal dictionary setting
        self.data[key] = value

    def __delitem__(self, key: str):
        """
        Deletes the "key" from CaseInsensitiveDict instance, case insensitive

        :param key: string, the key to delete from ParamDict instance,
                    case insensitive
        :type key: str

        :return None:
        """
        # deal with comments
        if key in self.comments:
            del self.comments[key]
        # delete key from keys
        del self.data[key]

    def __contains__(self, key: str) -> object:
        """
        True if the dictionary has the specified key, else False.

        :param key: string, the key for the value returned (case insensitive)

        :type key: str

        :return value: object, the value stored at position "key"
        """
        # deal with hierarchal keys
        if key not in self.data and key.startswith('HIERARCH '):
            key = key[len('HIERARCH '):]

        # Return True if key is in data
        if key in self.data:
            return True
        else:
            return False

    def __str__(self) -> str:
        """
        Return the string representation of the class
        :return: str, the string representation
        """
        # storage for string
        string = ''
        # loop around keys
        for key in self.keys():
            # get value
            value = self[key]
            # get comment
            comment = self.comments[key]
            # add to string
            string += '{0} = {1}\t\t\t\t/ {2}\n'.format(key, value, comment)
        # return string
        return string

    def __repr__(self) -> str:
        """
        Return the string representation of the class
        :return: str, the string representation
        """
        return self.__str__()

    @classmethod
    def from_fits(cls, header: fits.Header,
                  filename: Optional[str] = None) -> 'LBLHeader':
        """
        Construct a LBLHeader from a fits file

        :param header: fits.Header, the loaded fits header (astro.io.FitsHeader)
        :return: LBLHeader, the header
        """
        new = cls()
        # loop around keys and add them to the header dictionary
        for key in header:
            new[key] = copy.deepcopy(header[key])
        # add comments
        new.comments = dict()
        for key in header:
            new.comments[key] = copy.deepcopy(header.comments[key])
        # set filename
        new.filename = filename
        # construct LBLHeader
        return new

    @classmethod
    def from_store(cls, storekey: str, filename: str) -> 'LBLHeader':
        # construct LBLHeader
        return hdf_to_header(storekey, filename)

    def to_fits(self) -> fits.Header:
        header = fits.Header()
        # loop around keys and add them to the header dictionary
        for key in list(self.keys()):
            if len(key) > 8 and 'HIERARCH' not in key:
                outkey = 'hierarch ' + key
            else:
                outkey = key
            # exceptionally long keys we have to ignore
            if len(outkey) > 40:
                continue
            # cannot propagate these keys
            if key in ['COMMENT', 'HISTORY']:
                continue
            # Add key to dictionary
            header[outkey] = (self.data[key], self.comments[key])
        # return header
        return header

    def get_hkey(self,  key: Union[str, List[str]],
                 filename: Union[str, None] = None,
                 required: bool = True,
                 dtype: Any = None) -> Any:
        # deal with no filename
        if filename is None:
            filename = self.filename
            if filename is None:
                filename = 'Unknown'
        # ---------------------------------------------------------------------
        # deal with a list of keys (test all one by one for the correct key)
        if isinstance(key, list):
            drskey = ''
            for key_it in key:
                if key_it in self.data:
                    if self.data[key_it] != ['None', '', None]:
                        drskey = key_it
                        break
                if key_it[len('HIERARCH '):] in self.data:
                    drskey = key_it[len('HIERARCH '):]
                    break
            if len(drskey) == 0:
                emsg = 'Cannot find keys: {0} in header'
                raise LblException(emsg.format(' or '.join(key)))
        else:
            drskey = str(key)
        # ---------------------------------------------------------------------
        # deal with hierarch keys being removed
        found_key = False

        if drskey not in self.data:
            # if drs key starts with HIERARCH then try to remove it
            #   as some keys have hierarch removed
            if drskey.startswith('HIERARCH '):
                sdrskey = drskey[len('HIERARCH '):]
                if sdrskey in self.data:
                    drskey = sdrskey
                    found_key = True
        else:
            found_key = True
        # ---------------------------------------------------------------------
        # test for key in header
        if found_key:
            try:
                if dtype is None:
                    return self.data[drskey]
                else:
                    try:
                        return dtype(self.data[drskey])
                    except Exception as e:
                        emsg = ('Cannot convert key {0} to {1} from header: {2}'
                                '\n\t{3}: {4}')
                        eargs = [drskey, str(dtype), filename, type(e), str(e)]
                        raise LblException(emsg.format(*eargs))
            except Exception as e:
                emsg = 'Cannot use key {0} from header: {1} \n\t{2}: {3}'
                eargs = [drskey, filename, type(e), str(e)]
                raise LblException(emsg.format(*eargs))
        elif not required:
            return None
        else:
            emsg = 'Key {0} not found in header: {1}'
            eargs = [drskey, filename]
            raise LblException(emsg.format(*eargs))

    def get_hkey_2d(self, key: str,
                    dim1: int, dim2: int, dtype: Type = float,
                    filename: Union[str, None] = None) -> np.ndarray:
        """
        Get a 2D list from the header

        :param key: str, the key with formatting to load i.e. XXX{number:04d}
                   where number = (row number * number of columns) + column number
                   where column number = dim2 and row number = range(0, dim1)
        :param dim1: int, the number of elements in dimension 1
                     (number of rows)
        :param dim2: int, the number of columns in dimension 2
                     (number of columns)
        :param dtype: type, the type to force the data to be (i.e. float, int)
        :param filename: str, the filename of the header (for error reporting)

        :return: np.ndarray, the value array, shape=(dim1, dim2)
        """
        # deal with no filename
        if filename is None:
            filename = self.filename
            if filename is None:
                filename = 'Unknown'
        # test for key in header
        if dtype is None:
            dtype = str
        # test for hierarch key (and remove it)
        if 'HIERARCH' in key:
            key2 = key[len('HIERARCH '):]
        else:
            key2 = None
        # create 2d list
        values = np.zeros((dim1, dim2), dtype=dtype)
        # loop around the 2D array
        dim1, dim2 = values.shape
        for it in range(dim1):
            for jt in range(dim2):
                # construct the key name
                keyname = _test_for_formatting(key, it * dim2 + jt)
                # deal with HIERARCH keys
                if keyname not in self.data and key2 is not None:
                    keyname = _test_for_formatting(key2, it * dim2 + jt)
                # try to get the values
                try:
                    # set the value
                    values[it][jt] = dtype(self.data[keyname])
                except KeyError:
                    emsg = 'Cannot find key {0} in header: {1}'
                    eargs = [keyname, filename]
                    raise LblException(emsg.format(*eargs))
        # return values
        return values

    def filter_by_hkey(self, key: str,
                       values: Union[List[str], str]) -> bool:
        """
        Filter by a header key (returns True if valid)

        :param key: str, the key in the header to check
        :param values: a list of string values to check against the header value
        :return:
        """
        # deal with key not in header
        if key not in self.data:
            return False
        # deal with non list values
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
        # loop around values
        for value in values:
            if str(self.data[key]) == str(value):
                return True
        # if we get here return False
        return False


# =============================================================================
# Define functions
# =============================================================================
def check_file_exists(filename: str, required: bool = True) -> bool:
    """
    Check if a file exists

    :param filename: str, the filename
    :param required: bool, if required raise an error on not existing
    :return:
    """
    if os.path.exists(filename):
        return True
    elif required:
        emsg = 'File {0} cannot be found'
        eargs = [filename]
        raise LblException(emsg.format(*eargs))
    else:
        return False


def check_directory(directory: str) -> str:
    """
    Checks 1. if directory exists 2. if directory is a directory

    :param directory: str, the directory to check

    :return: str, returns the directory after checking
    """
    # make directory a real path
    directory = os.path.realpath(directory)
    # check that directory path exists
    if not os.path.exists(directory):
        emsg = 'Directory "{0}" does not exist'
        eargs = [directory]
        raise LblException(emsg.format(*eargs))
    # check that directory is actually a directory
    if not os.path.isdir(directory):
        emsg = 'Directory "{0}" does not exist'
        eargs = [directory]
        raise LblException(emsg.format(*eargs))
    # return directory
    return directory


def make_dir(path: str, directory: str, kind: str,
             subdir: Union[str, None] = None,
             verbose: bool = True) -> str:
    """
    Make a directory path will be path/directory/subdir or path/directory

    :param path: str, path directory will be created in
    :param directory: str, directory to create
    :param kind: str, what is the directory (for error message)
    :param subdir: str or None, if set adds a sub directory to path
    :param verbose: bool, if True prints when directory exists

    :return: str, the absolute path created
    """

    # construct absolute path
    if subdir is None:
        abspath = Path(path).joinpath(directory)
    else:
        abspath = Path(path).joinpath(directory, subdir)
    # check if directory exists
    if abspath.exists():
        if verbose:
            msg = '{0} directory exists (Path={1})'
            margs = [kind, str(abspath)]
            log.general(msg.format(*margs))
        # return absolute path directory
        return str(abspath)
    # else try to create directory
    else:
        try:
            # make directory path
            if subdir is None:
                abspath.mkdir()
            elif abspath.parent.exists():
                abspath.mkdir()
            else:
                abspath.parent.mkdir()
                abspath.mkdir()
            # return absolute path directory
            return str(abspath)
        except Exception as e:
            emsg = 'Cannot create {0} directory. Path={1} \n\t{2}: {3}'
            eargs = [kind, str(abspath), type(e), str(e)]
            raise LblException(emsg.format(*eargs))


def find_files(path_list: List[Path],
               prefix: Optional[str] = None,
               suffix: Optional[str] = None,
               contains: Optional[str] = None,
               hkeys: Optional[Dict[str, Union[List[str], str]]] = None,
               use_tqdm: bool = True) -> List[str]:
    # get tqdm
    tqdm = base.tqdm_module(use_tqdm, log.console_verbosity)
    # storage for valid files
    valid_files = []
    # loop around files
    for filename in tqdm(path_list):
        # assume file is valid at start
        valid = True
        # deal with prefix
        if prefix not in [None, 'None', '', 'Null']:
            if not str(filename).startswith(prefix):
                continue
        # deal with suffix
        if suffix not in [None, 'None', '', 'Null']:
            if not str(filename).endswith(suffix):
                continue
        # deal with contains
        if contains not in [None, 'None', '', 'Null']:
            if contains not in str(filename):
                continue
        # deal with header keys
        if hkeys is not None:
            # skip non-fits file
            if not str(filename).endswith('.fits'):
                continue
            # load header
            hdr = load_header(str(filename), 'fits file')
            hdr = LBLHeader.from_fits(hdr)

            valid = True
            # loop around hkeys
            for hkey in hkeys:
                # check value
                valid &= hdr.filter_by_hkey(hkey, hkeys[hkey])
        # add to valid files if we have got to here
        if valid:
            valid_files.append(filename)

    return valid_files


def clean_directory(path: str, logmsg: bool = True,
                    dir_suffix: str = '',
                    include_files: Optional[List[str]] = None):
    """
    Remove all files from a directory

    :param path: str, path to clean
    :param logmsg: bool, if True log cleaning
    :param dir_suffix: str, if not '' only clean directories that end with this
    :param include_files: list of str, if not None only clean files in this list

    :return: None, removes files
    """
    # if directory does not exist do not clean
    if not os.path.exists(path):
        return
    # log cleaning
    if len(dir_suffix) == 0:
        log.general('Cleaning directory {0}'.format(path))
    else:
        msg = 'Cleaning directory {0} --suffix={1}'
        log.general(msg.format(path, dir_suffix))
    # get all files in path that match filter
    files = []
    for _root, _dirs, _files in os.walk(path):
        # dir_suffix
        if _root.endswith(dir_suffix):
            for filename in _files:
                if include_files is not None:
                    if filename not in include_files:
                        continue
                files.append(os.path.join(_root, filename))

    # loop around files
    for filename in files:
        # if filename is a file then remove it
        if os.path.isfile(filename):
            # noinspection PyBroadException
            try:
                if logmsg:
                    log.general('\t\tRemoving file: {0}'.format(filename))
                os.remove(filename)
            except Exception as _:
                pass


# =============================================================================
# Define fits functions
# =============================================================================
def load_fits(filename: str,
              kind: Union[str, None] = None,
              extnum: Optional[int] = None,
              extname: Optional[str] = None) -> np.ndarray:
    """
    Standard way to load fits files

    :param filename: str, the filename
    :param kind: the kind (for error message)
    :param extnum: int, the extension number (if not given uses first)
    :param extname: the extension name (if not given uses extnum)

    :return: tuple, the data and the header
    """
    # deal with no kind
    if kind is None:
        kind = 'fits file'
    # try to load fits file
    try:
        if extnum is not None:
            data = fits.getdata(filename, ext=extnum)
        elif extname is not None:
            data = fits.getdata(filename, extname=extname)
        else:
            data = fits.getdata(filename)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    return np.array(data)


def load_header(filename: str,
                kind: Union[str, None] = None,
                extnum: Optional[int] = None,
                extname: Optional[str] = None) -> fits.Header:
    """
    Standard way to load fits file header only

    :param filename: str, the filename
    :param kind: the kind (for error message)
    :param extnum: int, the extension number (if not given uses first)
    :param extname: the extension name (if not given uses extnum)

    :return: FITS header, the header
    """
    # deal with no kind
    if kind is None:
        kind = 'fits file'
    # try to load fits file
    try:
        if extnum is not None:
            header = fits.getheader(filename, ext=extnum)
        elif extname is not None:
            header = fits.getheader(filename, extname=extname)
        else:
            header = fits.getheader(filename)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    return header.copy()


def copy_header(header1: fits.Header, header2: fits.Header) -> fits.Header:
    """
    Copy all non-forbidden header keys from header2 to header1

    :param header1: header to update
    :param header2: header to copy from

    :return: fits.Header, header1 updated with keys from header2
    """
    # loop around all keys from header 2
    for key in header2:
        # skip forbidden keys
        if key in FORBIDDEN_KEYS:
            continue
        # get value and comment from header2
        value = copy.deepcopy(header2[key])
        comment = str(header2.comments[key])
        # push into header 1
        with warnings.catch_warnings(record=True) as _:
            header1[key] = (value, comment)
    # return header 1
    return header1


# complex write inputs
FitsData = Union[List[Union[Table, np.ndarray, None]], Table, np.ndarray, None]
FitsHeaders = Union[LBLHeader, List[Union[LBLHeader, None]],
                    List[Union[fits.Header, None]], fits.Header, None]
FitsDtype = Union[List[Union[str, None]], str, None]


def write_fits(filename: str, data: FitsData = None,
               header: FitsHeaders = None, dtype: FitsDtype = None,
               names: Union[List[str], str, None] = None):
    """
    Write a fits file to disk

    :param filename: str, the filename to write to
    :param data: list of tables/images (or a single table/image) to write to
                 fits file
    :param header: list of headers (or single header) to write to fits file
    :param dtype: list of 'image' or 'table' to identify the data types for
                  writing to fits file
    :param names: list of names or None, if set this sets the EXTNAMEs for each
                  extension of the fits file

    :return: None, writes fits file to 'filename'
    """
    # -------------------------------------------------------------------------
    # deal with non list data
    if data is None:
        data = [None]
        names = [None]
    elif isinstance(data, (Table, np.ndarray)):
        data = [data]
        names = [names]
    # -------------------------------------------------------------------------
    # deal with non list header
    if header is None:
        header = [None] * len(data)
    elif isinstance(header, LBLHeader):
        header = [header.to_fits()]
    elif isinstance(header, fits.Header):
        header = [header]
    # deal with no names
    if names is None:
        names = [None] * len(data)
    # -------------------------------------------------------------------------
    # fix header unicode
    for it in range(len(header)):
        # deal with no header
        if header[it] is None:
            continue
        # loop around headers
        for key in header[it]:
            if isinstance(header[it][key], str):
                value = header[it][key].encode('ascii', 'ignore').decode('ascii')
                header[it][key] = value
    # -------------------------------------------------------------------------
    # deal with non list dtype
    if dtype is None:
        dtype = [None] * len(data)
    elif isinstance(dtype, str):
        dtype = [dtype]
    # -------------------------------------------------------------------------
    # deal with wrong length of headers
    if len(header) != len(data):
        emsg = 'header (len={0}) must be same length as data (len={1})'
        eargs = [len(header), len(data)]
        raise LblException(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # deal with wrong length of dtypes
    if len(dtype) != len(data):
        emsg = 'dtype (len={0}) must be same length as data (len={1})'
        eargs = [len(dtype), len(data)]
        raise LblException(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # push primary header
    hdu0 = fits.PrimaryHDU()
    # add data to primary header
    if data[0] is not None:
        # shouldn't really put any data here - but if we are make sure the data
        #   is a numpy array
        hdu0.data = np.array(data[0])
        # print warning
        wmsg = 'Do not put data in primary extension (File: {0})'
        wargs = [filename]
        log.warning(wmsg.format(*wargs))
    # add primary header
    if header[0] is not None:
        # must append keys for primary extension
        for key in header[0]:
            value = header[0][key]
            comment = header[0].comments[key]
            # skip keys already present
            if key in hdu0.header:
                continue
            # skip forbidden keys
            if key in FORBIDDEN_KEYS:
                continue
            # skip comments
            # noinspection PyProtectedMember
            if isinstance(value, fits.header._HeaderCommentaryCards):
                continue
            # add key
            with warnings.catch_warnings(record=True) as _:
                hdu0.header[key] = (value, comment)
    # add primary extension to hdu list
    hdus = [hdu0]
    # -------------------------------------------------------------------------
    # push other extensions
    if len(data) > 1:
        # loop around other extensions
        for ext in range(1, len(data)):
            # add name of extension if given
            if names[ext] is not None:
                # deal with no header
                if header[ext] is None:
                    header[ext] = fits.Header()
                # convert to fits header if needed
                elif isinstance(header[ext], LBLHeader):
                    header[ext] = header[ext].to_fits()
                # add extension name to header
                header[ext]['EXTNAME'] = names[ext]
            # deal with type = image
            if dtype[ext] == 'image':
                hdu_ext = fits.ImageHDU(data[ext], header=header[ext])
            elif dtype[ext] == 'table':
                hdu_ext = fits.BinTableHDU(data[ext], header=header[ext])
            else:
                # create error message
                emsg = ('dtype invalid must be "image" or "table" '
                        '(current value = {0})')
                eargs = [dtype[ext]]
                # raise error
                raise LblException(emsg.format(*eargs))
            # append to hdu list
            hdus.append(hdu_ext)
    # -------------------------------------------------------------------------
    # try to write to disk
    hdulist = fits.HDUList(hdus)
    try:
        with warnings.catch_warnings(record=True) as _:
            hdulist.writeto(filename, overwrite=True)
    except Exception as e:
        # construct error message
        emsg = 'Could not write file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))


def get_urlfile(url: str, name: str, savepath: str, required: bool = True):
    """
    Get the file from url

    :param url: str, the url to get the file from
    :param name: str, the name of the file (for logging)
    :param savepath:, str, the absolute path to the file save location

    :return: None
    """
    # check if we have the tapas file
    if not os.path.exists(savepath):
        # log that we are downloading tapas file
        msg = 'Downloading {0} file \n\tfrom: {1} \n\tto {2}'
        margs = [name, url, savepath]
        log.general(msg.format(*margs))
        # attempt to download the data
        try:
            wget.download(url, savepath)
            log.general('\nDownloaded tapas file.')
        except Exception as e:
            if not required:
                msg = 'Skipped {0} file. Missing from server.'
                margs = [name]
                log.general(msg.format(*margs))
                return
            emsg = 'Cannot download {0} file: {1}\n\tError {2}: {3}'
            eargs = [name, url, type(e), str(e)]
            raise base_classes.LblException(emsg.format(*eargs))


# =============================================================================
# Define hdf functions
# =============================================================================
def hdf_to_e2ds(store_key: str, filename: str, ext: int, norders: int,
                orderlist: List[int], npixels: int,
                subkey: Optional[str] = None) -> np.ndarray:
    # Open the HDFStore
    with pd.HDFStore(filename) as store:
        # deal with not having blaze_blue in HDFstore
        if store_key not in store:
            emsg = '{0} is not in HDFStore for {1}'
            eargs = [store_key, filename]
            raise LblException(emsg.format(*eargs))
        # get the blaze blue from store
        if subkey is None:
            image_df = store[store_key][ext]
        else:
            image_df = store[store_key][subkey][ext]
        # make blaze numpy array
        image = np.ones((norders, npixels))
        # loop around orders and populate blaze
        for it, order_num in enumerate(orderlist):
            image[it] = image_df[order_num]
    # return image
    return image


def hdf_to_header(storekey: str, filename: str):
    # Open the HDFStore
    with pd.HDFStore(filename) as store:
        # create a new LBL header
        new = LBLHeader()
        # deal with key not in store
        if storekey not in store:
            emsg = 'Cannot find header in file {0} using key {1}'
            eargs = [filename, storekey]
            raise LblException(emsg.format(*eargs))
        # get the header
        hdr_df = store[storekey]
        # loop around keys and add them to the header dictionary
        for key in hdr_df.index.values:
            outkey = str(key)
            # remove HIERARCH from key
            while outkey.startswith('HIERARCH '):
                outkey = outkey[len('HIERARCH '):]
            # copy key into new LBL Header
            new[outkey] = copy.deepcopy(hdr_df[key])
            new.comments[outkey] = ''
        # set filename
        new.filename = filename
    # return the new LBL header
    return new


def save_data_to_hdfstore(filename: str, columns: Dict[str, List[np.ndarray]],
                           indices: Dict[str, list], image_storekey: str,
                           header: LBLHeader, header_storekey: str):
    # -------------------------------------------------------------------------
    # get pandas multi-index
    all_indices = []
    for key in indices:
        all_indices.append(indices[key])
    all_combinations = list(itertools.product(*all_indices))
    index = pd.MultiIndex.from_tuples(all_combinations, names=indices.keys())
    # -------------------------------------------------------------------------
    # create the dataframe to hold the data and header
    df_image = pd.DataFrame(columns, index=index)
    df_header = pd.Series(header.data)
    # -------------------------------------------------------------------------
    # push into hdf
    with warnings.catch_warnings(record=True) as _:
        df_image.to_hdf(filename, key=image_storekey)
        df_header.to_hdf(filename, key=header_storekey)


# =============================================================================
# Define table functions
# =============================================================================
def load_table(filename: str, kind: Union[str, None] = None,
               fmt: str = 'fits', get_hdr: bool = False,
               extname: Optional[str] = None,
               ) -> Union[Table, Tuple[Table, LBLHeader]]:
    """
    Standard way to load table

    :param filename: str, the filename
    :param kind: the kind (for error message)
    :param fmt: str, the format of the table (i.e. csv or fits)
                   defaults to 'fits'
    :param get_hdr: bool, whether to get the header or not
    :param extname: str or None, if set load a specific extension

    :return: astropy.table.Table, the table loaded
    """
    # deal with no kind
    if kind is None:
        kind = '{0} table'.format(format)
    # try to load fits file
    try:
        with warnings.catch_warnings(record=True) as _:
            if extname is not None:
                table = Table.read(filename, format=fmt, hdu=extname)
            else:
                table = Table.read(filename, format=fmt)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    # deal with obtaining table header
    if get_hdr:
        # get header
        hdr = fits.getheader(filename)
        # convert hdr to LBLHeader
        hdr = LBLHeader.from_fits(hdr)
        # return table and header
        return table, hdr
    else:
        return table


def write_table(filename: str, table: Table, fmt: str = 'fits',
                overwrite: bool = True):
    """
    Standard way to write a table to disk

    :param filename: str, the absolute path to write the table to
    :param table: astropy.table.Table, the table to write
    :param fmt: str, astropy.table format
    :param overwrite: bool, if True overwrites existing file

    :return: None
    """
    try:
        table.write(filename, format=fmt, overwrite=overwrite)
    except Exception as e:
        emsg = 'Cannot write table {0} to disk \n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))


# =============================================================================
# Define table functions
# =============================================================================
def _test_for_formatting(key: str, number: Union[int, float]) -> str:
    """
    Specific test of a string that may either be:

    key   or {key}{number}

    e.g. if key = XXX{0}  --> XXX{number}
    e.g. if key = XXX --> XXX

    Note if XXX{0:.3f} number must be a float

    :param key: str, the key to test (and return if no formatting present)
    :param number: int or float, depending on format of key (and key present)
    :return: str, the ouput either modified (if with formatting) or "key"
    """
    # test the formatting by entering a number as format
    test_str = key.format(number)
    # if they are the same after test return key with the key and number in
    if test_str == key:
        return '{0}{1}'.format(key, number)
    # else we just return the test string
    else:
        return test_str


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

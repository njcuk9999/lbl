#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
from astropy.io import fits
from astropy.table import Table
import copy
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings

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
               use_tqdm: bool = True):
    # get tqdm
    tqdm = base.tqdm_module(use_tqdm, log.console_verbosity)
    # storage for valid files
    valid_files = []
    # loop around files
    for filename in tqdm(path_list):
        # assume file is valid at start
        valid = True
        # deal with prefix
        if prefix is not None:
            if not str(filename).startswith(prefix):
                continue
        # deal with suffix
        if suffix is not None:
            if not str(filename).endswith(suffix):
                continue
        # deal with contains
        if contains is not None:
            if contains not in str(filename):
                continue
        # deal with header keys
        if hkeys is not None:
            # skip non-fits file
            if not str(filename).endswith('.fits'):
                continue
            # load header
            hdr = load_header(str(filename), 'fits file')

            valid = True
            # loop around hkeys
            for hkey in hkeys:
                # check value
                valid &= filter_by_hkey(hdr, hkey, hkeys[hkey])
        # add to valid files if we have got to here
        if valid:
            valid_files.append(filename)

    return valid_files


# =============================================================================
# Define fits functions
# =============================================================================
def load_fits(filename: str,
              kind: Union[str, None] = None) -> Tuple[np.ndarray, fits.Header]:
    """
    Standard way to load fits files

    :param filename: str, the filename
    :param kind: the kind (for error message)

    :return: tuple, the data and the header
    """
    # deal with no kind
    if kind is None:
        kind = 'fits file'
    # try to load fits file
    try:
        data = fits.getdata(filename)
        header = fits.getheader(filename)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    return np.array(data), header.copy()


def load_header(filename: str,
                kind: Union[str, None] = None) -> fits.Header:
    """
    Standard way to load fits file header only

    :param filename: str, the filename
    :param kind: the kind (for error message)

    :return: tuple, the data and the header
    """
    # deal with no kind
    if kind is None:
        kind = 'fits file'
    # try to load fits file
    try:
        header = fits.getheader(filename)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    return header.copy()


def get_hkey(header: fits.Header, key: str,
             filename: Union[str, None] = None,
             required: bool = True) -> Any:
    """
    Get a key from the header and deal with errors

    :param header: fits.Header, a fits header
    :param key: str, the key to get from header
    :param filename: str, the filename associated with the header
    :param required: bool, if True this parameter is required and will raise
                     an exception when not found, if False will return None
                     on False

    :return: Any, the value of header[key]
    """
    # set function name
    _ = __NAME__ + '.get_hkey()'
    # deal with no filename
    if filename is None:
        filename = 'Unknown'
    # test for key in header
    if key in header:
        try:
            return header[key]
        except Exception as e:
            emsg = 'Cannot use key {0} from header: {1} \n\t{2}: {3}'
            eargs = [key, filename, type(e), str(e)]
            raise LblException(emsg.format(*eargs))
    elif not required:
        return None
    else:
        emsg = 'Key {0} not found in header: {1}'
        eargs = [key, filename]
        raise LblException(emsg.format(*eargs))


def get_hkey_2d(header: fits.Header, key: str,
                dim1: int, dim2: int, dtype: Type = float,
                filename: Union[str, None] = None) -> np.ndarray:
    """
    Get a 2D list from the header

    :param header: fits.Header - the header to load from
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
        filename = 'Unknown'
    # test for key in header
    if dtype is None:
        dtype = str
    # create 2d list
    values = np.zeros((dim1, dim2), dtype=dtype)
    # loop around the 2D array
    dim1, dim2 = values.shape
    for it in range(dim1):
        for jt in range(dim2):
            # construct the key name
            keyname = _test_for_formatting(key, it * dim2 + jt)
            # try to get the values
            try:
                # set the value
                values[it][jt] = dtype(header[keyname])
            except KeyError:
                emsg = 'Cannot find key {0} in header: {1}'
                eargs = [keyname, filename]
                raise LblException(emsg.format(*eargs))
    # return values
    return values


def filter_by_hkey(header: fits.Header, key: str,
                   values: Union[List[str], str]) -> bool:
    """
    Filter by a header key (returns True if valid)

    :param header: fits header, the header to check
    :param key: str, the key in the header to check
    :param values: a list of string values to check against the header value
    :return:
    """
    # deal with key not in header
    if key not in header:
        return False
    # deal with non list values
    if not isinstance(values, (list, np.ndarray)):
        values = [values]
    # loop around values
    for value in values:
        if str(header[key]) == str(value):
            return True
    # if we get here return False
    return False


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
FitsHeaders = Union[List[Union[fits.Header, None]], fits.Header, None]
FitsDtype = Union[List[Union[str, None]], str, None]


def write_fits(filename: str, data: FitsData = None,
               header: FitsHeaders = None, dtype: FitsDtype = None):
    # -------------------------------------------------------------------------
    # deal with non list data
    if data is None:
        data = [None]
    elif isinstance(data, (Table, np.ndarray)):
        data = [data]
    # -------------------------------------------------------------------------
    # deal with non list header
    if header is None:
        header = [None] * len(data)
    elif isinstance(header, fits.Header):
        header = [header]
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


# =============================================================================
# Define table functions
# =============================================================================
def load_table(filename: str, kind: Union[str, None] = None,
               fmt: str = 'fits', get_hdr: bool = False
               ) -> Union[Table, Tuple[Table, fits.Header]]:
    """
    Standard way to load table

    :param filename: str, the filename
    :param kind: the kind (for error message)
    :param fmt: str, the format of the table (i.e. csv or fits)
                   defaults to 'fits'
    :param get_hdr: bool, whether to get the header or not

    :return: astropy.table.Table, the table loaded
    """
    # deal with no kind
    if kind is None:
        kind = '{0} table'.format(format)
    # try to load fits file
    try:
        with warnings.catch_warnings(record=True) as _:
            table = Table.read(filename, format=fmt)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1} \n\t{2}: {3}'
        eargs = [kind, filename, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    # deal with obtaining table header
    if get_hdr:
        hdr = fits.getheader(filename)
        # deal with main header in extension 1 (not primary)
        if len(hdr) < 10:
            hdr = fits.getheader(filename, ext=1)
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

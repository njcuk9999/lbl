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
import numpy as np
import os
from pathlib import Path
from typing import Any, Tuple, Type, Union

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



def make_dir(path: str , directory: str, kind: str) -> str:
    """
    Make a directory

    :param path:
    :param directory:
    :param kind:
    :return:
    """

    # construct absolute path
    abspath = Path(path).joinpath(directory)
    # check if directory exists
    if abspath.exists():
        msg = '{0} directory exists (Path={1})'
        margs = [kind, str(abspath)]
        log.logger.info(msg.format(*margs))
        # return absolute path directory
        return str(abspath)
    # else try to create directory
    else:
        try:
            # make directory path
            abspath.mkdir()
            # return absolute path directory
            return str(abspath)
        except Exception as e:
            emsg = 'Cannot create {0} directory. Path={1} \n\t{2}: {3}'
            eargs = [kind, str(abspath), type(e), str(e)]
            raise LblException(emsg.format(*eargs))



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
        emsg = 'Cannot load {0}. Filename: {1}'
        eargs = [kind, filename]
        raise LblException(emsg.format(*eargs))
    return np.array(data), header.copy()


def get_hkey(header: fits.Header, key: str,
             filename: Union[str, None] = None) -> Any:
    """
    Get a key from the header and deal with errors

    :param header: fits.Header, a fits header
    :param key: str, the key to get from header

    :return: Any, the value of header[key]
    """
    # deal with no filename
    if filename is None:
        filename = 'Unknown'
    # test for key in header
    if key in header:
        try:
            return header[key]
        except Exception as e:
            emsg = 'Cannot use key {0} from header: {1}'
            eargs = [key, filename]
            raise LblException(emsg.format(*eargs))
    else:
        emsg = 'Key {0} not found in header: {1}'
        eargs = [key, filename]
        raise LblException(emsg.format(*eargs))


def get_hkey_2d(header: fits.Header,  key: str,
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


# =============================================================================
# Define table functions
# =============================================================================
def load_table(filename: str,
               kind: Union[str, None] = None,
               fmt: str = 'fits') -> Table:
    """
    Standard way to load table

    :param filename: str, the filename
    :param kind: the kind (for error message)
    :param fmt: str, the format of the table (i.e. csv or fits)
                   defaults to 'fits'

    :return: astropy.table.Table, the table loaded
    """
    # deal with no kind
    if kind is None:
        kind = '{0} table'.format(format)
    # try to load fits file
    try:
        table = Table.read(filename, format=fmt)
    except Exception as e:
        emsg = 'Cannot load {0}. Filename: {1}'
        eargs = [kind, filename]
        raise LblException(emsg.format(*eargs))
    return table


def write_table(filename: str, table: Table, fmt: str = 'fits'):
    """
    Standard way to write a table to disk

    :param filename: str, the absolute path to write the table to
    :param table: astropy.table.Table, the table to write

    :return: None
    """
    try:
        table.write(filename, format=fmt)
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

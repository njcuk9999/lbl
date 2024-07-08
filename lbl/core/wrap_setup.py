#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-07-05 at 03:36

@author: cook
"""
from typing import List, Optional
import argparse
import os
import getpass
import socket

from lbl.core import base
from lbl.core import base_classes
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'setup.py'
__STRNAME__ = 'LBL Compil'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get the log
log = base_classes.log
# -----------------------------------------------------------------------------


# =============================================================================
# Define functions
# =============================================================================
def get_instrument_list() -> List[str]:
    """
    Get the list of instruments available
    :return:
    """
    # get list of instruments
    instruments = []
    # loop around instrument classes
    for instrument in base.INSTRUMENTS:
        instruments.append(instrument)
    # return instruments
    return instruments


def get_data_source_list(instrument: str) -> List[str]:
    """
    Get the data sources for a given instrument

    :param instrument: str, the instrument name
    :return:
    """
    # check if instrument is in select.InstDict
    if instrument not in select.InstDict:
        emsg = 'Instrument {0} not in select.InstDict'.format(instrument)
        log.error(emsg)
    # get data sources
    return select.InstDict[instrument].keys()


def get_args(description: str):
    # start argument parser
    parser = argparse.ArgumentParser(description=description)
    # add instrument argument
    parser.add_argument('--instrument', type=str,
                   default='None',
                   action='store', help='The instrument name',
                   choices=get_instrument_list())
    # add data source argument
    parser.add_argument('--data_source', type=str,
                   default='Null',
                   action='store', help='The data source')
    # add the data directory
    parser.add_argument('--data_dir', type=str,
                   default='None',
                   action='store', help='The data directory')
    # add the name of the wrap file
    parser.add_argument('wrap_file', type=str,
                   default='None',
                   action='store', help='The wrap file name')
    # parse the arguments
    args = parser.parse_args()
    # push into dictionary and return
    return vars(args)


def get_user() -> str:
    """
    Get the user name and host name in os independent way
    :return:
    """
    username = getpass.getuser()
    hostname = socket.gethostname()

    return f'{username}@{hostname}'


def get_str_list(input_list):
    """
    Get the string list for the data sources
    :return:
    """
    # empty source string at first
    str_list = '[{0}]'
    # string items must have quotes
    str_items = ["'{0}'".format(item) for item in input_list]
    # return source string
    return str_list.format(', '.join(str_items))


def ask(name: str, required: bool, options: Optional[List[str]] = None):
    # ask for input
    if options is not None:
        print('\n\n')
        str_options = ', '.join(options)
        question = ('Please enter the {0} (from: {1})'.format(name, str_options))
        # loop around until we get a valid input
        while True:
            # get input
            value = input(question + ':\t').strip()
            # check if value is in options
            if value in options:
                return value
            # if not required and value is None return None
            if not required and value in [None, 'None', 'Null']:
                return None
            # print error
            print('Error: {0} not in options'.format(value))
    else:
        print('\n\n')
        question = ('Please enter the {0}'.format(name))
        # get input
        value = input(question + ':\t')
        # return value
        return value


def instruments_str() -> str:
    """
    Produce the comment for instruments
    :return:
    """
    # empty source string at first
    source_str = ''
    # loop around instruments
    for instrument in base.INSTRUMENTS:
        source_str += '\n\t#\t\t{0}'.format(instrument)
    # return instruments string
    return source_str


def data_sources_str():
    """
    Produce the comment for data_soruces
    :return:
    """
    # empty source string at first
    source_str = ''
    # loop around instruments
    for instrument in select.InstDict:
        # get data sources
        dsources = select.InstDict[instrument].keys()
        # add to source string
        source_str += '\n\t#\t\t{0}: {1}'.format(instrument, ' or '.join(dsources))
    # return source string
    return source_str


def create_wrap_file(params: dict, wrap_dict: dict):
    """
    Push the wrap_dict into the wrap_default.txt and save as a file

    :param params:
    :param wrap_dict:
    :return:
    """
    # get the resource path
    resource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'resources')
    # get the default text for the wrap file
    default_wrap_file = os.path.join(resource_path, 'wrap_default.txt')

    # load the default wrap file
    with open(default_wrap_file, 'r') as wrapf:
        wrap_text = wrapf.read()

    # using the wrap dictionary replace the values in the wrap text
    wrap_text = wrap_text.format(**wrap_dict)

    # write wrap_file
    with open(params['wrap_file'], 'w') as wrapf:
        wrapf.write(wrap_text)






# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-07-28 at 14:38

@author: cook
"""
import os
import shutil

import wget

from lbl.core import base
from lbl.core import base_classes
from lbl.resources import lbl_misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_demo.py'
__STRNAME__ = 'LBL Demo'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
# set the path to the data
URL_TO_DATA = '/etiennes/path/'
# set packed file format (zip, tar, gztar, bztar)
UNPACK_FORMAT = 'gztar'
# example demo wrap file
EXAMPLE_DEMO_WRAP = 'example_demo_wrap.py'


# =============================================================================
# Define functions
# =============================================================================
def main():
    # ----------------------------------------------------------------------
    # print splash
    lbl_misc.splash(name=__STRNAME__, instrument='None',
                    params=base_classes.ParamDict(), plogger=log)
    # ----------------------------------------------------------------------
    # set ask condition
    ask_condition = True
    uinput1 = None
    # ----------------------------------------------------------------------
    # loop until we have directory
    while ask_condition:
        # prompt user for data directory
        uinput1 = input('Select data directory: (Ctrl+C to exit)\t')
        # if data directory does not exist ask user to create it
        if not os.path.exists(uinput1):
            uinput2 = input(f'Create directory "{uinput1}"?\n[Y]es or [N]o:\t')
            # if yes create directory
            if 'Y' in uinput2.upper():
                ask_condition = False
                # create directory
                os.makedirs(uinput1)
        # we already have the directory
        else:
            ask_condition = False
    # -------------------------------------------------------------------------
    # deal with a Ctrl C
    if uinput1 is None or ask_condition:
        log.warning('Exiting demo setup. Goodbye')
    # -------------------------------------------------------------------------
    # set data directory
    data_dir = os.path.join(uinput1, 'lbl_demo')
    # make lbl_demo sub-directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # -------------------------------------------------------------------------
    # get the basename for the packed file
    basename = os.path.basename(URL_TO_DATA)
    # construct outpath for the packed file
    packed_file = os.path.join(data_dir, basename)
    # get the tar file
    log.info(f'Downloading {URL_TO_DATA} to {packed_file}')
    wget.download(URL_TO_DATA, out=packed_file)
    # -------------------------------------------------------------------------
    # unpack the packed file
    log.info(f'Unpacking archive')
    shutil.unpack_archive(packed_file, extract_dir=data_dir,
                          format=UNPACK_FORMAT)
    # -------------------------------------------------------------------------
    # deal with updating example demo wrap file
    # -------------------------------------------------------------------------
    # path to example demo wrap
    epath = os.path.join(data_dir, EXAMPLE_DEMO_WRAP)
    # set new lines
    newlines = []
    # modify example demo wrap file
    with open(epath, 'r') as efile:
        # read all lines in example demo wrap file
        elines = efile.readlines()
        # replace all instances of DATA_DIR
        for eline in elines:
            if '{DATA_DIR}' in eline:
                newline = eline.replace('{DATA_DIR}', data_dir)
                newlines.append(newline)
            else:
                newlines.append(eline)
    # remove old demo
    os.remove(epath)
    # write new demo
    with open(epath, 'w') as efile:
        for newline in newlines:
            efile.write(newline + '\n')


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    if __name__ == "__main__":
        # start main code
        ll = main()

# =============================================================================
# End of code
# =============================================================================

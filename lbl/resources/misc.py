#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-23

@author: cook
"""
from pathlib import Path
import shutil

from lbl.core import io

# =============================================================================
# Define variables
# =============================================================================
def copy_readme(data_dir: str):
    """
    Copies the data structure read me to here

    :param data_dir: str, the data directory

    :return:
    """
    # get the path to this directory
    python_filename = __file__
    # get in directory
    inpath = Path(python_filename).parent
    # get out directory
    outpath = Path(data_dir)
    # check / make the outpath
    io.make_dir(outpath, '', 'Data', verbose=False)
    # construct path to read me
    input_path = inpath.joinpath('data_str_readme.md')
    # construct new path to read me
    output_path = outpath.joinpath('README.md')
    # check for output path
    if output_path.exists():
        return
    # else copy the file
    shutil.copy(str(input_path), str(output_path))


# =============================================================================
# Define functions
# =============================================================================


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

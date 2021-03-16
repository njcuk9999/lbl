#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import os

from lbl.core.base_classes import LblException

# =============================================================================
# Define variables
# =============================================================================


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



# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

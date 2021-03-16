#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl.core import base_classes
from lbl.instruments import select


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = base_classes.Instrument
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
# add arguments (must be in parameters.py)
ARGS = ['INSTRUMENT', 'CONFIG_FILE', 'DATA_DIR']
# TODO: Fill out
DESCRIPTION = 'Use this code to compute the LBL'


# =============================================================================
# Define functions
# =============================================================================
def main(**kwargs):
    # deal with parsing arguments
    args = select.parse_args(ARGS, kwargs, DESCRIPTION)
    # load instrument
    inst = select.load_instrument(args)
    # run __main__
    try:
        return __main__(inst, inst.params)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected Error: {0}: {1}'
        eargs = [type(e), str(e)]
        raise LblException(emsg.format(*eargs))


def __main__(inst: Instrument, params: ParamDict):
    pass


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

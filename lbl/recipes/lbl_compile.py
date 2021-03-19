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
__NAME__ = 'lbl_compile.py'
__STRNAME__ = 'LBL Compil'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
InstrumentsList = select.InstrumentsList
InstrumentsType = select.InstrumentsType
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
# add arguments (must be in parameters.py)
ARGS = ['INSTRUMENT', 'CONFIG_FILE', 'DATA_DIR', 'PLOT']
# TODO: Etienne - Fill out
DESCRIPTION = 'Use this code to compile the LBL rdb file'


# =============================================================================
# Define functions
# =============================================================================
def main(**kwargs):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param kwargs: kwargs to parse to instrument - anything in params can be
                   parsed (overwrites instrumental and default parameters)
    :return:
    """
    # deal with parsing arguments
    args = select.parse_args(ARGS, kwargs, DESCRIPTION)
    # load instrument
    inst = select.load_instrument(args)
    # print splash
    select.splash(name=__STRNAME__, instrument=inst.name)
    # run __main__
    try:
        return __main__(inst)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected Error: {0}: {1}'
        eargs = [type(e), str(e)]
        raise LblException(emsg.format(*eargs))


def __main__(inst: InstrumentsType, **kwargs):
    """
    The main recipe function - all code dealing with recipe functionality
    should go here

    :param inst: Instrument instance
    :param kwargs: kwargs to parse to instrument (only use if inst is None)
                   anything in params can be parsed (overwrites instrumental
                   and default parameters)

    :return: all variables in local namespace
    """
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or inst.params is None:
        # deal with parsing arguments
        args = select.parse_args(ARGS, kwargs, DESCRIPTION)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # -------------------------------------------------------------------------
    # Step 1:
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    return locals()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

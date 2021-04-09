#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admin function here

Created on 2021-04-09

@author: cook
"""
from astropy.table import Table, vstack
import os
from typing import Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'resources.misc.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
log = base_classes.log

# -----------------------------------------------------------------------------
# define the working directory
WORKSPACE = '/data/spirou/data/misc/lbltest/'
# define the name for the full param config yaml
PARAM_FULL_YAML = 'full_config.yaml'
# define the parameter readme table file (copied into full README.md)
PARAM_README = 'param_readme.md'

# -----------------------------------------------------------------------------
# mode:
# mode 1: make parameter table for readme
#mode = 1
# mode 2: make yaml
mode = 2


# =============================================================================
# Define functions
# =============================================================================
def make_param_table(instrument: Union[str, None] = 'SPIROU') -> Table:
    """
    Make a parameter table from global and instrument parameters

    :param instrument:
    :return:
    """
    # get global params
    gparams = parameters.params.copy()

    if instrument is not None:
        # set up fake arguments
        args_compute = ['INSTRUMENT']
        # set up fake kwargs
        kwargs = dict(instrument=instrument)
        # deal with parsing arguments
        args = select.parse_args(args_compute, kwargs, '')
        # load instrument
        inst = select.load_instrument(args)
        # load params from instrument
        params = inst.params.copy()
    else:
        params = gparams.copy()

    # create table elemetns
    keys, dvalues, ivalues, comments = [], [], [], []
    for key in params:
        keys.append(str(key))
        dvalues.append(str(gparams[key]))
        ivalues.append(str(params[key]))
        comments.append(str(gparams.instances[key].description))

    table = Table()
    table['KEY'] = keys
    table['DEFAULT_VALUE'] = dvalues
    if instrument is not None:
        table['{0}_VALUE'.format(instrument)] = ivalues
    table['DESCRIPTION'] = comments

    return table


def make_readme_param_table(table: Table):

    # construct filename
    abspath = os.path.join(WORKSPACE, PARAM_README)
    # -------------------------------------------------------------------------
    # add --- separators
    # -------------------------------------------------------------------------
    sep_dict = dict()
    # loop around columns
    for col in table.colnames:
        maxlen = 0
        for value in table[col]:
            if len(value) > maxlen:
                maxlen = len(value)
        # add separate for this column
        sep_dict[col] = ['-' * maxlen]
    # convert to table
    toptable = Table()
    for col in table.colnames:
        toptable[col] = sep_dict[col]
    # write full table
    fulltable = vstack([toptable, table])
    fulltable.write(abspath, format='ascii.fixed_width', overwrite=True)


def make_full_config_yaml(table: Table):

    # get instrument col name
    icol = None
    for col in table.colnames:
        if col not in ['KEY', 'DEFAULT_VALUE', 'DESCRIPTION']:
            icol = str(col)
    lines = ['# -------------------------------------------']
    lines += ['# LBL Full config file (auto-generated)']
    lines += ['# -------------------------------------------']
    lines += ['', '']
    # loop around row
    for row in range(len(table)):
        # get values for this row
        key = table['KEY'][row]
        dvalue = table['DEFAULT_VALUE'][row]

        # add description
        lines += ['# {0}'.format(table['DESCRIPTION'][row])]
        # add suggested values
        if icol is not None:
            suggested_value = '## Default = {0}   {1} = {2}'
            suggested_kwargs = [dvalue, icol, table[icol][row]]
        else:
            suggested_value = '## Default = {0}'
            suggested_kwargs = [dvalue]
        lines += [suggested_value.format(*suggested_kwargs)]
        # add parameter
        if icol is not None:
            value = table[icol][row]
        else:
            value = dvalue
        # ---------------------------------------------------------------------
        # clean up values with * or -
        if '*' in value or '-' in value:
            value = "'{0}'".format(value)
        # ---------------------------------------------------------------------
        # clean up list values
        if value.startswith('[') and value.endswith(']'):
            listvalue = eval(value)
            if len(listvalue) == 0:
                lines += ['{0}:'.format(key)]
                lines += ['- None']
            else:
                lines += ['{0}:'.format(key)]
                for lvalue in listvalue:
                    lines += ['- {0}'.format(lvalue)]
        else:
            # add value to lines
            lines += ['{0}: {1}'.format(key, value)]
        # add a blank line
        lines += ['']
    # construct abspath
    abspath = os.path.join(WORKSPACE, PARAM_FULL_YAML)
    # save lines to file
    with open(abspath, 'w') as yamlfile:
        for line in lines:
            yamlfile.write(line + '\n')

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    if mode == 1:
        table = make_param_table('SPIROU')
        make_readme_param_table(table)
    if mode == 2:
        table = make_param_table('SPIROU')
        make_full_config_yaml(table)


# =============================================================================
# End of code
# =============================================================================

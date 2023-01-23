#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admin function here

Created on 2021-04-09

@author: cook
"""
import argparse
import os
import sys
from typing import List, Union

import numpy as np
from astropy.table import Table, vstack

from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters
from lbl.instruments import select
from lbl.resources import lbl_misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'resources.misc.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
log = base_classes.log
LblException = base_classes.LblException
QArg = lbl_misc.QuickArg

# -----------------------------------------------------------------------------
# define the working directory
WORKSPACE = '/data/lbl/data/misc'
# define the name for the full param config yaml
PARAM_FULL_YAML = 'full_config.yaml'
# define the parameter readme table file (copied into full README.md)
PARAM_README = 'param_readme.md'
# -----------------------------------------------------------------------------
# deal with arguments
parser = argparse.ArgumentParser(description='Admin LBL code - see arguments '
                                             'for options')
# add quick arguments
pargs = dict()
pargs['--create_dirs'] = QArg(helpstr='Create input directories')
pargs['--make_readme'] = QArg(helpstr='Run make read me script')
pargs['--make_full_yaml'] = QArg(helpstr='Run full config yaml script')
pargs['--config'] = QArg(action=None,
                         helpstr='The config yaml file (required for '
                                 'some scripts)')
# loop around args and add back to parser
for _key in pargs:
    parser.add_argument(_key, **pargs[_key].kwargs())


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
        cargs = select.parse_args(args_compute, kwargs, '')
        # load instrument
        inst = select.load_instrument(cargs)
        # load params from instrument
        params = inst.params.copy()
    else:
        params = gparams.copy()

    # create table elemetns
    keys, dvalues, ivalues, comments = [], [], [], []
    for pkey in params:
        keys.append(str(pkey))
        dvalues.append(str(gparams[pkey]))
        ivalues.append(str(params[pkey]))
        comments.append(str(gparams.instances[pkey].description))

    table = Table()
    table['KEY'] = keys
    table['DEFAULT_VALUE'] = dvalues
    if instrument is not None:
        table['{0}_VALUE'.format(instrument)] = ivalues
    table['DESCRIPTION'] = comments

    return table


def make_readme_param_table(tables: List[Table]):
    # construct filename
    abspath = os.path.join(WORKSPACE, PARAM_README)
    # -------------------------------------------------------------------------
    # add --- separators
    # -------------------------------------------------------------------------
    sep_dict = dict()

    final_table = Table()
    # loop around all tables
    for table in tables:
        # ---------------------------------------------------------------------
        # just adding a row of --- at the top of table (for each column)
        # ---------------------------------------------------------------------
        # loop around columns
        for col in table.colnames:
            # skip columns already in sep_dict
            if col in sep_dict:
                continue
            maxlen = 0
            for value in table[col]:
                if len(value) > maxlen:
                    maxlen = len(value)
            # add separate for this column
            sep_dict[col] = ['-' * maxlen]
        # ---------------------------------------------------------------------
        for col in table.colnames:
            # skip columns that are already in final table
            if col in final_table.colnames:
                continue
            # add columns
            final_table[col] = table[col]
    # -------------------------------------------------------------------------
    # convert sep_dict to table of ---
    toptable = Table()
    for col in final_table.colnames:
        toptable[col] = sep_dict[col]
    # -------------------------------------------------------------------------
    # move description column to end
    if 'DESCRIPTION' in final_table.colnames:
        description = np.array(final_table['DESCRIPTION'])
        del final_table['DESCRIPTION']
        final_table['DESCRIPTION'] = description
    # -------------------------------------------------------------------------
    # write full table
    fulltable = vstack([toptable, final_table])
    fulltable.write(abspath, format='ascii.fixed_width', overwrite=True)
    # write the final table to a csv and fits file
    final_table.write(abspath.replace('.md', '.fits'), format='fits',
                      overwrite=True)
    final_table.write(abspath.replace('.md', '.csv'), format='csv',
                      overwrite=True)


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


def create_directories(config_file: str = 'None'):
    """
    Create all directories using config file

    :param config_file:
    :return:
    """
    # make sure config file exists
    if not os.path.exists(config_file) or config_file == 'None':
        emsg = 'Config file must exists (current value = {0})'
        raise LblException(emsg.format(config_file))
    # deal with parsing arguments
    args = select.parse_args([], dict(config_file=config_file), '', parse=False)
    # load instrument
    inst = select.load_instrument(args)
    # make directories
    dparams = select.make_all_directories(inst)
    # check and log creation
    for dkey in dparams:
        # get the directory entry
        outdir = dparams[dkey]
        # create if if it doesn't exist
        if os.path.exists(outdir):
            log.general('Created dir: {0}'.format(outdir))
        else:
            log.warning('Could not create dir: {0}'.format(outdir))


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # get input arguments (all switches)
    _args = parser.parse_args()
    # reset args
    sys.argv = sys.argv[0:1]
    # get fake parameters
    _params = base_classes.ParamDict()
    _params['COMMAND_LINE_ARGS'] = lbl_misc.quick_args(_args, pargs)
    # splash
    lbl_misc.splash('LBL Admin', 'None', _params)
    # -------------------------------------------------------------------------
    # run create directories (if True)
    if _args.create_dirs:
        create_directories(_args.config)
    # -------------------------------------------------------------------------
    # run make read me (if True)
    if _args.make_readme:
        # loop around instruments
        _tables = []
        for instrument in base.INSTRUMENTS:
            log.general('Adding instrument {0}'.format(instrument))
            _table = make_param_table(instrument)
            _tables.append(_table)
        make_readme_param_table(_tables)
    # -------------------------------------------------------------------------
    # run make full yaml config file (if True)
    if _args.make_full_yaml:
        _table = make_param_table('SPIROU')
        make_full_config_yaml(_table)

# =============================================================================
# End of code
# =============================================================================

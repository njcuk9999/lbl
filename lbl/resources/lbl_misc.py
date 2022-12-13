#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-23

@author: cook
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import logger

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'resources.misc.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
log = base_classes.log


# =============================================================================
# Define functions
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
    io.make_dir(str(outpath), '', 'Data', verbose=False)
    # construct path to read me
    input_path = inpath.joinpath('data_str_readme.md')
    # construct new path to read me
    output_path = outpath.joinpath('README.md')
    # check for output path
    if output_path.exists():
        return
    # else copy the file
    shutil.copy(str(input_path), str(output_path))


def move_log(data_dir: str, recipe: str):
    """
    Move the log file to the data directory

    :param data_dir: str, the data directory
    :param recipe: str, the name of the code used
    :return:
    """
    global log
    # check that log dir exists
    log_path = io.make_dir(data_dir, 'log', 'log')
    # make log file
    timenow = base.Time.now().fits
    datenow = str(timenow).split('T')[0]
    # clean recipe name
    recipe = recipe.replace('.py', '').replace(' ', '_')
    # construct log file name
    logfile = 'LOG-{0}-{1}.log'.format(datenow, recipe)
    # construct aboslute path
    log_path = os.path.join(log_path, logfile)
    # move log file
    log = logger.Log(filename=log_path)
    # update global call
    base_classes.log = log


def splash(name: str, instrument: str, params: base_classes.ParamDict,
           plogger: Union[logger.Log, None] = None):
    # deal with no logger
    if plogger is None:
        plogger = log
    # print splash
    msgs = ['']
    msgs += ['*' * 79]
    msgs += ['\t{0}']
    msgs += ['\t\tVERSION: {1}']
    msgs += ['\t\tINSTRUMENT: {2}']
    msgs += ['*' * 79]
    msgs += ['']
    margs = [name, __version__, instrument]
    # loop through messages
    for msg in msgs:
        plogger.info(msg.format(*margs))

    # add user args
    if params is not None:
        if 'USER_KWARGS' in params:
            if len(params['USER_KWARGS']) > 0:
                plogger.info('User keyword arguments:')
                # loop around arguments and add
                for cmdmsg in params['USER_KWARGS']:
                    plogger.info(cmdmsg)

    # add command line arguments (if not None)
    if params is not None:
        if len(params['COMMAND_LINE_ARGS']) > 0:
            plogger.info('Command line arguments:')
            # loop around arguments and add
            for cmdmsg in params['COMMAND_LINE_ARGS']:
                plogger.info(cmdmsg)


def end(recipe: str, plogger: Union[logger.Log, None] = None):
    """
    print and end statement

    :param recipe: str, the recipe name
    :param plogger: the logger instance class

    :return: None - prints to screen / log
    """
    # deal with no logger
    if plogger is None:
        plogger = log
    # print splash
    msgs = ['']
    msgs += ['*' * 79]
    msgs += ['{0} finished successfully']
    msgs += ['*' * 79]
    msgs += ['']
    margs = [recipe]
    # loop through messages
    for msg in msgs:
        plogger.info(msg.format(*margs))
    return


class QuickArg:
    def __init__(self, action: Union[str, None] = 'store_true',
                 helpstr: str = ''):
        """
        quick arg construct when not using default approach (only use for tools)

        :param action: str, store true or store constant
        :param helpstr:
        """
        self.action = action
        self.helpstr = helpstr
        if action == 'store_true':
            self.switch = True
        else:
            self.switch = False

    def kwargs(self) -> dict:
        """
        Used to pass these to argparse as **kwargs
        :return:
        """
        return dict(action=self.action, help=self.helpstr)


def quick_args(args: Any, quickargs: Dict[str, QuickArg]):
    """
    Get display args for splash

    :param args: arg parser namespace
    :param quickargs: list of QuickArgs instances
    :return:
    """
    listargs = []
    for qarg in quickargs:
        arg = qarg.strip('--')
        if arg in args.__dict__:
            if quickargs[qarg].switch and not args.__dict__[arg]:
                continue
            elif quickargs[qarg].switch:
                listargs.append('{0}'.format(qarg))
            else:
                listargs.append('{0}={1}'.format(qarg, args.__dict__[arg]))
    return listargs


def check_runparams(rparams: Dict[str, Any], key: str) -> Any:
    """
    Check key run parameters

    :param key: str, the key in rparams
    :param rparams: dict, the rparams dictionary

    :return: the value
    """
    if key not in rparams:
        emsg = 'LBL_WRAP ERROR: Must define key {0} in rparams'
        eargs = [key]
        raise base_classes.LblException(emsg.format(*eargs))
    else:
        return rparams[key]


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

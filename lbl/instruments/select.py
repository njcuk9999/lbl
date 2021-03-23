#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-17

@author: cook
"""
import argparse
from copy import deepcopy
import os
from typing import Any, Dict, List, Union
import yaml

from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters
from lbl.core import io
from lbl.instruments import spirou
from lbl.instruments import harps
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.default.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# load classes
ParamDict = base_classes.ParamDict
log = base_classes.log
# instruments list
InstrumentsType = Union[default.Instrument, spirou.Spirou,
                        harps.Harps, None]
InstrumentsList = (default.Instrument, spirou.Spirou, harps.Harps)


# =============================================================================
# Define functions
# =============================================================================
def parse_args(argnames: List[str], kwargs: Dict[str, Any],
               description: Union[str, None] = None) -> ParamDict:
    """
    Parse the arguments 'args' using params (and their Const instances)
    if value in kwargs override default value in params

    :param argnames: list of strings, the arguments to add to command line
    :param kwargs: dictionary, the function call arguments to add
    :param description: str, the program description (for the help)
    :return:
    """
    # set function name
    func_name = __NAME__ + '.parse_args'
    # get parser
    parser = argparse.ArgumentParser(description=description)
    # get params
    params = parameters.params.copy()
    # get default values
    default_values = dict()
    for argname in params:
        default_values[argname] = deepcopy(params[argname])
    # storage of inputs
    inputs = ParamDict()
    cmd_inputs = ParamDict()
    # -------------------------------------------------------------------------
    # add args from sys.argv
    for argname in argnames:
        if argname in params:
            # get constant
            const = params.instances[argname]
            # skip constants without argument / dtype
            if const.argument is None or const.dtype is None:
                continue
            # deal with bool
            if const.dtype is bool:
                pkwargs = dict(dest=const.key, default=params[argname],
                               action='store_true', help=const.description)
            else:
                pkwargs = dict(dest=const.key, type=const.dtype,
                               default=params[argname],
                               action='store', help=const.description,
                               choices=const.options)
            # parse argument
            parser.add_argument(const.argument, **pkwargs)
    # -------------------------------------------------------------------------
    # update params value with kwargs
    for kwarg in kwargs:
        # force kwarg to upper case
        kwargname = kwarg.upper()
        # only if in params
        if kwargname in params:
            inputs.set(kwargname, kwargs[kwarg], source=func_name + ' [KWARGS]')
    # -------------------------------------------------------------------------
    # load parsed args into inputs
    args = vars(parser.parse_args())
    for argname in args:
        if argname in inputs:
            # check whether None is allowed
            if args[argname] is None:
                if argname not in inputs.not_none:
                    continue

        # we set the input
        cmd_inputs.set(argname, args[argname], source=func_name + '[CMD]')
    # -------------------------------------------------------------------------
    # we need to copy config_file into inputs - in general we copy command line
    #   arguments after - but we need the config one to load it
    if 'CONFIG_FILE' in cmd_inputs:
        if cmd_inputs['CONFIG_FILE'] is not None:
            inputs.set('CONFIG_FILE', value=cmd_inputs['CONFIG_FILE'])
    # -------------------------------------------------------------------------
    # deal with yaml file
    if 'CONFIG_FILE' in inputs:
        if inputs['CONFIG_FILE'] is not None:
            # get config file
            config_file = inputs['CONFIG_FILE']
            # check if exists
            if not io.check_file_exists(config_file):
                emsg = 'config file = "{0}" does not exist'
                eargs = [os.path.realpath(config_file)]
                raise base_classes.LblException(emsg.format(*eargs))
            # load yaml file
            with open(config_file, 'r') as yfile:
                yaml_inputs = yaml.load(yfile, yaml.FullLoader)
            # add these inputs to inputs
            for argname in yaml_inputs:
                # skip Nones
                if yaml_inputs[argname] not in [None, 'None']:
                    inputs.set(argname, yaml_inputs[argname],
                               source=func_name + '[YAML]')
    # -------------------------------------------------------------------------
    # storage of command line arguments
    inputs.set('COMMAND_LINE_ARGS', value=[], source=func_name)
    # now copy over rest of command line arguments
    for key in cmd_inputs:
        # skip keys that are None
        if cmd_inputs[key] is not None:
            # do not update if default value
            if key in default_values:
                # check that default value isn't set
                if cmd_inputs[key] == default_values[key]:
                    # skip values which are default (we don't want to override
                    #   values set from kwargs or yaml)
                    continue
            # get arg
            cmdarg = params.instances[key].argument
            # get value
            value = cmd_inputs[key]
            # get source
            source = cmd_inputs.instances[key].source
            # update input - if valid
            inputs.set(key, value=value, source=source)
            # log inputs added from command line
            if value in [None, '', 'None']:
                continue
            msg = '\t{0}="{1}"'.format(cmdarg, value)
            inputs['COMMAND_LINE_ARGS'].append(msg)
    # -------------------------------------------------------------------------
    # parse arguments
    return inputs


def load_instrument(args: ParamDict) -> InstrumentsType:
    """
    Load an instrument

    :param args: ParamDict, the parameter dictionary of inputs

    :return: Instrument instance
    """
    # deal with instrument not in args
    if 'INSTRUMENT' not in args:
        emsg = ('Instrument name must be be defined (yaml, input or '
                'command line)')
        raise base_classes.LblException(emsg)
    # set instrument
    instrument = args['INSTRUMENT']
    # get base params
    params = parameters.params.copy()
    # deal with instrument not being a string
    if not isinstance(instrument, str):
        emsg = 'Instrument name must be a string (value={0})'
        eargs = [instrument]
        raise base_classes.LblException(emsg.format(*eargs))
    # select SPIROU
    if instrument.upper() == 'SPIROU':
        inst = spirou.Spirou(params)
    # select HARPS
    elif instrument.upper() == 'HARPS':
        inst = harps.Harps(params)
    # else instrument is invalid
    else:
        emsg = 'Instrument name "{0}" invalid'
        eargs = [instrument]
        raise base_classes.LblException(emsg.format(*eargs))
    # override inst params with args (from input/cmd/yaml)
    for argname in args:
        if argname in inst.params:
            # get source
            inst.params.set(argname, args[argname],
                            source=args.instances[argname].source)
    # return instrument instances
    return inst


def splash(name: str, instrument: str, cmdargs: Union[List[str], None] = None):
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
        log.info(msg.format(*margs))
    # add command line arguments (if not None)
    if cmdargs is not None:
        log.info('Command line arguments:')
        # loop around arguments and add
        for cmdmsg in cmdargs:
            log.info(cmdmsg)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

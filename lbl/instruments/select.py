#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-17

@author: cook
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import logger
from lbl.core import parameters
from lbl.instruments import carmenes
from lbl.instruments import default
from lbl.instruments import espresso
from lbl.instruments import harps
from lbl.instruments import harpsn
from lbl.instruments import sophie
from lbl.instruments import nirps
from lbl.instruments import spirou
from lbl.instruments import maroonx
from lbl.resources import lbl_misc

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
LblException = base_classes.LblException
# instruments list
InstrumentsType = Union[default.Instrument,
                        spirou.Spirou, spirou.SpirouCADC,
                        harps.Harps_ORIG, harps.Harps_ESO,
                        espresso.Espresso,
                        carmenes.Carmenes,
                        nirps.NIRPS_HA, nirps.NIRPS_HA_ESO,
                        nirps.NIRPS_HE, nirps.NIRPS_HE_ESO,
                        harpsn.HarpsN_ORIG, harpsn.HarpsN_ESO,
                        maroonx.MaroonX,
                        sophie.Sophie]
InstrumentsList = (default.Instrument,
                   spirou.Spirou, spirou.SpirouCADC,
                   harps.Harps_ORIG, harps.Harps_ESO,
                   espresso.Espresso,
                   carmenes.Carmenes,
                   nirps.NIRPS_HA, nirps.NIRPS_HA_ESO,
                   nirps.NIRPS_HE, nirps.NIRPS_HE_ESO,
                   harpsn.HarpsN_ORIG, harpsn.HarpsN_ESO,
                   maroonx.MaroonXRed, maroonx.MaroonXBlue,
                   sophie.Sophie)

# Add all the instrument + source combinations and link them to instrument
#   classes
#   Format:  InstDict[{INSTRUMENT}][{DATA_SOURCE}]
InstDict = dict()
InstDict['SPIROU'] = dict()
InstDict['SPIROU']['APERO'] = spirou.Spirou
InstDict['SPIROU']['CADC'] = spirou.SpirouCADC
InstDict['NIRPS_HA'] = dict()
InstDict['NIRPS_HA']['APERO'] = nirps.NIRPS_HA
InstDict['NIRPS_HA']['ESO'] = nirps.NIRPS_HA_ESO
InstDict['NIRPS_HE'] = dict()
InstDict['NIRPS_HE']['APERO'] = nirps.NIRPS_HE
InstDict['NIRPS_HE']['ESO'] = nirps.NIRPS_HE_ESO
InstDict['HARPS'] = dict()
InstDict['HARPS']['ORIG'] = harps.Harps_ORIG
InstDict['HARPS']['ESO'] = harps.Harps_ESO
InstDict['CARMENES'] = dict()
InstDict['CARMENES']['None'] = carmenes.Carmenes
InstDict['ESPRESSO'] = dict()
InstDict['ESPRESSO']['None'] = espresso.Espresso
InstDict['HARPSN'] = dict()
InstDict['HARPSN']['ORIG'] = harpsn.HarpsN_ORIG
InstDict['HARPSN']['ESO'] = harpsn.HarpsN_ESO
InstDict['MAROONX'] = dict()
InstDict['MAROONX']['RED'] = maroonx.MaroonXRed
InstDict['MAROONX']['BLUE'] = maroonx.MaroonXBlue
InstDict['SOPHIE'] = dict()
InstDict['SOPHIE']['None'] = sophie.Sophie

# =============================================================================
# Define functions
# =============================================================================
def parse_args(argnames: List[str], kwargs: Dict[str, Any],
               description: Union[str, None] = None,
               parse: bool = True) -> ParamDict:
    """
    Parse the arguments 'args' using params (and their Const instances)
    if value in kwargs override default value in params

    PRIORITY: defaults << yaml << python << command-line

    :param argnames: list of strings, the arguments to add to command line
    :param kwargs: dictionary, the function call arguments to add
    :param description: str, the program description (for the help)
    :param parse: bool, if True parses arguments from command line (default),
                  if False does not check command line for arguments
    :return: ParamDict, the parameter dictionary of constants
    """
    # set function name
    func_name = __NAME__ + '.parse_args'
    # get parser
    if parse:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser = None
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
    if parse:
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
    # look for 'config_file' in kwargs
    for kwarg in kwargs:
        # force kwarg to upper case
        kwargname = kwarg.upper()
        # make sure these are in default_values
        if kwargname not in list(default_values.keys()):
            emsg = 'Python Argument "{0}" is invalid'
            raise LblException(emsg.format(kwarg))
        # only add config_file
        if kwargname == 'CONFIG_FILE':
            inputs.set(kwargname, kwargs[kwarg], source=func_name + ' [KWARGS]')
    # -------------------------------------------------------------------------
    # load parsed args into inputs
    if parse:
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
            # check if config file is path or str
            if not isinstance(config_file, (str, Path)):
                raise LblException('config file not a valid path or string')
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
                # upper case argument
                argname = argname.upper()
                # make sure these are in default_values
                if argname not in list(default_values.keys()):
                    emsg = 'Yaml Argument "{0}" is invalid'
                    raise LblException(emsg.format(argname))
                # skip Nones
                if yaml_inputs[argname] not in [None, 'None']:
                    inputs.set(argname, yaml_inputs[argname],
                               source=func_name + '[YAML]')
    # -------------------------------------------------------------------------
    # storage of command line arguments
    inputs.set('USER_KWARGS', value=[], source=func_name)
    # update params value with kwargs
    for kwarg in kwargs:
        # force kwarg to upper case
        kwargname = kwarg.upper()
        # make sure these are in default_values
        if kwargname not in list(default_values.keys()):
            emsg = 'Python Argument "{0}" is invalid'
            raise LblException(emsg.format(kwarg))
        # only if in params
        if kwargname in params:
            inputs.set(kwargname, kwargs[kwarg], source=func_name + ' [KWARGS]')
            # add to use kwargs
            msg = '\t{0}="{1}"'.format(kwargname, kwargs[kwarg])
            inputs['USER_KWARGS'].append(msg)
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


def load_instrument(args: ParamDict,
                    plogger: Union[logger.Log, None] = None) -> InstrumentsType:
    """
    Load an instrument

    :param args: ParamDict, the parameter dictionary of inputs
    :param plogger: Log instance or None - passed to keep parameters

    :return: Instrument instance
    """
    # deal with instrument not in args
    if 'INSTRUMENT' not in args:
        emsg = ('Instrument name must be defined (yaml, input or '
                'command line)')
        raise base_classes.LblException(emsg)
    if 'DATA_SOURCE' not in args:
        emsg = 'Data source must be defined (yaml, input or command line)'
        raise base_classes.LblException(emsg)
    # set instrument
    instrument = args['INSTRUMENT']
    data_source = args['DATA_SOURCE']
    # get base params
    params = parameters.params.copy()
    # deal with instrument not being a string
    if not isinstance(instrument, str):
        emsg = 'Instrument name must be a string (value={0})'
        eargs = [instrument]
        raise base_classes.LblException(emsg.format(*eargs))
    # select instrument (if instrument is allowed)
    if instrument in InstDict:
        # get the instrument dict
        source_dict = InstDict[instrument]
        # None should only be in there if there are no data sources
        if 'None' in source_dict.keys():
            inst = source_dict['None'](params)
        # use the data source to get instance
        elif data_source in source_dict:
            # get the instance from the source dictionary
            inst = source_dict[data_source](params)
        else:
            emsg = 'Data source "{0}" invalid'
            eargs = [data_source]
            raise base_classes.LblException(emsg.format(*eargs))
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
    # -------------------------------------------------------------------------
    # update log verbosity level and program name
    verbose = inst.params.get('VERBOSE', 2)
    program = inst.params.get('PROGRAM', None)
    if plogger is not None:
        plogger.update_console(verbose, program)
    else:
        log.update_console(verbose, program)
    # -------------------------------------------------------------------------
    # return instrument instances
    return inst


def make_all_directories(inst: Union[InstrumentsType],
                         skip_obj: bool = False) -> ParamDict:
    """
    Make all directories and return directory parameter dictionary

    :param inst: Instrument instance
    :param skip_obj: bool, if True skip making the OBJ directory

    :return:
    """
    # set fucntion name
    func_name = __NAME__ + '.make_all_directories()'
    # get params
    params = inst.params
    # get data directory
    data_dir = io.check_directory(params['DATA_DIR'])
    # copy over readme
    lbl_misc.copy_readme(data_dir)
    # make mask directory
    mask_dir = io.make_dir(data_dir, params['MASK_SUBDIR'], 'Mask')
    # make template directory
    template_dir = io.make_dir(data_dir, params['TEMPLATE_SUBDIR'],
                               'Templates')
    # make calib directory (for blaze and wave solutions)
    calib_dir = io.make_dir(data_dir, params['CALIB_SUBDIR'], 'Calib')
    # make science directory (for S2D files)
    science_dir = io.make_dir(data_dir, params['SCIENCE_SUBDIR'],
                              'Science')
    # make sub directory based on object science and object template
    if skip_obj:
        obj_subdir = None
    else:
        obj_subdir = inst.science_template_subdir()
    # make lblrv directory
    lblrv_dir = io.make_dir(data_dir, params['LBLRV_SUBDIR'], 'LBL RV',
                            subdir=obj_subdir)
    # make lbl reftable directory
    lbl_reftable_dir = io.make_dir(data_dir, params['LBLREFTAB_SUBDIR'],
                                   'LBL reftable')
    # make lbl rdb directory
    lbl_rdb_dir = io.make_dir(data_dir, params['LBLRDB_SUBDIR'],
                              'LBL rdb')
    # make the plot directory
    plot_dir = io.make_dir(data_dir, 'plots', 'Plot')
    # make the model directory
    model_dir = io.make_dir(data_dir, 'models', 'Model')
    # -------------------------------------------------------------------------
    # make sure we have all the model files
    inst.get_model_files(model_dir, inst.params['MODEL_REPO_URL'],
                         inst.params['MODEL_FILES'])
    # -------------------------------------------------------------------------
    # store output directories
    props = ParamDict()
    # set properties
    props.set('DATA_DIR', value=data_dir, source=func_name)
    props.set('MASK_DIR', value=mask_dir, source=func_name)
    props.set('TEMPLATE_DIR', value=template_dir, source=func_name)
    props.set('CALIB_DIR', value=calib_dir, source=func_name)
    props.set('SCIENCE_DIR', value=science_dir, source=func_name)
    props.set('OBJ_SUBDIR', value=obj_subdir, source=func_name)
    props.set('LBLRV_DIR', value=lblrv_dir, source=func_name)
    props.set('LBLRV_ALL', value=os.path.join(data_dir, params['LBLRV_SUBDIR']),
              source=func_name)
    props.set('LBLRT_DIR', value=lbl_reftable_dir, source=func_name)
    props.set('LBL_RDB_DIR', value=lbl_rdb_dir, source=func_name)
    props.set('PLOT_DIR', value=plot_dir, source=func_name)
    props.set('MODEL_DIR', value=model_dir, source=func_name)
    # return output directories
    return props


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

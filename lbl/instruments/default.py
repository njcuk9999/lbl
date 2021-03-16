#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selection of instrument functions

Created on 2021-03-15

@author: cook
"""
from astropy.io import fits
from astropy.table import Table
import argparse
import numpy as np
import os
from typing import Any, Dict, List, Tuple, Union
import yaml

from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters
from lbl.core import io
from lbl.instruments import spirou
from lbl.instruments import harps


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.default.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# load classes
ParamDict = base_classes.ParamDict

# =============================================================================
# Define classes
# =============================================================================
class Instrument:
    params = ParamDict()

    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return 'Instrument[{0}]'.format(self.name)

    def __repr__(self) -> str:
        return self.__str__()

    def _not_implemented(self, method):
        emsg = 'Must implement {0} in specific instrument class'
        raise NotImplemented(emsg.format(method))

    def mask_file(self, directory):
        """
        Make the absolute path for the mask file

        :param directory: str, the directory the file is located at

        :return: absolute path to mask file
        """
        _ = directory
        raise self._not_implemented('mask_file')

    def load_mask(self, filename: str) -> Table:
        """
        Load a mask

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        return io.load_table(filename, kind='mask table')

    def template_file(self, directory: str):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = directory
        raise self._not_implemented('template_file')

    def load_template(self, filename: str) -> Table:
        """
        Load a template

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        return io.load_table(filename, kind='template fits file')

    def blaze_file(self, directory: str):
        """
        Make the absolute path for the template file

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = directory
        raise self._not_implemented('blaze_file')

    # complex blaze return
    BlazeReturn = Union[Tuple[np.ndarray, fits.Header], None]

    def load_blaze(self, filename: str) -> BlazeReturn:
        """
        Load a blaze file

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        if filename is not None:
            return io.load_fits(filename, kind='blaze fits file')
        else:
            return None

    def science_files(self, directory: str):
        """
        List the absolute paths of all science files

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        _ = directory
        raise self._not_implemented('science_files')

    def load_science(self, filename: str) -> Tuple[np.ndarray, fits.Header]:
        """
        Load a science exposure

        :param filename: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (fits.Header)
        """
        return io.load_fits(filename, kind='science fits file')

    def ref_table_file(self, directory: str):
        """
        Make the absolute path for the ref_table file

        :param directory: str, the directory the file is located at

        :return: absolute path to ref_table file
        """
        _ = directory
        raise self._not_implemented('ref_table_file')

    def get_wave_solution(self, filename: str):
        """
        Get a wave solution from a file (for SPIROU this is from the header)
        :param filename: str, the absolute path to the file
        :return:
        """
        _ = filename
        raise self._not_implemented('get_wave_solution')

    def get_lblrv_file(self, science_filename: str, directory: str):
        """
        Construct the LBL RV file name and check whether it exists

        :param filename: str, the absolute path to the file
        :return:
        """
        _ = science_filename, directory
        raise self._not_implemented('get_wave_solution')


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
    # storage of inputs
    inputs = ParamDict()
    # update params value with kwargs
    for kwarg in kwargs:
        # force kwarg to upper case
        kwargname = kwarg.upper()
        # only if in params
        if kwargname in params:
            inputs.set(kwargname, kwargs[kwarg], source=func_name + ' [KWARGS]')
    # add args from sys.argv
    for argname in argnames:
        if argname in params:
            # get constant
            const = params.instances[argname]
            # skip constants without argument / dtype
            if const.argument is None or const.dtype is None:
                continue
            # parse argument
            parser.add_argument(const.argument, dest=const.key,
                                type=const.dtype, default=params[argname],
                                action='store', help=const.description)
    # load parsed args into inputs
    args = vars(parser.parse_args())
    for argname in args:
        inputs.set(argname, args[argname], source=func_name + '[CMD]')
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
    # parse arguments
    return inputs


def load_instrument(args: ParamDict) -> base_classes.Instrument:
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
            inst.params.set(argname, args[argname],
                            source=args.instances[argname].source)
    # return instrument instances
    return inst

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

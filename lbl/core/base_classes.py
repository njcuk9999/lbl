#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for use in lbl

Created on 2021-03-15

@author: cook
"""
from collections import UserDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from astropy.table import Table

from lbl.core import base
from lbl.core import logger

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
#
NO_THEME = [False, 'False', 'OFF', 'off', 'Off', 'None']
# get log
log = logger.Log(filename=base.LOG_FILE)


# =============================================================================
# Define classes
# =============================================================================
class LblException(Exception):
    def __init__(self, message: str, verbose: bool = True):
        """
        Constructor for LBL Exception
        :param message: str, the message to add to log + raise msg
        """
        self.message = message
        # log error
        if verbose:
            log.error(self.message)

    def __str__(self) -> str:
        """
        String representation of the Exception

        :return: return string representation
        """
        return self.message


class LblLowCCFSNR(LblException):
    def __init__(self, message):
        """
        Constructor for LBL Low CCF SNR Exception
        """
        super().__init__(message)

class LblCurveFitException(Exception):
    def __init__(self, message: str, x=None, y=None, f=None,
                 p0=None, func=None, error=None):
        """
        Constructor for LBL CurveFit Exception

        :param message: str, the message to add to log + raise msg
        :param x: the x data passed to curve_fit
        :param y: the y data passed to curve_fit
        :param f: the function passed to curve_fit
        :param p0: the p0 (guess) passed to curve_fit
        :param func: str, the function name where curve_fit was called
        :param error: str or None, if set this is the error that caused the
                      original exception
        """
        self.message = message
        self.x = x
        self.y = y
        self.f = f
        self.p0 = p0
        self.func = func
        self.error = error
        # log error
        log.error(self.message)

    def __str__(self) -> str:
        """
        String representation of the Exception

        :return: return string representation
        """
        return self.message


class Const:
    def __init__(self, key: str, source: Union[str, None] = None,
                 desc: Union[str, None] = None,
                 arg: Union[str, None] = None,
                 dtype: Union[Type, None] = None,
                 options: Union[list, None] = None,
                 comment: Union[str, None] = None,
                 fp_flag: Union[bool, None] = None):
        """
        Constant class (for storing properties of constants)

        :param key: str, the key to set in dictionary
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param options: list or None, the options (choices) to allow for
                argparse
        :param comment: str or None, if set this is the comment to add to a
                        fits header
        """
        self.key = deepcopy(key)
        # set source
        self.source = deepcopy(source)
        # set description
        self.description = deepcopy(desc)
        # set arg (for run time argument)
        self.argument = deepcopy(arg)
        # set the dtype
        self.dtype = dtype
        # the allowed options for argparse (choices)
        self.options = deepcopy(options)
        # the comment for a fits header
        self.comment = deepcopy(comment)
        # set the fp flag
        self.fp_flag = bool(fp_flag)

    def __str__(self) -> str:
        return 'Const[{0}]'.format(self.key)

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Const':
        """
        Deep copy a Const

        :return: new Const instance
        """
        return Const(self.key, self.source, self.description,
                     self.argument, self.dtype, self.options, self.comment,
                     self.fp_flag)

    def update(self, key: str, source: Union[str, None] = None,
               desc: Union[str, None] = None,
               arg: Union[str, None] = None,
               dtype: Union[Type, None] = None,
               options: Union[list, None] = None,
               comment: Union[str, None] = None,
               fp_flag: Union[bool, None] = None) -> 'Const':
        """
        Update a constant class - if value is None do not update

        :param key: str, the key to set in dictionary
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param options: list or None, the options (choices) to allow for
                argparse
        :param comment: str or None, if set this is the comment to add to a
                        fits header
        :param fp_flag: bool - if set means that this const should not be used
                        for FP (normally for header keys only)
        """
        if key is None:
            key = self.key
        # update source
        if source is None:
            source = self.source
        # set description
        if desc is None:
            desc = self.description
        # set arg (for run time argument)
        if arg is None:
            arg = self.argument
        # set the dtype
        if dtype is None:
            dtype = self.dtype
        # the allowed options for argparse (choices)
        if options is None:
            options = self.options
        # the comment for a fits header
        if comment is None:
            comment = self.comment
        # set the fp flag
        if fp_flag is None:
            fp_flag = self.fp_flag
        # return ne instance
        return Const(key, source, desc, arg, dtype, options, comment, fp_flag)


class ParamDict(UserDict):
    def __init__(self, *args, **kwargs):
        """
        Construct the parameter dictionary class

        :param args: args passed to dict constructor
        :param kwargs: kwargs passed to dict constructor
        """
        super().__init__(*args, **kwargs)
        # storage for constants
        self.instances = dict()
        # must be set (by instrument)
        self.not_none = []

    def set(self, key: str, value: Any, source: Union[str, None] = None,
            desc: Union[str, None] = None, arg: Union[str, None] = None,
            dtype: Union[Type, None] = None, not_none: bool = False,
            options: Union[list, None] = None,
            comment: Union[str, None] = None,
            fp_flag: Union[bool, None] = None):
        """
        Set a parameter in the dictionary y[key] = value

        :param key: str, the key to set in dictionary
        :param value: Any, the value to give to the dictionary item
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param not_none: bool, if True and value is None error will be raised
                         when getting parameter (so devs don't forget to
                         have this parameter defined by instrument)
        :param options: list or None, the options (choices) to allow for
                        argparse
        :param comment: str or None, if set this is the comment to add to a
                        fits header
        :param fp_flag: bool, if True this key should not be used for FP files

        :return: None - updates dict
        """
        # capitalize
        key = self._capitalize(key)
        # deal with storing not None
        if not_none:
            self.not_none.append(key)
        # set item
        self.__setitem__(key, value)
        # ---------------------------------------------------------------------
        # update / add instance
        # ---------------------------------------------------------------------
        # args for const
        cargs = [key, source, desc, arg, dtype, options, comment, fp_flag]
        # if instance already exists we just want to update keys that should
        #   be updated (i.e. not None)
        if key in self.instances and self.instances[key] is not None:
            self.instances[key] = self.instances[key].update(*cargs)
        # else we set instance from scratch
        else:
            self.instances[key] = Const(*cargs)

    def __setitem__(self, key: Any, value: Any):
        """
        Set an item from the dictionary using y[key] = value
        
        :param key: Any, the key for which to store its value
        :param value: Any, the value to store for this key
        
        :return: None - updates the dictionary 
        """
        # capitalize
        key = self._capitalize(key)
        # then do the normal dictionary setting
        self.data[key] = value

    def __getitem__(self, key: Any) -> Any:
        """
        Get an item from the dictionary using y[key]
        
        :param key: Any, the key for which to return its value
        
        :return: Any, the value of the given key 
        """
        # capitalize
        key = self._capitalize(key)
        # return from supers dictionary storage
        value = self.data[key]
        # deal with not none and value is None
        # if value is None:
        #     if key in self.not_none:
        #         emsg = ('Key {0} is None - it must be set by the instrument,'
        #                 'function inputs, command line or yaml file.')
        #         eargs = [key]
        #         raise LblException(emsg.format(*eargs))
        # return value
        return value

    def __contains__(self, key: str) -> bool:
        """
        Method to find whether CaseInsensitiveDict instance has key="key"
        used with the "in" operator
        if key exists in CaseInsensitiveDict True is returned else False
        is returned

        :param key: string, "key" to look for in CaseInsensitiveDict instance
        :type key: str

        :return bool: True if CaseInsensitiveDict instance has a key "key",
        else False
        :rtype: bool
        """
        # capitalize
        key = self._capitalize(key)
        # return True if key in keys else return False
        return key in self.data.keys()

    def __delitem__(self, key: str):
        """
        Deletes the "key" from CaseInsensitiveDict instance, case insensitive

        :param key: string, the key to delete from ParamDict instance,
                    case insensitive
        :type key: str

        :return None:
        """
        # capitalize
        key = self._capitalize(key)
        # delete key from keys
        del self.data[key]

    def copy(self) -> 'ParamDict':
        """
        Deep copy a parameter dictionary

        :return: new instance of ParamDict
        """
        new = ParamDict()
        keys, values = self.data.keys(), self.data.values()
        for key, value in zip(keys, values):
            # copy value
            new[key] = deepcopy(value)
            # copy instance
            if self.instances[key] is None:
                new.instances[key] = None
            else:
                new.instances[key] = self.instances[key].copy()
        # return parameter dictionary
        return new

    @staticmethod
    def _capitalize(key: str) -> str:
        """
        capitalize a key
        :param key: str, the key to capitalize
        :return: str, the capitalized key
        """
        if isinstance(key, str):
            return key.upper()
        else:
            return key

    def sources(self) -> Dict[str, str]:
        """
        Get the sources for this parameter dictionary (from instances)
        
        :return: dict, the source dictionary
        """
        source_dict = dict()
        for key in self.instances:
            source_dict[key] = self.instances[key].source
        return source_dict

    def __str__(self) -> str:
        """
        String representation of the parameter dictionary
        
        :return: str, the string representation of the parameter dictionary 
        """
        # get keys, values, sources
        keys = list(self.keys())
        values = list(self.values())
        sources = self.sources()
        string = 'ParamDict:'
        for it, key in enumerate(keys):
            # get source
            if key not in sources:
                source = 'Not set'
            else:
                source = sources[key]
            sargs = [key + ':', str(values[it])[:40], source]
            string += '\n{0:30s}\t{1:40s}\t// {2}'.format(*sargs)
        return string

    def __repr__(self) -> str:
        """
        String representation of the parameter dictionary
        
        :return: str, the string representation of the parameter dictionary 
        """
        return self.__str__()

    def param_table(self) -> Table:
        """
        Create a parameter table as a snapshot of the current parameters
        being used
        :return: a astropy.table table of the parameters currently being used
        """
        func_name = __NAME__ + '.ParamDict.param_table()'
        # storage
        keys, values, descriptions, sources, dtypes = [], [], [], [], []
        # get all values from paramets
        for key in list(self.data.keys()):
            # get key and value
            keys.append(key)
            values.append(str(self.data[key]))
            # deal with parameters that require an instance (parameters.py)
            if key in self.instances:
                descriptions.append(str(self.instances[key].description))
                sources.append(str(self.instances[key].source))
                dtypes.append(str(self.instances[key].dtype))
            else:
                descriptions.append('None')
                sources.append('Unknown')
                dtypes.append('Unknown')
        # add some from base
        keys += ['LBLVERSION', 'LBLDATE', 'LBLAUTHORS', 'TIMENOW']
        values += [base.__version__, base.__date__, base.__authors__,
                   base.Time.now().iso]
        descriptions += ['Current LBL version', 'Current date of LBL version',
                         'LBL authors', 'Time of parameter snapshot']
        sources += [func_name] * 4
        dtypes += ['str', 'str', 'str', 'str']
        # push into a table
        ptable = Table()
        ptable['NAME'] = keys
        ptable['VALUE'] = values
        ptable['DESCRIPTION'] = descriptions
        ptable['SOURCE'] = sources
        ptable['DATATYPE'] = dtypes
        # return ptable
        return ptable


class LBLError(Exception):
    def __init__(self, message):
        """
        Construct the LBL Error class 
        
        :param message: str, the message to print on error 
        """
        self.message = message

    def __str__(self) -> str:
        """
        String representation of the LBL Error class

        :return: str, the string representation of the LBL Error class
        """
        message = 'Error: {0}'.format(self.message)
        # return message
        return message


class HeaderTranslate:
    def __init__(self):
        self.original_keys: List[str] = []
        self.new_keys: List[str] = []
        self.functions: List[Any] = []

    def default_func(self, original_key: str, new_key: str,
                     value: Any) -> Tuple[Any, str]:
        _ = new_key
        comment = 'Translated from {0}'.format(original_key)
        return value, comment

    def add(self, original_key:str,  new_key: str, func: Optional[Any] = None):
        self.original_keys.append(original_key)
        self.new_keys.append(new_key)
        self.functions.append(func)

    def translate(self, header: Any) -> Any:
        # loop around original keys
        for it, original_key in enumerate(self.original_keys):
            # if key is in header update the key
            if original_key in header:
                new_key = self.new_keys[it]
                func = self.functions[it]
                if func is None:
                    func = self.default_func
                # get the value and the comment
                value, comment = func(original_key, new_key,
                                      header[original_key])
                header[new_key] = (value, comment)
        # return the header
        return header


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for use in lbl

Created on 2021-03-15

@author: cook
"""
from collections import UserDict
from copy import deepcopy
from typing import Any, List, Tuple, Type, Union

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
    def __init__(self, message):
        """
        Constructor for LBL Exception
        :param message: str, the message to add to log + raise msg
        """
        self.message = message
        # log error
        log.logger.error(self.message)

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
                 dtype: Union[Type, None] = None):
        """
        Constant class (for storing properties of constants)

        :param key: str, the key to set in dictionary
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
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
                     self.argument, self.dtype)


class ParamDict(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # storage for constants
        self.instances = dict()

    def set(self, key: str, value: Any,
            source: Union[str, None] = None,
            desc: Union[str, None] = None,
            arg: Union[str, None] = None,
            dtype: Union[Type, None] = None):
        """
        Set a parameter

        :param key: str, the key to set in dictionary
        :param value: Any, the value to give to the dictionary item
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set

        :return: None - updates dict
        """
        # capitalize
        key = self._capitalize(key)
        # set item
        self.__setitem__(key, value)
        # set instance
        self.instances[key] = Const(key, source, desc, arg, dtype)

    def __setitem__(self, key, value):
        # capitalize
        key = self._capitalize(key)
        # then do the normal dictionary setting
        super(ParamDict, self).__setitem__(key, value)

    def __getitem__(self, key) -> Any:
        # capitalize
        key = self._capitalize(key)
        # return from supers dictionary storage
        return super(ParamDict, self).__getitem__(key)

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
        return super(ParamDict, self).__contains__(key)

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
        super(ParamDict, self).__delitem__(key)

    def copy(self) -> 'ParamDict':
        """
        Deep copy a parameter dictionary

        :return: new instance of ParamDict
        """
        new = ParamDict()
        keys, values = self.keys(), self.values()
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
        if isinstance(key, str):
            return key.upper()

    def sources(self):
        source_dict = dict()
        for key in self.instances:
            source_dict[key] = self.instances[key].source
        return source_dict

    def __str__(self) -> str:
        # get keys, values, sources
        keys = list(self.keys())
        values = list(self.values())
        sources = self.sources()
        string = 'ParamDict:'
        for it, key in enumerate(keys):
            sargs = [key + ':', str(values[it])[:40], sources[key]]
            string += '\n{0:30s}\t{1:40s}\t// {2}'.format(*sargs)
        return string

    def __repr__(self):
        return self.__str__()


class LBLError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self) -> str:

        message = 'Error: {0}'.format(self.message)

        return message


class Instrument:
    params = ParamDict()

    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return 'Instrument[{0}]'.format(self.name)

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

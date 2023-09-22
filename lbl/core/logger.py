#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import logging
from typing import List, Union

from astropy.time import Time

from lbl.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# no theme values
NO_THEME = [False, 'False', 'OFF', 'off', 'Off', 'None']
# General level
GENERAL = 15
# CACHE for all messages
CACHE = []


# =============================================================================
# Define functions
# =============================================================================
class Log:
    def __init__(self, **kwargs):
        # get the logger
        self.logger = logging.getLogger(base.__package__)
        # set the default value to one below between INFO and DEBUG level
        self.baselevel = logging.DEBUG
        # define the levels
        self.INFO = logging.INFO
        self.WARNING = logging.WARNING
        # add a new level (GENERAL)
        self.GENERAL = GENERAL
        logging.addLevelName(self.GENERAL, 'GENERAL')
        # set logger to this as the lowest level
        self.logger.setLevel(self.baselevel)
        # deal with log overwrite
        self.format_class = ConsoleFormat
        # set the log format
        self.theme = kwargs.get('theme', None)
        self.confmt = self.format_class(theme=self.theme)
        self.filefmt = self.format_class(theme='OFF')
        # set program to None
        self.program = kwargs.get('program', None)
        # save path
        self.filepath = None
        #
        # add console
        self.console_verbosity = 2
        self._add_console(self.console_verbosity, level=self.GENERAL)
        # if we have a filename defined add a file logger
        if kwargs.get('filename', False):
            self.add_log_file(kwargs['filename'])

    def _add_console(self, verbose: int, level: int):
        # set console verbosity level
        self.console_verbosity = verbose
        # clean any handlers in logging currently
        for _ in range(len(self.logger.handlers)):
            self.logger.handlers.pop()
        # start the console log for debugging
        consolehandler = logging.StreamHandler()
        # set the name (so we can access it later)
        consolehandler.set_name('console')
        # set the format from console format
        consolehandler.setFormatter(self.confmt)
        # set the default level (INFO)
        consolehandler.setLevel(level)
        # add to the logger
        self.logger.addHandler(consolehandler)

    def _update_console(self, verbose: int, level: int,
                        program: Union[str, None] = None):
        """
        Update the console

        :param verbose:
        :param level:
        :param program:
        :return:
        """
        # set console verbosity level
        self.console_verbosity = verbose
        self.program = program
        # update console fmt
        self.confmt = self.format_class(theme=self.theme, program=program)
        self.filefmt = self.format_class(theme='OFF', program=program)
        # loop around handlers and change level
        for it in range(len(self.logger.handlers)):
            if type(self.logger.handlers[it]) == logging.StreamHandler:
                self.logger.handlers[it].setLevel(level)
            # set the format from console format
            self.logger.handlers[it].setFormatter(self.confmt)

    def add_log_file(self, filepath: str, level: Union[str, int, None] = None):
        # set file path
        self.filepath = filepath
        # get the File Handler
        filehandler = logging.FileHandler(str(filepath))
        # set the name (so we can access it later)
        filehandler.set_name('file')
        # set the log file format
        filehandler.setFormatter(self.filefmt)
        # if we have an integer just use it
        if isinstance(level, int):
            filehandler.setLevel(level)
        # set the level from level argument (using logging.{LEVEL}
        elif isinstance(level, str) and hasattr(logging, level):
            record = getattr(logging, level)
            filehandler.setLevel(record)
        # if it doesn't exist use the lowest level
        else:
            filehandler.setLevel(self.baselevel)
        # add to logger
        self.logger.addHandler(filehandler)

    def set_level(self, name: str = 'console',
                  level: Union[int, str, None] = 'DEBUG'):
        """
        Set the level of either "console" (for standard output) or "file"
        (for log file) - will print all messages at this level and above

        :param name: str, either "console" or "file" other names will do nothing
        :param level: str, int or None, the level to add if string must be
                      'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' if unset
                      does nothing
        :return:
        """
        # loop around all handlers
        for handler in self.logger.handlers:
            # get the name of this handler
            hname = handler.get_name()
            # if name matches with required name then update level
            if name.upper() == hname.upper():
                # if we have an integer level just use it
                if isinstance(level, int):
                    # update handler level
                    handler.setLevel(level)
                # if we have a level and it is valid update level
                elif level is not None and hasattr(logging, level):
                    # get level from logging
                    record = getattr(logging, level)
                    # update handler level
                    handler.setLevel(record)

    def general(self, message: str, *args, **kwargs):
        self.update_console(self.console_verbosity, self.program)
        # update cached log
        cache_logger(message, self.GENERAL, program=self.program)
        # noinspection PyProtectedMember
        self.logger._log(self.GENERAL, message, args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self.update_console(self.console_verbosity, self.program)
        # update cached log
        cache_logger(message, logging.INFO, program=self.program)
        # run logger
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.update_console(self.console_verbosity, self.program)
        # update cached log
        cache_logger(message, logging.WARNING, program=self.program)
        # run logger
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.update_console(self.console_verbosity, self.program)
        # update cached log
        cache_logger(message, logging.ERROR, program=self.program)
        # run logger
        self.logger.error(message, *args, **kwargs)

    def update_console(self, verbose: int = 2,
                       program: Union[str, None] = None):
        """
        Update the text printed to console
            0=only warnings/errors
            1=info/warnings/errors,'
            2=general/info/warning/errors

        :param verbose: int, either 0, 1, 2 see above for definition
        :param program: str, if set changes the program message (default None)

        :return: None, updates consolehandler
        """
        if verbose == 2:
            # set the default console level to GENERAL
            self._update_console(verbose, self.GENERAL, program)
        elif verbose == 1:
            # set the default console level to INFO
            self._update_console(verbose, self.INFO, program)
        elif verbose == 0:
            # set the default console level to WARNING
            self._update_console(verbose, self.WARNING, program)
        else:
            # set the default console level to GENERAL
            self._update_console(verbose, self.GENERAL, program)

    @staticmethod
    def get_cache() -> List[str]:
        """
        Get the cached log messages

        :return: list of str, the cached log messages
        """
        # copy cache to local variable
        cache = []
        for record in CACHE:
            cache.append(str(record))
        # return cache
        return cache


# Cache logger
def cache_logger(message: str, levelnum: int, program: Union[str, None] = None):
    """
    Store message formated properly to cache (for return to lbl code)

    :param message: str, the log message to store
    :param levelnum: int, the level number of the message
    :param program: str or None, if set the program id

    :return: None, updates CACHE
    """
    # need to global edit cache
    global CACHE
    # get time now
    timenow = Time.now().iso
    # -------------------------------------------------------------------------
    # deal with level num --> string
    # Replace the original format with one customized by logging level
    if levelnum == GENERAL:
        level = 'G'
    elif levelnum == logging.INFO:
        level = 'I'
    elif levelnum == logging.WARNING:
        level = 'W'
    elif levelnum >= logging.ERROR and levelnum != 999:
        level = 'E'
    elif levelnum == 999:
        level = 'D'
    else:
        level = 'G'
    # -------------------------------------------------------------------------
    if program is not None:
        record = f'{timenow}|{level}|{program}| {message}'
    else:
        record = f'{timenow}|{level}| {message}'
    # -------------------------------------------------------------------------
    # push to cache
    CACHE.append(record)


# Custom formatter
class ConsoleFormat(logging.Formatter):

    def __init__(self, fmt: str = "%(levelno)s: %(msg)s", theme=None,
                 program: Union[str, None] = None):
        # get colours
        self.cprint = Colors(theme=theme)
        # define default format
        if program is not None:
            self.fmt = '%(asctime)s.%(msecs)03d|%(levelname)-1.1s|'
            self.fmt += program + '| %(message)s'
        else:
            self.fmt = ('%(asctime)s.%(msecs)03d|%(levelname)-1.1s'
                        '| %(message)s')
        self.default = logging.Formatter(self.fmt, datefmt='%Y-%m-%d %H:%M:%S')
        # define empty format
        self.empty_fmt = '%(message)s'
        # define debug format
        self.debug_fmt = self.cprint.debug + self.fmt + self.cprint.endc
        # add a new level (GENERAL)
        self.GENERAL = 15
        self.general_fmt = self.cprint.okgreen + self.fmt + self.cprint.endc
        # define info format
        self.info_fmt = self.cprint.okblue + self.fmt + self.cprint.endc
        # define warning format
        self.warning_fmt = self.cprint.warning + self.fmt + self.cprint.endc
        # define error format
        self.error_fmt = self.cprint.fail + self.fmt + self.cprint.endc
        # define critical format
        self.critial_fmt = self.cprint.fail + self.fmt + self.cprint.endc
        # initialize parent
        logging.Formatter.__init__(self, fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        # noinspection PyProtectedMember
        format_orig = self._style._fmt
        # Replace the original format with one customized by logging level
        if record.levelno < logging.INFO:
            self._style._fmt = self.debug_fmt
        if record.levelno == self.GENERAL:
            self._style._fmt = self.general_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warning_fmt
        elif record.levelno >= logging.ERROR and record.levelno != 999:
            self._style._fmt = self.error_fmt
        elif record.levelno == 999:
            self._style._fmt = self.empty_fmt
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        logging.Formatter.datefmt = '%Y-%m-%d %H:%M:%S'
        # Restore the original format configured by the user
        self._style._fmt = format_orig
        return result


# Color class
class Colors:
    BLACK1 = '\033[90;1m'
    RED1 = '\033[1;91;1m'
    GREEN1 = '\033[92;1m'
    YELLOW1 = '\033[1;93;1m'
    BLUE1 = '\033[94;1m'
    MAGENTA1 = '\033[1;95;1m'
    CYAN1 = '\033[1;96;1m'
    WHITE1 = '\033[97;1m'
    BLACK2 = '\033[1;30m'
    RED2 = '\033[1;31m'
    GREEN2 = '\033[1;32m'
    YELLOW2 = '\033[1;33m'
    BLUE2 = '\033[1;34m'
    MAGENTA2 = '\033[1;35m'
    CYAN2 = '\033[1;36m'
    WHITE2 = '\033[1;37m'
    ENDC = '\033[0;0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, theme=None):
        if theme is None:
            self.theme = 'DARK'
        else:
            self.theme = theme
        self.endc = self.ENDC
        self.bold = self.BOLD
        self.underline = self.UNDERLINE
        self.header = self.MAGENTA1
        self.okblue = self.BLUE1
        self.okgreen = self.GREEN1
        self.ok = self.MAGENTA2
        self.warning = self.YELLOW1
        self.fail = self.RED1
        self.debug = self.BLACK1
        self.update_theme()

    def update_theme(self, theme=None):
        if theme is not None:
            self.theme = theme
        if self.theme == 'DARK':
            self.header = self.MAGENTA1
            self.okblue = self.BLUE1
            self.okgreen = self.GREEN1
            self.ok = self.MAGENTA2
            self.warning = self.YELLOW1
            self.fail = self.RED1
            self.debug = self.BLACK1
        elif self.theme in NO_THEME:
            self.header = ''
            self.okblue = ''
            self.okgreen = ''
            self.ok = ''
            self.warning = ''
            self.fail = ''
            self.debug = ''
        else:
            self.header = self.MAGENTA2
            self.okblue = self.MAGENTA2
            self.okgreen = self.BLACK2
            self.ok = self.MAGENTA2
            self.warning = self.BLUE2
            self.fail = self.RED2
            self.debug = self.GREEN2

    def print(self, message, colour=None):
        if colour in ['b', 'blue']:
            start = self.BLUE1
        elif colour in ['r', 'red']:
            start = self.RED1
        elif colour in ['g', 'green']:
            start = self.GREEN1
        elif colour in ['y', 'yellow']:
            start = self.YELLOW1
        elif colour in ['m', 'magenta']:
            start = self.MAGENTA1
        elif colour in ['k', 'black', 'grey']:
            start = self.BLACK1
        else:
            start = self.endc
        # return colour mesage
        return start + message + self.endc


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_setup.py

Creates a wrapper file to run LBL

Created on 2021-03-15

@author: cook
"""
import os.path

from astropy.time import Time

from lbl.core import base
from lbl.core import base_classes
from lbl.core import parameters
from lbl.core import wrap_setup
from lbl.core import io
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_setup.py'
__STRNAME__ = 'LBL Setup'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
InstrumentsList = select.InstrumentsList
InstrumentsType = select.InstrumentsType
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = io.log
# description for setup recipe
DESCRIPTION_COMPIL = 'Use this code to setup LBL wrapper script'
# sub-directories to create
SUB_DIRECTORIES = ['calib', 'science', 'templates']


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

    # print info
    log.info('This script creates a wrap file for LBL processing')

    # get the parameters from argument parser
    params = wrap_setup.get_args(description=DESCRIPTION_COMPIL)
    # get a list of available instruments
    instruments = wrap_setup.get_instrument_list()
    # -------------------------------------------------------------------------
    # Ask for the instrument
    # -------------------------------------------------------------------------
    if params['instrument'] in [None, 'None', 'Null']:
        # ask for instrument
        params['instrument'] = wrap_setup.ask('instrument',
                                              required=True,
                                              options=instruments)
    # -------------------------------------------------------------------------
    # Ask for the data source
    # -------------------------------------------------------------------------
    # get the options from instrument selection
    data_sources = wrap_setup.get_data_source_list(params['instrument'])
    # if we only have one data source set it to this value and don't ask
    if len(data_sources) == 1:
        # set data source
        params['data_source'] = data_sources[0]
    else:
        # conditions to ask user for data source
        cond1 = params['data_source'] in [None, 'None', 'Null']
        cond2 = params['data_source'] not in data_sources
        # warn user that data source is not in list
        if not cond1 and cond2:
            wmsg = 'Data source {0} not in list of available data sources'
            log.warning(wmsg.format(params['data_source']))
        # Ask for the data source
        if cond1 or cond2:
            # if only one data source set it
            if len(data_sources) == 1:
                # set data source
                params['data_source'] = data_sources[0]
            else:
                # ask for data source
                params['data_source'] = wrap_setup.ask('data_source',
                                                       required=True,
                                                       options=data_sources)
    # -------------------------------------------------------------------------
    # Ask for the data directory
    # -------------------------------------------------------------------------
    # conditions to ask user for data directory
    cond1 = params['data_dir'] in [None, 'None', 'Null']
    cond2 = not os.path.exists(params['data_dir'])
    # get the data directory
    while cond1 or cond2:
        # deal with data directory not existing (try to make it)
        if not cond1 and cond2:
            # noinspection PyBroadException
            try:
                question = ('Data directory {0} does not exist, '
                            'create it? [Y]es/[N]o:\t')
                uinput = input(question.format(params['data_dir']))
                if 'Y' in uinput.upper():
                    # print that we are creating directory
                    msg = 'Creating data directory: {0}'
                    log.general(msg.format(params['data_dir']))
                    # make the directory
                    os.makedirs(params['data_dir'])
                    # break from while loop
                    break
            except Exception as e:
                emsg = 'Data directory {0} does not exist and cannot be created'
                log.error(emsg.format(params['data_dir']))
                continue

        # ask for data directory
        params['data_dir'] = wrap_setup.ask('data_dir', required=True)
        # conditions to ask user for data directory
        cond1 = params['data_dir'] in [None, 'None', 'Null']
        cond2 = not os.path.exists(params['data_dir'])

    # -------------------------------------------------------------------------
    # Ask for the wrap file name
    # -------------------------------------------------------------------------
    # conditions to ask user for wrap file name
    cond1 = params['wrap_file'] in [None, 'None', 'Null']
    cond2 = os.path.exists(params['wrap_file'])
    # warn user that wrap file exists
    if (not cond1) and cond2:
        wmsg = 'Wrap file {0} already exists. Please delete or rename it first.'
        log.warning(wmsg.format(params['wrap_file']))
    # conditions to ask user for wrap file name
    if cond1 or cond2:
        # ask for wrap file name
        params['wrap_file'] = wrap_setup.ask('wrap_file', required=True)
    # must be a python file (.py)
    if not params['wrap_file'].endswith('.py'):
        params['wrap_file'] += '.py'
    # -------------------------------------------------------------------------
    # create a wrap directory
    base_wrap_file = os.path.basename(params['wrap_file'])
    wrap_dir = os.path.join(params['data_dir'], 'wrap')
    # update wrap_file name
    params['wrap_file'] = os.path.join(wrap_dir, base_wrap_file)
    # check if wrap directory exists
    if not os.path.exists(wrap_dir):
        # print progress
        log.general('\nCreating wrap directory: {0}'.format(wrap_dir))
        # make the directory
        os.makedirs(wrap_dir)
    # -------------------------------------------------------------------------
    # construct the calib and science directory paths
    for dirtype in SUB_DIRECTORIES:
        # get the directory
        dirpath = os.path.join(params['data_dir'], dirtype)
        # add to params
        params['{0}_dir'.format(dirtype)] = dirpath
        # check if directory exists
        if not os.path.exists(dirpath):
            # print progress
            msg = 'Creating {0} sub-directory: {1}'
            margs = [dirtype, dirpath]
            log.general(msg.format(*margs))
            # make the directory
            os.makedirs(dirpath)

    # -------------------------------------------------------------------------
    # get the instrument instance
    # -------------------------------------------------------------------------
    # get instrument class
    instrument = select.InstDict[params['instrument']][params['data_source']]
    # construct the instrumnet instance
    inst = instrument(parameters.params.copy())

    # -------------------------------------------------------------------------
    # Save the wrap file
    # -------------------------------------------------------------------------
    # construct a dictionary for the wrap file
    wrap_dict = dict()
    # documentation
    wrap_dict['TIME_NOW'] = Time.now().iso
    wrap_dict['LBL_VERSION'] = base.__version__
    wrap_dict['LBL_DATE'] = base.__date__
    wrap_dict['AUTHORS'] = base.__authors__
    wrap_dict['USER'] = wrap_setup.get_user()
    # -------------------------------------------------------------------------
    # instrument list
    wrap_dict['INSTRUMENTS'] = wrap_setup.instruments_str()
    wrap_dict['INSTRUMENT'] = params['instrument']
    # -------------------------------------------------------------------------
    # data sources
    wrap_dict['DATA_SOURCES'] = wrap_setup.data_sources_str()
    wrap_dict['DATA_SOURCE'] = params['data_source']
    # -------------------------------------------------------------------------
    # set the data directory
    wrap_dict['DATA_DIR'] = params['data_dir']
    # set the INPUT_FILE - defaults to all files in science/object directory
    wrap_dict['INPUT_FILE'] = '*'
    # The input science data are blaze corrected
    wrap_dict['BLAZE_CORRECTED'] = inst.params['BLAZE_CORRECTED']
    # Override the blaze filename
    wrap_dict['BLAZE_FILE'] = 'blaze.fits'
    # -------------------------------------------------------------------------
    # science criteria
    # -------------------------------------------------------------------------
    # set the data types
    wrap_dict['DATA_TYPES'] = wrap_setup.get_str_list(['SCIENCE'])
    # set the object science
    wrap_dict['OBJECT_SCIENCE'] = wrap_setup.get_str_list(['OBJECT_NAME'])
    # set the template
    wrap_dict['OBJECT_TEMPLATE'] = wrap_setup.get_str_list(['TEMPLATE_NAME'])
    # set the teffs
    wrap_dict['OBJECT_TEFF'] = '[3000]'
    # -------------------------------------------------------------------------
    # default run conditions
    # -------------------------------------------------------------------------
    # We don't reset by default (we don't want to remove uses data)
    wrap_dict['RUN_LBL_RESET'] = False
    # No tellu clean for spirou/nirps in apero/cadc
    wrap_dict['RUN_LBL_TELLUCLEAN'] = False
    if params['instrument'] in ['SPIROU', 'NIRPS_HE', 'NIRPS_HA']:
        if params['data_source'] in ['APERO', 'CADC']:
            wrap_dict['RUN_LBL_TELLUCLEAN'] = False
    # All others are True by default
    wrap_dict['RUN_LBL_TEMPLATE'] = True
    wrap_dict['RUN_LBL_MASK'] = True
    wrap_dict['RUN_LBL_COMPUTE'] = True
    wrap_dict['RUN_LBL_COMPILE'] = True
    wrap_dict['SKIP_LBL_TEMPLATE'] = True
    wrap_dict['SKIP_LBL_MASK'] = True
    wrap_dict['SKIP_LBL_COMPUTE'] = True
    wrap_dict['SKIP_LBL_COMPILE'] = True
    # -------------------------------------------------------------------------
    # other settings
    # -------------------------------------------------------------------------
    # default resproj tables
    resproj_tables = dict()
    resproj_tables['DTEMP3500'] = 'temperature_gradient_3500.fits'
    # set resproj tables
    wrap_dict['RESPROJ_TABLES'] = resproj_tables
    # set rotational broadening
    wrap_dict['ROTBROAD'] = '[{0}]'.format(','.join(['']))
    # set plotting
    wrap_dict['DO_PLOT'] = False
    # -------------------------------------------------------------------------
    # if instrument is Generic we have push keywords into the wrap file
    if params['instrument'] == 'Generic':
        wrap_dict = wrap_setup.generic_instrument(wrap_dict)
    # -------------------------------------------------------------------------
    # print progress
    log.info('Saving wrap file: {0}. '.format(params['wrap_file']))
    log.info('\t\tPlease edit the science criteria and run/skip info manually')
    log.info('\t\t\te.g. {OBJECT_NAME} and {TEMPLATE_NAME}')
    log.info('\t\t\tPlease get to know all options in this wrap file before '
             'running!')

    # push into file
    wrap_default = wrap_setup.create_wrap_file(params, wrap_dict)
    print('\n')
    # print a reminder about calib/science/template directories
    for dirtype in SUB_DIRECTORIES:
        pargs = (dirtype, params['{0}_dir'.format(dirtype)])
        log.info('Don\'t forget to copy your {0} data to: {1}'.format(*pargs))
        # deal with an extra reminder for science files
        if dirtype == 'science':
            log.general('\t\tRemember that science files should be in an '
                        '"OBJECT_NAME" sub-directory')

    # Now you can run lbl
    print('\n')
    log.info('You can now run LBL by running the following command:')
    log.general('\t\t python {0}'.format(params['wrap_file']))

    # For more info
    print('\n')
    log.info('For more information see https://lbl.exoplanets.ca')
    print('\n\n\n\n')

    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    # do not remove this line
    logmsg = log.get_cache()
    # return
    return locals()


def run():
    """
    Main function for running the setup from the command line
    (avoids returning locals)

    :return:
    """
    # run the main function
    _ = main()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    _ = main()

# =============================================================================
# End of code
# =============================================================================

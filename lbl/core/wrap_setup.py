#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-07-05 at 03:36

@author: cook
"""
from typing import Any, Dict, List, Optional
import argparse
import os
import getpass
import socket

from lbl.core import base
from lbl.core import io
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'setup.py'
__STRNAME__ = 'LBL Compil'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get the log
log = io.log
# -----------------------------------------------------------------------------


# =============================================================================
# Define functions
# =============================================================================
def get_instrument_list() -> List[str]:
    """
    Get the list of instruments available
    :return:
    """
    # get list of instruments
    instruments = []
    # loop around instrument classes
    for instrument in base.INSTRUMENTS:
        instruments.append(instrument)
    # return instruments
    return instruments


def get_data_source_list(instrument: str) -> List[str]:
    """
    Get the data sources for a given instrument

    :param instrument: str, the instrument name
    :return:
    """
    # check if instrument is in select.InstDict
    if instrument not in select.InstDict:
        emsg = 'Instrument {0} not in select.InstDict'.format(instrument)
        log.error(emsg)
    # get data sources
    return list(select.InstDict[instrument].keys())


def get_args(description: str):
    # start argument parser
    parser = argparse.ArgumentParser(description=description)
    # add instrument argument
    parser.add_argument('--instrument', type=str,
                   default='None',
                   action='store', help='The instrument name',
                   choices=get_instrument_list())
    # add data source argument
    parser.add_argument('--data_source', type=str,
                   default='Null',
                   action='store', help='The data source')
    # add the data directory
    parser.add_argument('--data_dir', type=str,
                   default='None',
                   action='store', help='The data directory')
    # add the name of the wrap file
    parser.add_argument('wrap_file', type=str,
                   default='None',
                   action='store', help='The wrap file name')
    # parse the arguments
    args = parser.parse_args()
    # push into dictionary and return
    return vars(args)


def get_user() -> str:
    """
    Get the user name and host name in os independent way
    :return:
    """
    username = getpass.getuser()
    hostname = socket.gethostname()

    return f'{username}@{hostname}'


def get_str_list(input_list):
    """
    Get the string list for the data sources
    :return:
    """
    # empty source string at first
    str_list = '[{0}]'
    # string items must have quotes
    str_items = ["'{0}'".format(item) for item in input_list]
    # return source string
    return str_list.format(', '.join(str_items))


def ask(name: str, required: bool, options: Optional[List[str]] = None):
    # ask for input
    if options is not None:
        print('\n\n')
        str_options = ', '.join(options)
        question = ('Please enter the {0} (from: {1})'.format(name, str_options))
        # loop around until we get a valid input
        while True:
            # get input
            value = input(question + ':\t').strip()
            # check if value is in options
            if value in options:
                return value
            # if not required and value is None return None
            if not required and value in [None, 'None', 'Null']:
                return None
            # print error
            print('Error: {0} not in options'.format(value))
    else:
        print('\n\n')
        question = ('Please enter the {0}'.format(name))
        # get input
        value = input(question + ':\t')
        # return value
        return value


def instruments_str() -> str:
    """
    Produce the comment for instruments
    :return:
    """
    # empty source string at first
    source_str = ''
    # loop around instruments
    for instrument in base.INSTRUMENTS:
        source_str += '\n\t#\t\t{0}'.format(instrument)
    # return instruments string
    return source_str


def data_sources_str():
    """
    Produce the comment for data_soruces
    :return:
    """
    # empty source string at first
    source_str = ''
    # loop around instruments
    for instrument in select.InstDict:
        # get data sources
        dsources = select.InstDict[instrument].keys()
        # add to source string
        source_str += '\n\t#\t\t{0}: {1}'.format(instrument, ' or '.join(dsources))
    # return source string
    return source_str


def create_wrap_file(params: dict, wrap_dict: dict):
    """
    Push the wrap_dict into the wrap_default.txt and save as a file

    :param params:
    :param wrap_dict:
    :return:
    """
    # deal with no other settings
    if 'OTHER_SETTINGS' not in wrap_dict:
        wrap_dict['OTHER_SETTINGS'] = ''

    # get the resource path
    resource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'resources')
    # get the default text for the wrap file
    default_wrap_file = os.path.join(resource_path, 'wrap_default.txt')

    # load the default wrap file
    with open(default_wrap_file, 'r') as wrapf:
        wrap_text = wrapf.read()

    # using the wrap dictionary replace the values in the wrap text
    wrap_text = wrap_text.format(**wrap_dict)

    # write wrap_file
    with open(params['wrap_file'], 'w') as wrapf:
        wrapf.write(wrap_text)


def generic_instrument(wrap_dict: Dict[str, Any]) -> Dict[str, Any]:
    """

    :param wrap_dict:
    :return:
    """

    # storage for text to add to wrap_dict
    OTHER_SETTINGS = """
    
    # IMPORTANT: For a generic instrument these MUST be set!

    # Instrument name (avoid spaces and punctuation other than underscores)
    # Example: 'ESPRESSO'
    # TYPE: STRING
    rparams['GENERIC_INSTRUMENT'] = None
    
    # Data source (default is 'None' but can be set if you have multiple modes per instrument)
    # Can be a different reduction pipeline or instrument mode or both
    # Example: 'None' or 'RED' or 'H' or 'CADC'
    # TYPE: STRING
    rparams['GENERIC_DATA_SOURCE'] = None
    
    # The minimum wavelength for the instrument/mode
    # Example: 377.189
    # Type: FLOAT
    rparams['GENERIC_WAVEMIN'] = None 
    
    # The maxmimum wavelength for the instrument/mode
    # Example: 790.788
    # Type: FLOAT
    rparams['GENERIC_WAVEMAX'] = None

    # add instrument earth location
    #    (for use in astropy.coordinates.EarthLocation)
    #    Must be in this list: https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json
    # Example: 'paranal' or 'gemini_south'
    # Type: STRING
    rparams['EARTH_LOCATION'] = None

    # define the High pass width in km/s
    # Example: 256
    # Type: INT
    rparams['HP_WIDTH'] = None

    # define the SNR cut off threshold
    # Example: 15
    # Type: FLOAT
    rparams['SNR_THRESHOLD'] = None

    # define which bands to use for the clean CCF
    # from lbl.core.astro.py
    # currently: r, i, y, h, k
    # Example: ['r']
    # Type: LIST of STRINGS
    rparams['CCF_CLEAN_BANDS'] = None

    # define the plot order for the compute rv model plot
    # Example: [60]
    # Type: LIST of INTS
    rparams['COMPUTE_MODEL_PLOT_ORDERS'] = None

    # define the compil minimum wavelength allowed for lines [nm]
    # Example: 450
    # Type: FLOAT
    rparams['COMPIL_WAVE_MIN'] = None

    # define the compil maximum wavelength allowed for lines [nm]
    # Example: 750
    # Type: FLOAT
    rparams['COMPIL_WAVE_MAX'] = None

    # define the maximum pixel width allowed for lines [pixels]
    # Example: 50
    # Type: INT
    rparams['COMPIL_MAX_PIXEL_WIDTH'] = None

    # define min likelihood of correlation with BERV (-1 to turn off)
    # Example: -1 
    # Type: INT
    rparams['COMPIL_CUT_PEARSONR'] = -1

    # define the CCF e-width to use for FP files
    # Example: 3.0 
    # Type: FLOAT
    rparams['COMPIL_FP_EWID'] = None

    # define whether to add the magic "binned wavelength" bands rv
    # Example: True
    # Type: BOOL
    rparams['COMPIL_ADD_UNIFORM_WAVEBIN'] = None

    # define the number of bins used in the magic "binned wavelength" bands
    # Example: 15
    # Type: INT
    rparams['COMPIL_NUM_UNIFORM_WAVEBIN'] = None

    # define the first band (from get_binned_parameters) to plot (band1)
    # u,g,r,i,z,y,j,h,k
    # Example: 'g'
    # Type: STRING
    rparams['COMPILE_BINNED_BAND1'] = None

    # define the second band (from get_binned_parameters) to plot (band2)
    #    this is used for colour   band2 - band3
    # u,g,r,i,z,y,j,h,k
    # Example: 'r'
    # Type: STRING
    rparams['COMPILE_BINNED_BAND2'] = None

    # define the third band (from get_binned_parameters) to plot (band3)
    #    this is used for colour   band2 - band3
    # u,g,r,i,z,y,j,h,k
    # Example: 'i'
    # Type: STRING
    rparams['COMPILE_BINNED_BAND3'] = None

    # define the reference wavelength used in the slope fitting in nm
    # Example: 650
    # Type: FLOAT
    rparams['COMPIL_SLOPE_REF_WAVE'] = None

    # define readout noise per instrument (assumes ~5e- and 10 pixels)
    # Example: 15
    # Type: FLOAT
    rparams['READ_OUT_NOISE'] = None

    # Define the minimum allowed SNR in a pixel to add it to the mask
    # Example: 20
    # Type: FLOAT
    rparams['MASK_SNR_MIN'] = None

    # blaze smoothing size (s1d template)
    # Example: 20
    # Type: FLOAT
    rparams['BLAZE_SMOOTH_SIZE'] = None

    # blaze threshold (s1d template)
    # Example: 0.2
    # Type: FLOAT
    rparams['BLAZE_THRESHOLD'] = None

    # define the size of the berv bins in m/s
    # Example: 3000
    # Type: FLOAT
    rparams['BERVBIN_SIZE'] = None

    # define whether to do the tellu-clean
    # Example: True
    # Type: BOOL
    rparams['DO_TELLUCLEAN'] = None

    # define the dv offset for tellu-cleaning in km/s
    # Example: 0.0
    # Type: FLOAT
    rparams['TELLUCLEAN_DV0'] = None

    # Define the lower wave limit for the absorber spectrum masks in nm
    # Example: 550
    # Type: FLOAT
    rparams['TELLUCLEAN_MASK_DOMAIN_LOWER'] = None

    # Define the upper wave limit for the absorber spectrum masks in nm
    # Example: 670
    # Type: FLOAT
    rparams['TELLUCLEAN_MASK_DOMAIN_UPPER'] = None

    # Define whether to force using airmass from header
    # Example: True
    # Type: BOOL
    rparams['TELLUCLEAN_FORCE_AIRMASS'] = None

    # Define the CCF scan range in km/s
    # Example: 150
    # Type: FLOAT
    rparams['TELLUCLEAN_CCF_SCAN_RANGE'] = None

    # Define the maximum number of iterations for the tellu-cleaning loop
    # Example: 20
    # Type: INT
    rparams['TELLUCLEAN_MAX_ITERATIONS'] = None

    # Define the kernel width in pixels
    # Example: 1.4
    # Type: FLOAT
    rparams['TELLUCLEAN_KERNEL_WID'] = None

    # Define the gaussian shape (2=pure gaussian, >2=boxy)
    # Example: 2.2
    # Type: FLOAT
    rparams['TELLUCLEAN_GAUSSIAN_SHAPE'] = None

    # Define the wave grid lower wavelength limit in nm
    # Example: 350
    # Type: FLOAT
    rparams['TELLUCLEAN_WAVE_LOWER'] = None

    # Define the wave griv upper wavelength limit
    # Example: 750
    # Type: FLOAT
    rparams['TELLUCLEAN_WAVE_UPPER'] = None

    # Define the transmission threshold exp(-1) at which tellurics are
    #     uncorrectable
    # Example: -1
    # Type: FLOAT
    rparams['TELLUCLEAN_TRANSMISSION_THRESHOLD'] = None

    # Define the sigma cut threshold above which pixels are removed from fit
    # Example: 10
    # Type: FLOAT
    rparams['TELLUCLEAN_SIGMA_THRESHOLD'] = None

    # Define whether to recenter the CCF on the first iteration
    # Example: False
    # Type: BOOL
    rparams['TELLUCLEAN_RECENTER_CCF'] = None

    # Define whether to recenter the CCF of others on the first iteration
    # Example: False
    # Type: BOOL
    rparams['TELLUCLEAN_RECENTER_CCF_FIT_OTHERS'] = None

    # Define the default water absorption to use
    # Example: 5.0
    # Type: FLOAT
    rparams['TELLUCLEAN_DEFAULT_WATER_ABSO'] = None

    # Define the lower limit on valid exponent of water absorbers
    # Example: 0.05
    # Type: FLOAT
    rparams['TELLUCLEAN_WATER_BOUNDS_LOWER'] = None

    # Define the upper limit on valid exponent of water absorbers
    # Example: 15
    # Type: FLOAT
    rparams['TELLUCLEAN_WATER_BOUNDS_UPPER'] = None

    # Define the lower limit on valid exponent of other absorbers
    # Example: 0.05
    # Type: FLOAT
    rparams['TELLUCLEAN_OTHERS_BOUNDS_LOWER'] = None
    
    # Define the upper limit on valid exponent of other absorbers
    # Example: 15
    # Type: FLOAT
    rparams['TELLUCLEAN_OTHERS_BOUNDS_UPPER'] = None
    
    # ---------------------------------------------------------------------
    # Parameters for the template construction
    # ---------------------------------------------------------------------
    # max number of bins for the median of the template. Avoids handling
    # too many spectra at once.
    # Example: 19
    # Type: INT
    rparams['TEMPLATE_MEDBINMAX'] = None

    # maximum RMS between the template and the median of the template
    # to accept the median of the template as a good template. If above
    # we iterate once more. Expressed in m/s
    # Example: 100
    # Type: FLOAT
    rparams['MAX_CONVERGENCE_TEMPLATE_RV'] = None
    
    """
    # add to wrap dict
    wrap_dict['OTHER_SETTINGS'] = OTHER_SETTINGS
    # return wrap dict
    return wrap_dict


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-07-05 at 03:36

@author: cook
"""
from typing import List, Optional
import argparse
import os
import getpass
import socket

from lbl.core import base
from lbl.core import base_classes
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


    # add instrument earth location
    #    (for use in astropy.coordinates.EarthLocation)
    #    Must be in this list: https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json
    EARTH_LOCATION = 'paranal'

    # The input science data are blaze corrected
    BLAZE_CORRECTED = False

    # define the High pass width in km/s
    HP_WIDTH = 256

    # define the SNR cut off threshold
    SNR_THRESHOLD = 10

    # define which bands to use for the clean CCF
    # from lbl.core.astro.py
    # currently: r, i, y, h, k
    CCF_CLEAN_BANDS = ['r']

    # define the plot order for the compute rv model plot
    COMPUTE_MODEL_PLOT_ORDERS = [60]

    # define the compil minimum wavelength allowed for lines [nm]
    COMPIL_WAVE_MIN = 450

    # define the compil maximum wavelength allowed for lines [nm]
    COMPIL_WAVE_MAX = 750

    # define the maximum pixel width allowed for lines [pixels]
    COMPIL_MAX_PIXEL_WIDTH = 50

    # define min likelihood of correlation with BERV
    COMPIL_CUT_PEARSONR = -1

    # define the CCF e-width to use for FP files
    COMPIL_FP_EWID = 3.0

    # define whether to add the magic "binned wavelength" bands rv
    COMPIL_ADD_UNIFORM_WAVEBIN = True

    # define the number of bins used in the magic "binned wavelength" bands
    COMPIL_NUM_UNIFORM_WAVEBIN = 15

    # define the first band (from get_binned_parameters) to plot (band1)
    # u,g,r,i,z,y,j,h,k
    COMPILE_BINNED_BAND1 = 'g'

    # define the second band (from get_binned_parameters) to plot (band2)
    #    this is used for colour   band2 - band3
    # u,g,r,i,z,y,j,h,k
    COMPILE_BINNED_BAND2 = 'r'

    # define the third band (from get_binned_parameters) to plot (band3)
    #    this is used for colour   band2 - band3
    # u,g,r,i,z,y,j,h,k
    COMPILE_BINNED_BAND3 = 'i'

    # define the reference wavelength used in the slope fitting in nm
    COMPIL_SLOPE_REF_WAVE = 650

    # define readout noise per instrument (assumes ~5e- and 10 pixels)
    READ_OUT_NOISE = 15

    # Define the minimum allowed SNR in a pixel to add it to the mask
    MASK_SNR_MIN = 20

    # blaze smoothing size (s1d template)
    BLAZE_SMOOTH_SIZE = 20

    # blaze threshold (s1d template)
    BLAZE_THRESHOLD = 0.2

    # define the size of the berv bins in m/s
    BERVBIN_SIZE = 3000

    # define whether to do the tellu-clean
    DO_TELLUCLEAN = True

    # define the dv offset for tellu-cleaning in km/s
    TELLUCLEAN_DV0 = 0

    # Define the lower wave limit for the absorber spectrum masks in nm
    TELLUCLEAN_MASK_DOMAIN_LOWERe = 550

    # Define the upper wave limit for the absorber spectrum masks in nm
    TELLUCLEAN_MASK_DOMAIN_UPPER = 670

    # Define whether to force using airmass from header
    TELLUCLEAN_FORCE_AIRMASS = True

    # Define the CCF scan range in km/s
    TELLUCLEAN_CCF_SCAN_RANGE = 150

    # Define the maximum number of iterations for the tellu-cleaning loop
    TELLUCLEAN_MAX_ITERATIONS = 20

    # Define the kernel width in pixels
    TELLUCLEAN_KERNEL_WID = 1.4

    # Define the gaussian shape (2=pure gaussian, >2=boxy)
    TELLUCLEAN_GAUSSIAN_SHAPE = 2.2

    # Define the wave grid lower wavelength limit in nm
    TELLUCLEAN_WAVE_LOWER = 350

    # Define the wave griv upper wavelength limit
    TELLUCLEAN_WAVE_UPPER = 750

    # Define the transmission threshold exp(-1) at which tellurics are
    #     uncorrectable
    TELLUCLEAN_TRANSMISSION_THRESHOLD = -1

    # Define the sigma cut threshold above which pixels are removed from fit
    TELLUCLEAN_SIGMA_THRESHOLD = 10

    # Define whether to recenter the CCF on the first iteration
    TELLUCLEAN_RECENTER_CCF = False

    # Define whether to recenter the CCF of others on the first iteration
    TELLUCLEAN_RECENTER_CCF_FIT_OTHERS = True

    # Define the default water absorption to use
    TELLUCLEAN_DEFAULT_WATER_ABSO = 5.0

    # Define the lower limit on valid exponent of water absorbers
    TELLUCLEAN_WATER_BOUNDS_LOWER = 0.05

    # Define the upper limit on valid exponent of water absorbers
    TELLUCLEAN_WATER_BOUNDS_UPPER = 15

    # Define the lower limit on valid exponent of other absorbers
    TELLUCLEAN_OTHERS_BOUNDS_LOWER = 0.05

    # Define the upper limit on valid exponent of other absorbers
    TELLUCLEAN_OTHERS_BOUNDS_UPPER = 15

    # ---------------------------------------------------------------------
    # Parameters for the template construction
    # ---------------------------------------------------------------------
    # max number of bins for the median of the template. Avoids handling
    # too many spectra at once.
    TEMPLATE_MEDBINMAX = 19

    # maximum RMS between the template and the median of the template
    # to accept the median of the template as a good template. If above
    # we iterate once more. Expressed in m/s
    MAX_CONVERGENCE_TEMPLATE_RV = 100


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

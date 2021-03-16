#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import os
from typing import Union

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import default
from lbl.science import general

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = base_classes.Instrument
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
# add arguments (must be in parameters.py)
ARGS = ['INSTRUMENT', 'CONFIG_FILE', 'DATA_DIR']
# TODO: Fill out
DESCRIPTION = 'Use this code to compute the LBL'


# =============================================================================
# Define functions
# =============================================================================
def main(**kwargs):
    # deal with parsing arguments
    args = default.parse_args(ARGS, kwargs, DESCRIPTION)
    # load instrument
    inst = default.load_instrument(args)
    # run __main__
    try:
        return __main__(inst, inst.params)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected Error: {0}: {1}'
        eargs = [type(e), str(e)]
        raise LblException(emsg.format(*eargs))


def __main__(inst: Union[Instrument, None] = None,
             params: Union[ParamDict, None] = None, **kwargs):
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or params is None:
        # deal with parsing arguments
        args = default.parse_args(ARGS, kwargs, DESCRIPTION)
        # load instrument
        inst = default.load_instrument(args)
        assert isinstance(inst, Instrument), 'inst must be Instrument class'
        params = inst.params

    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    # get data directory
    data_dir = params['DATA_DIR']
    # make mask directory
    mask_dir = io.make_dir(data_dir, 'masks', 'Mask')
    # make template directory
    template_dir = io.make_dir(data_dir, 'templates', 'Templates')
    # make calib directory (for blaze and wave solutions)
    calib_dir = io.make_dir(data_dir, 'calib', 'Calib')
    # make science directory (for S2D files)
    science_dir = io.make_dir(data_dir, 'science', 'Science')
    # make lblrv directory
    lblrv_dir = io.make_dir(data_dir, 'lblrv', 'LBL RV')
    # make lbl reftable directory
    lbl_reftable_dir = io.make_dir(data_dir, 'lblreftable', 'LBL reftable')
    # make lbl rdb directory
    lbl_rdb_dir = io.make_dir(data_dir, 'lblrdb', 'LBL rdb')

    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # mask filename
    mask_file = inst.mask_file(mask_dir)
    # template filename
    template_file = inst.template_file(template_dir)
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    # science filenames
    science_files = inst.science_files(science_dir)
    # reftable filename (None if not set)
    reftable_file, reftable_exists = inst.ref_table_file(lbl_reftable_dir)

    # -------------------------------------------------------------------------
    # Step 3: Load or make ref_dict
    # -------------------------------------------------------------------------
    ref_table = general.make_ref_dict(inst, reftable_file, reftable_exists,
                                      science_files, mask_file)
    # load the mask header
    _, mask_hdr = io.load_fits(mask_file, kind='mask fits file')
    # get info on template systvel for splining correctly
    systemic_vel = -1000 * io.get_hkey(mask_hdr, 'SYSTVEL')

    # -------------------------------------------------------------------------
    # Step 4: spline the template
    # -------------------------------------------------------------------------
    splines = general.spline_template(inst, template_file, systemic_vel)

    # -------------------------------------------------------------------------
    # Step 5: Loop around science files
    # -------------------------------------------------------------------------
    # load bad odometer codes
    bad_hdr_keys, bad_hdr_key = inst.load_bad_hdr_keys()

    # loop through each science file
    for it, science_file in enumerate(science_files):
        # ---------------------------------------------------------------------
        # log process
        msg = 'Processing file {0} / {1}'
        margs = [it + 1, len(science_file)]
        log.logger.info(msg.format(*margs))
        # ---------------------------------------------------------------------
        # get lbl rv file and check whether it exists
        lblrv_file, lblrv_exists = inst.get_lblrv_file(science_file, lblrv_dir)
        # if file exists and we are skipping done files
        if lblrv_exists and params['SKIP_DONE']:
            # log message about skipping
            log.logger.info('\t\tFile exists and skipping activated. '
                            'Skipping file.')
            # skip
            continue
        # ---------------------------------------------------------------------
        # load file
        sci_data, sci_hdr = io.load_fits(science_file, kind='science fits file')
        # ---------------------------------------------------------------------
        # check for bad files (via a header key)
        # ---------------------------------------------------------------------
        # check we have a bad hdr key
        if bad_hdr_key is not None and bad_hdr_key in sci_hdr:
            # get bad header key
            sci_bad_hdr_key = io.get_hkey(sci_hdr, bad_hdr_key)
            # if sci_bad_hdr_key in bad_hdr_keys
            if sci_bad_hdr_key in bad_hdr_keys:
                # log message about bad header key
                log.logger.info('\t\tFile is known to be bad. Skipping file.')
                # skip
                continue
        # ---------------------------------------------------------------------
        # quality control on snr
        # ---------------------------------------------------------------------
        # get snr key
        snr_key = params['KW_SNR']
        snr_limit = params['SNR_THRESHOLD']
        # check we have snr key in science header
        if snr_key in sci_hdr:
            # get snr value
            snr_value = io.get_hkey(sci_hdr, snr_key)
            # check if value is less than limit
            if snr_value < snr_limit:
                # log message
                msg = '\t\tSNR < {0} (SNR = {1}). Skipping file.'
                margs = [snr_limit, snr_value]
                log.logger.info(msg.format(*margs))
                # skip
                continue
        # ---------------------------------------------------------------------
        # compute rv
        # ---------------------------------------------------------------------
        # general.compute_rv

        # ---------------------------------------------------------------------
        # save to file
        # ---------------------------------------------------------------------


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

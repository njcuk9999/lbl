#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.science import general
from lbl.resources import misc


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_compute.py'
__STRNAME__ = 'LBL Compute'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
InstrumentsList = select.InstrumentsList
InstrumentsType = select.InstrumentsType
ParamDict = base_classes.ParamDict
LblException = base_classes.LblException
log = base_classes.log
# add arguments (must be in parameters.py)
ARGS_COMPUTE = [
                # core
                'INSTRUMENT', 'CONFIG_FILE',
                # directory
                'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
                'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
                # science
                'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'INPUT_FILE', 'TEMPLATE_FILE',
                'BLAZE_FILE', 'HP_WIDTH', 'USE_NOISE_MODEL',
                # plotting
                'PLOT', 'PLOT_COMPUTE_CCF', 'PLOT_COMPUTE_LINES',
                # other
                'SKIP_DONE'
                ]
# TODO: Etienne - Fill out
DESCRIPTION_COMPUTE = 'Use this code to compute the LBL rv'


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
    # deal with parsing arguments
    args = select.parse_args(ARGS_COMPUTE, kwargs, DESCRIPTION_COMPUTE)
    # load instrument
    inst = select.load_instrument(args)
    # move log file (now we have data directory)
    misc.move_log(inst.params['DATA_DIR'], __NAME__)
    # print splash
    misc.splash(name=__STRNAME__, instrument=inst.name,
                cmdargs=inst.params['COMMAND_LINE_ARGS'])
    # run __main__
    try:
        namespace =  __main__(inst)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected lbl_compute error: {0}: {1}'
        eargs = [type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    # end code
    misc.end(__NAME__)
    # return local namespace
    return namespace


def __main__(inst: InstrumentsType, **kwargs):
    """
    The main recipe function - all code dealing with recipe functionality
    should go here

    :param inst: Instrument instance
    :param kwargs: kwargs to parse to instrument (only use if inst is None)
                   anything in params can be parsed (overwrites instrumental
                   and default parameters)

    :return: all variables in local namespace
    """
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or inst.params is None:
        # deal with parsing arguments
        args = select.parse_args(ARGS_COMPUTE, kwargs, DESCRIPTION_COMPUTE)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    # get data directory
    data_dir = inst.params['DATA_DIR']
    # copy over readme
    misc.copy_readme(data_dir)
    # make mask directory
    mask_dir = io.make_dir(data_dir, inst.params['MASK_SUBDIR'], 'Mask')
    # make template directory
    template_dir = io.make_dir(data_dir, inst.params['TEMPLATE_SUBDIR'],
                               'Templates')
    # make calib directory (for blaze and wave solutions)
    calib_dir = io.make_dir(data_dir, inst.params['CALIB_SUBDIR'], 'Calib')
    # make science directory (for S2D files)
    science_dir = io.make_dir(data_dir, inst.params['SCIENCE_SUBDIR'],
                              'Science')
    # make sub directory based on object science and object template
    obj_subdir = inst.science_template_subdir()
    # make lblrv directory
    lblrv_dir = io.make_dir(data_dir, inst.params['LBLRV_SUBDIR'], 'LBL RV',
                            subdir=obj_subdir)
    # make lbl reftable directory
    lbl_reftable_dir = io.make_dir(data_dir, inst.params['LBLREFTAB_SUBDIR'],
                                   'LBL reftable')
    # make lbl rdb directory
    lbl_rdb_dir = io.make_dir(data_dir, inst.params['LBLREFTAB_SUBDIR'],
                              'LBL rdb')
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
    # Step 3: Load blaze file if set
    # -------------------------------------------------------------------------
    if blaze_file is not None:
        blaze = inst.load_blaze(blaze_file)
    else:
        blaze = None
    # -------------------------------------------------------------------------
    # Step 4: Load or make ref_dict
    # -------------------------------------------------------------------------
    ref_table = general.make_ref_dict(inst, reftable_file, reftable_exists,
                                      science_files, mask_file)
    # get the systemic velocity for mask
    systemic_vel = inst.get_mask_systemic_vel(mask_file)
    # -------------------------------------------------------------------------
    # Step 5: spline the template
    # -------------------------------------------------------------------------
    splines = general.spline_template(inst, template_file, systemic_vel)
    # -------------------------------------------------------------------------
    # Step 6: Loop around science files
    # -------------------------------------------------------------------------
    # load bad odometer codes
    bad_hdr_keys, bad_hdr_key = inst.load_bad_hdr_keys()
    # store all systemic velocities and mid exposure times in mjd
    systemic_all = np.full(len(science_files), np.nan)
    mjdate_all = np.zeros(len(science_files)).astype(float)
    # store the initial ccf_ewidth value
    ccf_ewidth = None
    # flag to take a completely new rv measurement
    reset_rv = True
    # time stats
    mean_time, std_time, time_left = np.nan, np.nan, ''
    all_durations = []
    count = 0
    # loop through each science file
    for it, science_file in enumerate(science_files):
        # ---------------------------------------------------------------------
        # 6.1 log process
        # ---------------------------------------------------------------------
        # number left
        nleft = len(science_files) - (it + 1)
        # standard loop message
        log.info('*' * 79)
        msg = 'Processing file {0} / {1}   ({2} left)'
        margs = [it + 1, len(science_files), nleft]
        log.info(msg.format(*margs))
        log.info('*' * 79)
        # add time stats
        if count > 3:
            msgs = ['\tDuration per file {0:.2f}+-{1:.2f} s']
            msgs += ['\tTime left to completion: {2}']
            margs = [mean_time, std_time, time_left]
            for msg in msgs:
                log.general(msg.format(*margs))
        # ---------------------------------------------------------------------
        # 6.2 get lbl rv file and check whether it exists
        # ---------------------------------------------------------------------
        lblrv_file, lblrv_exists = inst.get_lblrv_file(science_file, lblrv_dir)
        # if file exists and we are skipping done files
        if lblrv_exists and inst.params['SKIP_DONE']:
            # log message about skipping
            log.general('\t\tFile exists and skipping activated. '
                        'Skipping file.')
            # skip
            continue
        # ---------------------------------------------------------------------
        # 6.3 load science file
        # ---------------------------------------------------------------------
        sci_data, sci_hdr = io.load_fits(science_file, kind='science fits file')
        # ---------------------------------------------------------------------
        # 6.4 load blaze if not set above
        # ---------------------------------------------------------------------
        if blaze is None:
            blaze = inst.load_blaze_from_science(sci_hdr, calib_dir)
        # ---------------------------------------------------------------------
        # 6.5 check for bad files (via a header key)
        # ---------------------------------------------------------------------
        # check we have a bad hdr key
        if bad_hdr_key is not None and bad_hdr_key in sci_hdr:
            # get bad header key
            sci_bad_hdr_key = io.get_hkey(sci_hdr, bad_hdr_key)
            # if sci_bad_hdr_key in bad_hdr_keys
            if str(sci_bad_hdr_key) in bad_hdr_keys:
                # log message about bad header key
                log.general('\t\tFile is known to be bad. Skipping file.')
                # skip
                continue
        # ---------------------------------------------------------------------
        # 6.6 quality control on snr
        # ---------------------------------------------------------------------
        # get snr key
        snr_key = inst.params['KW_SNR']
        snr_limit = inst.params['SNR_THRESHOLD']
        # check we have snr key in science header
        if snr_key in sci_hdr:
            # get snr value
            snr_value = io.get_hkey(sci_hdr, snr_key)
            # check if value is less than limit
            if snr_value < snr_limit:
                # log message
                msg = '\t\tSNR < {0} (SNR = {1}). Skipping file.'
                margs = [snr_limit, snr_value]
                log.general(msg.format(*margs))
                # skip
                continue
        # ---------------------------------------------------------------------
        # 6.7 compute rv
        # ---------------------------------------------------------------------
        cout = general.compute_rv(inst, it, sci_data, sci_hdr, splines=splines,
                                  ref_table=ref_table, blaze=blaze,
                                  systemic_all=systemic_all,
                                  mjdate_all=mjdate_all, ccf_ewidth=ccf_ewidth,
                                  reset_rv=reset_rv)
        # get back ref_table and outputs
        ref_table, outputs = cout
        # ---------------------------------------------------------------------
        # update iterables (for next iteration)
        systemic_all = outputs['SYSTEMIC_ALL']
        mjdate_all = outputs['MJDATE_ALL']
        reset_rv = outputs['RESET_RV']
        ccf_ewidth = outputs['CCF_EW']
        all_durations.append(outputs['TOTAL_DURATION'])
        # ---------------------------------------------------------------------
        # 6.8 save to file
        # ---------------------------------------------------------------------
        inst.write_lblrv_table(ref_table, lblrv_file, sci_hdr, outputs)
        # ---------------------------------------------------------------------
        # 6.9 Time taken stats (For next iteration)
        # ---------------------------------------------------------------------
        if count > 2:
            # smart timing
            sout = general.smart_timing(all_durations, nleft)
            mean_time, std_time, time_left = sout
        count += 1
    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    return locals()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

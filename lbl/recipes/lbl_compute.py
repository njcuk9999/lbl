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
from lbl.resources import lbl_misc
from lbl.science import general

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
LblLowCCFSNR = base_classes.LblLowCCFSNR
log = base_classes.log
# add arguments (must be in parameters.py)
ARGS_COMPUTE = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
    'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE', 'INPUT_FILE', 'TEMPLATE_FILE',
    'BLAZE_FILE', 'HP_WIDTH', 'USE_NOISE_MODEL',
    # plotting
    'PLOT', 'PLOT_COMPUTE_CCF', 'PLOT_COMPUTE_LINES',
    # other
    'SKIP_DONE', 'VERBOSE', 'PROGRAM', 'MASK_FILE',
    # multiprocessing arguments
    'ITERATION', 'TOTAL',
]

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
    inst = select.load_instrument(args, plogger=log)
    # get data directory
    data_dir = io.check_directory(inst.params['DATA_DIR'])
    # move log file (now we have data directory)
    lbl_misc.move_log(data_dir, __NAME__)
    # print splash
    lbl_misc.splash(name=__STRNAME__, instrument=inst.name,
                    params=args, plogger=log)
    # run __main__
    try:
        namespace = __main__(inst)
    except LblException as e:
        raise LblException(e.message, verbose=False)
    except Exception as e:
        emsg = 'Unexpected {0} error: {1}: {2}'
        eargs = [__NAME__, type(e), str(e)]
        raise LblException(emsg.format(*eargs))
    # end code
    lbl_misc.end(__NAME__, plogger=log)
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
    dparams = select.make_all_directories(inst)
    mask_dir, template_dir = dparams['MASK_DIR'], dparams['TEMPLATE_DIR']
    calib_dir, science_dir = dparams['CALIB_DIR'], dparams['SCIENCE_DIR']
    lblrv_dir, lbl_reftable_dir = dparams['LBLRV_DIR'], dparams['LBLRT_DIR']
    lbl_rdb_dir, plot_dir = dparams['LBL_RDB_DIR'], dparams['PLOT_DIR']
    models_dir = dparams['MODEL_DIR']
    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # check data type
    general.check_data_type(inst.params['DATA_TYPE'])
    # mask filename
    mask_file = inst.mask_file(models_dir, mask_dir)
    # template filename
    template_file = inst.template_file(template_dir)
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    # science filenames
    science_files = inst.science_files(science_dir)
    # sort science files by date (speeds up computation)
    science_files = inst.sort_science_files(science_files)
    # reftable filename (None if not set)
    reftable_file, reftable_exists = inst.ref_table_file(lbl_reftable_dir,
                                                         mask_file)
    # -------------------------------------------------------------------------
    # Step 3: Load blaze file if set
    # -------------------------------------------------------------------------
    if blaze_file is not None:
        blaze = inst.load_blaze(blaze_file, science_file=science_files[0])
    else:
        blaze = None
    # -------------------------------------------------------------------------
    # Step 4: Load or make ref_dict
    # -------------------------------------------------------------------------
    ref_table = general.make_ref_dict(inst, reftable_file, reftable_exists,
                                      science_files, mask_file, calib_dir)

    # -------------------------------------------------------------------------
    # Step 5: Get the systemic velocity properties for template
    # -------------------------------------------------------------------------
    systemic_vel_props = general.get_systemic_vel_props(inst, template_file,
                                                        mask_file)

    # -------------------------------------------------------------------------
    # Step 5: spline the template
    # -------------------------------------------------------------------------
    splines = general.spline_template(inst, template_file,
                                      systemic_vel_props['MASK_SYS_VEL'],
                                      models_dir)
    # -------------------------------------------------------------------------
    # Step 6: Loop around science files
    # -------------------------------------------------------------------------
    # filter science files if in multi-mode
    science_files = general.filter_science_files(inst, science_files)

    # load bad odometer codes
    bad_hdr_keys, bad_hdr_key = inst.load_bad_hdr_keys()
    # store all systemic velocities and mid exposure times in mjd
    systemic_all = np.full(len(science_files), np.nan)
    mjdate_all = np.zeros(len(science_files)).astype(float)
    # store the initial ccf_ewidth value
    ccf_ewidth = None
    # flag to take a completely new rv measurement
    reset_rv = True
    # Inside the RV code, we'll measure the velocity of the template to have
    #     a proper systemic velocity on the first iteration of the first file,
    #     we'll compute it and have a finite value. If not finite
    #     we'll assume it's zero inside the code (we're not setting to zero
    #     as it could be zero for real) and measure the offset from there.
    model_velocity = np.inf
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
        # add which science file is being processed
        msg = '\t Science file = {0}'
        log.general(msg.format(science_file))
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
        # If output file exists then get the model velocity from here
        if lblrv_exists and not np.isfinite(model_velocity):
            lblrv_hdr = inst.load_header(lblrv_file, kind='lblrv fits file')
            model_velocity = lblrv_hdr.get_hkey(inst.params['KW_MODELVEL'],
                                                dtype=float)
            largs = [model_velocity]
            log.general('We read model velo = {0:.2f} m/s'.format(*largs))
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
        sci_data, sci_hdr = inst.load_science_file(science_file)
        # flag calibration file
        if inst.params['DATA_TYPE'] != 'SCIENCE':
            model_velocity = 0

        # ---------------------------------------------------------------------
        # 6.4 load blaze if not set above
        # ---------------------------------------------------------------------
        if blaze is None:
            blaze, _ = inst.load_blaze_from_science(science_file, sci_data,
                                                    sci_hdr, calib_dir)
        # ---------------------------------------------------------------------
        # 6.5 check for bad files (via a header key)
        # ---------------------------------------------------------------------
        # check we have a bad hdr key
        if bad_hdr_key is not None and bad_hdr_key in sci_hdr:
            # get bad header key
            sci_bad_hdr_key = sci_hdr.get_hkey(bad_hdr_key)
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
            snr_value = sci_hdr.get_hkey(snr_key, dtype=float)
            # check if value is less than limit
            if snr_value < snr_limit:
                # log message
                msg = '\t\tSNR < {0} (SNR = {1}). Skipping file.'
                margs = [snr_limit, snr_value]
                log.general(msg.format(*margs))
                # skip
                continue
            else:
                # log message
                msg = '\t\tSNR > {0} (SNR = {1:.4f}), passed SNR criteria'
                margs = [snr_limit, snr_value]
                log.general(msg.format(*margs))

        # ---------------------------------------------------------------------
        # 6.7 compute rv
        # ---------------------------------------------------------------------
        try:
            cout = general.compute_rv(inst, it, sci_data, sci_hdr,
                                      splines=splines,
                                      ref_table=ref_table, blaze=blaze,
                                      systemic_props=systemic_vel_props,
                                      systemic_all=systemic_all,
                                      mjdate_all=mjdate_all,
                                      ccf_ewidth=ccf_ewidth,
                                      reset_rv=reset_rv,
                                      model_velocity=model_velocity,
                                      science_file=science_file,
                                      mask_file=mask_file)
        except LblLowCCFSNR as e:
            emsg = e.message + '\n Skipping file.'
            log.warning(emsg)
            continue
        # get back ref_table and outputs
        ref_table, outputs = cout
        # ---------------------------------------------------------------------
        # update iterables (for next iteration)
        systemic_all = outputs['SYSTEMIC_ALL']
        mjdate_all = outputs['MJDATE_ALL']
        reset_rv = outputs['RESET_RV']
        ccf_ewidth = outputs['CCF_EW']
        model_velocity = outputs['MODEL_VELOCITY']

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
    # do not remove this line
    logmsg = log.get_cache()
    # return
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

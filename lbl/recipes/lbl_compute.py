#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import copy
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits
from scipy import stats

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
log = io.log
# add arguments (must be in parameters.py)
ARGS_COMPUTE = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'MASK_SUBDIR', 'TEMPLATE_SUBDIR', 'CALIB_SUBDIR',
    'SCIENCE_SUBDIR', 'LBLRV_SUBDIR', 'LBLREFTAB_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_COMPARISON', 'INPUT_FILE',
    'SCIENCE_TEMPLATE_FILE', 'COMPANION_TEMPLATE_FILE',
    'BLAZE_FILE', 'BLAZE_CORRECTED', 'HP_WIDTH', 'USE_NOISE_MODEL',
    # plotting
    'PLOT', 'PLOT_COMPUTE_CCF', 'PLOT_COMPUTE_LINES',
    # other
    'SKIP_DONE', 'VERBOSE', 'PROGRAM', 'MASK_FILE',
    # multiprocessing arguments
    'ITERATION', 'TOTAL', 'COMPUTE_NPROC',
]

DESCRIPTION_COMPUTE = 'Use this code to compute the LBL rv'

# reference table columns that a file only sets for some lines (the other
#   lines keep the value of the previous file): (columns, flag in outputs)
CARRIED_COLUMNS = [(['MEANXPIX', 'MEANBLAZE'], 'LINES_XPIX_SET'),
                   (['RMSRATIO', 'NPIXLINE', 'CHI2'], 'LINES_STATS_SET')]


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
        namespace = __main__(inst, recipe_kwargs=kwargs)
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


class ComputeRun:
    def __init__(self, nfiles: int):
        """
        What carries over from one science file to the next in the
        lbl_compute loop

        :param nfiles: int, the number of science files
        """
        # store all systemic velocities and mid exposure times in mjd
        self.systemic_all = np.full(nfiles, np.nan)
        self.mjdate_all = np.zeros(nfiles).astype(float)
        # store the initial ccf_ewidth value
        self.ccf_ewidth = None
        # flag to take a completely new rv measurement
        self.reset_rv = True
        # Inside the RV code, we'll measure the velocity of the template to
        #     have a proper systemic velocity on the first iteration of the
        #     first file, we'll compute it and have a finite value. If not
        #     finite we'll assume it's zero inside the code (we're not setting
        #     to zero as it could be zero for real) and measure the offset
        #     from there.
        self.model_velocity = np.inf
        # time stats
        self.mean_time, self.std_time, self.time_left = np.nan, np.nan, ''
        self.all_durations = []
        self.count = 0
        self.nfiles = nfiles


def setup_compute(inst: InstrumentsType,
                  science_files: Optional[List[str]] = None
                  ) -> Dict[str, Any]:
    """
    Steps 1 to 5 of lbl_compute: directories, filenames, blaze, reference
    table, systemic velocity properties and template splines

    :param inst: Instrument instance
    :param science_files: if given, the sorted list of science files (else
                          found and sorted here)

    :return: dictionary of what the loop over science files needs
    """
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst)
    mask_dir, template_dir = dparams['MASK_DIR'], dparams['TEMPLATE_DIR']
    calib_dir, science_dir = dparams['CALIB_DIR'], dparams['SCIENCE_DIR']
    lblrv_dir, lbl_reftable_dir = dparams['LBLRV_DIR'], dparams['LBLRT_DIR']
    models_dir = dparams['MODEL_DIR']
    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # check data type
    general.check_data_type(inst.params['DATA_TYPE'])
    # mask filename
    mask_file = inst.mask_file(models_dir, mask_dir)
    # template filename
    # TODO: Sort out inst.template_file
    science_template_file = inst.template_file(template_dir, 'science')
    comparison_template_file = inst.template_file(template_dir, 'comparison')
    # blaze filename (None if not set)
    blaze_file = inst.blaze_file(calib_dir)
    if science_files is None:
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
    # get systemic velocity for the science template
    sargs = [inst, science_template_file, mask_file]
    science_sys_vel_props = general.get_systemic_vel_props(*sargs)
    # get systemic velocity for the comparison template
    cargs = [inst, comparison_template_file, mask_file]
    comparison_sys_vel_props = general.get_systemic_vel_props(*cargs)

    # -------------------------------------------------------------------------
    # Step 5: spline the template
    # -------------------------------------------------------------------------
    splines = general.spline_template(inst, comparison_template_file,
                                      comparison_sys_vel_props['MASK_SYS_VEL'],
                                      models_dir)
    # return what the loop needs
    return dict(science_files=science_files, lblrv_dir=lblrv_dir,
                calib_dir=calib_dir, mask_file=mask_file, blaze=blaze,
                ref_table=ref_table, splines=splines,
                science_sys_vel_props=science_sys_vel_props,
                comparison_sys_vel_props=comparison_sys_vel_props)


def __main__(inst: InstrumentsType, **kwargs):
    """
    The main recipe function - all code dealing with recipe functionality
    should go here

    :param inst: Instrument instance
    :param kwargs: kwargs to parse to instrument (only use if inst is None)
                   anything in params can be parsed (overwrites instrumental
                   and default parameters). recipe_kwargs: the keyword
                   arguments given to main (needed to start the worker
                   processes when COMPUTE_NPROC > 1)

    :return: all variables in local namespace
    """
    recipe_kwargs = kwargs.pop('recipe_kwargs', None)
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
        recipe_kwargs = kwargs
    # -------------------------------------------------------------------------
    # Steps 1 to 5: directories, files, blaze, ref table, splines
    # -------------------------------------------------------------------------
    ctx = setup_compute(inst)
    # -------------------------------------------------------------------------
    # Step 6: Loop around science files
    # -------------------------------------------------------------------------
    # filter science files if in multi-mode
    science_files = general.filter_science_files(inst, ctx['science_files'])
    # load bad odometer codes
    ctx['bad_hdr_keys'], ctx['bad_hdr_key'] = inst.load_bad_hdr_keys()
    # what carries over between files
    run = ComputeRun(len(science_files))
    # number of processes
    nproc = inst.params['COMPUTE_NPROC']
    if nproc is None:
        nproc = 1
    multi_mode = inst.params['ITERATION'] >= 0 and inst.params['TOTAL'] >= 0
    with io.fast_fits_config():
        if nproc > 1 and not multi_mode and recipe_kwargs is not None:
            compute_parallel(inst, ctx, run, science_files, int(nproc),
                             recipe_kwargs)
        else:
            # loop through each science file
            for it, science_file in enumerate(science_files):
                compute_file(inst, ctx, run, it, science_file)
    # -------------------------------------------------------------------------
    # return local namespace
    # -------------------------------------------------------------------------
    # do not remove this line
    logmsg = log.get_cache()
    # return
    return locals()


def compute_file(inst: InstrumentsType, ctx: Dict[str, Any], run: ComputeRun,
                 it: int, science_file: str) -> Optional[Dict[str, Any]]:
    """
    Compute the line-by-line velocities of one science file and write its
    lblrv file (one pass of the lbl_compute loop)

    :param inst: Instrument instance
    :param ctx: dictionary from setup_compute (+ bad_hdr_keys, bad_hdr_key)
    :param run: what carries over between files (updated)
    :param it: int, the index of the file in the science file list
    :param science_file: str, the science file

    :return: the compute_rv outputs, or None if the file was skipped
    """
    ref_table = ctx['ref_table']
    blaze = ctx['blaze']
    # ---------------------------------------------------------------------
    # 6.1 log process
    # ---------------------------------------------------------------------
    # number left
    nleft = run.nfiles - (it + 1)
    # standard loop message
    log.info('*' * 79)
    msg = 'Processing file {0} / {1}   ({2} left)'
    margs = [it + 1, run.nfiles, nleft]
    log.info(msg.format(*margs))
    log.info('*' * 79)
    # add which science file is being processed
    msg = '\t Science file = {0}'
    log.general(msg.format(science_file))
    # add time stats
    if run.count > 3:
        msgs = ['\tDuration per file {0:.2f}+-{1:.2f} s']
        msgs += ['\tTime left to completion: {2}']
        margs = [run.mean_time, run.std_time, run.time_left]
        for msg in msgs:
            log.general(msg.format(*margs))
    # ---------------------------------------------------------------------
    # 6.2 get lbl rv file and check whether it exists
    # ---------------------------------------------------------------------
    lblrv_file, lblrv_exists = inst.get_lblrv_file(science_file,
                                                   ctx['lblrv_dir'])
    # If output file exists then get the model velocity from here
    if lblrv_exists and not np.isfinite(run.model_velocity):
        lblrv_hdr = inst.load_header(lblrv_file, kind='lblrv fits file')
        run.model_velocity = lblrv_hdr.get_hkey(inst.params['KW_MODELVEL'],
                                                dtype=float)
        largs = [run.model_velocity]
        log.general('We read model velo = {0:.2f} m/s'.format(*largs))
    # if file exists and we are skipping done files
    if lblrv_exists and inst.params['SKIP_DONE']:
        # get the lblrv_hash from the lblrv file
        lblrv_hdr = inst.load_header(lblrv_file, kind='lblrv fits file')
        lblrv_hash = lblrv_hdr.get_hkey(inst.params['KW_RAW_HASH'],
                                        required=False)
        # check hash of science file
        hash_identical = io.check_hash(science_file, lblrv_hash)
        # skip if the hash is identical
        if hash_identical:
            # log message about skipping
            log.general('\t\tLBL.fits file exists and skipping activated. '
                        'Skipping file.')
            # skip
            return None
    elif not lblrv_exists:
        log.general('\t\tLBL.fits file does not exist, computing.')
    else:
        log.general('\t\tLBL.fits file exists, overwriting')
    # ---------------------------------------------------------------------
    # 6.3 load science file
    # ---------------------------------------------------------------------
    sci_data, sci_hdr = inst.load_science_file(science_file)
    # get wave solution for reference file
    sci_wave = inst.get_wave_solution(science_file, sci_data, sci_hdr)
    # flag calibration file
    if inst.params['DATA_TYPE'] != 'SCIENCE':
        run.model_velocity = 0

    # ---------------------------------------------------------------------
    # 6.4 load blaze if not set above
    # ---------------------------------------------------------------------
    if blaze is None:
        bout = inst.load_blaze_from_science(science_file, sci_data,
                                            sci_hdr, ctx['calib_dir'])
        blazeimage, blaze_flag = bout
    # test for all ones (no blaze)
    elif np.sum(blaze.ravel()) == len(blaze.ravel()):
        blaze_flag = True
        blazeimage = np.array(blaze)
    else:
        blaze_flag = False
        blazeimage = np.array(blaze)
    # deal with not having blaze (for s1d weighting)
    if blaze_flag:
        sci_data, blazeimage = inst.no_blaze_corr(sci_data, sci_wave)

    # ---------------------------------------------------------------------
    # 6.5 check for bad files (via a header key)
    # ---------------------------------------------------------------------
    # check we have a bad hdr key
    bad_hdr_key = ctx['bad_hdr_key']
    if bad_hdr_key is not None and bad_hdr_key in sci_hdr:
        # get bad header key
        sci_bad_hdr_key = sci_hdr.get_hkey(bad_hdr_key)
        # if sci_bad_hdr_key in bad_hdr_keys
        if str(sci_bad_hdr_key) in ctx['bad_hdr_keys']:
            # log message about bad header key
            log.general('\t\tFile is known to be bad. Skipping file.')
            # skip
            return None
    # ---------------------------------------------------------------------
    # 6.6 quality control on snr
    # ---------------------------------------------------------------------
    # get snr key
    snr_key = inst.params['KW_SNR']
    snr_limit = inst.params['SNR_THRESHOLD']
    # check we have snr key in science header
    if (snr_key is not None) and (snr_key in sci_hdr):
        # get snr value
        snr_value = sci_hdr.get_hkey(snr_key, dtype=float)
        # check if value is less than limit
        if snr_value < snr_limit:
            # log message
            msg = '\t\tSNR < {0} (SNR = {1}). Skipping file.'
            margs = [snr_limit, snr_value]
            log.general(msg.format(*margs))
            # skip
            return None
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
                                  splines=ctx['splines'],
                                  ref_table=ref_table, blaze=blazeimage,
                                  systemic_props=ctx['science_sys_vel_props'],
                                  systemic_all=run.systemic_all,
                                  mjdate_all=run.mjdate_all,
                                  model_velocity=run.model_velocity,
                                  science_file=science_file,
                                  mask_file=ctx['mask_file'])
    except LblLowCCFSNR as e:
        emsg = e.message + '\n Skipping file.'
        log.warning(emsg)
        return None
    # get back ref_table and outputs
    ref_table, outputs = cout
    ctx['ref_table'] = ref_table
    # ---------------------------------------------------------------------
    # add the mask file
    outputs['MASK_FILE'] = os.path.basename(ctx['mask_file'])
    # add the rash hash to the outputs
    outputs['RAW_HASH'] = io.generate_checksum(science_file)
    # ---------------------------------------------------------------------
    # update iterables (for next iteration)
    run.systemic_all = outputs['SYSTEMIC_ALL']
    run.mjdate_all = outputs['MJDATE_ALL']
    run.reset_rv = outputs['RESET_RV']
    run.ccf_ewidth = outputs['CCF_EW']
    run.model_velocity = outputs['MODEL_VELOCITY']
    # ---------------------------------------------------------------------
    run.all_durations.append(outputs['TOTAL_DURATION'])
    # ---------------------------------------------------------------------
    # 6.8 save to file
    # ---------------------------------------------------------------------
    inst.write_lblrv_table(ref_table, lblrv_file, sci_hdr, outputs)
    outputs['LBLRV_FILE'] = lblrv_file
    # ---------------------------------------------------------------------
    # 6.9 Time taken stats (For next iteration)
    # ---------------------------------------------------------------------
    if run.count > 2:
        # smart timing
        sout = general.smart_timing(run.all_durations, nleft)
        run.mean_time, run.std_time, run.time_left = sout
    run.count += 1
    return outputs


# =============================================================================
# Parallel computation (COMPUTE_NPROC > 1)
# =============================================================================
def compute_parallel(inst: InstrumentsType, ctx: Dict[str, Any],
                     run: ComputeRun, science_files: List[str], nproc: int,
                     recipe_kwargs: Dict[str, Any]):
    """
    Run the lbl_compute loop over nproc processes, with the same results as
    the sequential loop.

    Two things carry over from one file to the next in the sequential loop:
    the model velocity (measured on the first computed file) and, in the
    reference table, the MEANXPIX / MEANBLAZE / RMSRATIO / NPIXLINE / CHI2
    values of the lines that a file does not set (they keep the value of the
    last file that set them; nothing reads them back in the computation).

    So: the first file is computed here (model velocity), the other files
    are split in nproc contiguous blocks, the first one run here and the
    others in worker processes that start from this reference table. Once
    all are done, in each lblrv file of a worker block the values of the
    lines that the block had not yet set are replaced by the values at the
    end of the previous block (and CHI2_VALID_CDF recomputed).

    :param inst: Instrument instance
    :param ctx: dictionary from setup_compute (+ bad_hdr_keys, bad_hdr_key)
    :param run: what carries over between files (updated)
    :param science_files: list of the science files
    :param nproc: int, number of processes
    :param recipe_kwargs: the keyword arguments of main (for the workers)
    """
    nfiles = len(science_files)
    it = 0
    # compute files until we have the model velocity (the first computed
    #   file) as the sequential loop does
    while it < nfiles and not np.isfinite(run.model_velocity):
        compute_file(inst, ctx, run, it, science_files[it])
        it += 1
    remaining = np.arange(it, nfiles)
    # not worth it for a few files
    if len(remaining) < 2 * nproc:
        for jt in remaining:
            compute_file(inst, ctx, run, int(jt), science_files[jt])
        return
    blocks = np.array_split(remaining, nproc)
    # start the workers on blocks 1 to nproc - 1
    workdir = tempfile.mkdtemp(prefix='lbl_compute_')
    workers = []
    for iblock in range(1, nproc):
        task = dict(recipe_kwargs=recipe_kwargs,
                    science_files=list(science_files),
                    indices=[int(jt) for jt in blocks[iblock]],
                    model_velocity=run.model_velocity,
                    ref_table=ctx['ref_table'],
                    bad_hdr_keys=ctx['bad_hdr_keys'],
                    bad_hdr_key=ctx['bad_hdr_key'])
        task_file = os.path.join(workdir, 'task_{0}.pkl'.format(iblock))
        result_file = os.path.join(workdir, 'result_{0}.pkl'.format(iblock))
        log_file = os.path.join(workdir, 'worker_{0}.log'.format(iblock))
        task['result_file'] = result_file
        with open(task_file, 'wb') as pfile:
            pickle.dump(task, pfile)
        # the worker must import this same lbl package
        lbl_path = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            [lbl_path] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                          if len(p) > 0])
        cmd = [sys.executable, '-c',
               'import sys; from lbl.recipes import lbl_compute; '
               'lbl_compute.worker_main(sys.argv[1])', task_file]
        with open(log_file, 'w') as lfile:
            proc = subprocess.Popen(cmd, env=env, stdout=lfile,
                                    stderr=subprocess.STDOUT)
        workers.append((iblock, proc, result_file, log_file))
        msg = 'Worker {0}: files {1} to {2} ({3} files, log: {4})'
        margs = [iblock, blocks[iblock][0] + 1, blocks[iblock][-1] + 1,
                 len(blocks[iblock]), log_file]
        log.general(msg.format(*margs))
    # block 0 here
    for jt in blocks[0]:
        compute_file(inst, ctx, run, int(jt), science_files[jt])
    # wait for the workers
    results = dict()
    for iblock, proc, result_file, log_file in workers:
        proc.wait()
        if proc.returncode != 0 or not os.path.exists(result_file):
            with open(log_file, 'r', errors='ignore') as lfile:
                tail = lfile.readlines()[-30:]
            emsg = 'lbl_compute worker {0} failed (log: {1}):\n{2}'
            raise LblException(emsg.format(iblock, log_file, ''.join(tail)))
        with open(result_file, 'rb') as pfile:
            results[iblock] = pickle.load(pfile)
    # carried values at the end of block 0
    ref_table = ctx['ref_table']
    state = dict()
    for columns, _ in CARRIED_COLUMNS:
        for col in columns:
            state[col] = np.array(ref_table[col])
    # fix the lblrv files of the other blocks, in order
    log.general('Setting the values carried over between blocks')
    for iblock in range(1, nproc):
        result = results[iblock]
        nfixed = 0
        for lblrv_file, not_set in result['files']:
            nfixed += _fix_carried_values(lblrv_file, not_set, state,
                                          result['start_state'])
        msg = '\tBlock {0}: {1} of {2} lblrv files updated'
        log.general(msg.format(iblock, nfixed, len(result['files'])))
        # values at the end of this block
        for columns, flag in CARRIED_COLUMNS:
            ever_set = result['ever_set'][flag]
            for col in columns:
                state[col] = np.where(ever_set, result['end_state'][col],
                                      state[col]).astype(state[col].dtype)
        # merge what the loop keeps
        run.systemic_all = np.where(np.isfinite(result['systemic_all']),
                                    result['systemic_all'], run.systemic_all)
        run.mjdate_all = np.where(result['mjdate_all'] != 0,
                                  result['mjdate_all'], run.mjdate_all)
    # the reference table as the sequential loop would leave it
    for col in state:
        ref_table[col][:] = state[col]
    shutil.rmtree(workdir, ignore_errors=True)


def _fix_carried_values(lblrv_file: str, not_set: Dict[str, np.ndarray],
                        state: Dict[str, np.ndarray],
                        start_state: Dict[str, np.ndarray]) -> bool:
    """
    In an lblrv file written by a worker, replace the carried values of the
    lines not yet set in its block (which the worker took from its starting
    reference table, start_state) by the values at the end of the previous
    block (state), and recompute CHI2_VALID_CDF, as compute_rv does

    :param lblrv_file: str, the lblrv file (updated in place if needed)
    :param not_set: {flag: bool array} lines not yet set in the block
    :param state: {column: array} values at the end of the previous block
    :param start_state: {column: array} values the worker started from

    :return: bool, whether the file was updated
    """
    # anything to change? (bit by bit, as they are written to the file)
    changes = []
    for columns, flag in CARRIED_COLUMNS:
        rows = not_set[flag]
        for col in columns:
            new = np.ascontiguousarray(state[col][rows])
            old = np.ascontiguousarray(start_state[col][rows],
                                       dtype=new.dtype)
            if new.tobytes() != old.tobytes():
                changes.append((col, rows))
    if len(changes) == 0:
        return False
    with fits.open(lblrv_file, mode='update') as hdulist:
        table = hdulist[1].data
        for col, rows in changes:
            column = np.array(table[col])
            column[rows] = state[col][rows]
            table[col][:] = column
        # as in compute_rv
        chi2 = np.array(table['CHI2'], dtype=np.float64)
        npixline = np.array(table['NPIXLINE'], dtype=np.int64)
        table['CHI2_VALID_CDF'][:] = 1 - stats.chi2.cdf(chi2, npixline)
    return True


def worker_main(task_file: str):
    """
    Worker process of compute_parallel: compute a block of science files
    and save what compute_parallel needs to fix the carried values

    :param task_file: str, pickle file with the task
    """
    with open(task_file, 'rb') as pfile:
        task = pickle.load(pfile)
    # parse arguments and load the instrument as main does
    sys.argv = [__NAME__]
    args = select.parse_args(ARGS_COMPUTE, task['recipe_kwargs'],
                             DESCRIPTION_COMPUTE)
    inst = select.load_instrument(args, plogger=log)
    science_files = task['science_files']
    ctx = setup_compute(inst, science_files=science_files)
    # same reference table and bad files as the main process
    ctx['ref_table'] = task['ref_table']
    ctx['bad_hdr_keys'] = task['bad_hdr_keys']
    ctx['bad_hdr_key'] = task['bad_hdr_key']
    run = ComputeRun(len(science_files))
    run.model_velocity = task['model_velocity']
    # carried values we start from
    start_state = dict()
    for columns, _ in CARRIED_COLUMNS:
        for col in columns:
            start_state[col] = np.array(ctx['ref_table'][col])
    # lines set at least once in this block
    nlines = len(ctx['ref_table']['ORDER'])
    ever_set = dict()
    for _, flag in CARRIED_COLUMNS:
        ever_set[flag] = np.zeros(nlines, dtype=bool)
    files = []
    with io.fast_fits_config():
        for it in task['indices']:
            outputs = compute_file(inst, ctx, run, it, science_files[it])
            if outputs is None:
                continue
            not_set = dict()
            for _, flag in CARRIED_COLUMNS:
                ever_set[flag] |= outputs[flag]
                not_set[flag] = ~ever_set[flag]
            files.append((outputs['LBLRV_FILE'], copy.deepcopy(not_set)))
    end_state = dict()
    for columns, _ in CARRIED_COLUMNS:
        for col in columns:
            end_state[col] = np.array(ctx['ref_table'][col])
    result = dict(files=files, ever_set=ever_set, start_state=start_state,
                  end_state=end_state, systemic_all=run.systemic_all,
                  mjdate_all=run.mjdate_all)
    with open(task['result_file'], 'wb') as pfile:
        pickle.dump(result, pfile)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    ll = main()

# =============================================================================
# End of code
# =============================================================================

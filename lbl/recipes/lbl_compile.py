#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
import os

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import general
from lbl.science import plot

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_compile.py'
__STRNAME__ = 'LBL Compil'
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
ARGS_COMPIL = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'LBLRV_SUBDIR', 'LBLRDB_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE',
    # plotting
    'PLOT', 'PLOT_COMPIL_CUMUL', 'PLOT_COMPIL_BINNED',
    # other
    'SKIP_DONE', 'RDB_SUFFIX', 'VERBOSE', 'PROGRAM',
]

DESCRIPTION_COMPIL = 'Use this code to compile the LBL rdb files'


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
    args = select.parse_args(ARGS_COMPIL, kwargs, DESCRIPTION_COMPIL)
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
        args = select.parse_args(ARGS_COMPIL, kwargs, DESCRIPTION_COMPIL)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
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
    # -------------------------------------------------------------------------
    # Step 2: set filenames
    # -------------------------------------------------------------------------
    # get all lblrv files for this object_science and object_template
    lblrv_files = inst.get_lblrv_files(lblrv_dir)

    # deal with no lblrv files (we cannot run compile)
    if len(lblrv_files) == 0:
        wmsg = 'No lblrv files found in {0}. Please run lbl_compute first.'
        wargs = [lblrv_dir]
        raise LblException(wmsg.format(*wargs))

    # get rdb files for this object_science and object_template
    rdbfiles = inst.get_lblrdb_files(lbl_rdb_dir)
    rdbfile1, rdbfile2, rdbfile3, rdbfile4, drift_file = rdbfiles

    # -------------------------------------------------------------------------
    # Step 3: Produce RDB file (.rdb and .fits)
    # -------------------------------------------------------------------------
    if os.path.exists(rdbfile1) and inst.params['SKIP_DONE']:
        # log file exists and we are skipping
        msg = 'File {0} exists, we will read it. To regenerate use --skip'
        margs = [rdbfile1]
        log.general(msg.format(*margs))
        # read the rdb file
        rdb_table = inst.load_lblrdb_file(rdbfile1)
    # else we generate the rdb file
    else:
        # generate table using make_rdb_table function
        rdb_data = general.make_rdb_table(inst, rdbfile1, lblrv_files, plot_dir)
        # get the rdb table out of rdb data
        rdb_table = rdb_data['RDB']
        # plot here based on table (not required when loading)
        plot.compil_binned_band_plot(inst, rdb_table)
        # log creation of rdb table
        msg = 'Writing RDB 1 file: {0}'
        log.info(msg.format(rdbfile1))
        # write rdb table
        io.write_table(rdbfile1, rdb_table, fmt='rdb')
        # write fits file
        if inst.params['WRITE_RDB_FITS']:
            inst.write_rdb_fits(rdbfile1, rdb_data)
        else:
            rdbfitsfile = rdbfile1.replace('.rdb', '.fits')
            log.general('Skipping {0}'.format(rdbfitsfile))

    # -------------------------------------------------------------------------
    # Step 4: Produce binned (per-epoch) RDB file
    # -------------------------------------------------------------------------
    # make the per-epoch RDB table
    rdb_table2 = general.make_rdb_table2(inst, rdb_table)
    # log creation of rdb table 2
    msg = 'Writing RDB 2 file: {0}'
    log.info(msg.format(rdbfile2))
    # write rdb table to file
    io.write_table(rdbfile2, rdb_table2, fmt='rdb')

    # -------------------------------------------------------------------------
    # Step 5: Produce drift file(s)
    # -------------------------------------------------------------------------
    if inst.params['DATA_TYPE'] == 'FP':
        # make the drift table
        drift_table = general.make_drift_table(inst, rdb_table)
        # log creation of drift table
        msg = 'Writing drift file: {0}'
        log.info(msg.format(drift_file))
        # write the drift table
        io.write_table(drift_file, drift_table, fmt='rdb')
    # else we have a file which can be corrected (if drift file exists)
    elif os.path.exists(drift_file):
        # load drift table
        log.general('Using drift table: {0}'.format(drift_file))
        drift_table = io.load_table(drift_file, kind='Drift table', fmt='rdb')
        # create rdb corrected for drift table
        rdb_table3 = general.correct_rdb_drift(inst, rdb_table, drift_table)
        # log creation of rdb corrected table
        msg = 'Writing RDB corrected for drift file: {0}'
        log.info(msg.format(rdbfile3))
        # write the corrected rdb file
        io.write_table(rdbfile3, rdb_table3, fmt='rdb')
        # make the per-epoch RDB corrected for drift table
        rdb_table4 = general.make_rdb_table2(inst, rdb_table3)
        # log creation of rdb table 2
        msg = 'Writing RDB 2 corrected for drift file: {0}'
        log.info(msg.format(rdbfile4))
        # write rdb table to file
        io.write_table(rdbfile4, rdb_table4, fmt='rdb')
    # log that we are not correcting for drift
    else:
        msg = 'No drift file found - not correcting drift'
        log.info(msg)
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

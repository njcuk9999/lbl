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
from lbl.science import general
from lbl.science import plot
from lbl.resources import misc

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
ARGS_COMPIL = [
               # core
               'INSTRUMENT', 'CONFIG_FILE',
               # directory
               'DATA_DIR', 'LBLRV_SUBDIR', 'LBLRDB_SUBDIR',
               # science
               'OBJECT_SCIENCE', 'OBJECT_TEMPLATE',
               # plotting
               'PLOT', 'PLOT_COMPIL_CUMUL', 'PLOT_COMPIL_BINNED',
               # other
               'SKIP_DONE', 'RDB_SUFFIX'
               ]
# TODO: Etienne - Fill out
DESCRIPTION_COMPIL = 'Use this code to compile the LBL rdb file'


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
    inst = select.load_instrument(args)
    # move log file (now we have data directory)
    misc.move_log(inst.params['DATA_DIR'], __NAME__)
    # print splash
    misc.splash(name=__STRNAME__, instrument=inst.name,
                  cmdargs=inst.params['COMMAND_LINE_ARGS'])
    # run __main__
    try:
        namespace = __main__(inst)
    except LblException as e:
        raise LblException(e.message)
    except Exception as e:
        emsg = 'Unexpected lbl_compile error: {0}: {1}'
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
        args = select.parse_args(ARGS_COMPIL, kwargs, DESCRIPTION_COMPIL)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    # get data directory
    data_dir = inst.params['DATA_DIR']
    # copy over readme
    misc.copy_readme(data_dir)
    # make sub directory based on object science and object template
    obj_subdir = inst.science_template_subdir()
    # make lblrv directory
    lblrv_dir = io.make_dir(data_dir, inst.params['LBLRV_SUBDIR'], 'LBL RV',
                            subdir=obj_subdir)
    # make lbl rdb directory
    lbl_rdb_dir = io.make_dir(data_dir, inst.params['LBLRDB_SUBDIR'], 'LBL rdb')
    # make the plot directory
    plot_dir = io.make_dir(data_dir, 'plots', 'Plot')
    # -------------------------------------------------------------------------
    # Step 2: set filenames
    # -------------------------------------------------------------------------
    # get all lblrv files for this object_science and object_template
    lblrv_files = inst.get_lblrv_files(lblrv_dir)
    # get rdb files for this object_science and object_template
    rdbfiles = inst.get_lblrdb_files(lbl_rdb_dir)
    rdbfile1, rdbfile2, rdbfile3, rdbfile4, drift_file = rdbfiles

    # -------------------------------------------------------------------------
    # Step 3: Produce RDB file
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
        rdb_table = general.make_rdb_table(inst, rdbfile1, lblrv_files,
                                           plot_dir)
        # plot here based on table (not required when loading)
        plot.compil_binned_band_plot(inst, rdb_table)
        # log creation of rdb table
        msg = 'Writing RDB 1 file: {0}'
        log.info(msg.format(rdbfile1))
        # write rdb table
        io.write_table(rdbfile1, rdb_table, fmt='rdb')

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
    if inst.params['OBJECT_SCIENCE'] == 'FP':
        # make the drift table
        drift_table = general.make_drift_table(inst, rdb_table)
        # log creation of drift table
        msg = 'Writing drift file: {0}'
        log.info(msg.format(drift_table))
        # write the drift table
        io.write_table(drift_file, drift_table, fmt='rdb')
    # else we have a file which can be corrected (if drift file exists)
    elif os.path.exists(drift_file):
        # load drift table
        drift_table = io.load_table(drift_file, kind='Drift table')
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
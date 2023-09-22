#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-08-24

@author: cook
"""
import os

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import general
from lbl.science import tellu_clean

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_telluclean.py'
__STRNAME__ = 'LBL Tellu Clean'
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
ARGS_TEMPLATE = [  # core
    'INSTRUMENT', 'CONFIG_FILE', 'DATA_SOURCE', 'DATA_TYPE',
    # directory
    'DATA_DIR', 'TEMPLATE_SUBDIR', 'SCIENCE_SUBDIR',
    # science
    'OBJECT_SCIENCE', 'OBJECT_TEMPLATE'
    # other
                      'VERBOSE', 'PROGRAM',
]

DESCRIPTION_TEMPLATE = 'Use this code to create the LBL template'


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
    args = select.parse_args(ARGS_TEMPLATE, kwargs, DESCRIPTION_TEMPLATE)
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
    # -------------------------------------------------------------------------
    # deal with debug
    if inst is None or inst.params is None:
        # deal with parsing arguments
        args = select.parse_args(ARGS_TEMPLATE, kwargs, DESCRIPTION_TEMPLATE)
        # load instrument
        inst = select.load_instrument(args)
        # assert inst type (for python typing later)
        amsg = 'inst must be a valid Instrument class'
        assert isinstance(inst, InstrumentsList), amsg
    # get tqdm
    tqdm = base.tqdm_module(inst.params['USE_TQDM'], log.console_verbosity)
    # -------------------------------------------------------------------------
    # check data before running
    # -------------------------------------------------------------------------
    # check data type
    general.check_data_type(inst.params['DATA_TYPE'])
    # check that we want to tellu-clean
    if not inst.params['DO_TELLUCLEAN']:
        emsg = 'DO_TELLUCLEAN set to False for instrument: {0}'
        eargs = [inst.name]
        raise base_classes.LblException(emsg.format(*eargs))
    # check that science object is not tellu-cleaned
    if inst.params['OBJECT_SCIENCE'].endswith('_tc'):
        emsg = ('Cannot tellu-clean a file that has a tellu-cleaned object '
                '(ends with _tc), OBJECT_SCIENCE = {0}')
        eargs = [inst.params['OBJECT_SCIENCE']]
        raise base_classes.LblException(emsg.format(*eargs))
    # check that we have science data (can only tellu-clean science data)
    if inst.params['DATA_TYPE'] != 'SCIENCE':
        emsg = ('Can only tellu-clean "SCIENCE" data. '
                'Please set "DATA_TYPE" to "SCIENCE"')
        raise base_classes.LblException(emsg)

    # -------------------------------------------------------------------------
    # Step 1: Set up data directory
    # -------------------------------------------------------------------------
    dparams = select.make_all_directories(inst)
    template_dir, science_dir = dparams['TEMPLATE_DIR'], dparams['SCIENCE_DIR']
    calib_dir = dparams['CALIB_DIR']
    model_dir = dparams['MODEL_DIR']
    # -------------------------------------------------------------------------
    # Step 2: Check and set filenames
    # -------------------------------------------------------------------------
    # template filename
    template_file = inst.template_file(template_dir, required=False)
    # science filenames (always non _tc files)
    science_files = inst.science_files(science_dir)

    # -------------------------------------------------------------------------
    # Step 3: get the TAPAS exponents
    # -------------------------------------------------------------------------
    spl_others, spl_water = tellu_clean.get_tapas_spl(inst, model_dir)
    # -------------------------------------------------------------------------
    # Step 4: Loop around science files and clean
    # -------------------------------------------------------------------------
    # loop around all science files
    for it, filename in enumerate(science_files):
        # construct the output directory
        outdir = inst.params['OBJECT_SCIENCE'] + '_tc'
        outpath = os.path.join(science_dir, outdir)
        # create this path if it doesn't exist
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        # construct the tellu-cleaned output filename
        tellu_clean_file = os.path.join(outpath, os.path.basename(filename))
        # ---------------------------------------------------------------------
        # if this output file exists and we are skipping done - skip
        if os.path.exists(tellu_clean_file) and inst.params['SKIP_DONE']:
            msg = 'File {0} exists and SKIP_DONE = True ({1} of {2})'
            margs = [tellu_clean_file, it + 1, len(science_files)]
            log.general(msg.format(*margs))
            continue
        # ---------------------------------------------------------------------
        # print progress
        msg = 'tellu-cleaning E2DS for file {0} of {1}'
        margs = [it + 1, len(science_files)]
        log.info(msg.format(*margs))
        # ---------------------------------------------------------------------
        # load the science image and header
        sci_image, sci_hdr = inst.load_science_file(filename)
        # get wave solution for reference file
        sci_wave = inst.get_wave_solution(filename, sci_image, sci_hdr)
        # push things into an e2ds input/output dictionary
        e2ds_dict = dict()
        e2ds_dict['flux'] = sci_image
        e2ds_dict['wavelength'] = sci_wave
        e2ds_dict['AIRMASS'] = sci_hdr.get_hkey(inst.params['KW_AIRMASS'],
                                                dtype=float)
        e2ds_dict['OBJECT'] = inst.params['OBJECT_SCIENCE']
        e2ds_dict['BERV'] = inst.get_berv(sci_hdr)
        e2ds_dict['FILENAME'] = filename
        # ---------------------------------------------------------------------
        # do the telluric correction (similar to APERO)
        e2ds_dict = tellu_clean.correct_tellu(inst, template_file,
                                              e2ds_dict, spl_others, spl_water,
                                              model_dir)
        # ---------------------------------------------------------------------
        # write the tellu-cleaned file to disk
        inst.write_tellu_cleaned(tellu_clean_file, e2ds_dict, sci_hdr,
                                 science_filename=filename)

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

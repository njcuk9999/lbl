#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from typing import List
import datetime
import glob
import os
import re
import sys

import numpy as np

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import vetted
from lbl.recipes import lbl_compile
from lbl.recipes import lbl_compute
from lbl.recipes import lbl_mask
from lbl.recipes import lbl_resmap
from lbl.recipes import lbl_telluclean
from lbl.recipes import lbl_template
from lbl.recipes import lbl_reset
from lbl.instruments import select
from lbl.resources import lbl_misc
from lbl.science import injection

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_wrap.py'
__STRNAME__ = 'LBL Wrapper'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Description of recipe
DESCRIPTION_MASK = 'Use this code to wrap around lbl'
# get the logger
log = io.log
# define keys to remove from run params
REMOVE_KEYS = [  # core
    'INSTRUMENT', 'DATA_DIR', 'DATA_TYPES', 'DATA_SOURCE',
    # science keys
    'OBJECT_SCIENCE', 'OBJECT_COMPARISON', 'OBJECT_TEFF',
    'BLAZE_CORRECTED', 'BLAZE_FILE',
    # run keys
    'RUN_LBL_RESET', 'RUN_LBL_RESMAP',
    'RUN_LBL_TELLUCLEAN', 'RUN_LBL_TEMPLATE', 'RUN_LBL_MASK',
    'RUN_LBL_COMPUTE', 'RUN_LBL_COMPILE',
    # skip keys
    'SKIP_LBL_TELLUCLEAN', 'SKIP_LBL_TEMPLATE', 'SKIP_LBL_MASK',
    'SKIP_LBL_COMPUTE', 'SKIP_LBL_COMPILE', 'SKIP_LBL_PCA2D',
    # general keys already used
    'SKIP_DONE', 'OVERWRITE', 'TELLUCLEAN_USE_TEMPLATE',
    # the pca2d precleaning, lbl_pca2d's and no other recipe's
    'PCA2D',
    # the end-of-run report
    'REPORT',
    # an injection test, the wrap's own (lbl.science.injection)
    'FAKE_KEPLERIAN', 'fake_keplerian']

# Define the default values
DEFAULTS = dict()
DEFAULTS['RUN_LBL_RESET'] = False
DEFAULTS['RUN_LBL_TELLUCLEAN'] = False
DEFAULTS['RUN_LBL_RESMAP'] = False
DEFAULTS['RUN_LBL_TEMPLATE'] = False
DEFAULTS['RUN_LBL_MASK'] = False
DEFAULTS['RUN_LBL_COMPUTE'] = False
DEFAULTS['RUN_LBL_COMPILE'] = False
#: THE REPORT IS MADE BY DEFAULT. It costs a few seconds at the end of a run
#: that took minutes, and it is what tells whether the run worked.
DEFAULTS['REPORT'] = True
DEFAULTS['SKIP_LBL_TEMPLATE'] = False
DEFAULTS['SKIP_LBL_MASK'] = False
DEFAULTS['SKIP_LBL_COMPUTE'] = False
DEFAULTS['SKIP_LBL_COMPILE'] = False
#: NEITHER TRUE NOR FALSE BY DEFAULT: the PCA2D correction is redone only if
#: the settings, the input files, nominal.yaml or the instrument's PCA2D
#: settings (its LBL class) have changed. True keeps the
#: one on disk even then, False always redoes it (lbl_pca2d.correct)
DEFAULTS['SKIP_LBL_PCA2D'] = None


# =============================================================================
# Define functions
# =============================================================================
def main(runparams: dict):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param runparams: dict, parameters to pass to lbl recipes

    :return:
    """
    # reset the sys.argv (arguments from command line aren't used)
    sys.argv = [__NAME__]
    # get key parameters
    instrument = lbl_misc.check_runparams(runparams, 'INSTRUMENT')
    data_dir = lbl_misc.check_runparams(runparams, 'DATA_DIR')
    data_source = lbl_misc.check_runparams(runparams, 'DATA_SOURCE')
    data_types = lbl_misc.check_runparams(runparams, 'DATA_TYPES')
    object_sciences = _object_list(
        lbl_misc.check_runparams(runparams, 'OBJECT_SCIENCE'))
    object_comparisons = lbl_misc.check_runparams(runparams, 'OBJECT_COMPARISON')
    object_teffs = lbl_misc.check_runparams(runparams, 'OBJECT_TEFF')
    blaze_corrs = lbl_misc.check_runparams(runparams, 'BLAZE_CORRECTED',
                                           required=False)
    blaze_files = lbl_misc.check_runparams(runparams, 'BLAZE_FILE',
                                           required=False)
    # -------------------------------------------------------------------------
    # push other keyword arguments into keyword arguments dictionary
    keyword_args = dict()
    for key in runparams:
        if key not in REMOVE_KEYS:
            keyword_args[key] = runparams[key]
    # make sure we have defaults if key not in runparams
    for key in DEFAULTS:
        if key not in runparams:
            runparams[key] = DEFAULTS[key]
    # -------------------------------------------------------------------------
    # sanity checks on runparams (certain things should not be set together)
    if runparams['RUN_LBL_MASK'] and 'MASK_FILE' in runparams:
        if runparams['MASK_FILE'] not in [None, 'None', '', 'Null']:
            emsg = ('LBL_WRAP ERROR: Cannot have RUN_LBL_MASK=True and '
                    'MASK_FILE={0} (Must be unset)')
            raise base_classes.LblException(emsg.format(runparams['MASK_FILE']))
    # the residual map recipe is not run by the wrap
    if runparams['RUN_LBL_RESMAP']:
        log.warning('RUN_LBL_RESMAP=True: lbl_resmap is not run by the wrap, '
                    'run it on its own')
    # a frozen observer basis (PCA2D input_name) this run cannot use, of
    #   another instrument or on another grid, stops the wrap here, before
    #   LBL's first step, rather than at the PCA2D fit, minutes later
    science = [data_types] if isinstance(data_types, str) else data_types
    if runparams.get('PCA2D') not in (None, False) and 'SCIENCE' in science:
        from lbl.recipes import lbl_pca2d
        lbl_pca2d.frozen_basis(runparams['PCA2D'], instrument, data_source,
                               data_dir, report=runparams.get('REPORT', True))
    # -------------------------------------------------------------------------
    # the reset cleans the products of every object in DATA_DIR: once, before
    #   the first object, or only the last object's products would survive
    if runparams['RUN_LBL_RESET']:
        lbl_reset.main(instrument=instrument, data_dir=data_dir,
                       data_source=data_source, **keyword_args)
    # -------------------------------------------------------------------------
    # AN INJECTION TEST (lbl.science.injection): FAKE_KEPLERIAN's Keplerian
    #   signals added to the BERV of every science object's spectra, which
    #   are measured as <object>_inj (a folder of links to its spectra, with
    #   the record of the signals): nothing the object's own runs made is
    #   touched, and its template becomes <object>_inj's when it was its own
    found = injection.signals(runparams.get(
        'FAKE_KEPLERIAN', runparams.get('fake_keplerian')))
    if found and any(signal.get('known') for signal in found) \
            and not runparams.get('PLANETS', False):
        log.warning('injection test: the signals are among the star\'s known '
                    'planets (known=True), but PLANETS is off: the template '
                    'does not follow known orbits, and these are not used')
    if found:
        object_sciences, object_comparisons = _inject(
            data_dir, runparams.get('SCIENCE_SUBDIR', 'science'),
            object_sciences, object_comparisons, data_types, found)
    # -------------------------------------------------------------------------
    # mark the expected length if a list
    olen = len(object_sciences)
    # the joint pca2d correction, once made: {object: (corrected, comparison)}
    pca2d_joint = None
    # the objects LBL has measured as delivered (PCA2D compare)
    measured = set()
    # loop around all files
    for num in range(olen):
        # get the science target
        object_science = object_sciences[num]
        # print wrapper splash
        lbl_misc.splash(name=__STRNAME__, instrument=instrument,
                        plogger=log)
        # print iteration we are running
        msg = 'Running [{0}] iteration {1}/{2}'
        margs = [object_science, num + 1, olen]
        log.info(msg.format(*margs))
        # wrap check args
        wkargs = dict(iteration=num, length=olen)
        # get this iterations values (and check if they are a list of matching
        #    length to object_sciences) or just a single value
        data_type = lbl_misc.wraplistcheck(data_types,
                                           'DATA_TYPES', **wkargs)
        object_comparison = lbl_misc.wraplistcheck(object_comparisons,
                                                   'OBJECT_COMPARISON',
                                                   **wkargs)
        object_teff = lbl_misc.wraplistcheck(object_teffs,
                                             'OBJECT_TEFF', **wkargs)
        blaze_corr = lbl_misc.wraplistcheck(blaze_corrs,
                                            'BLAZE_CORRECTED', **wkargs)
        blaze_file = lbl_misc.wraplistcheck(blaze_files,
                                            'BLAZE_FILE', **wkargs)
        # ---------------------------------------------------------------------
        # run all pre-cleaning steps
        if runparams['RUN_LBL_TELLUCLEAN'] and data_type == 'SCIENCE':
            # the template the cleaning uses is the star's own, made from its
            #   cleaned spectra (its lines are divided out of each spectrum);
            #   the comparison is what the velocities are measured against
            tc_template = _tc_name(object_science)
            tc_comparison = _tc_comparison(runparams, data_dir, instrument,
                                           object_science, object_comparison)
            # run telluric cleaning (without template)
            lbl_telluclean.main(instrument=instrument, data_dir=data_dir,
                                data_source=data_source,
                                data_type=data_type,
                                object_science=object_science,
                                object_comparison=object_comparison,
                                skip_done=False,
                                telluclean_use_template=False,
                                blaze_corrected=blaze_corr,
                                blaze_file=blaze_file,
                                **keyword_args)
            # make the template (if not present)
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science + '_tc',
                              object_comparison=tc_template,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
                              object_teff=object_teff,
                              overwrite=True,
                              **keyword_args)
            # re-run tellu clean with uncorrected science data now using our
            #  template (made from cleaned science data)
            lbl_telluclean.main(instrument=instrument, data_dir=data_dir,
                                data_source=data_source,
                                data_type=data_type,
                                object_science=object_science,
                                object_comparison=tc_template,
                                skip_done=False,
                                telluclean_use_template=True,
                                blaze_corrected=blaze_corr,
                                blaze_file=blaze_file,
                                **keyword_args)
            # the template made again, from the second pass's spectra: LBL's
            #   template step below keeps a template already on disk (unless
            #   OVERWRITE), so the first pass's would have stayed, and the
            #   mask, the velocities and PCA2D would have measured the second
            #   pass's spectra against it
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science + '_tc',
                              object_comparison=tc_template,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
                              object_teff=object_teff,
                              overwrite=True,
                              **keyword_args)
            # update object and comparison names
            object_science = _tc_name(object_science)
            object_comparison = tc_comparison
        # ---------------------------------------------------------------------
        # LBL's own steps, from the template to the report
        steps = dict(runparams=runparams, keyword_args=keyword_args,
                     instrument=instrument, data_dir=data_dir,
                     data_source=data_source, data_type=data_type,
                     object_teff=object_teff, blaze_corr=blaze_corr,
                     blaze_file=blaze_file)
        # ---------------------------------------------------------------------
        # THE PCA2D PRECLEANING, on the object as it now is: after the
        # telluric cleaning when there was one, so an optical instrument gets
        # <object>_tc_pca2d. Its corrected spectra are a science object of
        # their own, <object>_pca2d, and every step below measures that one.
        # Without a PCA2D key nothing of it is even imported, and the steps
        # run once, on the object.
        if runparams.get('PCA2D') not in (None, False) and \
                data_type == 'SCIENCE':
            from lbl.recipes import lbl_pca2d
            pca2d = runparams['PCA2D']
            delivered = object_science
            # THE DELIVERED SPECTRA ARE MEASURED AS WITH SMALL_RV_SHIFT TRUE,
            #   whatever it says: their velocities carry the tellurics' imprint
            #   where the star's lines cross them, which a template iterating on
            #   them takes in, and a star frame built on them copies into
            #   PCA2D's correction (TOI-904, NIRPS, 2026-11-24: 51 to 65 m/s
            #   rms after PCA2D, against 21 with the BERV alone). False
            #   applies once PCA2D has corrected them (_pca2d_passes)
            delivered_steps = dict(steps, keyword_args=_as_delivered(
                keyword_args))
            # the delivered object measured first, so that the report can set
            # its velocities beside the corrected ones (PCA2D compare=True).
            # Measured only: ONE REPORT per star, the corrected one's below,
            # whose own pages hold the delivered velocities; a second PDF
            # beside it (<object>_<template>.pdf next to
            # <object>_pca2d_<template>.pdf) only confused
            if lbl_pca2d.compare(pca2d) and object_science not in measured:
                _lbl_steps(object_science=object_science,
                           object_comparison=object_comparison,
                           report=False, **delivered_steps)
                measured.add(object_science)
            # A JOINT FIT: every other star measured as delivered too, BEFORE
            #   it, since a star whose OBJECT_COMPARISON is another star needs
            #   its velocity against that template to place it (lbl_pca2d
            #   star_template); their own iterations then skip that pass
            joint = isinstance(pca2d, dict) and pca2d.get('joint')
            if joint and pca2d_joint is None and lbl_pca2d.compare(pca2d) \
                    and not runparams['RUN_LBL_TELLUCLEAN']:
                for other in range(olen):
                    if object_sciences[other] in measured:
                        continue
                    kw_other = dict(iteration=other, length=olen)
                    if lbl_misc.wraplistcheck(data_types, 'DATA_TYPES',
                                              **kw_other) != 'SCIENCE':
                        continue
                    steps_other = dict(
                        delivered_steps,
                        object_teff=lbl_misc.wraplistcheck(
                            object_teffs, 'OBJECT_TEFF', **kw_other),
                        blaze_corr=lbl_misc.wraplistcheck(
                            blaze_corrs, 'BLAZE_CORRECTED', **kw_other),
                        blaze_file=lbl_misc.wraplistcheck(
                            blaze_files, 'BLAZE_FILE', **kw_other))
                    _lbl_steps(object_science=object_sciences[other],
                               object_comparison=lbl_misc.wraplistcheck(
                                   object_comparisons, 'OBJECT_COMPARISON',
                                   **kw_other), report=False, **steps_other)
                    measured.add(object_sciences[other])
            pca2d_args = dict(instrument=instrument, data_dir=data_dir,
                              data_source=data_source, pca2d=pca2d,
                              input_file=runparams.get('INPUT_FILE'),
                              science_subdir=runparams.get('SCIENCE_SUBDIR',
                                                           'science'),
                              report=runparams.get('REPORT', True),
                              skip=runparams['SKIP_LBL_PCA2D'],
                              planets=bool(runparams.get('PLANETS', False)))
            if isinstance(pca2d, dict) and pca2d.get('joint'):
                # every science object against ONE observer basis, done once,
                # at the first of them
                if pca2d_joint is None:
                    pca2d_joint = _pca2d_joint(
                        runparams, object_sciences, data_types,
                        object_comparisons, pca2d_args, keyword_args,
                        instrument, data_dir, data_source, blaze_corrs,
                        blaze_files)
                object_science, object_comparison = \
                    pca2d_joint[object_science]
            else:
                object_science, object_comparison = _pca2d_passes(
                    runparams, keyword_args, instrument, data_dir,
                    data_source, pca2d_args,
                    {object_science: object_comparison},
                    {object_science: (blaze_corr, blaze_file)},
                    joint=False)[object_science]
            lbl_report = _lbl_steps(object_science=object_science,
                                    object_comparison=object_comparison,
                                    **steps)
            # ONE REPORT: pca2d's (the star, delivered against corrected, the
            # river plots, the components...) followed by LBL's own pages
            if runparams.get('REPORT', True) and runparams['RUN_LBL_COMPILE']:
                lbl_pca2d.report(data_dir=data_dir, delivered=delivered,
                                 corrected=object_science,
                                 lbl_report=lbl_report)
        else:
            _lbl_steps(object_science=object_science,
                       object_comparison=object_comparison, **steps)


def _object_list(object_sciences) -> list:
    """OBJECT_SCIENCE as a list: a string is one object (not its letters)"""
    if isinstance(object_sciences, str):
        return [object_sciences]
    if isinstance(object_sciences, (list, tuple)):
        return list(object_sciences)
    emsg = ('LBL_WRAP ERROR: OBJECT_SCIENCE must be an object name or a list '
            'of them (got {0})')
    raise base_classes.LblException(emsg.format(repr(object_sciences)))


def _inject(data_dir: str, science_subdir: str, object_sciences: list,
            object_comparisons, data_types, found: list) -> tuple:
    """The injection test's objects: each science object's <object>_inj
    made (lbl.science.injection.prepare), and the run's names moved to
    them: (object_sciences, object_comparisons). A template that was a
    science object of the run becomes that object's injected one; another
    star's, or a vetted one, stays as it is.

    What LBL made from <object>_inj with other signals (or none) is put
    aside in DATA_DIR/pca2d/<object>_inj/superseded/<date>, since none of
    it may be reused; PCA2D makes its correction again by itself (the
    signals are in its request)."""
    science_root = os.path.join(data_dir, science_subdir)
    renamed = dict()
    count = len(object_sciences)
    for num, objname in enumerate(object_sciences):
        data_type = lbl_misc.wraplistcheck(data_types, 'DATA_TYPES',
                                           iteration=num, length=count)
        if data_type != 'SCIENCE':
            continue
        name, previous = injection.prepare(science_root, objname, found,
                                           log=log.info)
        renamed[objname] = name
        if previous != found:
            _put_aside(data_dir, name, science_subdir, previous)
    sciences = [renamed.get(objname, objname) for objname in object_sciences]
    if isinstance(object_comparisons, (list, tuple)):
        comparisons = [renamed.get(other, other)
                       for other in object_comparisons]
    else:
        comparisons = renamed.get(object_comparisons, object_comparisons)
    return sciences, comparisons


def _as_delivered(keyword_args: dict) -> dict:
    """The keyword arguments of LBL's steps on the delivered spectra of a
    PCA2D run: SMALL_RV_SHIFT True, the template on BERV alone (see
    _pca2d_passes)"""
    return dict(keyword_args, SMALL_RV_SHIFT=True)


def _velocity_template(runparams: dict, keyword_args: dict, instrument: str,
                       data_dir: str, data_source: str, objname: str,
                       blaze_corr, blaze_file) -> None:
    """SMALL_RV_SHIFT False: the own template of PCA2D's corrected spectra,
    made by iterating on their velocities, before PCA2D's second pass, which
    takes them for its star frame (lbl_pca2d.object_velocities: its star
    component is then built where the star is, and the PCA does not take in
    part of a real signal). Made when it is missing, or made again when it
    was made with the velocity taken as constant (SMALLRV in its header):
    whatever template the velocities are measured against
    (OBJECT_COMPARISON), only the object's own knows its spectra. Nothing
    when SMALL_RV_SHIFT is True."""
    if runparams.get('SMALL_RV_SHIFT', True) not in (False, 'False', 0):
        return
    path = _template_of(data_dir, objname, instrument)
    flat = False
    if path is not None:
        try:
            from astropy.io import fits
            flat = bool(fits.getheader(path).get('SMALLRV', False))
        except Exception:                                     # noqa: BLE001
            flat = True
        if not flat:
            return
    log.info('SMALL_RV_SHIFT = False: the template of {0} {1}, iterating on '
             'the velocities of its spectra, before PCA2D\'s second pass, '
             'whose star frame follows them'.format(objname, 'is made again (it was made with '
                                   'the velocity taken as constant)' if flat
                                   else 'is made'))
    lbl_template.main(instrument=instrument, data_dir=data_dir,
                      data_source=data_source, data_type='SCIENCE',
                      object_science=objname, object_comparison=objname,
                      blaze_corrected=blaze_corr, blaze_file=blaze_file,
                      object_teff=_teff_of(runparams, objname),
                      overwrite=flat, **keyword_args)


def _teff_of(runparams: dict, objname: str):
    """The Teff the wrap gives an object (OBJECT_TEFF, in the order of
    OBJECT_SCIENCE), or the object it was made from: <object>_pca2d (PCA2D's
    correction), <object>_inj (an injection test), <object>_tc (telluric
    cleaning), in any combination. None when it gives none"""
    names = runparams.get('OBJECT_SCIENCE')
    teffs = runparams.get('OBJECT_TEFF')
    if teffs is None or names is None:
        return None
    if not isinstance(teffs, (list, tuple)):
        return teffs
    names = list(names) if isinstance(names, (list, tuple)) else [names]
    name = str(objname)
    while True:
        if name in names and names.index(name) < len(teffs):
            return teffs[names.index(name)]
        for suffix in ('_pca2d', '_inj', '_tc'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        else:
            return None


def _put_aside(data_dir: str, name: str, science_subdir: str, previous):
    """What LBL made from an injected object with other signals, moved to
    DATA_DIR/pca2d/<name>/superseded/<date> (moved, not deleted: two
    injections are compared), said with its size"""
    import shutil
    from lbl.recipes import lbl_pca2d
    going = lbl_pca2d.lbl_products(data_dir, name, science_subdir)
    if not going:
        return
    stamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    aside = os.path.join(data_dir, 'pca2d', name, 'superseded', stamp)
    total = 0
    for path in going:
        target = os.path.join(aside, os.path.relpath(path, data_dir))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        total += lbl_pca2d.size_of(path)
        shutil.move(path, target)
    log.warning('injection test: what LBL made from {0} with {1} ({2} items, '
                '{3:.1f} GB) moved to {4}: made with other signals, none of '
                'it may be reused'.format(
                    name, injection.describe(previous) if previous
                    else 'no signal recorded', len(going), total / 1e9,
                    aside))


def _tc_name(objname: str) -> str:
    """The name of an object after the telluric cleaning"""
    return objname if objname.endswith('_tc') else objname + '_tc'


def _tc_comparison(runparams: dict, data_dir: str, instrument: str,
                   object_science: str, object_comparison: str) -> str:
    """
    The comparison of a telluric-cleaned object: the object's own cleaned
    template; a vetted template as it is (a stellar template, with no
    telluric lines to clean); another star's cleaned template, which needs
    that star cleaned first (its science/<star>_tc spectra, or its template)
    """
    if object_comparison in [None, '', 'None', 'Null']:
        return _tc_name(object_science)
    if vetted.is_vetted(object_comparison):
        return object_comparison
    if _tc_name(object_comparison) == _tc_name(object_science):
        return _tc_name(object_science)
    other = _tc_name(object_comparison)
    science_dir = os.path.join(data_dir,
                               runparams.get('SCIENCE_SUBDIR', 'science'),
                               other)
    if os.path.isdir(science_dir) or \
            _template_of(data_dir, other, instrument) is not None:
        return other
    emsg = ('LBL_WRAP ERROR: OBJECT_SCIENCE={0} with OBJECT_COMPARISON={1} and '
            'RUN_LBL_TELLUCLEAN=True: the velocities are measured against '
            'the template of the cleaned comparison, {2}, and there are '
            'neither {2} spectra ({3}) nor a {2} template. Clean {1} first '
            '(earlier in the OBJECT_SCIENCE list, or in an earlier wrap), or '
            'compare {0} with itself or a VETTED_ template')
    eargs = [object_science, object_comparison, other, science_dir]
    raise base_classes.LblException(emsg.format(*eargs))


def _missing_for_compute(keyword_args, **kwargs):
    """What lbl_compute will look for and not find: the templates of the
    science object and of the comparison (a vetted one is fetched by
    lbl_compute itself) and the comparison's mask, by the paths LBL's
    instrument class gives them, as lbl_compute's own setup asks them.
    ([objects whose template is missing], whether the mask is)."""
    from lbl.core import parameters
    names = {key.upper() for key in parameters.params}
    args = select.parse_args(lbl_compute.ARGS_COMPUTE,
                             dict({k: v for k, v in keyword_args.items()
                                   if k.upper() in names}, **kwargs),
                             '', parse=False)
    inst = select.load_instrument(args)
    data_dir = inst.params['DATA_DIR']
    template_dir = os.path.join(data_dir, inst.params['TEMPLATE_SUBDIR'])
    mask_dir = os.path.join(data_dir, inst.params['MASK_SUBDIR'])
    model_dir = os.path.join(data_dir, 'models')
    templates = []
    for tkind, objname in (('science', inst.params['OBJECT_SCIENCE']),
                           ('comparison', inst.params['OBJECT_COMPARISON'])):
        if vetted.is_vetted(objname) or objname in templates:
            continue
        path = inst.template_file(template_dir, tkind, required=False)
        if not os.path.exists(path):
            templates.append(objname)
    mask = inst.mask_file(model_dir, mask_dir, required=False)
    return templates, not os.path.exists(mask)


def _lbl_steps(runparams, keyword_args, instrument, data_dir, data_source,
               data_type, object_science, object_comparison, object_teff,
               blaze_corr, blaze_file, report=True):
    """Template, mask, compute, compile and report, as asked, on one object.

    A function of its own so that a PCA2D run can measure the delivered
    object and then the corrected one. Returns the report's PDF, or None.
    `report=False` measures without a report of its own: a PCA2D run's
    delivered object, whose velocities the corrected one's report holds.
    """
    lbl_report = None
    # ---------------------------------------------------------------------
    # if runparams['RUN_LBL_RESMAP']:
    #     lbl_resmap.main(instrument=instrument, data_dir=data_dir,
    #                     data_source=data_source,
    #                     data_type=data_type,
    #                     object_science=object_science,
    #                     object_comparison=object_comparison,
    #                     blaze_corrected=blaze_corr,
    #                     blaze_file=blaze_file,
    #                     overwrite=not runparams['SKIP_LBL_RESMAP'],
    #                     **keyword_args)
    # ---------------------------------------------------------------------
    # make the template (if not present)
    if runparams['RUN_LBL_TEMPLATE']:
        # Must produce the template for the science data and the template
        #   we use a set to do this (only runs once if they are the same)
        for _obj_template in {object_science, object_comparison}:
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science,
                              object_comparison=_obj_template,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
                              object_teff=object_teff,
                              overwrite=not runparams['SKIP_LBL_TEMPLATE'],
                              **keyword_args)
    # ---------------------------------------------------------------------
    # make the mask (if not present)
    if runparams['RUN_LBL_MASK']:
        # Must produce the mask for the science data and the template
        #   we use a set to do this (only runs once if they are the same)
        for _obj_comparison in {object_science, object_comparison}:
            lbl_mask.main(instrument=instrument, data_dir=data_dir,
                          data_source=data_source,
                          data_type=data_type,
                          object_science=object_science,
                          object_comparison=_obj_comparison,
                          object_teff=object_teff,
                          overwrite=not runparams['SKIP_LBL_MASK'],
                          **keyword_args)
    # ---------------------------------------------------------------------
    # # make the noise model (if not present)
    # if runparams['RUN_LBL_NOISE']:
    #     lbl_noise(instrument=instrument, data_dir=data_dir,
    #               object_science=object_science,
    #               object_comparison=object_comparison,
    #               **keyword_args)
    # ---------------------------------------------------------------------
    # A TEMPLATE OR A MASK COMPUTE NEEDS AND THAT IS NOT THERE (RUN_LBL_
    #   TEMPLATE or RUN_LBL_MASK off, and a PCA2D correction made again has
    #   moved the previous one's to superseded/, say): made now, said,
    #   rather than lbl_compute stopping on it after the correction
    if runparams['RUN_LBL_COMPUTE']:
        templates, mask = _missing_for_compute(
            keyword_args, instrument=instrument, data_dir=data_dir,
            data_source=data_source, data_type=data_type,
            object_science=object_science,
            object_comparison=object_comparison,
            blaze_corrected=blaze_corr, blaze_file=blaze_file)
        for _obj_template in templates:
            log.warning('the template of {0} is not there, and lbl_compute'
                        ' needs it: made now (RUN_LBL_TEMPLATE={1})'.format(
                            _obj_template, runparams['RUN_LBL_TEMPLATE']))
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science,
                              object_comparison=_obj_template,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file, overwrite=False,
                              object_teff=object_teff,
                              **keyword_args)
        if mask:
            log.warning('the mask of {0} is not there, and lbl_compute needs'
                        ' it: made now (RUN_LBL_MASK={1})'.format(
                            object_comparison, runparams['RUN_LBL_MASK']))
            lbl_mask.main(instrument=instrument, data_dir=data_dir,
                          data_source=data_source,
                          data_type=data_type,
                          object_science=object_science,
                          object_comparison=object_comparison,
                          object_teff=object_teff, overwrite=False,
                          **keyword_args)
    # ---------------------------------------------------------------------
    # run the compute code
    if runparams['RUN_LBL_COMPUTE']:
        lbl_compute.main(instrument=instrument, data_dir=data_dir,
                         data_source=data_source,
                         data_type=data_type,
                         object_science=object_science,
                         object_comparison=object_comparison,
                         blaze_corrected=blaze_corr,
                         blaze_file=blaze_file,
                         skip_done=runparams['SKIP_LBL_COMPUTE'],
                         object_teff=object_teff,
                         **keyword_args)
    # ---------------------------------------------------------------------
    # run the compile code
    if runparams['RUN_LBL_COMPILE']:
        lbl_compile.main(instrument=instrument, data_dir=data_dir,
                         data_source=data_source,
                         data_type=data_type,
                         object_science=object_science,
                         object_comparison=object_comparison,
                         skip_done=runparams['SKIP_LBL_COMPILE'],
                         **keyword_args)
    # ---------------------------------------------------------------------
    # THE REPORT, at the very end and never fatal. It is an account of the
    # run, not a computation step: if it cannot be drawn, the run stays
    # good and this is said, the run is not lost.
    if report and runparams.get('REPORT', True) and \
            runparams['RUN_LBL_COMPILE']:
        try:
            from lbl.report import make_report
            rdb = os.path.join(data_dir, 'lblrdb',
                               'lbl_%s_%s.rdb' % (object_science,
                                                  object_comparison))
            lbl_report = make_report(data_dir=data_dir, rdb_file=rdb,
                                     title='%s_%s' % (object_science,
                                                      object_comparison),
                                     science_dir=os.path.join(
                                         data_dir, 'science', object_science),
                                     tc_dir=os.path.join(
                                         data_dir, 'science',
                                         object_science + '_tc'),
                                     template_file=_template_of(
                                         data_dir, object_comparison,
                                         instrument),
                                     instrument=instrument,
                                     data_source=data_source,
                                     # the site of a Generic instrument,
                                     #   where its nights are cut; any
                                     #   other's class names its own
                                     earth_location=(
                                         runparams.get('EARTH_LOCATION')
                                         if str(instrument).upper()
                                         == 'GENERIC' else None),
                                     # PLANETS: the K of its confirmed
                                     #   planets of known ephemeris fitted
                                     ephemerides_of=(object_science if
                                                    runparams.get('PLANETS')
                                                    else None),
                                     # its TESS objects of interest,
                                     #   confirmed or not, always fitted
                                     tois_of=object_science,
                                     # its systemic velocity colours the
                                     #   series by V_tot (SIMBAD's)
                                     star=object_science)
        except Exception as _exc:
            log.warning('report failed: {0}'.format(_exc))
    return lbl_report


def _template_of(data_dir: str, objname: str, instrument: str):
    """The template LBL made of an object, for the report to say whether it
    was made on known orbits (PLANETS). Its name depends on the instrument
    (LBL_Template_<object>_nirps_he.fits, Template_<object>_HARPS_ESO.fits...)
    so it is looked for; the templates of the objects made from this one
    (<object>_pca2d..., <object>_tc) are not it. None if there is none.
    A vetted template's is in models/template_<INSTRUMENT>_<DATA_SOURCE>
    (lbl.core.vetted)."""
    folders = [os.path.join(data_dir, 'templates')] + sorted(glob.glob(
        os.path.join(data_dir, 'models', 'template_{0}_*'.format(
            str(instrument).upper()))))
    pattern = re.compile(r'^(LBL_)?Template_{0}_(.+)\.fits$'.format(
        re.escape(str(objname))))
    found = []
    for folder in folders:
        for name in (os.listdir(folder) if os.path.isdir(folder) else []):
            match = pattern.match(name)
            if match and not match.group(2).startswith(('pca2d', 'tc_',
                                                        'tc.')):
                found.append(os.path.join(folder, name))
    if not found:
        return None
    # the instrument's own, else the most recent
    mine = [path for path in found
            if str(instrument).lower() in os.path.basename(path).lower()]
    return max(mine or found, key=os.path.getmtime)


def _pca2d_joint(runparams: dict, object_sciences, data_types,
                 object_comparisons, pca2d_args: dict, keyword_args: dict,
                 instrument: str, data_dir: str, data_source: str,
                 blaze_corrs, blaze_files) -> dict:
    """The joint pca2d correction of every science object of this wrap.

    Not after the telluric cleaning: that is done one object at a time inside
    the loop, so the other objects would not be cleaned yet when the joint
    fit needs them all. Clean them first, in a wrap without PCA2D."""
    from lbl.recipes import lbl_pca2d
    if runparams['RUN_LBL_TELLUCLEAN']:
        emsg = ('LBL_WRAP ERROR: PCA2D joint=True cannot follow '
                'RUN_LBL_TELLUCLEAN in the same wrap. Run the telluric '
                'cleaning first, then the joint correction on the _tc objects')
        raise base_classes.LblException(emsg)
    olen = len(object_sciences)
    objects, comparisons, blazes = [], dict(), dict()
    for num in range(olen):
        wkargs = dict(iteration=num, length=olen)
        if lbl_misc.wraplistcheck(data_types, 'DATA_TYPES',
                                  **wkargs) != 'SCIENCE':
            continue
        objects.append(object_sciences[num])
        comparisons[object_sciences[num]] = lbl_misc.wraplistcheck(
            object_comparisons, 'OBJECT_COMPARISON', **wkargs)
        blazes[object_sciences[num]] = (
            lbl_misc.wraplistcheck(blaze_corrs, 'BLAZE_CORRECTED', **wkargs),
            lbl_misc.wraplistcheck(blaze_files, 'BLAZE_FILE', **wkargs))
    if len(objects) < 2:
        emsg = ('LBL_WRAP ERROR: PCA2D joint=True needs two science objects '
                'or more, there is {0}')
        raise base_classes.LblException(emsg.format(', '.join(objects)))
    return _pca2d_passes(runparams, keyword_args, instrument, data_dir,
                         data_source, pca2d_args, comparisons, blazes,
                         joint=True)


def _fine_velocities(runparams: dict) -> bool:
    """Whether the velocities are adjusted finely: SMALL_RV_SHIFT False,
    the template iterating on them"""
    return runparams.get('SMALL_RV_SHIFT', True) in (False, 'False', 0)


def _iterated(data_dir: str, objname: str, instrument: str) -> bool:
    """Whether the object's own template measured its spectra's velocities
    (RVITER in its header)"""
    path = _template_of(data_dir, objname, instrument)
    if path is None:
        return False
    try:
        from astropy.io import fits
        return bool(fits.getheader(path).get('RVITER', False))
    except Exception:                                         # noqa: BLE001
        return False


def _pca2d_passes(runparams: dict, keyword_args: dict, instrument: str,
                  data_dir: str, data_source: str, pca2d_args: dict,
                  comparisons: dict, blazes: dict, joint: bool) -> dict:
    """The PCA2D correction of these objects: {object: (corrected object,
    its comparison)}.

    THE FIRST PASS IS ON THE BERV ALONE (with the known orbits, PLANETS),
    whatever SMALL_RV_SHIFT says: the velocities of the delivered spectra
    carry the tellurics' imprint where the star's lines cross them (the BERV
    crossing), and a star frame built on them copies it into the correction
    (TOI-904, 14 NIRPS spectra, 2026-11-24: LBL at 51 to 65 m/s rms after a
    correction on them, against 21 m/s on the BERV alone; the template's
    velocities 60 m/s rms on the delivered spectra, 15 on the corrected
    ones). SMALL_RV_SHIFT True: that pass only.

    SMALL_RV_SHIFT False, A SECOND PASS FOR A STAR THAT MOVES: the
    corrected spectra's own template, iterating on their velocities
    (cleaner), and, when they spread by more than
    MAX_CONVERGENCE_TEMPLATE_RV (100 m/s, half their 16-84 % range), a
    second correction whose star frame follows them (lbl_pca2d
    velocities_from), which LBL then measures: a binary, a brown dwarf, a
    hot Jupiter. Under it, the first pass is kept: the velocities are then
    mostly their errors, and PCA2D copies the star frame into the
    correction (TOI-904, a BERV span of 1.8 km/s: the corrected spectra's
    velocities, 16 m/s rms and correlated with the BERV, followed, took LBL
    from 21.1 to 31.9 m/s rms, the same pattern 1.5 times larger,
    correlation 0.99). Each object of a joint fit on its own. When the
    corrected spectra's template already has its velocities (a previous
    run's), no first pass: the one they ask for."""
    from lbl.recipes import lbl_pca2d
    objects = list(comparisons)

    def correct(velocities_from):
        if joint:
            return lbl_pca2d.joint(objects=objects, comparisons=comparisons,
                                   velocities_from=velocities_from,
                                   **pca2d_args)
        name = objects[0]
        return {name: lbl_pca2d.main(object_science=name,
                                     object_comparison=comparisons[name],
                                     velocities_from=velocities_from,
                                     **pca2d_args)}

    # the BERV alone: no object's velocities for the star frame
    berv_only = {name: None for name in objects}
    if not _fine_velocities(runparams):
        return correct(berv_only)
    corrected = {name: lbl_pca2d.corrected_name(
        name, pca2d_args.get('pca2d'), pca2d_args.get('report', True))
        for name in objects}
    if all(_iterated(data_dir, corrected[name], instrument)
           for name in objects):
        log.info('SMALL_RV_SHIFT = False: the corrected spectra of {0} have '
                 'a template with their velocities (a previous run\'s): one '
                 'PCA2D pass'.format(', '.join(objects)))
        return correct(_follow(runparams, data_dir, instrument, corrected))
    log.info('SMALL_RV_SHIFT = False: PCA2D in two passes, the first on the '
             'BERV alone (the delivered spectra\'s velocities carry the '
             'tellurics\' imprint), the second on the velocities of the '
             'corrected spectra\'s template')
    names = correct(berv_only)
    for name in objects:
        _velocity_template(runparams, keyword_args, instrument, data_dir,
                           data_source, names[name][0], *blazes[name])
    follow = _follow(runparams, data_dir, instrument,
                     {name: names[name][0] for name in objects})
    if not any(follow.values()):
        # this line's start is what the window looks for (lbl_gui's runner
        #   drops the second pass's steps it planned)
        log.info('PCA2D: no second pass, the first is kept (no star moves by '
                 'more than MAX_CONVERGENCE_TEMPLATE_RV)')
        return names
    log.info('PCA2D, second pass: the star frame follows the velocities of '
             'the corrected spectra\'s template ({0})'.format(', '.join(
                 name for name in objects if follow[name])))
    return correct(follow)


def _follow(runparams: dict, data_dir: str, instrument: str,
            corrected: dict) -> dict:
    """{object: its corrected spectra, whose velocities PCA2D's star frame
    follows, when they spread by more than MAX_CONVERGENCE_TEMPLATE_RV
    (half their 16-84 % range); None, the BERV alone, otherwise}
    (_pca2d_passes)"""
    from lbl.core.parameters import params as defaults
    from lbl.recipes import lbl_pca2d
    limit = runparams.get('MAX_CONVERGENCE_TEMPLATE_RV')
    if limit is None:
        limit = defaults['MAX_CONVERGENCE_TEMPLATE_RV']
    out = dict()
    for name, other in corrected.items():
        found = lbl_pca2d.object_velocities(other, None, data_dir,
                                            instrument, lambda *a: None)
        values = 1000.0 * np.array(list(found.values()), dtype=float)
        values = values[np.isfinite(values)]
        spread = float(np.diff(np.percentile(values, [16, 84]))[0] / 2) \
            if values.size >= 3 else 0.0
        moves = spread > float(limit)
        out[name] = other if moves else None
        log.info('{0}: the velocities of its corrected spectra spread by '
                 '{1:.1f} m/s ({2} spectra), {3} MAX_CONVERGENCE_TEMPLATE_RV '
                 '({4:g} m/s): {5}'.format(
                     name, spread, values.size,
                     'over' if moves else 'under', float(limit),
                     'PCA2D\'s star frame follows them' if moves else
                     'the star frame stays on the BERV (velocities this '
                     'small are mostly their errors, which PCA2D would copy '
                     'into the correction)'))
    return out


def custom(runparams: dict, func, required_kwargs: List[str]):
    """
    Wrapper around __main__ recipe code (deals with errors and loads instrument
    profile)

    :param runparams: dict, parameters to pass to lbl recipes

    :return:
    """
    # reset the sys.argv (arguments from command line aren't used)
    sys.argv = [__NAME__]
    # get key parameters
    instrument = lbl_misc.check_runparams(runparams, 'INSTRUMENT')
    data_dir = lbl_misc.check_runparams(runparams, 'DATA_DIR')
    data_source = lbl_misc.check_runparams(runparams, 'DATA_SOURCE')
    data_types = lbl_misc.check_runparams(runparams, 'DATA_TYPES')
    object_sciences = _object_list(
        lbl_misc.check_runparams(runparams, 'OBJECT_SCIENCE'))
    object_comparisons = lbl_misc.check_runparams(runparams, 'OBJECT_COMPARISON')
    object_teffs = lbl_misc.check_runparams(runparams, 'OBJECT_TEFF')
    blaze_corrs = lbl_misc.check_runparams(runparams, 'BLAZE_CORRECTED',
                                           required=False)
    blaze_files = lbl_misc.check_runparams(runparams, 'BLAZE_FILE',
                                           required=False)
    # -------------------------------------------------------------------------
    # push other keyword arguments into keyword arguments dictionary
    keyword_args = dict()
    for key in runparams:
        if key not in REMOVE_KEYS:
            keyword_args[key] = runparams[key]
    # make sure we have defaults if key not in runparams
    for key in DEFAULTS:
        if key not in runparams:
            runparams[key] = DEFAULTS[key]
    # -------------------------------------------------------------------------
    # sanity checks on runparams (certain things should not be set together)
    if runparams['RUN_LBL_MASK'] and 'MASK_FILE' in runparams:
        if runparams['MASK_FILE'] not in [None, 'None', '', 'Null']:
            emsg = ('LBL_WRAP ERROR: Cannot have RUN_LBL_MASK=True and '
                    'MASK_FILE={0} (Must be unset)')
            raise base_classes.LblException(emsg.format(runparams['MASK_FILE']))
    # -------------------------------------------------------------------------
    # mark the expected length if a list
    olen = len(object_sciences)
    # loop around all files
    for num in range(olen):
        # get the science target
        object_science = object_sciences[num]
        # print wrapper splash
        lbl_misc.splash(name=__STRNAME__ + '[CUSTOM]', instrument=instrument,
                        plogger=log)
        # print iteration we are running
        msg = 'Running [{0}] iteration {1}/{2}'
        margs = [object_science, num + 1, olen]
        log.info(msg.format(*margs))
        # wrap check args
        wkargs = dict(iteration=num, length=olen)
        # get this iterations values (and check if they are a list of matching
        #    length to object_sciences) or just a single value
        data_type = lbl_misc.wraplistcheck(data_types,
                                           'DATA_TYPES', **wkargs)
        object_comparison = lbl_misc.wraplistcheck(object_comparisons,
                                                 'OBJECT_COMPARISON', **wkargs)
        object_teff = lbl_misc.wraplistcheck(object_teffs,
                                             'OBJECT_TEFF', **wkargs)
        blaze_corr = lbl_misc.wraplistcheck(blaze_corrs,
                                            'BLAZE_CORRECTED', **wkargs)
        blaze_file = lbl_misc.wraplistcheck(blaze_files,
                                            'BLAZE_FILE', **wkargs)
        # push these into a dictionary (for accessing)
        all_kwargs = dict()
        all_kwargs['instrument'] = instrument
        all_kwargs['data_source'] = data_source
        all_kwargs['data_dir'] = data_dir
        all_kwargs['data_type'] = data_type
        all_kwargs['object_science'] = object_science
        all_kwargs['object_comparison'] = object_comparison
        all_kwargs['object_teff'] = object_teff
        all_kwargs['blaze_corrected'] = blaze_corr
        all_kwargs['blaze_corrected'] = blaze_corr
        all_kwargs['blaze_file'] = blaze_file
        # get the required keyword arguemnts
        rkwargs = dict()
        for required_kwarg in required_kwargs:
            rkey = required_kwarg.lower()
            # if we have this key add it
            if rkey in all_kwargs:
                rkwargs[rkey] = all_kwargs[rkey]
        # add all keyword_args
        rkwargs.update(keyword_args)
        # ---------------------------------------------------------------------
        # run defined function
        func.main(**rkwargs)



# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set up parameters
    rparams = dict()
    rparams['INSTRUMENT'] = 'SPIROU'
    rparams['DATA_DIR'] = '/data/spirou/data/lbl/'
    # science criteria
    rparams['DATA_TYPES'] = ['FP', 'SCIENCE']
    rparams['OBJECT_SCIENCE'] = ['FP', 'GL699']
    rparams['OBJECT_COMPARISON'] = ['FP', 'GL699']
    rparams['OBJECT_TEFF'] = [300, 3224]
    # what to run
    rparams['RUN_LBL_TELLUCLEAN'] = True
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    # whether to skip done files
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # run main
    main(rparams)

# =============================================================================
# End of code
# =============================================================================

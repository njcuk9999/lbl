#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from typing import List
import glob
import os
import re
import sys

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.recipes import lbl_compile
from lbl.recipes import lbl_compute
from lbl.recipes import lbl_mask
from lbl.recipes import lbl_resmap
from lbl.recipes import lbl_telluclean
from lbl.recipes import lbl_template
from lbl.recipes import lbl_reset
from lbl.resources import lbl_misc

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
    'RUN_LBL_RESET',
    'RUN_LBL_TELLUCLEAN', 'RUN_LBL_TEMPLATE', 'RUN_LBL_MASK',
    'RUN_LBL_COMPUTE', 'RUN_LBL_COMPILE',
    # skip keys
    'SKIP_LBL_TELLUCLEAN', 'SKIP_LBL_TEMPLATE', 'SKIP_LBL_MASK',
    'SKIP_LBL_COMPUTE', 'SKIP_LBL_COMPILE', 'SKIP_LBL_PCA2D',
    # general keys already used
    'SKIP_DONE', 'OVERWRITE', 'TELLUCLEAN_USE_TEMPLATE',
    # the pca2d precleaning, lbl_pca2d's and no other recipe's
    'PCA2D',
    # le rapport de fin de run
    'REPORT']

# Define the default values
DEFAULTS = dict()
DEFAULTS['RUN_LBL_RESET'] = False
DEFAULTS['RUN_LBL_TELLUCLEAN'] = False
DEFAULTS['RUN_LBL_RESMAP'] = False
DEFAULTS['RUN_LBL_TEMPLATE'] = False
DEFAULTS['RUN_LBL_MASK'] = False
DEFAULTS['RUN_LBL_COMPUTE'] = False
DEFAULTS['RUN_LBL_COMPILE'] = False
#: LE RAPPORT EST FAIT PAR DEFAUT. Il coute quelques secondes a la fin d'un
#: run qui en a pris des minutes, et c'est lui qui dit si le run a marche.
DEFAULTS['REPORT'] = True
DEFAULTS['SKIP_LBL_TEMPLATE'] = False
DEFAULTS['SKIP_LBL_MASK'] = False
DEFAULTS['SKIP_LBL_COMPUTE'] = False
DEFAULTS['SKIP_LBL_COMPILE'] = False
#: NI VRAI NI FAUX PAR DEFAUT: la correction PCA2D n'est refaite que si les
#: reglages, les fichiers d'entree ou nominal.yaml ont change. True garde
#: celle sur disque meme alors, False la refait toujours (lbl_pca2d.correct)
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
    object_sciences = lbl_misc.check_runparams(runparams, 'OBJECT_SCIENCE')
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
        if runparams['RUN_LBL_RESET']:
            lbl_reset.main(instrument=instrument, data_dir=data_dir,
                           data_source=data_source, **keyword_args)
        # ---------------------------------------------------------------------
        # run all pre-cleaning steps
        if runparams['RUN_LBL_TELLUCLEAN'] and data_type == 'SCIENCE':
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
            # update template name
            if not object_comparison.endswith('_tc'):
                object_comparison = object_comparison + '_tc'
            # make the template (if not present)
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science + '_tc',
                              object_comparison=object_comparison,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
                              overwrite=True,
                              **keyword_args)
            # re-run tellu clean with uncorrected science data now using our
            #  template (made from cleaned science data)
            lbl_telluclean.main(instrument=instrument, data_dir=data_dir,
                                data_source=data_source,
                                data_type=data_type,
                                object_science=object_science,
                                object_comparison=object_comparison,
                                skip_done=False,
                                telluclean_use_template=True,
                                blaze_corrected=blaze_corr,
                                blaze_file=blaze_file,
                                **keyword_args)
            # the template made again, from the second pass's spectra: LBL's
            #   template step below keeps a template already on disk (unless
            #   OVERWRITE), so the first pass's would have stayed, and the
            #   mask, the velocities and PCA2D would have measured the second
            #   pass's spectra against it (GJ 1214, 2026-10-25)
            lbl_template.main(instrument=instrument, data_dir=data_dir,
                              data_source=data_source,
                              data_type=data_type,
                              object_science=object_science + '_tc',
                              object_comparison=object_comparison,
                              blaze_corrected=blaze_corr,
                              blaze_file=blaze_file,
                              overwrite=True,
                              **keyword_args)
            # update object name
            if not object_science.endswith('_tc'):
                object_science = object_science + '_tc'
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
        # run once, on the object, exactly as they always did.
        if runparams.get('PCA2D') not in (None, False) and \
                data_type == 'SCIENCE':
            from lbl.recipes import lbl_pca2d
            pca2d = runparams['PCA2D']
            delivered = object_science
            # the delivered object measured first, so that the report can set
            # its velocities beside the corrected ones (PCA2D compare=True)
            if lbl_pca2d.compare(pca2d) and object_science not in measured:
                _lbl_steps(object_science=object_science,
                           object_comparison=object_comparison, **steps)
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
                        steps,
                        object_teff=lbl_misc.wraplistcheck(
                            object_teffs, 'OBJECT_TEFF', **kw_other),
                        blaze_corr=lbl_misc.wraplistcheck(
                            blaze_corrs, 'BLAZE_CORRECTED', **kw_other),
                        blaze_file=lbl_misc.wraplistcheck(
                            blaze_files, 'BLAZE_FILE', **kw_other))
                    _lbl_steps(object_science=object_sciences[other],
                               object_comparison=lbl_misc.wraplistcheck(
                                   object_comparisons, 'OBJECT_COMPARISON',
                                   **kw_other), **steps_other)
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
                    pca2d_joint = _pca2d_joint(runparams, object_sciences,
                                               data_types, object_comparisons,
                                               pca2d_args)
                object_science, object_comparison = \
                    pca2d_joint[object_science]
            else:
                object_science, object_comparison = lbl_pca2d.main(
                    object_science=object_science,
                    object_comparison=object_comparison, **pca2d_args)
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


def _lbl_steps(runparams, keyword_args, instrument, data_dir, data_source,
               data_type, object_science, object_comparison, object_teff,
               blaze_corr, blaze_file):
    """Template, mask, compute, compile and report, as asked, on one object.

    The body of the loop as it always was, made a function so that a PCA2D
    run can measure the delivered object and then the corrected one. Returns
    the report's PDF, or None.
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
    # LE RAPPORT, a la toute fin et jamais fatal. C'est un compte rendu du
    # run, pas une etape de calcul: s'il ne peut pas se dessiner, le run
    # reste bon et on le dit, on ne le perd pas.
    if runparams.get('REPORT', True) and runparams['RUN_LBL_COMPILE']:
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
                                     # PLANETS: the K of its transiting
                                     #   planets and candidates fitted
                                     candidates_of=(object_science if
                                                    runparams.get('PLANETS')
                                                    else None))
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
                 object_comparisons, pca2d_args: dict) -> dict:
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
    objects, comparisons = [], dict()
    for num in range(olen):
        wkargs = dict(iteration=num, length=olen)
        if lbl_misc.wraplistcheck(data_types, 'DATA_TYPES',
                                  **wkargs) != 'SCIENCE':
            continue
        objects.append(object_sciences[num])
        comparisons[object_sciences[num]] = lbl_misc.wraplistcheck(
            object_comparisons, 'OBJECT_COMPARISON', **wkargs)
    if len(objects) < 2:
        emsg = ('LBL_WRAP ERROR: PCA2D joint=True needs two science objects '
                'or more, there is {0}')
        raise base_classes.LblException(emsg.format(', '.join(objects)))
    return lbl_pca2d.joint(objects=objects, comparisons=comparisons,
                           **pca2d_args)


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
    object_sciences = lbl_misc.check_runparams(runparams, 'OBJECT_SCIENCE')
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

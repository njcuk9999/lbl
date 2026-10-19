#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_pca2d

The two-frame precleaning of pca2d (lbl.pca2d), run on one science object
before LBL measures it. lbl_wrap calls it when the run parameters carry a
PCA2D dict, and measures the velocities on what it writes:

    rparams['PCA2D'] = dict(n_star=1, n_earth=3)

Without that key nothing here is imported and LBL is what it was.

    reads    DATA_DIR/science/<object>/         the object as LBL sees it, so
                                                <object>_tc after the telluric
                                                cleaning when that was asked
    writes   DATA_DIR/science/<object>_pca2d/   the corrected spectra, under
                                                their own names and in their
                                                own format, which LBL reads
                                                like any other object
             DATA_DIR/pca2d/<object>/           the request, the fit, the
                                                figures and the report
             DATA_DIR/pca2d/cache/              the cubes, rebuildable

The dict takes the component counts, a few switches, and any key of the
pca2d configuration (lbl/pca2d/nominal.yaml): by the short names of the
pca2d window, as 'section.key', or as a nested dict per section. Everything
not given is the nominal set, the one every comparison of the project ran on.

    n_star       star-frame components fitted (1)
    n_earth      observer-frame components fitted and divided out (3)
    figures      the run's figures and its report (True)
    overwrite    redo the correction even when the same one is on disk (False)
    name         a label: <object>_pca2d_<name>, for keeping several
                 corrections of one object side by side (None)
    joint        fit every OBJECT_SCIENCE of the wrap against one observer
                 basis, each with its own star spectrum (False)

A request already done, same settings and same input files, is not done
again. A different one moves the previous corrected object and every LBL
product made from it (lblrv, lblrdb, templates, masks...) aside, under
DATA_DIR/pca2d/<object>/superseded/, so that LBL's SKIP_DONE never mixes
velocities measured on one correction with a template made from another.

Created on 2026-10-19
"""
import copy
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import yaml

from lbl.core import base_classes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl_pca2d.py'
__STRNAME__ = 'LBL PCA2D'
LblException = base_classes.LblException
#: what the corrected object is called, after the object it comes from
SUFFIX = '_pca2d'
#: the switches of this recipe, and what they are when not given
DEFAULTS = dict(n_star=1, n_earth=3, figures=True, overwrite=False,
                name=None, joint=False)
#: the short names, as the pca2d window spells them, and the key of the
#: pca2d configuration each one is
ALIASES = {
    'velocity_term': 'twoframe.velocity_term',
    'iters': 'twoframe.iters',
    'shrink': 'correct.shrink',
    'nsig_cut': 'correct.nsig_cut',
    'bias_nsig': 'correct.bias_nsig',
    'column_frac': 'correct.column_frac',
    'column_chi2': 'correct.column_chi2',
    'weight': 'correct.weight',
    'mask': 'correct.mask',
    'mask_share': 'correct.mask_share',
    'width_kms': 'highpass.width_kms',
    'high_pass': 'highpass.width_kms',
    'dv': 'domain.dv',
    'nightly_stack': 'input.nightly_stack',
    'min_rjd': 'quality.min_rjd',
    'max_rjd': 'quality.max_rjd',
    'drs_modes': 'quality.drs_modes',
    'min_pixel_snr': 'weights.min_pixel_snr',
    'windows': 'output.windows',
    'cache_directory': 'output.cache_directory',
}
#: keys this recipe sets itself: where the spectra are and where things go
MANAGED = ('input.directory', 'input.object', 'output.directory',
           'output.fits_directory')
#: LBL's NIRPS modes, and the DRS mode the files of each one carry
NIRPS_MODES = {'NIRPS_HE': 'HE', 'NIRPS_HA': 'HA'}


# =============================================================================
# Define functions
# =============================================================================
def nominal_path() -> str:
    """The nominal configuration shipped with lbl.pca2d."""
    import lbl.pca2d
    return os.path.join(os.path.dirname(os.path.abspath(lbl.pca2d.__file__)),
                        'nominal.yaml')


def read_nominal() -> dict:
    from lbl.pca2d.config import read_yaml
    with open(nominal_path()) as handle:
        return read_yaml(handle) or {}


def known_keys(nominal: dict) -> dict:
    """{section: set of keys} that the pca2d configuration knows: its code
    defaults, the nominal file's general block and every instrument block."""
    from lbl.pca2d.config import DEFAULTS as CODE_DEFAULTS
    known = {}
    blocks = [CODE_DEFAULTS, nominal.get('general') or {}]
    blocks += list((nominal.get('instruments') or {}).values())
    for block in blocks:
        for section, values in block.items():
            if isinstance(values, dict):
                known.setdefault(section, set()).update(values)
    return known


def settings(pca2d) -> tuple:
    """(switches, overrides): the recipe's own switches, and the pca2d
    configuration keys asked for as {section: {key: value}}.

    PCA2D = True is the nominal set. An unknown key stops the run here,
    before anything is read: LBL does the same for its own parameters, and a
    misspelt key that was silently ignored would be a run of the nominal
    under another name.
    """
    if pca2d is True:
        pca2d = dict()
    if not isinstance(pca2d, dict):
        raise LblException('PCA2D must be a dict (or True for the nominal'
                           ' set), not {0!r}'.format(pca2d))
    nominal = read_nominal()
    known = known_keys(nominal)
    switches = dict(DEFAULTS)
    overrides = dict()
    unknown = []

    def put(path, value):
        section, _, key = path.partition('.')
        if path in MANAGED:
            raise LblException('PCA2D cannot set {0}: lbl_pca2d puts the'
                               ' spectra and the products where LBL expects'
                               ' them'.format(path))
        if section not in known or key not in known[section]:
            unknown.append(path)
            return
        overrides.setdefault(section, dict())[key] = value

    for name, value in pca2d.items():
        if name in DEFAULTS:
            switches[name] = value
        elif name in ALIASES:
            put(ALIASES[name], value)
        elif '.' in name:
            put(name, value)
        elif name in known and isinstance(value, dict):
            for key, sub in value.items():
                put('{0}.{1}'.format(name, key), sub)
        else:
            unknown.append(name)
    if unknown:
        raise LblException(
            'PCA2D does not know {0}. It takes {1}, the short names {2}, and'
            ' any key of lbl/pca2d/nominal.yaml as "section.key" or as a dict'
            ' per section ({3})'.format(
                ', '.join(sorted(unknown)), ', '.join(DEFAULTS),
                ', '.join(sorted(ALIASES)), ', '.join(sorted(known))))
    for name in ('n_star', 'n_earth'):
        switches[name] = int(switches[name])
    return switches, overrides


def input_files(science_dir: str, pattern: str) -> list:
    """The object's spectra, as LBL would glob them."""
    return sorted(glob.glob(os.path.join(science_dir, pattern or '*.fits')))


def request_stamp(request: dict, switches: dict, files: list) -> str:
    """What makes a correction this one: the settings, the input files (name,
    size, date) and the nominal file they sit on. Not the figures, which
    change nothing in the spectra."""
    blob = dict(request=request,
                counts=[switches['n_star'], switches['n_earth']],
                files=[(os.path.basename(f), os.path.getsize(f),
                        int(os.path.getmtime(f))) for f in files])
    with open(nominal_path(), 'rb') as handle:
        blob['nominal'] = hashlib.sha1(handle.read()).hexdigest()
    text = json.dumps(blob, sort_keys=True, default=str).encode()
    return hashlib.sha1(text).hexdigest()[:12]


def lbl_products(data_dir: str, name: str, science_subdir: str = 'science',
                 skip=('pca2d',)) -> list:
    """Everything LBL made from the object `name`, anywhere in its tree: a
    file or folder whose name holds `name` between separators.

    Not what it made from a longer name that starts with this one: the
    labelled corrections of the same star, <object>_pca2d_<label>, are other
    objects, and are taken out of each name before it is matched."""
    token = re.compile(r'(^|[_.\-]){0}([_.\-]|$)'.format(re.escape(name)))
    science = os.path.join(data_dir, science_subdir)
    longer = sorted((d for d in (os.listdir(science)
                                 if os.path.isdir(science) else [])
                     if d.startswith(name + '_')), key=len, reverse=True)
    found = []
    for sub in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, sub)
        if sub in skip or sub == science_subdir or not os.path.isdir(folder):
            continue
        for entry in sorted(os.listdir(folder)):
            bare = entry
            for other in longer:
                bare = bare.replace(other, '#')
            if token.search(bare):
                found.append(os.path.join(folder, entry))
    return found


def size_of(path: str) -> int:
    if os.path.isfile(path) or os.path.islink(path):
        return os.lstat(path).st_size
    total = 0
    for base, _, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def supersede(data_dir: str, corr_dir: str, name: str, work: str, log,
              science_subdir: str = 'science') -> None:
    """Move the previous corrected object, and what LBL made from it, aside.

    Moved and not deleted: a correction is minutes to hours, its velocities
    hours, and which of two of them was right is sometimes only known after.
    The folder is said, with its size, so that it can be removed by hand.
    """
    going = ([corr_dir] if os.path.isdir(corr_dir) else [])
    going += lbl_products(data_dir, name, science_subdir)
    if not going:
        return
    stamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    aside = os.path.join(work, 'superseded', stamp)
    total = 0
    for path in going:
        rel = os.path.relpath(path, data_dir)
        target = os.path.join(aside, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        total += size_of(path)
        shutil.move(path, target)
    log('the previous %s and what LBL made from it (%d items, %.1f GB) moved'
        ' to %s: another correction, so none of it may be reused'
        % (name, len(going), total / 1e9, aside), 'warn')


def request_block(overrides: dict, instrument: str, data_source,
                  input_file) -> dict:
    """An object's block of the request: what PCA2D changed, and what LBL
    already knows about the files, its reader and its NIRPS mode."""
    block = copy.deepcopy(overrides)
    lbl_block = block.setdefault('lbl', dict())
    lbl_block.setdefault('instrument', instrument)
    if data_source is not None:
        lbl_block.setdefault('data_source', data_source)
    if instrument in NIRPS_MODES:
        block.setdefault('quality', dict()).setdefault(
            'drs_modes', [NIRPS_MODES[instrument]])
    if input_file not in (None, '', '*', '*.fits'):
        block.setdefault('input', dict()).setdefault('pattern', input_file)
    return block


def write_request(path: str, objects: list, block: dict, cache_dir: str,
                  model_dir: str) -> None:
    """The request as a pca2d configuration file: the nominal one, with one
    block per object holding the same settings."""
    nominal = read_nominal()
    output = nominal.setdefault('general', dict()).setdefault('output', dict())
    output['cache_directory'] = cache_dir
    output['model_directory'] = model_dir
    # pca2d's own LBL stage is never run from here: LBL is the caller
    nominal['general'].setdefault('lbl', dict()).update(prepare=False,
                                                        run=False)
    nominal['objects'] = {name: copy.deepcopy(block) for name in objects}
    with open(path, 'w') as handle:
        handle.write('# what lbl_pca2d asked of pca2d for %s: the nominal'
                     ' configuration\n# of lbl/pca2d, with each object\'s'
                     ' block holding the PCA2D dict\n' % ', '.join(objects))
        yaml.safe_dump(nominal, handle, sort_keys=False,
                       default_flow_style=False)


def main(instrument: str, data_dir: str, object_science: str,
         object_comparison: str = None, data_source: str = None,
         pca2d=None, input_file: str = None, science_subdir: str = 'science',
         **kwargs) -> tuple:
    """Correct one object, and return the names LBL goes on with:
    (object_science, object_comparison)."""
    _ = kwargs
    names = correct(instrument, data_dir, [object_science],
                    {object_science: object_comparison}, data_source, pca2d,
                    input_file, science_subdir)
    return names[object_science]


def joint(instrument: str, data_dir: str, objects: list, comparisons: dict,
          data_source: str = None, pca2d=None, input_file: str = None,
          science_subdir: str = 'science', **kwargs) -> dict:
    """Correct several objects against one observer basis (pca2d.joint), each
    with its own star spectrum. {object: (object_science, object_comparison)}.
    """
    _ = kwargs
    return correct(instrument, data_dir, list(objects), dict(comparisons),
                   data_source, pca2d, input_file, science_subdir)


def correct(instrument, data_dir, objects, comparisons, data_source, pca2d,
            input_file, science_subdir) -> dict:
    """The correction of one object, or of several together. Returns
    {object: (corrected object, its comparison)}.

    The comparison follows the science object when it was the science object
    (the usual case, each object its own template); another star's template
    is left as it is, since that star was not corrected here.
    """
    from lbl.pca2d.logger import log

    switches, overrides = settings(pca2d)
    data_dir = os.path.abspath(data_dir)
    science_root = os.path.join(data_dir, science_subdir)
    label = re.sub(r'[^0-9A-Za-z._+-]', '_', str(switches['name'] or ''))
    suffix = SUFFIX + ('_' + label if label else '')
    pca2d_root = os.path.join(data_dir, 'pca2d')
    work_root = os.path.join(pca2d_root, '_' + label) if label else pca2d_root
    together = len(objects) > 1
    if together:
        from lbl.pca2d.joint import joint_name
        work = os.path.join(work_root, 'joint', joint_name(objects))
    else:
        work = os.path.join(work_root, objects[0])
    names = dict()
    for name in objects:
        if not os.path.isdir(os.path.join(science_root, name)):
            raise LblException('PCA2D: no folder {0}'.format(
                os.path.join(science_root, name)))
        comparison = comparisons.get(name)
        if comparison in (None, '', name):
            comparison = name + suffix
        else:
            log('OBJECT_COMPARISON of %s is %s, another object: its template'
                ' is used as it is, uncorrected' % (name, comparison), 'warn')
        names[name] = (name + suffix, comparison)

    block = request_block(overrides, instrument, data_source, input_file)
    os.makedirs(work, exist_ok=True)
    request_file = os.path.join(work, 'request.yaml')
    write_request(request_file, objects, block,
                  os.path.join(pca2d_root, 'cache'),
                  os.path.join(data_dir, 'models'))

    files = {name: input_files(os.path.join(science_root, name), input_file)
             for name in objects}
    stamp = request_stamp(dict(block, objects=list(objects)), switches,
                          [f for name in objects for f in files[name]])
    state_file = os.path.join(work, 'done.json')
    state = dict()
    if os.path.exists(state_file):
        with open(state_file) as handle:
            state = json.load(handle)
    present = {name: glob.glob(os.path.join(science_root, names[name][0],
                                            '*.fits')) for name in objects}
    if (not switches['overwrite'] and state.get('stamp') == stamp
            and all(present.values())
            and {n: len(p) for n, p in present.items()} == state.get('n_files')):
        log('PCA2D: this correction (%s) is already in %s, kept as it is.'
            ' overwrite=True makes it again'
            % (stamp, ', '.join(names[n][0] for n in objects)), 'info')
        return names
    for name in objects:
        supersede(data_dir, os.path.join(science_root, names[name][0]),
                  names[name][0], work, log, science_subdir)

    stages = ['cube', 'fit'] + (['figures'] if switches['figures'] else [])
    stages.append('correct')
    argv = (['--objects', ','.join(objects)] if together
            else ['--object', objects[0]])
    argv += ['--config', request_file, '--data-dir', science_root,
             '--out-dir', work_root, '--no-fits-dir',
             '--stages', ','.join(stages),
             '--n-star', str(switches['n_star']),
             '--n-earth', str(switches['n_earth']),
             '--corrected-dir', os.path.join(science_root,
                                             '{object}' + suffix),
             '--keep-names', '--any-object']
    log('PCA2D on %s%s: %d star and %d observer components -> %s'
        % (', '.join(objects), ' together' if together else '',
           switches['n_star'], switches['n_earth'],
           ', '.join(names[n][0] for n in objects)), 'info')
    if overrides:
        log('PCA2D asks %s' % ', '.join(
            '%s.%s = %r' % (s, k, v) for s, kv in sorted(overrides.items())
            for k, v in sorted(kv.items())), 'value')
    from lbl.pca2d import cli
    saved = sys.argv
    sys.argv = [__NAME__] + argv
    try:
        cli.main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise LblException('PCA2D on {0} stopped (exit {1}); the run'
                               ' folder is {2}'.format(', '.join(objects),
                                                       exc.code, work))
    finally:
        sys.argv = saved
    written = {name: sorted(glob.glob(os.path.join(science_root,
                                                   names[name][0], '*.fits')))
               for name in objects}
    empty = [names[n][0] for n in objects if not written[n]]
    if empty:
        raise LblException('PCA2D wrote no corrected spectrum into '
                           '{0}'.format(', '.join(empty)))
    from lbl.pca2d.provenance import stamp as code_stamp
    with open(state_file, 'w') as handle:
        json.dump(dict(stamp=stamp, objects=[names[n][0] for n in objects],
                       n_files={n: len(w) for n, w in written.items()},
                       code=code_stamp(), when=datetime.datetime.now(
                       ).isoformat(timespec='seconds'),
                       counts=[switches['n_star'], switches['n_earth']]),
                  handle, indent=1)
    for name in objects:
        log('PCA2D: %d corrected spectra of %d in %s; LBL goes on with %s'
            % (len(written[name]), len(files[name]),
               os.path.join(science_root, names[name][0]), names[name][0]),
            'value')
    return names

# =============================================================================
# End of code
# =============================================================================

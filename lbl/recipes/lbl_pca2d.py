#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_pca2d

The two-frame precleaning of pca2d (lbl.pca2d), run on one science object
before LBL measures it. lbl_wrap calls it when the run parameters carry a
PCA2D dict, and measures the velocities on what it writes:

    rparams['PCA2D'] = dict(n_star=1, n_earth=7)

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

    n_star       the star's spectra, the template included (1): 1 is the
                 template alone, not adjusted in depth; 2 adds one
                 star-frame component with a free amplitude, never divided
                 out; 0 is not a fit
    n_earth      observer-frame components fitted and divided out (7)
    compare      LBL measures the delivered object too, first, so that the
                 report sets the two series side by side (True)
    figures      the river plots and the other figures of the correction
                 (as REPORT, True unless the wrap turns the report off)
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

The wrap's SKIP_LBL_PCA2D overrides that choice:

    None     (or not given) the above: made again only if the settings, the
             input files or nominal.yaml changed
    True     the correction on disk is kept even if they changed, which a
             warning says
    False    made again whatever is on disk, as overwrite=True

A correction with no done.json beside it was interrupted, and is made again
in every case.

Created on 2026-10-19
"""
import copy
import datetime
import glob
import hashlib
import json
from typing import Optional
import os
import re
import shutil
import sys

import numpy as np
from astropy import constants
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
DEFAULTS = dict(n_star=1, n_earth=7, figures=True, overwrite=False,
                name=None, joint=False, compare=True, output_name=None,
                input_name=None)
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
    'star_column_frac': 'correct.star_column_frac',
    'star_column_chi2': 'correct.star_column_chi2',
    'star_bias_nsig': 'correct.star_bias_nsig',
    'weight': 'correct.weight',
    'mask': 'correct.mask',
    'mask_share': 'correct.mask_share',
    'width_kms': 'highpass.width_kms',
    'high_pass': 'highpass.width_kms',
    'dv': 'domain.dv',
    'nightly_stack': 'input.nightly_stack',
    'bundle_minutes': 'input.bundle_minutes',
    'bundle_berv': 'input.bundle_berv',
    'bundle_max': 'input.bundle_max',
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
#: {corrected object: its pca2d run folder}, for the report after LBL
RUNS = dict()


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


def compare(pca2d) -> bool:
    """Whether LBL measures the delivered object as well (PCA2D compare)."""
    return bool(pca2d.get('compare', True)) if isinstance(pca2d, dict) \
        else bool(pca2d)


def settings(pca2d, report=True) -> tuple:
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
    switches = dict(DEFAULTS, figures=bool(report))
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
    # n_star counts the star's spectra with the template; the fit's own
    #   twoframe.n_star counts the free components, and is not set directly
    from lbl.pca2d.config import free_star_components
    try:
        free_star_components(switches['n_star'])
    except ValueError as exc:
        raise LblException('PCA2D {0}'.format(exc))
    if 'n_star' in overrides.get('twoframe', dict()):
        raise LblException('PCA2D: give n_star, which counts the star\'s'
                           ' spectra with the template (1: the template'
                           ' alone), not twoframe.n_star')
    return switches, overrides


def star_template(data_dir: str, name: str, comparison: str,
                  instrument: str) -> dict:
    """
    Another star's template as this star's spectrum in PCA2D: the template
    LBL uses for OBJECT_COMPARISON (templates/, made by LBL or put there, an
    APERO one under LBL's name included), and the velocity that moves it
    onto this star: the star's velocity from an LBL pass on its delivered
    spectra against that template (lblrdb/lbl_<name>_<comparison>.rdb, the
    median; PCA2D compare=True makes it first) minus the template's own, the
    SYSTVELO of its LBL mask, composed relativistically.

    :return: dict(file, velocity, v_object, v_template), velocities in m/s
    """
    from lbl.recipes.lbl_wrap import _template_of
    from lbl.report import series

    template = _template_of(data_dir, comparison, instrument)
    if template is None:
        raise LblException(
            'PCA2D: OBJECT_COMPARISON of {0} is {1}, and there is no template'
            ' of {1} in {2}: make it, or put one there under LBL\'s name'
            .format(name, comparison, os.path.join(data_dir, 'templates')))
    masks = sorted(glob.glob(os.path.join(
        data_dir, 'masks', 'LBL_Mask_{0}_full_*.fits'.format(comparison))))
    masks = [m for m in masks if str(instrument).lower() in
             os.path.basename(m).lower()] or masks
    rdb = os.path.join(data_dir, 'lblrdb',
                       'lbl_{0}_{1}.rdb'.format(name, comparison))
    if not masks or not os.path.exists(rdb):
        raise LblException(
            'PCA2D: {0} against the template of {1} needs an LBL pass on the'
            ' delivered spectra first, for the velocities ({2}{3}): PCA2D'
            ' compare=True makes it'.format(
                name, comparison, rdb if not os.path.exists(rdb) else '',
                '' if masks else ' and the mask of {0}'.format(comparison)))
    from astropy.io import fits as _fits
    v_template = float(_fits.getheader(masks[0])['SYSTVELO'])
    v_object = float(np.nanmedian(series.read_rdb(rdb)['vrad']))
    c_ms = constants.c.value
    velocity = (v_object - v_template) / (1 - v_object * v_template
                                          / c_ms ** 2)
    return dict(file=template, velocity=float(velocity),
                v_object=v_object, v_template=v_template)


def input_files(science_dir: str, pattern: str) -> list:
    """The object's spectra, as LBL would glob them."""
    return sorted(glob.glob(os.path.join(science_dir, pattern or '*.fits')))


#: an INPUT_FILE that says nothing of the files: PCA2D would then take its
#:   own pattern for the instrument (nominal.yaml)
GENERIC_INPUT = (None, '', '*', '*.fits')


def files_pattern(paths: list) -> Optional[str]:
    """The pattern of spectra a wrap gives none for: '*' and the end their
    names share, its leading digits off (a date's or a number's). HARPS read
    as ORIG has e2ds files (*_e2ds_A.fits) where PCA2D's HARPS default is
    the ESO archive's *_S2D_A.fits: without it, no spectrum was found.
    None when the names share no more than .fits."""
    names = [os.path.basename(path) for path in paths]
    if not names:
        return None
    tail = os.path.commonprefix([name[::-1] for name in names])[::-1]
    tail = tail.lstrip('0123456789')
    if tail.lower() in ('', 'fits', '.fits', 's', 'ts', 'its'):
        return None
    return '*' + tail


def nominal_digest(instrument: Optional[str] = None) -> str:
    """The nominal configuration as a correction depends on it: PARSED (its
    comments are not in it), the general block and the instrument's own
    (all of them when none is named), WITHOUT their output sections (the
    windows and figures, which change nothing in the spectra). Until
    2026-10-23 the stamp hashed the file's bytes: a comment, a window or
    another instrument's setting made every correction of every instrument
    again."""
    nominal = read_nominal()
    general = dict(nominal.get('general') or {})
    general.pop('output', None)
    blocks = dict(general=general)
    wanted = {str(instrument).upper(), str(instrument).upper().split('_')[0]}
    for name, block in (nominal.get('instruments') or {}).items():
        block = dict(block or {})
        lbl_name = str((block.get('lbl') or {}).get('instrument') or name)
        if instrument and name.upper() not in wanted \
                and lbl_name.upper() not in wanted:
            continue
        block.pop('output', None)
        blocks[name] = block
    text = json.dumps(blocks, sort_keys=True, default=str).encode()
    return hashlib.sha1(text).hexdigest()


def request_stamp(request: dict, switches: dict, files: list,
                  instrument: Optional[str] = None) -> str:
    """What makes a correction this one: the settings, the input files (name,
    size, date) and the nominal configuration they sit on (nominal_digest:
    the instrument's, parsed). Not the figures, which change nothing in the
    spectra."""
    # the FREE star components: n_star counts the template since
    #   2026-10-22, and a correction made before then keeps its stamp when it
    #   is the same fit
    from lbl.pca2d.config import free_star_components
    blob = dict(request=request,
                counts=[free_star_components(switches['n_star']),
                        switches['n_earth']],
                files=[(os.path.basename(f), os.path.getsize(f),
                        int(os.path.getmtime(f))) for f in files])
    blob['nominal'] = nominal_digest(instrument)
    text = json.dumps(blob, sort_keys=True, default=str).encode()
    return hashlib.sha1(text).hexdigest()[:12]


#: the folders of DATA_DIR that LBL writes its products into, by object name
PRODUCT_FOLDERS = ('lblrv', 'lblrdb', 'lblreftable', 'templates', 'masks',
                   'plots', 'reports')


def lbl_products(data_dir: str, name: str, science_subdir: str = 'science',
                 folders=PRODUCT_FOLDERS) -> list:
    """Everything LBL made from the object `name`: a file or folder of one of
    LBL's product folders whose name holds `name` between separators.

    Only LBL's folders, never another one a person keeps in DATA_DIR: on
    2026-10-21 a comparison/ folder holding copies named after the object was
    swept up with the products. And not what LBL made from a longer name that
    starts with this one: the labelled corrections of the same star,
    <object>_pca2d_<label>, are other objects, and are taken out of each name
    before it is matched."""
    token = re.compile(r'(^|[_.\-]){0}([_.\-]|$)'.format(re.escape(name)))
    science = os.path.join(data_dir, science_subdir)
    longer = sorted((d for d in (os.listdir(science)
                                 if os.path.isdir(science) else [])
                     if d.startswith(name + '_')), key=len, reverse=True)
    found = []
    for sub in folders:
        folder = os.path.join(data_dir, sub)
        if not os.path.isdir(folder):
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


def pdf_binder() -> bool:
    """Whether pypdf is there to bind PDFs. LBL lists it as a dependency since
    pca2d joined it, but an environment installed before that has none, and
    nothing of a run may stop for want of a binder."""
    try:
        import pypdf                                          # noqa: F401
    except ImportError:
        return False
    return True


def run_folder(work: str):
    """The run's own folder, <work>/<M>-<N>: the one whose configuration was
    written last."""
    made = glob.glob(os.path.join(work, '*', 'resolved_config.yaml'))
    return os.path.dirname(max(made, key=os.path.getmtime)) if made else None


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


#: what an orbit carries into pca2d's configuration
ORBIT_KEYS = ('name', 'period', 'k', 'e', 'omega', 'tp', 't0', 'reference')


def object_orbits(name: str, files: list, data_dir: str, log) -> list:
    """The Keplerian orbits of an object's companions, from LBL's planet
    catalogs (lbl.science.planets), for pca2d's star frame
    (registration.orbits): looked up by the object's name and the OBJECT
    and DRSOBJN of its first file. [] when there is none."""
    from astropy.io import fits as _fits
    from lbl.science import planets
    header_names = []
    if files:
        for ext in (0, 1):
            try:
                header = _fits.getheader(files[0], ext)
            except Exception:                                 # noqa: BLE001
                continue
            header_names += [header.get('OBJECT'), header.get('DRSOBJN')]
    found = planets.orbits_for(planets.star_names(name, header_names),
                               data_dir)
    if found is None:
        return []
    orbits = []
    for orbit in found['orbits']:
        orbits.append({key: (float(orbit[key]) if key not in ('name',
                                                              'reference')
                             else str(orbit[key]))
                       for key in ORBIT_KEYS if orbit.get(key) is not None})
    log('PCA2D: the star frame of %s follows its %d known orbit(s) (%s), '
        'not the BERV alone' % (name, len(orbits), found['catalog']), 'info')
    return orbits


def object_velocities(name: str, comparison, data_dir: str, instrument: str,
                      log) -> dict:
    """The velocity of each of an object's spectra from a FREE FIT of them:
    LBL's template of the object, when it iterated on the velocities (its
    SCI_TABLE's VSYS, m/s; not when it was made on known orbits, VKEP),
    relative to their median, in km/s, by file name: pca2d's star frame
    (registration.velocities) then follows the star from one exposure to
    the next, whatever moves it. Only the object's own template knows its
    spectra: with another star's (OBJECT_COMPARISON), {}. {} as well when
    there is no template yet, or it holds no velocities."""
    if comparison not in (None, '', name):
        return dict()
    from lbl.recipes.lbl_wrap import _template_of
    path = _template_of(data_dir, name, instrument)
    if path is None:
        return dict()
    try:
        from astropy.table import Table
        table = Table.read(path, 'SCI_TABLE')
    except Exception:                                         # noqa: BLE001
        return dict()
    if 'VSYS' not in table.colnames or 'VKEP' in table.colnames \
            or 'FILENAME' not in table.colnames:
        return dict()
    vsys = np.asarray(table['VSYS'], dtype=float)
    good = np.isfinite(vsys)
    if good.sum() < 3 or not np.nanstd(vsys[good]) > 0:
        return dict()
    median = float(np.median(vsys[good]))
    out = {os.path.basename(str(f)): float(v - median) / 1000.0
           for f, v, ok in zip(table['FILENAME'], vsys, good) if ok}
    log('PCA2D: the star frame of %s follows the velocities its template'
        ' fitted freely (%d spectra, %.1f m/s rms, %s), not the BERV alone'
        % (name, len(out), float(np.std(vsys[good])), os.path.basename(path)),
        'info')
    return out


def write_request(path: str, objects: list, block: dict, cache_dir: str,
                  model_dir: str, suffix: str = SUFFIX,
                  orbits: dict = None, velocities: dict = None) -> None:
    """The request as a pca2d configuration file: the nominal one, with one
    block per object holding the same settings, and each object's orbits
    (registration.orbits) when PLANETS found some."""
    nominal = read_nominal()
    output = nominal.setdefault('general', dict()).setdefault('output', dict())
    output['cache_directory'] = cache_dir
    output['model_directory'] = model_dir
    # pca2d's own LBL stage is never run from here: LBL is the caller
    # and the corrected object's name, which is how the report finds its
    # velocities in LBL's tree (lbl.pca2d.lbl.object_names)
    nominal['general'].setdefault('lbl', dict()).update(prepare=False,
                                                        run=False,
                                                        suffix=suffix)
    nominal['objects'] = {name: copy.deepcopy(block) for name in objects}
    for name, found in (velocities or dict()).items():
        if found:
            nominal['objects'][name].setdefault('registration', dict())[
                'velocities'] = found
    for name, found in (orbits or dict()).items():
        if found:
            nominal['objects'][name].setdefault('registration', dict())[
                'orbits'] = found
            # their periods, which pca2d marks on its diagnostics
            #   (target.planets)
            target = nominal['objects'][name].setdefault('target', dict())
            target['planets'] = sorted(set(
                list(target.get('planets') or [])
                + [float(orbit['period']) for orbit in found]))
    with open(path, 'w') as handle:
        handle.write('# what lbl_pca2d asked of pca2d for %s: the nominal'
                     ' configuration\n# of lbl/pca2d, with each object\'s'
                     ' block holding the PCA2D dict\n' % ', '.join(objects))
        yaml.safe_dump(nominal, handle, sort_keys=False,
                       default_flow_style=False)


def main(instrument: str, data_dir: str, object_science: str,
         object_comparison: str = None, data_source: str = None,
         pca2d=None, input_file: str = None, science_subdir: str = 'science',
         report: bool = True, skip=None, planets: bool = False,
         **kwargs) -> tuple:
    """Correct one object, and return the names LBL goes on with:
    (object_science, object_comparison). skip is the wrap's
    SKIP_LBL_PCA2D, planets its PLANETS."""
    _ = kwargs
    names = correct(instrument, data_dir, [object_science],
                    {object_science: object_comparison}, data_source, pca2d,
                    input_file, science_subdir, report, skip, planets)
    return names[object_science]


def joint(instrument: str, data_dir: str, objects: list, comparisons: dict,
          data_source: str = None, pca2d=None, input_file: str = None,
          science_subdir: str = 'science', report: bool = True, skip=None,
          planets: bool = False, **kwargs) -> dict:
    """Correct several objects against one observer basis (pca2d.joint), each
    with its own star spectrum. {object: (object_science, object_comparison)}.
    skip is the wrap's SKIP_LBL_PCA2D.
    """
    _ = kwargs
    return correct(instrument, data_dir, list(objects), dict(comparisons),
                   data_source, pca2d, input_file, science_subdir, report,
                   skip, planets)


def correct(instrument, data_dir, objects, comparisons, data_source, pca2d,
            input_file, science_subdir, report=True, skip=None,
            planets=False) -> dict:
    """The correction of one object, or of several together. Returns
    {object: (corrected object, its comparison)}.

    skip is the wrap's SKIP_LBL_PCA2D: None makes the correction again only
    if the request changed, True keeps the one on disk even then, False makes
    it again in every case (as overwrite=True). planets is the wrap's
    PLANETS: each object's star frame then follows its known orbits.

    The comparison follows the science object when it was the science object
    (the usual case, each object its own template); another star's template
    is left as it is, since that star was not corrected here.
    """
    from lbl.pca2d.logger import log

    switches, overrides = settings(pca2d, report)
    if switches['figures'] and not pdf_binder():
        log('pypdf is not installed (pip install pypdf): the figures and the'
            ' report are made, but pca2d\'s report and LBL\'s stay two files'
            ' and the figures are not bound into one PDF', 'warn')
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
    star_templates = []
    for name in objects:
        if not os.path.isdir(os.path.join(science_root, name)):
            raise LblException('PCA2D: no folder {0}'.format(
                os.path.join(science_root, name)))
        comparison = comparisons.get(name)
        if comparison in (None, '', name):
            comparison = name + suffix
            star_templates.append(None)
        else:
            # ANOTHER STAR'S TEMPLATE: LBL measures the corrected spectra
            #   against it as it is, and PCA2D takes it as the star's
            #   spectrum, at this star's velocity
            spec = star_template(data_dir, name, comparison, instrument)
            star_templates.append(spec)
            log('OBJECT_COMPARISON of %s is %s, another star: its template'
                ' %s is the star\'s spectrum in PCA2D, moved by %+.0f m/s'
                ' (%s at %+.0f m/s in LBL, the template at %+.0f m/s, its'
                ' mask\'s SYSTVELO), and LBL measures the corrected spectra'
                ' against it' % (name, comparison,
                                 os.path.basename(spec['file']),
                                 spec['velocity'], name, spec['v_object'],
                                 spec['v_template']), 'info')
        names[name] = (name + suffix, comparison)
    if any(star_templates):
        overrides.setdefault('twoframe', dict())['star_templates'] = [
            None if spec is None else dict(file=spec['file'],
                                           velocity=spec['velocity'])
            for spec in star_templates]

    if skip not in (None, True, False):
        raise LblException('SKIP_LBL_PCA2D is True, False or None (not given),'
                           ' not {0!r}'.format(skip))
    if skip is True and switches['overwrite']:
        raise LblException('SKIP_LBL_PCA2D=True keeps the correction on disk'
                           ' and PCA2D overwrite=True makes it again: one or'
                           ' the other')
    # THE OBSERVER COMPONENTS OF ANOTHER FIT (input_name, pca2d.basis): taken
    #   frozen, the star's own block and the amplitudes fitted. All of them
    #   unless n_earth is given
    frozen = None
    if switches['input_name']:
        from lbl.pca2d import basis as pbasis
        try:
            frozen = pbasis.load(str(switches['input_name']), data_dir,
                                 instrument, str(data_source or ''))
        except FileNotFoundError as exc:
            raise LblException('PCA2D input_name: {0}'.format(exc))
        # the same instrument and the same DATA_SOURCE, or nothing
        try:
            frozen.check_compatible(instrument, data_source,
                                    others=pbasis.compatible_bases(
                                        data_dir, instrument, data_source))
        except ValueError as exc:
            raise LblException('PCA2D input_name: {0}'.format(exc))
        overrides.setdefault('twoframe', dict())['earth_basis'] = frozen.path
        if not (isinstance(pca2d, dict) and 'n_earth' in pca2d):
            switches['n_earth'] = frozen.n_earth
        log('PCA2D: the observer basis is %s, frozen' % frozen.describe(),
            'info')
    # NO INPUT_FILE IN THE WRAP: the files LBL read (all of the folder's)
    #   tell PCA2D which they are, rather than its default for the
    #   instrument, which is one data source's
    if input_file in GENERIC_INPUT:
        found = [f for name in objects for f in input_files(
            os.path.join(science_root, name), input_file)]
        guessed = files_pattern(found)
        if guessed:
            log('PCA2D: no INPUT_FILE given; the spectra of %s are read as'
                ' %s, the end their names share' % (', '.join(objects),
                                                    guessed), 'info')
            input_file = guessed
    block = request_block(overrides, instrument, data_source, input_file)
    os.makedirs(work, exist_ok=True)

    files = {name: input_files(os.path.join(science_root, name), input_file)
             for name in objects}
    orbits = {name: object_orbits(name, files[name], data_dir, log)
              for name in objects} if planets else dict()
    # the velocities LBL's template fitted freely, when it did (its own
    #   template, made without known orbits): the star frame follows them
    velocities = {name: object_velocities(name, comparisons.get(name),
                                          data_dir, instrument, log)
                  for name in objects}
    request = dict(block, objects=list(objects))
    # the orbits are part of the request only when there are some, so that a
    #   correction made before PLANETS existed keeps its stamp
    if any(orbits.values()):
        request['orbits'] = orbits
    if any(velocities.values()):
        request['velocities'] = velocities
    # the basis by its content, and how it is divided out (in full, since
    #   2026-10-22: shrunk by this star's own significance before, which
    #   dropped it): another basis, or a correction made before, is made again
    if frozen is not None:
        request['earth_basis'] = dict(
            id=frozen.ident, shrink=False,
            observer_first=bool(overrides.get('twoframe', dict()).get(
                'observer_first', True)))
    stamp = request_stamp(request, switches,
                          [f for name in objects for f in files[name]],
                          instrument)
    state_file = os.path.join(work, 'done.json')
    state = dict()
    if os.path.exists(state_file):
        with open(state_file) as handle:
            state = json.load(handle)
    present = {name: glob.glob(os.path.join(science_root, names[name][0],
                                            '*.fits')) for name in objects}
    # a correction that finished: its done.json counts its files, and they
    #   are all there (one that was interrupted has no done.json)
    on_disk = (all(present.values()) and {n: len(p) for n, p in
                                          present.items()} == state.get(
        'n_files'))
    same = state.get('stamp') == stamp
    corrected = ', '.join(names[n][0] for n in objects)
    if on_disk and not switches['overwrite'] and skip is not False and (
            same or skip is True):
        if same:
            log('PCA2D: this correction (%s) is already in %s, kept as it'
                ' is. SKIP_LBL_PCA2D=False makes it again'
                % (stamp, corrected), 'info')
        else:
            log('PCA2D: SKIP_LBL_PCA2D=True, so the correction in %s (%s,'
                ' made %s) is kept, although the settings, the input files'
                ' or nominal.yaml changed since (%s now). Without'
                ' SKIP_LBL_PCA2D, or with False, it is made again'
                % (corrected, state.get('stamp'), state.get('when', '?'),
                   stamp), 'warn')
        for name in objects:
            RUNS[names[name][0]] = state.get('run_dir') or run_folder(work)
        return names
    if skip is True and not on_disk:
        log('PCA2D: SKIP_LBL_PCA2D=True, but no finished correction in %s:'
            ' it is made' % corrected, 'info')
    request_file = os.path.join(work, 'request.yaml')
    write_request(request_file, objects, block,
                  os.path.join(pca2d_root, 'cache'),
                  os.path.join(data_dir, 'models'), suffix, orbits,
                  velocities)
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
    from lbl.pca2d.config import describe_star, free_star_components
    log('PCA2D on %s%s -> %s' % (', '.join(objects),
                                 ' together' if together else '',
                                 ', '.join(names[n][0] for n in objects)),
        'info')
    log('PCA2D star: %s' % describe_star(
        free_star_components(switches['n_star'])), 'info')
    log('PCA2D observer: %d components, divided out of the spectra%s'
        % (switches['n_earth'], ', FROZEN from %s' % frozen.describe()
           if frozen is not None else ', fitted'), 'info')
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
    run_dir = run_folder(work)
    # THE COMPONENTS OF THIS FIT, in DATA_DIR/pca2d/components (pca2d.basis):
    #   the observer basis, named by the PCA2D output_name or the object(s),
    #   for other stars to read frozen with input_name (unless it was itself
    #   read so; one of the name of a kept basis, in models/pca2d_<INSTRUMENT>
    #   _<DATA_SOURCE>, updates it there), and each star's own part
    from lbl.pca2d import basis as pbasis
    from lbl.pca2d.config import read_yaml
    resolved = os.path.join(run_dir, 'resolved_config.yaml')
    config = None
    if os.path.exists(resolved):
        with open(resolved) as handle:
            config = read_yaml(handle)
    output = re.sub(r'[^0-9A-Za-z._+-]', '_',
                    str(switches['output_name'] or ''))
    if frozen is None:
        observer_label = output or '+'.join(objects) + (
            '_' + label if label else '')
        saved = pbasis.save(run_dir, observer_label, data_dir, list(objects),
                            instrument=instrument, config=config,
                            data_source=str(data_source or ''))
        log('PCA2D: observer components saved, %s; another star reads them'
            ' with PCA2D input_name=%r' % (saved, observer_label), 'value')
    for saved in pbasis.save_stars(run_dir, data_dir, list(objects),
                                   instrument=instrument,
                                   name=output or label or None):
        log('PCA2D: star components saved, %s' % saved, 'value')
    for name in objects:
        RUNS[names[name][0]] = run_dir
    with open(state_file, 'w') as handle:
        json.dump(dict(stamp=stamp, objects=[names[n][0] for n in objects],
                       run_dir=run_dir,
                       n_files={n: len(w) for n, w in written.items()},
                       code=code_stamp(), when=datetime.datetime.now(
                       ).isoformat(timespec='seconds'),
                       n_star=switches['n_star'],
                       counts=[free_star_components(switches['n_star']),
                               switches['n_earth']]),
                  handle, indent=1)
    for name in objects:
        log('PCA2D: %d corrected spectra of %d in %s; LBL goes on with %s'
            % (len(written[name]), len(files[name]),
               os.path.join(science_root, names[name][0]), names[name][0]),
            'value')
    return names

def report(data_dir: str, delivered: str, corrected: str,
           lbl_report: str = None) -> str:
    """The report of a PCA2D run, after LBL: one PDF. Returns its path.

    pca2d's own report first, written again now that LBL has measured both
    objects: the star, the delivered and corrected velocities side by side
    and what changed, the river plots of the correction step by step, the
    components, their coefficients in time and their periodograms, the
    telluric and airglow residuals. Then LBL's own report of the corrected
    object. The whole goes where LBL put its report, so that the report of
    the run is one file. Never fatal: a report is an account of the run, and
    the velocities are already in lblrdb.
    """
    from lbl.pca2d.logger import log
    try:
        run_dir = RUNS.get(corrected)
        if not run_dir or not os.path.isdir(run_dir):
            log('no pca2d run folder for %s, so the report is LBL\'s alone'
                % corrected, 'warn')
            return lbl_report
        from lbl.pca2d import texreport
        # THE NAMES LBL USED, whatever the run's own configuration says: a
        # run made before the request carried lbl.suffix names the corrected
        # object <star>_PCA2D_<tag>, and its velocities would not be found
        with open(os.path.join(run_dir, 'resolved_config.yaml')) as handle:
            config = yaml.safe_load(handle)
        if corrected.startswith(delivered):
            config.setdefault('lbl', dict())['suffix'] = \
                corrected[len(delivered):]
        pdf = texreport.render(run_dir, config=config, lbl_dir=data_dir)
        if pdf is None:
            # no pdflatex: the figures bound in one PDF are the report
            found = [p for p in glob.glob(os.path.join(run_dir, '*.pdf'))
                     if not os.path.basename(p).startswith('components')]
            pdf = max(found, key=os.path.getmtime) if found else None
        if pdf is None:
            log('no pca2d report in %s' % run_dir, 'warn')
            return lbl_report
        if not lbl_report or not os.path.exists(lbl_report):
            log('the report of %s: %s' % (corrected, pdf), 'value')
            return pdf
        if not pdf_binder():
            log('pypdf is not installed (pip install pypdf), so the report is'
                ' two files: pca2d\'s %s and LBL\'s %s' % (pdf, lbl_report),
                'warn')
            return lbl_report
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.append(pdf)
        writer.append(lbl_report)
        whole = lbl_report + '.part'
        with open(whole, 'wb') as handle:
            writer.write(handle)
        os.replace(whole, lbl_report)
        log('the report of %s, pca2d\'s (the velocities of %s beside them,'
            ' the river plots, the components) and then LBL\'s, in one file:'
            ' %s' % (corrected, delivered, lbl_report), 'value')
        return lbl_report
    except Exception as exc:                                  # noqa: BLE001
        log('the pca2d pages could not be added to the report (%s: %s)'
            % (type(exc).__name__, exc), 'warn')
        return lbl_report


# =============================================================================
# End of code
# =============================================================================

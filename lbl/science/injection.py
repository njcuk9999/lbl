#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An injection test: Keplerian signals added to the BERV of a star's spectra,
to see what of them LBL measures, and what PCA2D's correction leaves.

FAKE_KEPLERIAN, given to lbl_wrap, holds the signals, one entry per signal in
each list (a single signal can be given as plain numbers):

    rparams['FAKE_KEPLERIAN'] = dict(k=[5.0, 3.0], period=[12.3, 40.0],
                                     phase=[0.0, 0.25], e=[0.0, 0.2],
                                     omega=[90.0, 30.0])

    k       semi-amplitude of the star [m/s]
    period  [days]
    phase   the orbit's phase at BJD 2460000.0 (EPOCH), counted from the
            inferior conjunction, where the star's velocity goes from
            receding to approaching through zero (the transit, for a
            transiting planet): t0 = EPOCH - phase * period. 0 by default
    e       eccentricity, 0 by default
    omega   argument of periastron of the star [deg], 90 by default

t0 (the conjunction) or tp (the periastron), in BJD, can be given instead of
the phase. known=True (for every signal, or a list of one per signal) adds
the signals to the star's known planets as LBL's PLANETS reads them
(lbl.science.planets.orbits_for), so that with PLANETS the template is made
on BERV + the known orbits + the injected signals, as it would be for a
planet whose orbit is known; the catalogs' files are not edited.

NOTHING IS WRITTEN INTO THE SPECTRA. The wrap makes science/<object>_inj, a
folder of links to the object's spectra holding fake_keplerian.yaml (RECORD),
and goes on with <object>_inj as the object, so that nothing of the object's
own products (templates, velocities, PCA2D correction) is touched. Every LBL
recipe and PCA2D take a spectrum's BERV from its instrument's get_berv
(lbl.instruments.default.Instrument), which adds to it, relativistically,
the signals' velocity at the spectrum's time (get_rjd_value, BJD) when the
spectrum's folder, or the folder it was derived from (<object>_inj_tc,
<object>_inj_pca2d), holds a record. The frame LBL and PCA2D take for the
star's is then off the star's by the signals: its lines move in it by them,
a purely achromatic velocity that LBL measures as one, added to the star's
(a larger BERV is a larger velocity). The telluric lines stay where they
are in the spectra.

Created on 2026-11-17

@author: artigau
"""
import datetime
import functools
import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
import yaml

from lbl.core import base_classes
from lbl.core import math as mp
from lbl.science import planets

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.science.injection.py'
#: the record of an injection, in the injected object's science folder
RECORD = 'fake_keplerian.yaml'
#: what the injected object's name adds to the star's
MARK = '_inj'
#: the date the phase is counted at [BJD] (2023-02-24)
EPOCH = 2460000.0
#: the keys of a signal, and what each one is
KEYS = dict(k='semi-amplitude of the star [m/s]',
            period='period [d]',
            phase='phase at BJD {0:.1f}, from the inferior conjunction'.format(
                EPOCH),
            e='eccentricity',
            omega='argument of periastron of the star [deg]',
            t0='time of inferior conjunction [BJD], instead of the phase',
            tp='time of periastron [BJD], instead of the phase',
            known='True: one of the star\'s known planets for PLANETS')
#: the names LBL derives from an object's (telluric cleaning, PCA2D): the
#:   folder an injected object's derived spectra come from is what is left
#:   once they are taken off
DERIVED = re.compile(r'(_tc|_pca2d[0-9A-Za-z._+-]*)+$')
#: the records read, by path: (modification time, signals)
_READ = dict()


# =============================================================================
# Define functions
# =============================================================================
class InjectionError(base_classes.LblException):
    """An injection test that cannot be done as asked: never taken for a
    spectrum without one (the readers that fall back on the header's BERV
    when get_berv fails let it through)"""


def _error(message: str):
    return InjectionError('FAKE_KEPLERIAN: ' + message)


def _number(value, key: str, index: int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise _error('{0} of signal {1} is {2!r}, not a number'.format(
            key, index + 1, value))
    if not np.isfinite(out):
        raise _error('{0} of signal {1} is {2}'.format(key, index + 1, out))
    return out


def _flag(value, key: str, index: int) -> bool:
    if isinstance(value, (bool, np.bool_)) or value in (0, 1):
        return bool(value)
    if str(value).strip().lower() in ('true', 'yes', 'on', '1'):
        return True
    if str(value).strip().lower() in ('false', 'no', 'off', '0', '', 'none'):
        return False
    raise _error('{0} of signal {1} is {2!r}, not True or False'.format(
        key, index + 1, value))


def signals(fake) -> List[Dict]:
    """
    The signals of a FAKE_KEPLERIAN, checked and complete: a list of dict
    (k, period, e, omega, and phase, t0 or tp). [] for none.

    :param fake: dict of lists (one entry per signal), dict of numbers (one
                 signal), list of dicts, or None

    :return: list of dict
    """
    if fake in (None, False, '', 'None', 'Null') or (
            isinstance(fake, (dict, list, tuple)) and not fake):
        return []
    if isinstance(fake, dict):
        lowered = {str(key).lower(): value for key, value in fake.items()}
        lengths = {key: len(value) for key, value in lowered.items()
                   if isinstance(value, (list, tuple))}
        if len(set(lengths.values())) > 1:
            raise _error('its lists have different lengths ({0}): one entry '
                         'per signal in each'.format(', '.join(
                            '{0}: {1}'.format(k, n)
                            for k, n in sorted(lengths.items()))))
        count = max(lengths.values()) if lengths else 1
        rows = [{key: (value[it] if isinstance(value, (list, tuple))
                       else value) for key, value in lowered.items()}
                for it in range(count)]
    elif isinstance(fake, (list, tuple)) and all(isinstance(row, dict)
                                                 for row in fake):
        rows = [{str(key).lower(): value for key, value in row.items()}
                for row in fake]
    else:
        raise _error('a dictionary (one list per key, one entry per signal) '
                     'or a list of dictionaries, not {0!r}'.format(fake))
    out = []
    for index, row in enumerate(rows):
        unknown = sorted(set(row) - set(KEYS))
        if unknown:
            raise _error('unknown key(s) {0} (known: {1})'.format(
                ', '.join(unknown), ', '.join(KEYS)))
        for key in ('k', 'period'):
            if row.get(key) is None:
                raise _error('signal {0} has no {1}'.format(index + 1, key))
        signal = dict(k=_number(row['k'], 'k', index),
                      period=_number(row['period'], 'period', index),
                      e=_number(row.get('e') or 0.0, 'e', index),
                      omega=_number(90.0 if row.get('omega') is None
                                    else row['omega'], 'omega', index))
        if signal['period'] <= 0:
            raise _error('the period of signal {0} is {1} d'.format(
                index + 1, signal['period']))
        if not 0 <= signal['e'] < 1:
            raise _error('the eccentricity of signal {0} is {1}, not in '
                         '[0, 1)'.format(index + 1, signal['e']))
        epochs = [key for key in ('phase', 't0', 'tp')
                  if row.get(key) is not None]
        if len(epochs) > 1:
            raise _error('signal {0} gives {1}: one of phase, t0 and tp'
                         .format(index + 1, ' and '.join(epochs)))
        epoch = epochs[0] if epochs else 'phase'
        signal[epoch] = _number(row.get(epoch) or 0.0, epoch, index)
        # one of the star's known planets: said only when it is
        if row.get('known') is not None and _flag(row['known'], 'known',
                                                  index):
            signal['known'] = True
        out.append(signal)
    return out


def orbit(signal: Dict) -> Dict:
    """
    A signal as an orbit of lbl.science.planets (period, k, e, omega, and
    t0 or tp)

    :param signal: dict, one of signals()

    :return: dict
    """
    out = {key: signal[key] for key in ('period', 'k', 'e', 'omega')}
    if 'tp' in signal:
        out['tp'] = signal['tp']
    elif 't0' in signal:
        out['t0'] = signal['t0']
    else:
        out['t0'] = EPOCH - signal.get('phase', 0.0) * signal['period']
    return out


def conjunction(signal: Dict) -> float:
    """
    A signal's time of inferior conjunction [BJD], the one just before or
    at the periastron when only that is given

    :param signal: dict, one of signals()

    :return: float
    """
    found = orbit(signal)
    if 't0' in found:
        return float(found['t0'])
    ecc = float(found['e'])
    omega = np.radians(float(found['omega']))
    nu_c = np.pi / 2 - omega
    ecc_c = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu_c / 2),
                           np.sqrt(1 + ecc) * np.cos(nu_c / 2))
    mean_c = ecc_c - ecc * np.sin(ecc_c)
    return float(found['tp']) + found['period'] * mean_c / (2 * np.pi)


def velocity(bjd, found: List[Dict]) -> np.ndarray:
    """
    The velocity the signals give the star [m/s], positive receding

    :param bjd: float or np.ndarray, the times [BJD]
    :param found: list of dict, signals()

    :return: np.ndarray
    """
    return planets.star_velocity(bjd, [orbit(signal) for signal in found])


def describe(found: List[Dict]) -> str:
    """The signals in a line: 'K = 5 m/s, P = 12.3 d, phase 0, e = 0'"""
    parts = []
    for signal in found:
        when = ('phase {0:g}'.format(signal['phase']) if 'phase' in signal
                else 't0 = {0:.4f}'.format(signal['t0']) if 't0' in signal
                else 'tp = {0:.4f}'.format(signal['tp']))
        text = 'K = {0:g} m/s, P = {1:g} d, {2}'.format(
            signal['k'], signal['period'], when)
        if signal['e'] > 0:
            text += ', e = {0:g}, omega = {1:g} deg'.format(signal['e'],
                                                           signal['omega'])
        if signal.get('known'):
            text += ', a known planet'
        parts.append(text)
    return '; '.join(parts)


def mean_longitude(signal: Dict, bjd: float) -> float:
    """
    A signal's mean longitude M + omega at a time [deg, 0 to 360]: the
    lambda of lbl.pca2d.keplerfit, whose circular orbit is
    K cos(lambda + 2 pi (t - t_ref) / P)

    :param signal: dict, one of signals()
    :param bjd: float, the time [BJD]

    :return: float
    """
    found = orbit(signal)
    mean = 2 * np.pi * (bjd - planets.periastron_time(found)) / found['period']
    return float(np.degrees(mean + np.radians(found['omega'])) % 360.0)


def _spread(samples) -> tuple:
    """(median, 16th, 84th percentiles) of samples"""
    return tuple(float(q) for q in np.percentile(samples, [50, 16, 84]))


def recovery(post: Dict, found: List[Dict], first: int = 0) -> List[Dict]:
    """
    What the MCMC of a series found of each injected signal
    (lbl.pca2d.keplerfit.posteriors, the signals its planets first to
    first + len(found) - 1, at their injected periods): the circular fit for
    a circular signal, the Keplerian one otherwise. For each signal, dict of
    (median, 16th, 84th percentiles): k, dk (recovered less injected, m/s),
    dlam (recovered less injected mean longitude at the fit's reference
    time, deg, within 180), e (Keplerian); and sigma, dk in units of K's
    half 16-84 width.

    :param post: dict, keplerfit.posteriors of one series
    :param found: list of dict, signals()
    :param first: int, the index of the first signal among the fit's planets

    :return: list of dict
    """
    t_ref = float(post['fits']['circular'].t_ref) + 2400000.0
    out = []
    for offset, signal in enumerate(found):
        index = first + offset
        scenario = 'keplerian' if signal['e'] > 0 else 'circular'
        samples = post[scenario]['samples']
        k = np.asarray(samples['K_%d' % index], dtype=float)
        lam = np.asarray(samples['lam_%d' % index], dtype=float)
        dlam = (lam - mean_longitude(signal, t_ref) + 180.0) % 360.0 - 180.0
        # the difference's own circular centre, so that a phase found near
        #   +-180 does not split in two
        centre = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(dlam)))))
        dlam = (dlam - centre + 180.0) % 360.0 - 180.0 + centre
        row = dict(signal=signal, scenario=scenario, k=_spread(k),
                   dk=_spread(k - signal['k']), dlam=_spread(dlam))
        half = 0.5 * (row['k'][2] - row['k'][1])
        row['sigma'] = (row['dk'][0] / half) if half > 0 else float('nan')
        if scenario == 'keplerian':
            row['e'] = _spread(samples['e_%d' % index])
        out.append(row)
    return out


def ratio(after: Dict, before: Dict, index: int, scenario: str) -> tuple:
    """(median, 16th, 84th percentiles) of K corrected / K delivered of one
    planet, from the two MCMCs' samples (independent chains)"""
    k_after = np.asarray(after[scenario]['samples']['K_%d' % index], float)
    k_before = np.asarray(before[scenario]['samples']['K_%d' % index], float)
    size = min(k_after.size, k_before.size)
    good = k_before[:size] > 0
    return _spread(k_after[:size][good] / k_before[:size][good])


# -----------------------------------------------------------------------------
# The record, in the injected object's folder
# -----------------------------------------------------------------------------
def injected(name: str) -> bool:
    """Whether an object's name is an injection test's (<star>_inj, and
    what is derived from it)"""
    return DERIVED.sub('', str(name)).endswith(MARK)


def read(folder: str) -> Optional[List[Dict]]:
    """
    The signals recorded in a science folder, or in the folder it was
    derived from (<star>_inj for <star>_inj_pca2d); None without a record.
    Read again only when the file changed.

    :param folder: str, a science folder

    :return: list of dict or None
    """
    if not folder:
        return None
    folder = os.path.normpath(str(folder))
    name = os.path.basename(folder)
    candidates = [folder]
    base = DERIVED.sub('', name)
    if base != name:
        candidates.append(os.path.join(os.path.dirname(folder), base))
    for candidate in candidates:
        path = os.path.join(candidate, RECORD)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        kept = _READ.get(path)
        if kept is not None and kept[0] == mtime:
            return kept[1]
        with open(path, encoding='utf-8') as handle:
            content = yaml.safe_load(handle) or dict()
        found = signals(content.get('signals'))
        _READ[path] = (mtime, found)
        return found
    return None


def known_orbits(folder: str) -> List[Dict]:
    """
    The injected signals recorded for a science folder that are to be among
    the star's known planets (known=True), as orbits of the planet catalogs
    (lbl.science.planets): [] for none

    :param folder: str, a science folder

    :return: list of dict (name, period, k, e, omega, t0 or tp, reference)
    """
    found = read(folder) or []
    out = []
    for index, signal in enumerate(found):
        if not signal.get('known'):
            continue
        one = orbit(signal)
        one.update(name='injected {0}'.format(index + 1),
                   reference='injection test, {0}'.format(RECORD))
        out.append(one)
    return out


def spectrum_folder(inst) -> Optional[str]:
    """
    The folder of the spectra an instrument is reading: the one PCA2D's
    readers set for each file (inst.spectrum_folder), else LBL's science
    folder of OBJECT_SCIENCE (DATA_DIR/SCIENCE_SUBDIR/OBJECT_SCIENCE)

    :param inst: Instrument

    :return: str or None
    """
    folder = getattr(inst, 'spectrum_folder', None)
    if folder:
        return folder
    try:
        data_dir = inst.params['DATA_DIR']
        objname = inst.params['OBJECT_SCIENCE']
        subdir = inst.params['SCIENCE_SUBDIR'] or 'science'
    except Exception:                                         # noqa: BLE001
        return None
    if not data_dir or objname in (None, '', 'None', 'Null'):
        return None
    return os.path.join(str(data_dir), str(subdir), str(objname).strip())


def add_to_berv(inst, header, berv):
    """
    The BERV of a spectrum with the injected signals' velocity at its time
    added, relativistically [m/s]: the BERV as it is for a spectrum of no
    injection test, or of a calibration

    :param inst: Instrument
    :param header: LBLHeader, the spectrum's (or its lblrv file's)
    :param berv: float, its BERV [m/s]

    :return: float
    """
    if berv is None:
        return berv
    try:
        if inst.params['DATA_TYPE'] not in (None, 'SCIENCE'):
            return berv
    except Exception:                                         # noqa: BLE001
        pass
    folder = spectrum_folder(inst)
    found = read(folder)
    if not found:
        return berv
    try:
        bjd = float(inst.get_rjd_value(header)) + 2400000.0
    except Exception as exc:                                  # noqa: BLE001
        raise _error('a spectrum of {0} gives no time ({1}): its injected '
                     'velocity cannot be had'.format(folder, exc))
    if not np.isfinite(bjd):
        raise _error('a spectrum of {0} has no finite time: its injected '
                     'velocity cannot be had'.format(folder))
    return float(mp.add_velocities(float(berv),
                                   float(velocity(bjd, found)[0])))


def with_signals(get_berv):
    """
    An instrument's get_berv, with the injected signals added
    (add_to_berv). A get_berv that calls another class's adds them once.

    :param get_berv: the class's own get_berv

    :return: function
    """
    if getattr(get_berv, 'injection', False):
        return get_berv

    @functools.wraps(get_berv)
    def wrapped(self, sci_hdr, *args, **kwargs):
        depth = self.__dict__.get('_berv_depth', 0)
        self._berv_depth = depth + 1
        try:
            berv = get_berv(self, sci_hdr, *args, **kwargs)
        finally:
            self._berv_depth = depth
        if depth:
            return berv
        return add_to_berv(self, sci_hdr, berv)

    wrapped.injection = True
    return wrapped


def _link(source: str, target: str) -> str:
    """target made a link to source: symbolic and relative, else hard, else
    a copy (a disk without links: exFAT, FAT). Returns which."""
    try:
        os.symlink(os.path.relpath(source, os.path.dirname(target)), target)
        return 'link'
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(source, target)
        return 'hard link'
    except (OSError, NotImplementedError):
        pass
    shutil.copy2(source, target)
    return 'copy'


def _same_file(source: str, target: str) -> bool:
    """Whether target is still source's link, hard link or copy"""
    try:
        if os.path.islink(target):
            return os.path.samefile(source, target)
        one, two = os.stat(source), os.stat(target)
    except OSError:
        return False
    return (one.st_ino == two.st_ino and one.st_dev == two.st_dev) or (
        one.st_size == two.st_size and int(one.st_mtime) == int(two.st_mtime))


def prepare(science_root: str, star: str, found: List[Dict], log=None):
    """
    The injected object of a star: science/<star>_inj, one link per file of
    science/<star> (kept in step with it), and its record. Returns (name,
    previous): the injected object's name, and the signals it held before,
    None when it held none (the caller then puts aside what LBL made from
    it, made with other signals or none).

    :param science_root: str, DATA_DIR/science
    :param star: str, the object
    :param found: list of dict, signals()
    :param log: a function(message) that says what was done, or None

    :return: tuple (str, list of dict or None)
    """
    if injected(star):
        raise _error('{0} is already an injection test\'s object: give the '
                     'star itself ({1})'.format(
                        star, DERIVED.sub('', star)[:-len(MARK)]))
    source = os.path.join(science_root, star)
    if not os.path.isdir(source):
        raise _error('no folder {0}'.format(source))
    name = star + MARK
    target = os.path.join(science_root, name)
    os.makedirs(target, exist_ok=True)
    previous = read(target) if os.path.exists(
        os.path.join(target, RECORD)) else None
    files = sorted(entry for entry in os.listdir(source)
                   if not entry.startswith('.') and entry != RECORD
                   and os.path.isfile(os.path.join(source, entry)))
    wanted = set(files)
    removed = 0
    for entry in os.listdir(target):
        path = os.path.join(target, entry)
        if entry == RECORD or entry.startswith('.') or os.path.isdir(path) \
                and not os.path.islink(path):
            continue
        if entry not in wanted or not _same_file(os.path.join(source, entry),
                                                 path):
            os.remove(path)
            removed += 1
    made = dict()
    for entry in files:
        path = os.path.join(target, entry)
        if os.path.lexists(path):
            continue
        how = _link(os.path.join(source, entry), path)
        made[how] = made.get(how, 0) + 1
    with open(os.path.join(target, RECORD), 'w', encoding='utf-8') as handle:
        handle.write('# an injection test (lbl_wrap FAKE_KEPLERIAN): the '
                     'spectra of this folder are\n# those of {0}, and LBL and '
                     'PCA2D add these Keplerian signals to their BERV\n# '
                     '(lbl.science.injection). Phases are at BJD {1:.1f}, '
                     'from the inferior conjunction.\n'.format(star, EPOCH))
        yaml.safe_dump(dict(star=star, signals=found,
                            written=datetime.datetime.now().isoformat(
                                timespec='seconds')),
                       handle, sort_keys=False, default_flow_style=None)
    if log is not None:
        said = ', '.join('{0} {1}{2}'.format(n, how, 's' if n > 1 else '')
                         for how, n in sorted(made.items()))
        log('injection test: {0} is {1} with {2} added to the BERV ({3} '
            'spectra{4}{5})'.format(name, star, describe(found), len(files),
                                    ', ' + said + ' made' if said else '',
                                    ', {0} gone'.format(removed)
                                    if removed else ''))
    return name, previous

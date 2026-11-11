#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The APERO names of stars: the astrometric database of APERO (the pipeline
of SPIRou and NIRPS), one YAML file per object, keyed by its APERO name (a
name without the characters that are trouble in a file name: GL699,
PROXIMA, TOI_756, 2MASS_J00140580M3245599), with its SIMBAD name, its
aliases and its place (J2000, Gaia's proper motion)

    from lbl.gui.fetch import apero_names
    apero_names.lookup('Gl382')         # the entry of AN Sex
    apero_names.near(150.38, -3.04)     # the entries within 30 arcsec

The database is APERO's own: the astrometrics folder of its assets, which
APERO downloads from its assets server as one tarball. Its YAML files come
first in the tarball, so a copy of them streams the first few MB of it and
stops (no APERO, no profile, no 900 MB); refresh() makes a new copy, and
the first lookup makes one when there is none. LBL_APERO_ASTROMETRICS
points to a folder of YAML files instead (an export made by APERO itself),
read as it is.

A name is found the way APERO finds it (drs_astrometrics.find_by_name):
cleaned (+ to P, - to M, any other punctuation and spaces to _, upper
case), in every variant of its spaces, signs and underscores, against the
APERO name, the original name, the SIMBAD name and every alias of each
entry; the verified entries first, then the pending, then the rejected.

After koloa.apero_names (github.com/eartigau/koloa), which reads the same
database for the same reasons.

Created on 2026-11-11
"""
import json
import math
import os
import re
import string
import tarfile
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

from .paths import cache

# =============================================================================
# Define variables
# =============================================================================
#: where the copy of the database is kept
FOLDER = cache('apero_astrometrics')
#: a folder of YAML files to read instead (an export of APERO's)
ENV_FOLDER = 'LBL_APERO_ASTROMETRICS'
#: the assets server of APERO (its DRS.ASSETS_URLS)
SERVER = 'http://206.12.93.77/ari/data/apero/assets/'
#: the folder of the database in the assets
SUBDIR = 'astrometrics'
#: its tiers, the first one found wins
TIERS = ('verified', 'pending', 'rejected')
#: the index of names and places kept beside the copy, and its version (a
#: new one when what is kept of an entry changes)
INDEX = 'lbl_index.json'
INDEX_VERSION = 1
#: characters APERO replaces (all punctuation but _, and the space)
BAD_CHARS = [' '] + list(string.punctuation.replace('_', ''))
NULL_TEXT = ('', 'None', 'Null', 'NULL', 'null', 'nan', 'NaN', 'inf')
#: the classes of entry that are no star to name a folder after
NOT_STARS = ('SKY', 'MOON', 'SUN', 'PLANET', 'CALIB', 'TELLURIC')
#: the index in memory: folder: (signature, index)
_CACHE: Dict[str, Any] = {}


# =============================================================================
# Define functions
# =============================================================================
def clean_object(raw: Any) -> str:
    """
    A name as APERO cleans it (drs_astrometrics.clean_object): stripped,
    + to p and - to m, any other punctuation and the spaces to _, upper
    case, _ collapsed and stripped; 'Null' for nothing

    :param raw: the name

    :return: str, the cleaned name
    """
    if raw is None or str(raw).strip() in NULL_TEXT:
        return 'Null'
    name = str(raw).strip().replace('+', 'p').replace('-', 'm')
    for char in BAD_CHARS:
        name = name.replace(char, '_')
    name = name.upper()
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')


def name_variants(raw: Any) -> List[str]:
    """
    The variants of a name APERO looks for (drs_astrometrics.
    name_search_variants): the cleaned name first, then every choice of
    the spaces (_ or none), + (P or none) and - (M or none), each also
    without its underscores

    :param raw: the name

    :return: list of str
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if text in NULL_TEXT:
        return []
    keep = {' ', '\t', '+', '-', '_'}
    norm = ''.join('_' if (char in BAD_CHARS and char not in keep) else char
                   for char in text)
    seen: List[str] = []
    canonical = clean_object(raw)
    if canonical and canonical != 'Null':
        seen.append(canonical)
    for space in ('_', ''):
        for plus in ('P', ''):
            for minus in ('M', ''):
                val = norm.replace(' ', space).replace('\t', space)
                val = val.replace('+', plus).replace('-', minus).upper()
                while '__' in val:
                    val = val.replace('__', '_')
                val = val.strip('_')
                for one in (val, val.replace('_', '')):
                    if one and one not in seen:
                        seen.append(one)
    return seen


def folder() -> str:
    """the folder of the database read: LBL_APERO_ASTROMETRICS, else the
    copy kept here"""
    return os.path.expanduser(os.environ.get(ENV_FOLDER) or FOLDER)


def yaml_files(path: str) -> List[str]:
    """
    The YAML files of the database, as APERO lists them (iter_yaml_files):
    those of its tiers (verified, pending, rejected), then those at its
    root that no tier has

    :return: list of str
    """
    if not os.path.isdir(path):
        return []
    out, seen = [], set()
    for sub in list(TIERS) + ['']:
        here = os.path.join(path, sub)
        if not os.path.isdir(here):
            continue
        for name in sorted(os.listdir(here)):
            if name.endswith('.yaml') and name not in seen:
                seen.add(name)
                out.append(os.path.join(here, name))
    return out


def newest_tarball(server: str = SERVER, timeout: float = 30.0) -> str:
    """the newest assets tarball of APERO's server (its index lists them,
    <unix time>_<...>_assets.tar.gz)"""
    with urllib.request.urlopen(server, timeout=timeout) as resp:
        page = resp.read().decode('utf-8', 'replace')
    names = sorted(set(re.findall(r'href="(\d+_\d+_assets\.tar\.gz)"',
                                  page)),
                   key=lambda name: float(name.split('_')[0]))
    if not names:
        raise ValueError(f'no assets tarball listed at {server}')
    return names[-1]


def refresh(server: str = SERVER, timeout: float = 60.0) -> Dict[str, Any]:
    """
    A new copy of the database: the YAML files of the astrometrics folder
    of APERO's newest assets tarball, streamed (they come first in it, a
    few MB) and the rest of it left; then the index of their names

    :param server: str, the assets server of APERO
    :param timeout: float [s]

    :return: dict, the tarball, the number of files and of entries, the
             folder
    """
    if os.environ.get(ENV_FOLDER):
        raise ValueError(f'{ENV_FOLDER} is set: the database is read from '
                         f'{folder()}, not copied here')
    name = newest_tarball(server, timeout)
    part = FOLDER + '.part'
    _remove(part)
    os.makedirs(part, exist_ok=True)
    count = 0
    with urllib.request.urlopen(server + name, timeout=timeout) as resp, \
            tarfile.open(fileobj=resp, mode='r|gz') as tar:
        inside = False
        for member in tar:
            parts = member.name.strip('./').split('/')
            if parts[0] != SUBDIR:
                # the folder is one stretch of the tarball: once left, done
                if inside:
                    break
                continue
            inside = True
            if not (member.isfile() and member.name.endswith('.yaml')):
                continue
            rest = parts[1:]
            # an entry at its root or in a tier, nothing else (its history)
            if len(rest) == 1 or (len(rest) == 2 and rest[0] in TIERS):
                target = os.path.join(part, *rest)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with tar.extractfile(member) as src, \
                        open(target, 'wb') as dst:
                    dst.write(src.read())
                count += 1
    if not count:
        raise ValueError(f'no astrometric entry in {name}')
    with open(os.path.join(part, 'source.json'), 'w') as handle:
        json.dump(dict(server=server, tarball=name, entries=count,
                       fetched=time.strftime('%Y-%m-%d %H:%M')), handle)
    # the new copy in place of the old one, whole
    if os.path.isdir(FOLDER):
        old = FOLDER + '.old'
        _remove(old)
        os.replace(FOLDER, old)
        os.replace(part, FOLDER)
        _remove(old)
    else:
        os.makedirs(os.path.dirname(FOLDER), exist_ok=True)
        os.replace(part, FOLDER)
    _CACHE.pop(FOLDER, None)
    index = load_index(FOLDER)
    return dict(tarball=name, files=count, objects=len(index['entries']),
                folder=FOLDER)


def _remove(path: str) -> None:
    """a copy of the database removed (its files, its tiers, itself)"""
    if not os.path.isdir(path):
        return
    for here, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(here, name))
        for name in dirs:
            os.rmdir(os.path.join(here, name))
    os.rmdir(path)


def _signature(path: str) -> str:
    """what changes when a file of the database does"""
    files = yaml_files(path)
    stamp = max((os.path.getmtime(name) for name in files), default=0.0)
    return f'{len(files)}:{stamp:.3f}'


def _value(entry: Dict[str, Any], key: str) -> Any:
    """a value of an entry ({value: ..., source: ...} or as it is)"""
    val = entry.get(key)
    return val.get('value') if isinstance(val, dict) else val


def _float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _place(entry: Dict[str, Any]):
    """(ra, dec) of an entry in J2000 [deg], its proper motion (pmra*,
    pmdec [mas/yr], None without): RA_J2000_DEG when APERO gives it, else
    its RA and DEC at their EPOCH (a JD) brought back to 2000"""
    pmra, pmdec = _float(_value(entry, 'PMRA')), _float(_value(entry, 'PMDE'))
    ra = _float(entry.get('RA_J2000_DEG'))
    dec = _float(entry.get('DEC_J2000_DEG'))
    if ra is None or dec is None:
        ra, dec = _float(_value(entry, 'RA')), _float(_value(entry, 'DEC'))
        epoch = _float(entry.get('EPOCH'))
        if ra is not None and dec is not None and epoch and \
                pmra is not None and pmdec is not None:
            years = (2451545.0 - epoch) / 365.25
            dec_new = dec + pmdec * years / 3.6e6
            ra = ra + pmra * years / 3.6e6 / max(
                math.cos(math.radians(dec)), 1e-6)
            dec = dec_new
    if ra is None or dec is None or abs(dec) > 90:
        return None, None, None
    pm = (pmra, pmdec) if pmra is not None and pmdec is not None else None
    return ra % 360.0, dec, pm


def load_index(path: Optional[str] = None) -> Dict[str, Any]:
    """
    The index of the database: each variant of each name to its APERO
    name, and each entry with its place (kept beside the copy, built again
    when a file changes)

    :param path: str or None, the folder (folder() by default)

    :return: dict, names (variant: APERO name), entries (APERO name:
             apero, simbad, original, status, klass, ra, dec, pmra, pmdec)
    """
    path = path or folder()
    sig = _signature(path)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == sig:
        return cached[1]
    keep = os.path.join(path, INDEX)
    if os.path.exists(keep):
        try:
            with open(keep) as handle:
                index = json.load(handle)
        except (OSError, ValueError):
            index = {}
        if index.get('signature') == sig and \
                index.get('version') == INDEX_VERSION:
            _CACHE[path] = (sig, index)
            return index
    import yaml
    loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
    names: Dict[str, str] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    for name in yaml_files(path):
        try:
            with open(name) as handle:
                entry = yaml.load(handle, Loader=loader)
        except Exception:  # a malformed file: left, as APERO does
            continue
        if not isinstance(entry, dict):
            continue
        tier = os.path.basename(os.path.dirname(name))
        apero = str(entry.get('APERO_NAME') or os.path.splitext(
            os.path.basename(name))[0])
        ra, dec, pm = _place(entry)
        entries.setdefault(apero, dict(
            apero=apero, simbad=entry.get('SIMBAD_NAME'),
            original=entry.get('ORIGINAL_NAME'),
            status=tier if tier in TIERS else '',
            klass=entry.get('APERO_CLASS'), ra=ra, dec=dec,
            pmra=pm[0] if pm else None, pmdec=pm[1] if pm else None))
        aliases = entry.get('ALIASES') or []
        if isinstance(aliases, str):
            aliases = aliases.split('|')
        for value in [apero, entry.get('ORIGINAL_NAME'),
                      entry.get('SIMBAD_NAME')] + list(aliases):
            for variant in name_variants(value):
                names.setdefault(variant, apero)
    index = dict(signature=sig, version=INDEX_VERSION, names=names,
                 entries=entries)
    try:
        with open(keep, 'w') as handle:
            json.dump(index, handle)
    except OSError:  # a folder that cannot be written: kept in memory
        pass
    _CACHE[path] = (sig, index)
    return index


def available(fetch: bool = True) -> bool:
    """whether there is a database to read (a copy made when there is none
    and fetch)"""
    if yaml_files(folder()):
        return True
    if not fetch or os.environ.get(ENV_FOLDER):
        return False
    try:
        refresh()
    except (OSError, ValueError, tarfile.TarError):
        return False
    return True


def is_star(entry: Dict[str, Any]) -> bool:
    """whether an entry is a star a folder is named after (not the sky,
    the Moon, a planet of the solar system, a calibration)"""
    klass = str(entry.get('klass') or '').upper()
    return not any(klass.startswith(word) for word in NOT_STARS)


def lookup(name: Any, fetch: bool = True) -> Optional[Dict[str, Any]]:
    """
    The entry of a name in the database, as APERO finds it

    :param name: the name (raw: Gl382, GJ 382, AN SEX, GL382...)
    :param fetch: bool, a copy of the database made when there is none

    :return: dict (apero, simbad, original, status, ra, dec, pmra, pmdec)
             or None
    """
    if not available(fetch):
        return None
    index = load_index()
    for variant in name_variants(name):
        apero = index['names'].get(variant)
        if apero is not None:
            return dict(index['entries'][apero])
    return None


def near(ra: float, dec: float, radius: float = 30.0,
         fetch: bool = True) -> List[Dict[str, Any]]:
    """
    The stars of the database within `radius` [arcsec] of a place (J2000,
    deg), the nearest first, each with its separation (sep, arcsec)

    :return: list of dict
    """
    if not available(fetch):
        return []
    index = load_index()
    stars = [e for e in index['entries'].values()
             if e.get('ra') is not None and is_star(e)]
    if not stars:
        return []
    ras = np.radians([e['ra'] for e in stars])
    decs = np.radians([e['dec'] for e in stars])
    ra0, dec0 = math.radians(ra), math.radians(dec)
    # the haversine: right at a few arcsec as at 180 degrees
    hav = np.sin((decs - dec0) / 2) ** 2 + np.cos(decs) * math.cos(dec0) \
        * np.sin((ras - ra0) / 2) ** 2
    sep = np.degrees(2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))) * 3600.0
    keep = np.argsort(sep)
    return [dict(stars[i], sep=float(sep[i])) for i in keep
            if sep[i] <= radius]


def simbad_target(entry: Dict[str, Any]) -> str:
    """the name of an entry for SIMBAD: its SIMBAD name without SIMBAD's
    prefixes (NAME, *, V*) and repeated spaces, else its APERO name"""
    name = str(entry.get('simbad') or '').strip()
    name = re.sub(r'^(NAME|\*\*?|V\*)\s+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name or str(entry.get('apero') or '')


# =============================================================================
# End of code
# =============================================================================

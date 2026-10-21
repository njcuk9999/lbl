#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The velocity of a star from the Keplerian orbits of its known companions.

With PLANETS = True, lbl_template looks the object up in the catalogs below,
by its name and by every synonym the catalogs give it:

    DATA_DIR/personal_planet_catalog.yaml     your own solutions, for this
                                              data directory (if there is one)
    lbl/resources/planets/personal_planet_catalog.yaml
                                              your own solutions, for every
                                              run of this LBL
    lbl/resources/planets/known_planets.yaml  the published ones (exoplanet.eu,
                                              with the NASA Exoplanet
                                              Archive's transit epochs and
                                              APERO's names), made by
                                              build_known_planets.py

The first catalog that knows the star gives ALL its orbits: a personal entry
replaces the published one, it does not add to it.

If the star is found with at least one complete orbit, every spectrum is
registered with its BERV and the sum of the Keplerian velocities at its time,
the two composed relativistically, and the template is made in one pass. If
not, the template iterates on the velocities of the spectra as it always has.
Either way the velocities LBL reports are corrected for the BERV only: the
orbits set where the template is built, and where compute starts.

Created on 2026-10-21

@author: artigau
"""
import os
import re
from typing import Dict, List, Optional

import numpy as np
import yaml

from lbl.core import io

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.science.planets.py'
# get log
log = io.log
#: where the catalogs that come with LBL are
PLANET_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'resources', 'planets')
#: the published orbits
PUBLISHED = os.path.join(PLANET_DIR, 'known_planets.yaml')
#: the name of a personal catalog, in DATA_DIR or beside the published one
PERSONAL = 'personal_planet_catalog.yaml'
#: what LBL appends to an object's name (telluric cleaning, PCA2D): the star
#:  is what is left once they are taken off
SUFFIXES = re.compile(r'(_tc|_pca2d[0-9A-Za-z._+-]*)+$')
#: the keys of an orbit, and what each one is
ORBIT_KEYS = dict(period='period [d]',
                  k='semi-amplitude of the star [m/s]',
                  e='eccentricity',
                  omega='argument of periastron of the star [deg]',
                  tp='time of periastron [BJD]',
                  t0='time of inferior conjunction (transit) [BJD]')


# =============================================================================
# Define functions
# =============================================================================
def normalise(name) -> str:
    """
    A name as the catalogs are matched on: upper case, letters and digits
    only. 'GJ 3622', 'Gl3622' and 'gj-3622' are not the same, but 'GJ 3622',
    'GJ3622' and 'gj_3622' are.

    :param name: the name

    :return: str, the normalised name
    """
    return re.sub(r'[^0-9A-Z]', '', str(name).upper())


def star_names(objname: str, header_names: Optional[List[str]] = None
               ) -> List[str]:
    """
    The names an object is looked up by: its own, without the suffixes LBL
    adds (_tc, _pca2d...), then the names its files carry

    :param objname: str, the object as LBL calls it
    :param header_names: list of str, names read in the files' headers

    :return: list of str, the names, first one first
    """
    names = [str(objname), SUFFIXES.sub('', str(objname))]
    for name in (header_names or []):
        if name not in (None, '', 'None'):
            names.append(str(name))
    # unique, in order
    return list(dict.fromkeys(names))


def _yaml_load(path: str) -> dict:
    """A catalog, or an empty dict if the file is empty"""
    loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
    with open(path, 'r') as handle:
        content = yaml.load(handle, Loader=loader)
    return content or dict()


def catalog_files(data_dir: Optional[str] = None) -> List[str]:
    """
    The catalogs, in the order they are searched

    :param data_dir: str, LBL's DATA_DIR

    :return: list of str, the catalogs that exist
    """
    files = []
    if data_dir:
        files.append(os.path.join(data_dir, PERSONAL))
    files += [os.path.join(PLANET_DIR, PERSONAL), PUBLISHED]
    return [path for path in files if os.path.exists(path)]


def find_star(names: List[str], data_dir: Optional[str] = None
              ) -> Optional[Dict]:
    """
    The first catalog entry that knows one of these names

    :param names: list of str, the names of the object
    :param data_dir: str, LBL's DATA_DIR (for a personal catalog there)

    :return: dict (star, catalog, entry), or None
    """
    wanted = [normalise(name) for name in names]
    for path in catalog_files(data_dir):
        stars = _yaml_load(path).get('stars') or dict()
        index = dict()
        for key, entry in stars.items():
            entry = entry or dict()
            aliases = [key, entry.get('apero')] + list(entry.get('names') or [])
            for alias in aliases:
                if alias not in (None, ''):
                    index.setdefault(normalise(alias), key)
        for name in wanted:
            if name in index:
                key = index[name]
                return dict(star=key, catalog=path, entry=stars[key] or dict())
    return None


def missing(orbit: dict) -> Optional[str]:
    """
    What an orbit lacks to give a velocity, or None if it is complete

    :param orbit: dict, one planet of a catalog

    :return: str or None
    """
    for key in ('period', 'k'):
        value = orbit.get(key)
        if value is None or not np.isfinite(float(value)):
            return 'no ' + key
    if float(orbit['period']) <= 0:
        return 'period <= 0'
    if orbit.get('tp') is None and orbit.get('t0') is None:
        return 'no epoch (tp or t0)'
    ecc = float(orbit.get('e') or 0.0)
    if not 0 <= ecc < 1:
        return 'eccentricity {0} not in [0, 1)'.format(ecc)
    if ecc > 0 and orbit.get('omega') is None:
        return 'eccentric (e = {0}) with no omega'.format(ecc)
    return None


def periastron_time(orbit: dict) -> float:
    """
    The time of periastron of an orbit, from tp or from t0, the inferior
    conjunction (the transit), where omega + nu = 90 deg

    :param orbit: dict, a complete orbit

    :return: float, the time of periastron [BJD]
    """
    if orbit.get('tp') is not None:
        return float(orbit['tp'])
    ecc = float(orbit.get('e') or 0.0)
    omega = np.radians(float(orbit.get('omega', 90.0)))
    # true, eccentric and mean anomalies at the conjunction
    nu_c = np.pi / 2 - omega
    ecc_c = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu_c / 2),
                           np.sqrt(1 + ecc) * np.cos(nu_c / 2))
    mean_c = ecc_c - ecc * np.sin(ecc_c)
    return float(orbit['t0']) - float(orbit['period']) * mean_c / (2 * np.pi)


def eccentric_anomaly(mean_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    """
    Kepler's equation, M = E - e sin E, solved for E by Newton's method

    :param mean_anomaly: np.ndarray, M [rad]
    :param ecc: float, the eccentricity

    :return: np.ndarray, E [rad]
    """
    mean_anomaly = np.mod(mean_anomaly, 2 * np.pi)
    # a start that converges for every eccentricity below 1
    ecc_anom = mean_anomaly + ecc * np.sin(mean_anomaly) if ecc < 0.8 else \
        np.full_like(mean_anomaly, np.pi)
    for _ in range(50):
        step = ((ecc_anom - ecc * np.sin(ecc_anom) - mean_anomaly)
                / (1 - ecc * np.cos(ecc_anom)))
        ecc_anom = ecc_anom - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return ecc_anom


def orbit_velocity(bjd: np.ndarray, orbit: dict) -> np.ndarray:
    """
    The velocity of the star due to one companion,
    K [cos(nu + omega) + e cos(omega)], positive receding

    :param bjd: np.ndarray, the times [BJD]
    :param orbit: dict, a complete orbit

    :return: np.ndarray, the velocity [m/s]
    """
    bjd = np.atleast_1d(np.array(bjd, dtype=float))
    ecc = float(orbit.get('e') or 0.0)
    omega = np.radians(float(orbit.get('omega', 90.0)))
    period = float(orbit['period'])
    mean_anomaly = 2 * np.pi * (bjd - periastron_time(orbit)) / period
    ecc_anom = eccentric_anomaly(mean_anomaly, ecc)
    nu = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(ecc_anom / 2),
                        np.sqrt(1 - ecc) * np.cos(ecc_anom / 2))
    return float(orbit['k']) * (np.cos(nu + omega) + ecc * np.cos(omega))


def star_velocity(bjd: np.ndarray, orbits: List[dict]) -> np.ndarray:
    """
    The velocity of the star, the sum over its companions

    :param bjd: np.ndarray, the times [BJD]
    :param orbits: list of dict, complete orbits

    :return: np.ndarray, the velocity [m/s]
    """
    velocity = np.zeros_like(np.atleast_1d(np.array(bjd, dtype=float)))
    for orbit in orbits:
        velocity += orbit_velocity(bjd, orbit)
    return velocity


def orbits_for(names: List[str], data_dir: Optional[str] = None
               ) -> Optional[Dict]:
    """
    The complete orbits of the first catalog entry that knows the object,
    said in the log. None if no catalog knows it or none of its orbits is
    complete: the template then iterates as it always has.

    :param names: list of str, the names of the object
    :param data_dir: str, LBL's DATA_DIR

    :return: dict (star, catalog, orbits, skipped), or None
    """
    found = find_star(names, data_dir)
    if found is None:
        log.general('PLANETS: {0} is in none of the catalogs ({1}): the '
                    'template iterates on the velocities'.format(
                        ', '.join(names), ', '.join(catalog_files(data_dir))))
        return None
    orbits, skipped = [], []
    for orbit in found['entry'].get('planets') or []:
        why = missing(orbit)
        if why is None:
            orbits.append(orbit)
        else:
            skipped.append((orbit.get('name', '?'), why))
    for name, why in skipped:
        log.warning('PLANETS: {0} of {1} is not used: {2}'.format(
            name, found['star'], why))
    if not orbits:
        log.general('PLANETS: {0} is in {1} but has no complete orbit: the '
                    'template iterates on the velocities'.format(
                        found['star'], found['catalog']))
        return None
    log.info('PLANETS: {0} found in {1}, {2} orbit(s); the template is made '
             'on BERV + their velocities, in one pass'.format(
                 found['star'], found['catalog'], len(orbits)))
    for orbit in orbits:
        log.general('\t{0}: P = {1} d, K = {2} m/s, e = {3}, omega = {4} deg'
                    ' ({5})'.format(orbit.get('name', '?'), orbit['period'],
                                    orbit['k'], orbit.get('e', 0),
                                    orbit.get('omega', 90),
                                    orbit.get('reference', 'no reference')))
    return dict(star=found['star'], catalog=found['catalog'], orbits=orbits,
                skipped=skipped)


# =============================================================================
# End of code
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make known_planets.yaml: the published orbits LBL's PLANETS option reads.

    python -m lbl.resources.planets.build_known_planets

Three sources, read when it runs (network needed):

    exoplanet.eu       the planets (published in a refereed paper), their
                       period, K, e, omega, time of periastron, masses and the
                       star's, and their names. Through its TAP service,
                       voparis-tap-planeto.obspm.fr, table exoplanet.epn_core
    NASA Exoplanet     the transit epochs: per planet, the reference whose t0
    Archive            and P give the best-known phase today (table ps), and
                       the default solution (K, e, omega, tp) where
                       exoplanet.eu lacks them. exoplanet.eu keeps its
                       transit epochs (tzero_tr) in SINGLE precision: at
                       2.46 million days that is a step of 0.25 d, half of the
                       values are exact multiples of 0.25 d, and the phase of
                       a 3-day planet would be off by up to 15 deg. Its other
                       epochs (t_peri, t_conj) are in double precision, but
                       t_conj does not say which conjunction it is: against
                       the transit it lands at any phase
    APERO's object     the APERO names and their aliases (Google sheet, tabs
    lists              main_list and pending_list), so that a star is found
                       by the name LBL knows it by

An orbit, per planet, is ONE PAPER'S Keplerian solution whenever one is
complete (P, K, an epoch, omega if eccentric), taken whole from the NASA
Exoplanet Archive, which keeps one row per paper (table ps): the NASA default
solution if it is complete, else the most recent complete one. `reference`
is that paper, with its bibcode. Otherwise the orbit is put together from
exoplanet.eu, part by part, `source` saying where each part comes from:

    period   exoplanet.eu, or the one that goes with the epoch when the
             epoch is NASA's
    k        exoplanet.eu; else the NASA default solution; else computed from
             a mass measured in radial velocity and the star's mass
    e        exoplanet.eu (0 if not given), omega the same (90 deg if e = 0).
             An eccentricity that is an upper limit (GJ 1214 b: 0.27 -0.27
             +0) is replaced by the NASA default solution's, else by 0
    epoch    the NASA transit ephemeris (t0) if there is one, the most precise
             phase there is, and an event that holds with any e and omega;
             else exoplanet.eu's time of periastron (tp) with exoplanet.eu's
             omega; else the NASA default solution's tp, with its e and omega
    reference  the planet's page on exoplanet.eu: its bib_reference is the
             catalog's own paper for every planet, so the paper behind each
             value is not known

A planet with no K or no epoch is listed under `not_used`, with why, so
that LBL can say that the star has a companion it cannot follow. A star with
no complete orbit is not written at all.

Created on 2026-10-21

@author: artigau
"""
import csv
import datetime
import html
import io
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import yaml

from lbl.science.planets import normalise, PUBLISHED

# =============================================================================
# Define variables
# =============================================================================
EXOEU_TAP = 'http://voparis-tap-planeto.obspm.fr/tap/sync'
NASA_TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
APERO_SHEET = ('https://docs.google.com/spreadsheets/d/'
               '1dOogfEwC7wAagjVFdouB1Y1JdF9Eva4uDW6CTZ8x2FM/gviz/tq?'
               'tqx=out:csv&sheet={0}')
APERO_TABS = ('main_list', 'pending_list')
EXOEU_COLUMNS = ('target_name, alt_target_name, star_name, detection_type, '
                 'mass_detection_type, publication_status, period, '
                 'eccentricity, eccentricity_error_min, '
                 'eccentricity_error_max, periastron, k, t_peri, mass, '
                 'mass_error_min, '
                 'mass_sin_i, mass_sin_i_error_min, star_mass, inclination, '
                 'ra, dec, external_link')
NASA_COLUMNS = ('pl_name, hostname, hd_name, hip_name, tic_id, gaia_dr3_id, '
                'pl_letter, pl_refname, pl_orbper, pl_rvamp, pl_orbeccen, '
                'pl_orbeccenlim, '
                'pl_orblper, pl_orbtper, pl_tranmid, ra, dec')
#: only what was published
PUBLISHED_STATUS = 'Published in a refereed paper'
#: a mass from which K can be computed: one measured in radial velocity.
#:  Not astrometry, which gave K2-321 b (P = 2.3 d) 7.8 Jupiter masses and
#:  K = 1.9 km/s, a limit read as a value; nor TTV
MEASURED_MASSES = ('Radial Velocity',)
#: gravitational parameters [m3/s2] (IAU 2015 nominal values)
GM_SUN = 1.3271244e20
GM_JUP = 1.2668653e17
#: how close a coordinate match must be [arcsec]: an APERO object (whose
#:  position is brought back to J2000), a NASA planet (whose period must also
#:  agree to 0.1%, so that the radius can be loose)
MATCH_RADIUS = 5.0
NASA_RADIUS = 30.0
#: epochs below this are not full JDs (truncated, MJD...) and are not used
MIN_JD = 2.4e6


# =============================================================================
# Define functions
# =============================================================================
def log(message: str):
    print(datetime.datetime.now().strftime('%y%m%d %H:%M:%S | ') + message,
          flush=True)


def fetch(url: str, data: Optional[dict] = None) -> str:
    """The text of a URL, a POST when data is given"""
    body = urllib.parse.urlencode(data).encode() if data else None
    with urllib.request.urlopen(url, data=body, timeout=600) as response:
        return response.read().decode('utf-8')


def tap(url: str, query: str) -> List[Dict[str, str]]:
    """The rows of a TAP query, as dicts of strings"""
    text = fetch(url, dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv',
                           MAXREC='1000000', QUERY=query))
    if text.lstrip().startswith('<'):
        raise RuntimeError('{0} answered: {1}'.format(url, text[:500]))
    return list(csv.DictReader(io.StringIO(text)))


def num(value) -> float:
    """A float, NaN when empty or not a number"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def host_of(planet_name: str) -> str:
    """The star of a planet's name: 'HD 10697 b' -> 'HD 10697',
    'TOI-4470.01' -> 'TOI-4470'"""
    name = re.sub(r'\.0?[0-9]{1,2}$', '', planet_name.strip())
    return re.sub(r'[\s_-]*\(?[a-z][0-9]?\)?$', '', name)


def planet_variants(name: str) -> List[str]:
    """A planet's name, and the same without the primary's letter:
    exoplanet.eu's 'HD 189733 Ab' is the NASA's 'HD 189733 b'. Only A: a
    planet of B ('GJ 229 Bb') is not the primary's ('GJ 229 b')"""
    return [name, re.sub(r'\s?A([b-z])$', r' \1', name.strip())]


def eccentricity_limit(row: dict) -> bool:
    """Whether exoplanet.eu's eccentricity is an upper limit read as a value:
    GJ 1214 b has e = 0.27 -0.27 +0, the limit of an early paper"""
    ecc = num(row['eccentricity'])
    low, high = num(row['eccentricity_error_min']), num(
        row['eccentricity_error_max'])
    return bool(ecc > 0 and (high == 0 or low >= ecc))


def semi_amplitude(msini_jup: float, star_mass: float, period: float,
                   ecc: float) -> float:
    """K [m/s] from M sin i [Jupiter masses], M star [Sun], P [d] and e"""
    mplanet = msini_jup * GM_JUP
    mtotal = star_mass * GM_SUN + mplanet
    return ((2 * math.pi / (period * 86400)) ** (1 / 3) * mplanet
            / mtotal ** (2 / 3) / math.sqrt(1 - ecc ** 2))


def apero_objects() -> List[Dict[str, str]]:
    """APERO's objects, from both tabs of its sheet"""
    rows = []
    for tab in APERO_TABS:
        rows += list(csv.DictReader(io.StringIO(fetch(
            APERO_SHEET.format(tab)))))
    return [row for row in rows if row.get('OBJNAME')]


def apero_index(objects: List[Dict[str, str]]):
    """{normalised alias: set of APERO names}, and the positions at J2000"""
    index, where = dict(), dict()
    for row in objects:
        name = row['OBJNAME'].strip()
        aliases = [name, row.get('ORIGINAL_NAME', '')]
        aliases += (row.get('ALIASES') or '').split('|')
        for alias in aliases:
            if alias.strip():
                index.setdefault(normalise(alias), set()).add(name)
        ra, dec = num(row.get('RA_DEG')), num(row.get('DEC_DEG'))
        if np.isfinite(ra) and np.isfinite(dec):
            # back to J2000 with the proper motion [mas/yr], EPOCH in JD
            years = (num(row.get('EPOCH')) - 2451545.0) / 365.25
            pmra, pmde = num(row.get('PMRA')), num(row.get('PMDE'))
            if np.isfinite(years) and np.isfinite(pmra) and np.isfinite(pmde):
                dec = dec - pmde * years / 3.6e6
                ra = ra - pmra * years / 3.6e6 / np.cos(np.radians(dec))
            where[name] = (ra, dec)
    return index, where


def separation(ra1, dec1, ra2, dec2) -> float:
    """Angular separation [arcsec]"""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos = (np.sin(dec1) * np.sin(dec2)
           + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))) * 3600)


def orbit(row: dict, nasa: Optional[dict], ephem: Optional[dict]) -> dict:
    """One planet's orbit, or its reason for not having one.

    An epoch only means something with the solution it comes from: a time of
    periastron goes with the omega (and e) of the same solution, so the two
    are always taken together. A transit (an inferior conjunction) is an
    event, and holds with any solution's e and omega.

    :param row: the planet's exoplanet.eu row
    :param nasa: its NASA default solution, or None
    :param ephem: its best NASA transit ephemeris (best_ephemerides), or None
    """
    out = dict(name=row['target_name'])
    sources = []
    period = num(row['period'])
    ecc = num(row['eccentricity'])
    ecc = 0.0 if not np.isfinite(ecc) else ecc
    omega = num(row['periastron'])
    # NASA's e, omega and time of periastron, together, when they are there
    nasa_ecc = num(nasa['pl_orbeccen']) if nasa is not None else math.nan
    nasa_ok = (np.isfinite(nasa_ecc)
               and str(nasa['pl_orbeccenlim']).strip() in ('0', '')
               and (nasa_ecc == 0 or np.isfinite(num(nasa['pl_orblper']))))
    nasa_tp = num(nasa['pl_orbtper']) if nasa is not None else math.nan
    use_nasa = False
    if eccentricity_limit(row):
        if nasa_ok and (ephem is not None or nasa_tp > MIN_JD):
            use_nasa = True
            sources.append('e and omega NASA {0} (exoplanet.eu: a limit)'
                           .format(strip_tags(nasa['pl_refname'])))
        else:
            # circular, with exoplanet.eu's own omega kept for its own tp
            ecc = 0.0
            sources.append('e = 0 (exoplanet.eu: a limit)')
    # --- the semi-amplitude
    k = num(row['k'])
    if np.isfinite(k) and k > 0:
        sources.append('K exoplanet.eu')
    elif nasa is not None and num(nasa['pl_rvamp']) > 0:
        k = num(nasa['pl_rvamp'])
        sources.append('K NASA ' + strip_tags(nasa['pl_refname']))
    else:
        k = math.nan
        star_mass = num(row['star_mass'])
        msini = math.nan
        if row['mass_detection_type'] in MEASURED_MASSES:
            if np.isfinite(num(row['mass_sin_i_error_min'])):
                msini = num(row['mass_sin_i'])
            elif np.isfinite(num(row['mass_error_min'])):
                incl = num(row['inclination'])
                incl = 90.0 if not np.isfinite(incl) else incl
                msini = num(row['mass']) * math.sin(math.radians(incl))
        if (np.isfinite(msini) and msini > 0 and np.isfinite(star_mass)
                and star_mass > 0 and np.isfinite(period) and period > 0
                and ecc < 1):
            k = semi_amplitude(msini, star_mass, period, ecc)
            sources.append('K from M sin i and M star, exoplanet.eu')
    # --- the epoch
    epoch = dict()
    if ephem is not None:
        # the transit: holds with any e and omega
        epoch['t0'], period = ephem['t0'], ephem['period']
        sources.append('t0 and P NASA ' + ephem['reference'])
        if use_nasa:
            ecc, omega = nasa_ecc, num(nasa['pl_orblper'])
    elif not use_nasa and num(row['t_peri']) > MIN_JD and (
            np.isfinite(omega)):
        epoch['tp'] = num(row['t_peri'])
        sources.append('tp exoplanet.eu')
    elif nasa_ok and nasa_tp > MIN_JD:
        # NASA's solution, whole: P, e, omega and tp
        epoch['tp'] = nasa_tp
        ecc, omega = nasa_ecc, num(nasa['pl_orblper'])
        if num(nasa['pl_orbper']) > 0:
            period = num(nasa['pl_orbper'])
        sources.append('P, e, omega and tp NASA '
                       + strip_tags(nasa['pl_refname']))
    if ecc == 0 and not np.isfinite(omega):
        omega = 90.0
    # --- what is missing
    why = []
    if not (np.isfinite(period) and period > 0):
        why.append('no period')
    if not np.isfinite(k):
        why.append('no K (and no mass measured in radial velocity)')
    if not epoch:
        if num(row['t_peri']) > MIN_JD and not np.isfinite(omega):
            why.append('a time of periastron with no omega')
        else:
            why.append('no epoch in double precision')
    if ecc > 0 and not np.isfinite(omega):
        why.append('eccentric with no omega')
    if not 0 <= ecc < 1:
        why.append('eccentricity {0}'.format(ecc))
    if why:
        out['why'] = '; '.join(why)
        if np.isfinite(period):
            out['period'] = float(period)
        if np.isfinite(k):
            out['k'] = round(float(k), 3)
        return out
    out.update(period=float(period), k=round(float(k), 3),
               e=round(float(ecc), 5), omega=round(float(omega), 3))
    out.update({key: round(float(val), 6) for key, val in epoch.items()})
    out['source'] = '; '.join(sources)
    if row.get('external_link'):
        out['reference'] = row['external_link']
    return out


def reference(refname: str) -> str:
    """'Mahajan et al. 2024 (2024ApJ...963L..37M)' from NASA's HTML field"""
    text = strip_tags(refname)
    bibcode = re.search(r'/abs/([^/"\s>]+)', refname or '')
    if bibcode:
        return '{0} ({1})'.format(text, urllib.parse.unquote(bibcode.group(1)))
    return text


def complete_solution(sol: dict) -> Optional[dict]:
    """A NASA solution (one paper) as an orbit, if it has all of it: P, K, an
    epoch, and omega when it is eccentric. An eccentricity given as a limit
    is a circular fit."""
    period, k = num(sol['pl_orbper']), num(sol['pl_rvamp'])
    if not (period > 0 and k > 0):
        return None
    ecc = num(sol['pl_orbeccen'])
    omega = num(sol['pl_orblper'])
    if not np.isfinite(ecc) or str(sol['pl_orbeccenlim']).strip() == '1':
        ecc = 0.0
    t0, tp = num(sol['pl_tranmid']), num(sol['pl_orbtper'])
    out = dict(period=period, k=k, e=ecc)
    if t0 > MIN_JD:
        # an event: holds whatever omega is, but e > 0 needs one for the shape
        out['t0'] = t0
    elif tp > MIN_JD and np.isfinite(omega):
        out['tp'] = tp
    else:
        return None
    if ecc > 0 and not np.isfinite(omega):
        return None
    out['omega'] = omega if np.isfinite(omega) else 90.0
    if not 0 <= ecc < 1:
        return None
    return out


def best_solution(solutions: List[dict]) -> Optional[dict]:
    """The paper whose solution is used: among the complete ones, the one
    with the most precise K, then the most recent. Not the NASA default as
    such: for GJ 3470 b it is Awiphan et al. 2016, K = 13.4 +- 1.2 m/s, where
    Bonfils et al. 2012 and Kosiarek et al. 2019 find 9.0 +- 0.8 and
    8.21 +- 0.47"""
    usable = []
    for sol in solutions:
        orbit_ = complete_solution(sol)
        if orbit_ is not None:
            error = abs(num(sol['pl_rvamperr1'])) / orbit_['k']
            error = error if np.isfinite(error) and error > 0 else np.inf
            usable.append(((-error, sol.get('pl_pubdate') or ''), orbit_,
                           sol))
    if not usable:
        return None
    _, orbit_, sol = max(usable, key=lambda item: item[0])
    orbit_['reference'] = reference(sol['pl_refname'])
    return orbit_


def best_ephemerides(rows: List[Dict[str, str]]) -> Dict[str, dict]:
    """
    Per planet, the NASA transit ephemeris (t0 and P of ONE reference) whose
    phase is best known today: the smallest sqrt(dt0**2 + (N dP)**2), N the
    number of periods since t0. A default solution can have no t0 at all
    (HD 189733 b's is a stellar study), and the most precise t0 can come with
    a period that has long since lost it.

    :param rows: NASA ps rows with pl_tranmid

    :return: {normalised planet name: dict(t0, period, reference)}
    """
    today = 2440587.5 + datetime.datetime.now().timestamp() / 86400
    best = dict()
    for row in rows:
        t0, period = num(row['pl_tranmid']), num(row['pl_orbper'])
        if not (t0 > MIN_JD and period > 0):
            continue
        dt0 = abs(num(row['pl_tranmiderr1']))
        dper = abs(num(row['pl_orbpererr1']))
        dt0 = dt0 if np.isfinite(dt0) else 1.0
        dper = dper if np.isfinite(dper) else period * 1e-4
        sigma = math.hypot(dt0, (today - t0) / period * dper)
        key = normalise(row['pl_name'])
        if key not in best or sigma < best[key]['sigma']:
            best[key] = dict(t0=t0, period=period, sigma=sigma,
                             reference=strip_tags(row['pl_refname']))
    return best


def strip_tags(text: str) -> str:
    """The text of NASA's HTML reference field"""
    return html.unescape(re.sub(r'<[^>]+>', '', text or '')).strip()


def build(path: str = PUBLISHED):
    """Query the three sources and write the catalog"""
    log('exoplanet.eu: querying {0}'.format(EXOEU_TAP))
    exoeu = tap(EXOEU_TAP, 'SELECT {0} FROM exoplanet.epn_core'.format(
        EXOEU_COLUMNS))
    exoeu = [row for row in exoeu
             if row['publication_status'] == PUBLISHED_STATUS]
    log('exoplanet.eu: {0} planets published in a refereed paper'.format(
        len(exoeu)))
    log('NASA Exoplanet Archive: querying {0}'.format(NASA_TAP))
    nasa = list(csv.DictReader(io.StringIO(fetch(
        NASA_TAP + '?' + urllib.parse.urlencode(dict(
            query='select {0} from ps where default_flag=1'.format(
                NASA_COLUMNS), format='csv'))))))
    log('NASA Exoplanet Archive: {0} default solutions'.format(len(nasa)))
    every = list(csv.DictReader(io.StringIO(fetch(
        NASA_TAP + '?' + urllib.parse.urlencode(dict(
            query='select pl_name, hostname, pl_refname, pl_pubdate, '
                  'default_flag, pl_orbper, pl_orbpererr1, pl_rvamp, '
                  'pl_rvamperr1, pl_orbeccen, '
                  'pl_orbeccenlim, pl_orblper, pl_orbtper, pl_tranmid, '
                  'pl_tranmiderr1 from ps', format='csv'))))))
    # a time of periastron shared by several planets of one paper is the
    #   epoch of its osculating elements, not a periastron: Rivera et al.
    #   2010 give 2450602.093 for all four planets of GJ 876
    shared = dict()
    for row in every:
        if num(row['pl_orbtper']) > MIN_JD:
            shared.setdefault((row['hostname'], row['pl_refname'],
                               round(num(row['pl_orbtper']), 4)), []).append(row)
    for rows in shared.values():
        if len(rows) > 1:
            for row in rows:
                row['pl_orbtper'] = ''
    # the same in the default solutions, which the orbits put together from
    #   exoplanet.eu fall back on
    bad = {key for key, rows in shared.items() if len(rows) > 1}
    for row in nasa:
        if (row['hostname'], row['pl_refname'],
                round(num(row['pl_orbtper']), 4)) in bad:
            row['pl_orbtper'] = ''
    solutions = dict()
    for row in every:
        solutions.setdefault(normalise(row['pl_name']), []).append(row)
    ephemerides = best_ephemerides(
        [row for row in every if num(row['pl_tranmid']) > MIN_JD])
    log('NASA Exoplanet Archive: {0} solutions (one per paper and planet), '
        'transit ephemerides of {1} planets'.format(len(every),
                                                     len(ephemerides)))
    log('APERO: reading the object lists ({0})'.format(', '.join(APERO_TABS)))
    objects = apero_objects()
    index, where = apero_index(objects)
    by_name = {row['OBJNAME']: row for row in objects}
    log('APERO: {0} objects'.format(len(objects)))
    # --- the NASA solution of each exoplanet.eu planet, by name, else by
    #     position and period
    # the same for exoplanet.eu: a t_peri shared by planets of one star
    #   (GL 229) is an epoch, not each one's periastron
    shared_eu = dict()
    for row in exoeu:
        if num(row['t_peri']) > MIN_JD:
            shared_eu.setdefault((row['star_name'],
                                  round(num(row['t_peri']), 4)), []).append(row)
    for rows in shared_eu.values():
        if len(rows) > 1:
            for row in rows:
                row['t_peri'] = ''
    nasa_by_name = {normalise(row['pl_name']): row for row in nasa}
    stars = dict()
    n_nasa, n_paper = 0, 0
    for row in exoeu:
        names = [row['target_name']] + [
            name.strip() for name in re.split(r'[#,]',
                                              row['alt_target_name'] or '')
            if name.strip()]
        match = None
        for name in [v for name in names for v in planet_variants(name)]:
            match = nasa_by_name.get(normalise(name))
            if match is not None:
                break
        if match is None and np.isfinite(num(row['ra'])):
            for cand in nasa:
                if (abs(num(cand['pl_orbper']) / num(row['period']) - 1) < 1e-3
                        and separation(num(row['ra']), num(row['dec']),
                                       num(cand['ra']), num(cand['dec']))
                        < NASA_RADIUS):
                    match = cand
                    break
        star = row['star_name'] or host_of(row['target_name'])
        entry = stars.setdefault(star, dict(names=set(), planets=[],
                                            ra=num(row['ra']),
                                            dec=num(row['dec']),
                                            nasa=set()))
        # one NASA planet for one exoplanet.eu planet: two entries of a star
        #   on the same NASA planet would add its velocity twice
        if match is not None:
            if match['pl_name'] in entry['nasa']:
                log('{0}: same NASA planet ({1}) as another planet of {2}, '
                    'left out'.format(row['target_name'], match['pl_name'],
                                      star))
                continue
            entry['nasa'].add(match['pl_name'])
        n_nasa += match is not None
        entry['names'].add(star)
        entry['names'].update(host_of(name) for name in names)
        if match is not None:
            for key in ('hostname', 'hd_name', 'hip_name', 'tic_id',
                        'gaia_dr3_id'):
                if match.get(key):
                    entry['names'].add(match[key])
        planet = None
        if match is not None:
            # one paper's solution, whole, if one is complete
            best = best_solution(solutions.get(normalise(match['pl_name']),
                                               []))
            if best is not None:
                planet = dict(name=row['target_name'])
                planet.update(
                    period=float(best['period']), k=round(float(best['k']), 3),
                    e=round(float(best['e']), 5),
                    omega=round(float(best['omega']), 3))
                for key in ('t0', 'tp'):
                    if key in best:
                        planet[key] = round(float(best[key]), 6)
                planet['source'] = 'NASA Exoplanet Archive, one paper'
                planet['reference'] = best['reference']
                n_paper += 1
        if planet is None:
            ephem = (ephemerides.get(normalise(match['pl_name']))
                     if match is not None else None)
            planet = orbit(row, match, ephem)
        entry['planets'].append(planet)
    log('NASA Exoplanet Archive: {0} of {1} planets matched, {2} with a '
        'complete solution from one paper'.format(n_nasa, len(exoeu),
                                                   n_paper))
    # --- the APERO name of each star, by its names, else by position
    claimed = dict()
    for star, entry in stars.items():
        found = set()
        for name in entry['names']:
            found |= index.get(normalise(name), set())
        if len(found) > 1:
            exact = {name for name in found
                     if normalise(name) == normalise(star)
                     or normalise(by_name[name].get('ORIGINAL_NAME', ''))
                     == normalise(star)}
            found = exact
        if not found and np.isfinite(entry['ra']):
            close = [name for name, (ra, dec) in where.items()
                     if separation(entry['ra'], entry['dec'], ra, dec)
                     < MATCH_RADIUS]
            found = set(close)
        if len(found) == 1:
            apero = found.pop()
            claimed.setdefault(apero, []).append(star)
            entry['apero'] = apero
    # one APERO object claimed by several stars (the components of a binary):
    #   kept only for the one that is named like it
    for apero, claimants in claimed.items():
        if len(claimants) > 1:
            keep = [star for star in claimants
                    if normalise(star) in (
                        normalise(apero),
                        normalise(by_name[apero].get('ORIGINAL_NAME', '')))]
            for star in claimants:
                if star not in keep[:1]:
                    stars[star].pop('apero', None)
    # --- a star's names without the component letter ('HD 189733 A' is also
    #     'HD 189733'), for the stars that are the only one to have it
    bare = dict()
    for star, entry in stars.items():
        for name in entry['names']:
            short = re.sub(r'\s?[A-C]$', '', name.strip())
            if short != name.strip():
                bare.setdefault(normalise(short), set()).add((star, short))
    for variants in bare.values():
        if len({star for star, _ in variants}) == 1:
            for star, short in variants:
                stars[star]['names'].add(short)
    # --- write
    out, n_orbits, n_skipped, n_apero = dict(), 0, 0, 0
    for star in sorted(stars):
        entry = stars[star]
        complete = [p for p in entry['planets'] if 'why' not in p]
        skipped = [p for p in entry['planets'] if 'why' in p]
        if not complete:
            continue
        apero = entry.get('apero')
        names = set(entry['names'])
        if apero:
            n_apero += 1
            row = by_name[apero]
            names.update([row.get('ORIGINAL_NAME', '')]
                         + (row.get('ALIASES') or '').split('|'))
        key = apero or star
        # the APERO name first, then the others
        names = sorted({name.strip() for name in names if name.strip()}
                       - {key})
        names = ([apero] if apero else []) + names
        record = dict(apero=apero, names=names, planets=complete)
        if skipped:
            record['not_used'] = skipped
        out[key] = record
        n_orbits += len(complete)
        n_skipped += len(skipped)
    header = HEADER.format(
        date=datetime.date.today().isoformat(), nstars=len(out),
        norbits=n_orbits, nskipped=n_skipped, napero=n_apero,
        nobjects=len(objects))
    with open(path, 'w') as handle:
        handle.write(header)
        yaml.safe_dump(dict(stars=out), handle, sort_keys=False,
                       default_flow_style=None, width=100, allow_unicode=True)
    log('{0}: {1} stars, {2} orbits, {3} planets not used, {4} stars with '
        'an APERO name'.format(path, len(out), n_orbits, n_skipped, n_apero))


HEADER = """\
# =============================================================================
# known_planets.yaml: the PUBLISHED Keplerian orbits, for LBL's PLANETS option
# =============================================================================
# MADE BY lbl/resources/planets/build_known_planets.py ON {date}: do not edit
# it, it is remade from the sources. Your own orbits, and corrections to these,
# go in personal_planet_catalog.yaml, which says how to enter them and is
# searched first.
#
# {nstars} stars with at least one complete orbit ({norbits} orbits); {napero} of
# them carry their APERO name ({nobjects} objects in APERO's lists).
# {nskipped} planets of these stars have no complete orbit and are listed under
# `not_used`, with why.
#
# Each orbit is, whenever one exists, ONE PAPER'S complete Keplerian solution
# (P, K, e, omega, tp or t0), taken whole from the NASA Exoplanet Archive
# (table ps, one row per paper): `source: NASA Exoplanet Archive, one paper`
# and `reference` the paper and its bibcode. Otherwise it is put together from
# exoplanet.eu (refereed planets; its transit epochs are in single precision
# and are not used) and the NASA archive (transit ephemerides, K, e, omega),
# `source` saying where each part comes from and `reference` being the
# planet's page on exoplanet.eu. The planets are exoplanet.eu's; the APERO
# names and aliases come from APERO's object lists.
#
# Per orbit: period [d], k [m/s, of the star], e, omega [deg, of the star,
# as radial-velocity fits give it], and tp [time of periastron, BJD] or t0
# [inferior conjunction, the transit, BJD, where omega + nu = 90 deg].
# v = K [cos(nu + omega) + e cos(omega)], positive receding.
# =============================================================================
"""


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else PUBLISHED)

# =============================================================================
# End of code
# =============================================================================

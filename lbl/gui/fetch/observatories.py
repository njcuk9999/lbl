"""
How the archives of the observatories are asked, and which SOPHIE spectra
of a star are public.

Each archive is asked with no account, for what is public:

- ESO's (HARPS, NIRPS, ESPRESSO) and the CADC's (SPIRou at the CFHT)
  through their TAP services (ADQL), their files from their own file
  services (ESO_FILE, CADC_FILE);
- the OHP's (SOPHIE at the 1.93 m) through the tables of its own web
  service: the table of the CCFs of its pipeline (one per exposure, with
  its velocity), and that of the spectra (their resolution, HR or HE).

A star is found by where the telescope was pointed (_pointed), not by its
name, which is the observer's own: within POINTED of where the star was
that night, or of its place at one of the years catalogue coordinates are
of (tng.EPOCHS).

sophie_spectra(star): the sequence numbers of the public SOPHIE exposures
of a star, by resolution, those of the star itself: past their year of
embargo (the rows of a programme still protected have no date), with a
velocity and an error, cross-correlated with the mask most of them have,
each spectrum once, and of the star by their velocity (tng.same_star).
"""
import csv
import io
import json
import socket
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import tng
from .log import log
from .paths import USER_AGENT


# =============================================================================
# Define variables
# =============================================================================
#: ESO's TAP service (ObsCore and the product files) and its files
ESO_OBS = 'https://archive.eso.org/tap_obs/sync'


ESO_FILE = 'https://dataportal.eso.org/dataPortal/file/'


#: the web service of SOPHIE's archive at the OHP
SOPHIE_URL = 'http://atlas.obs-hp.fr/sophie/sophie.cgi'


#: the CADC's TAP service (CAOM) and its files
CADC_TAP = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync'


CADC_FILE = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/raven/files/'


#: how far from where a star was the telescope may have been pointed
#:    [arcsec] (the coordinates of a night, as the telescope had them), and
#:    from its place in a catalogue
POINTED = 15.0


#: how long the OHP's archive may take to accept a connection [s]: its
#:    queries can take a while, its connection does not, and a server that
#:    is down would otherwise hold a search for the whole timeout of every
#:    question before saying so
OHP_CONNECT = 8.0


#: how long a server that accepted no connection is not asked again [s]
DOWN_FOR = 300.0


#: when each host last refused or ignored a connection [s since the epoch]
_DOWN: Dict[str, float] = {}


# =============================================================================
# Define functions
# =============================================================================
def _http(url: str, data: Optional[Dict[str, Any]] = None,
          where: str = 'archive', timeout: float = 60.0,
          tries: int = 2) -> bytes:
    """a page of an archive, asked again once when it stumbles (stays
    silent for timeout seconds, or the connection fails)"""
    body = urllib.parse.urlencode(data).encode() if data else None
    last: Exception = RuntimeError('not asked')
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, data=body, headers={'User-Agent': USER_AGENT}),
                    timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                raise RuntimeError(f'{where}: not public ({err.code})'
                                   ) from err
            last = err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last = err
        if attempt + 1 < tries:
            _time.sleep(3.0)
    raise RuntimeError(f'the {where} could not be reached ({last})')


def _tap(url: str, query: str, where: str, fmt: str = 'csv'
         ) -> List[Dict[str, Any]]:
    """a question to a TAP service (ADQL): its rows"""
    raw = _http(url, dict(REQUEST='doQuery', LANG='ADQL', FORMAT=fmt,
                          MAXREC=200000, QUERY=query), where).decode(
        'utf8', 'replace')
    if fmt == 'json':
        try:
            held = json.loads(raw)
        except ValueError as err:
            raise RuntimeError(f'the {where} could not be reached (it '
                               f'answered an error)') from err
        names = [col['name'] for col in held['metadata']]
        return [dict(zip(names, row)) for row in held['data']]
    if raw.lstrip().startswith('<'):
        raise RuntimeError(f'the {where} could not be reached (it answered '
                           f'an error)')
    return list(csv.DictReader(io.StringIO(raw)))


def _table(url: str, where: str) -> List[Dict[str, str]]:
    """a table of the archives of the OHP, asked as CSV: the lines that
    begin with # are about the question"""
    raw = _http(url, None, where).decode('utf8', 'replace')
    lines = [line for line in raw.splitlines()
             if line.strip() and not line.startswith('#')]
    return [{key.strip(): str(val).strip() for key, val in row.items()
             if key} for row in csv.DictReader(lines)]


def _num(value: Any) -> float:
    """a float, nan when it is not one"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _year(rjd: Any) -> np.ndarray:
    """the year of a date [BJD - 2400000]"""
    return 2000.0 + (np.asarray(rjd, float) - 51544.5) / 365.25


def _reach(star: Dict[str, Any], margin: float = POINTED) -> float:
    """how far a star may be from its place of 2000 by now, and a margin
    [deg]"""
    return (margin + (tng._now() - tng.EPOCH) / 1000.0 * float(np.hypot(
        float(star.get('pmra') or 0.0), float(star.get('pmdec') or 0.0)))
            ) / 3600.0


def _pointed(ra: Any, dec: Any, rjd: Any, star: Dict[str, Any],
             radius: float = POINTED) -> np.ndarray:
    """whether the telescope was pointed at a star: within radius of
    where it was that night, or at one of the years of a catalogue"""
    ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(
        np.asarray(dec, float))
    years = np.column_stack([np.full(len(ra), year) for year in tng.EPOCHS]
                            + [np.atleast_1d(_year(rjd))])
    return tng.away(ra, dec, star, years) < radius


def _most(values: Sequence[Any]) -> Any:
    """the value most of them have"""
    kinds, counts = np.unique(np.asarray(values, dtype=str),
                              return_counts=True)
    return str(kinds[np.argmax(counts)])


# -----------------------------------------------------------------------------
# The OHP: SOPHIE
# -----------------------------------------------------------------------------
def _reachable(base: str, where: str):
    """stop at once, saying so, when the archive's server does not accept
    a connection (not asked again for DOWN_FOR seconds)"""
    parts = urllib.parse.urlsplit(base)
    host = parts.hostname or ''
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    if _time.time() - _DOWN.get(host, -1e9) < DOWN_FOR:
        raise RuntimeError(f'the {where} does not answer ({host} accepted no '
                           f'connection a few minutes ago): try again later')
    try:
        socket.create_connection((host, port), timeout=OHP_CONNECT).close()
    except OSError as err:
        _DOWN[host] = _time.time()
        raise RuntimeError(f'the {where} does not answer ({host} accepted no '
                           f'connection in {OHP_CONNECT:.0f} s: {err}); its '
                           f'server is down, try again later') from err


def _ohp(base: str, table: str, star: Dict[str, Any], cols: str,
         where: str) -> List[Dict[str, str]]:
    """the rows of a table of an archive of the OHP in a box about the
    place of a star in 2000 (wide enough for where it is now)"""
    _reachable(base, where)
    ra, dec = float(star['ra']) / 15.0, float(star['dec'])
    sign, dec = ('-' if dec < 0 else '+'), abs(dec)
    place = (f'J{int(ra):02d} {int(ra * 60) % 60:02d} '
             f'{ra * 3600 % 60:04.1f} {sign}{int(dec):02d} '
             f'{int(dec * 60) % 60:02d} {dec * 3600 % 60:02.0f}')
    size = max(2.0, 2.0 * 60.0 * _reach(star, 30.0))
    return _table(f'{base}?n={table}&c=o&p={urllib.parse.quote(place)}'
                  f'&f={size:.1f}&a=csv[,]&d={cols}', where)


def sophie_spectra(star: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    The public SOPHIE exposures of a star, by resolution: those of the
    table of the CCFs of its archive past their year of embargo (the rows
    of a programme still protected have no date: left out), with a
    velocity and an error, cross-correlated with the mask most of them
    have, each spectrum once, of the star by their velocity

    :param star: dict, ra, dec [deg, J2000], pmra, pmdec [mas/yr], names

    :return: dict, {'HR': [seq, ...], 'HE': [...]}, the sequence numbers by
             date (a resolution with none left out)
    """
    rows = _ohp(SOPHIE_URL, 'sophiecc', star, 'seq,objname,bjd,rv,err,mask',
                'SOPHIE archive')
    modes = {row.get('seq'): str(row.get('mode') or '').upper()
             for row in _ohp(SOPHIE_URL, 'sophie', star, 'seq,mode',
                             'SOPHIE archive')} if rows else {}
    rows = [row for row in rows if _num(row.get('bjd')) > 2400000.0
            and 0 < _num(row.get('err')) < 0.09]
    if not rows:
        return {}
    mask = _most([row['mask'] for row in rows])
    # a spectrum cross-correlated twice with the mask: once
    seen, kept = set(), []
    for row in rows:
        if row['mask'] == mask and row['seq'] not in seen:
            seen.add(row['seq'])
            kept.append(row)
    bjd = np.array([_num(row['bjd']) for row in kept])
    vel = np.array([_num(row['rv']) for row in kept])
    err = np.array([_num(row['err']) for row in kept])
    names = np.array([str(row['objname']) for row in kept])
    mode = np.array([modes.get(row['seq'], '') or 'HR' for row in kept])
    finite = np.isfinite(bjd + vel + err) & (err > 0)
    out = {}
    for res in ('HR', 'HE'):
        left: List[str] = []
        good = tng.same_star(vel, names, finite & (mode == res), left)
        if left:
            log(f'SOPHIE {res}, OHP archive: exposures left out: '
                f'{"; ".join(left)}', 'warn')
        if np.any(good):
            order = np.argsort(bjd[good])
            out[res] = [str(kept[it]['seq'])
                        for it in np.where(good)[0][order]]
    return out

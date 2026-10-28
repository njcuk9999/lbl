"""
The velocities of the pipelines of other spectrographs, from the archives
of their observatories, for koloa (HARPS-N at the TNG is koloa.tng).

Each archive is asked with no account, for what is public, and gives the
velocity its pipeline made of each exposure, in a table or in the header
of a small file:

- eso(): the archive of ESO. HARPS from its catalogue of velocities
  (Barbieri 2023: the pipeline's, 2003 to 2023, one question), ESPRESSO,
  NIRPS and HARPS since 2026 from the header of the CCF file of each
  exposure (120 to 310 kB, fetched whole: the velocity is at the end of
  a long header). The files of ESPRESSO are of the pipeline of their
  day: those of espdr 3 are 24 m/s from those of espdr 2 for Barnard's
  star, and a dataset of their own;
- neid(): NEID at the WIYN, a table of the archive of NExScI;
- sophie(): SOPHIE at the 1.93 m of the OHP, a table of its archive
  (before and after its octagonal fibres of June 2011, SOPHIE+; its two
  resolutions apart);
- elodie(): ELODIE, its predecessor (1994 to 2006), a table of its
  archive, the exposures with the thorium lamp beside the star. The table
  has no error: it is that of Perrier et al. (2003, A&A 410, 1039), the
  instrumental error of ELODIE, 6 m/s, and the photon noise added in
  quadrature, here 311 m/s over the signal-to-noise the table gives (so
  that 70 Vir has their 2.8 m/s); the mean errors of four stars of Naef
  et al. (2003) come back within 1.3 m/s;
- cfht(): SPIRou at the CFHT, from the headers of the CCF files of APERO
  at the CADC.

A star is found by where the telescope was pointed, as in koloa.tng: the
object names are the observers'.

Each dataset has the indicators of its CCF where the archive gives them,
as koloa keeps those of DACE (koloa.tng.ccf_indicators): its width, its
contrast and its bisector span (the width alone for ELODIE; none for
NEID, whose table has none).

Two of them are asked only when they are named (koloa.published
.OPTIONAL). ESO has what DACE has of HARPS, ESPRESSO and NIRPS: the same
pipeline and the same spectra (for GJ 436, 169, 23 and 99 velocities
from each), so it is of use where DACE cannot be reached, or to see the
velocities of a pipeline as its archive has them. SPIRou at the CFHT:
the CADC answers a header in 2.5 s, half an hour for the 1800 exposures
of a star.

Two things differ from one archive to the next, and each dataset says
them (the dict that goes with it):

- perspective: whether its velocities still hold the perspective
  acceleration of the star (koloa.secular). Measured on Barnard's star
  (4.54 m/s/yr): the pipelines of ESO do (HARPS 4.0 to 4.2 m/s/yr,
  NIRPS 4.0 +- 0.2), NEID's does not (-0.16 +- 0.11). APERO is told how
  the star moves and does not either: its CCF velocities of SPIRou drift
  by 2.51 +- 0.06 m/s/yr over 1761 exposures, between the 1.2 of its
  line-by-line velocities and far from the 5.7 they would show with it.
  SOPHIE and ELODIE are taken to hold it, as the pipelines of their
  family do (SOPHIE+ drifts by 2.4 m/s/yr for GJ 411, where 1.35 is
  expected: its zero point moves too, and does not settle it);
- defer: the velocity of a CCF in the near infrared is a poor one (SPIRou
  here, like NIRPS on DACE): a file given of the same spectrograph (LBL)
  is believed first, and the dataset is left out altogether when there is
  one.

These are velocities to look at a star with, not a release: the archive
of the OHP asks that what it serves be not handed on.

Created on 2026-10-13

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module observatories): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import csv
import io
import json
import re
import socket
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from . import tng
from .data import RVData


ESO_OBS = 'https://archive.eso.org/tap_obs/sync'


ESO_FILE = 'https://dataportal.eso.org/dataPortal/file/'


SOPHIE_URL = 'http://atlas.obs-hp.fr/sophie/sophie.cgi'


CADC_TAP = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync'


CADC_FILE = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/raven/files/'


#: the days the instruments changed [BJD - 2400000]: the fibres of HARPS
#: (2015), the fibre link of ESPRESSO (2019), the octagonal fibres of
#: SOPHIE (2011)
HARPS_2015, ESPRESSO_2019, SOPHIE_2011 = 57174.5, 58660.0, 55730.0


#: how far from where a star was the telescope may have been pointed
#: [arcsec] (the coordinates of a night, as the telescope had them), and
#: from its place in a catalogue of positions
POINTED = 15.0


# =============================================================================
# Define functions
# =============================================================================
def _http(url: str, data: Optional[Dict[str, Any]] = None,
          where: str = 'archive', timeout: float = 60.0,
          tries: int = 2) -> bytes:
    """a page of an archive, asked again once when it stumbles (stays
    silent for timeout seconds, or the connection fails)"""
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, data=body, headers={'User-Agent': 'koloa'}),
                    timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                raise RuntimeError(f'{where}: not public ({err.code})'
                                   ) from err
            last: Exception = err
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


def _part(label: str, key: str, reference: str, time: Any, rv: Any,
          err: Any, names: Optional[Sequence[str]] = None,
          name: Optional[str] = None, perspective: bool = True,
          defer: bool = False, said: str = '', fwhm: Any = None,
          contrast: Any = None, bis: Any = None,
          ids: Optional[Sequence[str]] = None
          ) -> Optional[Tuple[RVData, Dict[str, Any]]]:
    """
    A dataset of an archive: its exposures with a velocity and an error,
    not those under a name whose velocities are of another star, nor one
    far from the others (as koloa.tng.series)

    :param time: [BJD - 2400000]; rv and err [m/s]
    :param names: the name the observer gave each, or None
    :param fwhm: the widths of the CCFs [km/s], contrast [%] and bis (their
                 bisector spans [km/s]), or None: the indicators of the
                 dataset (koloa.tng.ccf_indicators)
    :param ids: what names each exposure in its archive, or None: those
                kept, in the order of the velocities (files)

    :return: (RVData, dict: key, label, reference, n, left, perspective,
             defer, and files with ids), or None when nothing is left
    """
    time, rv, err = (np.asarray(val, float) for val in (time, rv, err))
    good = np.isfinite(time + rv + err) & (err > 0)
    left = [said] if said else []
    if np.sum(~good):
        left.append(f'{int(np.sum(~good))} with no velocity or no error')
    if names is not None and np.any(good):
        names = np.asarray(names, dtype=str)
        kinds, counts = np.unique(names[good], return_counts=True)
        main = np.median(rv[good & (names == kinds[np.argmax(counts)])])
        for kind in kinds:
            sel = good & (names == kind)
            if abs(np.median(rv[sel]) - main) > tng.OTHER * 1000.0:
                left.append(f'{int(np.sum(sel))} as {kind}, of another '
                            f'star by their velocity')
                good &= ~sel
    if np.any(good):
        far = good & (np.abs(rv - np.median(rv[good])) > tng.FAR * 1000.0)
        if np.any(far):
            left.append(f'{int(np.sum(far))} more than {tng.FAR:g} km/s '
                        f'from the others')
        good &= ~far
    if not np.any(good):
        return None
    order = np.argsort(time[good])
    more = tng.ccf_indicators(err[good][order], *(
        None if val is None else np.asarray(val, float)[good][order]
        for val in (fwhm, contrast, bis)))
    data = RVData(time[good][order], rv[good][order], err[good][order],
                  inst=np.full(int(np.sum(good)), label), name=name,
                  indicators=more)
    about = dict(key=key, label=label, reference=reference, n=int(data.n),
                 left='; '.join(left), perspective=bool(perspective),
                 defer=bool(defer))
    if ids is not None:
        about['files'] = [str(ids[it]) for it in np.where(good)[0][order]]
    return data, about


def _most(values: Sequence[Any]) -> Any:
    """the value most of them have"""
    kinds, counts = np.unique(np.asarray(values, dtype=str),
                              return_counts=True)
    return str(kinds[np.argmax(counts)])


def _told(parts: Sequence[Tuple[RVData, Dict[str, Any]]], none: str) -> str:
    """what was found, in words"""
    return ('; '.join(f'{about["label"]}: {about["n"]}'
                      for _, about in parts) or none)


# -----------------------------------------------------------------------------
# The OHP: SOPHIE and ELODIE
# -----------------------------------------------------------------------------
#: how long the OHP's archive may take to accept a connection [s]: its
#:    queries can take a while, its connection does not, and a server that
#:    is down held the search for two minutes (60 s, twice) before saying so
#:    (atlas.obs-hp.fr accepted none from 2026-10-26 to at least 2026-10-27)
OHP_CONNECT = 8.0
#: when each host last refused or ignored a connection [s since the epoch]
_DOWN: Dict[str, float] = {}


def _reachable(base: str, where: str):
    """stop at once, saying so, when the archive's server does not accept
    a connection (not asked again for five minutes)"""
    parts = urllib.parse.urlsplit(base)
    host = parts.hostname or ''
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    if _time.time() - _DOWN.get(host, -1e9) < 300.0:
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


def sophie(star: Dict[str, Any], folder: str, refresh: bool = False,
           name: Optional[str] = None
           ) -> Tuple[List[Tuple[RVData, Dict[str, Any]]], str]:
    """
    The velocities of SOPHIE (the 1.93 m of the OHP) of a star: the table
    of the CCFs of its archive, what is past its year of embargo (the
    rows of a programme still protected have no date: left out). SOPHIE
    and SOPHIE+ apart (the fibres of June 2011), and its two resolutions

    :return: tuple, the datasets and what was found, in words
    """
    rows = _ohp(SOPHIE_URL, 'sophiecc', star,
                'seq,objname,bjd,rv,err,mask,fwhm,contrast,span',
                'SOPHIE archive')
    modes = {row.get('seq'): str(row.get('mode') or '').upper()
             for row in _ohp(SOPHIE_URL, 'sophie', star, 'seq,mode',
                             'SOPHIE archive')} if rows else {}
    rows = [row for row in rows if _num(row.get('bjd')) > 2400000.0
            and 0 < _num(row.get('err')) < 0.09]
    parts = []
    if rows:
        mask = _most([row['mask'] for row in rows])
        other = sum(row['mask'] != mask for row in rows)
        # a spectrum cross-correlated twice with a mask: once
        seen, kept = set(), []
        for row in rows:
            if row['mask'] == mask and row['seq'] not in seen:
                seen.add(row['seq'])
                kept.append(row)
        rjd = np.array([_num(row['bjd']) for row in kept]) - 2400000.0
        mode = np.array([modes.get(row['seq'], '') or 'HR' for row in kept])
        for res in ('HR', 'HE'):
            for era, late in (('SOPHIE', False), ('SOPHIE+', True)):
                sel = (mode == res) & ((rjd >= SOPHIE_2011) == late)
                if not np.any(sel):
                    continue
                label = f'{era}{"" if res == "HR" else " HE"} (OHP)'
                got = _part(
                    label, re.sub(r'[^a-z+]+', '_', label.lower()).replace(
                        '+', 'p').strip('_'), 'SOPHIE archive (OHP)',
                    rjd[sel], [_num(row['rv']) * 1000.0 for row, ok
                               in zip(kept, sel) if ok],
                    [_num(row['err']) * 1000.0 for row, ok
                     in zip(kept, sel) if ok],
                    [row['objname'] for row, ok in zip(kept, sel) if ok],
                    name, said=(f'{other} with another mask than {mask}'
                                if other else ''),
                    ids=[row['seq'] for row, ok in zip(kept, sel) if ok],
                    **{key: [_num(row.get(col)) for row, ok
                             in zip(kept, sel) if ok]
                       for key, col in (('fwhm', 'fwhm'), ('bis', 'span'),
                                        ('contrast', 'contrast'))})
                if got:
                    parts.append(got)
    return parts, _told(parts, 'SOPHIE has no public velocity of the star')

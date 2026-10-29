"""
The public HARPS-N spectra of a star in the archive of its telescope, the
Telescopio Nazionale Galileo (La Palma), kept by the IA2 (INAF, Trieste).

The archive serves files, public (FREE) once their proprietary time is
over, private (PRIV) before; it has no table of velocities. The spectra LBL
reads are the S2D_A files of the ESPRESSO pipeline adapted to HARPS-N
(espdr, Dumusque et al. 2021). Which exposures are on the star is decided
from the headers of their CCF files (40 to 70 kB each), which hold the
pipeline's velocity of the exposure:

- files(star): the CCF files of fibre A pointed near the star (one ADQL
  question to the archive's TAP service);
- exposures(star, folder): what their headers say, fetched four at a time
  and kept in a table of the folder (tng_exposures.csv), so that a header
  is never fetched twice;
- espdr_files(rows): the exposures of the star itself, by date: those of a
  star (not of a lamp or the sky), passed by the pipeline's quality
  control, with a velocity and an error, cross-correlated with the mask
  most of them have, and not of another star (below).

A star is found by where the telescope was told to point, not by its
name: the object names are the observers' own, and one star can be under
several. The coordinates are the catalogue's, of the year 2000 for most,
of the night or of a Gaia release for some: an exposure is the star's when
its pointing lies within RADIUS of where the star was at one of these
(its proper motion is needed for the stars that move). The companion of a
binary, pointed at nearby, is told apart by its velocity: the exposures
under a name whose median velocity is more than OTHER from that of the
name with the most exposures are left out, and so is any exposure more
than FAR from the median (a CCF that failed).
"""
import csv
import gzip
import io
import os
import re
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .log import log
from .paths import USER_AGENT


# =============================================================================
# Define variables
# =============================================================================
#: the TAP service of the archive of the TNG (IA2)
TAP = 'http://archives.ia2.inaf.it/vo/tap/tng'


#: how far from where a star was a pointing may be [arcsec]
RADIUS = 8.0


#: the years the catalogue coordinates may be of, beside the night's:
#:    2000 for most, those of the Gaia releases
EPOCH = 2000.0


EPOCHS = (2000.0, 2015.0, 2015.5, 2016.0)


#: the headers fetched at once
WORKERS = 4


#: the lines that say how far the headers are: one every 1/STEPS of them
STEPS = 10


#: a velocity this far from the median of the star's is of another star,
#:    or of a CCF that failed [km/s]; and the exposures under a name whose
#:    median is this far from that of the name with the most are of
#:    another star (the companion of a binary, pointed at nearby) [km/s]
FAR = 2.0


OTHER = 0.5


#: the CCF file of fibre A of the espdr reduction, by its name
_NAME = re.compile(r'^r\.HARPN\..*_CCF_A\.fits\.gz$')


#: what is read from a header: the value, its keyword
_KEYS = dict(bjd='TNG QC BJD', rv='TNG QC CCF RV', err='TNG QC CCF RV ERROR',
             mask='TNG QC CCF MASK', dpr='TNG DPR TYPE',
             qc='TNG QC SCIRED CHECK')


#: the columns of the table of the exposures of a star
COLUMNS = ['file', 'policy', 'object', 'date', 'bjd', 'rv', 'err', 'mask',
           'dpr', 'qc', 'error']


#: the table of the exposures of a star, in its folder
TABLE = 'tng_exposures.csv'


# =============================================================================
# Define functions
# =============================================================================
def _ask(query: str, wait: float = 600.0, timeout: float = 60.0
         ) -> List[Dict[str, str]]:
    """
    A question to the archive (ADQL), asked the asynchronous way: a
    question with a condition is not answered the other way

    :return: list of dict, the rows
    """
    data = urllib.parse.urlencode(dict(
        REQUEST='doQuery', LANG='ADQL', FORMAT='csv', PHASE='RUN',
        QUERY=query)).encode()
    try:
        job = urllib.request.urlopen(urllib.request.Request(
            TAP + '/async', data=data, headers={'User-Agent': USER_AGENT}),
            timeout=timeout).geturl()
        start, phase = _time.time(), ''
        while _time.time() - start < wait:
            phase = urllib.request.urlopen(
                job + '/phase', timeout=timeout).read().decode().strip()
            if phase in ('COMPLETED', 'ERROR', 'ABORTED'):
                break
            _time.sleep(2.0)
        if phase != 'COMPLETED':
            raise RuntimeError(
                f'the TNG archive could not be reached (no answer after '
                f'{_time.time() - start:.0f} s: {phase or "no phase"})')
        text = urllib.request.urlopen(
            job + '/results/result', timeout=timeout).read().decode(
            'utf8', 'replace')
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise RuntimeError(f'the TNG archive could not be reached ({err})'
                           ) from err
    if text.lstrip().startswith('<'):
        raise RuntimeError('the TNG archive could not be reached (it '
                           'answered an error)')
    return list(csv.DictReader(io.StringIO(text)))


def _now() -> float:
    """this year, with its decimals"""
    return 1970.0 + _time.time() / (365.25 * 86400.0)


def _offsets(ra: Any, dec: Any, star: Dict[str, Any]
             ) -> Tuple[np.ndarray, np.ndarray]:
    """places of the sky from that of a star in EPOCH, east and north
    [arcsec]"""
    cosd = np.cos(np.radians(float(star['dec'])))
    east = (np.asarray(ra, float) - float(star['ra'])) * cosd * 3600.0
    east = (east + 648000.0) % 1296000.0 - 648000.0
    north = (np.asarray(dec, float) - float(star['dec'])) * 3600.0
    return east, north


def away(ra: Any, dec: Any, star: Dict[str, Any],
         years: Any = EPOCHS) -> np.ndarray:
    """
    How far places of the sky are from where a star was at the nearest of
    some years [arcsec]: its place in EPOCH (ra, dec of the star) moved by
    its proper motion (pmra, pmdec [mas/yr], none when not known)

    :param years: float or sequence, or an array with a year for each
                  place beside those of EPOCHS (a column of years: its
                  shape is (number of places, number of years))

    :return: np.ndarray [arcsec]
    """
    east, north = _offsets(ra, dec, star)
    span = (np.atleast_1d(np.asarray(years, float)) - EPOCH) / 1000.0
    pme = float(star.get('pmra') or 0.0) * span
    pmn = float(star.get('pmdec') or 0.0) * span
    return np.min(np.hypot(east[..., None] - pme, north[..., None] - pmn),
                  axis=-1)


def files(star: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    The espdr CCF files of fibre A pointed at a star (one question, about
    20 s)

    :param star: dict, ra, dec [deg, J2000], and pmra, pmdec [mas/yr]

    :return: list of dict: file, url, policy (FREE or PRIV), object, date
             (of the start of the exposure, UTC), by date
    """
    span = max(_now() - EPOCH, 0.0)
    reach = RADIUS + 1.0 + span / 1000.0 * float(np.hypot(
        float(star.get('pmra') or 0.0), float(star.get('pmdec') or 0.0)))
    half = np.radians(reach / 3600.0)
    ra, dec = np.radians(float(star['ra'])), np.radians(float(star['dec']))
    wide = half / max(np.cos(dec), 0.05)
    # LIKE does not tell the cases apart there, and _ is any character: the
    #   names are matched again here (_NAME)
    rows = _ask(
        "SELECT TOP 100000 OBJECT, policy, DATE_OBS, RA_RAD, DEC_RAD, "
        "file_url FROM tng.TNG_TAP WHERE INSTRUMENT = 'HARPN' AND file_url "
        "LIKE '%ccf%A.fits.gz' AND "
        f'RA_RAD BETWEEN {ra - wide:.7f} AND {ra + wide:.7f} AND '
        f'DEC_RAD BETWEEN {dec - half:.7f} AND {dec + half:.7f}')
    out = []
    for row in rows:
        name = row['file_url'].rsplit('/', 1)[-1]
        if not _NAME.match(name):
            continue
        date = row.get('DATE_OBS') or ''
        try:
            year = float(date[:4]) + (float(date[5:7]) - 0.5) / 12.0
            far = away(np.degrees(float(row['RA_RAD'])),
                       np.degrees(float(row['DEC_RAD'])), star,
                       EPOCHS + (year,))
        except ValueError:
            continue
        if float(far) < RADIUS:
            out.append(dict(file=name, url=row['file_url'],
                            policy=row.get('policy') or '',
                            object=row.get('OBJECT') or '', date=date))
    return sorted(out, key=lambda one: (one['date'], one['file']))


def header(raw: bytes) -> Dict[str, Any]:
    """
    The first header of a FITS file, gzipped or not, as a dict (the
    HIERARCH keywords without that word)
    """
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    out: Dict[str, Any] = {}
    for start in range(0, len(raw), 80):
        card = raw[start:start + 80].decode('ascii', 'replace')
        if card.startswith('END') and not card[3:].strip():
            break
        if card.startswith('HIERARCH') and '=' in card:
            key, value = card[8:].split('=', 1)
        elif card[8:10] == '= ':
            key, value = card[:8], card[10:]
        else:
            continue
        value = value.strip()
        if value.startswith("'"):
            end = value.find("'", 1)
            out[key.strip()] = value[1:end if end > 0 else None].strip()
            continue
        value = value.split(' /', 1)[0].strip()
        if value in ('T', 'F'):
            out[key.strip()] = value == 'T'
            continue
        try:
            out[key.strip()] = float(value) if re.search(
                r'[.eE]', value) else int(value)
        except ValueError:
            out[key.strip()] = value
    return out


def _fetch(one: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    """what the header of a CCF file says, fetched"""
    row = {key: '' for key in COLUMNS}
    row.update(file=one['file'], policy=one['policy'], object=one['object'],
               date=one['date'])
    try:
        with urllib.request.urlopen(urllib.request.Request(
                one['url'], headers={'User-Agent': USER_AGENT}),
                timeout=timeout) as response:
            head = header(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, EOFError,
            ValueError) as err:
        row['error'] = f'{type(err).__name__}: {err}'
        return row
    for key, word in _KEYS.items():
        value = head.get(word, '')
        row[key] = value if not isinstance(value, bool) else int(value)
    return row


def _progress(things: Iterable, total: int, words: str) -> Iterable:
    """the headers as they come, with a line every 1/STEPS of them: how
    far, how long so far and how long is left"""
    start, step = _time.time(), max(1, -(-int(total) // STEPS))
    for done, one in enumerate(things, 1):
        if done % step == 0 or done == total:
            took = _time.time() - start
            log(f'{words}: {done} of {total} headers, {took:.0f} s so far'
                + (f', about {took * (total - done) / done:.0f} s left'
                   if done < total else ''), 'info')
        yield one


def exposures(star: Dict[str, Any], folder: str) -> List[Dict[str, Any]]:
    """
    What the headers of the CCF files of a star say, kept in a table of
    the folder (TABLE): only the public files that are not in it yet are
    fetched, and the private ones are listed with nothing read

    :param star: dict, as for files()
    :param folder: str, where the table is kept

    :return: list of dict, a row per file (COLUMNS)
    """
    table = os.path.join(folder, TABLE)
    known: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(table):
        with open(table, encoding='utf-8') as handle:
            # what was read: not a file that failed, nor one that was
            #   private then
            known = {row['file']: row for row in csv.DictReader(handle)
                     if not row.get('error') and row.get('policy') == 'FREE'}
    listed = files(star)
    ask = [one for one in listed
           if one['policy'] == 'FREE' and one['file'] not in known]
    fetched: List[Dict[str, Any]] = []
    if ask:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            fetched = list(_progress(pool.map(_fetch, ask), len(ask),
                                     'HARPS-N, TNG archive'))
    new = {row['file']: row for row in fetched}
    rows = []
    for one in listed:
        row = new.get(one['file']) or known.get(one['file'])
        if row is None or (one['policy'] != 'FREE' and not row.get('rv')):
            # a private file: listed, nothing of it read
            row = {key: '' for key in COLUMNS}
            row.update(file=one['file'], object=one['object'],
                       date=one['date'])
        rows.append(dict(row, policy=one['policy']))
    os.makedirs(folder, exist_ok=True)
    with open(table, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in fetched if row['error']]
    if failed:
        log(f'TNG archive: {len(failed)} headers could not be read '
            f'({failed[0]["error"]})', 'warn')
    return rows


def _number(value: Any) -> float:
    """a float, nan when it is not one"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def same_star(vel: np.ndarray, names: np.ndarray, good: np.ndarray,
              left: List[str]) -> np.ndarray:
    """
    The exposures among `good` that are of the star by their velocity
    [km/s]: not those under a name whose median is more than OTHER from
    that of the name with the most exposures (the companion of a binary,
    pointed at nearby), nor one more than FAR from the median of the rest

    :param vel: np.ndarray, the velocities [km/s]
    :param names: np.ndarray of str, the name the observer gave each
    :param good: np.ndarray of bool, the exposures to look at
    :param left: list of str, why exposures were left out (appended to)

    :return: np.ndarray of bool, those kept
    """
    good = np.array(good, bool)
    if not np.any(good):
        return good
    kinds, counts = np.unique(names[good], return_counts=True)
    main = np.median(vel[good & (names == kinds[np.argmax(counts)])])
    for kind in kinds:
        sel = good & (names == kind)
        if abs(np.median(vel[sel]) - main) > OTHER:
            left.append(f'{int(np.sum(sel))} as {kind}, of another star by '
                        f'their velocity')
            good &= ~sel
    if np.any(good):
        far = good & (np.abs(vel - np.median(vel[good])) > FAR)
        if np.any(far):
            left.append(f'{int(np.sum(far))} more than {FAR:g} km/s from '
                        f'the others')
        good &= ~far
    return good


def espdr_files(rows: List[Dict[str, Any]]) -> List[str]:
    """
    The CCF files of the exposures of the star itself, from a table of
    exposures: those of a star, passed by the pipeline's quality control,
    with a velocity and an error, cross-correlated with the mask most of
    them have (another mask is another zero point), of the star by their
    velocity (same_star)

    :param rows: list of dict, from exposures()

    :return: list of str, the names of the CCF files, by date
    """
    mine = [row for row in rows if row.get('rv') not in ('', None)]
    if not mine:
        return []
    bjd = np.array([_number(row['bjd']) for row in mine])
    vel = np.array([_number(row['rv']) for row in mine])
    err = np.array([_number(row['err']) for row in mine])
    mask = np.array([str(row.get('mask') or '') for row in mine])
    star = np.array([str(row.get('dpr') or '').upper().startswith('STAR')
                     for row in mine])
    good = (np.isfinite(bjd + vel + err) & (err > 0) & star
            & np.array([str(row.get('qc', '')) not in ('0', '0.0')
                        for row in mine]))
    left: List[str] = []
    if np.sum(~good):
        left.append(f'{int(np.sum(~good))} with no velocity, no error or '
                    f'not of a star')
    if not np.any(good):
        return []
    kinds, counts = np.unique(mask[good], return_counts=True)
    best = str(kinds[np.argmax(counts)])
    other = good & (mask != best)
    if np.any(other):
        left.append(f'{int(np.sum(other))} with another mask than {best}')
    good &= mask == best
    names = np.array([str(row.get('object') or '') for row in mine])
    good = same_star(vel, names, good, left)
    if left:
        log(f'HARPS-N, TNG archive: exposures left out: {"; ".join(left)}',
            'warn')
    order = np.argsort(bjd[good])
    return [str(mine[it]['file']) for it in np.where(good)[0][order]]

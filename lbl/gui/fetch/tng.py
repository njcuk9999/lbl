"""
The velocities of HARPS-N from the archive of its telescope, for koloa.

HARPS-N is at the Telescopio Nazionale Galileo (La Palma), whose archive is
kept by the IA2 (INAF, Trieste). It serves files, not velocities: the raw
frames and what the pipeline made of them, public (FREE) once their
proprietary time is over, private (PRIV) before. The pipeline's velocity
of an exposure is in the header of its CCF file (40 to 70 kB), which is
all that is read here:

    from . import tng
    parts = tng.velocities(dict(ra=269.4521, dec=4.6934,
                                pmra=-801.6, pmdec=10362.4), 'gj699_tng')

The archive is asked in two steps. pointings() is every place HARPS-N was
pointed at with a CCF to show for it (one question, 40 s, kept a month):
with it, whether a star has anything is known at once, for a whole sample.
files() then lists the CCF files of one star (20 s), and exposures()
fetches the public ones, four at a time (a hundred in 15 s), and keeps what
their headers say in a table beside them, so that nothing is fetched twice.
A terminal shows a bar meanwhile (tqdm); the page of the GUI and the log
of a batch, which are read line by line, a line every tenth of the files.

A star is found by where the telescope was told to point, not by its name:
the object names are the observers' (Barnard's star is GJ699 and K4_02,
GJ 625 is M74, GJ 15 A is M22 and its companion M108). The coordinates
are the catalogue's, of the year 2000 for most, of the night or of a Gaia
release for some: an exposure is the star's when its pointing lies within
RADIUS of where the star was at one of these (its proper motion is needed
for the stars that move). The path of a star would not do: the companion
of a binary lies on it.

The spectra were reduced twice, and both are kept, each a dataset of its
own: the pipeline HARPS-N was built with (HARPN_3.7; keywords TNG DRS CCF
RVC, CCF NOISE, BJD) and the one of ESPRESSO adapted to it (espdr,
Dumusque et al. 2021; keywords TNG QC CCF RV, CCF RV ERROR, BJD). They are
two reductions of the same spectra, which koloa.datasets compares like any
two releases; their zero points differ (the line lists do), which an
offset per dataset takes. The older reduction dates in UTC, the newer in
TDB: 69 s apart, well within what two releases of a spectrum may differ
by.

These velocities are barycentric but still hold the perspective
acceleration of the star (4.5 m/s/yr for Barnard's star: measured, 4.70
+- 0.11 in the newer reduction over 7 years), which the velocities of
other pipelines do not: koloa takes it out where the series are put
together (koloa.secular).

For an M dwarf the pipeline uses its M2 mask, whose CCF is a poor match
(Rainer et al. 2020): a velocity made from the spectra themselves (the
published ones, by TERRA or SERVAL; LBL) is better, and koloa prefers it
when it has both.

Created on 2026-10-13

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module tng): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import csv
import gzip
import io
import json
import os
import re
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from . import log as klog
from .data import RVData
from .log import log
from .paths import cache


# =============================================================================
# Define variables
# =============================================================================
#: the TAP service of the archive of the TNG (IA2), and what it is called
TAP = 'http://archives.ia2.inaf.it/vo/tap/tng'


#: where the list of the pointings is kept, and for how long [days]
CACHE = cache('tng')


POINTINGS = os.path.join(CACHE, 'pointings.json')


MAX_AGE = 30.0


#: how far from where a star was a pointing may be [arcsec]
RADIUS = 8.0


#: the years the catalogue coordinates may be of, beside the night's:
#: 2000 for most, those of the Gaia releases
EPOCH = 2000.0


EPOCHS = (2000.0, 2015.0, 2015.5, 2016.0)


#: the year HARPS-N saw its first star
FIRST = 2012.0


#: the files fetched at once
WORKERS = 4


#: the lines that say how far the files are, where no bar can be drawn
STEPS = 10


#: a velocity this far from the median of its reduction is of another
#: star, or of a CCF that failed [km/s]; and the exposures under a name
#: whose median is this far from that of the name with the most are of
#: another star (the companion of a binary, pointed at where the star
#: was to be)
FAR = 2.0


OTHER = 0.5


#: the most a bisector span can be, in widths of its CCF: beyond, the
#: measure failed
BIS_MOST = 0.1


#: the two reductions: how koloa calls their datasets (the spectrograph
#: first: koloa.published.family reads it), their key, and where they are
#: from, in words
REDUCTIONS = {
    '3.7': dict(key='tng_drs37', label='HARPS-N (TNG 3.7)',
                reference='TNG archive (HARPS-N, pipeline 3.7)'),
    'espdr': dict(key='tng_espdr', label='HARPS-N (TNG espdr)',
                  reference='TNG archive (HARPS-N, pipeline espdr)'),
}


#: the CCF file of fibre A of each reduction, by its name
_NAMES = {'3.7': re.compile(r'^HARPN\..*_ccf_[A-Za-z0-9]+_A\.fits\.gz$'),
          'espdr': re.compile(r'^r\.HARPN\..*_CCF_A\.fits\.gz$')}


#: what is read from a header, for each reduction: the value, its keywords
_KEYS = {
    '3.7': dict(bjd='TNG DRS BJD', rv='TNG DRS CCF RVC',
                err='TNG DRS CCF NOISE', fwhm='TNG DRS CCF FWHM',
                contrast='TNG DRS CCF CONTRAST', mask='TNG DRS CCF MASK',
                version='TNG DRS VERSION', berv='TNG DRS BERV',
                dpr='TNG DPR TYPE', target='TNG OBS TARG NAME',
                program='TNG OBS PROG ID'),
    'espdr': dict(bjd='TNG QC BJD', rv='TNG QC CCF RV',
                  err='TNG QC CCF RV ERROR', fwhm='TNG QC CCF FWHM',
                  contrast='TNG QC CCF CONTRAST', bis='TNG QC CCF BIS SPAN',
                  mask='TNG QC CCF MASK', version='ESO PRO REC1 PIPE ID',
                  berv='TNG QC BERV', dpr='TNG DPR TYPE',
                  target='TNG OBS TARG NAME', program='TNG OBS PROG ID',
                  qc='TNG QC SCIRED CHECK'),
}


#: the columns of the table of the exposures of a star
COLUMNS = ['file', 'reduction', 'policy', 'object', 'date', 'bjd', 'rv',
           'err', 'fwhm', 'contrast', 'bis', 'mask', 'version', 'berv',
           'dpr', 'target', 'program', 'qc', 'error']


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
            TAP + '/async', data=data, headers={'User-Agent': 'koloa'}),
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


def _ccf(where: str = '') -> str:
    """the condition of a question for the CCF files of fibre A (LIKE does
    not tell the cases apart there, and _ is any character: the names are
    looked at again here)"""
    return ("INSTRUMENT = 'HARPN' AND file_url LIKE '%ccf%A.fits.gz'"
            + (f' AND {where}' if where else ''))


def pointings(refresh: bool = False) -> Dict[str, Any]:
    """
    Every place HARPS-N was pointed at with a CCF file to show for it,
    kept MAX_AGE days

    :param refresh: bool, ask the archive again

    :return: dict, ra and dec [deg] (np.ndarray), object (list of str),
             free and private (np.ndarray: their files, of both
             reductions), and when it was asked
    """
    kept = None
    if os.path.exists(POINTINGS) and not refresh:
        with open(POINTINGS) as handle:
            kept = json.load(handle)
        if _time.time() - kept.get('asked', 0) > MAX_AGE * 86400:
            kept = None
    if kept is None:
        log('asking the TNG archive where HARPS-N was pointed (once a '
            'month, under a minute)', 'info')
        rows = _ask('SELECT OBJECT, RA_RAD, DEC_RAD, policy, COUNT(*) AS '
                    f'nfile FROM tng.TNG_TAP WHERE {_ccf()} GROUP BY '
                    'OBJECT, RA_RAD, DEC_RAD, policy')
        found: Dict[Tuple[str, float, float], List[int]] = {}
        for row in rows:
            try:
                key = (row['OBJECT'], round(float(row['RA_RAD']), 9),
                       round(float(row['DEC_RAD']), 9))
                num = int(float(row['nfile']))
            except (KeyError, ValueError):
                continue
            # the Sun, the sky, the lamps: a pointing at the origin
            if key[1] == 0.0 and key[2] == 0.0:
                continue
            found.setdefault(key, [0, 0])[row['policy'] != 'FREE'] += num
        kept = dict(asked=_time.time(),
                    object=[key[0] for key in found],
                    ra=[float(np.degrees(key[1])) for key in found],
                    dec=[float(np.degrees(key[2])) for key in found],
                    free=[val[0] for val in found.values()],
                    private=[val[1] for val in found.values()])
        os.makedirs(CACHE, exist_ok=True)
        with open(POINTINGS, 'w') as handle:
            json.dump(kept, handle)
        log(f'TNG archive: {len(found)} pointings of HARPS-N', 'value')
    return dict(asked=kept['asked'], object=list(kept['object']),
                ra=np.array(kept['ra'], float),
                dec=np.array(kept['dec'], float),
                free=np.array(kept['free'], int),
                private=np.array(kept['private'], int))


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


def offpath(ra: Any, dec: Any, star: Dict[str, Any], start: float = FIRST,
            end: Optional[float] = None) -> np.ndarray:
    """
    How far places of the sky are from the path of a star between two
    years [arcsec] (from FIRST to now when not said)

    :return: np.ndarray [arcsec]
    """
    end = _now() if end is None else end
    east, north = _offsets(ra, dec, star)
    pm = np.array([float(star.get('pmra') or 0.0),
                   float(star.get('pmdec') or 0.0)]) / 1000.0
    first, step = pm * (start - EPOCH), pm * max(end - start, 0.0)
    length = float(np.sum(step ** 2))
    east, north = east - first[0], north - first[1]
    along = (np.clip((east * step[0] + north * step[1]) / length, 0.0, 1.0)
             if length > 0 else np.zeros_like(east))
    return np.hypot(east - along * step[0], north - along * step[1])


def near(star: Dict[str, Any], refresh: bool = False) -> Dict[str, Any]:
    """
    What the archive may have of a star, from the list of the pointings
    (no question asked once the list is there): the pointings at its place
    of one of EPOCHS, or on its path since HARPS-N began. The list has no
    dates, so a companion that crossed that path counts too: files()
    tells them apart, night by night

    :param star: dict, ra, dec [deg, J2000], and pmra, pmdec [mas/yr]

    :return: dict, free and private (whether it has public and private
             files), nfree and nprivate (how many: those of a companion
             too), objects (the names the observers gave it)
    """
    there = pointings(refresh)
    sel = ((away(there['ra'], there['dec'], star) < RADIUS)
           | (offpath(there['ra'], there['dec'], star) < RADIUS))
    return dict(free=bool(np.any(there['free'][sel] > 0)),
                private=bool(np.any(there['private'][sel] > 0)),
                nfree=int(np.sum(there['free'][sel])),
                nprivate=int(np.sum(there['private'][sel])),
                objects=sorted({there['object'][it]
                                for it in np.where(sel)[0]}))


def reduction_of(name: str) -> Optional[str]:
    """the reduction a file is of, from its name (None: not a CCF of
    fibre A)"""
    return next((key for key, pattern in _NAMES.items()
                 if pattern.match(name)), None)


def files(star: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    The CCF files of a star in the archive (one question, 20 s)

    :param star: dict, as for near()

    :return: list of dict: file, url, reduction, policy (FREE or PRIV),
             object, date (of the start of the exposure, UTC)
    """
    span = max(_now() - EPOCH, 0.0)
    reach = RADIUS + 1.0 + span / 1000.0 * float(np.hypot(
        float(star.get('pmra') or 0.0), float(star.get('pmdec') or 0.0)))
    half = np.radians(reach / 3600.0)
    ra, dec = np.radians(float(star['ra'])), np.radians(float(star['dec']))
    wide = half / max(np.cos(dec), 0.05)
    rows = _ask(
        'SELECT TOP 100000 OBJECT, policy, DATE_OBS, RA_RAD, DEC_RAD, '
        'file_url FROM tng.TNG_TAP WHERE ' + _ccf(
            f'RA_RAD BETWEEN {ra - wide:.7f} AND {ra + wide:.7f} AND '
            f'DEC_RAD BETWEEN {dec - half:.7f} AND {dec + half:.7f}'))
    out = []
    for row in rows:
        name = row['file_url'].rsplit('/', 1)[-1]
        red = reduction_of(name)
        if red is None:
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
            out.append(dict(file=name, url=row['file_url'], reduction=red,
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
    row.update(file=one['file'], reduction=one['reduction'],
               policy=one['policy'], object=one['object'], date=one['date'])
    try:
        with urllib.request.urlopen(urllib.request.Request(
                one['url'], headers={'User-Agent': 'koloa'}),
                timeout=timeout) as response:
            head = header(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, EOFError,
            ValueError) as err:
        row['error'] = f'{type(err).__name__}: {err}'
        return row
    for key, word in _KEYS[one['reduction']].items():
        value = head.get(word, '')
        row[key] = value if not isinstance(value, bool) else int(value)
    return row


def _steps(things: Iterable, total: int, words: str,
           key: Optional[str] = None) -> Iterable:
    """the files as they come, with a line every tenth of them: how far,
    how long so far and how long is left. For koloa's GUI, the row of the
    archive in its table instead (every second), and no line"""
    start, step = _time.time(), max(1, -(-int(total) // STEPS))
    gui = bool(key) and os.environ.get('KOLOA_PROGRESS', '') == 'gui'
    last = 0.0
    for done, one in enumerate(things, 1):
        now = _time.time()
        if gui:
            if now - last >= 1.0 or done == total:
                last = now
                klog.row(key, 'asking', '', done, total)
        elif done % step == 0 or done == total:
            took = now - start
            log(f'{words}: {done} of {total} files, {took:.0f} s so far'
                + (f', about {took * (total - done) / done:.0f} s left'
                   if done < total else ''), 'info')
        yield one


def _progress(things: Iterable, total: int, words: str,
              key: Optional[str] = None) -> Iterable:
    """
    How far the files are, as they come (when koloa talks): a bar in a
    terminal (tqdm), which draws over itself there; a line every tenth of
    the files where the output is read line by line (the log of a batch:
    a bar would be a line at each of its steps), or where there is no
    tqdm; and for koloa's GUI, the row of the archive (key) in its table
    """
    if not klog.VERBOSE:
        return things
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            return tqdm(things, total=total, desc=words, unit='file')
    return _steps(things, total, words, key)


def exposures(star: Dict[str, Any], folder: str, refresh: bool = False,
              told: Optional[Dict[str, Any]] = None
              ) -> List[Dict[str, Any]]:
    """
    What the headers of the CCF files of a star say, kept in a table of
    the folder (tng_exposures.csv); only the public files that are not in
    it yet are fetched, and the private ones are listed with nothing read.
    In LBL's window the archive is always asked: koloa first reads what its
    users shared of the star (koloa.shared, a team's repository), which is
    left out here.

    :param star: dict, as for near()
    :param folder: str, where the table is kept
    :param refresh: bool, fetch again what is there (the archive, not
                    what was shared)
    :param told: dict or None, filled with how the files came: shared
                 (from the shared tables), fetched (from the archive),
                 asked (whether the archive was asked)

    :return: list of dict, a row per file (COLUMNS)
    """
    table = os.path.join(folder, 'tng_exposures.csv')
    known: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(table) and not refresh:
        with open(table) as handle:
            # what was read: not a file that failed, nor one that was
            #   private then
            known = {row['file']: row for row in csv.DictReader(handle)
                     if not row.get('error') and row.get('policy') == 'FREE'}
    told = told if told is not None else {}
    told.update(shared=0, fetched=0, asked=False)
    told['asked'] = True
    listed = files(star)
    ask = [one for one in listed
           if one['policy'] == 'FREE' and one['file'] not in known]
    fetched: List[Dict[str, Any]] = []
    if ask:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            fetched = list(_progress(pool.map(_fetch, ask), len(ask),
                                     'HARPS-N, TNG archive', 'tng'))
    new = {row['file']: row for row in fetched}
    rows = []
    for one in listed:
        row = new.get(one['file']) or known.get(one['file'])
        if row is None or (one['policy'] != 'FREE' and not row.get('rv')):
            # a private file: listed, nothing of it read
            row = {key: '' for key in COLUMNS}
            row.update(file=one['file'], reduction=one['reduction'],
                       object=one['object'], date=one['date'])
        row = dict(row, policy=one['policy'])
        rows.append(row)
    _keep(rows, table)
    told['fetched'] = sum(1 for row in fetched if not row['error'])
    failed = [row for row in fetched if row['error']]
    if failed:
        log(f'TNG archive: {len(failed)} files could not be read '
            f'({failed[0]["error"]})', 'warn')
    return rows


def _keep(rows: List[Dict[str, Any]], table: str):
    """the table of the exposures of a star, written"""
    os.makedirs(os.path.dirname(table) or '.', exist_ok=True)
    with open(table, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def ccf_indicators(err: Any, fwhm: Any = None, contrast: Any = None,
                   bis: Any = None) -> Dict[str, Tuple[np.ndarray,
                                                       np.ndarray]]:
    """
    The indicators of a CCF as koloa keeps them (those of DACE): its
    width fwhm [m/s], its bisector span bis [m/s], its contrast [%], each
    with the error the pipelines give it: twice that of the velocity for
    the width and the bisector, and the contrast times the relative error
    of the width. A bisector span of more than BIS_MOST of the width is
    a measure that failed, not a span (the pipeline of ESPRESSO gives
    1.8 km/s for most CCFs of Barnard's star with its M2 mask, and 4 m/s
    for the others): left blank

    :param err: the errors of the velocities [m/s]
    :param fwhm: the widths [km/s], or None
    :param contrast: the contrasts [%], or None
    :param bis: the bisector spans [km/s], or None

    :return: dict, {name: (values, errors)} for those given and not all
             missing
    """
    err = np.asarray(err, float)
    out = {}
    if fwhm is not None:
        fwhm = np.asarray(fwhm, float) * 1000.0
        out['fwhm'] = (fwhm, np.where(np.isfinite(fwhm), 2.0 * err, np.nan))
    if bis is not None:
        bis = np.asarray(bis, float) * 1000.0
        if fwhm is not None:
            with np.errstate(invalid='ignore'):
                bis = np.where(np.abs(bis) > BIS_MOST * fwhm, np.nan, bis)
        out['bis'] = (bis, np.where(np.isfinite(bis), 2.0 * err, np.nan))
    if contrast is not None and fwhm is not None:
        contrast = np.asarray(contrast, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out['contrast'] = (contrast, np.abs(contrast) * 2.0 * err / fwhm)
    return {key: val for key, val in out.items()
            if np.any(np.isfinite(val[0]))}


def _number(value: Any) -> float:
    """a float, nan when it is not one"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def series(rows: List[Dict[str, Any]], name: Optional[str] = None
           ) -> List[Tuple[RVData, Dict[str, Any]]]:
    """
    The datasets of a table of exposures: one per reduction, the
    exposures on a star with the mask most of them have (two masks are
    two zero points), a velocity and an error

    :param rows: list of dict, from exposures()
    :param name: str or None, the name of the star

    Each has the indicators of its CCF (ccf_indicators): its width and
    contrast, and its bisector span in the newer reduction

    :return: list of (RVData [m/s, BJD - 2400000], dict: reduction, key,
             reference, label, n, mask, files (those kept, in the order of
             the velocities), version, objects, left (why exposures were
             left out, in words))
    """
    out = []
    for red, about in REDUCTIONS.items():
        mine = [row for row in rows if row['reduction'] == red
                and row.get('rv') not in ('', None)]
        if not mine:
            continue
        bjd = np.array([_number(row['bjd']) for row in mine])
        vel = np.array([_number(row['rv']) for row in mine])
        err = np.array([_number(row['err']) for row in mine])
        mask = np.array([str(row.get('mask') or '') for row in mine])
        star = np.array([str(row.get('dpr') or '').upper().startswith('STAR')
                         for row in mine])
        good = (np.isfinite(bjd + vel + err) & (err > 0) & star
                & np.array([str(row.get('qc', '')) not in ('0', '0.0')
                            for row in mine]))
        left = []
        if np.sum(~good):
            left.append(f'{int(np.sum(~good))} with no velocity, no error '
                        f'or not of a star')
        if not np.any(good):
            continue
        kinds, counts = np.unique(mask[good], return_counts=True)
        best = str(kinds[np.argmax(counts)])
        other = good & (mask != best)
        if np.any(other):
            left.append(f'{int(np.sum(other))} with another mask than '
                        f'{best}')
        good &= mask == best
        names = np.array([str(row.get('object') or '') for row in mine])
        kinds, counts = np.unique(names[good], return_counts=True)
        main = np.median(vel[good & (names == kinds[np.argmax(counts)])])
        for kind in kinds:
            sel = good & (names == kind)
            if abs(np.median(vel[sel]) - main) > OTHER:
                left.append(f'{int(np.sum(sel))} as {kind}, of another '
                            f'star by their velocity')
                good &= ~sel
        far = good & (np.abs(vel - np.median(vel[good])) > FAR)
        if np.any(far):
            left.append(f'{int(np.sum(far))} more than {FAR:g} km/s from '
                        f'the others')
        good &= ~far
        if not np.any(good):
            continue
        order = np.argsort(bjd[good])
        version = [str(row.get('version') or '') for row, ok
                   in zip(mine, good) if ok]
        more = ccf_indicators(
            err[good][order] * 1000.0,
            *(np.array([_number(row.get(key)) for row in mine])[good][order]
              for key in ('fwhm', 'contrast', 'bis')))
        data = RVData(bjd[good][order] - 2400000.0,
                      vel[good][order] * 1000.0, err[good][order] * 1000.0,
                      inst=np.full(int(np.sum(good)), about['label']),
                      name=name, indicators=more)
        out.append((data, dict(
            about, reduction=red, n=int(data.n), mask=best,
            files=[str(mine[it]['file'])
                   for it in np.where(good)[0][order]],
            version=max(set(version), key=version.count),
            objects=sorted({str(row['object']) for row, ok
                            in zip(mine, good) if ok}),
            left='; '.join(left))))
    return out

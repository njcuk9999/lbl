"""
LBL velocities of a star from the public spectra of the archives: the
spectra fetched, and lbl-rapide (the branch of LBL made faster; Artigau
et al. 2022) run on them.

koloa never imports LBL: LBL needs versions of its own (Python 3.12,
numpy 2.3, numba, astropy 7), in an environment of its own (lbl-rapide by
default). koloa writes the parameters of the run in a short script, as one
writes a wrap file, and runs it with that Python, in a process of its own.
The rdb LBL makes is kept in the folder of the star, under its name, one
file per dataset koloa knows (archives/GJ_436/lbl/GJ_436_HARPSN_lbl.rdb;
SOPHIE and SOPHIE+ apart), to be given to the analysis as a file; the
names of their datasets are in labels.json beside them.

    koloa "GJ 436" --lbl HARPSN SOPHIE --archives archives --lbl-dir lbl_data

An instrument is one or more groups, a run of LBL each (its template, its
mask): the spectra of one setting (SOPHIE's HR and HE apart). Each class
of LBL has a folder of its own in lbl-dir (lbl-dir/HARPSN_ESO,
lbl-dir/SOPHIE_None), with the models of LBL shared between them
(lbl-dir/models, a few hundred MB, fetched by LBL the first time).

The instruments:

HARPSN: the espdr S2D_A files of the archive of the TNG (blaze corrected,
on a barycentric wavelength grid: LBL's HARPSN ESO source, no calibration
file), 10 MB each, of the exposures koloa keeps of the star from their
CCFs (koloa.tng.series: those on the star, with its mask, the companion of
a binary left out by its velocity); the telluric lines cleaned by LBL
first.

SOPHIE: the e2ds of the archive of the OHP (1.3 MB each once kept), of the exposures
koloa keeps from the table of its CCFs (koloa.observatories.sophie: past
their year of embargo, a programme still protected left out), HR and HE
apart; the archive puts the blaze beside the spectrum in one file: koloa
writes it as the calibration file LBL reads (calib/<seq>_blaze_A.fits,
named in the header). SOPHIE and SOPHIE+ (the fibres of June 2011) are
two datasets of the rdb.

SPIROU: the t.fits of APERO at the CADC (the telluric lines taken out by
APERO; 18 MB each), those public, pointed at where the star was; LBL's
SPIROU CADC source reads their blaze in them.

NIRPS: the 2D spectra of ESO (in the ancillary tar of each product, its
files fetched alone by Range requests): the S2D with its blaze divided by
ESO's model of the telluric lines (S2D_TELL_SPECTRUM_A: the S2D of ESO are
not corrected for them, and LBL does not clean them for NIRPS), the lines
deeper than TELLURIC_FLOOR left out, and the blaze of the night as the
calibration file the header names; HE and HA apart (LBL's NIRPS_HE and
NIRPS_HA ESO).

HARPS: the e2ds of ESO's HARPS pipeline (the DRS that made its archive:
the ANCILLARY.HARPSTAR tar of each spectrum, its e2ds fetched alone by Range
requests, 1.4 MB), LBL's HARPS ORIG source. ESO does not distribute the
blaze of the night it names (HIERARCH ESO DRS BLAZE FILE): koloa writes one
under that name, the smooth envelope of each order of the night's first
spectrum (empirical_blaze), which LBL normalises per order and whose
residual its line-by-line fit absorbs; the telluric lines cleaned by LBL
first. HAM and EGGS apart; HARPS03 and HARPS15 (the fibres of June 2015)
are two datasets.

ESPRESSO: the S2D_A of ESO (blaze corrected, 65 MB each), a run per mode
and binning (HR 2x1, UHR...), the telluric lines cleaned by LBL first;
ESPRESSO18 and ESPRESSO19 (the intervention of 2019) are two datasets.

Created on 2026-10-14

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module lblrun): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import gzip
import io
import json
import os
import re
import shutil
import time as _time
import urllib.error
import urllib.request
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from . import observatories as obs
from . import tng
from .log import log
from .paths import USER_AGENT
from .stars import resolve


# =============================================================================
# Define variables
# =============================================================================
#: the files of the archive of the TNG, and the e2ds of SOPHIE's
TNG_FILES = 'http://archives.ia2.inaf.it/files/tng/'


SOPHIE_E2DS = ('http://atlas.obs-hp.fr/sophie/sophie.cgi?c=i&a=mime:'
               'application/fits&o=sophie:%5Be2ds,{seq}%5D')


#: the instruments: what they are called, and about how large one of
#:    their spectra is once fetched [MB]
INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    'HARPSN': dict(label='HARPS-N', size=9.8),
    'SOPHIE': dict(label='SOPHIE', size=4.2),
    'SPIROU': dict(label='SPIRou', size=18.4),
    'NIRPS': dict(label='NIRPS', size=24.0),
    'HARPS': dict(label='HARPS', size=1.5),
    'ESPRESSO': dict(label='ESPRESSO', size=65.0),
}


#: the least transmission of ESO's telluric model kept in a NIRPS
#:    spectrum; where the model is lower, the pixel is left out
TELLURIC_FLOOR = 0.5


#: the bytes of a tar of ESO read first: the header of its first file
HEAD_BYTES = 512 + 2880 * 24


# =============================================================================
# Define functions
# =============================================================================
def object_name(target: str) -> str:
    """the name of the star for LBL (letters and digits: GJ436)"""
    return re.sub(r'[^A-Za-z0-9]+', '', str(target)) or 'star'


def folder_name(target: str) -> str:
    """the folder of a star in the cache: its name as given, spaces as
    underscores"""
    return re.sub(r'[^A-Za-z0-9+\-.]+', '_', target.strip()).strip('_')


def identify(target: str) -> Dict[str, Any]:
    """the star as SIMBAD knows it, with how it moves (stars.resolve)"""
    return resolve(target)


def _star(ident: Dict[str, Any]) -> Dict[str, Any]:
    """the star as the archives are asked about it"""
    return dict(ra=ident['ra'], dec=ident['dec'], pmra=ident.get('pmra'),
                pmdec=ident.get('pmdec'),
                names=[ident.get('main') or ''] + list(
                    ident.get('aliases') or []))


# -----------------------------------------------------------------------------
# Fetching a file, or a part of one
# -----------------------------------------------------------------------------
def _download(url: str, path: str, timeout: float = 300.0,
              gunzip: bool = False) -> str:
    """a file of an archive, gunzipped if asked, written whole or not at
    all; '' or why not"""
    part = path + '.part'
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={'User-Agent': USER_AGENT}),
                timeout=timeout) as response, open(part, 'wb') as handle:
            source = gzip.GzipFile(fileobj=response) if gunzip else response
            shutil.copyfileobj(source, handle, 1 << 20)
        os.replace(part, path)
        return ''
    except (urllib.error.URLError, TimeoutError, OSError, EOFError) as err:
        if os.path.exists(part):
            os.remove(part)
        return f'{type(err).__name__}: {err}'


def _range(url: str, start: int, size: int, timeout: float = 300.0
           ) -> bytes:
    """bytes of a file of an archive (an HTTP Range request)"""
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={'User-Agent': USER_AGENT, 'Range':
                          f'bytes={start}-{start + size - 1}'}),
            timeout=timeout) as response:
        data = response.read()
    if len(data) != size:
        raise OSError(f'{len(data)} bytes of {size} ({url})')
    return data


def tar_members(url: str) -> List[Tuple[str, int, int]]:
    """
    The files of a tar of an archive, read from their headers alone (a
    Range request of 512 bytes each): name, where its data start, size

    :return: list of (name, offset, size)
    """
    out, where = [], 0
    for _ in range(1000):
        head = _range(url, where, 512)
        if not head.strip(b'\0'):
            break
        size = int(head[124:136].rstrip(b'\0 ').decode() or '0', 8)
        name = head[:100].rstrip(b'\0').decode('utf8', 'replace')
        kind = head[156:157]
        if kind in (b'L', b'x'):
            # a long name (GNU) or the attributes of the next (pax)
            extra = _range(url, where + 512, size).rstrip(b'\0').decode(
                'utf8', 'replace')
            where += 512 + -(-size // 512) * 512
            head = _range(url, where, 512)
            size = int(head[124:136].rstrip(b'\0 ').decode() or '0', 8)
            name = head[:100].rstrip(b'\0').decode('utf8', 'replace')
            if kind == b'L':
                name = extra
            for line in extra.splitlines():
                key, _, val = line.partition(' ')[2].partition('=')
                if key == 'path':
                    name = val
                elif key == 'size':
                    size = int(val)
        else:
            prefix = head[345:500].rstrip(b'\0').decode('utf8', 'replace')
            name = f'{prefix}/{name}' if prefix else name
        out.append((name.lstrip('./'), where + 512, size))
        where += 512 + -(-size // 512) * 512
    return out


def _member(members: List[Tuple[str, int, int]], *ends: str
            ) -> Optional[Tuple[str, int, int]]:
    """the file of a tar whose name ends one of the ways given"""
    return next((one for end in ends for one in members
                 if one[0].endswith(end)), None)


def _safe(name: str) -> str:
    """a name of a file with only plain characters (LBL asks for them)"""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', name)


# -----------------------------------------------------------------------------
# Which exposures are the star's
# -----------------------------------------------------------------------------
def _designation(name: str) -> Optional[Tuple[str, str]]:
    """a name as a catalogue and a number (GJ 699, Gl699, GJ699-HE: GJ and
    699; TOI-1452: TOI and 1452; GJ 15 B: GJ and 15B), None when it is not
    one (K4_02, BARNARD)"""
    found = re.match(r'^\s*([A-Za-z]+)[\s_-]*(\d+)(?:[\s_.-]*([A-Ca-c])\b)?',
                     str(name))
    if not found:
        return None
    family = found[1].upper()
    family = {'GL': 'GJ', 'GLIESE': 'GJ'}.get(family, family)
    return family, found[2].lstrip('0') + (found[3] or '').upper()


def _other(name: str, mine: Dict[str, set]) -> bool:
    """whether the name of a target is that of another star: of a
    catalogue the star is in, under another number (GJ 15 B beside GJ 15
    A); a name of no catalogue of the star, or none, is not (an observer's
    own: K4_02)"""
    found = _designation(name)
    if found is None or found[0] not in mine:
        return False
    number = found[1]
    for one in mine[found[0]]:
        # a component given on one side only is the same star
        if number == one or number.rstrip('ABC') == one or \
                one.rstrip('ABC') == number:
            return False
    return True


def _named(found: List[Dict[str, Any]], star: Dict[str, Any],
           what: str) -> List[Dict[str, Any]]:
    """
    The exposures pointed at a star that are not of another star by their
    name: one of a catalogue the star is in, under another number (the
    companion of a binary), would spoil the template of LBL; the
    observers' own names are kept (GJ699-HE, Gl699)

    :param found: the exposures, each with its target
    """
    mine: Dict[str, set] = {}
    for name in star.get('names') or []:
        got = _designation(name)
        if got:
            mine.setdefault(got[0], set()).add(got[1])
    kept = [one for one in found if not _other(one['target'], mine)]
    gone = sorted({one['target'] for one in found} -
                  {one['target'] for one in kept})
    if gone:
        log(f'{what}: {len(found) - len(kept)} exposures of another star '
            f'pointed nearby left out ({", ".join(gone)})', 'warn')
    return kept


def harpsn_spectra(target: str, star: Dict[str, Any], root: str
                   ) -> List[str]:
    """
    The espdr S2D_A files of HARPS-N of a star: those of the exposures on
    the star (tng.espdr_files), whose headers are kept in the star's folder
    of the cache (root/<target>/tng.TABLE)

    :return: list of str, the names of the files (gzipped, as the archive
             has them), by date
    """
    folder = os.path.join(root, folder_name(target))
    rows = tng.exposures(star, folder)
    return [name.replace('_CCF_A.fits', '_S2D_A.fits')
            for name in tng.espdr_files(rows)]


def spirou_spectra(star: Dict[str, Any]) -> List[str]:
    """
    The t.fits of SPIRou of a star at the CADC (APERO's spectra, the
    telluric lines taken out): those public, pointed at where the star was

    :return: list of str, their URIs (cadc:CFHT/<odometer>t.fits), by date
    """
    rows = obs._tap(obs.CADC_TAP, (
        "SELECT o.target_name AS target, "
        "o.targetPosition_coordinates_cval1 AS ra, "
        "o.targetPosition_coordinates_cval2 AS dec, p.time_bounds_lower AS "
        "mjd, p.dataRelease AS release, a.uri AS uri FROM caom2.Observation "
        "AS o JOIN caom2.Plane AS p ON o.obsID = p.obsID JOIN "
        "caom2.Artifact AS a ON a.planeID = p.planeID WHERE o.collection = "
        "'CFHT' AND o.instrument_name = 'SPIRou' AND p.productID LIKE '%t' "
        "AND a.contentType = 'application/fits' AND "
        f"INTERSECTS(CIRCLE('ICRS', {float(star['ra']):.6f}, "
        f"{float(star['dec']):.6f}, {obs._reach(star):.5f}), "
        "p.position_bounds) = 1"), 'CFHT archive (CADC)')
    now = _time.strftime('%Y-%m-%dT%H:%M:%S', _time.gmtime())
    found = [dict(mjd=obs._num(row['mjd']), uri=str(row['uri']),
                  target=str(row.get('target') or '')) for row in rows
             if str(row.get('release') or '9') <= now
             and str(row['uri']).endswith('t.fits')
             and obs._pointed(obs._num(row['ra']), obs._num(row['dec']),
                              obs._num(row['mjd']), star)[0]]
    kept = _named(found, star, 'SPIRou')
    return list(dict.fromkeys(one['uri'] for one in sorted(
        kept, key=lambda one: one['mjd'])))


def eso_spectra(star: Dict[str, Any], ins: str,
                category: str = 'ANCILLARY.2DECHELLE.TAR'
                ) -> List[Dict[str, Any]]:
    """
    The exposures of an instrument of ESO (NIRPS, ESPRESSO; HARPS with
    category ANCILLARY.HARPSTAR, its DRS's tar) of a star with their 2D
    spectra (the ancillary tar of their product), public, pointed at where
    the star was; with what the header of the first file of each tar says
    (its mode, its binning), one Range request a tar

    :return: list of dict: id (the tar), target, mjd, mode, bin, object, by
             date
    """
    ra, dec = float(star['ra']), float(star['dec'])
    rows = obs._tap(obs.ESO_OBS, (
        "SELECT o.target_name AS target, o.t_min AS mjd, o.obs_release_date "
        "AS release, o.s_ra AS ra, o.s_dec AS dec, f.archive_id AS tar FROM "
        "ivoa.ObsCore AS o JOIN phase3v2.product_files AS f ON f.product_id"
        " = o.dp_id WHERE CONTAINS(POINT('ICRS', o.s_ra, o.s_dec), "
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {obs._reach(star):.5f}))=1 AND "
        f"o.instrument_name = '{ins}' AND f.eso_category = "
        f"'{category}'"), 'ESO archive', 'json')
    now = _time.strftime('%Y-%m-%dT%H:%M:%S', _time.gmtime())
    found = {}
    for row in rows:
        if str(row.get('release') or '9') > now or not obs._pointed(
                obs._num(row['ra']), obs._num(row['dec']),
                obs._num(row['mjd']), star)[0]:
            continue
        found[str(row['tar'])] = dict(id=str(row['tar']),
                                      target=str(row['target']),
                                      mjd=obs._num(row['mjd']))

    def head(one):
        try:
            raw = _range(obs.ESO_FILE + one['id'], 0, HEAD_BYTES, 120.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        cards = tng.header(raw[512:])
        return dict(one, mode=str(cards.get('ESO INS MODE') or '').strip(),
                    bin=f'{cards.get("ESO DET BINX", 1)}'
                        f'{cards.get("ESO DET BINY", 1)}',
                    object=str(cards.get('OBJECT') or ''))
    with ThreadPoolExecutor(max_workers=8) as pool:
        out = [one for one in pool.map(head, _named(list(found.values()),
                                                     star, ins)) if one]
    return sorted(out, key=lambda one: one['mjd'])


# -----------------------------------------------------------------------------
# One spectrum of each instrument into the data folder: '' or why not
# -----------------------------------------------------------------------------
def _get_tng(one: Dict[str, Any], science: str, calib: str) -> str:
    """an S2D_A of the TNG, gunzipped"""
    return _download(TNG_FILES + one['id'], os.path.join(science, one['name']),
                     gunzip=True)


def _get_sophie(one: Dict[str, Any], science: str, calib: str) -> str:
    """an e2ds of SOPHIE: its spectrum (fibre A) as the science file, its
    blaze as the calibration file its header names; a spectrum of a
    programme still protected (no BERV, its date rounded) left out"""
    try:
        from astropy.io import fits
    except ImportError:
        return 'astropy is needed to read the spectra of SOPHIE'
    try:
        with urllib.request.urlopen(urllib.request.Request(
                SOPHIE_E2DS.format(seq=one['id']),
                headers={'User-Agent': USER_AGENT}), timeout=300) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return f'{type(err).__name__}: {err}'
    if not raw:
        return 'not public yet'
    try:
        with fits.open(io.BytesIO(raw)) as hdul:
            head = hdul[0].header.copy()
            bjd = head.get('HIERARCH OHP DRS BJD')
            if head.get('HIERARCH OHP DRS BERV') is None or bjd is None \
                    or float(bjd) == int(float(bjd)):
                return 'of a programme still protected (no BERV)'
            blaze = f'{one["id"]}_blaze_A.fits'
            head['HIERARCH OHP DRS BLAZE FILE'] = blaze
            fits.PrimaryHDU(hdul['BLAZE_A'].data).writeto(
                os.path.join(calib, blaze), overwrite=True)
            path = os.path.join(science, one['name'])
            fits.PrimaryHDU(hdul[0].data, head).writeto(path + '.part',
                                                        overwrite=True)
            os.replace(path + '.part', path)
    except (OSError, KeyError, ValueError) as err:
        return f'{type(err).__name__}: {err}'
    return ''


def _get_cadc(one: Dict[str, Any], science: str, calib: str) -> str:
    """a t.fits of SPIRou at the CADC (a signed address it redirects to);
    one of an older reduction, whose FluxAB LBL cannot read (no MJDMID),
    left out"""
    path = os.path.join(science, one['name'])
    why = _download(obs.CADC_FILE + one['id'], path)
    if why:
        return why
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            head = hdul['FluxAB'].header
            if 'MJDMID' not in head:
                version = hdul[0].header.get('VERSION', '?')
                os.remove(path)
                return f'APERO {version}: an older reduction LBL does not read'
    except ImportError:
        pass
    except (OSError, KeyError) as err:
        os.remove(path)
        return f'{type(err).__name__}: {err}'
    return ''


def _get_nirps(one: Dict[str, Any], science: str, calib: str) -> str:
    """
    A spectrum of NIRPS from ESO: its S2D with the blaze (S2D_BLAZE_A)
    divided by ESO's model of the telluric lines (S2D_TELL_SPECTRUM_A),
    the pixels whose transmission is below TELLURIC_FLOOR left out, as the
    science file; the blaze of its night, as the calibration file its
    header names
    """
    try:
        from astropy.io import fits
        import numpy as np
    except ImportError:
        return 'astropy is needed to read the spectra of ESO'
    url = obs.ESO_FILE + one['id']
    try:
        members = tar_members(url)
        flux = _member(members, '_S2D_BLAZE_A.fits')
        tell = _member(members, '_S2D_TELL_SPECTRUM_A.fits')
        # the blaze of the night (not the S2D with its blaze)
        blaze = next((part for part in members
                      if part[0].endswith('_BLAZE_A.fits')
                      and '_S2D_' not in part[0]), None)
        if not flux or not tell or not blaze:
            return 'no S2D_BLAZE_A, telluric model or blaze in its tar'
        bname = os.path.basename(blaze[0])
        if not os.path.exists(os.path.join(calib, bname)):
            # the blaze of a night, which several spectra share: each its
            #   own part, the last one written wins
            part = os.path.join(calib, f'{bname}.{uuid.uuid4().hex}.part')
            with open(part, 'wb') as handle:
                handle.write(_range(url, blaze[1], blaze[2]))
            os.replace(part, os.path.join(calib, bname))
        with fits.open(io.BytesIO(_range(url, flux[1], flux[2]))) as hdul, \
                fits.open(io.BytesIO(_range(url, tell[1], tell[2]))) as model:
            trans = np.asarray(model['SCIDATA'].data, float)
            bad = ~(trans > TELLURIC_FLOOR)
            with np.errstate(divide='ignore', invalid='ignore'):
                for ext in ('SCIDATA', 'ERRDATA'):
                    data = np.asarray(hdul[ext].data, float) / trans
                    data[bad] = np.nan
                    hdul[ext].data = data.astype(np.float32)
            hdul[0].header['HIERARCH LBL TELL FLOOR'] = (
                TELLURIC_FLOOR, 'divided by S2D_TELL_SPECTRUM_A')
            path = os.path.join(science, one['name'])
            hdul.writeto(path + '.part', overwrite=True)
            os.replace(path + '.part', path)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError,
            ValueError) as err:
        return f'{type(err).__name__}: {err}'
    return ''


def empirical_blaze(e2ds, chunks: int = 32, degree: int = 4,
                    percentile: float = 90.0):
    """
    A blaze from a spectrum, for a pipeline whose archive does not give it
    (HARPS at ESO): for each order, the given percentile of the flux in
    chunks along it, a polynomial through them, kept above a hundredth of
    its peak. Smooth: lines and molecular bands lower a chunk's percentile
    a little, never the shape. Its scale does not matter (LBL normalises a
    blaze per order)

    :param e2ds: np.ndarray, (orders, pixels)
    :return: np.ndarray, the same shape, float32
    """
    import numpy as np
    data = np.asarray(e2ds, dtype=float)
    out = np.ones_like(data)
    pixels = np.arange(data.shape[1])
    edges = np.linspace(0, data.shape[1], chunks + 1).astype(int)
    for order in range(data.shape[0]):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            part = data[order, lo:hi]
            part = part[np.isfinite(part)]
            if part.size > 10:
                xs.append(0.5 * (lo + hi))
                ys.append(np.percentile(part, percentile))
        if len(xs) <= degree or not np.nanmax(ys) > 0:
            continue
        x0 = data.shape[1] / 2.0
        fit = np.polyval(np.polyfit((np.array(xs) - x0) / x0, ys, degree),
                         (pixels - x0) / x0)
        out[order] = np.maximum(fit, 0.01 * np.nanmax(fit))
    return out.astype(np.float32)


def _get_harps(one: Dict[str, Any], science: str, calib: str) -> str:
    """
    A spectrum of HARPS from ESO: the e2ds of its DRS tar, as it is, and
    the blaze of its night where the header names it (empirical_blaze of
    this spectrum when the night has none yet: ESO does not give it)
    """
    try:
        from astropy.io import fits
    except ImportError:
        return 'astropy is needed to read the spectra of ESO'
    url = obs.ESO_FILE + one['id']
    try:
        members = tar_members(url)
        flux = _member(members, '_e2ds_A.fits')
        if not flux:
            return 'no e2ds_A in its tar'
        raw = _range(url, flux[1], flux[2])
        with fits.open(io.BytesIO(raw)) as hdul:
            bname = str(hdul[0].header.get('HIERARCH ESO DRS BLAZE FILE')
                        or '').strip()
            if not bname:
                return 'no blaze named in its header'
            blaze = os.path.join(calib, os.path.basename(bname))
            if not os.path.exists(blaze):
                head = fits.Header()
                head['HIERARCH LBL BLAZE'] = (
                    'empirical', 'envelope of ' + flux[0][-40:])
                part = f'{blaze}.{uuid.uuid4().hex}.part'
                fits.PrimaryHDU(empirical_blaze(hdul[0].data),
                                header=head).writeto(part)
                os.replace(part, blaze)
            # the DRS writes cards astropy cannot write back (ESO INS TEMP1
            #   RMS...): fixed here, or LBL stops on the first file it
            #   rewrites (its telluric cleaning)
            path = os.path.join(science, one['name'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                hdul.verify('silentfix+ignore')
                hdul.writeto(path + '.part', overwrite=True,
                             output_verify='silentfix+ignore')
        os.replace(path + '.part', path)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError,
            ValueError) as err:
        return f'{type(err).__name__}: {err}'
    return ''


def _get_espresso(one: Dict[str, Any], science: str, calib: str) -> str:
    """a spectrum of ESPRESSO from ESO: the S2D_A of its tar (blaze
    corrected), as it is"""
    url = obs.ESO_FILE + one['id']
    try:
        members = tar_members(url)
        flux = _member(members, '_S2D_A.fits') or next(
            (part for part in members
             if os.path.basename(part[0]).startswith('ES_S2DA')), None)
        if not flux:
            return 'no S2D_A in its tar'
        path = os.path.join(science, one['name'])
        with open(path + '.part', 'wb') as handle:
            handle.write(_range(url, flux[1], flux[2], 900.0))
        os.replace(path + '.part', path)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return f'{type(err).__name__}: {err}'
    return ''


# -----------------------------------------------------------------------------
# The runs of LBL of an instrument for a star
# -----------------------------------------------------------------------------
def groups(target: str, inst: str, root: str,
           ident: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    The runs of LBL of an instrument for a star: for each, the LBL
    instrument and data source that read its spectra, its file pattern,
    the parameters its wrap sets, its spectra (id in the archive, name
    here) and how one of them is fetched

    :param target: str, the star as it was asked for
    :param inst: str, one of INSTRUMENTS
    :param root: str, a folder of the cache for what a search keeps
    :param ident: dict or None, the star (identify(target) when None)

    :return: list of dict: key (the setting, '' for the main one), obj (the
             object name LBL gets), lbl, source, input, params, spectra
             (list of dict: id, name), get (function: one spectrum, the
             science folder, the calibration folder; '' or why not)
    """
    obj = object_name(target)
    star = _star(ident or identify(target))
    if inst == 'HARPSN':
        names = harpsn_spectra(target, star, root)
        return [dict(key='', obj=obj, lbl='HARPSN', source='ESO',
                     input='*S2D_A.fits', params={},
                     spectra=[dict(id=name, name=name[:-3])
                              for name in names],
                     get=_get_tng)]
    if inst == 'SOPHIE':
        found = obs.sophie_spectra(star)
        return [dict(key=key, obj=obj + key, lbl='SOPHIE', source='None',
                     input='*_e2ds_A.fits', params={},
                     spectra=[dict(id=seq, name=f'{seq}_e2ds_A.fits')
                              for seq in found[mode]],
                     get=_get_sophie)
                for mode, key in (('HR', ''), ('HE', 'HE'))
                if found.get(mode)]
    if inst == 'SPIROU':
        uris = spirou_spectra(star)
        return [dict(key='', obj=obj, lbl='SPIROU', source='CADC',
                     input='*t.fits', params={},
                     spectra=[dict(id=uri, name=uri.rsplit('/', 1)[-1])
                              for uri in uris],
                     get=_get_cadc)]
    if inst == 'NIRPS':
        found = eso_spectra(star, 'NIRPS')
        out = []
        for mode, key in (('HE', ''), ('HA', 'HA')):
            mine = [one for one in found if one['mode'] == mode]
            if mine:
                out.append(dict(
                    key=key, obj=obj + key, lbl=f'NIRPS_{mode}', source='ESO',
                    input='*TCORR_A.fits',
                    params=dict(BLAZE_CORRECTED=False),
                    spectra=[dict(id=one['id'], name=_safe(
                        f'r.NIRPS.{one["id"]}_S2D_BLAZE_TCORR_A.fits'))
                        for one in mine],
                    get=_get_nirps))
        return out
    if inst == 'HARPS':
        found = eso_spectra(star, 'HARPS', 'ANCILLARY.HARPSTAR')
        out = []
        for mode, key in (('HAM', ''), ('EGGS', 'EGGS')):
            mine = [one for one in found
                    if (one['mode'].upper() == 'EGGS') == (mode == 'EGGS')]
            if mine:
                out.append(dict(
                    key=key, obj=obj + key, lbl='HARPS', source='ORIG',
                    input='*_e2ds_A.fits', params={},
                    spectra=[dict(id=one['id'], name=_safe(
                        f'HARPS.{one["id"]}_e2ds_A.fits')) for one in mine],
                    get=_get_harps))
        return out
    if inst == 'ESPRESSO':
        found = eso_spectra(star, 'ESPRESSO')
        kinds = [one['mode'].replace('SINGLE', '').replace('MULTI', '')
                 + one['bin'] for one in found]
        out = []
        if kinds:
            # the setting most spectra have is the main run
            main = max(set(kinds), key=kinds.count)
            for kind in sorted(set(kinds), key=lambda one: (one != main, one)):
                key = '' if kind == main else kind
                out.append(dict(
                    key=key, obj=obj + key, lbl='ESPRESSO', source='None',
                    input='*S2D_A.fits', params={},
                    spectra=[dict(id=one['id'], name=_safe(
                        f'ESPRESSO.{one["id"]}_S2D_A.fits'))
                        for one, it in zip(found, kinds) if it == kind],
                    get=_get_espresso))
        return out
    raise ValueError(f'no instrument {inst} for LBL: {", ".join(INSTRUMENTS)}')

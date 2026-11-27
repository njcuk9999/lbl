#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Equivalent widths of the activity lines, in the formalism of Schofer et al.
(2019, A&A 623, A44, section 3.2): each line has an integration window and
two pseudo-continuum ranges, one on each side, fixed in the catalog
(lbl/resources/activity_lines.yaml, vacuum wavelengths, the star's rest
frame). Every science spectrum gets them, with no option to set:

    measure()     lbl_compute, once the velocity has converged: for each
                  line, in the order where it falls closest to the centre,
                  its equivalent width against the template. Written to the
                  ACTIVITY extension of the lblrv file, one row per line of
                  the catalog: the catalog's columns and MEASURED
    add_to_rdb()  lbl_compile: the same, as columns of the rdb (one row per
                  spectrum), <column>_<line> for the lines measured at least
                  once, the line's space as an underscore: EW_He_1.08333,
                  sEW_He_1.08333, dEW_He_1.08333, EWT_He_1.08333...
                  sEW is EW's error (and dEW's), so the rdb per night (lbl2)
                  bins EW and dEW weighted by it

THE MEASUREMENT
---------------
In the star's frame (the template's, at the velocity the RVs converged
to), with T the template, S the spectrum (before LBL's sigma clipping, which
would cut a flare out) and B the blaze:

    PC(lam)  the template's pseudo-continuum: the 90th percentile of T in
             each pseudo-continuum range, joined by a straight line between
             the ranges' centres (Schofer et al. 2019, eq. 2: robust to the
             noise and to the lines within the ranges). On the template
             itself (its own pixels, merged over the orders: an order of the
             spectrum need not hold the ranges)
    q(lam)   the spectrum's level against the template: the median of
             S / (T B) in each range, a straight line between the two. A
             range the order does not hold (its valid pixels end before it)
             is replaced, for q only, by as wide a stretch of the order's
             valid pixels nearest to it on that side of the window (half the
             window's width from it at least). S / (q B) is the spectrum
             brought to the template's level, as Schofer et al. normalise
             both spectra to their pseudo-continuum ranges

    EWT  = sum over the window of a_i (1 - T_i / PC_i), on the template's
           own pixels: the same for every spectrum
    dEW  = sum over the window of a_i (T_i - S_i / (q_i B_i)) / PC_i, on
           the order's pixels
    EW   = EWT + dEW
         ~ sum over the window of a_i (1 - S_i / (q_i B_i PC_i))

with a_i the length of pixel i inside the window (a boxcar: the overlap of
the pixel with it). In mA, positive for absorption: emission is negative.
EWT is the template's pseudo-equivalent width, dEW what the spectrum absorbs
more than the template (Schofer et al.'s pEW', the template in place of an
inactive star's spectrum), EW the spectrum's own. A window pixel that is
not valid in the spectrum counts as the template's (dEW 0 there).

    sEW (= sdEW)  the error of EW and of dEW: the noise of LBL's model (rms)
                  over the window, and that of q's level
    EWTSYS        the systematic error of EWT from the continuum: how far it
                  moves with another continuum, the 95th percentile of T
                  within SYS_HALF_WIDTH of the line's centre (LBL's before
                  2026-11-27). Not divided by the number of spectra: the
                  same for all of them. EW carries it too, dEW does not
    FRACVALID     the fraction of the window with valid pixels: below
                  MIN_FRAC_VALID, no measure; the pixels LBL's sigma
                  clipping rejected are kept (a flare would be cut out),
                  CHI2 (the residuals of the window against rms, per pixel)
                  shows them

The order is the one where the line, without the BERV, falls closest to the
centre: the same order for every spectrum, as long as the window is inside
it and MIN_FRAC_VALID of the window is valid; the next closest otherwise.
The published ranges can be farther apart than an order is long (Pa gamma's,
6.5 nm, against the 11 nm a NIRPS order holds of valid pixels): the template
holds them all.

The windows hold the whole line, wide enough for the broadest emission
(Schofer et al. 2019); the catalog says where each window and range comes
from (lbl/doc/ACTIVITY.md).

Created on 2026-10-22, Schofer's formalism on 2026-11-27

@author: artigau
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from lbl.core import io
from lbl.core import math as mp

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.science.activity_lines.py'
# get log
log = io.log
#: the catalog of lines
CATALOG = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'resources', 'activity_lines.yaml')
#: the lblrv extension of the measurements
EXTNAME = 'ACTIVITY'
#: the columns of the catalog kept in the lblrv extension (the window and
#:   the two pseudo-continuum ranges' edges, nm, vacuum, the star's frame)
CATALOG_COLUMNS = ['NAME', 'LABEL', 'SPECIES', 'KIND', 'BAND', 'WAVE',
                   'WIN_LO', 'WIN_HI', 'CONT1_LO', 'CONT1_HI', 'CONT2_LO',
                   'CONT2_HI']
#: the columns measured on each line, and their units
MEASURED = dict(EW='mAngstrom', sEW='mAngstrom', dEW='mAngstrom',
                sdEW='mAngstrom', EWT='mAngstrom', EWTSYS='mAngstrom',
                FRACVALID='', CHI2='')
#: nm to mA
NM_TO_MA = 1e4
#: the percentile of the template in a pseudo-continuum range taken as its
#:   level (Schofer et al. 2019)
PC_PERCENTILE = 90
#: EWTSYS: the other continuum, this percentile of the template within this
#:   half width [nm] of the line's centre
SYS_PERCENTILE = 95
SYS_HALF_WIDTH = 2.5
#: least fraction of the window with valid pixels
MIN_FRAC_VALID = 0.9
#: least number of valid pixels in a pseudo-continuum range
MIN_SIDE_PIX = 3
# the catalog, read once
_LINES: Optional[List[Dict]] = None
_VERSION: Optional[int] = None


# =============================================================================
# Define functions
# =============================================================================
def _interval(value, what: str) -> Tuple[float, float]:
    """[lo, hi] of the catalog, checked"""
    lo, hi = (float(v) for v in value)
    if not lo < hi:
        raise ValueError('{0}: [{1}, {2}] is not an interval '
                         '({3})'.format(what, lo, hi, CATALOG))
    return lo, hi


def load_lines() -> List[Dict]:
    """
    The lines of the catalog, in its order

    :return: list of dicts, one per line: NAME, LABEL, SPECIES, KIND, BAND,
             WAVE [nm, vacuum], WINDOW (lo, hi), CONT ((lo, hi), (lo, hi)),
             WIDTH (the window's) [nm], REF, URL
    """
    global _LINES, _VERSION
    if _LINES is None:
        with open(CATALOG, 'r') as yfile:
            content = yaml.safe_load(yfile)
        lines = []
        for name, line in content['lines'].items():
            window = _interval(line['window'], name + ' window')
            ranges = [_interval(r, name + ' continuum')
                      for r in line['continuum']]
            if len(ranges) != 2 or not (ranges[0][1] <= window[0] and
                                        ranges[1][0] >= window[1]):
                emsg = ('{0}: two pseudo-continuum ranges are needed, one '
                        'blueward of the window and one redward ({1})')
                raise ValueError(emsg.format(name, CATALOG))
            lines.append(dict(NAME=name, LABEL=line['label'],
                              SPECIES=line['species'], KIND=line['kind'],
                              BAND=line['band'], WAVE=float(line['wave']),
                              WINDOW=window, CONT=tuple(ranges),
                              WIDTH=window[1] - window[0],
                              REF=line.get('ref', ''),
                              URL=line.get('url', '')))
        _LINES = lines
        _VERSION = int(content['version'])
    return _LINES


def catalog_version() -> int:
    """The version of the catalog (its definitions: a measure made with
    another one is made again)"""
    load_lines()
    return _VERSION


def window(line: Dict) -> Tuple[float, float]:
    """The window a line's equivalent width is integrated over, (lo, hi)
    [nm, vacuum, the star's frame]"""
    return line['WINDOW']


def continuum(line: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """A line's two pseudo-continuum ranges, blueward and redward of its
    window, ((lo, hi), (lo, hi)) [nm, vacuum, the star's frame]"""
    return line['CONT']


def _catalog_row(line: Dict) -> Dict:
    """The catalog's columns of a line, flat (the lblrv extension's)"""
    (c1lo, c1hi), (c2lo, c2hi) = line['CONT']
    return dict(NAME=line['NAME'], LABEL=line['LABEL'],
                SPECIES=line['SPECIES'], KIND=line['KIND'], BAND=line['BAND'],
                WAVE=line['WAVE'], WIN_LO=line['WINDOW'][0],
                WIN_HI=line['WINDOW'][1], CONT1_LO=c1lo, CONT1_HI=c1hi,
                CONT2_LO=c2lo, CONT2_HI=c2hi)


def _pixel_edges(wave: np.ndarray) -> np.ndarray:
    """
    Edges of the pixels of an increasing wavelength vector, halfway between
    their centres

    :param wave: np.ndarray, the wavelengths of the pixels

    :return: np.ndarray, len(wave) + 1 edges
    """
    edges = np.empty(len(wave) + 1)
    edges[1:-1] = 0.5 * (wave[1:] + wave[:-1])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return edges


def _side_level(wave: np.ndarray, ratio: np.ndarray, noise: np.ndarray,
                band: np.ndarray):
    """
    The level of S / (T B) in one side band: its median, where it is taken
    and its error

    :return: None if too few valid pixels, else (wavelength, level, error)
    """
    band = band & np.isfinite(ratio) & np.isfinite(noise)
    if np.sum(band) < MIN_SIDE_PIX:
        return None
    # error of a median of gaussian noise: sqrt(pi / 2) times that of the
    #   mean
    error = np.sqrt(np.pi / 2) * np.median(noise[band]) / np.sqrt(np.sum(band))
    return np.mean(wave[band]), np.median(ratio[band]), error


def template_terms(line: Dict, t_wave: np.ndarray,
                   t_flux: np.ndarray) -> Optional[Dict]:
    """
    What a line's equivalent widths take from the template, on the template
    itself (its own pixels, merged over the orders, in the star's frame):
    its pseudo-continuum (the 90th percentile of the template in each
    range, at the range's centre), its equivalent width EWT in the window,
    and EWTSYS. The same for every spectrum, whatever its orders.

    :param line: dict, the line (load_lines)
    :param t_wave: np.ndarray, the template's wavelengths [nm], increasing
    :param t_flux: np.ndarray, the template, without the blaze

    :return: None if a range or the window lacks template pixels, else dict
             (x_left, pc_left, x_right, pc_right, EWT, EWTSYS) [nm, mA]
    """
    lo, hi = window(line)
    (c1lo, c1hi), (c2lo, c2hi) = continuum(line)
    good = np.isfinite(t_wave) & np.isfinite(t_flux)
    t_wave, t_flux = t_wave[good], t_flux[good]
    left = t_flux[(t_wave >= c1lo) & (t_wave <= c1hi)]
    right = t_flux[(t_wave >= c2lo) & (t_wave <= c2hi)]
    if len(left) < MIN_SIDE_PIX or len(right) < MIN_SIDE_PIX:
        return None
    out = dict(x_left=0.5 * (c1lo + c1hi), x_right=0.5 * (c2lo + c2hi),
               pc_left=float(np.percentile(left, PC_PERCENTILE)),
               pc_right=float(np.percentile(right, PC_PERCENTILE)))
    # the window on the template's pixels
    i0 = max(int(np.searchsorted(t_wave, lo)) - 2, 0)
    i1 = min(int(np.searchsorted(t_wave, hi)) + 2, len(t_wave))
    if i1 - i0 < 3:
        return None
    wave, flux = t_wave[i0:i1], t_flux[i0:i1]
    edges = _pixel_edges(wave)
    if edges[0] > lo or edges[-1] < hi:
        return None
    length = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo),
                     0, None)
    pcont = pseudo_continuum(out, wave)
    ewt = np.sum(length * (1 - flux / pcont))
    near = np.abs(t_wave - line['WAVE']) < SYS_HALF_WIDTH
    cont_sys = np.percentile(t_flux[near], SYS_PERCENTILE)
    ewt_sys = np.sum(length * (1 - flux / cont_sys))
    out['EWT'] = float(ewt * NM_TO_MA)
    out['EWTSYS'] = float(abs(ewt_sys - ewt) * NM_TO_MA)
    return out


def pseudo_continuum(terms: Dict, wave: np.ndarray) -> np.ndarray:
    """The template's pseudo-continuum at these wavelengths: a straight line
    between its levels at the two ranges' centres (template_terms)"""
    return terms['pc_left'] + (terms['pc_right'] - terms['pc_left']) * (
            wave - terms['x_left']) / (terms['x_right'] - terms['x_left'])


def _side_band(wave: np.ndarray, usable: np.ndarray, rng: Tuple,
               side: int, lo: float, hi: float) -> np.ndarray:
    """
    The pixels of the order where the spectrum's level q is taken on one
    side of the window: the pseudo-continuum range, or where the order does
    not hold MIN_SIDE_PIX valid pixels of it, as wide a stretch of the
    order's valid pixels nearest the range on that side (at least half the
    window's width from the window, so that the line's wings stay out)

    :param wave: np.ndarray, the order's wavelengths, increasing
    :param usable: np.ndarray, bool, the pixels of the order with a ratio
    :param rng: (lo, hi), the range [nm]
    :param side: int, -1 blueward, +1 redward of the window
    :param lo, hi: float, the window [nm]

    :return: np.ndarray, bool, the pixels (none when the order has none
             there)
    """
    band = usable & (wave >= rng[0]) & (wave <= rng[1])
    if np.sum(band) >= MIN_SIDE_PIX:
        return band
    width, gap = rng[1] - rng[0], 0.5 * (hi - lo)
    if side < 0:
        pool = usable & (wave <= lo - gap)
        if not pool.any():
            return pool
        start = max(wave[pool].min(), rng[0])
        return pool & (wave >= start) & (wave <= start + width)
    pool = usable & (wave >= hi + gap)
    if not pool.any():
        return pool
    end = min(wave[pool].max(), rng[1])
    return pool & (wave <= end) & (wave >= end - width)


def _measure_line(line: Dict, terms: Dict, wave: np.ndarray,
                  spectrum: np.ndarray, template: np.ndarray,
                  blaze: np.ndarray, rms: np.ndarray) -> Optional[Dict]:
    """
    The equivalent widths of one line in one order of a spectrum (see the
    module's docstring)

    :param line: dict, the line (load_lines)
    :param terms: dict, what the line takes from the template
                  (template_terms)
    :param wave: np.ndarray, the wavelengths of the order in the template's
                 frame [nm]
    :param spectrum: np.ndarray, the spectrum of the order (not clipped)
    :param template: np.ndarray, the template on the order, without the
                     blaze
    :param blaze: np.ndarray, the blaze of the order
    :param rms: np.ndarray, LBL's noise model of the order

    :return: None if the window is not in the order, or no pseudo-continuum
             range has valid pixels of the spectrum there, else the measured
             columns (dEW NaN if less than MIN_FRAC_VALID of the window is
             valid)
    """
    lo, hi = window(line)
    (c1lo, c1hi), (c2lo, c2hi) = continuum(line)
    # the finite part of the order, increasing
    finite = np.where(np.isfinite(wave))[0]
    if len(finite) < 10:
        return None
    sel = slice(finite[0], finite[-1] + 1)
    wave = wave[sel]
    spectrum = spectrum[sel]
    template, blaze, rms = template[sel], blaze[sel], rms[sel]
    if wave[-1] < wave[0]:
        wave = wave[::-1]
        spectrum = spectrum[::-1]
        template, blaze, rms = template[::-1], blaze[::-1], rms[::-1]
    edges = _pixel_edges(wave)
    # the window must be inside the order
    if edges[0] > lo or edges[-1] < hi:
        return None
    # length of each pixel inside the window (a boxcar)
    length = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo),
                     0, None)
    in_window = length > 0
    # the spectrum's level against the template, from the ranges where the
    #   order has valid pixels (both, or the one)
    with np.errstate(divide='ignore', invalid='ignore'):
        model = template * blaze
        ratio = spectrum / model
        ratio_noise = rms / model
    usable = np.isfinite(ratio) & np.isfinite(ratio_noise)
    side_l = _side_level(wave, ratio, ratio_noise,
                         _side_band(wave, usable, (c1lo, c1hi), -1, lo, hi))
    side_r = _side_level(wave, ratio, ratio_noise,
                         _side_band(wave, usable, (c2lo, c2hi), 1, lo, hi))
    if side_l is None and side_r is None:
        return None
    centre = 0.5 * (lo + hi)
    if side_l is None or side_r is None:
        _, level, level_err = side_l if side_r is None else side_r
        qlevel = np.full(len(wave), level)
    else:
        frac = (centre - side_l[0]) / (side_r[0] - side_l[0])
        qlevel = side_l[1] + (side_r[1] - side_l[1]) * (wave - side_l[0]) / (
                side_r[0] - side_l[0])
        level = side_l[1] + (side_r[1] - side_l[1]) * frac
        level_err = np.sqrt(((1 - frac) * side_l[2]) ** 2 +
                            (frac * side_r[2]) ** 2)
    pcont = pseudo_continuum(terms, wave)
    out = dict(EW=np.nan, sEW=np.nan, dEW=np.nan, sdEW=np.nan,
               EWT=terms['EWT'], EWTSYS=terms['EWTSYS'], FRACVALID=np.nan,
               CHI2=np.nan)
    # the valid pixels of the window
    valid = in_window & np.isfinite(spectrum) & np.isfinite(model)
    valid &= np.isfinite(rms) & (model > 0) & (blaze > 0) & (qlevel > 0)
    valid &= pcont > 0
    frac_valid = np.sum(length[valid]) / np.sum(length[in_window])
    out['FRACVALID'] = float(frac_valid)
    if frac_valid < MIN_FRAC_VALID:
        return out
    # what the spectrum lacks against the template, in pseudo-continuum
    #   units
    continuum_level = qlevel[valid] * blaze[valid] * pcont[valid]
    resid = qlevel[valid] * model[valid] - spectrum[valid]
    dew = np.sum(length[valid] * resid / continuum_level)
    var = np.sum((length[valid] * rms[valid] / continuum_level) ** 2)
    var += ((level_err / level) *
            np.sum(length[valid] * template[valid] / pcont[valid])) ** 2
    out['dEW'] = float(dew * NM_TO_MA)
    out['sdEW'] = float(np.sqrt(var) * NM_TO_MA)
    out['EW'] = out['EWT'] + out['dEW']
    out['sEW'] = out['sdEW']
    out['CHI2'] = float(np.mean((resid / rms[valid]) ** 2))
    return out


def measure(wavegrid: np.ndarray, spectrum: np.ndarray,
            template: np.ndarray, blaze: np.ndarray, rms: np.ndarray,
            velocity: float, berv: float,
            template_1d: Tuple[np.ndarray, np.ndarray]) -> Table:
    """
    The equivalent widths of all the lines of the catalog in one spectrum,
    against the template (see the module's docstring)

    :param wavegrid: np.ndarray, the wavelengths of the spectrum in the
                     observer's frame [nm], (orders, pixels)
    :param spectrum: np.ndarray, the spectrum, before LBL's sigma clipping
    :param template: np.ndarray, the template at the converged velocity,
                     without the blaze
    :param blaze: np.ndarray, the blaze
    :param rms: np.ndarray, LBL's noise model
    :param velocity: float, the shift that brings the observer's wavelengths
                     to the template's frame [m/s], as mp.doppler_shift
                     takes it
    :param berv: float, the BERV [m/s] (the order is chosen without it)
    :param template_1d: (wavelengths [nm], flux): the template itself, in
                        the same frame as the template on the orders (the
                        star's), without the blaze: the pseudo-continuum and
                        EWT are measured on it

    :return: Table, one row per line of the catalog: the catalog's columns
             and the measured ones (NaN when no order holds the line)
    """
    lines = load_lines()
    t_wave, t_flux = (np.asarray(a, dtype=float) for a in template_1d)
    order = np.argsort(t_wave)
    t_wave, t_flux = t_wave[order], t_flux[order]
    npix = wavegrid.shape[1]
    # the template's frame, and the same without the BERV
    wave_t = mp.doppler_shift(wavegrid, velocity)
    wave_ref = mp.doppler_shift(wavegrid, velocity + berv)
    centre = 0.5 * (npix - 1)
    xpix = np.arange(npix, dtype=float)
    # the range of each order
    finite = np.isfinite(wave_ref)
    wmin = np.min(np.where(finite, wave_ref, np.inf), axis=1)
    wmax = np.max(np.where(finite, wave_ref, -np.inf), axis=1)
    rows = []
    for line in lines:
        row = _catalog_row(line)
        row.update({col: np.nan for col in MEASURED})
        terms = template_terms(line, t_wave, t_flux)
        if terms is None:
            rows.append(row)
            continue
        # orders holding the line, the closest to their centre first
        candidates = []
        for order_num in np.where((wmin < line['WAVE']) &
                                  (wmax > line['WAVE']))[0]:
            good = np.isfinite(wave_ref[order_num])
            order_wave = wave_ref[order_num][good]
            order_x = xpix[good]
            if order_wave[-1] < order_wave[0]:
                order_wave, order_x = order_wave[::-1], order_x[::-1]
            x_ref = np.interp(line['WAVE'], order_wave, order_x)
            candidates.append((abs(x_ref - centre), order_num))
        first = None
        for _, order_num in sorted(candidates):
            out = _measure_line(line, terms, wave_t[order_num],
                                spectrum[order_num], template[order_num],
                                blaze[order_num], rms[order_num])
            if out is None:
                continue
            if first is None:
                first = out
            if np.isfinite(out['dEW']):
                first = out
                break
        # no order valid enough: the closest one says why
        if first is not None:
            row.update(first)
        rows.append(row)
    table = Table(rows=rows, names=CATALOG_COLUMNS + list(MEASURED))
    for col in CATALOG_COLUMNS[5:]:
        table[col].unit = 'nm'
    for col, unit in MEASURED.items():
        if len(unit) > 0:
            table[col].unit = unit
    return table


def log_measured(table: Table):
    """
    One line for the log of lbl_compute: how many lines were measured, and
    the one furthest from the template in sigma

    :param table: Table, from measure
    """
    ew = np.array(table['dEW'], dtype=float)
    sig = np.array(table['sdEW'], dtype=float)
    good = np.isfinite(ew) & np.isfinite(sig) & (sig > 0)
    msg = '\tActivity lines: {0} of {1} measured'.format(np.sum(good),
                                                          len(table))
    if np.any(good):
        nsig = np.where(good, np.abs(ew) / np.where(good, sig, 1), -1)
        it = int(np.argmax(nsig))
        msg += ', furthest: {0} {1:+.1f} +- {2:.1f} mA'.format(
            table['NAME'][it], ew[it], sig[it])
    log.general(msg)


def extension_header(science_file: str) -> fits.Header:
    """
    The header of the ACTIVITY extension of an lblrv file

    :param science_file: str, the science file measured

    :return: fits.Header
    """
    header = fits.Header()
    header['SCIFILE'] = (os.path.basename(science_file), 'science file')
    header['ACTCAT'] = (os.path.basename(CATALOG), 'catalog of lines')
    header['ACTVERS'] = (catalog_version(), 'version of the catalog')
    header['ACTUNIT'] = ('mAngstrom', 'unit of EW, dEW, EWT and errors')
    return header


def has_activity(lblrv_file: str) -> bool:
    """
    Whether an lblrv file has the ACTIVITY extension with the columns of
    MEASURED, made with this version of the catalog (files written before,
    or with other windows, do not: they are measured again)

    :param lblrv_file: str, the lblrv file

    :return: bool
    """
    try:
        with fits.open(lblrv_file) as hdulist:
            if EXTNAME not in hdulist:
                return False
            names = hdulist[EXTNAME].columns.names
            version = hdulist[EXTNAME].header.get('ACTVERS')
            return (all(col in names for col in MEASURED)
                    and version == catalog_version())
    except Exception as _:
        return False


def rdb_column(col: str, name: str) -> str:
    """
    The rdb column of a measured column of a line: <column>_<line>, the
    line's space as an underscore (EW, He 1.08333 -> EW_He_1.08333)
    """
    return '{0}_{1}'.format(col, name.replace(' ', '_'))


def rdb_has_activity(rdbfile: str) -> bool:
    """
    Whether an rdb file has the activity columns of this version (one
    compiled before them, or before EW, does not, and is compiled again)

    :param rdbfile: str, the rdb file

    :return: bool
    """
    try:
        with open(rdbfile, 'r') as rfile:
            names = rfile.readline().rstrip('\n').split('\t')
    except Exception as _:
        return False
    return any(name.startswith('EW_') for name in names)


def add_to_rdb(rdb_table: Table, lblrv_files: List[str]) -> int:
    """
    Add the activity lines to the rdb table of lbl_compile, as columns
    <column>_<line> (rdb_column) for every column of MEASURED and every line
    measured in at least one spectrum. Each row is matched to its lblrv file
    by FILENAME; the date, the files and the BERV are already in the rdb.

    :param rdb_table: Table, the rdb table (one row per lblrv file), changed
                      in place
    :param lblrv_files: list of str, the lblrv files

    :return: int, the number of lines added
    """
    # the measures of each lblrv file: {basename: {line: row}}
    measures, missing = dict(), 0
    for lblrv_file in lblrv_files:
        with fits.open(lblrv_file) as hdulist:
            if EXTNAME not in hdulist:
                missing += 1
                continue
            data = hdulist[EXTNAME].data
            names = np.array(data['NAME']).astype(str)
            if not all(col in data.columns.names for col in MEASURED):
                missing += 1
                continue
            measures[os.path.basename(lblrv_file)] = {
                name: {col: float(data[col][row]) for col in MEASURED}
                for row, name in enumerate(names)}
    if missing > 0:
        msg = ('{0} of {1} lblrv files have no activity lines (computed '
               'before them): rerun lbl_compute on them')
        log.warning(msg.format(missing, len(lblrv_files)))
    filenames = np.array(rdb_table['FILENAME']).astype(str)
    added = 0
    for line in load_lines():
        values = {col: np.full(len(filenames), np.nan) for col in MEASURED}
        for row, filename in enumerate(filenames):
            measure_row = measures.get(filename, dict()).get(line['NAME'])
            if measure_row is None:
                continue
            for col in MEASURED:
                values[col][row] = measure_row[col]
        if not np.isfinite(values['dEW']).any():
            continue
        for col in MEASURED:
            rdb_table[rdb_column(col, line['NAME'])] = values[col]
        added += 1
    msg = 'Activity lines: {0} of {1} measured, added to the rdb'
    log.general(msg.format(added, len(load_lines())))
    return added

# =============================================================================
# End of code
# =============================================================================

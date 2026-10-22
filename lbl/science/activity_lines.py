#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Differential equivalent widths of the activity lines.

The lines are those of lbl/resources/activity_lines.yaml: the literature's
activity indicators from u to K, at their vacuum wavelengths. Every science
spectrum gets them, with no option to set:

    measure()     lbl_compute, once the velocity has converged: for each
                  line, in the order where it falls closest to the centre,
                  the change of its equivalent width against the template.
                  Written to the ACTIVITY extension of the lblrv file, one row
                  per line of the catalog: the catalog's columns and
                  MEASURED
    add_to_rdb()  lbl_compile: the same, as columns of the rdb (one row per
                  spectrum), <column>_<line> for the lines measured at least
                  once, the line's space as an underscore: dEW_He_1.08333,
                  sdEW_He_1.08333, FRACVALID_He_1.08333, CHI2_He_1.08333.
                  sdEW is dEW's error, so the rdb per night (lbl2) bins dEW
                  weighted by it

THE MEASUREMENT
---------------
In the frame of the template, at the velocity the RVs converged to, with S
the spectrum (before LBL's sigma clipping, which would cut a flare out), B
the blaze and T the template:

    q        the local level of S / (T B): its median in two side bands,
             from h + w/2 to h + 5 w/2 on each side of the line (w the
             width of the window, h its half extent), linear across the
             line. LBL's own low-pass of that ratio (5 line widths, 60 to
             230 km/s on NIRPS) would follow a line as broad as H alpha;
             side bands scaled to each window do not
    C        the continuum of the template near the line: the 95th
             percentile of T within 2.5 nm
    r        (q T B - S) / (q B C): what the spectrum lacks against the
             template, in units of the continuum

    dEW = sum over the pixels of a_i r_i

with a_i the length of pixel i inside the window: its overlap with the
window for a boxcar (h = w / 2), its width times a gaussian of FWHM w and
peak 1 for a gaussian (h = 1.5 w). In mA, positive when the line absorbs
more than in the template: a line filling in with emission gives a negative
dEW. Its error, sdEW, carries the noise of LBL's model (rms) and that of q.
FRACVALID is the fraction of the window with valid pixels; the pixels LBL's
sigma clipping rejected are kept (a flare would be cut out otherwise), CHI2
(the residuals of the window against rms, per pixel) shows them.

The order is the one where the line, without the BERV, falls closest to the
centre: the same order for every spectrum, as long as the window and some
of its side bands are inside it and 90% of the window is valid; the next
closest otherwise.

Created on 2026-10-22

@author: artigau
"""
import os
from typing import Dict, List, Optional

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
#: the columns of the catalog kept in the lblrv extension
CATALOG_COLUMNS = ['NAME', 'LABEL', 'SPECIES', 'KIND', 'BAND', 'FUNCTION',
                   'WAVE', 'WIDTH']
#: the columns measured on each line, and their units
MEASURED = dict(dEW='mAngstrom', sdEW='mAngstrom', FRACVALID='', CHI2='')
#: nm to mA
NM_TO_MA = 1e4
#: half width [nm] of the region where the continuum of the template is taken
CONT_HALF_WIDTH = 2.5
#: percentile of the template taken as its continuum
CONT_PERCENTILE = 95
#: least fraction of the window with valid pixels
MIN_FRAC_VALID = 0.9
#: least number of valid pixels in a side band
MIN_SIDE_PIX = 3
# the catalog, read once
_LINES: Optional[List[Dict]] = None


# =============================================================================
# Define functions
# =============================================================================
def load_lines() -> List[Dict]:
    """
    The lines of the catalog, in its order

    :return: list of dicts, one per line (NAME, LABEL, SPECIES, KIND, BAND,
             FUNCTION, WAVE [nm, vacuum], WIDTH [nm])
    """
    global _LINES
    if _LINES is None:
        with open(CATALOG, 'r') as yfile:
            catalog = yaml.safe_load(yfile)['lines']
        lines = []
        for name, line in catalog.items():
            if line['function'] not in ('boxcar', 'gaussian'):
                emsg = '{0}: function must be boxcar or gaussian, not {1} ({2})'
                raise ValueError(emsg.format(name, line['function'], CATALOG))
            lines.append(dict(NAME=name, LABEL=line['label'],
                              SPECIES=line['species'], KIND=line['kind'],
                              BAND=line['band'], FUNCTION=line['function'],
                              WAVE=float(line['wave']),
                              WIDTH=float(line['width'])))
        _LINES = lines
    return _LINES


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
    # error of a median of gaussian noise
    error = 1.2533 * np.median(noise[band]) / np.sqrt(np.sum(band))
    return np.mean(wave[band]), np.median(ratio[band]), error


def _measure_line(line: Dict, wave: np.ndarray, spectrum: np.ndarray,
                  template: np.ndarray, blaze: np.ndarray,
                  rms: np.ndarray) -> Optional[Dict]:
    """
    The differential equivalent width of one line in one order

    :param line: dict, the line (load_lines)
    :param wave: np.ndarray, the wavelengths of the order in the template's
                 frame [nm]
    :param spectrum: np.ndarray, the spectrum of the order (not clipped)
    :param template: np.ndarray, the template, without the blaze
    :param blaze: np.ndarray, the blaze of the order
    :param rms: np.ndarray, LBL's noise model of the order

    :return: None if the window or its side bands are not in the order, else
             the measured columns (dEW NaN if less than MIN_FRAC_VALID of the
             window is valid)
    """
    lam0, width = line['WAVE'], line['WIDTH']
    gaussian = line['FUNCTION'] == 'gaussian'
    half = 1.5 * width if gaussian else 0.5 * width
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
    if edges[0] > lam0 - half or edges[-1] < lam0 + half:
        return None
    # length of each pixel inside the window
    if gaussian:
        dist = wave - lam0
        profile = np.exp(-4 * np.log(2) * (dist / width) ** 2)
        length = np.where(np.abs(dist) <= half,
                          profile * (edges[1:] - edges[:-1]), 0.0)
    else:
        length = (np.minimum(edges[1:], lam0 + half) -
                  np.maximum(edges[:-1], lam0 - half))
        length = np.clip(length, 0, None)
    window = length > 0
    # the local level of S / (T B), from the side bands
    with np.errstate(divide='ignore', invalid='ignore'):
        model = template * blaze
        ratio = spectrum / model
        ratio_noise = rms / model
    dist = np.abs(wave - lam0)
    band = (dist >= half + 0.5 * width) & (dist <= half + 2.5 * width)
    left = _side_level(wave, ratio, ratio_noise, band & (wave < lam0))
    right = _side_level(wave, ratio, ratio_noise, band & (wave > lam0))
    if left is None and right is None:
        return None
    if left is None or right is None:
        _, level, level_err = left if right is None else right
        qlevel = np.full(len(wave), level)
    else:
        frac = (lam0 - left[0]) / (right[0] - left[0])
        qlevel = left[1] + (right[1] - left[1]) * (wave - left[0]) / (
                right[0] - left[0])
        level = left[1] + (right[1] - left[1]) * frac
        level_err = np.sqrt(((1 - frac) * left[2]) ** 2 +
                            (frac * right[2]) ** 2)
    # the continuum of the template
    near = (dist < CONT_HALF_WIDTH) & np.isfinite(template)
    cont = np.percentile(template[near], CONT_PERCENTILE) if np.any(near) \
        else np.nan
    # the valid pixels of the window
    valid = window & np.isfinite(spectrum) & np.isfinite(model)
    valid &= np.isfinite(rms) & (model > 0) & (blaze > 0) & (qlevel > 0)
    frac_valid = np.sum(length[valid]) / np.sum(length[window])
    out = dict(dEW=np.nan, sdEW=np.nan, FRACVALID=float(frac_valid),
               CHI2=np.nan)
    if frac_valid < MIN_FRAC_VALID or not np.isfinite(cont) or cont <= 0:
        return out
    # what the spectrum lacks against the template, in continuum units
    continuum = qlevel[valid] * blaze[valid] * cont
    resid = qlevel[valid] * model[valid] - spectrum[valid]
    ew = np.sum(length[valid] * resid / continuum)
    var = np.sum((length[valid] * rms[valid] / continuum) ** 2)
    var += ((level_err / level) *
            np.sum(length[valid] * template[valid] / cont)) ** 2
    out['dEW'] = float(ew * NM_TO_MA)
    out['sdEW'] = float(np.sqrt(var) * NM_TO_MA)
    out['CHI2'] = float(np.mean((resid / rms[valid]) ** 2))
    return out


def measure(wavegrid: np.ndarray, spectrum: np.ndarray,
            template: np.ndarray, blaze: np.ndarray, rms: np.ndarray,
            velocity: float, berv: float) -> Table:
    """
    The differential equivalent widths of all the lines of the catalog in one
    spectrum, against the template (see the module's docstring)

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

    :return: Table, one row per line of the catalog: the catalog's columns
             and the measured ones (NaN when no order holds the line)
    """
    lines = load_lines()
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
        row = {col: line[col] for col in CATALOG_COLUMNS}
        row.update({col: np.nan for col in MEASURED})
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
            out = _measure_line(line, wave_t[order_num], spectrum[order_num],
                                template[order_num], blaze[order_num],
                                rms[order_num])
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
    table['WAVE'].unit = 'nm'
    table['WIDTH'].unit = 'nm'
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
    header['ACTUNIT'] = ('mAngstrom', 'unit of dEW and sdEW')
    return header


def has_activity(lblrv_file: str) -> bool:
    """
    Whether an lblrv file has the ACTIVITY extension with the columns of
    MEASURED (files written before do not)

    :param lblrv_file: str, the lblrv file

    :return: bool
    """
    try:
        with fits.open(lblrv_file) as hdulist:
            if EXTNAME not in hdulist:
                return False
            names = hdulist[EXTNAME].columns.names
            return all(col in names for col in MEASURED)
    except Exception as _:
        return False


def rdb_column(col: str, name: str) -> str:
    """
    The rdb column of a measured column of a line: <column>_<line>, the
    line's space as an underscore (dEW, He 1.08333 -> dEW_He_1.08333)
    """
    return '{0}_{1}'.format(col, name.replace(' ', '_'))


def rdb_has_activity(rdbfile: str) -> bool:
    """
    Whether an rdb file has the activity columns (one compiled before them
    does not, and is compiled again)

    :param rdbfile: str, the rdb file

    :return: bool
    """
    try:
        with open(rdbfile, 'r') as rfile:
            names = rfile.readline().rstrip('\n').split('\t')
    except Exception as _:
        return False
    return any(name.startswith('dEW_') for name in names)


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

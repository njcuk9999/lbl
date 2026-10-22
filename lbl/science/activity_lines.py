#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Differential equivalent widths of the activity lines.

The lines are those of lbl/resources/activity_lines.yaml: the literature's
activity indicators from u to K, at their vacuum wavelengths. Every science
spectrum gets them, with no option to set:

    measure()         lbl_compute, once the velocity has converged: for each
                      line, in the order where it falls closest to the
                      centre, the change of its equivalent width against the
                      template. Written to the ACTIVITY extension of the
                      lblrv file, one row per line of the catalog
    compile_tables()  lbl_compile, in the lblrdb directory:
                      lbl_<object>_<template>_activity.csv, one row per
                      spectrum (its date, its files, all that was measured on
                      each line), and lbl_<object>_<template>_activity_lines
                      .csv, one row per line (the catalog, how often the line
                      was measured, how much it varies)

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

    EW_DIFF = sum over the pixels of a_i r_i

with a_i the length of pixel i inside the window: its overlap with the
window for a boxcar (h = w / 2), its width times a gaussian of FWHM w and
peak 1 for a gaussian (h = 1.5 w). In mA, positive when the line absorbs
more than in the template: a line filling in with emission gives a negative
EW_DIFF. Its error carries the noise of LBL's model (rms) and that of q.

The order is the one where the line, without the BERV, falls closest to the
centre: the same order for every spectrum, as long as the window and some
of its side bands are inside it and 90% of the window is valid; the next
closest otherwise.

Created on 2026-10-22

@author: artigau
"""
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

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
MEASURED = dict(ORDER='', XPIX='pix', WAVE_OBS='nm', EW_DIFF='mAngstrom',
                SIG_EW_DIFF='mAngstrom', FRAC_VALID='', NPIX='', NCLIP='',
                SNR='', CHI2='')
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
                  clipped: np.ndarray, template: np.ndarray,
                  blaze: np.ndarray, rms: np.ndarray) -> Optional[Dict]:
    """
    The differential equivalent width of one line in one order

    :param line: dict, the line (load_lines)
    :param wave: np.ndarray, the wavelengths of the order in the template's
                 frame [nm]
    :param spectrum: np.ndarray, the spectrum of the order (not clipped)
    :param clipped: np.ndarray, the spectrum after LBL's sigma clipping
    :param template: np.ndarray, the template, without the blaze
    :param blaze: np.ndarray, the blaze of the order
    :param rms: np.ndarray, LBL's noise model of the order

    :return: None if the window or its side bands are not in the order, else
             the measured columns (EW_DIFF NaN if less than MIN_FRAC_VALID of
             the window is valid)
    """
    lam0, width = line['WAVE'], line['WIDTH']
    gaussian = line['FUNCTION'] == 'gaussian'
    half = 1.5 * width if gaussian else 0.5 * width
    xpix = np.arange(len(wave), dtype=float)
    # the finite part of the order, increasing
    finite = np.where(np.isfinite(wave))[0]
    if len(finite) < 10:
        return None
    sel = slice(finite[0], finite[-1] + 1)
    wave, xpix = wave[sel], xpix[sel]
    spectrum, clipped = spectrum[sel], clipped[sel]
    template, blaze, rms = template[sel], blaze[sel], rms[sel]
    if wave[-1] < wave[0]:
        wave, xpix = wave[::-1], xpix[::-1]
        spectrum, clipped = spectrum[::-1], clipped[::-1]
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
    # where the line is in the order, in the observer's frame
    xline = np.interp(lam0, wave, xpix)
    out = dict(XPIX=float(xline), FRAC_VALID=float(frac_valid),
               NPIX=int(np.sum(window)),
               NCLIP=int(np.sum(window & np.isfinite(spectrum) &
                                ~np.isfinite(clipped))),
               EW_DIFF=np.nan, SIG_EW_DIFF=np.nan, SNR=np.nan, CHI2=np.nan)
    if frac_valid < MIN_FRAC_VALID or not np.isfinite(cont) or cont <= 0:
        return out
    # what the spectrum lacks against the template, in continuum units
    continuum = qlevel[valid] * blaze[valid] * cont
    resid = qlevel[valid] * model[valid] - spectrum[valid]
    ew = np.sum(length[valid] * resid / continuum)
    var = np.sum((length[valid] * rms[valid] / continuum) ** 2)
    var += ((level_err / level) *
            np.sum(length[valid] * template[valid] / cont)) ** 2
    out['EW_DIFF'] = float(ew * NM_TO_MA)
    out['SIG_EW_DIFF'] = float(np.sqrt(var) * NM_TO_MA)
    out['SNR'] = float(np.median(continuum / rms[valid]))
    out['CHI2'] = float(np.mean((resid / rms[valid]) ** 2))
    return out


def measure(wavegrid: np.ndarray, spectrum: np.ndarray, clipped: np.ndarray,
            template: np.ndarray, blaze: np.ndarray, rms: np.ndarray,
            velocity: float, berv: float) -> Table:
    """
    The differential equivalent widths of all the lines of the catalog in one
    spectrum, against the template (see the module's docstring)

    :param wavegrid: np.ndarray, the wavelengths of the spectrum in the
                     observer's frame [nm], (orders, pixels)
    :param spectrum: np.ndarray, the spectrum, before LBL's sigma clipping
    :param clipped: np.ndarray, the spectrum after LBL's sigma clipping (NaN
                    where it rejected pixels)
    :param template: np.ndarray, the template at the converged velocity,
                     without the blaze
    :param blaze: np.ndarray, the blaze
    :param rms: np.ndarray, LBL's noise model
    :param velocity: float, the shift that brings the observer's wavelengths
                     to the template's frame [m/s], as mp.doppler_shift
                     takes it
    :param berv: float, the BERV [m/s] (the order is chosen without it)

    :return: Table, one row per line of the catalog: the catalog's columns
             and the measured ones (ORDER = -1 when no order holds the line)
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
        row.update(ORDER=-1, XPIX=np.nan, WAVE_OBS=np.nan, EW_DIFF=np.nan,
                   SIG_EW_DIFF=np.nan, FRAC_VALID=np.nan, NPIX=0, NCLIP=0,
                   SNR=np.nan, CHI2=np.nan)
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
                                clipped[order_num], template[order_num],
                                blaze[order_num], rms[order_num])
            if out is None:
                continue
            out['ORDER'] = int(order_num)
            out['WAVE_OBS'] = float(np.interp(out['XPIX'], xpix,
                                              wavegrid[order_num]))
            if first is None:
                first = out
            if np.isfinite(out['EW_DIFF']):
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
    ew = np.array(table['EW_DIFF'], dtype=float)
    sig = np.array(table['SIG_EW_DIFF'], dtype=float)
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
    header['ACTUNIT'] = ('mAngstrom', 'unit of EW_DIFF and SIG_EW_DIFF')
    return header


def has_activity(lblrv_file: str) -> bool:
    """
    Whether an lblrv file has the ACTIVITY extension (files written before
    it existed do not)

    :param lblrv_file: str, the lblrv file

    :return: bool
    """
    try:
        with fits.open(lblrv_file) as hdulist:
            return EXTNAME in hdulist
    except Exception as _:
        return False


def compile_tables(inst, lblrv_files: List[str], rdbfile: str):
    """
    The activity tables of lbl_compile, beside the rdb file:

        <rdb name>_activity.csv        one row per spectrum, by date: rjd (as
                                       in the rdb), date (UTC, mid-exposure),
                                       FILENAME (the lblrv file, as in the
                                       rdb), SCIENCE_FILE, BERV [m/s], then
                                       for each line measured at least once
                                       <name>_<column> for every measured
                                       column (the name's space as _)
        <rdb name>_activity_lines.csv  one row per line measured at least
                                       once: the catalog's columns, the order
                                       used most often, NMEAS (spectra with a
                                       measure), EW_MEAN (weighted mean),
                                       EW_RMS, EW_MED_SIG and EW_CHI2 (the
                                       reduced chi2 about the weighted mean:
                                       well above 1 for a line that varies)

    :param inst: Instrument instance
    :param lblrv_files: list of str, the lblrv files
    :param rdbfile: str, the rdb file (lbl_<object>_<template>.rdb)
    """
    activity_file = rdbfile.replace('.rdb', '_activity.csv')
    lines_file = rdbfile.replace('.rdb', '_activity_lines.csv')
    mid_exp_key = inst.params['KW_MID_EXP_TIME']
    # the header keys LBL's parameters name: all that the date and the BERV
    #   need, without converting the ~1500 cards of the science header
    #   (that took 20 times as long as the rest)
    keys = sorted({value for key, value in inst.params.items()
                   if key.startswith('KW_') and isinstance(value, str)})
    rows, tables, missing = [], [], 0
    for lblrv_file in lblrv_files:
        with fits.open(lblrv_file) as hdulist:
            if EXTNAME not in hdulist:
                missing += 1
                continue
            table = Table.read(hdulist, hdu=EXTNAME)
            science_file = hdulist[EXTNAME].header.get('SCIFILE', '')
            primary = hdulist[0].header
            small = fits.Header()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for key in keys:
                    if key in primary:
                        small[key] = (primary[key], primary.comments[key])
        header = io.LBLHeader.from_fits(small, lblrv_file)
        mjd = header.get_hkey(mid_exp_key, dtype=float)
        rows.append(dict(rjd=inst.get_rjd_value(header),
                         date=Time(mjd, format='mjd').isot,
                         FILENAME=os.path.basename(lblrv_file),
                         SCIENCE_FILE=science_file,
                         BERV=inst.get_berv(header)))
        tables.append(table)
    if missing > 0:
        msg = ('{0} of {1} lblrv files have no {2} extension (computed before '
               'the activity lines): rerun lbl_compute on them')
        log.warning(msg.format(missing, len(lblrv_files), EXTNAME))
    if len(tables) == 0:
        log.general('No activity lines to compile')
        return
    # the lines measured at least once
    lines = [line for line in load_lines()
             if any(np.isfinite(_value(tab, line['NAME'], 'EW_DIFF'))
                    for tab in tables)]
    # one row per spectrum
    order = np.argsort([row['rjd'] for row in rows])
    out = Table()
    for col in ['rjd', 'date', 'FILENAME', 'SCIENCE_FILE', 'BERV']:
        out[col] = [rows[it][col] for it in order]
    for line in lines:
        prefix = line['NAME'].replace(' ', '_') + '_'
        for col in MEASURED:
            out[prefix + col] = [_value(tables[it], line['NAME'], col)
                                 for it in order]
    msg = 'Writing activity table ({0} spectra, {1} lines): {2}'
    log.info(msg.format(len(out), len(lines), activity_file))
    io.write_table(activity_file, out, fmt='csv')
    # one row per line
    summary = Table()
    for col in CATALOG_COLUMNS:
        summary[col] = [line[col] for line in lines]
    stats = dict(ORDER=[], NMEAS=[], EW_MEAN=[], EW_RMS=[], EW_MED_SIG=[],
                 EW_CHI2=[])
    for line in lines:
        prefix = line['NAME'].replace(' ', '_') + '_'
        ew = np.array(out[prefix + 'EW_DIFF'], dtype=float)
        sig = np.array(out[prefix + 'SIG_EW_DIFF'], dtype=float)
        orders = np.array(out[prefix + 'ORDER'], dtype=int)
        good = np.isfinite(ew) & np.isfinite(sig) & (sig > 0)
        used, counts = np.unique(orders[good], return_counts=True)
        stats['ORDER'].append(int(used[np.argmax(counts)]) if len(used) else -1)
        stats['NMEAS'].append(int(np.sum(good)))
        if np.sum(good) < 2:
            for key in ['EW_MEAN', 'EW_RMS', 'EW_MED_SIG', 'EW_CHI2']:
                stats[key].append(np.nan)
            continue
        weights = 1 / sig[good] ** 2
        mean = np.sum(weights * ew[good]) / np.sum(weights)
        stats['EW_MEAN'].append(mean)
        stats['EW_RMS'].append(np.std(ew[good]))
        stats['EW_MED_SIG'].append(np.median(sig[good]))
        stats['EW_CHI2'].append(np.sum(((ew[good] - mean) / sig[good]) ** 2)
                                / (np.sum(good) - 1))
    for key, values in stats.items():
        summary[key] = values
    msg = 'Writing activity lines table: {0}'
    log.info(msg.format(lines_file))
    io.write_table(lines_file, summary, fmt='csv')


def _value(table: Table, name: str, col: str):
    """
    The value of a column for one line in an ACTIVITY table (NaN, or -1 for
    ORDER, if the line is not in it)
    """
    rows = np.where(np.array(table['NAME']).astype(str) == name)[0]
    if len(rows) == 0:
        return -1 if col == 'ORDER' else np.nan
    return table[col][rows[0]]

# =============================================================================
# End of code
# =============================================================================

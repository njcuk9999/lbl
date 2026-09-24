#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report of what is in the data, written at the end of the processing

This is not a paper: it is the accountant's report of an LBL run. Each
section states what was measured, with the numbers next to the figures. The
figures are also bundled, one file each, in a tar archive next to the pdf, so
that they can go into a paper as they are.

Created on 2026-09-24

@author: artigau
"""
import os
import shutil
import subprocess
import tarfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from astropy.table import Table

from lbl.core import astro
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import select
from lbl.science import general

# do not require a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.report.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
LblException = base_classes.LblException
InstrumentsType = select.InstrumentsType
log = io.log
# speed of light in km/s
speed_of_light_kms = mp.speed_of_light_ms / 1000.0
# -----------------------------------------------------------------------------
# the references of the report: each one is cited in the text with cite() and
#   listed, numbered, at the end
# the papers to cite when the velocities of a run are published: what they
#   are for, how they are cited, and their bibcode on ADS (the box at the
#   top of the report links to them)
CITE_PAPERS = [('LBL', 'Artigau et al. 2022', 'lbl', '2022AJ....164...84A'),
               ('the temperature indicators', 'Artigau et al. 2024', 'dtemp',
                '2024AJ....168..252A'),
               ('APERO (SPIRou, NIRPS)', 'Cook et al. 2022', 'apero',
                '2022PASP..134k4509C')]

# the width of the debug plot of the lines, in resolution elements of the
#   instrument (its resolution is a parameter, APPROX_RESOLUTION)
LINE_PLOT_ELEMENTS = 30.0

# where a bibcode is read on ADS
URL_ADS = 'https://ui.adsabs.harvard.edu/abs/{0}/abstract'

REFERENCES = [
    ('lbl', 'E. Artigau, C. Cadieux, N. J. Cook et al., '
            '\\emph{Line-by-line velocity measurements: an outlier-resistant '
            'method for precision velocimetry}, AJ 164, 84 (2022), '
            'arXiv:2207.13524'),
    ('dtemp', 'E. Artigau, C. Cadieux, N. J. Cook et al., '
              '\\emph{Measuring sub-Kelvin variations in stellar temperature '
              'with high-resolution spectroscopy}, AJ 168, 252 (2024), '
              'arXiv:2409.07260'),
    ('apero', 'N. J. Cook, E. Artigau, R. Doyon et al., \\emph{APERO: a '
               'PipelinE to Reduce Observations - demonstration with SPIRou}, '
               'PASP 134, 114509 (2022), arXiv:2211.01358'),
    ('lomb', 'N. R. Lomb, \\emph{Least-squares frequency analysis of '
             'unequally spaced data}, Ap\\&SS 39, 447 (1976)'),
    ('scargle', 'J. D. Scargle, \\emph{Studies in astronomical time series '
                'analysis. II}, ApJ 263, 835 (1982)'),
    ('gls', 'M. Zechmeister and M. Kurster, \\emph{The generalised '
            'Lomb-Scargle periodogram}, A\\&A 496, 577 (2009)'),
    ('fip', 'N. C. Hara, N. Unger, J.-B. Delisle, R. F. Diaz and '
            'D. Segransan, \\emph{Improving exoplanet detection '
            'capabilities with the false '
            'inclusion probability}, A\\&A 663, A14 (2022), arXiv:2105.06995'),
    ('serval', 'M. Zechmeister, A. Reiners, P. J. Amado et al., '
               '\\emph{Spectrum radial velocity analyser (SERVAL)}, '
               'A\\&A 609, A12 (2018), arXiv:1710.10114'),
    ('exoplaneteu', 'J. Schneider, C. Dedieu, P. Le Sidaner, R. Savalle and '
                    'I. Zolotukhin, \\emph{Defining and cataloging '
                    'exoplanets: the exoplanet.eu database}, A\\&A 532, '
                    'A79 (2011)'),
    ('nasa', 'R. L. Akeson, X. Chen, D. Ciardi et al., \\emph{The NASA '
             'Exoplanet Archive: data and tools for exoplanet research}, '
             'PASP 125, 989 (2013)'),
    ('simbad', 'M. Wenger, F. Ochsenbein, D. Egret et al., \\emph{The SIMBAD '
               'astronomical database}, A\\&AS 143, 9 (2000)'),
    ('phoenix', 'T.-O. Husser, S. Wende-von Berg, S. Dreizler et al., '
                '\\emph{A new extensive library of PHOENIX stellar '
                'atmospheres and synthetic spectra}, A\\&A 553, A6 (2013)')]
URL_EXOPLANET_EU = 'https://exoplanet.eu/catalog/'
URL_SIMBAD_TAP = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
# the NASA exoplanet archive, for the references of the published planets
URL_NASA_TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
# how close a star of exoplanet.eu must be to the target [arcsec]
MATCH_RADIUS = 30.0
# the periods marked on every periodogram [days]: a day and the year and its
#   first harmonics, which the sampling and the Earth put in the data
MARKED_PERIODS = [(1.0, 'day'), (365.25 / 3, 'year/3'), (365.25 / 2, 'year/2'),
                  (365.25, 'year')]
# -----------------------------------------------------------------------------
# the columns of the rdb that get a time series, a periodogram and a FIP.
#   name, error column, long name, unit (the per-band velocities and the
#   residual projections are added to this list when they are in the table)
INDICATORS = [('vrad', 'svrad', 'Radial velocity', 'm/s'),
              ('d2v', 'sd2v', 'Second derivative of the line profile '
                              '(differential line width)', 'm$^2$/s$^2$'),
              ('d3v', 'sd3v', 'Third derivative of the line profile',
               'm$^3$/s$^3$'),
              ('dW', 'sdW', 'Differential line width', 'm$^2$/s$^2$'),
              ('fwhm', 'sig_fwhm', 'FWHM of the CCF', 'm/s'),
              ('contrast', 'sig_contrast', 'Contrast of the CCF', ' '),
              ('CRX', 'sCRX', 'Chromatic index (CRX)', 'm/s/dex'),
              ('vrad_achromatic', 'svrad_achromatic',
               'Achromatic radial velocity', 'm/s'),
              ('vrad_chromatic_slope', 'svrad_chromatic_slope',
               'Chromatic slope of the radial velocity', 'm/s/dex')]
# how many indicators are plotted on their own page at most (the rest are in
#   the summary table only)
MAX_INDICATOR_PLOTS = 24
# the velocities of a slice of the domain (vrad_1673nm) or of a part of the
#   detector (vrad_h_0-2044) stay in the summary table but get no figure:
#   there are dozens of them and they say the same thing as the bands
NO_FIGURE_PATTERNS = ['nm', '-']


# =============================================================================
# Define the data of the report
# =============================================================================
def get_report_data(inst: InstrumentsType, dparams: Dict[str, str]
                    ) -> Dict[str, Any]:
    """
    Everything the report needs, read the LBL way

    :param inst: Instrument instance
    :param dparams: dict, the directories of this run

    :return: dict, the data of the report
    """
    # storage
    rdata = dict()
    # -------------------------------------------------------------------------
    # the rdb files of this object
    # -------------------------------------------------------------------------
    rdbfiles = inst.get_lblrdb_files(dparams['LBL_RDB_DIR'])
    rdbfile1, rdbfile2 = rdbfiles[0], rdbfiles[1]
    if not os.path.exists(rdbfile1):
        emsg = 'No rdb file for this object: {0} (run lbl_compile first)'
        raise LblException(emsg.format(rdbfile1))
    # per file and per night
    rdata['rdb'] = inst.load_lblrdb_file(rdbfile1)
    rdata['rdbfile'] = rdbfile1
    if os.path.exists(rdbfile2):
        rdata['rdb2'] = inst.load_lblrdb_file(rdbfile2)
        rdata['rdbfile2'] = rdbfile2
    else:
        rdata['rdb2'], rdata['rdbfile2'] = None, None
    # -------------------------------------------------------------------------
    # the berv of each file, the LBL way (from the lblrv headers)
    # -------------------------------------------------------------------------
    lblrv_dir = dparams['LBLRV_DIR']
    berv = np.full(len(rdata['rdb']), np.nan)
    header_berv = np.full(len(rdata['rdb']), np.nan)
    lblrv_files = []
    for row in range(len(rdata['rdb'])):
        basename = str(rdata['rdb']['FILENAME'][row])
        lblrv_file = os.path.join(lblrv_dir, basename)
        lblrv_files.append(lblrv_file)
        if not os.path.exists(lblrv_file):
            continue
        # the berv comes from the instrument, as everywhere else in LBL
        rvhdr = inst.load_header(lblrv_file, kind='lbl rv fits file')
        berv[row] = inst.get_berv(rvhdr)
        # the berv of the header, with the key of the instrument: for the
        #   modes whose wave solution is already barycentric, get_berv gives
        #   zero and this one is the motion of the Earth
        bervkey = inst.params['KW_BERV']
        if bervkey not in [None, 'None'] and bervkey in rvhdr:
            try:
                # the header key is in km/s, get_berv gives m/s: the report
                #   keeps everything in m/s, as LBL does
                header_berv[row] = rvhdr.get_hkey(bervkey, dtype=float) * 1000
            except Exception as _:
                pass
    # the BERV LBL used is the one that puts a spectrum in the rest frame of
    #   the star (the river plots): it is zero for the modes whose wave
    #   solution is already barycentric
    rdata['berv_lbl'] = berv
    # the BERV of the figures: the one LBL used, unless it does not move, in
    #   which case the motion of the Earth comes from the header
    rdata['berv_source'] = 'the BERV used by LBL (get_berv)'
    if np.nanmax(berv) - np.nanmin(berv) < 1.0e-6:
        if np.nanmax(header_berv) - np.nanmin(header_berv) > 1.0e-6:
            berv = header_berv
            rdata['berv_source'] = ('the {0} key of the headers: this mode '
                                    'has a barycentric wave solution, so the '
                                    'BERV of LBL is zero and the wavelengths '
                                    'are already corrected'
                                    ''.format(inst.params['KW_BERV']))
        else:
            rdata['berv_source'] = 'no BERV found (it does not move)'
    rdata['berv'] = berv
    rdata['lblrv_files'] = lblrv_files
    # -------------------------------------------------------------------------
    # the mask, the template and the systemic velocity
    # -------------------------------------------------------------------------
    rdata['object'] = str(inst.params['OBJECT_SCIENCE'])
    rdata['template_object'] = str(inst.params['OBJECT_COMPARISON'])
    rdata['instrument'] = '{0} ({1})'.format(inst.params['INSTRUMENT'],
                                             inst.params['DATA_SOURCE'])
    # the names LBL itself uses for them, so that the table of the report and
    #   the wrap script of the user speak of the same things
    rdata['instrument_name'] = str(inst.params['INSTRUMENT'])
    rdata['data_source'] = str(inst.params['DATA_SOURCE'])
    rdata['data_type'] = str(inst.params['DATA_TYPE'])
    # the mask of the compute step (and its systemic velocity)
    try:
        mask_file = inst.mask_file(dparams['MODEL_DIR'], dparams['MASK_DIR'])
        rdata['mask_file'] = os.path.basename(mask_file)
        rdata['systemic_velocity'] = inst.get_mask_systemic_vel(mask_file)
    except Exception as _:
        rdata['mask_file'] = 'unknown'
        rdata['systemic_velocity'] = np.nan
    # the template
    try:
        template_file = inst.template_file(dparams['TEMPLATE_DIR'],
                                           'comparison')
        rdata['template_file'] = os.path.basename(template_file)
    except Exception as _:
        rdata['template_file'] = 'unknown'
    # -------------------------------------------------------------------------
    # the indicators that are in this rdb file
    # -------------------------------------------------------------------------
    rdata['indicators'] = find_indicators(rdata['rdb'])
    # return the data
    return rdata


def find_indicators(rdb: Table) -> List[Tuple[str, str, str, str]]:
    """
    The indicators of INDICATORS that are in the rdb file, plus the residual
    projections (DTEMP...) and the per-band velocities it holds

    :param rdb: astropy Table, the rdb table (one row per file)

    :return: list of tuples, the name, error column, long name and unit
    """
    indicators = []
    # the ones we know about
    for name, ename, longname, unit in INDICATORS:
        if name in rdb.colnames and ename in rdb.colnames:
            indicators.append((name, ename, longname, unit))
    # the residual projections (DTEMP3000, DTEMP3000_j, ...) and the per-band
    #   velocities (vrad_h, vrad_1673nm, ...), which depend on the run
    for name in rdb.colnames:
        ename = 's' + name
        if name.startswith('s') or ename not in rdb.colnames:
            continue
        if name in [ind[0] for ind in indicators]:
            continue
        if name.startswith('DTEMP'):
            longname = 'Temperature gradient projection {0}'.format(name)
            indicators.append((name, ename, longname, 'K'))
        elif name.startswith('vrad_'):
            longname = 'Radial velocity, {0}'.format(name[5:])
            indicators.append((name, ename, longname, 'm/s'))
    return indicators


# =============================================================================
# Define the statistics of the report
# =============================================================================
def time_series(rdb: Table, name: str, ename: str
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The time, value and error of an indicator, without its invalid points

    :param rdb: astropy Table, the rdb table
    :param name: str, the column of the indicator
    :param ename: str, the column of its error

    :return: tuple, 1. the time [days], 2. the value, 3. the error
    """
    time = np.array(rdb['rjd'], dtype=float)
    value = np.array(rdb[name], dtype=float)
    error = np.array(rdb[ename], dtype=float)
    # only the points with a value, an error and a time
    good = np.isfinite(time) & np.isfinite(value) & np.isfinite(error)
    good &= error > 0
    return time[good], value[good], error[good]


def remove_drift(time: np.ndarray, value: np.ndarray, error: np.ndarray
                 ) -> Tuple[np.ndarray, float, float, Optional[np.ndarray],
                            Optional[np.ndarray], float]:
    """
    Take a long term drift out of a time series

    A straight line in time is fitted with the error bars as weights and
    subtracted, so that a drift (a companion of a long period, an instrument
    that moves) does not spread its power over the whole periodogram.

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error

    :return: tuple, 1. the value without the drift, 2. the drift [value/day],
             3. its uncertainty, 4. the two coefficients of the line, 5. their
             covariance, 6. the time the line is centred on [days]
    """
    if len(value) < 5:
        return value, np.nan, np.nan, None, None, 0.0
    # a straight line in time, with the error bars as weights, centred on the
    #   middle of the observations so that the two parameters are independent
    tmean = float(np.mean(time))
    with warnings.catch_warnings(record=True) as _:
        coeffs, cov = np.polyfit(time - tmean, value, 1, w=1 / error, cov=True)
    drift, sdrift = float(coeffs[0]), float(np.sqrt(cov[0, 0]))
    # the value without it
    detrended = value - np.polyval(coeffs, time - tmean)
    return detrended, drift, sdrift, coeffs, cov, tmean


def basic_stats(time: np.ndarray, value: np.ndarray,
                error: np.ndarray) -> Dict[str, float]:
    """
    The numbers of the report for one indicator

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error

    :return: dict, the statistics
    """
    stats = dict()
    stats['n'] = len(value)
    if len(value) < 2:
        for key in ['median', 'rms', 'robust_rms', 'error', 'chi2', 'excess']:
            stats[key] = np.nan
        return stats
    stats['median'] = float(np.median(value))
    stats['rms'] = float(np.std(value))
    # robust dispersion (1.4826 x the median absolute deviation)
    stats['robust_rms'] = float(mp.estimate_sigma(value))
    stats['error'] = float(np.median(error))
    # weighted mean and the chi2 around it
    weight = 1.0 / error ** 2
    mean = float(np.sum(weight * value) / np.sum(weight))
    stats['chi2'] = float(np.sum(((value - mean) / error) ** 2) /
                          (len(value) - 1))
    # the dispersion that the error bars do not explain
    excess = stats['robust_rms'] ** 2 - stats['error'] ** 2
    stats['excess'] = float(np.sqrt(excess)) if excess > 0 else 0.0
    return stats


def period_grid(time: np.ndarray, params: base_classes.ParamDict
                ) -> np.ndarray:
    """
    The periods of the periodograms and of the FIP

    The grid is regular in frequency, with a step of a tenth of the width of
    a peak (1 / the time span), which is the usual oversampling.

    :param time: np.ndarray, the time of the observations [days]
    :param params: ParamDict, the parameters of the instrument

    :return: np.ndarray, the periods [days]
    """
    # the time span of the observations
    baseline = float(np.max(time) - np.min(time))
    # the longest period: twice the time span unless it is given
    period_max = params['REPORT_PERIOD_MAX']
    if period_max in [None, 'None', '']:
        period_max = 2 * baseline
    period_min = params['REPORT_PERIOD_MIN']
    # a regular grid in frequency, oversampled by 10
    fmin, fmax = 1 / float(period_max), 1 / float(period_min)
    nfreq = int(np.ceil(10 * baseline * (fmax - fmin))) + 1
    frequency = np.linspace(fmin, fmax, max(nfreq, 100))
    # return the periods
    return 1 / frequency


def lomb_scargle(time: np.ndarray, value: np.ndarray, error: np.ndarray,
                 periods: np.ndarray) -> np.ndarray:
    """
    The floating mean periodogram (a sine and an offset fitted at each period)

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error
    :param periods: np.ndarray, the periods [days]

    :return: np.ndarray, the power of the periodogram (0 to 1)
    """
    from astropy.timeseries import LombScargle
    with warnings.catch_warnings(record=True) as _:
        model = LombScargle(time, value, error, fit_mean=True)
        power = model.power(1 / periods, normalization='standard')
    return np.array(power)


def fip_periodogram(time: np.ndarray, value: np.ndarray, error: np.ndarray,
                    periods: np.ndarray, prior: float = 0.5
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    The false inclusion probability (FIP) of a signal, period bin by period bin

    The FIP of a period interval is one minus the probability that a signal is
    present with a period in that interval (Hara et al. 2022). Here that
    probability is computed with a single sinusoid: at each period the model
        value = offset + A cos(2 pi t / P) + B sin(2 pi t / P)
    is compared with the model without a signal (offset only). The offset and
    the amplitudes are linear parameters and are integrated out analytically
    with wide gaussian priors, which gives the Bayes factor of the period. The
    periods are taken with a prior flat in log(period), and the probability
    that a signal is in a bin follows from Bayes' rule with the prior
    probability that a signal is present at all.

    The bins have the width of a peak of the periodogram (1 / the time span),
    as in Hara et al. (2022). This is the single-planet version of their
    framework: it does not search several signals at once, so a second signal
    is not taken out before the FIP of the first is computed.

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error
    :param periods: np.ndarray, the periods [days]
    :param prior: float, the prior probability that a signal is present

    :return: tuple, 1. the periods of the bins [days], 2. the FIP of each bin
    """
    # -------------------------------------------------------------------------
    # the noise: the error bars, plus the dispersion they do not explain
    #   (a jitter, so that the Bayes factors are not driven by it)
    # -------------------------------------------------------------------------
    weight = 1.0 / error ** 2
    mean = np.sum(weight * value) / np.sum(weight)
    excess = mp.estimate_sigma(value) ** 2 - np.median(error) ** 2
    jitter2 = excess if excess > 0 else 0.0
    variance = error ** 2 + jitter2
    # the value without its weighted mean
    yvalue = value - mean
    # the width of the priors of the linear parameters: wide, so that they
    #   do not drive the result (ten times the dispersion of the data)
    tau2 = (10 * np.std(yvalue)) ** 2
    if tau2 <= 0:
        return np.array([]), np.array([])
    # -------------------------------------------------------------------------
    # the Bayes factor of each period
    # -------------------------------------------------------------------------
    # the model without a signal: an offset only
    logz0 = log_evidence(np.ones((len(time), 1)), yvalue, variance, tau2)
    logbf = np.zeros(len(periods))
    for it, period in enumerate(periods):
        phase = 2 * np.pi * time / period
        design = np.array([np.ones(len(time)), np.cos(phase), np.sin(phase)]).T
        logbf[it] = log_evidence(design, yvalue, variance, tau2) - logz0
    # -------------------------------------------------------------------------
    # the probability that a signal is in each bin
    # -------------------------------------------------------------------------
    # the prior of the periods is flat in log(period)
    logp = np.log(periods)
    # the integral of the Bayes factor over all the periods (the prior is
    #   normalised over the grid)
    bfmax = np.max(logbf)
    weights = np.exp(logbf - bfmax)
    norm = np.abs(trapezoid(np.ones(len(logp)), logp))
    total = np.abs(trapezoid(weights, logp)) / norm
    # the bins have the width of a peak of the periodogram
    baseline = float(np.max(time) - np.min(time))
    frequency = 1 / periods
    edges = np.arange(np.min(frequency), np.max(frequency) + 1 / baseline,
                      1 / baseline)
    # the posterior probability that a signal is present at all
    odds_all = (prior / (1 - prior)) * total * np.exp(bfmax)
    probability = odds_all / (1 + odds_all)
    bin_periods, fip = [], []
    for it in range(len(edges) - 1):
        inside = (frequency >= edges[it]) & (frequency < edges[it + 1])
        if np.sum(inside) < 2:
            continue
        # the integral of the Bayes factor over this bin
        inbin = np.abs(trapezoid(weights[inside], logp[inside])) / norm
        # the probability that a signal is present and in this bin
        tip = probability * (inbin / total)
        bin_periods.append(float(np.mean(periods[inside])))
        fip.append(float(min(max(1 - tip, 0.0), 1.0)))
    return np.array(bin_periods), np.array(fip)


def trapezoid(yvalue: np.ndarray, xvalue: np.ndarray) -> float:
    """
    The integral of y over x by the trapezoid rule

    (np.trapz before numpy 2, np.trapezoid after it)

    :param yvalue: np.ndarray, the values
    :param xvalue: np.ndarray, where they are

    :return: float, the integral
    """
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(yvalue, xvalue))
    return float(np.trapz(yvalue, xvalue))


def log_evidence(design: np.ndarray, value: np.ndarray, variance: np.ndarray,
                 tau2: float) -> float:
    """
    Logarithm of the evidence of a linear model with gaussian priors

    The model is value = design . beta + noise, with independent gaussian
    noise of the given variance and gaussian priors of variance tau2 on the
    parameters beta. The parameters are integrated out analytically:
        Z = N(value | 0, Sigma + tau2 . design . design^T)
    and the determinant and the inverse are computed with the small matrices
    of the parameters, not with the big matrix of the points.

    :param design: np.ndarray, the design matrix (points x parameters)
    :param value: np.ndarray, the values
    :param variance: np.ndarray, the variance of each point
    :param tau2: float, the variance of the priors of the parameters

    :return: float, the logarithm of the evidence
    """
    # design^T Sigma^-1 design and design^T Sigma^-1 value
    dtw = design.T / variance
    amat = dtw @ design
    bvec = dtw @ value
    # the matrix of the posterior of the parameters
    kmat = amat + np.eye(design.shape[1]) / tau2
    # the chi2 of the model without the parameters, and the gain they bring
    chi2 = np.sum(value ** 2 / variance)
    try:
        gain = bvec @ np.linalg.solve(kmat, bvec)
        sign, logdet = np.linalg.slogdet(kmat)
    except np.linalg.LinAlgError:
        return -np.inf
    if sign <= 0:
        return -np.inf
    # ln Z, dropping the terms that do not depend on the model
    logz = -0.5 * (chi2 - gain) - 0.5 * logdet
    logz = logz - 0.5 * design.shape[1] * np.log(tau2)
    return float(logz)


def window_function(time: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    The spectral window of the observation times

    :param time: np.ndarray, the time of the observations [days]
    :param periods: np.ndarray, the periods [days]

    :return: np.ndarray, the power of the window (1 at an infinite period)
    """
    power = np.zeros(len(periods))
    for it, period in enumerate(periods):
        phase = 2 * np.pi * time / period
        real = np.sum(np.cos(phase)) / len(time)
        imag = np.sum(np.sin(phase)) / len(time)
        power[it] = real ** 2 + imag ** 2
    return power


def correlation(xvalue: np.ndarray, yvalue: np.ndarray) -> Tuple[float, float]:
    """
    The Pearson correlation of two vectors and its p-value

    :param xvalue: np.ndarray, the first vector
    :param yvalue: np.ndarray, the second vector

    :return: tuple, 1. the correlation, 2. the p-value
    """
    from scipy.stats import pearsonr
    good = np.isfinite(xvalue) & np.isfinite(yvalue)
    if np.sum(good) < 5:
        return np.nan, np.nan
    with warnings.catch_warnings(record=True) as _:
        rvalue, pvalue = pearsonr(xvalue[good], yvalue[good])
    return float(rvalue), float(pvalue)


# =============================================================================
# Define the known planets (exoplanet.eu)
# =============================================================================
def to_rjd(value: float) -> float:
    """
    A time of a catalogue, as the rjd of the rdb files (jd - 2400000)

    Catalogues write their epochs as a full jd (2455093.8), as bjd - 2450000
    (5093.8) or as an rjd already (55093.8): the three are told apart by
    their size.

    :param value: float, the time of the catalogue

    :return: float, the time [rjd], or nan
    """
    try:
        time = float(value)
    except Exception as _:
        return np.nan
    if not np.isfinite(time) or time <= 0:
        return np.nan
    # a full jd
    if time > 2.4e6:
        return time - 2400000.0
    # an rjd (the rdb files of any instrument are in this range)
    if time > 4.0e4:
        return time
    # bjd - 2450000
    return time + 50000.0


def planet_ephemerides(planets: Optional[Table]
                       ) -> List[Dict[str, Any]]:
    """
    The planets that can be folded in phase: their period and, when the
    catalogue has one, the epoch of their transit or of their conjunction

    :param planets: astropy Table or None, the planets of exoplanet.eu

    :return: list of dict, one per planet (name, period, t0, t0_source)
    """
    out = []
    if planets is None or 'orbital_period' not in planets.colnames:
        return out
    for row in range(len(planets)):
        period = planets['orbital_period'][row]
        if not np.isfinite(period) or period <= 0:
            continue
        # the epoch of the fold: the transit first, then the conjunction,
        #   then the passage at the periastron
        t0, t0_source = np.nan, 'the first observation'
        for column, source in [('tzero_tr', 'the transit'),
                               ('tconj', 'the conjunction'),
                               ('tperi', 'the periastron')]:
            if column not in planets.colnames:
                continue
            t0 = to_rjd(planets[column][row])
            if np.isfinite(t0):
                t0_source = source
                break
        out.append(dict(name=str(planets['name'][row]),
                        period=float(period), t0=t0, t0_source=t0_source))
    # the shortest period first
    return sorted(out, key=lambda item: item['period'])


def fit_sinusoids(time: np.ndarray, value: np.ndarray, error: np.ndarray,
                  periods: List[float]
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit one sinusoid of a known period per planet, all at once

    The periods are held at the value of the catalogue, so the model is
    linear in its parameters: an offset and, for each planet, the amplitude
    of a cosine and of a sine. The error bars are the weights.

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error
    :param periods: list of float, the period of each planet [days]

    :return: tuple, 1. the parameters (offset, then cos and sin of each
             planet), 2. their covariance
    """
    # the design matrix: the offset and the two terms of each period
    columns = [np.ones(len(time))]
    for period in periods:
        angle = 2 * np.pi * time / period
        columns += [np.cos(angle), np.sin(angle)]
    design = np.array(columns).T
    # the error bars as weights
    weight = 1.0 / error
    try:
        wdesign = design * weight[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(wdesign, value * weight, rcond=None)
        cov = np.linalg.inv(np.dot(wdesign.T, wdesign))
    except Exception as _:
        return None, None
    return coeffs, cov


def sinusoid_amplitude(coeffs: np.ndarray, cov: np.ndarray, index: int
                       ) -> Tuple[float, float]:
    """
    The amplitude of one of the sinusoids of fit_sinusoids, and its error

    :param coeffs: np.ndarray, the parameters of the fit
    :param cov: np.ndarray, their covariance
    :param index: int, the rank of the planet (its first parameter is at
                  1 + 2 * index)

    :return: tuple, the amplitude and its error
    """
    first = 1 + 2 * index
    acoeff, bcoeff = coeffs[first], coeffs[first + 1]
    amplitude = float(np.sqrt(acoeff ** 2 + bcoeff ** 2))
    if amplitude == 0 or cov is None:
        return amplitude, np.nan
    # the amplitude is sqrt(a^2 + b^2): its error comes from the covariance
    #   of a and b
    variance = (acoeff ** 2 * cov[first, first]
                + bcoeff ** 2 * cov[first + 1, first + 1]
                + 2 * acoeff * bcoeff * cov[first, first + 1])
    return amplitude, float(np.sqrt(max(variance, 0)) / amplitude)


def phase_bins(phase: np.ndarray, value: np.ndarray, error: np.ndarray,
               nbins: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The weighted mean of a phase folded series in bins of phase

    :param phase: np.ndarray, the phase (0 to 1)
    :param value: np.ndarray, the value
    :param error: np.ndarray, the error
    :param nbins: int, the number of bins

    :return: tuple, the middle of the bins, the mean and its error
    """
    edges = np.linspace(0, 1, nbins + 1)
    centres, means, errors = [], [], []
    for it in range(nbins):
        inside = (phase >= edges[it]) & (phase < edges[it + 1])
        inside &= np.isfinite(value) & np.isfinite(error) & (error > 0)
        if np.sum(inside) == 0:
            continue
        weight = 1.0 / error[inside] ** 2
        centres.append(0.5 * (edges[it] + edges[it + 1]))
        means.append(float(np.sum(value[inside] * weight) / np.sum(weight)))
        errors.append(float(1.0 / np.sqrt(np.sum(weight))))
    return np.array(centres), np.array(means), np.array(errors)


def simbad_target(objname: str) -> Optional[Dict[str, Any]]:
    """
    The target in SIMBAD: its main name, its coordinates and its identifiers

    The coordinates are what the planets of exoplanet.eu are matched on, which
    is safer than a name.

    :param objname: str, the name of the object

    :return: dict or None, main_id, ra, dec, sp_type and ids
    """
    import urllib.parse
    import urllib.request
    # the names to try: the object, and the object without the suffixes LBL
    #   adds to it
    for name in [objname, simple_name(objname)]:
        query = ("SELECT TOP 1 b.main_id, b.ra, b.dec, b.sp_type FROM basic "
                 "AS b JOIN ident AS i ON b.oid = i.oidref WHERE "
                 "i.id = '{0}'".format(name.replace("'", "")))
        params = dict(request='doQuery', lang='adql', format='csv',
                      query=query)
        url = URL_SIMBAD_TAP + '?' + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as handle:
                lines = handle.read().decode('utf-8').splitlines()
        except Exception as e:
            log.warning('SIMBAD could not be reached: {0}'.format(str(e)))
            return None
        # the header and one row
        if len(lines) < 2:
            continue
        values = lines[1].replace('"', '').split(',')
        if len(values) < 4:
            continue
        try:
            target = dict(main_id=values[0], ra=float(values[1]),
                          dec=float(values[2]), sp_type=values[3],
                          name=name)
        except ValueError:
            continue
        # every identifier of this star, to match a catalogue by name too
        target['ids'] = simbad_identifiers(name)
        msg = 'SIMBAD: {0} is {1} at ({2:.5f}, {3:.5f}), {4}'
        log.general(msg.format(objname, target['main_id'], target['ra'],
                               target['dec'], target['sp_type']))
        return target
    log.warning('SIMBAD does not know {0}'.format(objname))
    return None


def simbad_identifiers(name: str) -> List[str]:
    """
    Every identifier SIMBAD has for a star

    :param name: str, the name of the star

    :return: list of str, the identifiers
    """
    import urllib.parse
    import urllib.request
    query = ("SELECT i2.id FROM ident AS i1 JOIN ident AS i2 ON "
             "i1.oidref = i2.oidref WHERE i1.id = '{0}'"
             "".format(name.replace("'", "")))
    params = dict(request='doQuery', lang='adql', format='csv', query=query)
    url = URL_SIMBAD_TAP + '?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as handle:
            lines = handle.read().decode('utf-8').splitlines()
    except Exception as _:
        return []
    return [line.replace('"', '').strip() for line in lines[1:]]


def exoplanet_eu_planets(inst: InstrumentsType, objname: str,
                         directory: str,
                         target: Optional[Dict[str, Any]] = None
                         ) -> Optional[Table]:
    """
    The planets of this star in the catalogue of exoplanet.eu

    The whole catalogue is downloaded once (it is a few megabytes) and kept in
    the report directory, then the rows of this star are taken from it. The
    name of the star is matched without its spaces, dashes and case, and the
    usual prefixes of the LBL object names are removed.

    :param inst: Instrument instance
    :param objname: str, the name of the object
    :param directory: str, where the catalogue is kept
    :param target: dict or None, the target in SIMBAD (its coordinates are
                   what the rows are matched on first)

    :return: astropy Table or None, the planets of this star
    """
    import urllib.request
    # the catalogue, downloaded once
    url = inst.params['REPORT_EXOPLANET_EU_URL']
    catalogue = os.path.join(directory, 'exoplanet_eu_catalog.csv')
    if not os.path.exists(catalogue):
        msg = 'Downloading the catalogue of exoplanet.eu \n\t{0}'
        log.general(msg.format(url))
        try:
            urllib.request.urlretrieve(url, catalogue)
        except Exception as e:
            wmsg = 'Could not download the catalogue of exoplanet.eu: {0}'
            log.warning(wmsg.format(str(e)))
            return None
    # read it
    try:
        table = Table.read(catalogue, format='ascii.csv', fast_reader=False)
    except Exception as e:
        log.warning('Could not read {0}: {1}'.format(catalogue, str(e)))
        return None
    # the names this star could be under: its own, and every identifier
    #   SIMBAD has for it
    targets = set(name_variants(objname))
    if target is not None:
        for identifier in [target['main_id']] + target.get('ids', []):
            targets |= set(name_variants(identifier))
    # the rows of this star, by position first (safer than a name)
    keep = np.zeros(len(table), dtype=bool)
    if target is not None and 'ra' in table.colnames:
        with warnings.catch_warnings(record=True) as _:
            ra = np.array(table['ra'], dtype=float)
            dec = np.array(table['dec'], dtype=float)
        # the angular distance to the target [arcsec]
        cosdec = np.cos(np.radians(target['dec']))
        dra = (ra - target['ra']) * cosdec
        ddec = dec - target['dec']
        distance = np.sqrt(dra ** 2 + ddec ** 2) * 3600
        keep |= np.isfinite(distance) & (distance < MATCH_RADIUS)
    # and by name, for the rows without a position
    for row in range(len(table)):
        if keep[row]:
            continue
        names = set(name_variants(str(table['star_name'][row])))
        if 'star_alternate_names' in table.colnames:
            alternates = str(table['star_alternate_names'][row])
            for alternate in alternates.split(','):
                names |= set(name_variants(alternate))
        if len(names & targets) > 0:
            keep[row] = True
    if np.sum(keep) == 0:
        return None
    return table[keep]


def name_variants(name: str) -> List[str]:
    """
    The ways a star name can be written, simplified, to compare two catalogues

    The name itself, the name without the letter of its component (the
    catalogue has Kepler-21 A where LBL has Kepler-21), and Gl for GJ, which
    are the same catalogue of nearby stars.

    :param name: str, the name

    :return: list of str, the variants
    """
    simple = simple_name(name)
    if len(simple) == 0:
        return []
    variants = {simple}
    # without the letter of the component (kepler21a -> kepler21)
    if len(simple) > 1 and simple[-1].isalpha() and simple[-2].isdigit():
        variants.add(simple[:-1])
    # Gl and GJ are the same catalogue
    for variant in list(variants):
        if variant.startswith('gl'):
            variants.add('gj' + variant[2:])
        elif variant.startswith('gj'):
            variants.add('gl' + variant[2:])
    return sorted(variants)


def nasa_archive_planets(names: List[str]) -> Optional[Table]:
    """
    The published planets of a star in the NASA exoplanet archive, with the
    reference of their discovery and of their parameters

    exoplanet.eu says which planets are known; the archive is asked for the
    papers behind them, which it gives with a link to ADS. The default
    parameter set of each planet is taken (default_flag = 1).

    :param names: list of str, the names the star could be under

    :return: astropy Table or None, one row per planet
    """
    import urllib.parse
    import urllib.request
    # the names to ask for, without the ones that are clearly not a host name
    hosts = []
    for name in names:
        name = str(name).strip()
        if len(name) == 0 or "'" in name:
            continue
        if name not in hosts:
            hosts.append(name)
    if len(hosts) == 0:
        return None
    # one query for all the names
    inlist = ', '.join(["'{0}'".format(host) for host in hosts[:40]])
    query = ('select pl_name, hostname, disc_year, disc_refname, pl_refname, '
             'pl_orbper, pl_bmassj, pl_rvamp, discoverymethod from ps where '
             'default_flag = 1 and hostname in ({0})'.format(inlist))
    params = dict(query=query, format='csv')
    url = URL_NASA_TAP + '?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as handle:
            text = handle.read().decode('utf-8')
    except Exception as e:
        wmsg = 'The NASA archive could not be reached: {0}'
        log.warning(wmsg.format(str(e)))
        return None
    if not text.lower().startswith('pl_name'):
        return None
    try:
        table = Table.read(text, format='ascii.csv', fast_reader=False)
    except Exception as _:
        return None
    if len(table) == 0:
        return None
    msg = 'NASA archive: {0} planet(s) of {1}'
    hosts = ', '.join(np.unique(table['hostname']))
    log.general(msg.format(len(table), hosts))
    return table


def same_planet(name1: str, name2: str) -> bool:
    """
    Whether two catalogues mean the same planet

    A planet is a star and a letter, and the catalogues do not write the star
    the same way: exoplanet.eu has 'Kepler-21 Ab' (the component A of the
    star, planet b) where the NASA archive has 'Kepler-21 b'. The letter of
    the planet is compared, and the star with the variants of its name.

    :param name1: str, the name in one catalogue
    :param name2: str, the name in the other

    :return: bool, True if they are the same planet
    """
    def _split(name):
        simple = simple_name(name)
        # the letter of the planet is the last character
        if len(simple) < 2 or not simple[-1].isalpha():
            return simple, ''
        return simple[:-1], simple[-1]
    host1, letter1 = _split(name1)
    host2, letter2 = _split(name2)
    if letter1 != letter2 or len(letter1) == 0:
        return False
    # the star, with the variants of its name (Kepler-21 A and Kepler-21)
    return len(set(name_variants(host1)) & set(name_variants(host2))) > 0


def reference_text(refname: str) -> Tuple[str, str]:
    """
    The author and year of a reference of the NASA archive, and its link

    The archive gives them as a piece of html, such as
    '<a refstr=... href=https://ui.adsabs.harvard.edu/abs/... >Howell et al.
    2012</a>'

    :param refname: str, the reference as the archive gives it

    :return: tuple, 1. the author and year, 2. the url (empty if there is none)
    """
    text = str(refname)
    if '<a' not in text:
        return text.strip(), ''
    # what is between the tags is the author and the year
    label = text.split('>')[-2].split('<')[0] if '>' in text else text
    # and the link is the href
    url = ''
    for piece in text.split():
        if piece.startswith('href='):
            url = piece[len('href='):].strip('">')
    return label.strip(), url


def simple_name(name: str) -> str:
    """
    A name without its spaces, dashes, underscores and case, and without the
    suffixes LBL adds to an object name

    :param name: str, the name

    :return: str, the simplified name
    """
    name = str(name).strip().lower()
    # the telluric cleaning adds _tc to the object name
    for suffix in ['_tc', '_corrected']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    for char in [' ', '-', '_', '.']:
        name = name.replace(char, '')
    return name


# =============================================================================
# Define the river plots
# =============================================================================
def ratio_to_velocity(ratio: np.ndarray) -> np.ndarray:
    """
    The velocity of a ratio of wavelengths, relativistic

    :param ratio: np.ndarray, the wavelength over the wavelength at rest

    :return: np.ndarray, the velocity [km/s]
    """
    return speed_of_light_kms * (ratio ** 2 - 1) / (ratio ** 2 + 1)


def velocity_to_wavelength(velocity: np.ndarray, wave_centre: float
                           ) -> np.ndarray:
    """
    The wavelength a velocity puts a line at, relativistic

    :param velocity: np.ndarray, the velocity [km/s]
    :param wave_centre: float, the wavelength at rest [nm]

    :return: np.ndarray, the wavelength [nm]
    """
    beta = np.asarray(velocity) / speed_of_light_kms
    return wave_centre * np.sqrt((1 + beta) / (1 - beta))


def river_plot_data(inst: InstrumentsType, dparams: Dict[str, str],
                    rdata: Dict[str, Any], wave_centre: float
                    ) -> Optional[Dict[str, np.ndarray]]:
    """
    A river plot: the spectra of the star around one wavelength, stacked in
    time, in the rest frame of the star

    The spectra are read with the accessors of LBL (science_files,
    load_science_file, get_wave_solution and get_berv), and the rest frame is
    the one of the compute step: the wavelengths are shifted by -sys_rv, with
    sys_rv = berv - vtot, where vtot is the velocity of the star of that epoch
    (the rdb velocity, which is -vtot in the convention of compute_rv).

    :param inst: Instrument instance
    :param dparams: dict, the directories of this run
    :param rdata: dict, the data of the report
    :param wave_centre: float, the wavelength at the centre [nm]

    :return: dict or None, the velocity grid, the time and the stacked flux
    """
    # -------------------------------------------------------------------------
    # the files to read, and their velocity
    # -------------------------------------------------------------------------
    science_files = rdata['science_files']
    calib_dir = dparams['CALIB_DIR']
    rjd = rdata['river_rjd']
    velocity = rdata['river_velocity']
    berv = rdata['river_berv']
    # the velocity grid of the plot
    width = inst.params['REPORT_RIVER_WIDTH']
    # the step is the pixel of the instrument, roughly
    step = speed_of_light_kms / inst.params['APPROX_RESOLUTION'] / 2.0
    grid = np.arange(-width / 2, width / 2 + step, step)
    # -------------------------------------------------------------------------
    # the spectra, one by one
    # -------------------------------------------------------------------------
    river = np.full((len(science_files), len(grid)), np.nan)
    unread = []
    for it, filename in enumerate(science_files):
        # the science data and its wave solution, the LBL way (a file that
        #   is not there any more, or that cannot be read, is skipped: the
        #   report is written after the run, sometimes long after)
        try:
            sci_image, sci_hdr = inst.load_science_file(filename)
            wavegrid = inst.get_wave_solution(filename, sci_image, sci_hdr)
            # the blaze of this file, the LBL way (lbl_compute, step 6.4)
            blaze = rdata['river_blaze']
            if blaze is None:
                bout = inst.load_blaze_from_science(filename, sci_image,
                                                    sci_hdr, calib_dir)
                blazeimage, blaze_flag = bout
            # test for all ones (no blaze)
            elif np.sum(blaze.ravel()) == len(blaze.ravel()):
                blaze_flag = True
                blazeimage = np.array(blaze)
            else:
                blaze_flag = False
                blazeimage = np.array(blaze)
            # deal with not having a blaze
            if blaze_flag:
                sci_image, blazeimage = inst.no_blaze_corr(sci_image,
                                                           wavegrid)
            # the spectrum, out of the shape of the orders
            with warnings.catch_warnings(record=True) as _:
                sci_image = sci_image / blazeimage
        except Exception as _:
            unread.append(os.path.basename(filename))
            continue
        # the rest frame of the star, as in compute_rv (everything in m/s)
        sys_rv = berv[it] + velocity[it]
        restwave = mp.doppler_shift(wavegrid, -sys_rv)
        # the order that holds the centre with the most margin around it
        order_num = best_order(restwave, wave_centre)
        if order_num is None:
            continue
        # the velocity of each point of that order, and its flux
        owave, oflux = restwave[order_num], sci_image[order_num]
        dvelo = ratio_to_velocity(owave / wave_centre)
        inside = np.abs(dvelo) < width
        if np.sum(inside) < 10:
            continue
        # normalised by its own continuum (the 90th percentile of the window)
        with warnings.catch_warnings(record=True) as _:
            norm = np.nanpercentile(oflux[inside], 90)
        if not np.isfinite(norm) or norm == 0:
            continue
        # on the grid of the plot
        river[it] = np.interp(grid, dvelo[inside], oflux[inside] / norm,
                              left=np.nan, right=np.nan)
    # -------------------------------------------------------------------------
    if len(unread) > 0:
        wmsg = '{0} of the {1} spectra could not be read (the first one is '
        wmsg += '{2})'
        log.warning(wmsg.format(len(unread), len(science_files), unread[0]))
    # nothing read: no plot
    if not np.any(np.isfinite(river)):
        return None
    return dict(grid=grid, rjd=rjd, river=river, wave_centre=wave_centre,
                berv=rdata['river_berv_show'])


def best_order(wavegrid: np.ndarray, wave_centre: float) -> Optional[int]:
    """
    The order of a wave solution that holds a wavelength with the most margin
    on both sides of it

    :param wavegrid: np.ndarray, the wave solution (orders x pixels)
    :param wave_centre: float, the wavelength [nm]

    :return: int or None, the order
    """
    # a one dimensional wave solution has one order
    if wavegrid.ndim == 1:
        wavegrid = wavegrid.reshape(1, -1)
    best, margin = None, 0.0
    for order_num in range(wavegrid.shape[0]):
        owave = wavegrid[order_num]
        with warnings.catch_warnings(record=True) as _:
            wmin, wmax = np.nanmin(owave), np.nanmax(owave)
        if not (wmin < wave_centre < wmax):
            continue
        # the margin on the poorest side
        omargin = min(wave_centre - wmin, wmax - wave_centre)
        if omargin > margin:
            best, margin = order_num, omargin
    return best


def river_bands(inst: InstrumentsType, rdata: Dict[str, Any]
                ) -> List[Tuple[str, float]]:
    """
    The photometric bands of astro.bands that the instrument covers, each with
    the wavelength at its exact middle

    :param inst: Instrument instance
    :param rdata: dict, the data of the report

    :return: list of tuples, the name of the band and its middle wavelength
    """
    # the domain of the instrument
    wavemin = inst.params['COMPIL_WAVE_MIN']
    wavemax = inst.params['COMPIL_WAVE_MAX']
    bands = []
    for band in astro.bands:
        # the exact middle of the band: the mean wavelength of the band when
        #   it is a wavelength of that band, the middle of its two edges
        #   otherwise (a band whose mean or edge is mistyped does not send
        #   the river plots and the line plots to the wrong place)
        middle = 0.5 * (band.minimum + band.maximum)
        if band.minimum < band.mean < band.maximum:
            middle = band.mean
        # only the bands whose middle the instrument covers
        if wavemin < middle < wavemax:
            bands.append((band.name, float(middle)))
    return bands


def river_centres(inst: InstrumentsType, rdata: Dict[str, Any]
                  ) -> List[Tuple[str, int, float]]:
    """
    Three wavelengths per band, spread over the part of the band the
    instrument covers: one river plot each

    :param inst: Instrument instance
    :param rdata: dict, the data of the report

    :return: list of tuples, the band, the rank inside the band (1 to 3) and
             the wavelength [nm]
    """
    wavemin = inst.params['COMPIL_WAVE_MIN']
    wavemax = inst.params['COMPIL_WAVE_MAX']
    centres = []
    for bandname, _ in river_bands(inst, rdata):
        # the part of the band the instrument has
        band = None
        for item in astro.bands:
            if item.name == bandname:
                band = item
        if band is None:
            continue
        low = max(band.minimum, wavemin)
        high = min(band.maximum, wavemax)
        if not high > low:
            continue
        # a quarter, a half and three quarters of the way through it
        for rank, fraction in enumerate([0.25, 0.5, 0.75]):
            centres.append((bandname, rank + 1,
                            float(low + fraction * (high - low))))
    return centres


# =============================================================================
# Define the figures
# =============================================================================
def save_figure(fig: Any, figdir: str, name: str) -> str:
    """
    Save a figure of the report (pdf, so that it can go into a paper as it is)

    :param fig: the matplotlib figure
    :param figdir: str, the directory of the figures
    :param name: str, the name of the figure (without its extension)

    :return: str, the file written
    """
    filename = os.path.join(figdir, '{0}.pdf'.format(name))
    fig.savefig(filename)
    plt.close(fig)
    return filename


def mark_periods(frame: Any, planets: Optional[Table]):
    """
    Mark on a periodogram the periods that are not the star: a day, and the
    year and its first two harmonics, which the sampling and the motion of
    the Earth put in the data. The known planets are marked as well.

    :param frame: the matplotlib axis
    :param planets: astropy Table or None, the planets of exoplanet.eu

    :return: None, draws on the axis
    """
    for period, label in MARKED_PERIODS:
        frame.axvline(period, color='0.4', ls='--', lw=0.8, alpha=0.8)
        frame.text(period, 0.02, ' {0}'.format(label),
                   transform=frame.get_xaxis_transform(), color='0.3',
                   fontsize=6, va='bottom', rotation=90)
    mark_planets(frame, planets)


def mark_planets(frame: Any, planets: Optional[Table]):
    """
    Mark the periods of the known planets on a periodogram

    :param frame: the matplotlib axis
    :param planets: astropy Table or None, the planets of exoplanet.eu

    :return: None, draws on the axis
    """
    if planets is None or 'orbital_period' not in planets.colnames:
        return
    for row in range(len(planets)):
        period = planets['orbital_period'][row]
        if not np.isfinite(period) or period <= 0:
            continue
        frame.axvline(float(period), color='tab:red', ls=':', lw=1.0,
                      alpha=0.8)
        frame.text(float(period), 0.97, ' {0}'.format(planets['name'][row]),
                   transform=frame.get_xaxis_transform(), color='tab:red',
                   fontsize=7, va='top', rotation=90)


def wants_figure(name: str) -> bool:
    """
    Whether an indicator gets a figure of its own

    The velocities of a slice of the domain (vrad_1673nm) and of a part of
    the detector (vrad_h_0-2044) stay in the summary table: there are dozens
    of them and they say what the bands already say.

    :param name: str, the name of the indicator

    :return: bool, True if it gets a figure
    """
    if not name.startswith('vrad_'):
        return True
    suffix = name[5:]
    # a slice of the domain, in nm, or a range of pixels of the detector
    if suffix.endswith('nm') or '-' in suffix:
        return False
    return True


def date_axis(frame: Any):
    """
    The calendar dates of a time series, on top of the frame, with the rjd
    staying at the bottom

    :param frame: the matplotlib frame, once its data are in

    :return: None
    """
    low, high = frame.get_xlim()
    # the ticks of the rjd axis that are inside the plot
    ticks = [tick for tick in frame.get_xticks() if low <= tick <= high]
    if len(ticks) == 0:
        return
    # the same instants, as dates a human reads
    times = base.AstropyTime(np.array(ticks) + 2400000, format='jd')
    labels = [str(item)[:10] for item in np.atleast_1d(times.iso)]
    twin = frame.twiny()
    twin.set_xlim(low, high)
    twin.set_xticks(ticks)
    twin.set_xticklabels(labels, fontsize=8, rotation=30, ha='left')
    twin.set_xlabel('date')
    # the title belongs to the top frame, which the dates now occupy: it
    #   moves to the date axis, above the labels. The room it needs is
    #   measured on the labels themselves (in points, which do not change
    #   when the figure is laid out again) rather than left to matplotlib,
    #   which puts a long title through them
    title = frame.get_title()
    if len(title) == 0:
        return
    frame.set_title('')
    pad = 34.0
    try:
        figure = frame.figure
        renderer = figure.canvas.get_renderer()
        height = 0.0
        for label in twin.get_xticklabels() + [twin.xaxis.label]:
            extent = label.get_window_extent(renderer=renderer)
            height = max(height, extent.height)
        # the labels and the word date, one above the other, in points
        pad = 2.2 * height * 72.0 / figure.dpi + 6.0
    except Exception as _:
        pass
    twin.set_title(title, pad=pad)


def plot_indicator(rdata: Dict[str, Any], indicator: Tuple[str, str, str, str],
                   periods: np.ndarray, figdir: str,
                   planets: Optional[Table], prior: float,
                   figure: bool = True) -> Dict[str, Any]:
    """
    The figure of one indicator: its time series, its periodogram and its FIP

    :param rdata: dict, the data of the report
    :param indicator: tuple, the name, error column, long name and unit
    :param periods: np.ndarray, the periods of the periodogram [days]
    :param figdir: str, the directory of the figures
    :param planets: astropy Table or None, the known planets
    :param prior: float, the prior probability of a signal (FIP)
    :param figure: bool, whether this indicator gets a figure of its own

    :return: dict, the numbers of this indicator
    """
    name, ename, longname, unit = indicator
    # the time series
    time, value, error = time_series(rdata['rdb'], name, ename)
    out = dict(name=name, longname=longname, unit=unit)
    out.update(basic_stats(time, value, error))
    if len(value) < 5:
        out['figure'] = None
        out['best_period'] = np.nan
        out['min_fip'] = np.nan
        out['drift'] = np.nan
        return out
    # the long term drift, taken out before the periodogram so that it does
    #   not spread its power over every period
    dout = remove_drift(time, value, error)
    detrended, drift, sdrift, dcoeffs, dcov, tmean = dout
    out['drift'], out['sdrift'] = drift, sdrift
    # the periodogram and the FIP, without the drift
    power = lomb_scargle(time, detrended, error, periods)
    fip_periods, fip = fip_periodogram(time, detrended, error, periods, prior)
    out['best_period'] = float(periods[np.argmax(power)])
    out['max_power'] = float(np.max(power))
    if len(fip) > 0:
        out['min_fip'] = float(np.min(fip))
        out['fip_period'] = float(fip_periods[np.argmin(fip)])
    else:
        out['min_fip'], out['fip_period'] = np.nan, np.nan
    # the correlation with the radial velocity
    if name != 'vrad' and 'vrad' in rdata['rdb'].colnames:
        rvalue = np.array(rdata['rdb']['vrad'], dtype=float)
        xvalue = np.array(rdata['rdb'][name], dtype=float)
        out['corr_rv'], out['pvalue_rv'] = correlation(xvalue, rvalue)
    else:
        out['corr_rv'], out['pvalue_rv'] = np.nan, np.nan
    # -------------------------------------------------------------------------
    # the figure
    # -------------------------------------------------------------------------
    if not figure:
        out['figure'] = None
        return out
    fig, frames = plt.subplots(2, 1, figsize=(9, 6))
    # the time series, with the drift that was taken out
    frames[0].errorbar(time, value, yerr=error, fmt='.', ms=3, color='k',
                       elinewidth=0.5, capsize=0, alpha=0.5)
    if np.isfinite(drift) and dcov is not None:
        # the line over the whole time span, and its one sigma envelope from
        #   the covariance of its two parameters
        tvec = np.linspace(np.min(time), np.max(time), 100)
        trend = np.polyval(dcoeffs, tvec - tmean)
        design = np.array([tvec - tmean, np.ones(len(tvec))])
        envelope = np.sqrt(np.sum(design * (dcov @ design), axis=0))
        label = ('drift {0:.3g} +- {1:.2g} per day ({2:.1f} sigma)'
                 ''.format(drift, sdrift, abs(drift / sdrift)))
        frames[0].plot(tvec, trend, '-', color='tab:red', lw=1.0, label=label)
        frames[0].fill_between(tvec, trend - envelope, trend + envelope,
                               color='tab:red', alpha=0.2, lw=0,
                               label='1 sigma of the drift')
        frames[0].legend(fontsize=8)
    frames[0].set(xlabel='rjd [days]',
                  ylabel='{0} [{1}]'.format(name, unit),
                  title='{0}: {1} points, rms {2:.4g}, median error '
                        '{3:.4g}'.format(longname, out['n'], out['rms'],
                                         out['error']))
    # faint lines to guide the eye, below the points
    frames[0].grid(color='grey', alpha=0.3, lw=0.5)
    frames[0].set_axisbelow(True)
    # the same axis, in dates, on top (the title moves up with it)
    date_axis(frames[0])
    # the periodogram, with the FIP on the right axis
    frames[1].semilogx(periods, power, '-', color='tab:blue', lw=0.8)
    frames[1].set(xlabel='period [days]',
                  ylabel='periodogram power (drift removed)')
    mark_periods(frames[1], planets)
    if len(fip) > 0:
        twin = frames[1].twinx()
        twin.semilogx(fip_periods, -np.log10(np.maximum(fip, 1e-20)), '-',
                      color='tab:orange', lw=1.0)
        twin.set_ylabel('$-\\log_{10}$(FIP)', color='tab:orange')
        twin.axhline(2, color='tab:orange', ls='--', lw=0.8)
        twin.tick_params(axis='y', labelcolor='tab:orange')
    fig.tight_layout()
    out['figure'] = save_figure(fig, figdir, 'indicator_{0}'.format(name))
    return out


def plot_phase_folds(rdata: Dict[str, Any], planets: Optional[Table],
                     figdir: str) -> List[Dict[str, Any]]:
    """
    The velocities, drift taken out, folded on the period of each planet

    One sinusoid per planet is fitted at once, with the periods held at the
    value of the catalogue. In the fold of a planet the sinusoids of the
    other planets are taken out of the points, so that each fold shows that
    planet alone.

    :param rdata: dict, the data of the report
    :param planets: astropy Table or None, the planets of exoplanet.eu
    :param figdir: str, the directory of the figures

    :return: list of dict, one per planet (its figure and its numbers)
    """
    out = []
    ephemerides = planet_ephemerides(planets)
    if len(ephemerides) == 0:
        return out
    # the velocities, without their drift (the time series of the report)
    time, value, error = time_series(rdata['rdb'], 'vrad', 'svrad')
    if len(time) < 10:
        return out
    detrended = remove_drift(time, value, error)[0]
    # one sinusoid per planet, all at once
    periods = [item['period'] for item in ephemerides]
    coeffs, cov = fit_sinusoids(time, detrended, error, periods)
    if coeffs is None:
        return out
    # what each planet is worth, at every time of the run
    models = []
    for it, period in enumerate(periods):
        angle = 2 * np.pi * time / period
        models.append(coeffs[1 + 2 * it] * np.cos(angle)
                      + coeffs[2 + 2 * it] * np.sin(angle))
    # -------------------------------------------------------------------------
    for it, item in enumerate(ephemerides):
        period = item['period']
        # the other planets, out of the points of this fold
        others = np.zeros(len(time))
        for jt in range(len(models)):
            if jt != it:
                others = others + models[jt]
        points = detrended - coeffs[0] - others
        # the phase, from the epoch of the catalogue when there is one
        t0 = item['t0']
        if not np.isfinite(t0):
            t0 = float(np.min(time))
        phase = np.mod((time - t0) / period, 1.0)
        # the amplitude of this planet, from the fit
        amplitude, samplitude = sinusoid_amplitude(coeffs, cov, it)
        # the figure
        fig, frame = plt.subplots(figsize=(8, 4.5))
        frame.errorbar(phase, points, yerr=error, fmt='.', ms=3, color='k',
                       elinewidth=0.5, capsize=0, alpha=0.5,
                       label='{0} points'.format(len(phase)))
        # the same points, averaged in bins of phase
        bphase, bvalue, berror = phase_bins(phase, points, error)
        if len(bphase) > 0:
            frame.errorbar(bphase, bvalue, yerr=berror, fmt='o', ms=5,
                           color='tab:blue', elinewidth=1.2, capsize=2,
                           label='binned in phase')
        # the sinusoid of this planet
        pgrid = np.linspace(0, 1, 200)
        angle = 2 * np.pi * (t0 + pgrid * period) / period
        curve = (coeffs[1 + 2 * it] * np.cos(angle)
                 + coeffs[2 + 2 * it] * np.sin(angle))
        frame.plot(pgrid, curve, '-', color='tab:red', lw=1.2,
                   label='K = {0:.2f} +- {1:.2f} m/s'.format(amplitude,
                                                             samplitude))
        # the one sigma envelope of that sinusoid, from the covariance of
        #   its two parameters
        if cov is not None:
            first = 1 + 2 * it
            design = np.array([np.cos(angle), np.sin(angle)])
            small = cov[first:first + 2, first:first + 2]
            envelope = np.sqrt(np.sum(design * np.dot(small, design), axis=0))
            frame.fill_between(pgrid, curve - envelope, curve + envelope,
                               color='tab:red', alpha=0.2, lw=0,
                               label='1 sigma of the sinusoid')
        title = '{0}: P = {1:.6g} d, phase 0 at {2}'
        frame.set(xlabel='phase', ylabel='vrad, drift removed [m/s]',
                  xlim=[0, 1],
                  title=title.format(item['name'], period, item['t0_source']))
        frame.grid(color='grey', alpha=0.3, lw=0.5)
        frame.set_axisbelow(True)
        frame.legend(fontsize=8)
        fig.tight_layout()
        name = 'phase_{0}'.format(item['name'].replace(' ', '_'))
        item['figure'] = save_figure(fig, figdir, name)
        item['amplitude'] = amplitude
        item['samplitude'] = samplitude
        item['nothers'] = len(models) - 1
        out.append(item)
    return out


def plot_rv_vs_berv(rdata: Dict[str, Any], figdir: str) -> Dict[str, Any]:
    """
    The radial velocity against the barycentric velocity of the Earth

    :param rdata: dict, the data of the report
    :param figdir: str, the directory of the figures

    :return: dict, the numbers of this section
    """
    rdb = rdata['rdb']
    berv = rdata['berv']
    vrad = np.array(rdb['vrad'], dtype=float)
    svrad = np.array(rdb['svrad'], dtype=float)
    out = dict()
    # the berv is in m/s everywhere in LBL, and in km/s on the figure
    berv = berv / 1000.0
    good = np.isfinite(berv) & np.isfinite(vrad)
    out['n'] = int(np.sum(good))
    if out['n'] < 5:
        out['figure'] = None
        out['corr'], out['pvalue'], out['slope'] = np.nan, np.nan, np.nan
        return out
    out['corr'], out['pvalue'] = correlation(berv, vrad)
    out['berv_min'] = float(np.nanmin(berv))
    out['berv_max'] = float(np.nanmax(berv))
    # the slope of the velocity against the berv (only if the berv moves)
    if out['berv_max'] - out['berv_min'] < 1.0e-6:
        out['figure'], out['slope'] = None, np.nan
        return out
    with warnings.catch_warnings(record=True) as _:
        coeffs, cov = np.polyfit(berv[good], vrad[good], 1, cov=True)
    out['slope'] = float(coeffs[0])
    out['sslope'] = float(np.sqrt(cov[0, 0]))
    # the figure
    fig, frame = plt.subplots(figsize=(7, 5))
    frame.errorbar(berv[good], vrad[good], yerr=svrad[good], fmt='.', ms=4,
                   color='k', elinewidth=0.5, capsize=0, alpha=0.7)
    xvec = np.linspace(np.nanmin(berv), np.nanmax(berv), 100)
    frame.plot(xvec, np.polyval(coeffs, xvec), '-', color='tab:red', lw=1.0,
               label='slope {0:.3f} +- {1:.3f} m/s per km/s'
                     ''.format(out['slope'], out['sslope']))
    # the one sigma envelope of the straight line, from the covariance of
    #   its two parameters
    design = np.array([xvec, np.ones(len(xvec))])
    envelope = np.sqrt(np.sum(design * (cov @ design), axis=0))
    frame.fill_between(xvec, np.polyval(coeffs, xvec) - envelope,
                       np.polyval(coeffs, xvec) + envelope, color='tab:red',
                       alpha=0.2, lw=0, label='1 sigma of the fit')
    frame.set(xlabel='BERV [km/s]', ylabel='radial velocity [m/s]',
              title='Radial velocity against the BERV (correlation '
                    '{0:.3f})'.format(out['corr']))
    frame.legend()
    fig.tight_layout()
    out['figure'] = save_figure(fig, figdir, 'rv_vs_berv')
    return out


def plot_window(rdata: Dict[str, Any], periods: np.ndarray,
                figdir: str, planets: Optional[Table]) -> Dict[str, Any]:
    """
    The spectral window of the observations, and the periodogram of the BERV
    sampled at those same times

    A peak in either of them is a period the sampling can put into the data on
    its own: it is there to be compared with the periodograms of the
    indicators.

    :param rdata: dict, the data of the report
    :param periods: np.ndarray, the periods [days]
    :param figdir: str, the directory of the figures
    :param planets: astropy Table or None, the known planets

    :return: dict, the numbers of this section
    """
    time = np.array(rdata['rdb']['rjd'], dtype=float)
    berv = rdata['berv']
    out = dict()
    # the window of the sampling
    wpower = window_function(time, periods)
    out['window_peak_period'] = float(periods[np.argmax(wpower)])
    out['window_peak'] = float(np.max(wpower))
    # the periodogram of the berv on that sampling
    good = np.isfinite(berv) & np.isfinite(time)
    if np.sum(good) > 5:
        errors = np.full(np.sum(good), 1.0)
        bpower = lomb_scargle(time[good], berv[good], errors, periods)
        out['berv_peak_period'] = float(periods[np.argmax(bpower)])
    else:
        bpower = np.zeros(len(periods))
        out['berv_peak_period'] = np.nan
    # the figure
    fig, frames = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    frames[0].semilogx(periods, wpower, '-', color='k', lw=0.8)
    frames[0].set(ylabel='window power',
                  title='Spectral window of the {0} observations'.format(
                      len(time)))
    mark_periods(frames[0], planets)
    frames[1].semilogx(periods, bpower, '-', color='tab:green', lw=0.8)
    frames[1].set(xlabel='period [days]', ylabel='periodogram power',
                  title='BERV sampled at the times of the observations')
    mark_periods(frames[1], planets)
    fig.tight_layout()
    out['figure'] = save_figure(fig, figdir, 'window_and_berv')
    return out


def plot_river(river: Dict[str, np.ndarray], bandname: str, index: int,
               figdir: str) -> str:
    """
    The river plot of one wavelength: the spectra stacked, in the rest frame
    of the star, sorted in BERV

    :param river: dict, the output of river_plot_data
    :param bandname: str, the name of the band
    :param index: int, the rank of this wavelength inside the band
    :param figdir: str, the directory of the figures

    :return: str, the file written
    """
    grid, rjd, image = river['grid'], river['rjd'], river['river']
    berv = np.array(river['berv'], dtype=float)
    # the spectra are sorted in BERV: the telluric lines of the Earth then
    #   walk across the plot, while the lines of the star stay put
    sorted_by = 'BERV'
    if np.nanmax(berv) - np.nanmin(berv) < 1.0e-6 or not np.all(
            np.isfinite(berv)):
        # a mode whose wave solution is already barycentric, and no BERV in
        #   the headers either: time is all there is to sort on
        berv, sorted_by = np.array(rjd, dtype=float), 'time'
    order = np.argsort(berv)
    image, berv = image[order], berv[order]
    # the median spectrum, and what each spectrum has that it does not
    with warnings.catch_warnings(record=True) as _:
        median = np.nanmedian(image, axis=0)
        residual = image - median
        # each spectrum is brought back to zero as well: an epoch that sits
        #   above or below the others (the flux calibration of that night,
        #   say) is not what the residual river plot is about
        residual -= np.nanmedian(residual, axis=1)[:, None]
        vmin, vmax = np.nanpercentile(image, [2, 98])
        rmin, rmax = np.nanpercentile(residual, [5, 95])
    # the residuals are shown around zero, in units of the median spectrum
    rlimit = max(abs(rmin), abs(rmax))
    # three panels of the same width: the spectra, their residuals and the
    #   median, each with a column for its colour bar (the last one empty)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9),
                             height_ratios=[3, 3, 1], width_ratios=[40, 1])
    frames, cframes = axes[:, 0], axes[:, 1]
    cframes[2].axis('off')
    extent = [grid[0], grid[-1], 0, len(berv)]
    spcimage = frames[0].imshow(image, aspect='auto', origin='lower',
                                cmap='inferno', vmin=vmin, vmax=vmax,
                                interpolation='nearest', extent=extent)
    frames[0].set(ylabel='spectrum (sorted in {0})'.format(sorted_by))
    frames[0].set_title('{0} band, {1:.2f} nm, rest frame of the star, '
                        '{2} spectra'.format(bandname, river['wave_centre'],
                                             len(berv)), fontsize=10)
    cbar = fig.colorbar(spcimage, cax=cframes[0])
    cbar.set_label('flux / continuum')
    resimage = frames[1].imshow(residual, aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-rlimit, vmax=rlimit,
                                interpolation='nearest', extent=extent)
    frames[1].set(ylabel='spectrum (sorted in {0})'.format(sorted_by))
    frames[1].set_title('median spectrum and row medians taken out '
                        '(5 to 95 percentile)', fontsize=9)
    # the colour bar of the residuals, in units of the median spectrum
    cbar = fig.colorbar(resimage, cax=cframes[1])
    cbar.set_label('residual / median spectrum')
    # the median spectrum, in wavelength, over the same width as the images
    wavegrid = velocity_to_wavelength(grid, river['wave_centre'])
    with warnings.catch_warnings(record=True) as _:
        frames[2].plot(wavegrid, median, '-', color='k', lw=0.8)
    frames[2].set(xlabel='wavelength [nm]', ylabel='median flux',
                  xlim=[wavegrid[0], wavegrid[-1]])
    frames[0].set(xticklabels=[])
    frames[1].set(xlabel='velocity [km/s]')
    fig.tight_layout()
    name = 'river_{0}{1}'.format(bandname, index)
    return save_figure(fig, figdir, name)


# =============================================================================
# Define the report itself
# =============================================================================
def cite(*keys: str) -> str:
    """
    The number of one or more references, as cited in the text

    :param keys: str, the keys of REFERENCES

    :return: str, the citation, such as [2, 3]
    """
    order = [key for key, _ in REFERENCES]
    numbers = [str(order.index(key) + 1) for key in keys if key in order]
    return '[{0}]'.format(', '.join(numbers))


def bibliography() -> str:
    """
    The reference section at the end of the report

    :return: str, the LaTeX
    """
    lines = ['\\section*{References}', '\\begin{enumerate}',
             '\\setlength{\\itemsep}{1pt}']
    for _, text in REFERENCES:
        lines.append('\\item %s' % text)
    lines.append('\\end{enumerate}')
    return '\n'.join(lines)


def latex_escape(text: str) -> str:
    """
    Escape the characters of a text that LaTeX would read as its own

    :param text: str, the text

    :return: str, the text for LaTeX
    """
    out = str(text)
    for char in ['\\', '&', '%', '$', '#', '_', '{', '}']:
        out = out.replace(char, '\\' + char)
    return out.replace('~', '\\textasciitilde ').replace('^', '\\^{}')


def latex_link(text: str, url: str) -> str:
    """
    A piece of text, as a link when there is a url

    :param text: str, the text
    :param url: str, the url (can be empty)

    :return: str, the LaTeX
    """
    if len(text.strip()) == 0:
        return '--'
    if len(url.strip()) == 0:
        return latex_escape(text)
    return '\\href{%s}{%s}' % (url.replace('%', '\\%'),
                              latex_escape(text))


def number(value: Any, fmt: str = '{0:.4g}') -> str:
    """
    A number for a LaTeX table, or a dash when there is none

    :param value: the value
    :param fmt: str, the format

    :return: str, the number
    """
    try:
        if value is None or not np.isfinite(value):
            return '--'
        return fmt.format(value)
    except (TypeError, ValueError):
        return '--'


def disclaimer() -> str:
    """
    The warning at the top of the report: it is a starting point, not a result

    :return: str, the LaTeX
    """
    lines = ['\\begin{center}',
             '\\setlength{\\fboxrule}{1.2pt}',
             '\\fcolorbox{red}{red!5}{\\parbox{0.93\\textwidth}{\\centering',
             '{\\large \\textbf{This is just to get started with the '
             'analysis.}}\\\\[5pt]',
             '{\\large \\textbf{\\textcolor{red}{Do \\emph{not} publish these '
             'results out of the box without checking them yourself.}}}'
             '\\\\[7pt]',
             'Everything here is automatic: the periodograms, the FIPs, the '
             'planets and the drifts are what the numbers say, not what a '
             'human has vetted. Look at the spectra, at the outliers and at '
             'the systematics before believing any of it.\\\\[7pt]',
             '\\textbf{And if you do publish it,} please cite %s.'
             % papers_to_cite(),
             '}}',
             '\\end{center}',
             '\\vspace{6pt}']
    return '\n'.join(lines)


def papers_to_cite() -> str:
    """
    The papers to cite, as they are written in a paper: the author and the
    year, linked to their page on ADS

    :return: str, the LaTeX
    """
    items = []
    for what, citation, _, bibcode in CITE_PAPERS:
        items.append('{0} for {1}'
                     ''.format(latex_link(citation,
                                          URL_ADS.format(bibcode)), what))
    return '{0} and {1}'.format(', '.join(items[:-1]), items[-1])


def latex_header(title: str, subtitle: str) -> str:
    """
    The preamble and the title of the report

    :param title: str, the title
    :param subtitle: str, the line below the title

    :return: str, the LaTeX
    """
    lines = ['\\documentclass[11pt,a4paper]{article}',
             '\\usepackage[margin=2.2cm]{geometry}',
             '\\usepackage{graphicx}',
             '\\usepackage{longtable}',
             '\\usepackage{booktabs}',
             '\\usepackage{pdflscape}',
             '\\usepackage[colorlinks=true,urlcolor=blue]{hyperref}',
             '\\usepackage{float}',
             '\\usepackage{xcolor}',
             '\\setlength{\\parindent}{0pt}',
             '\\setlength{\\parskip}{4pt}',
             '\\begin{document}',
             '\\begin{center}',
             '{\\LARGE \\textbf{%s}}\\\\[4pt]' % title,
             '{\\large %s}' % subtitle,
             '\\end{center}',
             '\\vspace{4pt}', '\\hrule', '\\vspace{8pt}',
             disclaimer()]
    return '\n'.join(lines)


def latex_table(caption: str, header: List[str], rows: List[List[str]],
                align: Optional[str] = None,
                size: Optional[str] = None,
                landscape: bool = False) -> str:
    """
    A table of the report

    :param caption: str, the caption
    :param header: list of str, the header of each column
    :param rows: list of list of str, the rows
    :param align: str or None, the alignment of the columns
    :param size: str or None, a LaTeX font size for the table (footnotesize)
    :param landscape: bool, if True the table gets a page of its own, sideways

    :return: str, the LaTeX
    """
    if align is None:
        align = 'l' + 'r' * (len(header) - 1)
    lines = []
    if landscape:
        lines.append('\\begin{landscape}')
    if size is not None:
        lines.append('{\\%s' % size)
    lines += ['\\begin{longtable}{%s}' % align,
             '\\caption{%s}\\\\' % caption,
             '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule', '\\endfirsthead',
             '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule',
             '\\endhead']
    for row in rows:
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{longtable}']
    if size is not None:
        lines.append('}')
    if landscape:
        lines.append('\\end{landscape}')
    return '\n'.join(lines)


def latex_figure(filename: str, caption: str, width: str = '0.95') -> str:
    """
    A figure of the report

    :param filename: str, the file of the figure
    :param caption: str, the caption
    :param width: str, the width as a fraction of the text

    :return: str, the LaTeX
    """
    if filename is None:
        return ''
    name = os.path.basename(filename)
    lines = ['\\begin{figure}[H]', '\\centering',
             '\\includegraphics[width=%s\\textwidth]{figures/%s}' % (width,
                                                                     name),
             '\\caption{%s}' % caption, '\\end{figure}']
    return '\n'.join(lines)


def compile_pdf(texfile: str) -> Optional[str]:
    """
    Compile the report with pdflatex (twice, for the table of contents and the
    references of the figures)

    :param texfile: str, the LaTeX file

    :return: str or None, the pdf file, or None if pdflatex is not there
    """
    if shutil.which('pdflatex') is None:
        wmsg = ('pdflatex was not found: the report is in {0}, compile it '
                'where a LaTeX distribution is installed')
        log.warning(wmsg.format(texfile))
        return None
    directory = os.path.dirname(texfile)
    command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
               os.path.basename(texfile)]
    for _ in range(2):
        process = subprocess.run(command, cwd=directory,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        if process.returncode != 0:
            # the last lines of the log say what LaTeX did not like
            output = process.stdout.decode('utf-8', errors='replace')
            log.warning('pdflatex failed:\n{0}'.format(output[-2000:]))
            return None
    return texfile.replace('.tex', '.pdf')


def make_tarball(figures: List[str], tarfilename: str) -> str:
    """
    Bundle the figures in a tar archive, so that they can go into a paper

    :param figures: list of str, the figures
    :param tarfilename: str, the archive to write

    :return: str, the archive written
    """
    with tarfile.open(tarfilename, 'w:gz') as archive:
        for figure in figures:
            if figure is None or not os.path.exists(figure):
                continue
            archive.add(figure, arcname=os.path.basename(figure))
    return tarfilename


# =============================================================================
# Define the sections of the report
# =============================================================================
def make_report(inst: InstrumentsType, dparams: Dict[str, str]) -> str:
    """
    Write the report of this object: the pdf and the tar archive of its
    figures

    :param inst: Instrument instance
    :param dparams: dict, the directories of this run

    :return: str, the report directory
    """
    params = inst.params
    # -------------------------------------------------------------------------
    # the directories of the report
    # -------------------------------------------------------------------------
    report_dir = io.make_dir(dparams['DATA_DIR'], params['REPORT_SUBDIR'],
                             'Report')
    objname = '{0}_{1}'.format(params['OBJECT_SCIENCE'],
                               params['OBJECT_COMPARISON'])
    outdir = io.make_dir(report_dir, objname, 'Report', verbose=False)
    figdir = io.make_dir(outdir, 'figures', 'Report figures', verbose=False)
    # the figures of an older run are taken out, so that the directory and
    #   the tar archive hold this report and nothing else
    for oldfig in os.listdir(figdir):
        if oldfig.endswith('.pdf'):
            os.remove(os.path.join(figdir, oldfig))
    # -------------------------------------------------------------------------
    # the data, and the periods of the periodograms
    # -------------------------------------------------------------------------
    log.general('Reading the data of the report')
    rdata = get_report_data(inst, dparams)
    time = np.array(rdata['rdb']['rjd'], dtype=float)
    periods = period_grid(time, params)
    prior = params['REPORT_FIP_PRIOR']
    # the target in SIMBAD (its coordinates are what the planets are matched
    #   on) and the known planets of this star
    target = simbad_target(rdata['object'])
    rdata['target'] = target
    planets = None
    if params['REPORT_EXOPLANET_EU']:
        planets = exoplanet_eu_planets(inst, rdata['object'], report_dir,
                                       target=target)
    # storage of the LaTeX and of the figures
    body, figures = [], []
    # -------------------------------------------------------------------------
    # 1. what was processed
    # -------------------------------------------------------------------------
    body.append('\\section{What was processed}')
    span = float(np.nanmax(time) - np.nanmin(time))
    nights = len(np.unique(np.floor(time)))
    # the first and the last observation, in human dates as well
    first = base.AstropyTime(np.nanmin(time) + 2400000, format='jd').iso[:19]
    last = base.AstropyTime(np.nanmax(time) + 2400000, format='jd').iso[:19]
    # the first column is the name LBL gives to each thing (the keys of the
    #   wrap script), so that the report and the run speak the same language
    rows = [['\\texttt{OBJECT\\_SCIENCE} (the star)',
             latex_escape(rdata['object'])],
            ['\\texttt{OBJECT\\_COMPARISON} (the template)',
             latex_escape(rdata['template_object'])],
            ['\\texttt{INSTRUMENT}',
             latex_escape(rdata['instrument_name'])],
            ['\\texttt{DATA\\_SOURCE}', latex_escape(rdata['data_source'])],
            ['\\texttt{DATA\\_TYPE}', latex_escape(rdata['data_type'])]]
    # what SIMBAD knows about this star
    if target is not None:
        rows += [['SIMBAD name', latex_escape(target['main_id'])],
                 ['Coordinates (ICRS)',
                  '{0:.6f} {1:+.6f} (deg)'.format(target['ra'],
                                                  target['dec'])],
                 ['Spectral type', latex_escape(target['sp_type'])]]
    rows += [
            ['Spectra', '{0}'.format(len(time))],
            ['Nights', '{0}'.format(nights)],
            ['First observation',
             '{0} (rjd {1:.4f})'.format(first, np.nanmin(time))],
            ['Last observation',
             '{0} (rjd {1:.4f})'.format(last, np.nanmax(time))],
            ['Time span', '{0:.1f} days'.format(span)],
            ['Template file', latex_escape(rdata['template_file'])],
            ['Mask file', latex_escape(rdata['mask_file'])],
            ['Systemic velocity of the mask',
             '{0} m/s'.format(number(rdata['systemic_velocity']))],
            ['rdb file', latex_escape(os.path.basename(rdata['rdbfile']))],
            ['LBL version', latex_escape(base.__version__)]]
    body.append(latex_table('What this report is about. The names in '
                            'typewriter font are the LBL parameters of the '
                            'run, the ones of the wrap script',
                            ['Item', 'Value'], rows, align='ll'))
    # -------------------------------------------------------------------------
    # 2. the known planets
    # -------------------------------------------------------------------------
    body.append('\\section{Known and suspected planets}')
    nasa = None
    if target is None:
        matched = ('matched on the name of the star (SIMBAD {0} did not '
                   'answer)'.format(cite('simbad')))
    else:
        matched = ('matched on the position of the star from SIMBAD {3} '
                   '({0:.5f} {1:+.5f}, within {2:.0f} arcsec) and on its '
                   'names'.format(target['ra'], target['dec'], MATCH_RADIUS,
                                  cite('simbad')))
    body.append('From the catalogue of exoplanet.eu, \\url{%s}, %s.'
                % (URL_EXOPLANET_EU, matched))
    if planets is None:
        body.append('No planet of this star is in the catalogue (which does '
                    'not mean there is none: the name may be written '
                    'differently there).')
    else:
        # the papers behind them, from the NASA exoplanet archive
        hostnames = [str(name) for name in np.unique(planets['star_name'])]
        if target is not None:
            hostnames += [target['main_id']] + target.get('ids', [])
        nasa = nasa_archive_planets(hostnames)
        header = ['Planet', 'Period [d]', 'Mass [M$_{\\rm Jup}$]',
                  'K [m/s]', 'Detection', 'Status', 'Discovery',
                  'Parameters']
        rows = []
        for row in range(len(planets)):
            def _get(colname, fmt='{0:.4g}'):
                if colname not in planets.colnames:
                    return '--'
                return number(planets[colname][row], fmt)
            # the row of this planet in the NASA archive, by name
            nrow = None
            if nasa is not None:
                for it in range(len(nasa)):
                    if same_planet(str(planets['name'][row]),
                                   str(nasa['pl_name'][it])):
                        nrow = it
                        break
            # the amplitude of its signal and the two references
            if nrow is None:
                amplitude, discovery, parameters = '--', '--', '--'
                # the catalogue says at least when it was announced
                year = _getstr(planets, 'discovered', row)
                publication = _getstr(planets, 'publication', row)
                if year != '--':
                    discovery = latex_escape('{0} ({1})'.format(publication,
                                                                year))
            else:
                amplitude = number(nasa['pl_rvamp'][nrow], '{0:.2f}')
                disc = reference_text(nasa['disc_refname'][nrow])
                pref = reference_text(nasa['pl_refname'][nrow])
                discovery = latex_link(*disc)
                parameters = latex_link(*pref)
            rows.append([latex_escape(planets['name'][row]),
                         _get('orbital_period'), _get('mass'), amplitude,
                         latex_escape(_getstr(planets, 'detection_type', row)),
                         latex_escape(_getstr(planets, 'planet_status', row)),
                         discovery, parameters])
        body.append(latex_table('The planets of this star. The periods, the '
                                'masses and the status come from '
                                'exoplanet.eu; K, the amplitude of the '
                                'velocity signal, and the two references come '
                                'from the NASA exoplanet archive (the '
                                'reference of the discovery and the one of '
                                'the parameters kept there). The references '
                                'link to ADS', header, rows,
                                align='lrrrlllll', size='footnotesize',
                                landscape=True))
        body.append('Their periods are marked on every periodogram of this '
                    'report. K says what the velocities of this run would '
                    'have to reach to see them. The catalogues are %s and '
                    '%s.' % (cite('exoplaneteu'), cite('nasa')))
    # -------------------------------------------------------------------------
    # 3. the phase folds of the known planets
    # -------------------------------------------------------------------------
    folds = plot_phase_folds(rdata, planets, figdir)
    if len(folds) > 0:
        body.append('\\section{The velocities folded on the planets}')
        text = ('The velocities of the run, with their long term drift taken '
                'out, folded on the period of each planet. The periods are '
                'held at the value of the catalogue; the amplitude and the '
                'phase of every planet are fitted at once, one sinusoid '
                'each, with the error bars as weights. ')
        if len(folds) > 1:
            text += ('The fold of a planet has the sinusoids of the other '
                     '{0} taken out of its points, so that each fold shows '
                     'that planet alone. '.format(len(folds) - 1))
        text += ('The blue points are the average of the black ones in bins '
                 'of phase, and the red curve is the sinusoid of the fit, '
                 'with the one sigma envelope of its two parameters. A '
                 'circular orbit is all this assumes: an eccentric planet '
                 'does not look like its curve.')
        body.append(text)
        # what the fit says, next to what the catalogue says
        rows = []
        for fold in folds:
            catalogue = '--'
            if nasa is not None:
                for it in range(len(nasa)):
                    if same_planet(fold['name'], str(nasa['pl_name'][it])):
                        catalogue = number(nasa['pl_rvamp'][it], '{0:.2f}')
                        break
            if np.isfinite(fold['samplitude']) and fold['samplitude'] > 0:
                fitted = '{0:.2f} +- {1:.2f}'.format(fold['amplitude'],
                                                     fold['samplitude'])
                sigma = '{0:.1f}'.format(fold['amplitude']
                                         / fold['samplitude'])
            else:
                fitted, sigma = '{0:.2f}'.format(fold['amplitude']), '--'
            rows.append([latex_escape(fold['name']),
                         '{0:.6g}'.format(fold['period']), fitted, sigma,
                         catalogue, latex_escape(fold['t0_source'])])
        body.append(latex_table('What the velocities of this run say about '
                                'each planet, with its period held at the '
                                'value of the catalogue. K is the amplitude '
                                'of the fitted sinusoid, and the last '
                                'columns are the amplitude of the NASA '
                                'archive and the epoch the fold starts from',
                                ['Planet', 'Period [d]', 'K fitted [m/s]',
                                 'K / error', 'K catalogue [m/s]',
                                 'Phase 0 at'], rows,
                                align='lrrrrl', size='footnotesize'))
        for fold in folds:
            caption = ('{0} folded on {1:.6g} days, the drift of the run '
                       'taken out'.format(latex_escape(fold['name']),
                                          fold['period']))
            if fold['nothers'] > 0:
                caption += ', and the other planets as well'
            caption += ('. Black: every point. Blue: the average in bins of '
                        'phase. Red: the fitted sinusoid, with its one '
                        'sigma envelope.')
            body.append(latex_figure(fold['figure'], caption))
            figures.append(fold['figure'])
    # -------------------------------------------------------------------------
    # 4. the indicators, one by one
    # -------------------------------------------------------------------------
    body.append('\\section{Radial velocity and the other indicators}')
    body.append('Each indicator gets its time series, its periodogram %s '
                'and its false inclusion probability (FIP) %s. The FIP of a '
                'period interval is one minus the probability that a signal '
                'is in it: a FIP below 0.01, the dashed line of the figures, '
                'is the usual threshold of a detection.'
                % (cite('lomb', 'scargle', 'gls'), cite('fip')))
    log.general('Measuring the indicators')
    results = []
    for indicator in rdata['indicators']:
        result = plot_indicator(rdata, indicator, periods, figdir, planets,
                                prior, figure=wants_figure(indicator[0]))
        results.append(result)
    # the summary table of every indicator
    header = ['Indicator', 'N', 'Median', 'rms', 'Robust', 'Error',
              'Excess', 'Drift per day', 'Drift error', 'Drift sigma',
              'Corr. RV', 'Period [d]', 'Min FIP']
    rows = []
    for result in results:
        rows.append([latex_escape(result['name']), '{0}'.format(result['n']),
                     number(result.get('median')), number(result.get('rms')),
                     number(result.get('robust_rms')),
                     number(result.get('error')),
                     number(result.get('excess')),
                     number(result.get('drift'), '{0:.3g}'),
                     number(result.get('sdrift'), '{0:.2g}'),
                     number(abs(result.get('drift', np.nan) /
                                result.get('sdrift', np.nan))
                            if result.get('sdrift') else np.nan,
                            '{0:.1f}'),
                     number(result.get('corr_rv'), '{0:.2f}'),
                     number(result.get('best_period'), '{0:.3f}'),
                     number(result.get('min_fip'), '{0:.2e}')])
    body.append(latex_table('Every indicator of the rdb file. Robust is the '
                            'robust dispersion, Error the median error bar, '
                            'Excess the dispersion the error bars do not '
                            'explain, Corr. RV the correlation with the '
                            'radial velocity, and Period the strongest period '
                            'of the periodogram, which is measured after the '
                            'drift is taken out', header, rows,
                            size='footnotesize', landscape=True))
    # the figures, the most telling ones first (the radial velocity, then the
    #   indicators with the lowest FIP)
    def _fip_key(res):
        fip = res.get('min_fip')
        return 1.0 if fip is None or not np.isfinite(fip) else fip
    ordered = sorted(results, key=lambda res: (res['name'] != 'vrad',
                                               _fip_key(res)))
    nplots = 0
    for result in ordered:
        if result.get('figure') is None or nplots >= MAX_INDICATOR_PLOTS:
            continue
        nplots += 1
        caption = ('{0} ({1}). Top: the time series. Bottom: the periodogram '
                   '(blue) and $-\\log_{{10}}$(FIP) (orange); the dashed line '
                   'is FIP = 0.01.'.format(latex_escape(result['longname']),
                                           latex_escape(result['name'])))
        body.append(latex_figure(result['figure'], caption))
        figures.append(result['figure'])
    # -------------------------------------------------------------------------
    # 5. the radial velocity against the berv
    # -------------------------------------------------------------------------
    body.append('\\section{Radial velocity against the BERV}')
    body.append('The BERV of each spectrum comes from %s, read from the '
                'header of its lblrv file. A slope here means the velocities '
                'move with the ' % latex_escape(rdata['berv_source']))
    body.append('position of the Earth, which is the signature of something '
                'fixed in the frame of the observatory (tellurics, sky '
                'lines, a wavelength solution).')
    bervout = plot_rv_vs_berv(rdata, figdir)
    rows = [['Points', '{0}'.format(bervout['n'])],
            ['BERV range', '{0} to {1} km/s'.format(
                number(bervout.get('berv_min')),
                number(bervout.get('berv_max')))],
            ['Correlation with the velocity',
             number(bervout.get('corr'), '{0:.3f}')],
            ['p-value', number(bervout.get('pvalue'), '{0:.2e}')],
            ['Slope', '{0} m/s per km/s'.format(number(bervout.get('slope')))]]
    body.append(latex_table('Radial velocity against the BERV',
                            ['Item', 'Value'], rows, align='ll'))
    body.append(latex_figure(bervout['figure'],
                             'Radial velocity against the BERV, with the '
                             'straight line fitted to it.'))
    figures.append(bervout['figure'])
    # -------------------------------------------------------------------------
    # 6. the window of the sampling, and the berv on that sampling
    # -------------------------------------------------------------------------
    body.append('\\section{Sampling}')
    body.append('The spectral window says which periods the sampling alone '
                'can put in the data; the periodogram of the BERV taken at '
                'the times of the observations says the same for the yearly '
                'motion of the Earth. A peak in the periodogram of an '
                'indicator that is also here is not a signal of the star.')
    windout = plot_window(rdata, periods, figdir, planets)
    rows = [['Strongest period of the window',
             '{0} days'.format(number(windout.get('window_peak_period'),
                                      '{0:.3f}'))],
            ['Strongest period of the BERV',
             '{0} days'.format(number(windout.get('berv_peak_period'),
                                      '{0:.3f}'))]]
    body.append(latex_table('The sampling of the observations',
                            ['Item', 'Value'], rows, align='ll'))
    body.append(latex_figure(windout['figure'],
                             'Top: the spectral window of the observations. '
                             'Bottom: the BERV sampled at the times of the '
                             'observations.'))
    figures.append(windout['figure'])
    # -------------------------------------------------------------------------
    # 7. the river plots
    # -------------------------------------------------------------------------
    body.append('\\section{River plots}')
    width = params['REPORT_RIVER_WIDTH']
    body.append('For each photometric band the instrument covers, three '
                'wavelengths of that band (a quarter, a half and three '
                'quarters of the way through it), each over %.0f km/s, '
                'stacked and put in the rest frame of the star. The spectra '
                'are sorted in BERV rather than in time, so that what '
                'belongs to the Earth walks across the plot while the lines '
                'of the star stay put. The spectra, their wavelength '
                'solution, their blaze and their BERV are read with the '
                'accessors of LBL, the spectra are divided by the blaze as '
                'lbl\\_compute loads it, and the rest frame is the one of '
                'the compute step.' % width)
    river_outputs = river_plots(inst, dparams, rdata, figdir)
    for bandname, figure, nspectra, wave_centre in river_outputs:
        caption = ('{0} band, centred on {1:.2f} nm, {2} spectra. Top: each '
                   'row is a spectrum, sorted in BERV. Middle: the same, '
                   'with the median spectrum taken out and each row brought '
                   'back to its own median, so that an epoch sitting above '
                   'or below the others does not hide the residuals. '
                   'Bottom: the median spectrum.'
                   ''.format(bandname, wave_centre, nspectra))
        body.append(latex_figure(figure, caption))
        figures.append(figure)
    if len(river_outputs) == 0:
        body.append('No band of the instrument could be read for the river '
                    'plots.')
    # -------------------------------------------------------------------------
    # 8. the lines of one file
    # -------------------------------------------------------------------------
    body.append('\\section{The lines of one spectrum}')
    body.append('The debug plot of lbl\\_compute (PLOT\\_COMPUTE\\_LINES), '
                'for the spectrum whose signal to noise is the median of the '
                'run. Its velocities are computed again here, with the same '
                'mask, reference table, template splines and blaze as the '
                'run. Each line of the mask is drawn in its own colour, so '
                'the edges of the lines are where the colours change. One '
                'order per photometric band is drawn, the order closest to '
                'the middle of the band, over %.0f resolution elements at '
                'the middle of that order (the resolution of the instrument '
                'is a parameter of the run, %.0f here). The dashed lines are '
                'the edges of the lines of the mask.'
                % (LINE_PLOT_ELEMENTS, params['APPROX_RESOLUTION']))
    linefigs, linefile = line_edge_plot(inst, dparams, rdata, figdir)
    if len(linefigs) == 0:
        body.append('The plot could not be made for this run.')
    for linefig, band, order, wmin, wmax, nlines in linefigs:
        if band == '':
            where = 'order {0}'.format(order)
        else:
            where = 'the {0} band (order {1})'.format(band, order)
        caption = ('The lines of {0} in {1}, over {5:.0f} resolution '
                   'elements at the middle of the order ({2:.2f} to '
                   '{3:.2f} nm, {4} lines). Top: the spectrum, line by line, '
                   'over the template (grey). Bottom: the difference between '
                   'the two.'.format(latex_escape(linefile), where, wmin,
                                     wmax, nlines, LINE_PLOT_ELEMENTS))
        body.append(latex_figure(linefig, caption))
        figures.append(linefig)
    # -------------------------------------------------------------------------
    # 9. how the numbers were obtained
    # -------------------------------------------------------------------------
    body.append('\\section{How the numbers were obtained}')
    body.append('\\textbf{Periodogram.} The floating mean periodogram, with '
                'the error bars as weights %s. A straight line in time is '
                'fitted to every indicator and taken out before it, so that '
                'a drift does not spread its power over every period; the '
                'drift, its uncertainty and its significance are in the '
                'table of the indicators and on their time series.'
                % cite('lomb', 'scargle', 'gls'))
    body.append('\\textbf{FIP.} %s The probability that a signal is present '
                'with a period in a given interval is computed here with a '
                'single sinusoid: at each period a model with a sine and an '
                'offset is compared with a model with the offset alone, the '
                'linear parameters are integrated out analytically with wide '
                'gaussian priors, the periods are taken with a prior flat in '
                'the logarithm of the period, and the dispersion the error '
                'bars do not explain is added to them as a jitter. The bins '
                'have the width of a peak of the periodogram. This is the '
                'single-signal version of the framework: several signals are '
                'not searched at once.' % cite('fip'))
    body.append('\\textbf{Chromatic index and line width.} %s LBL measures '
                'the same quantities: CRX is the slope of the velocity with '
                'the logarithm of the wavelength, and the second derivative '
                'of the line profile (d2v, dW) is its differential line '
                'width.' % cite('serval'))
    body.append('\\textbf{Known planets.} The catalogue of exoplanet.eu %s, '
                'downloaded in full and matched on the position of the star '
                'and on its names; the papers behind them and the amplitude '
                'of their signal come from the NASA exoplanet archive %s. '
                'The position, the spectral type and the names of the star '
                'come from SIMBAD %s, and the stellar model LBL uses for the '
                'systemic velocity of the mask is a PHOENIX model %s. The '
                'velocities themselves come from LBL %s.'
                % (cite('exoplaneteu'), cite('nasa'), cite('simbad'),
                   cite('phoenix'), cite('lbl')))
    body.append('\\textbf{The papers to cite.} The velocities of this run '
                'come from LBL %s, the temperatures from the DTemp method '
                '%s, and the spectra, for SPIRou and NIRPS, from APERO %s.'
                % (cite('lbl'), cite('dtemp'), cite('apero')))
    # -------------------------------------------------------------------------
    # 10. the references
    # -------------------------------------------------------------------------
    body.append(bibliography())
    # -------------------------------------------------------------------------
    # write the LaTeX, compile it and bundle the figures
    # -------------------------------------------------------------------------
    title = 'LBL report: {0}'.format(latex_escape(rdata['object']))
    subtitle = '{0}, {1} spectra, {2} nights, from {3} to {4}'.format(
        latex_escape(rdata['instrument']), len(time), nights, first[:10],
        last[:10])
    subtitle += '\\\\[2pt] written on {0}'.format(base.Time.now().iso[:19])
    texfile = os.path.join(outdir, 'lbl_report_{0}.tex'.format(objname))
    with open(texfile, 'w') as tex:
        tex.write(latex_header(title, subtitle))
        tex.write('\n\n' + '\n\n'.join(body) + '\n\n')
        tex.write('\\end{document}\n')
    log.general('Writing {0}'.format(texfile))
    # compile it
    pdffile = compile_pdf(texfile)
    if pdffile is not None:
        log.info('Report: {0}'.format(pdffile))
    # bundle the figures
    tarname = 'lbl_figures_{0}.tar.gz'.format(objname)
    tarfilename = os.path.join(outdir, tarname)
    make_tarball(figures, tarfilename)
    log.info('Figures: {0}'.format(tarfilename))
    # return the directory of the report
    return outdir


def _getstr(table: Table, colname: str, row: int) -> str:
    """
    A string of a table, or a dash when there is none

    :param table: astropy Table, the table
    :param colname: str, the column
    :param row: int, the row

    :return: str, the value
    """
    if colname not in table.colnames:
        return '--'
    value = str(table[colname][row]).strip()
    return value if len(value) > 0 and value != 'nan' else '--'


def river_plots(inst: InstrumentsType, dparams: Dict[str, str],
                rdata: Dict[str, Any], figdir: str
                ) -> List[Tuple[str, str, int, float]]:
    """
    The river plots of every band the instrument covers

    The science files are read once for all the bands (they are the slow part)
    and they are sub-sampled evenly in time above REPORT_MAX_RIVER_FILES.

    :param inst: Instrument instance
    :param dparams: dict, the directories of this run
    :param rdata: dict, the data of the report
    :param figdir: str, the directory of the figures

    :return: list of tuples, the band, the figure, the number of spectra and
             the central wavelength
    """
    # three wavelengths per band the instrument covers
    centres = river_centres(inst, rdata)
    if len(centres) == 0:
        return []
    # -------------------------------------------------------------------------
    # the files to read: those of the rdb file, sub-sampled evenly in time
    # -------------------------------------------------------------------------
    rdb = rdata['rdb']
    science_dir = dparams['SCIENCE_DIR']
    # the science files, the LBL way
    all_files = inst.science_files(science_dir)
    # the basename of each science file, to find the one of each rdb row
    basenames = dict()
    for filename in all_files:
        basenames[os.path.basename(filename)] = filename
    files, rjd, velocity, berv = [], [], [], []
    bervshow = []
    for row in range(len(rdb)):
        # the lblrv file is named after the science file
        name = str(rdb['FILENAME'][row])
        for suffix in ['_lbl.fits']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        # the science file of this row
        found = None
        for basename in basenames:
            if basename.startswith(name.split('_' + rdata['object'])[0]):
                found = basenames[basename]
                break
        # a file that is not on disk any more (a broken link, say) is left
        #   out here, once, rather than for every band
        if found is None or not os.path.exists(found):
            continue
        files.append(found)
        rjd.append(float(rdb['rjd'][row]))
        velocity.append(float(rdb['vrad'][row]))
        # the BERV LBL used: the one the wavelengths of the file still hold
        berv.append(float(rdata['berv_lbl'][row]))
        # the BERV of the Earth, to sort the spectra on
        bervshow.append(float(rdata['berv'][row]))
    if len(files) == 0:
        log.warning('No science file found for the river plots')
        return []
    # sub-sample evenly in time
    maxfiles = inst.params['REPORT_MAX_RIVER_FILES']
    if maxfiles not in [None, 'None'] and len(files) > int(maxfiles):
        keep = np.linspace(0, len(files) - 1, int(maxfiles)).astype(int)
        keep = np.unique(keep)
    else:
        keep = np.arange(len(files))
    rdata['science_files'] = [files[it] for it in keep]
    rdata['river_rjd'] = np.array(rjd)[keep]
    rdata['river_velocity'] = np.array(velocity)[keep]
    rdata['river_berv'] = np.array(berv)[keep]
    # the BERV of the figures (the motion of the Earth, whether or not LBL
    #   had to apply it): the spectra are sorted on it
    rdata['river_berv_show'] = np.array(bervshow)[keep]
    # the blaze of the run, as lbl_compute loads it (step 3 of that recipe):
    #   the river plots divide the spectra by it, so that what is seen is
    #   the spectrum and not the shape of the orders
    rdata['river_blaze'] = None
    try:
        blaze_file = inst.blaze_file(dparams['CALIB_DIR'])
        if blaze_file is not None:
            rdata['river_blaze'] = inst.load_blaze(
                blaze_file, science_file=rdata['science_files'][0])
    except Exception as e:
        wmsg = 'The blaze could not be loaded: {0}: {1}'
        log.warning(wmsg.format(type(e), str(e)))
    # -------------------------------------------------------------------------
    # one river plot per band
    # -------------------------------------------------------------------------
    outputs = []
    for bandname, index, wave_centre in centres:
        msg = 'River plot of the {0} band ({1:.2f} nm), {2} spectra'
        log.general(msg.format(bandname, wave_centre,
                               len(rdata['science_files'])))
        river = river_plot_data(inst, dparams, rdata, wave_centre)
        if river is None:
            wmsg = 'No spectrum could be read around {0:.2f} nm'
            log.warning(wmsg.format(wave_centre))
            continue
        figure = plot_river(river, bandname, index, figdir)
        nspectra = int(np.sum(np.any(np.isfinite(river['river']), axis=1)))
        outputs.append((bandname, figure, nspectra, wave_centre))
    return outputs


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================


# =============================================================================
# Define the debug plot of the lines, for one file
# =============================================================================
def median_snr_file(inst: InstrumentsType, rdata: Dict[str, Any],
                    science_files: List[str]) -> Optional[str]:
    """
    The science file whose signal to noise is the median of the run

    The signal to noise is the one LBL uses, the KW_EXT_SNR key of the header,
    which the rdb file also holds as a column. If it is not there, the file
    with the median velocity uncertainty is taken instead.

    :param inst: Instrument instance
    :param rdata: dict, the data of the report
    :param science_files: list of str, the science files

    :return: str or None, the science file
    """
    rdb = rdata['rdb']
    # the signal to noise of each file, the LBL way
    snr_key = inst.params['KW_EXT_SNR']
    if snr_key not in [None, 'None'] and snr_key in rdb.colnames:
        value = np.array(rdb[snr_key], dtype=float)
        label = 'signal to noise'
    else:
        # the uncertainty of the velocity says the same thing the other way
        value = -np.array(rdb['svrad'], dtype=float)
        label = 'velocity uncertainty'
    good = np.isfinite(value)
    if np.sum(good) == 0:
        return None
    # the file at the median
    rank = np.argsort(value[good])
    middle = np.where(good)[0][rank[len(rank) // 2]]
    msg = 'Debug plot of the lines: the file at the median {0} ({1:.1f})'
    log.general(msg.format(label, value[middle]))
    # the science file of that row
    name = str(rdb['FILENAME'][middle])
    for suffix in ['_lbl.fits']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    stem = name.split('_' + rdata['object'])[0]
    for filename in science_files:
        if os.path.basename(filename).startswith(stem):
            return filename
    return None


def line_edge_plot(inst: InstrumentsType, dparams: Dict[str, str],
                   rdata: Dict[str, Any], figdir: str
                   ) -> Tuple[List[Tuple[Any, ...]], Optional[str]]:
    """
    The debug plot of the lines (the one of PLOT_COMPUTE_LINES) for the file
    at the median signal to noise

    The velocities of that one file are computed again, with the same setup as
    lbl_compute (the same mask, reference table, template splines and blaze),
    and the vectors the debug plot uses come back in the outputs of
    compute_rv. Each line is drawn in its own colour, so the edges of the
    lines are where the colours change. One order per photometric band is
    drawn, the one closest to the middle of the band, over
    LINE_PLOT_ELEMENTS resolution elements of the instrument.

    :param inst: Instrument instance
    :param dparams: dict, the directories of this run
    :param rdata: dict, the data of the report
    :param figdir: str, the directory of the figures

    :return: tuple, 1. the figures (the file, the band, the order, the first
             and the last wavelength, the number of lines), 2. the spectrum
             they are of
    """
    # the setup of lbl_compute, needed to compute one file
    mask_dir, template_dir = dparams['MASK_DIR'], dparams['TEMPLATE_DIR']
    calib_dir, science_dir = dparams['CALIB_DIR'], dparams['SCIENCE_DIR']
    models_dir, lblrt_dir = dparams['MODEL_DIR'], dparams['LBLRT_DIR']
    # the file to draw
    science_files = inst.science_files(science_dir)
    # the files that are not on disk any more (a broken link, say) go first:
    #   sorting them reads their header
    science_files = [item for item in science_files if os.path.exists(item)]
    if len(science_files) == 0:
        log.warning('No science file could be read for the debug plot of '
                    'the lines')
        return [], None
    try:
        science_files = inst.sort_science_files(science_files)
    except Exception as e:
        wmsg = 'The science files could not be sorted: {0}: {1}'
        log.warning(wmsg.format(type(e), str(e)))
    science_file = median_snr_file(inst, rdata, science_files)
    if science_file is None:
        log.warning('No science file found for the debug plot of the lines')
        return [], None
    # -------------------------------------------------------------------------
    # everything compute_rv needs (a run whose mask or template is not next
    #   to its rdb file cannot be done again: say so and move on)
    # -------------------------------------------------------------------------
    try:
        mask_file = inst.mask_file(models_dir, mask_dir)
        science_template_file = inst.template_file(template_dir, 'science')
        comparison_template_file = inst.template_file(template_dir,
                                                      'comparison')
        blaze_file = inst.blaze_file(calib_dir)
        reftable_file, reftable_exists = inst.ref_table_file(lblrt_dir,
                                                             mask_file)
        if blaze_file is not None:
            blaze = inst.load_blaze(blaze_file, science_file=science_files[0])
        else:
            blaze = None
        ref_table = general.make_ref_dict(inst, reftable_file,
                                          reftable_exists, science_files,
                                          mask_file, calib_dir)
        sargs = [inst, science_template_file, mask_file]
        science_sys_vel_props = general.get_systemic_vel_props(*sargs)
        cargs = [inst, comparison_template_file, mask_file]
        comparison_sys_vel_props = general.get_systemic_vel_props(*cargs)
        splines = general.spline_template(
            inst, comparison_template_file,
            comparison_sys_vel_props['MASK_SYS_VEL'], models_dir)
        # the file itself, as lbl_compute reads it
        sci_data, sci_hdr = inst.load_science_file(science_file)
        sci_wave = inst.get_wave_solution(science_file, sci_data, sci_hdr)
        if blaze is None:
            blazeimage, blaze_flag = inst.load_blaze_from_science(
                science_file, sci_data, sci_hdr, calib_dir)
        elif np.sum(blaze.ravel()) == len(blaze.ravel()):
            blazeimage, blaze_flag = np.array(blaze), True
        else:
            blazeimage, blaze_flag = np.array(blaze), False
        if blaze_flag:
            sci_data, blazeimage = inst.no_blaze_corr(sci_data, sci_wave)
    except Exception as e:
        wmsg = 'The debug plot of the lines needs the mask and the template '
        wmsg += 'of the run: {0}: {1}'
        log.warning(wmsg.format(type(e), str(e)))
        return [], None
    # -------------------------------------------------------------------------
    # the velocities of that file, for the vectors of the debug plot
    # -------------------------------------------------------------------------
    log.general('Computing {0} again for the debug plot of the lines'
                ''.format(os.path.basename(science_file)))
    systemic_all = np.full(1, np.nan)
    mjdate_all = np.zeros(1, dtype=float)
    try:
        _, outputs = general.compute_rv(inst, 0, sci_data, sci_hdr,
                                        splines=splines,
                                        ref_table=ref_table,
                                        blaze=blazeimage,
                                        systemic_props=science_sys_vel_props,
                                        systemic_all=systemic_all,
                                        mjdate_all=mjdate_all,
                                        model_velocity=np.inf,
                                        science_file=science_file,
                                        mask_file=mask_file)
    except Exception as e:
        wmsg = 'The debug plot of the lines could not be made: {0}: {1}'
        log.warning(wmsg.format(type(e), str(e)))
        return [], None
    plot_dict = outputs.get('PLOT_DICT', None)
    if plot_dict is None or 'WAVEGRID' not in plot_dict:
        log.warning('No vectors for the debug plot of the lines')
        return [], None
    # -------------------------------------------------------------------------
    # the figures, as plot.compute_line_plot draws them: one order per
    #   photometric band (the order closest to the middle of the band), and
    #   a window of so many resolution elements at the middle of it
    # -------------------------------------------------------------------------
    wavegrid = plot_dict['WAVEGRID']
    model = plot_dict['MODEL']
    line_orders = np.array(plot_dict['LINE_ORDERS'])
    ww_ord_line = plot_dict['WW_ORD_LINE']
    spec_ord_line = plot_dict['SPEC_ORD_LINE']
    model_ord_line = plot_dict['MODEL_ORD_LINE']
    if wavegrid.ndim == 1:
        wavegrid = wavegrid.reshape(1, -1)
        model = np.array(model).reshape(1, -1)
    # the middle of each order
    with warnings.catch_warnings(record=True) as _:
        omin = np.nanmin(wavegrid, axis=1)
        omax = np.nanmax(wavegrid, axis=1)
    omid = 0.5 * (omin + omax)
    # one order per band: the one whose middle is closest to the middle of
    #   the band, among the orders that have lines
    with_lines = set(np.unique(line_orders).tolist())
    chosen = []
    for band_name, band_middle in river_bands(inst, rdata):
        distance, order_num = np.inf, None
        for ord_num in range(wavegrid.shape[0]):
            if ord_num not in with_lines or not np.isfinite(omid[ord_num]):
                continue
            if abs(omid[ord_num] - band_middle) < distance:
                distance = abs(omid[ord_num] - band_middle)
                order_num = ord_num
        # the same bands as the river plots: the closest order to the middle
        #   of the band, even when the middle falls between two orders
        if order_num is None:
            continue
        if order_num in [item[1] for item in chosen]:
            continue
        chosen.append((band_name, order_num))
    # nothing matched a band (a spectrum of a single order, say): the order
    #   with the most lines does the job
    if len(chosen) == 0 and len(with_lines) > 0:
        counts = [(int(np.sum(line_orders == num)), int(num))
                  for num in with_lines]
        chosen = [('', sorted(counts)[-1][1])]
    # -------------------------------------------------------------------------
    figures = []
    # the width of the window: so many resolution elements of the
    #   instrument, whatever the order and the instrument are
    resolution = inst.params['APPROX_RESOLUTION']
    for band_name, ord_num in chosen:
        # the middle of the order, over LINE_PLOT_ELEMENTS resolution
        #   elements (one element is the wavelength over the resolution)
        centre = omid[ord_num]
        half = 0.5 * LINE_PLOT_ELEMENTS * centre / resolution
        # an order shorter than that is drawn whole
        half = min(half, 0.5 * (omax[ord_num] - omin[ord_num]))
        wmin, wmax = centre - half, centre + half
        fig, frames = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # the template
        frames[0].plot(wavegrid[ord_num], model[ord_num], color='grey', lw=3,
                       alpha=0.3, label='template')
        # each line in its own colour: the edges are where the colours change
        nlines, label = 0, 'line'
        edges, fluxes = [], []
        for line_it in range(len(line_orders)):
            if line_orders[line_it] != ord_num:
                continue
            wwline = ww_ord_line[line_it]
            # only the lines of the window
            if np.nanmax(wwline) < wmin or np.nanmin(wwline) > wmax:
                continue
            colour = ['red', 'green', 'blue'][nlines % 3]
            frames[0].plot(wwline, spec_ord_line[line_it], color=colour,
                           lw=0.8, label=label)
            frames[1].plot(wwline,
                           spec_ord_line[line_it] - model_ord_line[line_it],
                           color=colour, lw=0.8)
            # where this line starts and stops, and what it is worth
            edges += [float(np.nanmin(wwline)), float(np.nanmax(wwline))]
            fluxes.append(np.array(spec_ord_line[line_it], dtype=float))
            nlines, label = nlines + 1, None
        if nlines == 0:
            plt.close(fig)
            continue
        # the edges of the lines, as dashed lines on both panels
        for edge in np.unique(np.round(edges, 6)):
            for frame in frames:
                frame.axvline(edge, color='grey', ls='--', lw=0.4,
                              alpha=0.6, zorder=0)
        # the flux axis holds what is drawn, with a tenth of its span to
        #   spare, the template of the window included
        inside = (wavegrid[ord_num] > wmin) & (wavegrid[ord_num] < wmax)
        shown = np.concatenate(fluxes + [np.array(model[ord_num])[inside]])
        with warnings.catch_warnings(record=True) as _:
            flow, fhigh = np.nanmin(shown), np.nanmax(shown)
        if np.isfinite(flow) and np.isfinite(fhigh) and fhigh > flow:
            margin = 0.05 * (fhigh - flow)
            frames[0].set_ylim(flow - margin, fhigh + margin)
        if band_name == '':
            title = 'Lines of {0} (order {1})'
            title = title.format(os.path.basename(science_file), ord_num)
        else:
            title = 'Lines of {0} ({1} band, order {2}, {3:.0f} resolution '
            title += 'elements)'
            title = title.format(os.path.basename(science_file), band_name,
                                 ord_num, LINE_PLOT_ELEMENTS)
        frames[0].set(ylabel='flux', title=title, xlim=[wmin, wmax])
        frames[0].legend(loc='best')
        frames[1].axhline(0, color='k', lw=0.5)
        frames[1].set(xlabel='wavelength [nm]', ylabel='spectrum - template',
                      xlim=[wmin, wmax])
        fig.tight_layout()
        name = 'lines_debug_{0}'.format(band_name if band_name else ord_num)
        figures.append((save_figure(fig, figdir, name), band_name, ord_num,
                        float(wmin), float(wmax), nlines))
    return figures, os.path.basename(science_file)

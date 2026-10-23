#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPaDOnS (CFHT) instrument class: the intensity spectra, <odometer>i.fits,
that CADC distributes, reduced at CFHT by Upena with Libre-ESpRIT.

What these files are, and what the class makes of them (checked on AD Leo,
files of 2005 and 2019, 2026-10-23):

* ONE TABLE, ALL ORDERS END TO END. The primary HDU is (columns x samples),
  its rows named by the COLn cards: Wavelength, Star (or Intensity) and
  ErrorBar, Normalized and UnNormalized, with and without the automatic
  wavelength correction from the telluric lines; the star+sky mode adds
  Star+sky and Sky. The class takes the UnNormalized intensity with the
  automatic correction, cuts the table into its 40 orders (m = 61 to 22)
  where the wavelength goes back or jumps (the red orders do not overlap,
  so going back is not enough), and puts each order on a row of NPIX
  samples, anchored on the order's nominal centre so that a given pixel
  holds about the same wavelength in every file. The samples are kept as
  Libre-ESpRIT wrote them (a uniform step of 1.86 km/s); the rest of the row
  is NaN, its wavelengths continued at the same step.

* AIR, HELIOCENTRIC. Libre-ESpRIT writes air wavelengths moved to the
  heliocentric frame by (1 + v/c), v being the "Heliocentric velocity of
  observer towards star" of the header's comments. The class undoes that
  shift (in the same first-order form, since that is what was applied) and
  converts to vacuum, so that the spectrum is in the observer's frame where
  the tellurics are: against TAPAS, the O2 B band of the delivered files sat
  at -27.5 and -29.5 km/s for 27.307 and 29.931 km/s of heliocentric
  velocity, and in air (in vacuum the best match is 82 km/s away, the
  air-to-vacuum shift, and follows nothing). The BERV is then the full
  barycentric correction, computed with astropy at CFHT for the middle of
  the exposure.

* THE ZERO POINT, FROM THE TELLURICS. ESPaDOnS has no simultaneous
  reference, and its wavelengths move from one exposure to the next by
  hundreds of m/s, which Libre-ESpRIT's own correction from the telluric
  lines leaves in place: on 39 spectra of AD Leo (2005-2022) the O2 lines
  sat between -540 and +780 m/s of their rest wavelengths, and LBL's
  velocities followed them with a slope of -0.92 (r = -0.98, 204 m/s of
  rms). The class measures the velocity of the tellurics by least squares
  (fit_tellurics): the star's template divided out (LBL's, made by a
  previous run, moved by the BERV), the depths of the dry and wet absorbers
  of TAPAS fitted first with no velocity, then one velocity for both,
  iterated, with a continuum per order. It moves the whole spectrum by that
  velocity so that the tellurics sit at rest (relativistic Doppler), and
  writes it in the header: TELLDV, the RV correction from the tellurics
  [m/s], TELLDVE its error, TELLTPL the template divided out ('none' on a
  first run, which measures with the star's lines in and is then run
  again). TELLDV and TELLDVE go into the rdb; telluric_figure draws a fit.

* BLAZE. The UnNormalized intensity is already divided by the flat, and has
  no blaze left in it. The class gives LBL the blaze back, measured from the
  file's own error bars: in the photon-noise limit flux / error^2 follows
  the blaze whatever the depth of the lines, so that, smoothed along each
  order, it weighs each sample by the light it received.

* SNR. From the comments, "SNR per spec/ccd pxl in order # 31 (731 nm)", the
  order at CFHT's reference wavelength (SNRWAVE = 730 nm).

The polarimetric products (<odometer>p.fits) are not read: the four
exposures of a sequence each have their own i.fits.

Created on 2026-10-23

@author: artigau
"""
import functools
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy import units as uu
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits

from lbl.core import astro
from lbl.core import base
from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.espadons.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get time from base
Time = base.AstropyTime
# get classes
Instrument = default.Instrument
LblException = base_classes.LblException
log = io.log
#: the speed of light in km/s
C_KMS = 299792.458
#: the diffraction orders, bluest first, and m * lambda at an order's centre
#:    [nm] (61 at 371.5 nm to 22 at 1030 nm, the SNR comments of the files)
ORDERS = np.arange(61, 21, -1)
M_LAMBDA = 22660.0
#: the samples per order row: the longest order holds about 6650
NPIX = 8192
#: Libre-ESpRIT's step in ln(lambda), 1.86 km/s, for the rows' anchors
DLNL = 6.2e-6
#: the order whose SNR the header gives LBL (731 nm, CFHT's SNRWAVE)
SNR_ORDER = 31
#: CFHT on Maunakea (astropy's site registry, 'cfht')
CFHT = EarthLocation.from_geodetic(lon=-155.47166666666666 * uu.deg,
                                   lat=19.826666666666668 * uu.deg,
                                   height=4215.0 * uu.m)
#: the keys the class adds to each header
KEY_MJDMID = 'MJDMID'
KEY_BERV = 'BERV'
KEY_BJD = 'BJD'
KEY_HELIO = 'HELIOVEL'
KEY_SNR = 'SNR{0}'.format(SNR_ORDER)
#: the zero point from the tellurics: the RV correction they gave [m/s], its
#:    error [m/s], and the template divided out to measure it (or 'none')
KEY_TELLDV = 'TELLDV'
KEY_TELLDVE = 'TELLDVE'
KEY_TELLTPL = 'TELLTPL'
#: the telluric fit (fit_tellurics): where [nm], the degree of each order's
#:    continuum, the least absorption that makes an order take part, the
#:    transmission under which a sample is saturated and left out, the
#:    iterations and the clipping [sigmas]
TELLURIC_RANGE = (580.0, 1050.0)
TELLURIC_DEGREE = 3
TELLURIC_MIN_DEPTH = 0.03
TELLURIC_MIN_TRANS = 0.15
TELLURIC_ITERATIONS = 20
TELLURIC_CLIP = 5.0
#: the TAPAS file (models/ of DATA_DIR) and the star's templates, in the
#:    order they are looked for, set by the class when it starts
TAPAS_FILE = None
TEMPLATE_FILES = ()

# =============================================================================
# Define functions
# =============================================================================
def columns(header: fits.Header) -> Dict[Tuple[str, bool, bool], int]:
    """
    The rows of the data, from the COLn cards: {(name, normalized,
    autowave): row}, the name in lower case ('wavelength', 'star',
    'intensity', 'errorbar1'...)
    """
    out = dict()
    for num in range(1, 100):
        key = 'COL{0}'.format(num)
        if key not in header:
            break
        comment = header.comments[key].lower()
        normalized = 'unnormalized' not in comment
        autowave = 'no autowave' not in comment
        out[(str(header[key]).strip().lower(), normalized, autowave)] = num - 1
    return out


def comment_value(header: fits.Header, pattern: str) -> Optional[float]:
    """The first number a COMMENT card gives after `pattern` (a regex with
    one group), or None"""
    for card in header.get('COMMENT', []):
        found = re.search(pattern, str(card))
        if found:
            return float(found.group(1))
    return None


def heliocentric_velocity(header: fits.Header) -> float:
    """The velocity [km/s] Libre-ESpRIT moved the wavelengths by"""
    value = comment_value(header, r'Heliocentric velocity of observer towards'
                                  r' star\s*:\s*([-+0-9.]+)')
    if value is None:
        raise LblException('ESPaDOnS file without the heliocentric velocity'
                           ' of its reduction in its comments: its'
                           ' wavelengths cannot be put back in the'
                           ' observer\'s frame')
    return value


def order_snr(header: fits.Header, order: int = SNR_ORDER) -> float:
    """The SNR per spectral bin Libre-ESpRIT gives for one order (the first
    of the values it writes), NaN when absent"""
    value = comment_value(header, r'SNR per spec/ccd pxl in order #\s*{0}\b'
                                  r'[^:]*:\s*I>\s*([0-9.]+)'.format(order))
    return np.nan if value is None else value


def mid_exposure(header: fits.Header) -> float:
    """MJD (UTC) of the middle of the exposure"""
    start = float(header['MJDATE'])
    if 'MJDEND' in header:
        return 0.5 * (start + float(header['MJDEND']))
    return start + 0.5 * float(header.get('EXPTIME', 0.0)) / 86400.0


@functools.lru_cache(maxsize=512)
def barycentric(ra: str, dec: str, mjdmid: float) -> Tuple[float, float]:
    """(BERV [km/s], BJD_TDB [JD]) at CFHT, astropy's barycentric correction
    (the velocity to add to an observed one)"""
    coord = SkyCoord(ra, dec, unit=(uu.hourangle, uu.deg), frame='icrs')
    obstime = Time(mjdmid, format='mjd', scale='utc', location=CFHT)
    # the location rides on obstime (astropy refuses it twice)
    berv = coord.radial_velocity_correction(kind='barycentric',
                                            obstime=obstime)
    bjd = obstime.tdb + obstime.light_travel_time(coord, kind='barycentric')
    return float(berv.to(uu.km / uu.s).value), float(bjd.jd)


def order_blaze(flux: np.ndarray, error: np.ndarray
                ) -> Tuple[np.ndarray, float]:
    """
    The blaze of one order from its error bars, normalised to 1 at its peak,
    and that peak: the photons per unit of the delivered flux (N = F^2 /
    sigma^2 = F b g, so g = F / (sigma^2 b)), NaN without error bars.

    In the photon-noise limit sigma^2 = F / b for a flux F already divided
    by the blaze b, so F / sigma^2 = b whatever the depth of a line; where
    the read noise counts it falls below, and the robust fit clips it.
    Ones when the order has too few usable samples.
    """
    npts = len(flux)
    good = (np.isfinite(flux) & np.isfinite(error) & (error > 0)
            & (flux > 0))
    if np.sum(good) < 100:
        return np.ones(npts), np.nan
    raw = np.full(npts, np.nan)
    raw[good] = flux[good] / error[good] ** 2
    xpix = np.arange(npts, dtype=float) / npts
    try:
        coeffs, _ = mp.robust_polyfit(xpix[good], raw[good], 6, 4)
    except Exception:                                        # noqa: BLE001
        return np.ones(npts), np.nan
    blaze = np.polyval(coeffs, xpix)
    peak = np.nanmax(blaze)
    if not np.isfinite(peak) or peak <= 0:
        return np.ones(npts), np.nan
    blaze = blaze / peak
    # the polynomial can dip at the very ends: no weight below 1%
    return np.clip(blaze, 0.01, None), float(peak)


@functools.lru_cache(maxsize=2)
def tapas(path: str, resolution: float = 65000.0) -> Dict[str, np.ndarray]:
    """TAPAS over TELLURIC_RANGE at the instrument's resolution (a gaussian of
    FWHM c / R in velocity): the log of the transmission of each absorber
    and its derivative against ln(lambda)"""
    from astropy.table import Table
    from scipy.ndimage import gaussian_filter1d
    table = Table.read(path, hdu=1)
    wave = np.asarray(table['WAVELENGTH'], dtype=float)
    keep = ((wave > TELLURIC_RANGE[0] - 5.0)
            & (wave < TELLURIC_RANGE[1] + 5.0))
    lnw = np.log(wave[keep])
    step = np.median(np.diff(lnw)) * C_KMS
    sigma = (C_KMS / resolution / (2 * np.sqrt(2 * np.log(2)))) / step
    out = dict(lnw=lnw)
    for name, key in (('ABSO_OTHERS', 'others'), ('ABSO_WATER', 'water')):
        trans = gaussian_filter1d(np.asarray(table[name],
                                             dtype=float)[keep], sigma)
        out[key] = np.log(np.clip(trans, 1e-4, None))
        out['d' + key] = np.gradient(out[key], lnw)
    return out


@functools.lru_cache(maxsize=4)
def stellar_template(path: str, mtime: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """An LBL template (barycentric frame): (ln wavelength, ln flux) where
    the flux is positive; `mtime` only makes a rewritten file read again"""
    from astropy.table import Table
    _ = mtime
    table = Table.read(path, 'TEMPLATE')
    wave = np.asarray(table['wavelength'], dtype=float)
    flux = np.asarray(table['flux'], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux) & (flux > 0) & (wave > 0)
    return np.log(wave[good]), np.log(flux[good])


def fit_tellurics(wave: np.ndarray, flux: np.ndarray, blaze: np.ndarray,
                  model: Dict[str, np.ndarray],
                  template: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  berv: float = 0.0, details: bool = False,
                  gain: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    The velocity of the telluric lines of a spectrum [km/s], by least
    squares, one velocity for every absorber:

        ln(F / blaze) = continuum_o(x) + ln S(lambda D(v) D(BERV))
                        + a ln T_others(lambda D(v)) + b ln T_water(lambda D(v))

    with F the flux (the blaze in it), S the star's template when there is
    one (LBL's, barycentric; left out otherwise), T TAPAS at the instrument's
    resolution, a and b the depths of the two absorbers, a polynomial
    continuum per order in its own pixel coordinate, and D(v) the
    relativistic Doppler factor, so that the spectrum's wavelengths times
    D(v) put the tellurics at rest.

    First the depths alone, at v = 0 (the wavelengths as they are). Then v
    is added and iterated until it moves by less than 1 cm/s: each step is
    Gauss-Newton's on the tellurics (their derivative in v), with the star
    moved to v but taking no part in measuring it, so that its own velocity
    cannot leak into the zero point. Left out of the derivative, the star
    makes that iteration converge only linearly (on AD Leo, 639, 530, 480...
    m/s, still moving after ten steps), so the velocity where the step
    vanishes is found by the secant method on the steps. Weights are the
    photon counts (the variance of ln F is 1 / counts; the flux times
    `gain`, photons per unit of flux, per row, so that the chi2 is one for
    photon noise), those the model PREDICTS after the first step: weighted by
    the counts measured, the noise weighs itself, and the velocity came out
    7 to 11 m/s off on a synthetic spectrum at 1% noise (10 to 16 sigmas),
    the bias going as 1 / SNR^2. The velocity is found twice: with nothing
    clipped, then with the samples beyond TELLURIC_CLIP robust sigmas of
    that aligned model left out, held for the second search (see the code
    for why neither the clipping at v = 0 nor at every step will do).
    Saturated
    ones (transmission under TELLURIC_MIN_TRANS at the depths of the first
    step), are left out. The error is the covariance's, scaled by the
    reduced chi2.

    :return: dict(dv, sdv [km/s], depths (a, b) at v = 0 and fitted, chi2,
             nsamples, norders, history [km/s, per step], template: bool;
             with details, the samples for the figure)
    """
    low, high = TELLURIC_RANGE
    lnw_all, y_all, w_all, x_all, o_all = [], [], [], [], []
    for row in range(wave.shape[0]):
        good = (np.isfinite(flux[row]) & (flux[row] > 0) & (blaze[row] > 0.2)
                & (wave[row] > low) & (wave[row] < high))
        if np.sum(good) < 200:
            continue
        lnw = np.log(wave[row][good])
        trans = np.exp(np.interp(lnw, model['lnw'], model['others'])
                       + np.interp(lnw, model['lnw'], model['water']))
        if 1.0 - np.min(trans) < TELLURIC_MIN_DEPTH:
            continue
        pix = np.where(good)[0].astype(float)
        scale = 1.0
        if gain is not None and np.isfinite(gain[row]) and gain[row] > 0:
            scale = float(gain[row])
        lnw_all.append(lnw)
        y_all.append(np.log(flux[row][good] / blaze[row][good]))
        w_all.append(flux[row][good] * scale)
        x_all.append(2 * (pix - pix.min()) / max(np.ptp(pix), 1.0) - 1)
        o_all.append(np.full(len(pix), len(lnw_all) - 1))
    if not lnw_all:
        return dict(dv=np.nan, sdv=np.nan, depths=(np.nan, np.nan),
                    depths0=(np.nan, np.nan), chi2=np.nan, nsamples=0,
                    norders=0, history=[], template=template is not None)
    lnw, y_raw = np.concatenate(lnw_all), np.concatenate(y_all)
    counts = np.concatenate(w_all)
    weight = np.array(counts)
    xpix, order = np.concatenate(x_all), np.concatenate(o_all)
    norders = len(lnw_all)
    ncont = norders * (TELLURIC_DEGREE + 1)
    # the continua: a block of powers of x per order
    cont = np.zeros((len(lnw), ncont))
    for deg in range(TELLURIC_DEGREE + 1):
        cont[np.arange(len(lnw)), order * (TELLURIC_DEGREE + 1) + deg] = \
            xpix ** deg
    shift_berv = np.arctanh(berv / C_KMS)

    def data(vel):
        """ln(F / blaze), the star divided out at the wavelengths of `vel`,
        and where the star is known"""
        if template is None:
            return y_raw, np.ones(len(y_raw), dtype=bool)
        at = lnw + np.arctanh(vel / C_KMS) + shift_berv
        star = np.interp(at, template[0], template[1], left=np.nan,
                         right=np.nan)
        return y_raw - star, np.isfinite(star)

    def tellurics(vel):
        at = lnw + np.arctanh(vel / C_KMS)
        return [np.interp(at, model['lnw'], model[key])
                for key in ('others', 'water', 'dothers', 'dwater')]

    def solve(design, yy, use, wgt):
        """Weighted least squares; by pseudo-inverse, since an absorber
        absent from the orders used leaves its column empty"""
        sw = np.sqrt(wgt[use])
        aa = design[use] * sw[:, None]
        params = np.linalg.lstsq(aa, yy[use] * sw, rcond=None)[0]
        return params, np.linalg.pinv(aa.T @ aa)

    def design(vel, params=None):
        """The columns at `vel`: the continua, the two absorbers and, given
        the parameters of a previous solution, the derivative of the
        tellurics in v at its depths (d ln T / d ln lambda times
        d ln(lambda D(v)) / dv = 1 / (c (1 - v^2 / c^2)))"""
        lno, lnh, dlno, dlnh = tellurics(vel)
        cols = [cont, lno[:, None], lnh[:, None]]
        if params is not None:
            cols.append(((params[ncont] * dlno + params[ncont + 1] * dlnh)
                         / (C_KMS * (1 - (vel / C_KMS) ** 2)))[:, None])
        return np.hstack(cols), lno, lnh

    def predicted(yy, mat, params):
        """The counts the model predicts, the next step's weights"""
        out = counts * np.exp(-(yy - mat @ params))
        out[~np.isfinite(out)] = 0.0
        return out

    def converge(vel, mask, params, wgt):
        """The velocity where Gauss-Newton's step vanishes, by the secant
        method on the steps, the samples `mask` held fixed"""
        previous, path = None, []
        for _ in range(TELLURIC_ITERATIONS):
            yy, known = data(vel)
            mat, _, _ = design(vel, params)
            use = mask & known
            params, cov = solve(mat, yy, use, wgt)
            wgt = predicted(yy, mat, params)
            step = float(params[-1])
            move = step
            if previous is not None and step != previous[1]:
                move = -step * (vel - previous[0]) / (step - previous[1])
            previous = (vel, step)
            vel += move
            path.append(vel)
            if abs(move) < 1e-5:
                break
        return vel, params, cov, wgt, mat, yy, use, path

    # 0. the depths alone, at v = 0, weighted by the counts measured
    yy, known = data(0.0)
    mat, lno, lnh = design(0.0)
    params, cov = solve(mat, yy, known, counts)
    depths0 = (float(params[ncont]), float(params[ncont + 1]))
    # saturated at these depths: out for good
    keep = known & (params[ncont] * lno + params[ncont + 1] * lnh
                    >= np.log(TELLURIC_MIN_TRANS))
    weight = predicted(yy, mat, params)
    # A. the velocity, nothing clipped: a clipping made on a model still off
    #   by the velocity sought keeps the samples whose noise went against
    #   that offset, and pulls the velocity back towards zero (7 to 11 m/s
    #   on a synthetic spectrum at 1% noise, the same for every seed)
    vel, params, cov, weight, mat, yy, use, path_a = converge(
        0.0, keep, params, weight)
    # B. clipped once, on the residuals of that aligned model, and held:
    #   clipped again at every step, the samples coming in and out made the
    #   steps jump and the velocity wander by +-20 m/s for a 5 m/s error
    resid = (yy - mat @ params) * np.sqrt(weight)
    sigma = 1.4826 * np.nanmedian(np.abs(resid[use]
                                         - np.nanmedian(resid[use])))
    clip = np.abs(resid) < TELLURIC_CLIP * sigma
    vel, params, cov, weight, mat, yy, use, path_b = converge(
        vel, keep & clip, params, weight)
    history = [0.0] + path_a + path_b
    nsamp = int(np.sum(use))
    resid = (yy - mat @ params) * np.sqrt(weight)
    chi2 = float(np.sum(resid[use] ** 2) / max(nsamp - len(params), 1))
    out = dict(dv=vel, sdv=float(np.sqrt(cov[-1, -1] * chi2)),
               depths=(float(params[ncont]), float(params[ncont + 1])),
               depths0=depths0, chi2=chi2, nsamples=nsamp, norders=norders,
               history=history, clipped_from=1 + len(path_a),
               template=template is not None)
    if details:
        # the observed transmission (the star and the continuum divided
        #   out) and the two models, at v = 0 and at v, sample by sample
        yy, _ = data(vel)
        lno, lnh, _, _ = tellurics(vel)
        lno0, lnh0, _, _ = tellurics(0.0)
        continuum = cont @ params[:ncont]
        out.update(wave=np.exp(lnw), observed=np.exp(yy - continuum),
                   model0=np.exp(depths0[0] * lno0 + depths0[1] * lnh0),
                   model=np.exp(params[ncont] * lno + params[ncont + 1] * lnh),
                   used=use)
    return out


@functools.lru_cache(maxsize=8)
def read_espadons(filename: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One ESPaDOnS i.fits as LBL wants it: (wave, flux, blaze), each
    (len(ORDERS), NPIX), the wavelengths in nm, vacuum, observer's frame,
    their zero point set on the tellurics (zero_point), the flux with its
    blaze put back (flux = blaze x the delivered UnNormalized intensity),
    the blaze normalised per order. See the module's docstring.
    """
    return _read(filename, zero=True)


def _read(filename: str, zero: bool = True):
    """read_espadons, or with zero=False the rows before the zero point and
    the header: (wave, flux, blaze, header)"""
    with fits.open(filename) as hdul:
        names = [hdu.name for hdu in hdul]
        # a file LBL wrote (the telluric-cleaned spectra, write_tellu_cleaned)
        #   holds the rows themselves
        if 'WAVE' in names and 'BLAZE' in names:
            return (np.asarray(hdul['WAVE'].data, dtype=float),
                    np.asarray(hdul[1].data, dtype=float),
                    np.asarray(hdul['BLAZE'].data, dtype=float))
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    cols = columns(header)
    if str(header.get('REDUCTIO', 'Intensity')).strip() != 'Intensity':
        raise LblException('{0} is not an intensity spectrum (REDUCTIO = {1}):'
                           ' the class reads the i.fits'.format(
                               filename, header.get('REDUCTIO')))
    try:
        wave = data[cols[('wavelength', False, True)]]
        name = ('star' if ('star', False, True) in cols else 'intensity')
        flux = data[cols[(name, False, True)]]
    except KeyError:
        raise LblException('{0}: no UnNormalized wavelength and intensity'
                           ' with the automatic wavelength correction in its'
                           ' COLn cards'.format(filename))
    ename = ('errorbar1' if ('errorbar1', False, True) in cols
             else 'errorbar')
    error = (data[cols[(ename, False, True)]]
             if (ename, False, True) in cols else np.full_like(flux, np.nan))
    # back to the observer's frame, as Libre-ESpRIT moved it, then to vacuum
    wave = wave / (1.0 + heliocentric_velocity(header) / C_KMS)
    wave = wave * mp.air_index(wave)
    # the orders: where the wavelength goes back or jumps
    lnw = np.log(np.where(wave > 0, wave, np.nan))
    step = np.diff(lnw)
    typical = np.nanmedian(step[step > 0])
    breaks = np.where(~(step > 0) | (step > 5 * typical))[0] + 1
    # the rows, every one with a wavelength grid even when the file lacks
    #   the order, so that the wave map is finite everywhere
    index = np.arange(NPIX)
    anchors = np.log(M_LAMBDA / ORDERS) - (NPIX // 2) * DLNL
    out_wave = np.exp(anchors[:, None] + index[None, :] * DLNL)
    out_flux = np.full((len(ORDERS), NPIX), np.nan)
    out_blaze = np.ones((len(ORDERS), NPIX))
    # photons per unit of flux, order by order (order_blaze)
    gain = np.full(len(ORDERS), np.nan)
    filled = np.zeros(len(ORDERS), dtype=bool)
    for seg in np.split(np.arange(len(wave)), breaks):
        seg = seg[np.isfinite(lnw[seg])]
        if len(seg) < 100:
            continue
        order = int(np.round(M_LAMBDA / np.exp(np.median(lnw[seg]))))
        rows = np.where(ORDERS == order)[0]
        if len(rows) == 0 or filled[rows[0]]:
            continue
        row = rows[0]
        lseg = lnw[seg]
        dl = np.median(np.diff(lseg))
        start = int(np.round((lseg[0] - anchors[row]) / dl))
        # what does not fit on the row is dropped (never seen: NPIX is
        #   1.2 times the longest order)
        first = max(0, -start)
        last = min(len(seg), NPIX - start)
        seg, lseg = seg[first:last], lseg[first:last]
        start = max(start, 0)
        stop = start + len(seg)
        # the empty ends continued at the order's step from its own first
        #   and last samples: continued from the first alone, an order
        #   whose step varies along it overtook its last sample
        out_wave[row, :start] = np.exp(lseg[0] - (start - index[:start]) * dl)
        out_wave[row, start:stop] = wave[seg]
        out_wave[row, stop:] = np.exp(lseg[-1]
                                      + (index[stop:] - (stop - 1)) * dl)
        out_flux[row, start:stop] = flux[seg]
        blaze, gain[row] = order_blaze(flux[seg], error[seg])
        out_blaze[row, start:stop] = blaze
        out_flux[row, start:stop] *= blaze
        filled[row] = True
    if not np.any(filled):
        raise LblException('{0}: no ESPaDOnS order found'.format(filename))
    GAINS[filename] = gain
    if not zero:
        return out_wave, out_flux, out_blaze, header
    # the zero point: the tellurics put at rest (see the module)
    fit = zero_point(filename, out_wave, out_flux, out_blaze, header)
    if np.isfinite(fit['dv']):
        out_wave = out_wave * np.exp(np.arctanh(fit['dv'] / C_KMS))
    return out_wave, out_flux, out_blaze


#: the zero point of each file read (fit_tellurics), for its header
TELLURIC_FITS = dict()
#: the photons per unit of flux of each file read, order by order
GAINS = dict()


def template_file() -> Optional[str]:
    """The first of TEMPLATE_FILES on disk, or None"""
    for path in TEMPLATE_FILES:
        if path and os.path.exists(path):
            return path
    return None


def zero_point(filename: str, wave: np.ndarray, flux: np.ndarray,
               blaze: np.ndarray, header: fits.Header,
               template: Optional[str] = None, tapas_file: Optional[str] = None,
               details: bool = False) -> Dict[str, Any]:
    """
    The zero point of a file from its tellurics (fit_tellurics), with the
    star's template divided out when there is one (TEMPLATE_FILES, else said:
    a first run makes the template, and the next measures against it), kept
    for its header. dv is NaN, and said, without TAPAS.
    """
    tapas_file = tapas_file or TAPAS_FILE
    if tapas_file is None or not os.path.exists(tapas_file):
        log.warning('ESPaDOnS: no TAPAS file ({0}), so the zero point of {1}'
                    ' is not measured on its tellurics: expect hundreds of'
                    ' m/s of scatter'.format(tapas_file, filename))
        fit = dict(dv=np.nan, sdv=np.nan, template='none')
        TELLURIC_FITS[filename] = fit
        return fit
    template = template or template_file()
    star = None
    if template is not None:
        star = stellar_template(template, os.path.getmtime(template))
    else:
        log.warning('ESPaDOnS {0}: no template of the star yet, so its'
                    ' tellurics are measured with its lines in: run again'
                    ' once the template is made, for the zero point to'
                    ' tens of m/s'.format(os.path.basename(filename)))
    berv, _ = barycentric(str(header['RA']), str(header['DEC']),
                          round(mid_exposure(header), 8))
    fit = fit_tellurics(wave, flux, blaze, tapas(tapas_file), star, berv,
                        details=details, gain=GAINS.get(filename))
    fit['template'] = (os.path.basename(template) if template is not None
                       else 'none')
    TELLURIC_FITS[filename] = fit
    log.general('ESPaDOnS {0}: tellurics at {1:+.1f} +- {2:.1f} m/s, depths'
                ' {3:.2f} (dry) and {4:.2f} (water), {5} samples in {6}'
                ' orders, chi2 {7:.2f}, {8} iterations, template {9}'.format(
                    os.path.basename(filename), 1000 * fit['dv'],
                    1000 * fit['sdv'], fit['depths'][0], fit['depths'][1],
                    fit['nsamples'], fit['norders'], fit['chi2'],
                    len(fit['history']) - 1, fit['template']))
    return fit


#: the windows the figure shows: (name, from, to [nm])
FIGURE_WINDOWS = (('O2 B', 687.2, 688.6), ('O2 A', 762.2, 763.6),
                  ('H2O', 822.6, 824.4))
#: before and after (the colour-blind safe pair of lbl.pca2d's figures)
BEFORE, AFTER = '#2a78d6', '#eb6834'


def telluric_figure(plt, filename: str, template: Optional[str] = None,
                    tapas_file: Optional[str] = None, title: str = ''):
    """
    The zero point of one file, drawn: in three windows, the observed
    transmission (the spectrum divided by the star's template and the
    continuum) with the tellurics at the fitted depths and v = 0, and at the
    fitted velocity; their residuals; and the velocity along the iterations.

    :return: the figure
    """
    wave, flux, blaze, header = _read(os.path.abspath(filename), zero=False)
    fit = zero_point(os.path.abspath(filename), wave, flux, blaze, header,
                     template=template, tapas_file=tapas_file, details=True)
    if 'wave' not in fit:
        return None
    fig = plt.figure(figsize=(10, 7.5))
    grid = fig.add_gridspec(3, len(FIGURE_WINDOWS), height_ratios=[2, 1, 1.3],
                            hspace=0.45, wspace=0.25)
    for col, (name, low, high) in enumerate(FIGURE_WINDOWS):
        inside = (fit['wave'] > low) & (fit['wave'] < high)
        order = np.argsort(fit['wave'][inside])
        wave_w = fit['wave'][inside][order]
        used = fit['used'][inside][order]
        top = fig.add_subplot(grid[0, col])
        bottom = fig.add_subplot(grid[1, col], sharex=top)
        for key, colour, label in (('model0', BEFORE, 'v = 0'),
                                   ('model', AFTER, 'v = {0:+.0f} m/s'.format(
                                       1000 * fit['dv']))):
            top.plot(wave_w, fit[key][inside][order], color=colour, lw=1.0,
                     label=label)
            bottom.plot(wave_w[used], (fit['observed'][inside][order]
                                       - fit[key][inside][order])[used],
                        '.', ms=2, color=colour)
        top.plot(wave_w[used], fit['observed'][inside][order][used], '.',
                 ms=2.5, color='#1f1f1f', label='observed')
        top.plot(wave_w[~used], fit['observed'][inside][order][~used], '.',
                 ms=2.5, color='#b0b0b0', label='left out')
        top.set_title(name, fontsize=10)
        bottom.axhline(0, color='#5f6368', lw=0.6)
        bottom.set_xlabel('wavelength (nm), observer frame', fontsize=8)
        if col == 0:
            top.set_ylabel('spectrum / star / continuum', fontsize=8)
            bottom.set_ylabel('observed - model', fontsize=8)
            top.legend(fontsize=7, loc='lower left')
        for ax in (top, bottom):
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.15)
    ax = fig.add_subplot(grid[2, :2])
    steps = np.arange(len(fit['history']))
    ax.plot(steps, 1000 * np.array(fit['history']), 'o-', color=AFTER)
    ax.axhspan(1000 * (fit['dv'] - fit['sdv']), 1000 * (fit['dv'] + fit['sdv']),
               color=AFTER, alpha=0.15)
    ax.axvline(fit['clipped_from'] - 0.5, color='#5f6368', lw=0.8, ls=':')
    ax.text(fit['clipped_from'] - 0.4, ax.get_ylim()[1], ' outliers clipped',
            va='top', fontsize=7, color='#5f6368')
    ax.set_xlabel('step (0: the depths alone, at v = 0)', fontsize=8)
    ax.set_ylabel('velocity of the tellurics (m/s)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)
    text = fig.add_subplot(grid[2, 2])
    text.axis('off')
    text.text(0, 1, '\n'.join([
        'TELLDV = {0:+.1f} +- {1:.1f} m/s'.format(1000 * fit['dv'],
                                                1000 * fit['sdv']),
        'depths: dry {0:.2f}, water {1:.2f}'.format(*fit['depths']),
        '  (at v = 0: {0:.2f}, {1:.2f})'.format(*fit['depths0']),
        '{0} samples, {1} orders'.format(fit['nsamples'], fit['norders']),
        'reduced chi2 {0:.2f}'.format(fit['chi2']),
        'star divided out: {0}'.format(fit['template'])]),
        va='top', fontsize=8, family='monospace')
    fig.suptitle('{0}{1}, {2}: the zero point from the tellurics'.format(
        title + ' ' if title else '', os.path.basename(filename),
        header.get('DATE-OBS', '')), fontsize=11)
    return fig


# =============================================================================
# Define ESPaDOnS class
# =============================================================================
class Espadons(Instrument):
    def __init__(self, params: base_classes.ParamDict,
                 args: base_classes.ParamDict):
        # call to super function
        super().__init__('ESPADONS')
        # extra parameters (specific to instrument)
        self.default_template_name = 'LBL_Template_{0}_espadons.fits'
        self.default_mask_name = 'LBL_Mask_{obj}_{mtype}_espadons.fits'
        self.default_sample_wave_name = 'sample_wave_grid_espadons.fits'
        # define wave limits in nm
        self.wavemin = 369.0
        self.wavemax = 1048.0
        # set parameters for instrument
        self.params = params
        # override params
        self.param_override()
        # update from args
        self.update_from_args(args)
        # the transmission the zero point is measured against
        global TAPAS_FILE, TEMPLATE_FILES
        if self.params['DATA_DIR'] not in (None, '', 'None'):
            data_dir = str(self.params['DATA_DIR'])
            TAPAS_FILE = os.path.join(data_dir, 'models',
                                      self.params['TELLUCLEAN_TAPAS_FILE'])
            # the star's template to divide out: the cleaned object's (made
            #   by a previous run), else the delivered one's
            names = []
            for key in ('OBJECT_COMPARISON', 'OBJECT_SCIENCE'):
                value = self.params[key]
                if value in (None, '', 'None'):
                    continue
                base_name = str(value)
                if base_name.endswith('_tc'):
                    base_name = base_name[:-3]
                names += [base_name + '_tc', base_name]
            TEMPLATE_FILES = tuple(os.path.join(
                data_dir, 'templates', self.default_template_name.format(name))
                for name in dict.fromkeys(names))

    # -------------------------------------------------------------------------
    # ESPADONS SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    def param_override(self):
        """
        Parameter override for ESPaDOnS parameters
        (update default params)

        :return: None - updates self.params
        """
        # set function name
        func_name = __NAME__ + '.Espadons.override()'
        # set parameters to update
        self.param_set('INSTRUMENT', 'ESPADONS', source=func_name)
        # add instrument earth location
        #    (for use in astropy.coordinates.EarthLocation)
        self.param_set('EARTH_LOCATION', 'cfht')
        # define the default science input files: the intensity spectra
        self.param_set('INPUT_FILE', '*i.fits', source=func_name)
        # the blaze is given back by load_blaze_from_science (see the module)
        self.param_set('BLAZE_CORRECTED', False, source=func_name)
        # define the mask table format
        self.param_set('REF_TABLE_FMT', 'csv', source=func_name)
        # define the mask type
        self.param_set('SCIENCE_MASK_TYPE', 'full', source=func_name)
        self.param_set('FP_MASK_TYPE', 'neg', source=func_name)
        self.param_set('LFC_MASK_TYPE', 'neg', source=func_name)
        # define the default mask url and filename
        self.param_set('DEFAULT_MASK_FILE', source=func_name,
                       value='mdwarf_harps.fits')
        # define the High pass width in km/s
        self.param_set('HP_WIDTH', 500, source=func_name)
        # approximate mean resolution in lambda/dlambda (65 000 in the
        #   polarimetric and star+sky modes, 80 000 star only)
        self.param_set('APPROX_RESOLUTION', 65000, source=func_name)
        # define the SNR cut off threshold
        self.param_set('SNR_THRESHOLD', 10, source=func_name)
        # define which bands to use for the clean CCF (see astro.ccf_regions)
        self.param_set('CCF_CLEAN_BANDS', ['r'], source=func_name)
        # define the plot order for the compute rv model plot (row of m = 31)
        self.param_set('COMPUTE_MODEL_PLOT_ORDERS',
                       [int(np.where(ORDERS == SNR_ORDER)[0][0])],
                       source=func_name)
        # define the compil minimum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MIN', 380.0, source=func_name)
        # define the compil maximum wavelength allowed for lines [nm]
        self.param_set('COMPIL_WAVE_MAX', 1040.0, source=func_name)
        # define the maximum pixel width allowed for lines [pixels]
        self.param_set('COMPIL_MAX_PIXEL_WIDTH', 500, source=func_name)
        # define min likelihood of correlation with BERV
        self.param_set('COMPIL_CUT_PEARSONR', -1, source=func_name)
        # define the CCF e-width to use for FP files
        self.param_set('COMPIL_FP_EWID', 5.0, source=func_name)
        # define whether to add the magic "binned wavelength" bands rv
        self.param_set('COMPIL_ADD_UNIFORM_WAVEBIN', True)
        # define the number of bins used in the magic "binned wavelength" bands
        self.param_set('COMPIL_NUM_UNIFORM_WAVEBIN', 15)
        # define the first band (from get_binned_parameters) to plot (band1)
        self.param_set('COMPILE_BINNED_BAND1', 'r', source=func_name)
        # define the second band (from get_binned_parameters) to plot (band2)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND2', 'r', source=func_name)
        # define the third band (from get_binned_parameters) to plot (band3)
        #    this is used for colour   band2 - band3
        self.param_set('COMPILE_BINNED_BAND3', 'z', source=func_name)
        # define the reference wavelength used in the slope fitting in nm
        self.param_set('COMPIL_SLOPE_REF_WAVE', 650, source=func_name)
        # define the name of the sample wave grid file (saved to the calib dir)
        self.param_set('SAMPLE_WAVE_GRID_FILE', self.default_sample_wave_name,
                       source=func_name)
        # no simultaneous calibration on ESPaDOnS
        self.param_set('FP_REF_LIST', ['FP'], source=func_name)
        self.param_set('FP_STD_LIST', ['FP'], source=func_name)
        # define readout noise per instrument (assumes ~4e- and 10 pixels)
        self.param_set('READ_OUT_NOISE', 13, source=func_name)
        # Define the wave url for the stellar models
        self.param_set('STELLAR_WAVE_URL', source=func_name,
                       value='ftp://phoenix.astro.physik.uni-goettingen.de/'
                             'HiResFITS/')
        # Define the wave file for the stellar models (using wget)
        self.param_set('STELLAR_WAVE_FILE', source=func_name,
                       value='WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        # Define the stellar model url
        self.param_set('STELLAR_MODEL_URL', source=func_name,
                       value='ftp://phoenix.astro.physik.uni-goettingen.de/'
                             'HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
                             '{ZSTR}{ASTR}/')
        # Define the minimum allowed SNR in a pixel to add it to the mask
        self.param_set('MASK_SNR_MIN', value=5, source=func_name)
        # Define the stellar model file name (using wget, with appropriate
        #     format  cards)
        self.param_set('STELLAR_MODEL_FILE', source=func_name,
                       value='lte{TEFF}-{LOGG}-{ZVALUE}{ASTR}'
                             '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
        # Define the object surface gravity (log g) (stellar model)
        self.param_set('OBJECT_LOGG', value=4.5, source=func_name)
        # Define the object Z (stellar model)
        self.param_set('OBJECT_Z', value=0.0, source=func_name)
        # Define the object alpha (stellar model)
        self.param_set('OBJECT_ALPHA', value=0.0, source=func_name)
        # blaze smoothing size (s1d template)
        self.param_set('BLAZE_SMOOTH_SIZE', value=20, source=func_name)
        # blaze threshold (s1d template)
        self.param_set('BLAZE_THRESHOLD', value=0.2, source=func_name)
        # define the size of the berv bins in m/s
        self.param_set('BERVBIN_SIZE', value=3000)
        # ---------------------------------------------------------------------
        # define whether to do the tellu-clean: the spectra are not corrected
        self.param_set('DO_TELLUCLEAN', value=True, source=func_name)
        # define the dv offset for tellu-cleaning in km/s
        self.param_set('TELLUCLEAN_DV0', value=0, source=func_name)
        # Define the lower wave limit for the absorber spectrum masks in nm
        self.param_set('TELLUCLEAN_MASK_DOMAIN_LOWER', value=500,
                       source=func_name)
        # Define the upper wave limit for the absorber spectrum masks in nm
        self.param_set('TELLUCLEAN_MASK_DOMAIN_UPPER', value=700,
                       source=func_name)
        # Define whether to force using airmass from header
        self.param_set('TELLUCLEAN_FORCE_AIRMASS', value=False,
                       source=func_name)
        # Define the CCF scan range in km/s
        self.param_set('TELLUCLEAN_CCF_SCAN_RANGE', value=50,
                       source=func_name)
        # Define the maximum number of iterations for the tellu-cleaning loop
        self.param_set('TELLUCLEAN_MAX_ITERATIONS', value=20, source=func_name)
        # Define the kernel width in pixels
        self.param_set('TELLUCLEAN_KERNEL_WID', value=1.4, source=func_name)
        # Define the gaussian shape (2=pure gaussian, >2=boxy)
        self.param_set('TELLUCLEAN_GAUSSIAN_SHAPE', value=2.2,
                       source=func_name)
        # Define the wave grid lower wavelength limit in nm
        self.param_set('TELLUCLEAN_WAVE_LOWER', value=350, source=func_name)
        # Define the wave griv upper wavelength limit (ESPaDOnS reaches the
        #   water bands of 900-1050 nm)
        self.param_set('TELLUCLEAN_WAVE_UPPER', value=1050, source=func_name)
        # Define the transmission threshold exp(-1) at which tellurics are
        #     uncorrectable
        self.param_set('TELLUCLEAN_TRANSMISSION_THRESHOLD', value=-1,
                       source=func_name)
        # Define the sigma cut threshold above which pixels are removed from fit
        self.param_set('TELLUCLEAN_SIGMA_THRESHOLD', value=10,
                       source=func_name)
        # Define whether to recenter the CCF on the first iteration
        self.param_set('TELLUCLEAN_RECENTER_CCF', value=False,
                       source=func_name)
        # Define whether to recenter the CCF of others on the first iteration
        self.param_set('TELLUCLEAN_RECENTER_CCF_FIT_OTHERS', value=False,
                       source=func_name)
        # Define the default water absorption to use
        self.param_set('TELLUCLEAN_DEFAULT_WATER_ABSO', value=0.5,
                       source=func_name)
        # Define the lower limit on valid exponent of water absorbers
        self.param_set('TELLUCLEAN_WATER_BOUNDS_LOWER', value=0.01,
                       source=func_name)
        # Define the upper limit on valid exponent of water absorbers
        self.param_set('TELLUCLEAN_WATER_BOUNDS_UPPER', value=15,
                       source=func_name)
        # Define the lower limit on valid exponent of other absorbers
        self.param_set('TELLUCLEAN_OTHERS_BOUNDS_LOWER', value=0.05,
                       source=func_name)
        # Define the upper limit on valid exponent of other absorbers
        self.param_set('TELLUCLEAN_OTHERS_BOUNDS_UPPER', value=15,
                       source=func_name)
        # ---------------------------------------------------------------------
        # Parameters for the template construction
        # ---------------------------------------------------------------------
        # max number of bins for the median of the template. Avoids handling
        # too many spectra at once.
        self.param_set('TEMPLATE_MEDBINMAX', 19, source=func_name)
        # maximum RMS between the template and the median of the template
        # to accept the median of the template as a good template. If above
        # we iterate once more. Expressed in m/s
        self.param_set('MAX_CONVERGENCE_TEMPLATE_RV', 100, source=func_name)
        # ---------------------------------------------------------------------
        # Header keywords: those the class adds (load_header) and CFHT's
        # ---------------------------------------------------------------------
        # define the key that gives the mid exposure time in MJD
        self.param_set('KW_MID_EXP_TIME', KEY_MJDMID, source=func_name)
        # define the start time of the observation
        self.param_set('KW_MJDATE', 'MJDATE', source=func_name)
        # define snr keyword
        self.param_set('KW_SNR', KEY_SNR, source=func_name)
        # define berv keyword
        self.param_set('KW_BERV', KEY_BERV, source=func_name)
        # define the Blaze calibration file (none: from the error bars)
        self.param_set('KW_BLAZE_FILE', 'FILENAME', source=func_name)
        # define the exposure time of the observation
        self.param_set('KW_EXPTIME', 'EXPTIME', source=func_name)
        # define the airmass of the observation
        self.param_set('KW_AIRMASS', 'AIRMASS', source=func_name)
        # define the human date of the observation
        self.param_set('KW_DATE', 'DATE-OBS', source=func_name)
        # define the tau_h20 of the observation
        self.param_set('KW_TAU_H2O', 'TLPEH2O', source=func_name)
        # define the tau_other of the observation
        self.param_set('KW_TAU_OTHERS', 'TLPEOTR', source=func_name)
        # define the DPRTYPE of the observation
        self.param_set('KW_DPRTYPE', 'OBSTYPE', source=func_name)
        # define the filename of the wave solution
        self.param_set('KW_WAVEFILE', 'FILENAME', source=func_name)
        # define the original object name
        self.param_set('KW_OBJNAME', 'OBJECT', source=func_name)
        # define the SNR goal per pixel per frame (can not exist - will be
        #   set to zero)
        self.param_set('KW_SNRGOAL', 'SNRGOAL', source=func_name)
        # define the SNR in chosen order
        self.param_set('KW_EXT_SNR', KEY_SNR, source=func_name)
        # define the barycentric julian date
        self.param_set('KW_BJD', KEY_BJD, source=func_name)
        # define the reference header key (must also be in rdb table) to
        #    distinguish FP calibration files from FP simultaneous files
        self.param_set('KW_REF_KEY', 'OBSTYPE', source=func_name)
        # velocity of template from CCF
        self.param_set('KW_MODELVEL', 'MODELVEL', source=func_name)
        # the temperature of the object
        self.param_set('KW_TEMPERATURE', None, source=func_name)

    # -------------------------------------------------------------------------
    # ESPADONS SPECIFIC METHODS
    # -------------------------------------------------------------------------
    def load_header(self, filename: str, kind: str = 'fits file',
                    extnum: int = 0, extname: str = None) -> io.LBLHeader:
        """
        Load a header, with the keys the class adds to a science file's:
        the middle of the exposure (MJDMID), the barycentric correction and
        time (BERV in km/s, BJD), the heliocentric velocity the reduction
        moved the wavelengths by (HELIOVEL) and the SNR at 731 nm (SNR31)

        :param filename: str, the filename to load
        :param kind: str, the kind of file we are loading
        :param extnum: int, the extension number to load
        :param extname: str, the extension name to load
        :return:
        """
        # get header
        hdr = io.load_header(filename, kind, extnum, extname)
        # a science exposure as CADC delivers it: what LBL needs that the
        #   file does not say (a file LBL wrote has the keys already)
        if ('MJDATE' in hdr and 'RA' in hdr and 'COL1' in hdr
                and KEY_MJDMID not in hdr):
            mjdmid = mid_exposure(hdr)
            berv, bjd = barycentric(str(hdr['RA']), str(hdr['DEC']),
                                    round(mjdmid, 8))
            hdr[KEY_MJDMID] = (mjdmid, 'MJD (UTC) at mid exposure, LBL')
            hdr[KEY_BERV] = (berv, '[km/s] barycentric correction, LBL')
            hdr[KEY_BJD] = (bjd, 'BJD_TDB at mid exposure, LBL')
            hdr[KEY_HELIO] = (heliocentric_velocity(hdr),
                              '[km/s] undone heliocentric shift, LBL')
            # the zero point measured on the tellurics (read_espadons)
            path = os.path.abspath(filename)
            if path not in TELLURIC_FITS:
                read_espadons(path)
            fit = TELLURIC_FITS.get(path, dict(dv=np.nan, sdv=np.nan,
                                               template='none'))
            finite = np.isfinite(fit['dv'])
            hdr[KEY_TELLDV] = (1000 * fit['dv'] if finite else 0.0,
                               '[m/s] RV correction from the tellurics, LBL')
            hdr[KEY_TELLDVE] = (1000 * fit['sdv'] if finite else 0.0,
                                '[m/s] its error, LBL')
            hdr[KEY_TELLTPL] = (str(fit.get('template', 'none'))[:60],
                                'star divided out to measure it, LBL')
            snr = order_snr(hdr)
            hdr[KEY_SNR] = (0.0 if not np.isfinite(snr) else snr,
                            'SNR at {0} (order #{1}), LBL'.format(
                                '731 nm', SNR_ORDER))
        # return the LBL Header class
        return io.LBLHeader.from_fits(hdr, filename)

    def load_science_file(self, science_file: str
                          ) -> Tuple[np.ndarray, io.LBLHeader]:
        """
        Load a science exposure: its orders as rows (read_espadons), the
        blaze put back

        :param science_file: str, absolute path to filename

        :return: tuple, data (np.ndarray) and header (io.LBLHeader)
        """
        _, flux, _ = read_espadons(os.path.abspath(science_file))
        sci_hdr = self.load_header(science_file, kind='science fits file')
        return np.array(flux), sci_hdr

    def write_tellu_cleaned(self, write_tellu_file: str, props: dict,
                            sci_hdict: io.LBLHeader,
                            science_filename: Optional[str] = None):
        """
        Write the telluric-cleaned spectrum: the header, the cleaned rows,
        and the wave map and the blaze they go with (WAVE, BLAZE), which an
        ESPaDOnS file does not carry as images and read_espadons reads back

        :param write_tellu_file: str, the file and path to write to
        :param props: dictionnary output from the TELLUCLEANed code
        :param sci_hdict: io.LBLHeader, the science file's header
        :param science_filename: str, the science filename
        :return:
        """
        source = science_filename or props['FILENAME']
        wave, _, blaze = read_espadons(os.path.abspath(source))
        # convert hdict to header
        header = io.copy_header(fits.Header(), sci_hdict.to_fits())
        # standard lbl keys
        header = self.lbl_hkeys(header, output='LBL_TELLU_CLEAN')
        # define the telluric parameters
        header = self.set_hkey(header, 'KW_TAU_H2O',
                               props['pre_cleaned_exponent_water'])
        header = self.set_hkey(header, 'KW_TAU_OTHERS',
                               props['pre_cleaned_exponent_others'])
        flux_hdr, wave_hdr, blaze_hdr = (fits.Header(), fits.Header(),
                                         fits.Header())
        flux_hdr['EXTNAME'] = 'FLUX'
        wave_hdr['EXTNAME'] = 'WAVE'
        blaze_hdr['EXTNAME'] = 'BLAZE'
        log.general('Saving tellu-cleaned file: {0}'.format(write_tellu_file))
        io.write_fits(write_tellu_file,
                      data=[None, props['pre_cleaned_flux'], wave, blaze],
                      header=[header, flux_hdr, wave_hdr, blaze_hdr],
                      dtype=[None, 'image', 'image', 'image'])

    def mask_file(self, model_directory: str, mask_directory: str,
                  required: bool = True) -> str:
        """
        Make the absolute path for the mask file

        :param model_directory: str, the directory the model is located at
        :param mask_directory: str, the directory the mask should be copied to
        :param required: bool, if True checks that file exists on disk

        :return: absolute path to mask file
        """
        # copy the default mask file to the mask directory
        self.copy_default_mask(model_directory, mask_directory,
                               self.params['DEFAULT_MASK_FILE'])
        # get data type
        data_type = self.params['DATA_TYPE']
        # get type of mask
        mask_type = self.params['{0}_MASK_TYPE'.format(data_type)]
        # deal with no object
        if self.params['MASK_FILE'] not in [None, 'None', '']:
            # define base name
            basename = self.params['MASK_FILE']
            # if basename is full path use this
            if os.path.exists(basename):
                abspath = str(basename)
            else:
                # get absolute path
                abspath = os.path.join(mask_directory, basename)
        elif self.params['OBJECT_COMPARISON'] is None:
            raise LblException('OBJECT_COMPARISON name must be defined')
        else:
            objname = self.params['OBJECT_COMPARISON']
            # define base name
            basename = self.default_mask_name.format(obj=objname,
                                                     mtype=mask_type)
            # get absolute path
            abspath = os.path.join(mask_directory, basename)
        # check that this file exists
        if required:
            io.check_file_exists(abspath, 'mask')
        # return absolute path
        return abspath

    def blaze_file(self, directory: str) -> Union[str, None]:
        """
        No blaze file: the blaze comes from each file's error bars

        :param directory: str, the directory the file is located at

        :return: None
        """
        _ = directory
        return None

    def load_blaze(self, filename: str, science_file: Optional[str] = None,
                   normalize: bool = True) -> Union[np.ndarray, None]:
        """
        No blaze file: the blaze comes from each file's error bars
        (load_blaze_from_science)

        :return: None
        """
        _ = filename, science_file, normalize
        return None

    def get_mask_systemic_vel(self, mask_file: str) -> float:
        """
        Get the systemic velocity in m/s of the mask

        :param mask_file: the absolute path to the mask file

        :return: float, systemic velocity in m/s
        """
        # get systemic velocity key
        sysvelkey = self.params['KW_SYSTEMIC_VELO']
        # load the mask header
        mask_hdr = self.load_header(mask_file, kind='mask fits file')
        # get info on template systvel for splining correctly
        systemic_vel = mask_hdr.get_hkey(sysvelkey, dtype=float)
        # return systemic velocity in m/s
        return systemic_vel

    def science_files(self, directory: str) -> np.ndarray:
        """
        List the absolute paths of all science files

        :param directory: str, the directory the file is located at

        :return: absolute path to template file
        """
        # deal with no object
        if self.params['OBJECT_SCIENCE'] is None:
            raise LblException('OBJECT_SCIENCE name must be defined')
        else:
            objname = self.params['OBJECT_SCIENCE']
        # deal the input file string
        if self.params['INPUT_FILE'] is None:
            raise LblException('INPUT_FILE must be defined')
        # check that the object sub-directory exists
        abspath = io.make_dir(directory, objname, 'Science object')
        # set up basename
        basename = os.path.basename(self.params['INPUT_FILE'])
        # add to abspath
        abspath = os.path.join(abspath, basename)
        # look for files
        files = glob.glob(abspath)
        # deal with no files found
        if len(files) == 0:
            emsg = 'No science objects found for {0}. Search string={1}'
            eargs = [objname, abspath]
            raise LblException(emsg.format(*eargs))
        else:
            # sort files
            files = np.sort(files)
            # return numpy array of files
            return files

    def sort_science_files(self, science_files: List[str]) -> List[str]:
        """
        Sort science files by the middle of their exposure

        :param science_files: list of strings - list of science files

        :return: list of strings - sorted list of science files
        """
        times = []
        # loop around science files
        for science_file in science_files:
            # load header
            sci_hdr = self.load_header(science_file)
            # get mid exposure time
            times.append(sci_hdr.get_hkey(self.params['KW_MID_EXP_TIME'],
                                          science_file, dtype=float))
        # apply sort mask
        science_files = np.array(science_files)[np.argsort(times)]
        # return sorted files
        return list(science_files)

    def load_blaze_from_science(self, science_file: str,
                                sci_image: np.ndarray,
                                sci_hdr: io.LBLHeader,
                                calib_directory: str,
                                normalize: bool = True
                                ) -> Tuple[np.ndarray, bool]:
        """
        The blaze of a science file, from its own error bars
        (read_espadons), normalised per order

        :param science_file: str, the science file
        :param sci_image: np.array - the science image
        :param sci_hdr: io.LBLHeader - the science file header
        :param calib_directory: str, the directory containing calibration files
        :param normalize: bool, ignored (always normalised per order)

        :return: the blaze and a flag whether blaze is set to ones (science
                 image already blaze corrected)
        """
        _ = sci_image, sci_hdr, calib_directory, normalize
        _, _, blaze = read_espadons(os.path.abspath(science_file))
        return np.array(blaze), False

    def get_wave_solution(self, science_filename: Optional[str] = None,
                          data: Optional[np.ndarray] = None,
                          header: Optional[io.LBLHeader] = None
                          ) -> np.ndarray:
        """
        The wave map of a science file (read_espadons): nm, vacuum, the
        observer's frame

        :param science_filename: str, the absolute path to the file
        :param header: io.LBLHeader, not used
        :param data: np.ndarray, not used

        :return: np.ndarray, the wave map. Shape = (num orders x num pixels)
        """
        _ = data, header
        if science_filename is None:
            raise LblException('ESPaDOnS: the wave solution is read from the'
                               ' science file, which must be given')
        wave, _, _ = read_espadons(os.path.abspath(science_filename))
        return np.array(wave)

    def load_bad_hdr_keys(self) -> Tuple[list, Any]:
        """
        No bad keys for ESPaDOnS

        :return: tuple, 1. the list of bad values, 2. the bad key in
                 a file header to check against bad values
        """
        return [], None

    def get_berv(self, sci_hdr: io.LBLHeader) -> float:
        """
        The barycentric correction in m/s (load_header): the spectra are in
        the observer's frame

        :param sci_hdr: io.LBLHeader, the science header

        :return: float, the BERV in m/s
        """
        return sci_hdr.get_hkey(self.params['KW_BERV'], dtype=float) * 1000

    def populate_sci_table(self, filename: str, tdict: dict,
                           sci_hdr: io.LBLHeader, berv: float = 0.0) -> dict:
        """
        Populate the science table

        :param filename: str, the filename of the science image
        :param tdict: dictionary, the storage dictionary for science table
                      can be empty or have previous rows to append to
        :param sci_hdr: fits Header, the header of the science image
        :param berv: float, the berv value to add to storage dictionary

        :return: dict, a dictionary table of the science parameters
        """
        # these are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_DATE', 'KW_DPRTYPE', 'KW_OBJNAME', 'KW_EXT_SNR']
        # add the filename
        tdict = self.add_dict_list_value(tdict, 'FILENAME', filename)
        # loop around header keys
        for drs_key in drs_keys:
            # if key is in params we can add the value to keys
            if drs_key in self.params:
                key = self.params[drs_key]
            else:
                key = str(drs_key)
            # get value from header
            value = sci_hdr.get(key, 'NULL')
            # add to tdict
            tdict = self.add_dict_list_value(tdict, drs_key, value)
        # add the berv separately
        tdict = self.add_dict_list_value(tdict, 'BERV', berv)
        # return updated storage dictionary
        return tdict

    def rdb_columns(self) -> Tuple[np.ndarray, List[bool]]:
        """
        Define the fits header columns names to add to the RDB file
        These should be references to keys in params

        :return: tuple, 1. np.array of strings (the keys), 2. list of bools
                 the flags whether these keys should be used with FP files
        """
        # there are defined in params
        drs_keys = ['KW_MJDATE', 'KW_MID_EXP_TIME', 'KW_EXPTIME',
                    'KW_AIRMASS', 'KW_DATE', 'KW_BERV', 'KW_DPRTYPE',
                    'KW_TAU_H2O', 'KW_TAU_OTHERS', 'KW_NITERATIONS',
                    'KW_RESET_RV',
                    'KW_SYSTEMIC_VELO', 'KW_WAVEFILE', 'KW_OBJNAME',
                    'KW_EXT_SNR', 'KW_BJD', 'KW_CCF_EW', KEY_TELLDV,
                    KEY_TELLDVE]
        # convert to actual keys (not references to keys)
        keys = []
        fp_flags = []
        for drs_key in drs_keys:
            # initial set fp flag to False
            fp_flag = False
            # ignore keys that are None
            if drs_key is None:
                continue
            # if key is in params we can add the value to keys
            if drs_key in self.params:
                keys.append(self.params[drs_key])
                # we can also look for fp flag - this is either True or False
                #    if True we skip this key for FP files - default is False
                #    (i.e. not to skip)
                instance = self.params.instances[drs_key]
                if instance is not None:
                    if instance.fp_flag is not None:
                        fp_flag = instance.fp_flag
            else:
                keys.append(drs_key)
            # append fp flags
            fp_flags.append(fp_flag)
        # return a numpy array
        return np.array(keys), fp_flags

    def fix_lblrv_header(self, header: io.LBLHeader) -> io.LBLHeader:
        """
        Fix the LBL RV header

        :param header: io.LBLHeader, the LBL RV fits file header

        :return: io.LBLHeader, the updated LBL RV fits file header
        """
        # get keys from params
        kw_snrgoal = self.params['KW_SNRGOAL']
        kw_ccf_ew = self.params['KW_CCF_EW']
        # ---------------------------------------------------------------------
        # an SNR goal is only there for the queue's observations
        if kw_snrgoal not in header:
            header[kw_snrgoal] = 0
        # ---------------------------------------------------------------------
        # deal with not having CCF_EW
        if kw_ccf_ew not in header:
            header[kw_ccf_ew] = 5.5 / mp.fwhm_value() * 1000
        # ---------------------------------------------------------------------
        # return header
        return header

    def get_rjd_value(self, header: io.LBLHeader) -> float:
        """
        Get the rjd from KW_BJD (or KW_MID_EXP_TIME without it)
        time returned is in MJD (not JD)

        :param header: io.LBLHeader - the LBL rv header
        :return:
        """
        # get keys from params
        kw_mjdmid = self.params['KW_MID_EXP_TIME']
        kw_bjd = self.params['KW_BJD']
        # get mjdmid and bjd
        mid_exp_time = header.get_hkey(kw_mjdmid, dtype=float)
        bjd = header.get_hkey(kw_bjd, dtype=float)
        if isinstance(bjd, str) or np.isnan(bjd):
            # return RJD = MJD + 0.5
            return float(mid_exp_time) + 0.5
        else:
            # convert bjd to mjd
            bjd_mjd = Time(bjd, format='jd').mjd
            # return RJD = MJD + 0.5
            return float(bjd_mjd) + 0.5

    def get_plot_date(self, header: io.LBLHeader):
        """
        Get the matplotlib plotting date

        :param header: io.LBLHeader - the LBL rv header

        :return: float, the plot date
        """
        # get mjdate key
        kw_mjdate = self.params['KW_MJDATE']
        # get mjdate
        mjdate = header.get_hkey(kw_mjdate, dtype=float)
        # convert to plot date and take off JD?
        plot_date = Time(mjdate, format='mjd').plot_date
        # return float plot date
        return float(plot_date)

    def get_binned_parameters(self) -> Dict[str, list]:
        """
        Defines a "binning dictionary" splitting up the array by:

        Each binning dimension has [str names, start value, end value]

        - bands  (in wavelength)
            [bands / blue_end / red_end]

        - cross order regions (in pixels): the whole row only, since a row's
          pixels are Libre-ESpRIT's samples and not the detector's

        :return: dict, the binned dictionary
        """
        # ---------------------------------------------------------------------
        # define regions, and blue/red band ends
        bout = astro.choose_bands(astro.bands, self.wavemin, self.wavemax)
        bands, blue_end, red_end, use_regions = bout
        # ---------------------------------------------------------------------
        # return all this information (in a dictionary)
        binned = dict()
        binned['bands'] = list(bands)
        binned['blue_end'] = list(blue_end)
        binned['red_end'] = list(red_end)
        binned['region_names'] = ['']
        binned['region_low'] = [0]
        binned['region_high'] = [NPIX]
        binned['use_regions'] = list(use_regions)
        # ---------------------------------------------------------------------
        # return this binning dictionary
        return binned


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================

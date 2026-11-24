#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The template without resampling a spectrum: BSD, BERV Spectral Drizzle
(TEMPLATE_METHOD = 'BSD'; 'SCATTER', its old name, is still accepted).

Every pixel of every spectrum keeps its own wavelength, moved to the star's
frame (BERV and the star's velocity, relativistic): for one order, the pixels
of all the spectra are one cloud of points (wavelength, flux), which the
spread of the BERV over the observations samples far more finely than the
pixels. The template on its grid, and its first three derivatives, are a
local polynomial fitted through that cloud around each point of the grid
(lbl.core.gaussian_savgol): no spectrum is interpolated, the derivatives are
those of the fitted polynomial, and the uncertainty of each point comes with
it. This replaces the median of spectra resampled into bins of BERV
(TEMPLATE_METHOD = 'BERVBIN').

Per order, in TEMPLATE_BSD_NITER passes:
    1. the fit through the cloud, each point weighted by the inverse square
       of its noise (a low pass of its residuals), with a clip of the local
       outliers at TEMPLATE_BSD_CLIP times the RMS of the residuals of
       their window (in units of their noise), in a
       Gaussian window of TEMPLATE_BSD_FWHM resolution elements;
    2. a pixel of the detector whose residual is off in most spectra (the
       median over spectra of the residual in units of the noise beyond
       TEMPLATE_BSD_OBS_CLIP) is left out of all spectra: what stays on
       the detector while the star moves (telluric residuals, bad pixels);
    3. each spectrum is divided by a low pass (HP_WIDTH) of its ratio to the
       template, for what the blaze division leaves.
The orders are then combined on the grid with the weights the s1d uses
(the blaze and the smooth fall at the order edges): all orders, the odd
ones and the even ones.

With SMALL_RV_SHIFT False and no known orbits, the velocities of the spectra
are first measured on the orders around COMPIL_SLOPE_REF_WAVE (a scan of the
shift, then Bouchy et al. 2001 pixel by pixel on the template's Savitzky-
Golay derivative, the pixels combined by LBL's mixture model, odd_ratio_mean,
the steps iterated until they converge), pass after pass, until 95 % of the
spectra changed by less than 2 sigma (TEMPLATE_RV_CONV_FRAC,
TEMPLATE_RV_CONV_NSIG). They start from the object's velocity box
(lbl.science.velocity_box, a CCF against the PHOENIX model the first time)
and are written back to it. They are kept in the template's SCI_TABLE
(VSYS), where PCA2D reads them for its star frame.

The window is about one resolution element, not the 3 steps of the grid of
templatepy: in a narrow window each spectrum's own noise is copied sharply
into the template, at that spectrum's pixels, which pulls the velocity of the
spectrum towards the template's and shrinks all velocities (by 16 % on 40
NIRPS spectra of GJ 1028 at 0.4 resolution element, against 2 % at one
resolution element and with BERVBIN).

After templatepy (github.com/eartigau/templatepy, E. Artigau).

@author: artigau
"""
import os
import warnings
from typing import Any, Callable, Dict, List

import numpy as np

from lbl.core import base_classes
from lbl.core import io
from lbl.core import math as mp
from lbl.core.gaussian_savgol import scatter_savgol
from lbl.science import apero

# =============================================================================
# Define variables
# =============================================================================
log = io.log
C_MS = mp.speed_of_light_ms
#: the points of a cloud: the derivatives returned (0 to 3)
NDERIV = 3


# =============================================================================
# Define functions
# =============================================================================
def bsd_param(params, name: str):
    """
    TEMPLATE_BSD_<name>, or TEMPLATE_SCATTER_<name>, its name from when the
    method was called SCATTER, when a wrap still sets that one

    :param params: ParamDict, the parameters of the instrument
    :param name: str, FWHM, NITER, CLIP, OBS_CLIP or MEMORY

    :return: the value of the parameter
    """
    old = params['TEMPLATE_SCATTER_' + name]
    return params['TEMPLATE_BSD_' + name] if old is None else old


def star_shift(berv: np.ndarray, rv0: np.ndarray) -> np.ndarray:
    """
    The shift in ln(wavelength) that takes an observed wavelength to the
    star's frame: ln of doppler_shift(wave, add_velocities(-berv, rv0)) / wave,
    as the BERVBIN template moves its spectra

    :param berv: np.ndarray, the BERV of each spectrum [m/s]
    :param rv0: np.ndarray, the star's velocity in each spectrum [m/s]

    :return: np.ndarray, the shift of each spectrum
    """
    velocity = mp.add_velocities(-np.asarray(berv, dtype=float),
                                 np.asarray(rv0, dtype=float))
    return -np.arctanh(velocity / C_MS)


def _noise(ratio: np.ndarray, width: int) -> np.ndarray:
    """
    The noise of each pixel of an order of one spectrum, in units of its
    normalised flux: a low pass of the absolute deviations (scaled from a
    mean absolute value to a sigma), never below a tenth of its median

    :param ratio: np.ndarray, deviations of the flux (from a neighbour or from
                  the template)
    :param width: int, the low pass width [pixels]

    :return: np.ndarray, the noise (NaN where the flux is NaN)
    """
    noise = mp.lowpassfilter(np.abs(ratio), width) * np.sqrt(np.pi / 2)
    if not np.any(np.isfinite(noise)):
        return noise
    floor = 0.1 * np.nanmedian(noise)
    noise = np.where(noise > floor, noise, floor)
    noise[~np.isfinite(ratio)] = np.nan
    return noise


class OrderData:
    """The pixels of some orders of all the spectra, as the fits read them:
    the observed ln(wavelength) less a reference per order, the normalised
    flux (flux / blaze over its median) and its noise, the level of the order
    in each spectrum against the whole spectrum, and the weight of each order
    for the combination of the orders"""

    def __init__(self, orders: List[int], nfiles: int, npix: int):
        self.orders = list(orders)
        shape = (len(self.orders), nfiles, npix)
        self.xobs = np.full(shape, np.nan, dtype=np.float32)
        self.flux = np.full(shape, np.nan, dtype=np.float32)
        self.noise = np.full(shape, np.nan, dtype=np.float32)
        self.ref = np.zeros(len(self.orders))
        # the median of flux / blaze in the order over that of the spectrum
        self.level = np.full(shape[:2], np.nan)
        # the mean (over spectra) of the s1d weight of each pixel, normalised
        #   to its maximum, and its mean observed ln(wavelength)
        self.weight = np.zeros((len(self.orders), npix))
        self.xmean = np.zeros((len(self.orders), npix))
        self.count = np.zeros((len(self.orders), npix))


def read_orders(inst, science_files: List[str], load_science: Callable,
                blaze, calib_dir: str, orders: List[int], hp_pix: np.ndarray,
                tqdm, sci_table: Dict[str, Any] = None, berv=None,
                desc: str = 'Reading spectra') -> OrderData:
    """
    Read the given orders of all the spectra

    :param inst: the instrument
    :param science_files: list of str, the spectra
    :param load_science: callable, filename -> (image, header)
    :param blaze: np.ndarray or None, a blaze for all the files (as
                  lbl_template loads it), else from each file
    :param calib_dir: str, the calibration directory
    :param orders: list of int, the orders to keep
    :param hp_pix: np.ndarray, the low pass width of each order [pixels]
    :param tqdm: the progress bar
    :param sci_table: dict or None, filled with each file's row when given
    :param berv: np.ndarray, the BERV of each file (for sci_table)
    :param desc: str, the progress bar's label

    :return: OrderData
    """
    data = None
    for it, filename in enumerate(tqdm(science_files, desc=desc, unit='file')):
        image, hdr = load_science(filename)
        wave = inst.get_wave_solution(filename, image, hdr)
        # the blaze as the BERVBIN template takes it
        if blaze is None:
            blazeimage, blaze_flag = inst.load_blaze_from_science(
                filename, image, hdr, calib_dir, normalize=False)
        elif np.sum(blaze.ravel()) == len(blaze.ravel()):
            blazeimage, blaze_flag = np.array(blaze), True
        else:
            blazeimage, blaze_flag = np.array(blaze), False
        if blaze_flag:
            image, blazeimage = inst.no_blaze_corr(image, wave)
        image = np.array(image, dtype=float)
        image[image == 0] = np.nan
        if sci_table is not None:
            inst.populate_sci_table(filename, sci_table, hdr, berv=berv[it])
        if data is None:
            data = OrderData(orders, len(science_files), image.shape[1])
            for io_, order in enumerate(orders):
                data.ref[io_] = np.log(np.nanmedian(wave[order]))
        # the s1d weights: the blaze, falling smoothly to zero at the order
        #   edges and where the blaze is under BLAZE_THRESHOLD
        slope = apero._s1d_slopevector(inst.params, image, blazeimage)
        # the level of the whole spectrum, for the level of each order
        with np.errstate(invalid='ignore', divide='ignore'):
            full = image / blazeimage
            glevel = np.nanmedian(full[(slope > 0) & np.isfinite(full)])
        for io_, order in enumerate(orders):
            with np.errstate(invalid='ignore', divide='ignore'):
                flux = image[order] / blazeimage[order]
                good = (slope[order] > 0) & np.isfinite(flux) & \
                    np.isfinite(wave[order]) & (wave[order] > 0)
            flux[~good] = np.nan
            if np.sum(good) < 10:
                continue
            level = np.nanmedian(flux)
            if not np.isfinite(level) or level <= 0:
                continue
            flux = flux / level
            data.level[io_, it] = level / glevel
            xobs = np.log(np.where(wave[order] > 0, wave[order], np.nan))
            data.xobs[io_, it] = xobs - data.ref[io_]
            data.flux[io_, it] = flux
            dev = np.append(np.diff(flux), np.nan) / np.sqrt(2)
            data.noise[io_, it] = _noise(dev, int(hp_pix[order]))
            wgt = slope[order] * blazeimage[order]
            wgt = np.where(np.isfinite(wgt), wgt, 0.0)
            if np.nanmax(wgt) > 0:
                data.weight[io_] += wgt / np.nanmax(wgt)
                data.xmean[io_] += np.where(np.isfinite(xobs), xobs, 0.0)
                data.count[io_] += np.isfinite(xobs)
    for io_ in range(len(orders)):
        with np.errstate(invalid='ignore', divide='ignore'):
            data.xmean[io_] = data.xmean[io_] / data.count[io_]
        data.weight[io_] = data.weight[io_] / max(1, len(science_files))
    return data


def _cloud(data: OrderData, io_: int, shifts: np.ndarray):
    """The points of one order in the star's frame, sorted: x, flux, weight"""
    x = data.xobs[io_].astype(float) + data.ref[io_] + shifts[:, None]
    flux = data.flux[io_].astype(float)
    noise = data.noise[io_].astype(float)
    good = np.isfinite(x) & np.isfinite(flux) & np.isfinite(noise) & \
        (noise > 0)
    order = np.argsort(x[good], kind='stable')
    return (np.ascontiguousarray(x[good][order]),
            np.ascontiguousarray(flux[good][order]),
            np.ascontiguousarray(1.0 / noise[good][order] ** 2))


def fit_order(data: OrderData, io_: int, shifts: np.ndarray,
              lngrid: np.ndarray, params: Dict[str, Any], hp_pix: int,
              niter: int) -> Dict[str, Any]:
    """
    The template of one order on the grid, in niter passes (fit, clip of the
    pixels that are off in most spectra, low pass of each spectrum's ratio to
    the template). The flux and noise of the order in data are updated.

    :param data: OrderData
    :param io_: int, the order's index in data
    :param shifts: np.ndarray, the shift of each spectrum (star_shift)
    :param lngrid: np.ndarray, ln of the template's grid
    :param params: dict, fwhm (window FWHM in ln wavelength), clip, obs_clip
    :param hp_pix: int, the low pass width [pixels]
    :param niter: int, the passes

    :return: dict(i0, i1: the slice of the grid, derivs (i1 - i0, 4): the
             template and its derivatives in ln(wavelength), errs: their
             uncertainties), or None without enough points
    """
    out = None
    for ite in range(niter):
        x, flux, weight = _cloud(data, io_, shifts)
        if len(x) < 100:
            return None
        i0 = int(np.searchsorted(lngrid, x[0]))
        i1 = int(np.searchsorted(lngrid, x[-1]))
        if i1 - i0 < 10:
            return None
        derivs, errs = scatter_savgol(x, flux, lngrid[i0:i1], params['fwhm'],
                                      weights=weight, polyorder=3,
                                      nderiv=NDERIV, sigma_clip=params['clip'],
                                      clip_iter=2, presorted=True,
                                      max_gap=params.get('max_gap'),
                                      min_balance=params.get('min_balance',
                                                             0.0))
        out = dict(i0=i0, i1=i1, derivs=derivs, errs=errs)
        if ite == niter - 1:
            break
        # the template at every pixel of every spectrum
        ok = np.isfinite(derivs[:, 0])
        if np.sum(ok) < 10:
            return None
        xs = data.xobs[io_].astype(float) + data.ref[io_] + shifts[:, None]
        model = np.interp(xs, lngrid[i0:i1][ok], derivs[ok, 0],
                          left=np.nan, right=np.nan)
        flux = data.flux[io_].astype(float)
        noise = data.noise[io_].astype(float)
        with np.errstate(invalid='ignore', divide='ignore'), \
                warnings.catch_warnings():
            # pixels with no spectrum (order edges) give all-NaN medians
            warnings.simplefilter('ignore', RuntimeWarning)
            resid = (flux - model) / noise
            # the pixels of the detector that are off in most spectra
            off = np.abs(np.nanmedian(resid, axis=0)) > params['obs_clip']
            flux[:, off] = np.nan
            # the low pass of each spectrum's ratio to the template, and the
            #   noise from what is left
            ratio = flux / model
        for ifile in range(flux.shape[0]):
            if not np.any(np.isfinite(ratio[ifile])):
                continue
            lowpass = mp.lowpassfilter(ratio[ifile], hp_pix)
            with np.errstate(invalid='ignore', divide='ignore'):
                flux[ifile] = flux[ifile] / lowpass
                data.noise[io_, ifile] = _noise(ratio[ifile] / lowpass - 1,
                                                hp_pix)
        data.flux[io_] = flux
    return out


def spectrum_1d(inst, filename: str, load_science: Callable, blaze,
                calib_dir: str):
    """
    One spectrum as the CCF against the model takes it: every order divided
    by its blaze (as read_orders does) and by its own median, where the s1d
    weights are not nil; all the orders together

    :return: (wavelength [nm], normalised flux), 1D
    """
    image, hdr = load_science(filename)
    wave = inst.get_wave_solution(filename, image, hdr)
    if blaze is None:
        blazeimage, blaze_flag = inst.load_blaze_from_science(
            filename, image, hdr, calib_dir, normalize=False)
    elif np.sum(blaze.ravel()) == len(blaze.ravel()):
        blazeimage, blaze_flag = np.array(blaze), True
    else:
        blazeimage, blaze_flag = np.array(blaze), False
    if blaze_flag:
        image, blazeimage = inst.no_blaze_corr(image, wave)
    image = np.array(image, dtype=float)
    image[image == 0] = np.nan
    slope = apero._s1d_slopevector(inst.params, image, blazeimage)
    waves, fluxes = [], []
    for order in range(image.shape[0]):
        with np.errstate(invalid='ignore', divide='ignore'):
            flux = image[order] / blazeimage[order]
            good = (slope[order] > 0) & np.isfinite(flux) & \
                np.isfinite(wave[order]) & (wave[order] > 0) & (flux > 0)
        if np.sum(good) < 10:
            continue
        level = np.nanmedian(flux[good])
        if not np.isfinite(level) or level <= 0:
            continue
        waves.append(wave[order][good])
        fluxes.append(flux[good] / level)
    if not waves:
        return np.array([]), np.array([])
    return np.concatenate(waves), np.concatenate(fluxes)


def start_velocities(inst, science_files: List[str], load_science: Callable,
                     blaze, calib_dir: str, berv: np.ndarray,
                     rv0: np.ndarray, objname: str, data_dir: str,
                     model_dir: str, tqdm):
    """
    Where the passes on the velocities start: each spectrum's velocity from
    the object's box (lbl.science.velocity_box), the spectra not in it
    measured first by a CCF against the PHOENIX model at the star's Teff
    (general.sysvel_ccf, as the mask's systemic velocity is) and put in it.
    The template keeps its frame (VSYS relative): the start is the box's
    velocity less its systemic velocity (the median of the CCF velocities).
    Known orbits (rv0) are kept for the spectra the box has from a CCF only,
    theirs being finer; a spectrum without either starts at rv0.

    :return: (the start [m/s], the systemic velocity [m/s] or NaN)
    """
    from lbl.science import general, velocity_box
    start = np.array(rv0, dtype=float)
    if not data_dir or not objname:
        return start, np.nan
    box = velocity_box.read(data_dir, objname)
    found = velocity_box.velocities(box, science_files)
    missing = [it for it in range(len(science_files))
               if not found['found'][it]]
    if missing:
        try:
            m_wavemap, m_spectrum = general.get_stellar_models(inst,
                                                               model_dir)
        except Exception as exc:                          # noqa: BLE001
            log.warning('Velocities of the spectra: no PHOENIX model for the '
                        'CCF ({0}): the passes start from {1}'.format(
                            exc, 'the known orbits' if np.any(start)
                            else 'the BERV alone'))
            m_wavemap = None
        if m_wavemap is not None:
            reference, entries, ccf_v = None, dict(), []
            for it in tqdm(missing, desc='CCF of the spectra against the '
                                         'model', unit='file'):
                try:
                    wave, flux = spectrum_1d(inst, science_files[it],
                                             load_science, blaze, calib_dir)
                    if reference is None:
                        lo = max(float(np.min(wave)), float(np.min(m_wavemap)))
                        hi = min(float(np.max(wave)), float(np.max(m_wavemap)))
                        reference = general.sysvel_reference(
                            lo, hi, m_wavemap, m_spectrum)
                    vel, _, _, signif = general.sysvel_ccf(wave, flux,
                                                           reference)
                except Exception as exc:                  # noqa: BLE001
                    log.warning('CCF of {0} against the model failed: {1}'
                                .format(os.path.basename(
                                    str(science_files[it])), exc))
                    continue
                if not signif > 5:
                    log.warning('CCF of {0} against the model: a peak at '
                                '{1:.1f} sigma only, not kept'.format(
                                    os.path.basename(str(science_files[it])),
                                    signif))
                    continue
                vstar = velocity_box.star_velocity(1000.0 * vel, berv[it])
                ccf_v.append(vstar)
                entries[science_files[it]] = dict(v=vstar, berv=berv[it])
            systemic = None
            if box.get('systemic') is None and ccf_v:
                systemic = dict(v=float(np.median(ccf_v)), by='ccf',
                                n=len(ccf_v),
                                teff=inst.params.get('OBJECT_TEFF'))
            if entries or systemic is not None:
                box = velocity_box.update(data_dir, objname, entries, 'ccf',
                                          systemic=systemic)
            log.general('Velocities of the spectra: {0} of {1} measured by a '
                        'CCF against the PHOENIX model, in {2}'.format(
                            len(entries), len(missing),
                            velocity_box.path(data_dir, objname)))
        found = velocity_box.velocities(box, science_files)
    sysvel = (box.get('systemic') or dict()).get('v')
    if sysvel is None or not found['found'].any():
        return start, np.nan
    sysvel = float(sysvel)
    relative = found['v'] - sysvel
    use = found['found'] & ~((found['by'] == 'ccf') & (start != 0))
    start[use] = relative[use]
    counts = ', '.join('{0} {1}'.format(int(np.sum(found['by'] == kind)), kind)
                       for kind in ('lbl', 'template', 'ccf')
                       if np.any(found['by'] == kind))
    log.general('Velocities of the spectra: the passes start from the box '
                '({0}; {1} without), systemic velocity {2:.3f} km/s'.format(
                    counts, int(np.sum(~found['found'])), sysvel / 1000.0))
    return start, sysvel


#: Bouchy's steps on one spectrum (measure_velocities): taken until a step is
#:   this fraction of the spectrum's uncertainty or less, or below
#:   BOUCHY_FLOOR [m/s], BOUCHY_MAX_STEPS at most
BOUCHY_TOLERANCE = 0.01
BOUCHY_FLOOR = 1e-3
BOUCHY_MAX_STEPS = 20


def measure_velocities(data: OrderData, fits: Dict[int, Dict[str, Any]],
                       shifts: np.ndarray, lngrid: np.ndarray,
                       scan: bool, scan_step: float,
                       errors: np.ndarray = None,
                       steps_taken: np.ndarray = None) -> np.ndarray:
    """
    The velocity of each spectrum against the templates of some orders: the
    shift of the template that matches the spectrum (a scan of +-50 steps
    first when scan is True, then Bouchy et al. 2001 pixel by pixel,
    combined by a mixture model, iterated until it converges)

    :param data: OrderData of the orders
    :param fits: dict, order index in data -> fit_order output
    :param shifts: np.ndarray, the current shift of each spectrum
    :param lngrid: np.ndarray, ln of the template's grid
    :param scan: bool, scan the shift first
    :param scan_step: float, the step of the scan [ln wavelength]
    :param errors: np.ndarray or None, filled with each spectrum's
                   uncertainty [m/s] (the mixture model's, last step)
    :param steps_taken: np.ndarray or None, filled with the number of
                        Bouchy steps each spectrum took

    :return: np.ndarray, the velocity to add to each spectrum's [m/s]
    """
    nfiles = len(shifts)
    delta = np.zeros(nfiles)
    steps = np.arange(-50, 51) * scan_step
    unconverged = []
    for ifile in range(nfiles):
        xs, ys, ws, tm, td = [], [], [], [], []
        for io_, fit in fits.items():
            if fit is None:
                continue
            ok = np.isfinite(fit['derivs'][:, 0]) & \
                np.isfinite(fit['derivs'][:, 1])
            if np.sum(ok) < 10:
                continue
            x = data.xobs[io_, ifile].astype(float) + data.ref[io_] + \
                shifts[ifile]
            flux = data.flux[io_, ifile].astype(float)
            noise = data.noise[io_, ifile].astype(float)
            good = np.isfinite(x) & np.isfinite(flux) & np.isfinite(noise)
            xs.append(x[good])
            ys.append(flux[good])
            ws.append(1.0 / noise[good] ** 2)
            tm.append((lngrid[fit['i0']:fit['i1']][ok], fit['derivs'][ok, 0],
                       fit['derivs'][ok, 1]))
        if not xs:
            continue
        shift = 0.0
        if scan:
            chi2 = np.zeros(len(steps))
            for k, step in enumerate(steps):
                for x, y, w, (grid, d0, _) in zip(xs, ys, ws, tm):
                    model = np.interp(x - step, grid, d0, left=np.nan,
                                      right=np.nan)
                    chi2[k] += np.nansum(w * (y - model) ** 2)
            kmin = int(np.clip(np.argmin(chi2), 1, len(steps) - 2))
            fit = np.polyfit(steps[kmin - 1:kmin + 2], chi2[kmin - 1:kmin + 2],
                             2)
            shift = -fit[1] / (2 * fit[0]) if fit[0] > 0 else steps[kmin]
        # BOUCHY ET AL. (2001), PIXEL BY PIXEL, COMBINED BY A MIXTURE MODEL.
        #   y(x) = T(x - shift - d) ~ T(x - shift) - d T'(x - shift): each
        #   pixel gives its own shift d_i = -(y_i - T_i) / T'_i, of
        #   uncertainty sigma_i / |T'_i|, and their weighted mean is the
        #   least-squares projection on the derivative (Bouchy's optimal
        #   weights, (T' / sigma)^2). Combined by LBL's mixture model
        #   (odd_ratio_mean: a Gaussian and a flat floor of outliers,
        #   Artigau et al. 2022) rather than by the plain weighted mean: a
        #   cosmic ray, a telluric residual or a bad pixel in one spectrum
        #   does not pull its velocity, which goes to PCA2D's star frame.
        #   Bouchy's step is the first order of the shift's expansion, right
        #   only for a small shift: it is iterated, the template and its
        #   derivative taken again at each new shift, until the step is
        #   BOUCHY_TOLERANCE of the spectrum's uncertainty or less
        #   (BOUCHY_MAX_STEPS at most). From 3 km/s off, with lines 4.5 km/s
        #   wide, two steps leave 169 m/s, three 0.3 m/s, five nothing
        converged = False
        for nstep in range(1, BOUCHY_MAX_STEPS + 1):
            pix_dv, pix_err = [], []
            for x, y, w, (grid, d0, d1) in zip(xs, ys, ws, tm):
                model = np.interp(x - shift, grid, d0, left=np.nan,
                                  right=np.nan)
                deriv = np.interp(x - shift, grid, d1, left=np.nan,
                                  right=np.nan)
                good = (np.isfinite(model) & np.isfinite(deriv) &
                        (deriv != 0) & (w > 0))
                # where T' is about zero (the continuum) a pixel says
                #   nothing: its d_i is infinite and its weight nil, and the
                #   mixture model leaves it out
                with np.errstate(over='ignore', divide='ignore',
                                 invalid='ignore'):
                    pix_dv.append(-(y[good] - model[good]) / deriv[good])
                    pix_err.append(1.0 / (np.sqrt(w[good]) *
                                          np.abs(deriv[good])))
            if not pix_dv or not sum(len(v) for v in pix_dv):
                break
            step, step_err = mp.odd_ratio_mean(np.concatenate(pix_dv),
                                               np.concatenate(pix_err))
            if not np.isfinite(step):
                break
            shift += step
            if errors is not None:
                errors[ifile] = C_MS * step_err
            if steps_taken is not None:
                steps_taken[ifile] = nstep
            limit = BOUCHY_TOLERANCE * step_err \
                if np.isfinite(step_err) else 0.0
            if abs(step) <= limit or abs(step) * C_MS < BOUCHY_FLOOR:
                converged = True
                break
        if not converged:
            unconverged.append(ifile)
        delta[ifile] = shift
    if unconverged:
        msg = ('Velocities of the spectra: {0} of {1} still moving after {2} '
               'Bouchy steps (spectra {3})')
        log.warning(msg.format(len(unconverged), nfiles, BOUCHY_MAX_STEPS,
                               ', '.join(str(i + 1) for i in unconverged)))
    # a feature displaced by +shift in ln(wavelength) is a velocity
    #   c tanh(shift) to add to the star's
    return C_MS * np.tanh(delta)


#: the keys of a spectrum's time in the table of spectra, in order of
#:   preference (LBL's names of the instrument's keywords first)
TIME_KEYS = ('KW_MID_EXP_TIME', 'KW_MJDATE', 'KW_BJD', 'MJDMID', 'MJDATE',
             'MJD-OBS', 'BJD', 'MJD')


def _times(sci_table: Dict[str, Any], nfiles: int):
    """Each spectrum's MJD from the table of spectra (a BJD brought back to
    an MJD), and what it is; (NaN, '') when the table has none"""
    for key in TIME_KEYS:
        values = sci_table.get(key)
        if values is None or len(values) != nfiles:
            continue
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).any():
            continue
        if np.nanmedian(values) > 2.4e6:
            values = values - 2400000.5
        return values, 'MJD ({0})'.format(key)
    return np.full(nfiles, np.nan), ''


def _log_velocities(sci_table: Dict[str, Any], rv0: np.ndarray,
                    rv_err: np.ndarray, times: np.ndarray,
                    time_label: str) -> None:
    """Each spectrum's velocity as the template measured it, in the log:
    its number, file, time, velocity less the median and uncertainty"""
    names = [os.path.basename(str(f)) for f in sci_table.get('FILENAME', [])]
    median = np.nanmedian(rv0)
    log.general('Velocities of the spectra (Bouchy 2001 pixel by pixel, '
                'mixture model), less their median {0:.1f} m/s:'.format(
                    median))
    for it in range(len(rv0)):
        name = names[it] if it < len(names) else ''
        when = ('MJD {0:.5f}'.format(times[it]) if np.isfinite(times[it])
                else '')
        log.general('\t{0:4d}/{1}  {2}  {3}  {4:+9.2f} +- {5:6.2f} m/s'
                    .format(it + 1, len(rv0), name, when, rv0[it] - median,
                            rv_err[it]))
    good = np.isfinite(rv0)
    log.general('Velocities of the spectra: rms {0:.2f} m/s, median '
                'uncertainty {1:.2f} m/s, {2} spectra'.format(
                    float(np.nanstd(rv0[good])),
                    float(np.nanmedian(rv_err)), int(good.sum())))


def _plot_velocities(inst, data: OrderData, fits: Dict[int, Any],
                     reg: List[int], berv: np.ndarray, rv0: np.ndarray,
                     rv_err: np.ndarray, lngrid: np.ndarray,
                     times: np.ndarray, time_label: str,
                     changes: List[float], history: List[np.ndarray] = None,
                     steps_per_pass: List[np.ndarray] = None,
                     within: List[float] = None, nsig: float = None,
                     need: float = None) -> None:
    """The velocities measured, pass by pass, and the spectra registered at
    them, on the middle order of those measured (lbl.science.plot, for
    PCA2D's and the window's eyes): never fatal"""
    try:
        from lbl.science import plot
        io_ = len(reg) // 2
        cloud_x, cloud_y, cloud_c = np.array([]), np.array([]), np.array([])
        grid_x, grid_y = np.array([]), np.array([])
        if fits.get(io_) is not None:
            # the pixels of every spectrum at its velocity, each with its
            #   spectrum's epoch (its MJD, else its number) for the colour
            shifts = star_shift(berv, rv0)
            x = (data.xobs[io_].astype(float) + data.ref[io_]
                 + shifts[:, None])
            flux = data.flux[io_].astype(float)
            epoch = times if np.isfinite(times).any() else \
                np.arange(1, len(rv0) + 1, dtype=float)
            colour = np.broadcast_to(np.asarray(epoch, dtype=float)[:, None],
                                     x.shape)
            good = np.isfinite(x) & np.isfinite(flux) & np.isfinite(colour)
            cloud_x, cloud_y, cloud_c = x[good], flux[good], colour[good]
            fit = fits[io_]
            grid_x = lngrid[fit['i0']:fit['i1']]
            grid_y = fit['derivs'][:, 0]
        # AN INJECTION TEST: the signals added to the BERV of these
        #   spectra (lbl.science.injection), drawn over the velocities
        #   against MJD at their zero point (less their median at the
        #   spectra's epochs)
        injected = None
        from lbl.science import injection
        found = injection.read(injection.spectrum_folder(inst))
        if found and np.isfinite(times).any():
            curve_t = np.linspace(np.nanmin(times), np.nanmax(times), 3000)
            curve_v = injection.velocity(curve_t + 2400000.5, found)
            at = injection.velocity(times[np.isfinite(times)] + 2400000.5,
                                    found)
            injected = (curve_t, curve_v - np.nanmedian(at),
                        'injected: ' + injection.describe(found))
        plot.template_velocity_plot(
            inst, rv0, rv_err, times, time_label, changes, cloud_x, cloud_y,
            grid_x, grid_y, inst.params['OBJECT_SCIENCE'], cloud_c=cloud_c,
            colour_label=(time_label or 'MJD') if np.isfinite(times).any()
            else 'spectrum', injected=injected, history=history,
            steps_per_pass=steps_per_pass, within=within, nsig=nsig,
            need=need)
    except Exception as exc:                                  # noqa: BLE001
        log.warning('the plot of the template\'s velocities was not made: '
                    '{0}'.format(exc))


def build(inst, science_files: List[str], load_science: Callable, blaze,
          calib_dir: str, wavegrid: np.ndarray, berv: np.ndarray,
          rv0: np.ndarray, iterate_rv: bool, tqdm, objname: str = None,
          data_dir: str = None, model_dir: str = None) -> Dict[str, Any]:
    """
    The template of a star from its spectra, by local polynomials through
    the clouds of their pixels (see the module's docstring)

    :param inst: the instrument
    :param science_files: list of str, the spectra
    :param load_science: callable, filename -> (image, header)
    :param blaze: np.ndarray or None, as lbl_template loads it
    :param calib_dir: str, the calibration directory
    :param wavegrid: np.ndarray, the template's grid (uniform in velocity)
    :param berv: np.ndarray, the BERV of each spectrum [m/s]
    :param rv0: np.ndarray, the star's velocity in each spectrum [m/s]
    :param iterate_rv: bool, measure the velocities of the spectra first
    :param tqdm: the progress bar
    :param objname: str or None, the object (its velocity box)
    :param data_dir: str or None, DATA_DIR (the velocity boxes)
    :param model_dir: str or None, the models (PHOENIX, for the first CCF)

    :return: dict(flux, eflux, flux_odd, eflux_odd, flux_even, eflux_even,
             savgol_fluxes: the template and its derivatives per step of the
             grid (flux_savgol_d0..3 and their odd and even versions, as
             compute_rv reads them, and eflux_savgol_d1, the uncertainty of
             flux_savgol_d1, for the mask), sci_table, rv0)
    """
    params = inst.params
    nfiles = len(science_files)
    rv0 = np.array(rv0, dtype=float)
    lngrid = np.log(wavegrid)
    dln = np.log(wavegrid[1] / wavegrid[0])
    # the window: TEMPLATE_BSD_FWHM resolution elements, 1 / R in
    #   ln(wavelength)
    if not params['APPROX_RESOLUTION']:
        emsg = ('TEMPLATE_METHOD = BSD needs the resolving power of the '
                'instrument (APPROX_RESOLUTION), or use TEMPLATE_METHOD = '
                'BERVBIN')
        raise base_classes.LblException(emsg)
    fit_params = dict(fwhm=(bsd_param(params, 'FWHM') /
                            params['APPROX_RESOLUTION']),
                      clip=bsd_param(params, 'CLIP'),
                      obs_clip=bsd_param(params, 'OBS_CLIP'))
    # no value across a hole of the pixels (TEMPLATE_BSD_MAX_GAP window
    #   FWHM), in ln(wavelength)
    fit_params['max_gap'] = (float(params['TEMPLATE_BSD_MAX_GAP'] or 0)
                             * fit_params['fwhm'])
    #   nor at the edge of one: the least share of the window's weight on
    #   each side (TEMPLATE_BSD_MIN_BALANCE)
    fit_params['min_balance'] = float(params['TEMPLATE_BSD_MIN_BALANCE'] or 0)
    niter = int(bsd_param(params, 'NITER'))
    # the geometry, the reference wavelength of each order and the low pass
    #   width of each order in pixels (HP_WIDTH)
    image0, hdr0 = load_science(science_files[-1])
    wave0 = inst.get_wave_solution(science_files[-1], image0, hdr0)
    nord, npix = wave0.shape
    with np.errstate(invalid='ignore', divide='ignore'):
        pix_ms = np.nanmedian(np.gradient(wave0, axis=1) / wave0, axis=1) * C_MS
    hp_pix = np.full(nord, 101, dtype=int)
    good = np.isfinite(pix_ms) & (pix_ms > 0)
    hp_pix[good] = np.clip(params['HP_WIDTH'] * 1000 / pix_ms[good], 11,
                           max(11, npix // 4)).astype(int)
    # the orders read at once: what fits in TEMPLATE_BSD_MEMORY [GB]
    per_order = nfiles * npix * 3 * 4
    chunk = max(1, int(bsd_param(params, 'MEMORY') * 1e9 // per_order))
    msg = ('BSD template (BERV Spectral Drizzle) from the pixels of {0} '
           'spectra: {1} orders, '
           'window FWHM {2:.2f} km/s, {3} passes, {4} orders read at a time')
    log.info(msg.format(nfiles, nord, fit_params['fwhm'] * C_MS / 1e3, niter,
                        chunk))
    sci_table = dict()
    # -------------------------------------------------------------------------
    # the velocities of the spectra, on the orders around COMPIL_SLOPE_REF_WAVE
    if iterate_rv:
        ref_wave = params['COMPIL_SLOPE_REF_WAVE']
        medwave = np.nanmedian(wave0, axis=1)
        reg = [o for o in range(nord) if np.isfinite(medwave[o]) and
               abs(medwave[o] / ref_wave - 1) < 0.05]
        if not reg:
            reg = list(np.argsort(np.abs(medwave - ref_wave))[:3])
        data = read_orders(inst, science_files, load_science, blaze,
                           calib_dir, reg, hp_pix, tqdm, sci_table, berv,
                           desc='Reading spectra (velocities)')
        changes, within = [], []
        rv_err = np.full(nfiles, np.nan)
        # the start: the velocities of the object's box (a CCF against the
        #   PHOENIX model the first time), so that a star that moves by km/s
        #   is not smeared in the first template
        rv0, sysvel = start_velocities(inst, science_files, load_science,
                                       blaze, calib_dir, berv, rv0, objname,
                                       data_dir, model_dir, tqdm)
        nsig = float(params['TEMPLATE_RV_CONV_NSIG'])
        need = float(params['TEMPLATE_RV_CONV_FRAC'])
        npass = int(params['TEMPLATE_RV_MAX_PASSES'])
        # each pass's velocities (the start first) and the Bouchy steps
        #   each spectrum took, for the plot
        history = [np.array(rv0, dtype=float)]
        steps_per_pass = []
        converged = False
        for ite in range(npass):
            shifts = star_shift(berv, rv0)
            fits = {io_: fit_order(data, io_, shifts, lngrid, fit_params,
                                   hp_pix[order], 2)
                    for io_, order in enumerate(reg)}
            taken = np.zeros(nfiles, dtype=int)
            dv = measure_velocities(data, fits, shifts, lngrid,
                                    scan=(ite == 0), scan_step=dln,
                                    errors=rv_err, steps_taken=taken)
            rv0 = mp.add_velocities(rv0, dv)
            history.append(np.array(rv0, dtype=float))
            steps_per_pass.append(taken)
            n1, p1 = np.nanpercentile(dv, [16, 84])
            sig = (p1 - n1) / 2.0
            changes.append(float(sig))
            # converged when TEMPLATE_RV_CONV_FRAC of the spectra changed by
            #   less than TEMPLATE_RV_CONV_NSIG times their uncertainty
            ok = np.isfinite(dv) & np.isfinite(rv_err) & (rv_err > 0)
            frac = float(np.mean(np.abs(dv[ok]) < nsig * rv_err[ok])) \
                if ok.any() else 0.0
            within.append(frac)
            msg = ('Velocities of the spectra, pass {0}: change RMS {1:.1f} '
                   'm/s, {2:.1f} % of them within {3:g} sigma')
            log.general(msg.format(ite + 1, sig, 100 * frac, nsig))
            if frac >= need:
                converged = True
                break
        if not converged:
            log.warning('Velocities of the spectra: {0:.1f} % within {1:g} '
                        'sigma after {2} passes, not the {3:.0f} % asked '
                        '(TEMPLATE_RV_CONV_FRAC)'.format(
                            100 * within[-1] if within else 0.0, nsig, npass,
                            100 * need))
        times, time_label = _times(sci_table, nfiles)
        _log_velocities(sci_table, rv0, rv_err, times, time_label)
        sci_table['VSYS_ERR'] = np.array(rv_err)
        # the velocities measured, into the object's box: the template's
        #   frame plus the box's systemic velocity
        if data_dir and objname and np.isfinite(sysvel):
            from lbl.science import velocity_box
            entries = {science_files[it]: dict(
                v=sysvel + rv0[it], err=rv_err[it], mjd=times[it],
                berv=berv[it]) for it in range(nfiles)}
            velocity_box.update(data_dir, objname, entries, 'template')
        _plot_velocities(inst, data, fits, reg, berv, rv0, rv_err, lngrid,
                         times, time_label, changes, history,
                         steps_per_pass, within, nsig, need)
        del data
    shifts = star_shift(berv, rv0)
    # -------------------------------------------------------------------------
    # the template, order by order, combined on the grid: all orders, the odd
    #   ones and the even ones; sums of weight * value and of weight
    ngrid = len(wavegrid)
    sums = {key: np.zeros((ngrid, NDERIV + 1)) for key in ('', 'odd', 'even')}
    wsum = {key: np.zeros(ngrid) for key in ('', 'odd', 'even')}
    vsum = {key: np.zeros((ngrid, NDERIV + 1)) for key in ('', 'odd', 'even')}
    for start in range(0, nord, chunk):
        orders = list(range(start, min(nord, start + chunk)))
        desc = 'Reading spectra, orders {0} to {1} of {2}'.format(
            orders[0], orders[-1], nord - 1)
        data = read_orders(inst, science_files, load_science, blaze,
                           calib_dir, orders, hp_pix, tqdm,
                           sci_table if not sci_table else None, berv,
                           desc=desc)
        for io_, order in enumerate(tqdm(orders, desc='Fitting orders',
                                         unit='order')):
            fit = fit_order(data, io_, shifts, lngrid, fit_params,
                            hp_pix[order], niter)
            if fit is None:
                continue
            i0, i1 = fit['i0'], fit['i1']
            # the order's weight on its piece of the grid
            xm = data.xmean[io_] + np.nanmedian(shifts)
            good = np.isfinite(xm) & (data.count[io_] > 0)
            if np.sum(good) < 10:
                continue
            srt = np.argsort(xm[good])
            weight = np.interp(lngrid[i0:i1], xm[good][srt],
                               data.weight[io_][good][srt], left=0, right=0)
            # the order back to its level in the spectra (each spectrum's
            #   order was divided by its own median), so that the orders
            #   agree where they overlap
            scale = np.nanmedian(data.level[io_])
            if not np.isfinite(scale) or scale <= 0:
                continue
            derivs, errs = fit['derivs'] * scale, fit['errs'] * scale
            ok = np.all(np.isfinite(derivs), axis=1) & \
                np.all(np.isfinite(errs), axis=1) & (weight > 0)
            weight = np.where(ok, weight, 0.0)
            derivs = np.where(ok[:, None], derivs, 0.0)
            errs = np.where(ok[:, None], errs, 0.0)
            # the derivatives per step of the grid, as compute_rv reads them
            per_step = dln ** np.arange(NDERIV + 1)
            derivs, errs = derivs * per_step, errs * per_step
            for key in ('', 'odd' if order % 2 else 'even'):
                sums[key][i0:i1] += weight[:, None] * derivs
                wsum[key][i0:i1] += weight
                vsum[key][i0:i1] += (weight[:, None] * errs) ** 2
        del data
    # -------------------------------------------------------------------------
    out = dict(sci_table=sci_table, rv0=rv0, savgol_fluxes=dict())
    min_weight = params['TEMPLATE_WEIGHT_MIN']
    for key in ('', 'odd', 'even'):
        name = 'flux_' + key if key else 'flux'
        with np.errstate(invalid='ignore', divide='ignore'):
            valid = wsum[key] >= min_weight
            values = sums[key] / wsum[key][:, None]
            values[~valid] = np.nan
            errors = np.sqrt(vsum[key]) / wsum[key][:, None]
            errors[~valid] = np.nan
        out[name] = values[:, 0]
        out['e' + name] = errors[:, 0]
        for k in range(NDERIV + 1):
            out['savgol_fluxes']['{0}_savgol_d{1}'.format(name, k)] = \
                values[:, k]
        # the uncertainty of the first derivative, for the mask's lines
        if not key:
            out['savgol_fluxes']['eflux_savgol_d1'] = errors[:, 1]
    return out


# =============================================================================
# End of code
# =============================================================================

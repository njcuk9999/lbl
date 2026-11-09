#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The template without resampling a spectrum (TEMPLATE_METHOD = 'SCATTER').

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

Per order, in TEMPLATE_SCATTER_NITER passes:
    1. the fit through the cloud, each point weighted by the inverse square
       of its noise (a low pass of its residuals), with a clip of the local
       outliers at TEMPLATE_SCATTER_CLIP times the RMS of the residuals of
       their window (in units of their noise), in a
       Gaussian window of TEMPLATE_SCATTER_FWHM resolution elements;
    2. a pixel of the detector whose residual is off in most spectra (the
       median over spectra of the residual in units of the noise beyond
       TEMPLATE_SCATTER_OBS_CLIP) is left out of all spectra: what stays on
       the detector while the star moves (telluric residuals, bad pixels);
    3. each spectrum is divided by a low pass (HP_WIDTH) of its ratio to the
       template, for what the blaze division leaves.
The orders are then combined on the grid with the weights the s1d uses
(the blaze and the smooth fall at the order edges): all orders, the odd
ones and the even ones.

With SMALL_RV_SHIFT False and no known orbits, the velocities of the spectra
are first measured on the orders around COMPIL_SLOPE_REF_WAVE (a scan of the
shift, then its projection on the derivative) until they agree to
MAX_CONVERGENCE_TEMPLATE_RV.

The window is about one resolution element, not the 3 steps of the grid of
templatepy: in a narrow window each spectrum's own noise is copied sharply
into the template, at that spectrum's pixels, which pulls the velocity of the
spectrum towards the template's and shrinks all velocities (by 16 % on 40
NIRPS spectra of GJ 1028 at 0.4 resolution element, against 2 % at one
resolution element and with BERVBIN).

After templatepy (github.com/eartigau/templatepy, E. Artigau).

@author: artigau
"""
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
                                      clip_iter=2, presorted=True)
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


def measure_velocities(data: OrderData, fits: Dict[int, Dict[str, Any]],
                       shifts: np.ndarray, lngrid: np.ndarray,
                       scan: bool, scan_step: float) -> np.ndarray:
    """
    The velocity of each spectrum against the templates of some orders: the
    least-squares shift of the template that matches the spectrum (a scan of
    +-50 steps first when scan is True, then its projection on the
    derivative)

    :param data: OrderData of the orders
    :param fits: dict, order index in data -> fit_order output
    :param shifts: np.ndarray, the current shift of each spectrum
    :param lngrid: np.ndarray, ln of the template's grid
    :param scan: bool, scan the shift first
    :param scan_step: float, the step of the scan [ln wavelength]

    :return: np.ndarray, the velocity to add to each spectrum's [m/s]
    """
    nfiles = len(shifts)
    delta = np.zeros(nfiles)
    steps = np.arange(-50, 51) * scan_step
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
        # the projection on the derivative, twice: y(x) = T(x - shift)
        for _ in range(2):
            num, den = 0.0, 0.0
            for x, y, w, (grid, d0, d1) in zip(xs, ys, ws, tm):
                model = np.interp(x - shift, grid, d0, left=np.nan,
                                  right=np.nan)
                deriv = np.interp(x - shift, grid, d1, left=np.nan,
                                  right=np.nan)
                good = np.isfinite(model) & np.isfinite(deriv)
                num += np.sum(w[good] * (y[good] - model[good]) * deriv[good])
                den += np.sum(w[good] * deriv[good] ** 2)
            if den > 0:
                shift -= num / den
        delta[ifile] = shift
    # a feature displaced by +shift in ln(wavelength) is a velocity
    #   c tanh(shift) to add to the star's
    return C_MS * np.tanh(delta)


def build(inst, science_files: List[str], load_science: Callable, blaze,
          calib_dir: str, wavegrid: np.ndarray, berv: np.ndarray,
          rv0: np.ndarray, iterate_rv: bool, tqdm) -> Dict[str, Any]:
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
    # the window: TEMPLATE_SCATTER_FWHM resolution elements, 1 / R in
    #   ln(wavelength)
    if not params['APPROX_RESOLUTION']:
        emsg = ('TEMPLATE_METHOD = SCATTER needs the resolving power of the '
                'instrument (APPROX_RESOLUTION), or use TEMPLATE_METHOD = '
                'BERVBIN')
        raise base_classes.LblException(emsg)
    fit_params = dict(fwhm=(params['TEMPLATE_SCATTER_FWHM'] /
                            params['APPROX_RESOLUTION']),
                      clip=params['TEMPLATE_SCATTER_CLIP'],
                      obs_clip=params['TEMPLATE_SCATTER_OBS_CLIP'])
    niter = int(params['TEMPLATE_SCATTER_NITER'])
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
    # the orders read at once: what fits in TEMPLATE_SCATTER_MEMORY [GB]
    per_order = nfiles * npix * 3 * 4
    chunk = max(1, int(params['TEMPLATE_SCATTER_MEMORY'] * 1e9 // per_order))
    msg = ('Template from the scattered pixels of {0} spectra: {1} orders, '
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
        for ite in range(5):
            shifts = star_shift(berv, rv0)
            fits = {io_: fit_order(data, io_, shifts, lngrid, fit_params,
                                   hp_pix[order], 2)
                    for io_, order in enumerate(reg)}
            dv = measure_velocities(data, fits, shifts, lngrid,
                                    scan=(ite == 0), scan_step=dln)
            rv0 = mp.add_velocities(rv0, dv)
            n1, p1 = np.nanpercentile(dv, [16, 84])
            sig = (p1 - n1) / 2.0
            msg = 'Velocities of the spectra, pass {0}: change RMS {1:.1f} m/s'
            log.general(msg.format(ite + 1, sig))
            if sig < params['MAX_CONVERGENCE_TEMPLATE_RV']:
                break
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

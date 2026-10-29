#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An APERO template as an LBL template: another star's spectrum for LBL and
PCA2D (OBJECT_COMPARISON another star) without its spectra.

APERO gives two templates of a star:

    Template_s1dv_<OBJ>_sc1d_v_file_AB.fits   the s1d on a constant velocity
                                              grid, all orders together
                                              (wavelength, flux, eflux, rms)
    Template_<OBJ>_tellu_obj_AB.fits          the e2ds, order by order on
                                              APERO's reference wavelength
                                              solution (Chebyshev polynomial
                                              of the pixel on [-1, 1], its
                                              coefficients WAVE0000... in the
                                              header), blaze corrected

LBL's template is the s1d plus the same spectrum from the even orders alone
and from the odd ones alone (flux_even, flux_odd: the two parities see the
lines at different depths where the orders overlap), and the Savitzky-Golay
derivatives of all three (USE_SAVGOL_TEMPLATE). Here the s1d gives the grid
and flux; the e2ds, when there is one, gives flux_even and flux_odd, each
order put on the s1d grid and scaled to the s1d's level over it. The e2ds is
moved first onto the s1d's frame: its reference wavelength solution is not
the s1d's (GL 514: 1.1 km/s apart), the offset measured order by order and
its median applied. Without an e2ds, flux_even and flux_odd are the s1d.

Created on 2026-10-22

@author: artigau
"""
import os
from typing import Optional

import numpy as np
from astropy import constants
from astropy.io import fits
from astropy.table import Table
from numpy.polynomial import chebyshev

from lbl.core import io
from lbl.core import math as mp

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.science.template_import.py'
log = io.log
#: the speed of light [km/s]
C_KMS = constants.c.to('km/s').value
#: the velocities searched for the e2ds's offset from the s1d [km/s]
OFFSET_RANGE = np.arange(-5.0, 5.001, 0.02)


# =============================================================================
# Define functions
# =============================================================================
def e2ds_wavelengths(header: fits.Header, npix: int) -> np.ndarray:
    """
    APERO's wavelength solution of an e2ds file, order by order: a Chebyshev
    polynomial of the pixel mapped onto [-1, 1] (WAVEORDN orders, WAVEDEGN
    degree, coefficients WAVE0000...)

    :return: np.ndarray (orders, npix) [nm]
    """
    nord, deg = int(header['WAVEORDN']), int(header['WAVEDEGN'])
    coeffs = np.array([header['WAVE%04d' % i]
                       for i in range(nord * (deg + 1))]).reshape(nord, deg + 1)
    xnorm = 2 * np.arange(npix) / (npix - 1) - 1
    return np.array([chebyshev.chebval(xnorm, coeffs[o]) for o in range(nord)])


def _order_offset(wave: np.ndarray, flux: np.ndarray, s1d_wave: np.ndarray,
                  s1d_flux: np.ndarray) -> float:
    """The velocity [km/s] that best brings one e2ds order onto the s1d, by
    correlation, or NaN"""
    good = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    if np.sum(good) < 500:
        return np.nan
    wave, flux = wave[good], flux[good] / np.nanmedian(flux[good])
    corr = np.full(len(OFFSET_RANGE), np.nan)
    for it, dv in enumerate(OFFSET_RANGE):
        model = np.interp(wave * (1 + dv / C_KMS), s1d_wave, s1d_flux,
                          left=np.nan, right=np.nan)
        ok = np.isfinite(model)
        if np.sum(ok) > 500:
            corr[it] = np.corrcoef(flux[ok], model[ok])[0, 1]
    if not np.isfinite(corr).any() or np.nanmax(corr) < 0.5:
        return np.nan
    return float(OFFSET_RANGE[int(np.nanargmax(corr))])


def apero_to_lbl(s1d_file: str, out_file: str, approx_resolution: float,
                 e2ds_file: Optional[str] = None) -> str:
    """
    Write an LBL template from APERO's s1d template, and its e2ds one for
    the even and odd orders (see the module's docstring)

    :param s1d_file: str, Template_s1dv_<OBJ>_sc1d_v_file_AB.fits
    :param out_file: str, the LBL template to write
                     (templates/LBL_Template_<OBJ>_<instrument>.fits)
    :param approx_resolution: float, the instrument's resolution (LBL's
                              APPROX_RESOLUTION), for the derivatives' window
    :param e2ds_file: str or None, Template_<OBJ>_tellu_obj_AB.fits

    :return: str, out_file
    """
    s1d = Table.read(s1d_file, hdu=1)
    wave = np.asarray(s1d['wavelength'], dtype=float)
    flux = np.asarray(s1d['flux'], dtype=float)
    eflux = np.asarray(s1d['eflux'], dtype=float)
    rms = np.asarray(s1d['rms'], dtype=float)
    header = fits.getheader(s1d_file)
    parities = dict()
    note = 'no e2ds template: flux_even and flux_odd are the s1d'
    if e2ds_file is not None and os.path.exists(e2ds_file):
        with fits.open(e2ds_file) as hdul:
            e2ds = np.asarray(hdul[1].data, dtype=float)
            ehead = hdul[0].header
        ewave = e2ds_wavelengths(ehead, e2ds.shape[1])
        good = np.isfinite(flux) & (flux > 0)
        offsets = np.array([_order_offset(ewave[o], e2ds[o], wave[good],
                                          flux[good])
                            for o in range(e2ds.shape[0])])
        offset = float(np.nanmedian(offsets))
        ewave = ewave * (1 + offset / C_KMS)
        log.general('APERO e2ds template on the s1d frame: moved by {0:+.2f} '
                    'km/s (median of {1} orders, rms {2:.2f})'.format(
                        offset, int(np.isfinite(offsets).sum()),
                        float(np.nanstd(offsets))))
        for parity, name in ((0, 'even'), (1, 'odd')):
            out = np.full_like(flux, np.nan)
            for order in range(parity, e2ds.shape[0], 2):
                ok = np.isfinite(ewave[order]) & np.isfinite(e2ds[order])
                if np.sum(ok) < 100:
                    continue
                lo, hi = np.min(ewave[order][ok]), np.max(ewave[order][ok])
                inside = (wave >= lo) & (wave <= hi)
                values = np.interp(wave[inside], ewave[order][ok],
                                   e2ds[order][ok])
                # the s1d's level over this order
                ref = flux[inside]
                both = np.isfinite(ref) & np.isfinite(values) & (values > 0)
                if np.sum(both) < 100:
                    continue
                scale = np.nanmedian(ref[both] / values[both])
                out[inside] = values * scale
            parities[name] = out
        note = ('flux_even and flux_odd from the e2ds template, moved by '
                '{0:+.2f} km/s onto the s1d'.format(offset))
    else:
        parities = dict(even=flux.copy(), odd=flux.copy())
    table = Table()
    table['wavelength'] = wave
    table['flux'] = flux
    table['eflux'] = eflux
    table['rms'] = rms
    for name in ('odd', 'even'):
        table['flux_' + name] = parities[name]
        table['eflux_' + name] = eflux
        table['rms_' + name] = rms
    # the derivatives, as LBL's own templates have them
    #   (Instrument.calculate_savgol_template): a window of one resolution
    #   element on the template's velocity grid
    dv_grid = np.nanmedian(np.diff(np.log(wave))) * mp.speed_of_light_ms
    window = int(np.round((mp.speed_of_light_ms / dv_grid) /
                          float(approx_resolution)))
    window = window if window % 2 == 1 else window + 1
    for key in ('flux', 'flux_odd', 'flux_even'):
        for deriv in range(4):
            table['{0}_savgol_d{1}'.format(key, deriv)] = \
                mp.gaussian_weighted_savgol(np.asarray(table[key]), window,
                                            polyorder=3, deriv=deriv,
                                            delta=1.0)
    primary = fits.Header()
    for key in ('OBJECT', 'OBJNAME', 'DRSOBJN', 'PP_OBJN', 'PP_TEFF',
                'PP_RV', 'VERSION', 'INSTRUME'):
        if key in header:
            primary[key] = (header[key], header.comments[key])
    # the number of spectra the template was made of (LBL's KW_NTFILES,
    #   LBLNTMPL): LBL's mask takes the template's noise from it
    try:
        nfiles = len(fits.getdata(s1d_file, 'TEMPLATE_TABLE'))
    except Exception:                                     # noqa: BLE001
        nfiles = int(header.get('FTTTEMPN', 1) or 1)
    primary['LBLNTMPL'] = (int(nfiles), 'spectra in the APERO template')
    primary['LBL_OBJC'] = (str(header.get('DRSOBJN', '')), 'template object')
    primary['TPLSRC'] = (os.path.basename(s1d_file), 'APERO s1d template')
    if e2ds_file is not None:
        primary['TPLSRCE'] = (os.path.basename(e2ds_file),
                              'APERO e2ds template')
    primary['TPLNOTE'] = note[:68]
    hdus = fits.HDUList([fits.PrimaryHDU(header=primary),
                         fits.BinTableHDU(table, name='TEMPLATE')])
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    hdus.writeto(out_file, overwrite=True)
    log.general('LBL template from APERO\'s: {0} ({1})'.format(out_file,
                                                                note))
    return out_file

# =============================================================================
# End of code
# =============================================================================

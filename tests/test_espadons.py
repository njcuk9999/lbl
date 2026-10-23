#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The ESPaDOnS class's reader (lbl.instruments.espadons) on a file built like
the i.fits CADC distributes: all orders in one table, air wavelengths moved
by the heliocentric velocity of the comments, the intensity divided by the
blaze.

Created on 2026-10-23

@author: artigau
"""
import os

import numpy as np
from astropy.io import fits

from lbl.core import math as mp
from lbl.instruments import espadons as es

C_KMS = 299792.458
HELIO = 25.0


def _order(m, npts, rng):
    """One order: air wavelengths, as observed, and the blaze across it"""
    centre = es.M_LAMBDA / m
    lnw = np.log(centre) + (np.arange(npts) - npts / 2) * es.DLNL
    xx = np.linspace(-1, 1, npts)
    blaze = 1.0 - 0.6 * xx ** 2
    return np.exp(lnw), blaze


def _file(tmp_path, orders=(61, 60, 23, 22), npts=3000, seed=1):
    """A CADC-like i.fits with a few orders: two that overlap in the blue,
    two with a gap between them in the red"""
    rng = np.random.default_rng(seed)
    waves, fluxes, errors, blazes = [], [], [], []
    for m in orders:
        wave_obs, blaze = _order(m, npts, rng)
        photons = 4000.0 * blaze
        counts = rng.poisson(photons).astype(float)
        # delivered: divided by the blaze, error the photon noise
        fluxes.append(counts / blaze / 4000.0)
        errors.append(np.sqrt(counts) / blaze / 4000.0)
        # air wavelengths moved to the heliocentric frame by (1 + v/c)
        waves.append(wave_obs / mp.air_index(wave_obs) * (1 + HELIO / C_KMS))
        blazes.append(blaze)
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    error = np.concatenate(errors)
    data = np.array([wave, flux, error, wave, flux, error])
    header = fits.Header()
    header['REDUCTIO'] = 'Intensity'
    for num, (name, comment) in enumerate(
            [('Wavelength', 'Normalized'), ('Intensity', 'Normalized'),
             ('ErrorBar', 'Normalized'), ('Wavelength', 'UnNormalized'),
             ('Intensity', 'UnNormalized'), ('ErrorBar', 'UnNormalized')]):
        header['COL{0}'.format(num + 1)] = (name, comment)
    header['COMMENT'] = ('Heliocentric velocity of observer towards star :'
                         ' {0:.3f} km/s'.format(HELIO))
    header['COMMENT'] = 'SNR per spec/ccd pxl in order # 31 (731 nm): I> 123 / 147'
    path = os.path.join(str(tmp_path), '1234567i.fits')
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path)
    return path, orders, blazes


def test_the_orders_are_found_and_put_back_in_the_observer_frame(tmp_path):
    path, orders, blazes = _file(tmp_path)
    wave, flux, blaze = es.read_espadons(path)
    assert wave.shape == (len(es.ORDERS), es.NPIX)
    # every row has a finite, increasing wave map, filled or not
    assert np.all(np.diff(wave, axis=1) > 0)
    filled = np.isfinite(flux).any(axis=1)
    assert sorted(es.ORDERS[filled]) == sorted(orders)
    for m in orders:
        row = int(np.where(es.ORDERS == m)[0][0])
        ok = np.isfinite(flux[row])
        expected, _ = _order(m, ok.sum(), None)
        # vacuum, observer's frame: back to what was observed, to float32
        assert np.allclose(wave[row][ok], expected, rtol=2e-7)


def test_the_blaze_comes_back_from_the_error_bars(tmp_path):
    path, orders, blazes = _file(tmp_path)
    _, flux, blaze = es.read_espadons(path)
    for m, truth in zip(orders, blazes):
        row = int(np.where(es.ORDERS == m)[0][0])
        ok = np.isfinite(flux[row])
        assert np.allclose(blaze[row][ok], truth / truth.max(), atol=0.03)


def test_the_header_gets_the_snr_and_the_heliocentric_velocity(tmp_path):
    path, _, _ = _file(tmp_path)
    header = fits.getheader(path)
    assert es.heliocentric_velocity(header) == HELIO
    assert es.order_snr(header) == 123
    assert np.isnan(es.order_snr(header, order=30))


def test_the_tellurics_give_the_zero_point_back():
    """A spectrum whose telluric lines are 300 m/s off: the correlation with
    the transmission finds them there, band by band"""
    rng = np.random.default_rng(3)
    grid = np.exp(np.arange(np.log(680.0), np.log(975.0), 2e-6))
    lines = np.ones_like(grid)
    for centre in rng.uniform(680.0, 975.0, 3000):
        lines *= 1 - 0.5 * np.exp(-0.5 * ((grid - centre) / 0.004) ** 2)
    model = dict(wavelength=grid, ABSO_OTHERS=lines, ABSO_WATER=lines)
    wave = np.exp(np.arange(np.log(680.0), np.log(975.0), es.DLNL))
    shift = 0.3
    # the spectrum's wavelengths are short by the Doppler factor of 300 m/s
    flux = np.interp(wave * np.exp(np.arctanh(shift / C_KMS)), grid, lines)
    flux = flux + rng.normal(0, 0.005, flux.size)
    found, bands = es.telluric_velocity(wave, flux, np.ones_like(flux), model)
    assert abs(found - shift) < 0.01
    assert all(abs(val - shift) < 0.03 for val in bands)

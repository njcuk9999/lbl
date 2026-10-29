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
    """A spectrum whose telluric lines are 300 m/s off, a star in front of
    them: with the star's template divided out, the least squares find the
    velocity and the depths of the two absorbers"""
    rng = np.random.default_rng(3)
    lnw = np.arange(np.log(600.0), np.log(1000.0), 2e-6)
    def lines(number, depth, width):
        out = np.zeros_like(lnw)
        for centre in rng.uniform(lnw[0], lnw[-1], number):
            out += np.log(1 - depth * np.exp(-0.5 * ((lnw - centre)
                                                    / width) ** 2))
        return out
    others, water = lines(800, 0.5, 6e-6), lines(1500, 0.3, 6e-6)
    model = dict(lnw=lnw, others=others, water=water,
                 dothers=np.gradient(others, lnw),
                 dwater=np.gradient(water, lnw))
    star = lines(4000, 0.4, 1e-5)
    berv, shift, depths = 12.0, 0.3, (1.3, 0.7)
    # 3 orders of the observed spectrum, its wavelengths short by D(300 m/s)
    wave = np.exp(np.linspace(np.log(610.0), np.log(990.0), 3 * 60000)
                  ).reshape(3, 60000)
    true = np.log(wave) + np.arctanh(shift / C_KMS)
    lnflux = (np.interp(true + np.arctanh(berv / C_KMS), lnw, star)
              + depths[0] * np.interp(true, lnw, others)
              + depths[1] * np.interp(true, lnw, water))
    flux = 1e4 * np.exp(lnflux) * (1 + rng.normal(0, 0.01, wave.shape))
    fit = es.fit_tellurics(wave, flux, np.ones_like(flux), model,
                           template=(lnw, star), berv=berv)
    assert abs(fit['dv'] - shift) < 3 * fit['sdv'] + 0.002
    assert fit['sdv'] < 0.01
    assert np.allclose(fit['depths'], depths, atol=0.02)
    assert fit['history'][0] == 0.0

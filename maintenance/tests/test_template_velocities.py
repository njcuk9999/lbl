#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The velocities of the spectra when the template iterates on them
(SMALL_RV_SHIFT False, lbl.science.template_scatter.measure_velocities):
Bouchy et al. 2001 pixel by pixel on the template's derivative, combined by
LBL's mixture model, so that bad pixels do not pull a spectrum's velocity;
and these velocities reaching PCA2D's star frame whatever the comparison.

Created on 2026-11-18

@author: artigau
"""
import numpy as np

from lbl.core.math import speed_of_light_ms
from lbl.science import template_scatter as ts


def _template(x):
    """a spectrum of narrow lines in ln(wavelength), and its derivative"""
    centres = np.linspace(-0.0045, 0.0045, 60)
    width = 6e-6
    flux, deriv = np.ones_like(x), np.zeros_like(x)
    for k, c in enumerate(centres):
        depth = 0.3 + 0.4 * ((k * 7) % 5) / 5.0
        g = depth * np.exp(-0.5 * ((x - c) / width) ** 2)
        flux -= g
        deriv += g * (x - c) / width ** 2
    return flux, deriv


def _data(true_shifts, noise, bad=0, seed=0):
    rng = np.random.default_rng(seed)
    npix = 6000
    data = ts.OrderData([0], len(true_shifts), npix)
    grid = np.linspace(-0.005, 0.005, 40001)
    d0, d1 = _template(grid)
    for ifile, shift in enumerate(true_shifts):
        x = np.sort(rng.uniform(-0.0048, 0.0048, npix))
        flux, _ = _template(x - shift)
        flux = flux + rng.normal(0.0, noise, npix)
        if bad:
            # cosmic rays and dead pixels: far off, few
            hit = rng.choice(npix, bad, replace=False)
            flux[hit] += rng.choice([-1.0, 1.0], bad) * rng.uniform(0.3, 1.0,
                                                                    bad)
        data.xobs[0, ifile] = x
        data.flux[0, ifile] = flux
        data.noise[0, ifile] = noise
    fits = {0: dict(derivs=np.column_stack([d0, d1]), i0=0, i1=grid.size)}
    return data, fits, grid


def test_the_velocities_are_found():
    true = np.array([0.0, 3e-6, -5e-6, 1.2e-5])
    data, fits, grid = _data(true, noise=0.005)
    dv = ts.measure_velocities(data, fits, np.zeros(true.size), grid,
                               scan=True, scan_step=2.5e-7)
    expected = speed_of_light_ms * np.tanh(true)
    assert np.all(np.abs(dv - expected) < 15.0)


def test_bad_pixels_do_not_pull_them():
    """bad pixels on the lines' flanks (1 % of them, 15 to 170 sigma off),
    after the scan (shifts below a line width): the mixture model keeps the
    velocities at the photon noise's precision, where least squares (the
    method before, the same estimator without the mixture) is pulled by
    tens of m/s (0.8 against 22 m/s rms over 40 draws, 2026-11-18)"""
    rng = np.random.default_rng(42)
    errors, pulled = [], []
    for seed in range(12):
        true = rng.uniform(-1e-6, 1e-6, 4)
        data, fits, grid = _data(true, noise=0.003, seed=seed)
        bad = np.random.default_rng(1000 + seed)
        d1 = fits[0]['derivs'][:, 1]
        for ifile in range(true.size):
            x = data.xobs[0, ifile].astype(float)
            slope = np.abs(np.interp(x, grid, d1))
            flank = np.where(slope > np.percentile(slope, 90))[0]
            hit = bad.choice(flank, 60, replace=False)
            data.flux[0, ifile, hit] += bad.choice([-1.0, 1.0], 60) * \
                bad.uniform(0.05, 0.5, 60)
        dv = ts.measure_velocities(data, fits, np.zeros(true.size), grid,
                                   scan=False, scan_step=2.5e-7)
        expected = speed_of_light_ms * np.tanh(true)
        errors.append(dv - expected)
        # least squares on the same pixels, for comparison
        for ifile in range(true.size):
            x = data.xobs[0, ifile].astype(float)
            y = data.flux[0, ifile].astype(float)
            shift = 0.0
            for _ in range(2):
                model = np.interp(x - shift, grid, fits[0]['derivs'][:, 0])
                deriv = np.interp(x - shift, grid, d1)
                shift -= np.sum((y - model) * deriv) / np.sum(deriv ** 2)
            pulled.append(speed_of_light_ms * np.tanh(shift) -
                          expected[ifile])
    assert np.std(errors) < 2.0
    assert np.std(pulled) > 5 * np.std(errors)


def test_pca2d_takes_the_own_template_s_velocities(tmp_path, monkeypatch):
    """whatever template the velocities are measured against, PCA2D's star
    frame follows those the object's own template fitted"""
    from astropy.table import Table
    from lbl.recipes import lbl_pca2d, lbl_wrap
    path = tmp_path / 'LBL_Template_STAR_x.fits'
    table = Table(dict(FILENAME=['a.fits', 'b.fits', 'c.fits'],
                       VSYS=[10.0, 30.0, 20.0]))
    from astropy.io import fits
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table,
                                                      name='SCI_TABLE')]
                 ).writeto(path)
    monkeypatch.setattr(lbl_wrap, '_template_of',
                        lambda data_dir, name, instrument: str(path))
    said = []
    found = lbl_pca2d.object_velocities('STAR', 'VETTED_GL699',
                                        str(tmp_path), 'NIRPS_HE',
                                        lambda *a: said.append(a))
    assert found == {'a.fits': -0.01, 'b.fits': 0.01, 'c.fits': 0.0}


def test_the_wrap_makes_the_velocity_template_first(tmp_path, monkeypatch):
    """SMALL_RV_SHIFT False: the object's own template is made (or made
    again when it was made flat) before PCA2D and before anything is
    measured against it; SMALL_RV_SHIFT True: nothing"""
    from lbl.recipes import lbl_wrap
    calls = []
    monkeypatch.setattr(lbl_wrap.lbl_template, 'main',
                        lambda **kw: calls.append(kw))
    monkeypatch.setattr(lbl_wrap, '_template_of', lambda *a: None)
    args = dict(keyword_args=dict(SMALL_RV_SHIFT=False),
                instrument='NIRPS_HE', data_dir=str(tmp_path),
                data_source='CADC', objname='STAR', blaze_corr=None,
                blaze_file=None)
    lbl_wrap._velocity_template(dict(SMALL_RV_SHIFT=True), **args)
    assert calls == []
    lbl_wrap._velocity_template(dict(SMALL_RV_SHIFT=False), **args)
    assert calls[0]['object_science'] == calls[0]['object_comparison'] \
        == 'STAR'
    assert calls[0]['overwrite'] is False
    # a template made flat: made again
    from astropy.io import fits
    flat = tmp_path / 'LBL_Template_STAR.fits'
    fits.PrimaryHDU(header=fits.Header([('SMALLRV', True)])).writeto(flat)
    monkeypatch.setattr(lbl_wrap, '_template_of', lambda *a: str(flat))
    lbl_wrap._velocity_template(dict(SMALL_RV_SHIFT=False), **args)
    assert calls[1]['overwrite'] is True
    # one made iterating: kept
    fits.PrimaryHDU(header=fits.Header([('SMALLRV', False)])).writeto(
        flat, overwrite=True)
    lbl_wrap._velocity_template(dict(SMALL_RV_SHIFT=False), **args)
    assert len(calls) == 2


def test_a_shift_of_kilometres_per_second_is_iterated_to_convergence():
    """Bouchy's step is the first order of the shift's expansion, right only
    for a small shift: from 3 km/s off with no scan, two steps (the method
    before) leave 50 m/s, the steps taken to convergence leave the noise's
    precision"""
    true = np.array([1.0e-5, -1.0e-5, 6.0e-6])
    data, fits, grid = _data(true, noise=0.001)
    errors = np.full(true.size, np.nan)
    dv = ts.measure_velocities(data, fits, np.zeros(true.size), grid,
                               scan=False, scan_step=2.5e-7, errors=errors)
    expected = speed_of_light_ms * np.tanh(true)
    assert np.all(np.abs(dv - expected) < 3 * errors + 0.5)


def test_the_orders_the_velocities_are_measured_on():
    """TEMPLATE_RV_ORDERS = ref (the default): the whole orders whose median
    wavelength is within 5 % of COMPIL_SLOPE_REF_WAVE; bands, the orders
    with a quarter of their pixels within a nominal photometric band
    (between the deep telluric bands), their pixels outside left out; all,
    every order, at most as many as fit"""
    import pytest
    from lbl.core.base_classes import LblException
    # NIRPS-like orders, 970 to 1920 nm: y, J and H, the gaps between
    starts = np.linspace(970.0, 1900.0, 60)
    wave = np.array([np.linspace(lo, lo * 1.012, 400) for lo in starts])
    median = np.median(wave, axis=1)
    reg, which = ts.velocity_orders(dict(COMPIL_SLOPE_REF_WAVE=1600.0),
                                    wave, 1000)
    assert which == 'ref'
    assert reg == [o for o in range(len(starts))
                   if abs(median[o] / 1600.0 - 1) < 0.05]
    assert 0 < len(reg) < len(starts)
    with pytest.raises(LblException):
        ts.velocity_orders(dict(COMPIL_SLOPE_REF_WAVE=None), wave, 10)
    inside = ts.in_bands(wave)
    assert inside[np.abs(wave - 1600.0) < 1].all()
    assert not inside[np.abs(wave - 1400.0) < 20].any()
    assert not inside[np.abs(wave - 1880.0) < 20].any()
    reg, which = ts.velocity_orders(dict(TEMPLATE_RV_ORDERS='bands'), wave,
                                    1000)
    assert which == 'bands'
    share = inside.mean(axis=1)
    assert reg == [o for o in range(len(starts)) if share[o] >= 0.25]
    reg, _ = ts.velocity_orders(dict(TEMPLATE_RV_ORDERS='all'), wave, 7)
    assert len(reg) == 7 and reg[0] == 0 and reg[-1] == len(starts) - 1
    with pytest.raises(LblException):
        ts.velocity_orders(dict(TEMPLATE_RV_ORDERS='some'), wave, 10)

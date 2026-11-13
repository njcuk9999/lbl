#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The noise model of the fit's weights (lbl.pca2d.noise): from residuals
whose variance is alpha sigma^2 + gamma(lambda) + beta(lambda) sigma^4, the
three are found again; the weights of load_cube follow it, the cut on the
pixel SNR stays on the photon sigma.

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_pca2d_noise.py

Created on 2026-11-12

@author: artigau
"""
import numpy as np
import pytest

from lbl.pca2d import noise


def _residuals(alpha, gamma, beta, seed=2):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 400, 3000
    wave = np.exp(np.linspace(np.log(1500.0), np.log(1600.0), n_cols))
    # the photon sigma spans a factor 10, as it does across an order
    sigma = np.exp(rng.uniform(np.log(0.003), np.log(0.03), (n_rows, n_cols)))
    g = gamma(wave)[None, :]
    b = beta(wave)[None, :]
    var = alpha * sigma ** 2 + g + b * sigma ** 4
    return rng.normal(0.0, np.sqrt(var)), sigma, wave


def test_the_three_terms_are_found_again():
    gamma = lambda w: np.full_like(w, 0.004 ** 2)              # noqa: E731
    beta = lambda w: 2000.0 * (w - 1500.0) / 100.0             # noqa: E731
    resid, sigma, wave = _residuals(1.3, gamma, beta)
    model = noise.measure(resid, sigma, wave)
    assert model.alpha == pytest.approx(1.3, rel=0.1)
    mid = np.array([1530.0, 1570.0])
    assert np.sqrt(model.gamma(mid)) == pytest.approx(0.004, rel=0.25)
    # the additive term follows its slope (each node alone is noisier: it
    #   dominates only at the largest sigma)
    assert model.beta(np.array([1590.0])) > 3 * model.beta(np.array([1510.0]))
    # what the weights use, the variance, against the true one
    true = 1.3 * sigma ** 2 + gamma(wave)[None, :] + (beta(wave)[None, :]
                                                       * sigma ** 4)
    ratio = model.variance(sigma, wave) / true
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    assert np.percentile(ratio, 95) < 1.15 and np.percentile(ratio, 5) > 0.85


def test_photon_noise_alone_gives_the_photon_model():
    resid, sigma, wave = _residuals(1.0, lambda w: 0 * w, lambda w: 0 * w)
    model = noise.measure(resid, sigma, wave)
    assert model.alpha == pytest.approx(1.0, rel=0.05)
    assert np.all(model.variance(sigma, wave) / sigma ** 2 < 1.1)


def test_the_photon_scale_never_falls_below_its_floor():
    """one sigma everywhere: the photon scale and the modal noise cannot be
    told apart; the photon one is kept at its floor rather than at 0"""
    rng = np.random.default_rng(1)
    sigma = np.full((300, 2000), 0.01)
    resid = rng.normal(0.0, 0.012, sigma.shape)
    wave = np.linspace(1500.0, 1510.0, 2000)
    model = noise.measure(resid, sigma, wave)
    assert model.alpha >= noise.ALPHA_MIN
    assert np.median(model.variance(sigma, wave)) == pytest.approx(
        0.012 ** 2, rel=0.1)


def test_the_model_travels_in_fit_npz(tmp_path):
    model = noise.NoiseModel(1.2, [1500.0, 1600.0], [1e-5, 2e-5], [0.0, 10.0])
    path = tmp_path / "fit.npz"
    np.savez(path, **model.arrays())
    back = noise.from_fit(np.load(path))
    assert back.alpha == 1.2
    assert np.allclose(back.beta(np.array([1550.0])), 5.0, rtol=0.02)
    np.savez(tmp_path / "old.npz", b=np.zeros(2))
    assert noise.from_fit(np.load(tmp_path / "old.npz")) is None

"""lbl.core.math.upper_envelope, the pseudo-continuum of the template that
compute_rv uses as norm: on a spectrum of absorption lines over a known
continuum, it follows the continuum, stays positive and does not overshoot."""
import numpy as np

from lbl.core import math as mp


def _spectrum(npix=4096, seed=5):
    rng = np.random.default_rng(seed)
    xpix = np.arange(npix, dtype=float)
    # a slow continuum, as the template's (it has no blaze)
    continuum = 1.3 + 0.05 * np.sin(xpix / npix * np.pi)
    flux = np.ones(npix)
    for centre in rng.uniform(0, npix, 400):
        depth = rng.uniform(0.05, 0.8)
        flux -= depth * np.exp(-0.5 * ((xpix - centre) / 2.5) ** 2)
    return continuum, continuum * np.clip(flux, 0.02, None)


def test_the_envelope_follows_the_continuum():
    continuum, spec = _spectrum()
    env = mp.upper_envelope(spec, len(spec) // 4)
    assert np.all(env > 0)
    # the 95th percentile of a box takes the high end of the continuum in
    #   it, and the envelope is constant beyond the first and last boxes:
    #   1.5% away from the ends, 2% at the ends
    inner = slice(len(spec) // 8, -len(spec) // 8)
    assert np.max(np.abs(env / continuum - 1)[inner]) < 0.015
    assert np.max(np.abs(env / continuum - 1)) < 0.02
    # the median of the spectrum, which the lines pull down, would not
    assert np.median(spec) / np.median(continuum) < 0.95


def test_gaps_and_edges_do_not_make_it_overshoot():
    continuum, spec = _spectrum()
    spec[:300] = np.nan
    spec[1800:2300] = np.nan
    spec[-50:] = np.nan
    env = mp.upper_envelope(spec, len(spec) // 4)
    assert np.all(np.isfinite(env))
    assert np.max(env) <= np.nanmax(spec)
    assert np.min(env) > 0.9 * np.min(continuum)


def test_too_few_finite_pixels_give_nan():
    spec = np.full(1000, np.nan)
    spec[10:12] = 1.0
    assert np.all(np.isnan(mp.upper_envelope(spec, 250)))

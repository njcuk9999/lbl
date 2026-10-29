"""The rough CCF of a model (lbl.science.general.rough_ccf_rv) on a template
with no data on part of the domain: the model is 0 there (its spline's
outside), and those zeros once made the CCF peak on the edge of its scan.
GJ 1214 with HARPS, 2026-10-25: the template of the PCA2D-corrected spectra
had nothing below ~550 nm, the model velocity came out at -260 km/s, every
line window was placed that far from its line, and the errors were four
times larger."""
from types import SimpleNamespace

import numpy as np
from astropy import constants

from lbl.science import general

C = constants.c.value


def _inst():
    return SimpleNamespace(params=dict(
        ROUGH_CCF_MIN_RV=-3e5, ROUGH_CCF_MAX_RV=3e5,
        ROUGH_CCF_EWIDTH_GUESS=2000, COMPUTE_CCF_HP_SCALE=30,
        CCF_SNR_MIN=7, PLOT=False, PLOT_COMPUTE_CCF=False, BERV=0.0))


def _model(velocity, lines, nord=20, npix=4096):
    """Orders from 500 to 700 nm at a constant velocity step, a continuum of
    100 and Gaussian lines (half depth, 2 km/s wide) shifted by velocity."""
    edges = np.geomspace(500.0, 700.0, nord + 1)
    wave = np.array([np.geomspace(edges[i], edges[i + 1], npix)
                     for i in range(nord)])
    shifted = lines * np.sqrt((1 + velocity / C) / (1 - velocity / C))
    flux = np.full_like(wave, 100.0)
    for line in shifted:
        dv = (wave / line - 1) * C
        near = np.abs(dv) < 10000
        flux[near] *= 1 - 0.5 * np.exp(-0.5 * (dv[near] / 2000.0) ** 2)
    return wave, flux


def test_a_model_without_data_on_half_its_domain_finds_its_velocity():
    rng = np.random.default_rng(1214)
    # the star's lines, in the half of the domain the model covers
    lines = np.sort(rng.uniform(605.0, 695.0, 200))
    # and mask lines on either side of where its data start (600 nm): as
    #   the scan moves them across, from the zeros to the continuum, they
    #   add a ramp to the CCF, whose high-pass leaves its extremes at the
    #   scan's edges
    edge = np.sort(rng.uniform(595.0, 605.0, 400))
    wave, model = _model(15000.0, lines)
    model[wave < 600.0] = 0.0
    mask = np.concatenate([lines, edge])
    weights = np.ones_like(mask)
    found, _ = general.rough_ccf_rv(_inst(), wave, model, mask, weights,
                                    kind='model')
    assert abs(found + 15000.0) < 300.0

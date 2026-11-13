#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The noise of each sample, measured on the fit's own residuals.

THE PHOTON MODEL IS NOT THE NOISE. The cube's sigma of ln(flux) is
1 / sqrt(f T), f T the counts before the telluric division (so the photons
the telluric correction amplifies are counted), scaled per order to the
pipeline's SNR at the order's centre (tfits._order_sigma). Two things are
missing from it:

- MODAL NOISE, proportional to the flux, so constant in ln(f): a fixed
  fraction of the flux, at its own level in each order;
- ADDITIVE NOISE, independent of the star: the read noise of the extraction
  (summed over the ten or twenty detector pixels of a spectral pixel, so
  the detector's read noise per pixel does not give it), the dark, and the
  photons of the backgrounds the pipeline subtracted, the thermal emission
  above all, which rises steeply at the red end of K. Its variance in ln(f)
  goes as 1 / (f T)^2, as sigma_p^4.

On consecutive SPIRou exposures of TOI-2120, the scatter of their ratio was
1.16 times the photon model at 6000 e- per pixel and 1.77 times below 380
e-; at 2260 nm, on TOI-1452, 3 to 5 times. A fit weighted by the photon
model alone gives the faint samples (order edges, line cores, the end of
K) far more weight than they carry, and its reduced chi2 is not 1 (5.0 on
the SPIRou baseline of SNR 20-40, against 1.2 to 1.5 on NIRPS's, which has
no K band).

So the variance of ln(f) of a sample at wavelength lambda is

    alpha sigma_p^2 + gamma(lambda) + beta(lambda) sigma_p^4

sigma_p the photon model: alpha one photon scale (the flat field, the
extraction's weights, the resampling onto the grid), gamma the modal noise
and beta the additive term, each linear in ln(lambda) between nodes about
one order apart, so that each follows its slope from one order to the next.
Written in sigma_p, it needs neither the gain nor the units of the flux and
holds for every instrument: a pipeline whose errors already count the
read noise and the backgrounds gets beta near 0 and alpha near 1.

MEASURED ON THE FIT, not on pairs of exposures (the visible spectrographs
take one exposure a night): after a first fit with the photon weights, the
residual of every exposure is noise and whatever the components do not
describe, which the weights must count too. Its robust variance in bins of
sigma_p within each stretch between two nodes, corrected for the degrees of
freedom the fit took, is fitted by non-negative least squares. The fit is
then made again with these weights, and every later reader of the cube
(figures, masks, correction) weighs the samples the same way: the model is
saved in fit.npz with the other settings of the weights.

The cut on the pixel SNR (weights.min_pixel_snr) stays on the photon model:
the noise model changes the weights, not which samples are masked.

@author: artigau
"""
from __future__ import annotations

import numpy as np

#: the spacing of the nodes, in ln(lambda): about one order of NIRPS or
#: HARPS, half of SPIRou's
NODE_STEP = 0.01
#: the least photon scale: the photon noise does not vanish, it can only
#: look smaller on the grid than on the pixels (the resampling averages
#: neighbours). Without it, where sigma_p spans too little to tell the
#: photon scale from the modal noise, the fit can put everything in the
#: modal term, and a faint exposure would weigh as much as a bright one
ALPHA_MIN = 0.5
#: the levels of sigma_p in each stretch between two nodes
N_LEVELS = 6
#: a bin needs this many samples
MIN_BIN = 200
#: the columns read to measure it: this many contiguous blocks across the
#: grid, of this many columns each
N_BLOCKS = 200
BLOCK_COLUMNS = 200


class NoiseModel:
    """alpha, and gamma and beta at nodes in wavelength, linear in
    ln(lambda) between them (NaN at a node no data reached)"""

    def __init__(self, alpha, node_wave, node_gamma, node_beta):
        self.alpha = float(alpha)
        self.node_wave = np.asarray(node_wave, dtype=float)
        self.node_gamma = np.asarray(node_gamma, dtype=float)
        self.node_beta = np.asarray(node_beta, dtype=float)

    def _at(self, values, wave):
        good = np.isfinite(self.node_wave) & np.isfinite(values)
        if not good.any():
            return np.zeros_like(np.asarray(wave, dtype=float))
        return np.interp(np.log(wave), np.log(self.node_wave[good]),
                         values[good])

    def gamma(self, wave):
        return self._at(self.node_gamma, wave)

    def beta(self, wave):
        return self._at(self.node_beta, wave)

    def variance(self, sigma_p, wave):
        """the variance of ln(f) of samples of photon sigma `sigma_p` (rows
        x columns) at the columns' wavelengths `wave`"""
        s2 = np.asarray(sigma_p, dtype=float) ** 2
        return (self.alpha * s2 + self.gamma(wave)[None, :]
                + self.beta(wave)[None, :] * s2 * s2)

    def arrays(self):
        """what fit.npz keeps of it"""
        return dict(noise_alpha=self.alpha, noise_node_wave=self.node_wave,
                    noise_node_gamma=self.node_gamma,
                    noise_node_beta=self.node_beta)

    def describe(self, bands=((380, 700), (950, 1350), (1450, 1800),
                              (1950, 2300), (2300, 2600))):
        """the photon scale, and the modal noise (% of the flux) and the
        additive term per band"""
        words = ["alpha %.2f" % self.alpha]
        for lo, hi in bands:
            inside = ((self.node_wave >= lo) & (self.node_wave < hi)
                      & np.isfinite(self.node_gamma))
            if not inside.any():
                continue
            words.append("%d-%d nm: modal %.2f%%, beta %.3g"
                         % (lo, hi,
                            100 * np.sqrt(np.median(self.node_gamma[inside])),
                            np.median(self.node_beta[inside])))
        return "; ".join(words)


def from_fit(fit):
    """the model a fit.npz recorded (np.load's object), or None"""
    files = list(getattr(fit, "files", None) or [])
    if "noise_node_wave" not in files:
        return None
    wave = np.asarray(fit["noise_node_wave"], dtype=float)
    if not wave.size:
        return None
    return NoiseModel(float(np.asarray(fit["noise_alpha"])), wave,
                      fit["noise_node_gamma"], fit["noise_node_beta"])


def measure_columns(n_columns, edge=0):
    """the columns the model is measured on: N_BLOCKS contiguous blocks of
    BLOCK_COLUMNS spread over the grid (contiguous, so that reading them
    from the cube's memory maps reads few pages); all of them for a small
    grid"""
    usable = np.arange(edge, n_columns - edge)
    if usable.size <= N_BLOCKS * BLOCK_COLUMNS:
        return usable
    starts = np.linspace(usable[0], usable[-1] - BLOCK_COLUMNS, N_BLOCKS)
    return np.unique(np.concatenate([np.arange(int(s), int(s) + BLOCK_COLUMNS)
                                     for s in starts]))


def _robust_var(values):
    med = np.median(values)
    return (1.4826 * np.median(np.abs(values - med))) ** 2


def fit_bins(bins, node_wave):
    """alpha, gamma and beta at the nodes from the bins [(wave, n,
    variance, mean sigma_p^2, mean sigma_p^4)], by non-negative least
    squares, each bin weighted by the inverse of the error of its variance,
    var sqrt(2 / n)"""
    from scipy.optimize import nnls

    n_nodes = node_wave.size
    log_nodes = np.log(node_wave)
    rows, target = [], []
    for wave, n, var, s2, s4 in bins:
        pos = np.interp(np.log(wave), log_nodes, np.arange(n_nodes))
        lo = int(np.floor(pos))
        hi = min(lo + 1, n_nodes - 1)
        frac = pos - lo
        row = np.zeros(1 + 2 * n_nodes)
        row[0] = s2
        row[1 + lo] += 1 - frac
        row[1 + hi] += frac
        row[1 + n_nodes + lo] += s4 * (1 - frac)
        row[1 + n_nodes + hi] += s4 * frac
        weight = np.sqrt(n / 2.0) / var
        rows.append(row * weight)
        # alpha = ALPHA_MIN + a non-negative excess
        target.append((var - ALPHA_MIN * s2) * weight)
    rows = np.array(rows)
    solution, _ = nnls(rows, np.array(target), maxiter=50 * rows.shape[1])
    solution[0] += ALPHA_MIN
    reached = np.any(rows[:, 1:1 + n_nodes] != 0, axis=0)
    gamma = np.where(reached, solution[1:1 + n_nodes], np.nan)
    beta = np.where(reached, solution[1 + n_nodes:], np.nan)
    return NoiseModel(solution[0], node_wave, gamma, beta)


def measure(residual, sigma_p, wave, dof=0.0):
    """
    The noise model from a fit's residuals.

    :param residual: (rows, columns), the data less the model, at the
                     samples the fit weighted (anything elsewhere)
    :param sigma_p: (rows, columns), their photon sigma, NaN or <= 0 where
                    the fit gave no weight
    :param wave: (columns,), the columns' wavelengths
    :param dof: the fraction of each column's variance the fit itself took
                (its parameters per column over its rows): the residual's
                variance is divided by 1 - dof

    :return: NoiseModel, or None when too few samples
    """
    residual = np.asarray(residual, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)
    wave = np.asarray(wave, dtype=float)
    live = (np.isfinite(residual) & np.isfinite(sigma_p) & (sigma_p > 0)
            & np.isfinite(wave)[None, :])
    if live.sum() < 10 * MIN_BIN:
        return None
    w_used = wave[np.any(live, axis=0)]
    lo, hi = np.log(w_used.min()), np.log(w_used.max())
    node_wave = np.exp(np.arange(lo, hi + NODE_STEP, NODE_STEP))
    keep = 1.0 / max(1.0 - float(dof), 0.5)
    bins = []
    for i in range(node_wave.size - 1):
        cols = (wave >= node_wave[i]) & (wave < node_wave[i + 1])
        if not cols.any():
            continue
        here = live[:, cols]
        if here.sum() < N_LEVELS * MIN_BIN:
            continue
        r = residual[:, cols][here]
        s = sigma_p[:, cols][here]
        lam = np.broadcast_to(wave[cols][None, :], here.shape)[here]
        edges = np.quantile(s, np.linspace(0, 1, N_LEVELS + 1))
        for a, b in zip(edges[:-1], edges[1:]):
            pick = (s >= a) & (s <= b)
            if pick.sum() < MIN_BIN:
                continue
            var = _robust_var(r[pick]) * keep
            if not np.isfinite(var) or var <= 0:
                continue
            bins.append((float(np.median(lam[pick])), int(pick.sum()), var,
                         float(np.mean(s[pick] ** 2)),
                         float(np.mean(s[pick] ** 4))))
    if len(bins) < 3:
        return None
    return fit_bins(bins, node_wave)

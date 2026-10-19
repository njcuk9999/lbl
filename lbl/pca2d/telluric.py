"""Data-driven telluric absorption map.

The first run on Proxima made it obvious that this is not optional. In the
barycentric frame the leading PCA components correlated with BERV (r = -0.68)
and with ambient humidity (r = -0.45), and their periodograms peaked at 121.8,
91.2, 73.0 and 45.8 days: to within the resolution these are 365.25/3, /4, /5
and /8, i.e. harmonics of the annual cycle that drives BERV and the seasonal
water-vapour column. Those components describe Chile, not Proxima.

The method here needs no external model. Telluric transmission follows
Beer-Lambert,

    ln T(lambda, airmass) = -tau(lambda) * airmass,

so at a *fixed observer-frame wavelength* the high-passed log flux should depend
linearly on airmass with slope -tau(lambda). The stellar spectrum, by contrast,
walks back and forth across the observer-frame grid by +-21 km/s over the year,
so it does not produce a coherent linear airmass dependence at a fixed sample.
Regressing every observer-frame sample against airmass therefore recovers a map
of the telluric optical depth, straight from the science frames.

The resulting tau is then applied per exposure: each spectrum gets the
transmission it actually had, exp(-tau * airmass_n), shifted into whatever frame
the cube is registered to.

Caveats, deliberately not hidden:
  * BERV and airmass are not perfectly independent for a single-site target, so
    some stellar signal can leak into tau. The observer-frame control run is the
    check on that.
  * A pure airmass regression cannot see water-vapour changes at fixed airmass;
    it recovers the mean tau, not the variable part. Regressing against the
    header humidity as a second term would help and is the obvious next step.
  * A real transmission model (molecfit, TAPAS, APERO's own telluric products)
    remains the better answer. Use `weights.telluric_file` for that.
"""

from __future__ import annotations

import os

import numpy as np

from .grids import doppler
from .logger import log

C_KMS = 299792.458


def _cache_path(config, key):
    from . import cache as _cache
    return os.path.join(_cache.ensure(config), "telluric_%s.npz" % key)


def build_tau_map(config, cache_suffix=""):
    """Regress the observer-frame cube against airmass; return (grid, tau).

    tau is the telluric optical depth per unit airmass, on the observer-frame
    grid. Positive means absorption.
    """
    from . import cube as _cube
    from .config import cache_key

    # same preprocessing, but never shifted: tellurics must stand still
    observer_config = {k: dict(v) if isinstance(v, dict) else v
                       for k, v in config.items()}
    observer_config["registration"] = dict(config["registration"])
    observer_config["registration"]["frame"] = "observer"
    observer_config["weights"] = dict(config["weights"])
    observer_config["weights"]["telluric_file"] = None
    observer_config["weights"]["telluric_mode"] = "none"

    path = _cache_path(config, cache_key(observer_config) + cache_suffix)
    if config["output"]["use_cache"] and os.path.exists(path):
        blob = np.load(path)
        log("loaded telluric tau map from %s" % path)
        return blob["grid"], blob["tau"]

    log("building the telluric tau map from the observer-frame cube")
    grid, data, sigma, _trans, meta = _cube.build_cube(observer_config)
    data = np.asarray(data, dtype=np.float64)

    airmass = np.asarray(meta["airmass"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = 1.0 / np.asarray(sigma, dtype=np.float64) ** 2
    weights[~np.isfinite(weights)] = 0.0
    weights[~np.isfinite(data)] = 0.0
    data = np.nan_to_num(data)

    good_rows = np.isfinite(airmass)
    if good_rows.sum() < 10:
        log("too few spectra with a valid airmass; tau map is all zeros", "warn")
        return grid, np.zeros(grid.size)

    airmass = airmass[good_rows]
    data = data[good_rows]
    weights = weights[good_rows]

    # weighted linear fit y = a + b * airmass, one per wavelength sample,
    # done as sums over spectra so it stays a handful of matrix-vector products
    s0 = weights.sum(axis=0)
    s1 = weights.T @ airmass
    s2 = weights.T @ (airmass ** 2)
    sy = (weights * data).sum(axis=0)
    sxy = (weights * data).T @ airmass

    determinant = s0 * s2 - s1 ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(determinant > 0, (s0 * sxy - s1 * sy) / determinant, 0.0)
    tau = -slope                                    # ln T = -tau * airmass
    tau = np.where(np.isfinite(tau), tau, 0.0)
    tau = np.clip(tau, 0.0, None)                   # emission is not absorption

    span = float(np.percentile(tau, 99.9))
    log("tau map: median %.2e, 99.9th percentile %.3f, %.2f%% of samples above 0.01"
        % (np.median(tau), span, 100 * np.mean(tau > 0.01)), "value")

    if config["output"]["use_cache"]:
        np.savez_compressed(path, grid=grid, tau=tau)
        log("cached telluric tau map -> %s" % path)
    return grid, tau


def transmission_for(grid, tau_grid, tau, target_grid, berv, target_berv, airmass):
    """Transmission of one exposure, sampled on the cube's grid."""
    wave_obs = target_grid * doppler(target_berv) / doppler(berv)
    tau_here = np.interp(wave_obs, tau_grid, tau, left=0.0, right=0.0)
    if not np.isfinite(airmass):
        airmass = 1.0
    return np.exp(-tau_here * airmass)


def apply_from_data(weights, grid, meta, config):
    """Multiply the PCA weights by transmission ** telluric_power, per exposure."""
    tau_grid, tau = build_tau_map(config)
    power = float(config["weights"]["telluric_power"])
    cut = config["weights"]["telluric_min_transmission"]

    observer_frame = config["registration"]["frame"] == "observer"
    target = 0.0 if observer_frame else float(config["registration"]["target_berv"])

    n_zeroed = 0
    for n in range(weights.shape[0]):
        berv = 0.0 if observer_frame else float(meta["berv"][n])
        trans = transmission_for(
            grid, tau_grid, tau, grid, berv, target, float(meta["airmass"][n])
        )
        if cut is not None:
            deep = trans < float(cut)
            n_zeroed += int(deep.sum())
            weights[n][deep] = 0.0
        weights[n] *= trans ** power

    log("telluric weighting applied (power %.1f)" % power)
    if cut is not None:
        log("zero-weight from transmission < %.2f: %.3f%%"
            % (cut, 100.0 * n_zeroed / weights.size), "value")
    return weights

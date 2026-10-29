#!/usr/bin/env python
"""Periodogram of every coefficient, with the star's known signals marked.

    python -m lbl.pca2d.figures.coeff_periodogram --fit outputs/TOI2120/1-3v/fit.npz \
        --planets 5.7998 --out outputs/TOI2120/1-3v/coeff_periodogram.pdf

The question this answers is the one that decides whether a correction is safe
to apply: does any component of the basis vary on the period of a known planet?

A component that does is a component that will subtract that planet out of the
spectra, and the velocities will come back cleaner precisely because the signal
is gone. The MAD would improve. Nothing in the residuals would look wrong. This
plot is the cheapest place to catch it, before any LBL pass is spent.

Marked on every panel: the known planet periods in red, one year and half a
year in blue, since BERV, the water column and solar elevation are all annual
and a component tracking them is expected rather than alarming.

Power is Lomb-Scargle on the nightly coefficient series, normalised so that the
false-alarm level from a bootstrap can be drawn on the same axis. The bootstrap
shuffles the coefficient values against fixed dates, which preserves the window
function; that matters here because these are ground-based series with a strong
one-day and one-year sampling structure.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

from lbl.pca2d.logger import log

SEASONAL = [(365.25, "1 yr"), (182.6, "1/2 yr")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", required=True, help="fit.npz written by the fit")
    p.add_argument("--planets", type=float, nargs="*", default=[],
                   help="known planet periods in days, marked in red")
    p.add_argument("--prot", type=float, default=None,
                   help="published rotation period in days, marked in orange")
    p.add_argument("--pmin", type=float, default=1.2)
    p.add_argument("--pmax", type=float, default=1000.0)
    p.add_argument("--bootstrap", type=int, default=500,
                   help="trials for the false-alarm level; 0 skips it")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def nightly(bjd, values):
    """One point per night, since that is the unit the fit works in."""
    nights = np.unique(bjd)
    return nights, np.array([values[bjd == n].mean() for n in nights])


def fap_level(t, y, freq, trials, rng, quantile=0.99):
    """Bootstrap false-alarm level: the 99th percentile of the highest peak.

    Values are shuffled against FIXED dates, so every trial keeps the real
    window function. Shuffling the dates instead would destroy it and give a
    threshold far too optimistic for a ground-based series.
    """
    if trials <= 0:
        return np.nan
    peaks = np.empty(trials)
    for i in range(trials):
        peaks[i] = LombScargle(t, rng.permutation(y)).power(freq).max()
    return float(np.quantile(peaks, quantile))


def main(argv=None):
    args = parse_args(argv)
    fit = np.load(args.fit)
    keep = ~fit["rejected"] if "rejected" in fit.files else np.ones(fit["bjd"].size, bool)
    bjd = fit["bjd"][keep]
    a, b = fit["a"][keep], fit["b"][keep]
    n_star, n_earth = a.shape[1], b.shape[1]
    comps = ([("a%d" % (k + 1), a[:, k]) for k in range(n_star)]
             + [("b%d" % (j + 1), b[:, j]) for j in range(n_earth)])
    if not comps:
        log("no coefficients to draw: the fit has no component at all", "warn")
        return

    # the nights, which belong to the exposures and not to any one coefficient:
    # the nominal fit has NO star coefficient (n_star 0), and asking a[:, 0] for
    # the dates failed on every one of them
    t = np.unique(bjd)
    baseline = t.max() - t.min()
    freq = np.linspace(1.0 / args.pmax, 1.0 / args.pmin, 20000)
    per = 1.0 / freq
    rng = np.random.default_rng(1)

    n = len(comps)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 1.15 * n + 1.2), sharex=True,
                             squeeze=False)
    log("  %-4s %10s %10s %10s" % ("comp", "peak (d)", "power", "> FAP 1%"))
    for ax, (name, values) in zip(axes[:, 0], comps):
        _, y = nightly(bjd, values)
        power = LombScargle(t, y).power(freq)
        level = fap_level(t, y, freq, args.bootstrap, rng)
        ax.plot(per, power, lw=0.8, color="#1f4e9c" if name[0] == "b" else "k")
        if np.isfinite(level):
            ax.axhline(level, color="0.55", lw=0.6, ls=":")
        for p in args.planets:
            ax.axvline(p, color="#b3261e", lw=0.9, ls="--", alpha=0.75)
        if args.prot:
            ax.axvline(args.prot, color="#e08a00", lw=0.9, ls="--", alpha=0.75)
        for p, _label in SEASONAL:
            ax.axvline(p, color="#1f4e9c", lw=0.7, ls="-.", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlim(args.pmin, args.pmax)
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=12, va="center")
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        peak = per[np.argmax(power)]
        flag = ""
        for p in args.planets:
            # "at the planet" means within one frequency resolution element
            if abs(1.0 / peak - 1.0 / p) < 1.0 / baseline:
                flag = "  <-- AT A PLANET PERIOD"
        ax.text(0.995, 0.86, "peak %.3f d%s" % (peak, flag), fontsize=6.5,
                ha="right", va="top", transform=ax.transAxes,
                color="#b3261e" if flag else "0.3")
        log("  %-4s %10.3f %10.4f %10s%s"
              % (name, peak, power.max(),
                 "yes" if np.isfinite(level) and power.max() > level else "no", flag))

    axes[-1, 0].set_xlabel("period (d)", fontsize=9)
    title = "coefficient periodograms, nightly"
    if args.planets:
        title += "   red: known planets at %s d" % ", ".join("%.4g" % p for p in args.planets)
    if args.prot:
        title += "   orange: $P_{rot}$ %.1f d" % args.prot
    fig.suptitle(title + "\nblue dash-dot: 1 yr and 1/2 yr;"
                 " dotted: bootstrap 1% false-alarm level", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = args.out or (os.path.splitext(args.fit)[0].replace("fit", "")
                       + "coeff_periodogram.pdf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log("wrote %s" % out)


if __name__ == "__main__":
    main()

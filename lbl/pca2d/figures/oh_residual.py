#!/usr/bin/env python
"""Does what the model fails to remove line up with the sky's own emission?

    python -m lbl.pca2d.figures.oh_residual --cube <cube> --fit <fit.npz> \
        --source-dir data/TOI2120 --out outputs/TOI2120/1-3v/oh_residual.pdf

SPIRou t.fits carry an `OHLine` extension, the model of the atmospheric OH
airglow that was subtracted. That emission is not faint: on TOI-2120 it reaches
twelve times the stellar flux at 1700.91 nm and 61 per cent of it at
1219.64 nm. Whatever fraction of it the subtraction gets wrong is left in the
spectrum, and it is left at a wavelength fixed in the observer's frame, which is
exactly the kind of thing this method is supposed to catch.

This asks whether it does. For every column of the cube it compares the residual
after the full two-frame model, in units of the estimated noise, against the
OH-to-flux ratio interpolated from the source files. If the residual is
independent of the airglow the two are uncorrelated; if the airglow is what is
left over, the residual rises with it.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr

from lbl.pca2d.grids import pixel_shift
from lbl.pca2d.logger import log
from lbl.pca2d.tfits import extensions_for
from lbl.pca2d.twoframe import (LanczosShifter, carry_template, fit_means,
                                 load_cube, mean_rows, star_model,
                                 fit_templates, subtract_carried)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--n-files", type=int, default=12,
                   help="exposures sampled for the OH profile; the median is used")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def oh_over_flux(source_dir, grid, n_files):
    """Median OH-to-flux ratio, interpolated onto the cube grid."""
    files = sorted(glob.glob(os.path.join(source_dir, "*.fits")))
    if not files:
        raise SystemExit("no spectra in %s" % source_dir)
    step = max(1, len(files) // n_files)
    stack = []
    for path in files[::step][:n_files]:
        with fits.open(path) as hdulist:
            if "OHLine" not in [h.name for h in hdulist]:
                raise SystemExit("%s has no OHLine extension; SPIRou only"
                                 % os.path.basename(path))
            e_flux, e_wave, _, _ = extensions_for(hdulist)
            wave = np.asarray(hdulist[e_wave].data, float).ravel()
            oh = np.asarray(hdulist["OHLine"].data, float).ravel()
            flux = np.asarray(hdulist[e_flux].data, float).ravel()
        good = (np.isfinite(wave) & np.isfinite(oh) & np.isfinite(flux)
                & (flux > 0) & (wave > 0))
        order = np.argsort(wave[good])
        stack.append(np.interp(grid, wave[good][order],
                               (oh[good] / flux[good])[order],
                               left=np.nan, right=np.nan))
    return np.nanmedian(np.vstack(stack), axis=0)


def main(argv=None):
    args = parse_args(argv)
    grid, data, w, meta = load_cube(args.cube)
    n, m = data.shape
    fit = np.load(args.fit)
    dv = float(fit["dv"])
    delta = -pixel_shift(np.asarray(fit["berv"], float), dv)
    shifter = LanczosShifter(m, a=8, max_shift=int(np.ceil(np.abs(delta).max())) + 2)
    templates, tgroup = fit_templates(fit, meta, n, m)
    means, group = fit_means(fit, meta, n, m)
    model = (mean_rows(means, group, n)
             + star_model(fit["P"], fit["a"], shifter, delta, n, m)
             + fit["b"] @ fit["Q"])
    if np.any(templates):
        # + S_n T_g, a chunk at a time: carried whole it is rows x groups x grid
        subtract_carried(model, -templates, tgroup, shifter, delta, 64)
    live = w > 0
    resid = np.where(live, data - model, np.nan)
    sigma = np.where(live, 1.0 / np.sqrt(np.where(live, w, 1.0)), np.nan)
    with np.errstate(invalid="ignore"):
        col = np.nanstd(resid, axis=0) / np.nanmedian(sigma, axis=0)

    ratio = oh_over_flux(args.source_dir, grid, args.n_files)
    ok = np.isfinite(col) & np.isfinite(ratio) & (live.sum(axis=0) > 20)
    rho = spearmanr(col[ok], ratio[ok]).statistic
    log("  %d usable columns, Spearman rho = %+.3f" % (ok.sum(), rho))

    edges = [0.0, 0.01, 0.05, 0.2, 1.0, 5.0, np.inf]
    centres, meds, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_ = ok & (ratio >= lo) & (ratio < hi)
        if m_.sum() < 50:
            continue
        centres.append(np.sqrt(max(lo, 1e-3) * min(hi, 20.0)))
        meds.append(np.median(col[m_]))
        counts.append(int(m_.sum()))
        log("  OH/flux %6.2f - %-8.2f  %7d columns  residual/sigma = %.2f"
              % (lo, hi, m_.sum(), meds[-1]))

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2))
    ax = axes[0]
    sub = np.where(ok)[0][::37]
    ax.plot(np.maximum(ratio[sub], 1e-4), col[sub], ".", ms=1.5, alpha=0.25,
            color="0.5")
    ax.plot(centres, meds, "o-", color="#b3261e", lw=1.6, ms=6,
            label="median per decade")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("OH emission / stellar flux", fontsize=9)
    ax.set_ylabel(r"residual / $\sigma$", fontsize=9)
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.legend(fontsize=8)
    ax.set_title("what the model leaves behind rises with the airglow"
                 "   (Spearman $\\rho$ = %+.2f)" % rho, fontsize=9)

    ax = axes[1]
    step = 2.0
    bins = np.arange(grid[0], grid[-1] + step, step)
    idx = np.clip(np.digitize(grid, bins) - 1, 0, bins.size - 2)
    prof = np.full(bins.size - 1, np.nan)
    for k in range(bins.size - 1):
        m_ = ok & (idx == k)
        if m_.sum() > 20:
            prof[k] = np.nanmax(ratio[m_])
    ax.semilogy(0.5 * (bins[:-1] + bins[1:]), prof, lw=0.8, color="#1f4e9c")
    ax.axhline(1.0, color="#b3261e", lw=0.8, ls="--")
    ax.text(grid[0], 1.3, "airglow brighter than the star", fontsize=7,
            color="#b3261e")
    ax.set_xlabel("wavelength (nm)", fontsize=9)
    ax.set_ylabel("peak OH / flux per 2 nm", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    log("wrote %s" % args.out)


if __name__ == "__main__":
    main()

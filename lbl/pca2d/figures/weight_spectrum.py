#!/usr/bin/env python
"""Which parts of the spectrum actually drive the fit, and do they drive the RVs.

    python -m lbl.pca2d.figures.weight_spectrum --cube <cube> --fit <fit.npz> \
        --mask lbl/masks/<the object's LBL mask>.fits \
        --out outputs/<object>/<tag>/weight_spectrum.pdf

A weighted fit is not democratic. A tenth of the columns can carry half the
weight, and if they do, that tenth is what the components describe, whatever the
rest of the spectrum looks like. This figure says where the weight is, where
each block's power ends up, and, when an LBL mask is given, where the velocities
come from.

The overlay of the last two is the point. The PCA and LBL both weight by their
own idea of the noise, and if they disagree about which region is informative,
then the correction is being applied where the velocities are not measured.
With photon-noise weights alone, for instance, the star block can put nearly
all of its weighted power in one band while LBL takes most of its lines from
another; the overlap in the last panel measures how far the two agree.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lbl.pca2d.grids import pixel_shift
from lbl.pca2d.logger import log
from lbl.pca2d.plotting import rows_of
from lbl.pca2d.twoframe import LanczosShifter, load_cube


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--mask", default=None,
                   help="an LBL mask, to overlay where the velocities come from")
    p.add_argument("--bin", type=float, default=20.0, help="binning in nm")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def binned(grid, values, step, how="sum"):
    """Bin a per-column quantity. `sum` for extensive things like weight and
    power, `mean` for intensive ones like coverage, which is a fraction and
    would otherwise grow with the number of columns in the bin."""
    edges = np.arange(grid[0], grid[-1] + step, step)
    idx = np.clip(np.digitize(grid, edges) - 1, 0, edges.size - 2)
    total = np.bincount(idx, weights=values, minlength=edges.size - 1)
    if how == "mean":
        count = np.bincount(idx, minlength=edges.size - 1)
        total = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), total


def main(argv=None):
    args = parse_args(argv)
    # the cube whole, then the rows the FIT used: its own list is the authority,
    # and re-deriving the selection would tie the figure to the signal-to-noise
    # cut as it is now rather than to the one the fit was made with
    grid, data, w, meta = load_cube(args.cube, min_snr_frac=0.0)
    fit = np.load(args.fit)
    select = rows_of(meta, fit)
    if select is not None:
        data, w, meta = data[select], w[select], meta[select]
    P, Q, a, b = fit["P"], fit["Q"], fit["a"], fit["b"]
    dv = float(fit["dv"])
    delta = -pixel_shift(np.asarray(fit["berv"], dtype=float), dv)
    shifter = LanczosShifter(data.shape[1], a=8,
                             max_shift=int(np.ceil(np.abs(delta).max())) + 2)

    # weighted power each block puts on each column, which is what decides
    # what the components describe. With NO star component (n_star 0, the
    # default), there is no star block to weigh, and the einsum would be asked
    # to contract a zero-length axis: the star's power is left at zero.
    star = np.zeros(data.shape[1])
    if P.ndim == 2 and P.shape[0]:
        prepared = shifter.prepare(P)
        for start in range(0, data.shape[0], 32):
            stop = min(start + 32, data.shape[0])
            carried = shifter.carry(prepared, delta[start:stop])
            star += np.einsum("nm,nkm->m", w[start:stop],
                              (a[start:stop, :, None] * carried) ** 2)
    earth = np.einsum("nm,nm->m", w, (b @ Q) ** 2)

    x, wt = binned(grid, w.sum(axis=0), args.bin)
    _, ps = binned(grid, star, args.bin)
    _, pe = binned(grid, earth, args.bin)
    cov = (w > 0).sum(axis=0) / max(data.shape[0], 1)
    _, cv = binned(grid, cov, args.bin, how="mean")

    lines = None
    if args.mask and os.path.exists(args.mask):
        from astropy.io import fits
        t = fits.getdata(args.mask, 1)
        names = t.dtype.names or []
        for c in ("ll_mask_s", "wavelength", "wave", "ll"):
            if c in names:
                wl = np.asarray(t[c], dtype=float)
                _, lines = binned(grid, np.zeros_like(grid), args.bin)
                lines, _ = np.histogram(wl, bins=np.arange(grid[0],
                                                           grid[-1] + args.bin,
                                                           args.bin))
                break

    n = 4 if lines is not None else 3
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 2.0 * n + 1.0), sharex=True)

    ax = axes[0]
    ax.fill_between(x, 100 * wt / wt.sum(), color="#1f4e9c", alpha=0.75, lw=0)
    ax.set_ylabel("weight\n(%% per %.0f nm)" % args.bin, fontsize=8)
    order = np.argsort(-wt)
    cum = np.cumsum(wt[order]) / wt.sum()
    half = x[order][:np.searchsorted(cum, 0.5) + 1]
    ax.set_title("half the weight comes from %.0f-%.0f nm; %d of %d bins carry it"
                 % (half.min(), half.max(), half.size, x.size), fontsize=9)

    ax = axes[1]
    ax.fill_between(x, 100 * ps / max(ps.sum(), 1e-30), color="0.3", alpha=0.8,
                    lw=0, label="star block")
    ax.plot(x, 100 * pe / max(pe.sum(), 1e-30), color="#b3261e", lw=1.2,
            label="observer block")
    ax.set_ylabel("weighted power\n(%% per %.0f nm)" % args.bin, fontsize=8)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(x, 100 * cv, color="#1f8f4e", lw=1.2)
    ax.set_ylabel("coverage\n(%% of rows)", fontsize=8)
    ax.set_ylim(0, 105)

    if lines is not None:
        ax = axes[3]
        ax.fill_between(x, 100 * lines / lines.sum(), color="#e08a00", alpha=0.8,
                        lw=0)
        ax.set_ylabel("LBL lines\n(%% per %.0f nm)" % args.bin, fontsize=8)
        # the number that matters: do the two agree about where to look
        overlap = float(np.sum(np.minimum(ps / max(ps.sum(), 1e-30),
                                          lines / max(lines.sum(), 1))))
        ax.set_title("overlap between the star block's power and the LBL lines:"
                     " %.0f%%" % (100 * overlap), fontsize=9)

    axes[-1].set_xlabel("wavelength (nm)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    log("wrote %s" % args.out)

    log("  %-14s %9s %9s %9s %9s" % ("band (nm)", "weight", "star", "observer",
                                       "LBL lines"))
    # LBL's photometric bands (u to K), those the grid reaches
    from lbl.core import astro
    for band in astro.bands:
        lo, hi = band.minimum, band.maximum
        m = (x >= lo) & (x < hi)
        if not m.any():
            continue
        row = (100 * wt[m].sum() / wt.sum(), 100 * ps[m].sum() / max(ps.sum(), 1e-30),
               100 * pe[m].sum() / max(pe.sum(), 1e-30))
        extra = ("%8.1f%%" % (100 * lines[m].sum() / lines.sum())
                 if lines is not None else "        -")
        log("  %5.0f - %-6.0f %8.1f%% %8.1f%% %8.1f%% %s"
              % (lo, hi, row[0], row[1], row[2], extra))


if __name__ == "__main__":
    main()

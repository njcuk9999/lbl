#!/usr/bin/env python
"""One exposure, before and after the correction, in each window of interest.

Drawn in the STAR'S rest frame, like every other figure in the set, so the same
wavelength means the same stellar feature from one page to the next.

    python -m lbl.pca2d.figures.sample_before_after --cube <cube> --fit <fit.npz> \
        --windows 1200.3:2 1593.6:2 1669.5:5 --out sample.pdf

The river plots of the sequence figure show all the spectra as an image. This
shows what the correction did to ONE spectrum, which is the thing an eye reads
fastest: the flux as it arrived, the flux as it leaves, and the model that was
taken out between them.

One page per window. The exposure drawn is the one whose correction has the
median amplitude, so the page is representative rather than flattering: picking
the largest correction would show the method at its most impressive and tell you
nothing about a typical night.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from lbl.pca2d.logger import log  # noqa: E402
from lbl.pca2d.twoframe import carried_means, fit_means, fit_templates, mean_rows  # noqa: E402
from lbl.pca2d.grids import parse_window, pixel_shift, window_block
from lbl.pca2d.plotting import live_mask, sample_source_file, window_parity
from lbl.pca2d.twoframe import (LanczosShifter, cube_grid, load_cube, row_parity,
                            star_model)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--windows", nargs="+", required=True,
                   help="centre:width pairs in nm, e.g. 1200.3:2 1669.5:5")
    p.add_argument("--row", type=int, default=None,
                   help="cube row to draw; default is the median correction")
    p.add_argument("--frame", choices=["star", "observer"], default="star",
                   help="which rest frame to draw in; the star's by default")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fit = np.load(args.fit)
    dv = float(fit["dv"])
    delta = -pixel_shift(np.asarray(fit["berv"], dtype=float), dv)
    grid = cube_grid(args.cube)
    _, _, _, meta = load_cube(args.cube, columns=np.arange(1))
    n = len(meta)
    parity = row_parity(meta, n)
    means, group = fit_means(fit, meta, n, grid.size)
    templates = fit_templates(fit, meta, n, grid.size)
    # for windows an order overlap covers twice, see window_parity
    sample_path = sample_source_file(args.cube, meta)
    names = [str(v) for v in fit["filename"]] if "filename" in fit.files else []
    pages = [parse_window(spec) for spec in args.windows]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        for centre, width in pages:
            lo, hi = centre - 0.5 * width, centre + 0.5 * width
            # The block around the window and nothing else: the cube read there
            # and the model evaluated there, exact inside the window for any
            # shift the Earth can produce (see grids.window_block).
            block = window_block(grid, centre, width, dv)
            if block is None:
                log("  %.1f-%.1f nm: outside the grid, skipped" % (lo, hi), "warn")
                continue
            a0, b0 = block
            g, data, w, _ = load_cube(args.cube, columns=np.arange(a0, b0))
            m = b0 - a0
            shifter = LanczosShifter(m, a=8,
                                     max_shift=int(np.ceil(np.abs(delta).max())) + 2)
            model = (star_model(fit["P"][:, a0:b0], fit["a"], shifter, delta, n, m)
                     + fit["b"] @ fit["Q"][:, a0:b0])
            # both means are part of the model: with --mean iterate the
            # components are centred and hold none of the static content
            model += np.asarray(mean_rows(means[:, a0:b0], group, n))
            if np.any(templates[0][:, a0:b0]):
                model += carried_means(shifter.prepare(templates[0][:, a0:b0]),
                                       templates[1], shifter, delta, 0, n)
            if args.frame == "star":
                w = shifter.rows(w, -delta)
                w = np.where(live_mask(w), w, 0.0)   # see plotting.live_mask
                data = shifter.rows(data, -delta)
                model = shifter.rows(model, -delta)
                data[w <= 0] = 0.0
            live = live_mask(w)
            band = (g >= lo) & (g <= hi)
            # The row with the median correction IN THIS WINDOW, not over the
            # whole spectrum, which would need the model on the whole grid;
            # for a page about this window, this window's is the more telling
            # choice anyway.
            with np.errstate(invalid="ignore"):
                amplitude = np.array([np.nanstd(np.where(live[i] & band, model[i],
                                                         np.nan))
                                      for i in range(n)])
            covers = (live[:, band].sum(axis=1) > 0.5 * band.sum())
            # In an order overlap both parities cover the window, one near its
            # order's middle and one at an edge. Draw the middle one, so the
            # page is not a picture of the worse of two measurements.
            own, off, n_cov = (window_parity(centre, sample_path) if sample_path
                               else (None, float("nan"), 0))
            if own is not None and n_cov > 1 and (covers & (parity == own)).any():
                covers = covers & (parity == own)
                log("  %.1f-%.1f nm: two orders reach it, drawing parity %d"
                      " (%.2f of a half-width from the order centre)"
                      % (lo, hi, own, off))
            if args.row is not None:
                row = args.row
            elif covers.any():
                candidates = np.where(covers)[0]
                order = candidates[np.argsort(amplitude[candidates])]
                row = int(order[order.size // 2])
            else:
                log("  %.1f-%.1f nm: no row covers this window, skipped"
                      % (lo, hi))
                continue
            label = names[row] if row < len(names) else "row %d" % row
            win = band & live[row]
            if win.sum() < 10:
                log("  %.1f-%.1f nm: only %d live columns, skipped"
                      % (lo, hi, win.sum()))
                continue
            x = g[win]
            before = data[row][win]
            corr = model[row][win]
            after = before - corr

            fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=True,
                                     gridspec_kw={"height_ratios": [2.0, 1.0]})
            ax = axes[0]
            ax.plot(x, before, lw=0.9, color="0.35", label="before")
            ax.plot(x, after, lw=0.9, color="#1f4e9c", label="after")
            ax.axhline(0, color="0.85", lw=0.6)
            ax.set_ylabel(r"$\ln(f/\mathrm{savgol}\,f)$", fontsize=9)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
            ax.set_title("%s   %.2f-%.2f nm\nscatter %.4f before, %.4f after"
                         "  (%+.1f%%)"
                         % (label, lo, hi, np.std(before), np.std(after),
                            100 * (np.std(after) / max(np.std(before), 1e-30) - 1)),
                         fontsize=9)

            ax = axes[1]
            ax.plot(x, corr, lw=0.9, color="#b3261e")
            ax.axhline(0, color="0.85", lw=0.6)
            ax.set_ylabel("removed", fontsize=9)
            ax.set_xlabel("wavelength (nm), %s frame" % args.frame, fontsize=9)
            ax.text(0.005, 0.06, "rms %.4f" % np.std(corr), fontsize=7.5,
                    transform=ax.transAxes, color="0.35")

            for a in axes:
                a.tick_params(labelsize=8)
                a.grid(alpha=0.15)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            log("  %.2f-%.2f nm: scatter %.4f -> %.4f (%+.1f%%), removed rms %.4f"
                  % (lo, hi, np.std(before), np.std(after),
                     100 * (np.std(after) / max(np.std(before), 1e-30) - 1),
                     np.std(corr)))
    log("wrote %s" % args.out)


if __name__ == "__main__":
    main()

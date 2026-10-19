#!/usr/bin/env python
"""Every step from the file to the corrected spectrum, on one page per window.

    python -m lbl.pca2d.figures.sequence --cube <cube> --fit <fit.npz> \
        --windows 1267:2 1669.5:5 --out sequence.pdf

The other figures each show one stage. This shows the chain, in the order the
pipeline applies it, on the same rows and the same wavelength axis, so that
what each stage removed is the difference between two panels the eye can put
side by side.

    1  the high pass: ln(f) minus a Savitzky-Golay of ln(f), and nothing else
    2  the reconstruction: both blocks and the observer frame's parity mean
       (and the star frame's too, under --mean iterate)
    3  minus the OBSERVER block: what a corrected file holds
    4  minus the STAR block: everything that is not the star
    5  minus everything: what nobody explained

Panel 3 is the product. The correction removes the observer block, its mean
per parity included, and nothing else: the star block, its own per-parity mean
and components, is the spectrum LBL is about to measure.

Everything is in the STAR'S rest frame with the rows ordered by barycentric
velocity, so a stellar feature is vertical and anything anchored to the Earth
slants.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


from lbl.pca2d.logger import log  # noqa: E402
from lbl.pca2d.grids import parse_window, pixel_shift, window_block  # noqa: E402
from lbl.pca2d.plotting import (rows_of,  # noqa: F401
                            live_mask, nan_cmap, raw_log_flux_window,  # noqa: E402
                            sample_source_file, window_parity)
from lbl.pca2d.twoframe import (LanczosShifter, cube_grid, fit_means,  # noqa: E402
                            load_cube, mean_rows, row_parity, star_model,
                            carried_means, fit_templates)


#: before and after in the flux traces: slots 1 and 2 of a validated
#: colour-blind safe categorical palette (CVD delta E 24.7 between them)
BEFORE, AFTER = "#2a78d6", "#eb6834"

#: the panels that show what is left once a block is taken out, and the
#: correction itself, all a few times smaller than the spectrum: they get their
#: own stretch, twice the width of their 5-95 percentile range, centred on zero
RESIDUAL_PANELS = ("nostar", "resid", "applied")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--windows", nargs="+", required=True,
                   help="centre:width pairs in nm")
    p.add_argument("--source-dir", default=None)
    p.add_argument("--n-overplot", type=int, default=5)
    p.add_argument("--out", required=True)
    # what the correct stage divides out, so that panels 3 and 6 show exactly
    # that; cli.shrink_args hands the same flags to both
    p.add_argument("--shrink", action="store_true")
    p.add_argument("--shrink-smooth", action="store_true")
    p.add_argument("--smooth-components", default=None)
    p.add_argument("--resolution", type=float, default=None)
    # The correction's options are shared with this figure ON PURPOSE, so that
    # panel 3 is what a corrected file holds (cli.shrink_args). Two of them are
    # about how the amplitudes were MEASURED rather than about what is divided
    # out, and this figure draws the fit's own amplitudes: it takes them so that
    # the shared list parses, and says on the page which ones it drew. Without
    # this, the whole figure died on "unrecognized arguments" and a report came
    # out with no river plot at all (SMETHELLS_20, 2026-09-15).
    p.add_argument("--weight", choices=("flux", "velocity"), default="flux")
    p.add_argument("--velocity-floor", type=float, default=None)
    p.add_argument("--mask", default=None,
                   choices=("exposure", "common", "none"),
                   help="which samples the corrected files blank, so the panels"
                        " hide the same ones (reconstruct --mask)")
    p.add_argument("--mask-share", type=float, default=1.0,
                   help="with --mask common, the fraction of exposures that"
                        " must have weighted a sample for it to be kept"
                        " (reconstruct --mask-share), so the panels hide"
                        " exactly what the corrected files hide")
    return p.parse_args(argv)


def _filled(block, ok):
    """`block` with its gaps at each row's own median, ready to be shifted.

    A shift operator has to be handed finite numbers, and which finite number
    matters: in ln flux zero means a flux of one, which is fifty times the
    continuum here, and the Lanczos kernel spreads it over sixteen samples on
    either side. The row's own median is the value that disturbs nothing.
    """
    block = np.asarray(block, dtype=float)
    with np.errstate(invalid="ignore"):
        fill = np.nanmedian(np.where(ok, block, np.nan), axis=-1, keepdims=True)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return np.where(ok, block, np.broadcast_to(fill, block.shape))


def panel(ax, image, x, scale, title, cmap=None, unit="rows", ylabel=True):
    im = ax.imshow(image, aspect="auto", cmap=cmap or nan_cmap("RdBu_r"),
                   vmin=-scale, vmax=scale, origin="upper",
                   extent=[x[0], x[-1], image.shape[0] - 0.5, -0.5])
    ax.set_title(title, fontsize=8.5)
    # WHICH rows: a cube built with nightly stacking has one row per night, and
    # a reader counting 259 of them for Proxima's 782 spectra had nothing on the
    # page to tell them why (2026-09-13)
    # On ONE panel, not on every one: the label is longer than a panel is tall,
    # so five of them written down the same axis overlap each other and the
    # panels between them. The caller labels the middle image.
    if ylabel:
        ax.set_ylabel("%s, ordered by BERV" % unit, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def window_arrays(cube, fit, means, group, grid, dv, delta, centre, width,
                  templates=None, correct=None, select=None):
    """The panels of one window, on the grid block around it, in the star's frame.

    Everything one page needs, computed on window_block's columns alone: the
    cube read there through a memory map, the star and observer blocks
    evaluated there, and every panel carried there by each exposure's BERV.
    The block is wide enough for the largest shift the Earth can produce, so
    inside the window this is the full-grid computation it replaces; in the
    margin around it it is not, and nothing there is drawn.

    It used to be the whole grid for every panel: eight full (rows x samples)
    arrays for a page that shows a few thousand columns, five minutes and more
    memory than the machine had. None when the window misses the grid.
    """
    block = window_block(grid, centre, width, dv)
    if block is None:
        return None
    a0, b0 = block
    g, data, w0, _ = load_cube(cube, columns=np.arange(a0, b0),
                               min_snr_frac=0.0)
    if select is not None:
        data, w0 = data[select], w0[select]
    n, m = data.shape
    shifter = LanczosShifter(m, a=8, max_shift=int(np.ceil(np.abs(delta).max())) + 2)
    offset = mean_rows(means[:, a0:b0], group, n)
    star = star_model(fit["P"][:, a0:b0], fit["a"], shifter, delta, n, m)
    star_mean = templates is not None and bool(np.any(templates[0][:, a0:b0]))
    if star_mean:
        # the star-frame mean of each row's parity is part of the star block
        star = star + carried_means(shifter.prepare(templates[0][:, a0:b0]),
                                    templates[1], shifter, delta, 0, n)
    earth = fit["b"] @ fit["Q"][:, a0:b0]
    n_star, n_earth = int(fit["P"].shape[0]), int(fit["Q"].shape[0])
    # What the correct stage divides out of the files: the observer block as
    # the fit has it, or shrunk where it is not significant (--shrink and
    # its variants), by the very function reconstruct.correct_many calls.
    # Column by column, so this block with its margin gives the window what
    # the whole grid gives the files.
    applied = offset + earth
    shrunk = bool(correct) and bool(correct.get("shrink")
                                    or correct.get("smooth_which"))
    if shrunk:
        from lbl.pca2d.shrink import correction_basis
        Q_correct, _ = correction_basis(
            fit["Q"][:, a0:b0], fit["b"], w0, correct.get("chi2_scale", 1.0),
            bool(correct.get("shrink")), bool(correct.get("shrink_smooth")),
            correct.get("smooth_which") or (), correct.get("fwhm"))
        applied = offset + fit["b"] @ Q_correct

    # Panel 4 is panel 3's mirror: the star block taken out instead of the
    # observer one, so what is left is everything that is not the star. In the
    # star's frame that is the slanted part, the atmosphere and the instrument.
    # Each array is built only when it is about to be carried and dropped once
    # it has been; the carried copies are float32, several digits more than a
    # colour map shows.
    steps = [
        ("given", lambda: data,
         "1. the high pass: $\\ln(f)$ minus a Savitzky-Golay of $\\ln(f)$"),
        ("model", lambda: star + offset + earth,
         "2. the reconstruction: %d star + %d observer, and %s"
         % (n_star, n_earth,
            ("both frames' parity means" if np.any(means[:, a0:b0])
             else "the star's spectrum per parity") if star_mean
            else "the observer parity mean")),
        ("corrected", lambda: data - applied,
         "3. minus the OBSERVER block: what a corrected file holds"),
        ("nostar", lambda: data - star,
         "4. minus the STAR block: everything that is not the star"),
        ("resid", lambda: data - offset - star - earth,
         "5. minus everything: what nobody explained"),
        ("applied", lambda: applied,
         "6. what the correction divides out%s"
         % (": shrunk where the component is not significant (all exposures)"
            if shrunk
            else ", the observer block and its mean")),
    ]
    home = {}
    alive = live_mask(shifter.rows(w0, -delta))
    if correct and str(correct.get("mask")) == "common":
        # the files blank every sample any exposure left unweighted, so the
        # panels hide it in every row too, carried the same way.
        #
        # WITHIN A GROUP, which is what a corrected file's mask is taken over:
        # one order parity of one object. Taken over every row instead, this
        # asked a sample to be alive in rows of the OTHER parity, which never
        # cover that wavelength at all, since orders n and n+2 do not overlap.
        # The intersection was then empty everywhere the two parities do not
        # meet: the three H-band windows of the joint run came out with no row
        # to draw and vanished from the report, and the J-band ones kept only
        # the 15-20% of columns that two orders reach (2026-09-13).
        live = live_mask(w0)
        shared = np.empty_like(w0)
        # NOT `g`, which is this function's grid: named that, the loop replaced
        # the grid with a group number and every window came back "too few
        # columns" (2026-09-13, an hour after the fix above)
        # `share` EST LA FRACTION DE POSES qui doit avoir pondere un
        # echantillon, comme dans reconstruct.fit_weights_mask: 1.0 est
        # l'intersection stricte, et en dessous le jeu de raies reste UNIQUE
        # mais decide a la majorite. Les panneaux doivent cacher exactement ce
        # que les fichiers corriges cachent, sinon ils montrent autre chose.
        share = float(correct.get("mask_share") or 1.0)
        for label in np.unique(group):
            rows_here = group == label
            if share >= 1.0:
                keep = live[rows_here].all(axis=0)
            else:
                votes = live[rows_here].sum(axis=0)
                keep = votes >= max(1, int(np.ceil(share * int(rows_here.sum()))))
            shared[rows_here] = keep.astype(w0.dtype)
        alive &= live_mask(shifter.rows(shared, -delta))
    for name, build, _ in steps:
        z = shifter.rows(build(), -delta)
        z[~alive] = np.nan
        home[name] = z.astype(np.float32)
        del z
    return {"a0": a0, "b0": b0, "grid": g, "home": home, "earth": earth + offset,
            "shifter": shifter, "titles": [(name, t) for name, _, t in steps]}


def load_context(cube, fit_path, source_dir=None, shrink=False, shrink_smooth=False,
                 smooth_components=None, resolution=None, mask=None,
                 mask_share=1.0):
    """What every page shares: the fit, its means, the rows and their BERV, and
    how the correct stage divides the observer block out (panels 3 and 6):
    the same options reconstruct takes, the weights scaled by the fit's median
    reduced chi2 as it scales them."""
    fit = np.load(fit_path)
    dv = float(fit["dv"])
    which = [int(v) - 1 for v in str(smooth_components or "").split(",") if v.strip()]
    correct = None
    if shrink or which or (mask and mask != "exposure"):
        from lbl.pca2d.resolution import fwhm_samples
        chi2 = np.asarray(fit["chi2_red"], dtype=float)
        good = np.isfinite(chi2) & (chi2 > 0)
        correct = {"shrink": bool(shrink), "shrink_smooth": bool(shrink_smooth),
                   "smooth_which": which,
                   "fwhm": fwhm_samples(resolution, dv) if resolution else None,
                   "chi2_scale": float(np.median(chi2[good])) if good.any() else 1.0,
                   "mask": str(mask or "exposure"),
                   "mask_share": float(mask_share or 1.0)}
    grid = cube_grid(cube)
    # the rows and their metadata; one column is the cheapest way to get them
    # NO cut here: the fit's own row list decides what is drawn. Re-deriving the
    # selection ties a figure to whatever the cut was the day the fit was made,
    # and the day it changed every existing fit stopped being drawable.
    _, _, _, meta = load_cube(cube, columns=np.arange(1), min_snr_frac=0.0)
    select = rows_of(meta, fit)
    if select is not None:
        log("  keeping the %d rows the fit used, of the cube's %d"
            % (len(select), len(meta)))
        meta = meta[select]
    n = len(meta)
    means, group = fit_means(fit, meta, n, grid.size)
    return {"cube": cube, "fit": fit, "dv": dv, "grid": grid,
            "delta": -pixel_shift(np.asarray(fit["berv"], dtype=float), dv),
            "parity": row_parity(meta, n),
            "berv": np.asarray(meta["berv"], dtype=float),
            "means": means, "group": group, "select": select,
            "templates": fit_templates(fit, meta, n, grid.size),
            "names": [os.path.basename(str(v)) for v in meta["filename"]],
            "objects": (np.asarray([str(v) for v in meta["object"]])
                        if "object" in getattr(meta, "colnames", []) else None),
            "per_row": (np.asarray(meta["n_exposures"], dtype=float)
                        if "n_exposures" in getattr(meta, "colnames", [])
                        else np.ones(n)),
            "sample_path": sample_source_file(cube, meta, source_dir),
            "source_dir": source_dir, "correct": correct}


def pages_for(objects):
    """[(row mask or None, label or None)]: the pages one window is drawn on.

    One per object and then one of all of them together when the cube holds
    several, since rows of three stars on one axis are three sets of lines at
    three systemic velocities, which nobody can read; the page of all of them
    stays because what the observer block does is common to them. One page,
    unlabelled, when there is one object, which is every solo run.
    """
    names = [n for n in dict.fromkeys(objects) if n] if objects is not None else []
    if len(names) < 2:
        return [(None, names[0] if names else None)]
    objects = np.asarray(objects)
    return ([(objects == n, n) for n in names]
            + [(None, " + ".join(names) + "   (all)")])


def draw_window(ctx, centre, width, n_overplot=5, only=None, label=None,
                panels=None):
    """One page: the five panels, then a few of the rows in flux.

    `panels` keeps the first few of them and drops the rest, with the flux row
    underneath either way: the first three ARE the method (what arrives, what
    the model says is there, what a corrected file holds), and the two after
    them are diagnostics that a page explaining the idea does not need.

    Returns the figure, or None when the window has nothing to draw. The report
    puts it in its PDF; the site saves the same figure as SVG, so the two can
    never show different things.

    `only` keeps the rows of one object of a joint cube and `label` names it in
    the title: three campaigns drawn on one axis, ordered by BERV, interleave
    three different stars' lines, and the H-band pages of the first joint report
    were unreadable for it. The page of all of them together is drawn as well,
    since what the observer block does is common to them.
    """
    fit, grid, dv, delta = ctx["fit"], ctx["grid"], ctx["dv"], ctx["delta"]
    parity, berv = ctx["parity"], ctx["berv"]
    lo, hi = centre - 0.5 * width, centre + 0.5 * width
    arrays = window_arrays(ctx["cube"], fit, ctx["means"], ctx["group"], grid, dv,
                           delta, centre, width, templates=ctx["templates"],
                           correct=ctx.get("correct"), select=ctx.get("select"))
    if arrays is None:
        log("  %.1f-%.1f nm: outside the grid, skipped" % (lo, hi), "warn")
        return None
    g, home = arrays["grid"], arrays["home"]
    win = (g >= lo) & (g <= hi)
    if win.sum() < 10:
        log("  %.1f-%.1f nm: too few columns, skipped" % (lo, hi))
        return None
    x = g[win]
    cover = np.isfinite(home["given"][:, win]).sum(axis=1) / win.sum()
    keep = cover > 0.5 * max(cover.max(), 1e-9)
    own, off, ncov = (window_parity(centre, ctx["sample_path"])
                      if ctx["sample_path"] else (None, float("nan"), 0))
    # Where two orders reach this window, keep the one whose middle it
    # sits nearest: the other measures the same wavelengths at an order
    # edge, where the blaze has fallen away.
    # parity % 2, not parity: a joint cube labels its rows 2 * object + parity,
    # so comparing the label itself to an ORDER parity of 0 or 1 kept the first
    # object alone and left every other object's page with no row to draw
    if own is not None and ncov > 1 and (keep & (parity % 2 == own)).sum() >= 6:
        keep &= parity % 2 == own
        log("  %.1f-%.1f nm: two orders reach it, drawing the one"
              " %.2f of a half-width from its centre" % (lo, hi, off))
    if only is not None:
        keep = keep & np.asarray(only, dtype=bool)
    rows = np.where(keep)[0][np.argsort(berv[keep])]
    if rows.size < 6:
        log("  %.1f-%.1f nm: too few rows, skipped" % (lo, hi))
        return None
    per_row = np.asarray(ctx.get("per_row", np.ones(len(berv))), dtype=float)
    spectra = int(np.nansum(per_row[rows]))
    stacked = spectra > rows.size
    unit = "nights" if stacked else "exposures"
    count = ("%d %s of %d spectra" % (rows.size, unit, spectra) if stacked
             else "%d exposures" % rows.size)
    jw = np.where(win)[0]
    block = {k: home[k][np.ix_(rows, jw)] for k in home}
    finite = block["given"][np.isfinite(block["given"])]
    scale = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    pooled = np.concatenate([block[k][np.isfinite(block[k])]
                             for k in RESIDUAL_PANELS if k in block] or [np.zeros(0)])
    resid_scale = (float(np.diff(np.percentile(pooled, [5, 95]))[0])
                   if pooled.size else scale)

    titles = arrays["titles"]
    if panels:
        titles = titles[:int(panels)]
    n_img = len(titles)
    fig, axes = plt.subplots(n_img + 1, 1,
                             figsize=(9.4, 1.85 * n_img + 3.0),
                             sharex=True,
                             gridspec_kw={"height_ratios":
                                          [1.0] * n_img + [1.7]})
    im = im_resid = None
    middle = n_img // 2
    for r, (name, title) in enumerate(titles):
        extra = ("" if name in ("given", "model")
                 else "   scatter %.4f" % np.nanstd(block[name]))
        if name in RESIDUAL_PANELS:
            im_resid = panel(axes[r], block[name], x, resid_scale, title + extra,
                             unit=unit, ylabel=r == middle)
        else:
            im = panel(axes[r], block[name], x, scale, title + extra, unit=unit,
                       ylabel=r == middle)

    # the same rows in flux, before and after, from the raw flux the
    # cube build kept around this window (see cache.py)
    ax = axes[-1]
    pick = min(n_overplot, rows.size)
    spread = rows[np.linspace(0, rows.size - 1, pick).astype(int)]
    raw = raw_log_flux_window(ctx["cube"], ctx["source_dir"], ctx["names"],
                              parity, grid, arrays["a0"], arrays["b0"])
    sub, earth = arrays["shifter"], arrays["earth"]
    drawn = 0
    for cube_row in spread:
        blk = raw[cube_row]
        if not np.isfinite(blk).any():
            continue
        ok = np.isfinite(blk)
        lnf = sub.rows(_filled(blk[None, :], ok[None, :]),
                       -delta[[cube_row]])[0]
        live = sub.rows(ok.astype(float)[None, :],
                        -delta[[cube_row]])[0] > 0.99
        lnf = np.where(live, lnf, np.nan)
        atm = sub.rows(earth[[cube_row]], -delta[[cube_row]])[0]
        f_in = np.exp(lnf[jw])
        f_out = np.exp((lnf - atm)[jw])
        norm = np.nanmedian(f_in)
        if not np.isfinite(norm) or norm <= 0:
            continue
        # the first two slots of a colour-blind safe categorical palette,
        # validated as a pair: before blue, after orange
        ax.plot(x, f_in / norm, lw=0.8, color=BEFORE, alpha=0.6,
                label="before" if drawn == 0 else None)
        ax.plot(x, f_out / norm, lw=0.8, color=AFTER, alpha=0.6,
                label="observer block removed" if drawn == 0 else None)
        drawn += 1
    if drawn:
        ax.set_ylabel("flux / its own median", fontsize=8.5)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.set_title("%d of those rows in FLUX, spread over BERV %+.1f to"
                     " %+.1f km/s" % (drawn, berv[spread].min(),
                                      berv[spread].max()), fontsize=8.5)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "could not re-read the source spectra",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes)
    axes[-1].set_xlabel("wavelength (nm), star frame", fontsize=9)
    # attached to EVERY axes, the trace panel included. A colourbar
    # steals width from the axes it is given, so leaving one out makes
    # it wider than the rest and the wavelength axes stop lining up
    # down the page, which is the whole point of the figure.
    bar = fig.colorbar(im, ax=list(axes), fraction=0.022, pad=0.015,
                       label=r"$\ln(f/\mathrm{savgol}\,f)$")
    # laid out beside every panel so that the wavelength axes still line up;
    # now it spans the panels it describes, and the residual stretch gets a
    # bar of its own beside panels 4 and 5, in the same column
    box = bar.ax.get_position()
    span = lambda names: [axes[r].get_position() for r, (name, _) in enumerate(titles)
                          if (name in RESIDUAL_PANELS) == names]
    top, low = span(False), span(True)
    if top:
        y0, y1 = min(p.y0 for p in top), max(p.y1 for p in top)
        bar.ax.set_position([box.x0, y0, box.width, y1 - y0])
    if low and im_resid is not None:
        y0, y1 = min(p.y0 for p in low), max(p.y1 for p in low)
        fig.colorbar(im_resid, cax=fig.add_axes([box.x0, y0, box.width, y1 - y0]),
                     label="residual, ln f")
    # the star's name on its own line, big and bold and centred: with four pages
    # per window, which star a page is about has to be readable at a glance and
    # not found inside a sentence
    if label:
        fig.suptitle(label, fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.968 if label else 0.985,
             "%.2f-%.2f nm, %s: every step, in order, in the STAR'S REST FRAME\n"
             "vertical structure belongs to the star; anything slanted does not"
             % (lo, hi, count), ha="center", va="top", fontsize=10)
    log("  %s%.1f-%.1f nm: given %.4f | corrected %.4f (%+.0f%%) |"
          " residual %.4f"
          % ("%s " % label if label else "", lo, hi, np.nanstd(block["given"]),
             np.nanstd(block["corrected"]),
             100 * (np.nanstd(block["corrected"]) /
                    max(np.nanstd(block["given"]), 1e-30) - 1),
             np.nanstd(block["resid"])))
    return fig


def main(argv=None):
    args = parse_args(argv)
    ctx = load_context(args.cube, args.fit, args.source_dir, args.shrink,
                       args.shrink_smooth, args.smooth_components, args.resolution,
                       args.mask, args.mask_share)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pages = pages_for(ctx.get("objects"))
    with PdfPages(args.out) as pdf:
        for centre, width in (parse_window(spec) for spec in args.windows):
            for only, label in pages:
                fig = draw_window(ctx, centre, width, args.n_overplot, only, label)
                if fig is None:
                    continue
                if args.weight != "flux":
                    # Said on the page, because it is the one thing here that
                    # panel 3 does NOT show: with correct.weight: velocity the
                    # files are corrected with amplitudes refitted per exposure
                    # in that metric, and these panels are drawn with the fit's
                    # own. The blocks are the same; the amplitudes in front of
                    # them are measured differently.
                    fig.text(0.01, 0.003,
                             "panels drawn with the FIT's amplitudes; the files"
                             " were corrected with amplitudes refitted on"
                             " (dF/dv)\u00b2 (correct.weight: %s)" % args.weight,
                             fontsize=7, color="#b26a00", ha="left")
                pdf.savefig(fig)
                plt.close(fig)
    log("wrote %s" % args.out)
    return None


if __name__ == "__main__":
    main()

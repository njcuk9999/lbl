#!/usr/bin/env python
"""LBL velocities of runs that differ only in their component counts, in one PDF.

    python -m lbl.pca2d.lblscan --object NAME \
        --tags s2-o2v s2-o3v s2-o4v s3-o2v s3-o3v s3-o4v s4-o3v \
        --suffix _PCA2D_{tag} --out outputs/NAME/lbl_scan.pdf

A scan changes one count at a time, and the scans are read from the tags:
every star count that two runs or more share is an observer scan, and every
observer count that three runs or more share is a star scan (with two, its runs
are already in the observer scans). Every run is compared on the exposures all
of them have, and against the delivered spectra on those same exposures.

    the numbers   rms, robust sigma, nightly rms and median error of each run,
                  and the two things that say whether its fit can be trusted:
                  how far its velocity term moved the star (rms of vrad_fit)
                  and how closely a star coefficient follows the barycentric
                  velocity
    the counts    those numbers against the component count, one scan per row
    the matrix    rms and nightly rms on the grid of star by observer counts
    each scan     one panel of velocities per run
    the sky       every star coefficient against the observing conditions
    STRPCA        LBL's STRPCA amplitudes against the fit's own coefficients
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from .logger import log
from lbl.core.math import mad_to_sigma

#: delivered and corrected: slots 1 and 2 of the validated colour-blind safe
#: categorical palette the report's flux traces use (figures.sequence)
BEFORE, AFTER = "#2a78d6", "#eb6834"
#: a run's tag: s<M>-o<N>[v], M the star's spectra with the template (n_star),
#: N the observer components, v for the velocity term; also read in the short
#: form <K>-<N>[v], K the free star components alone (M - 1)
TAG = re.compile(r"^s(\d+)-o(\d+)(v?)$")
OLD_TAG = re.compile(r"^(\d+)-(\d+)(v?)$")
#: the numbers every run is summarised by, in the order they are shown
METRICS = (("rms", "rms"), ("robust", "robust sigma"),
           ("nightly_rms", "nightly rms"), ("median_error", "median error"))
#: the variants of a compilation: a colour-blind safe categorical palette
#: (Okabe and Ito, without its yellow); the original is always neutral grey
VARIANTS = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
ORIGINAL = "#4d4d4d"


# ------------------------------------------------------------- the velocities
def rdb_rows(path):
    """(rjd, vrad, svrad, table) of an LBL rdb, the finite rows only."""
    t = Table.read(path, format="ascii.rdb")
    rjd = np.asarray(t["rjd"], float)
    v = np.asarray(t["vrad"], float)
    e = np.asarray(t["svrad"], float)
    ok = np.isfinite(rjd) & np.isfinite(v) & np.isfinite(e) & (e > 0)
    return rjd[ok], v[ok], e[ok], t[ok]


def read_rdb(path):
    """(rjd, vrad, svrad) of an LBL rdb, the finite rows only."""
    return rdb_rows(path)[:3]


def nightly(rjd, v, e):
    """Weighted nightly means as (rjd, velocity, error), a night being a day of
    rjd + 0.5."""
    night = np.floor(rjd + 0.5).astype(int)
    out = []
    for n in np.unique(night):
        s = night == n
        w = 1 / e[s] ** 2
        out.append((rjd[s].mean(), np.sum(w * v[s]) / w.sum(), 1 / np.sqrt(w.sum())))
    return np.array(out).T


def velocity_stats(t, v, e):
    """rms, robust sigma (1.4826 MAD), median error and the rms of nightly
    weighted means, all about the median."""
    d = v - np.median(v)
    nt = nightly(t, v, e)
    return {"n": d.size, "rms": float(np.std(d)),
            "robust": float(mad_to_sigma * np.median(np.abs(d - np.median(d)))),
            "median_error": float(np.median(e)),
            "nightly_rms": float(np.std(nt[1] - np.median(nt[1]))),
            "nights": int(nt.shape[1])}


def amplitude_at(t, v, e, period):
    """(K, sigma_K) of a sinusoid at a FIXED period, weighted by 1/e^2.

    Used to ask the only question that matters about a correction applied to a
    star that has a signal: is the signal still there afterwards?
    """
    w = 1.0 / np.maximum(np.asarray(e, float), 1e-6) ** 2
    ph = 2 * np.pi * np.asarray(t, float) / float(period)
    A = np.column_stack([np.cos(ph), np.sin(ph), np.ones_like(ph)])
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ ((A * w[:, None]).T @ np.asarray(v, float))
    K = float(np.hypot(p[0], p[1]))
    dof = max(len(ph) - 3, 1)
    # the reduced chi^2, and nothing else: dividing it by the mean weight as
    # well would scale the error by the noise itself, which a test at a noise
    # of 1 m/s cannot see
    scale = float(np.sum(w * (v - A @ p) ** 2) / dof)
    var = (p[0] ** 2 * cov[0, 0] + p[1] ** 2 * cov[1, 1]
           + 2 * p[0] * p[1] * cov[0, 1])
    return K, float(np.sqrt(max(scale * var, 0.0)) / max(K, 1e-9))


def signal_stats(t, v, e, periods=(), degree=2):
    """What a correction did to a star that MOVES, which the rms cannot say.

    A correction can absorb part of a long-term astrophysical trend (the
    curvature a companion on a long orbit draws over a season): the raw rms
    then falls while the scatter within a night and the error per exposure do
    not improve, or get worse. A lower rms alone is therefore not a figure of
    merit, since it rewards eating astrophysics, so this reports, beside the
    rms:

      drift        peak-to-peak of a low-order polynomial in time, which is what
                   a companion on a long orbit looks like over one campaign
      residual     the scatter once that is removed: the honest number
      in_night     what one night's exposures disagree by, which no signal of
                   months can touch, and the cleanest measure of a correction
      amplitudes   K at each period the target is KNOWN to have
                   (target.planets in the configuration)

    A correction that shrinks `drift` or any amplitude is removing signal,
    whatever it does to the rms.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    nights = np.floor(t).astype(int)
    tc = t - t.mean()
    trend = np.polyval(np.polyfit(tc, v, degree), tc) if t.size > degree + 1 \
        else np.zeros_like(v)
    inside = [float(np.std(v[nights == n], ddof=1))
              for n in np.unique(nights) if (nights == n).sum() > 2]
    out = dict(velocity_stats(t, v, e))
    out.update({"drift": float(np.ptp(trend)),
                "residual": float(np.std(v - trend, ddof=1)),
                "in_night": float(np.median(inside)) if inside else float("nan"),
                "amplitudes": [amplitude_at(t, v, e, P) for P in periods]})
    return out


# -------------------------------------------------------------------- the runs
def parse_tag(tag):
    """(n_star, observer components) of a run's tag, n_star the star's spectra
    with the template: s1-o7 -> (1, 7), and the short form 0-7 -> (1, 7)."""
    m = TAG.match(str(tag))
    if m:
        return int(m.group(1)), int(m.group(2))
    m = OLD_TAG.match(str(tag))
    if not m:
        raise SystemExit("%r is not a run tag like s1-o7" % tag)
    return int(m.group(1)) + 1, int(m.group(2))


def scan_groups(tags):
    """The scans a set of runs holds, as [(kind, fixed count, tags)].

    An observer scan for every star count that two runs or more share, its runs
    by observer count; a star scan for every observer count that three runs or
    more share, its runs by star count. Observer scans first, by star count.
    """
    counts = {t: parse_tag(t) for t in tags}
    out = []
    for n_star in sorted({c[0] for c in counts.values()}):
        members = sorted((t for t in tags if counts[t][0] == n_star), key=lambda t: counts[t][1])
        if len(members) >= 2:
            out.append(("observer", n_star, members))
    for n_earth in sorted({c[1] for c in counts.values()}):
        members = sorted((t for t in tags if counts[t][1] == n_earth), key=lambda t: counts[t][0])
        if len(members) >= 3:
            out.append(("star", n_earth, members))
    return out


def load_run(obj, tag, suffix, outputs, lbl_dir):
    """What one run left: its LBL velocities, its fit and its coefficients.

    None when LBL has not measured it yet.
    """
    name = obj + suffix.format(tag=tag)
    rdb = os.path.join(lbl_dir, "lblrdb", "lbl_%s_%s.rdb" % (name, name))
    if not os.path.exists(rdb):
        log("no LBL velocities for %s yet (%s): left out" % (tag, rdb), "warn")
        return None
    t, v, e, table = rdb_rows(rdb)
    n_star, n_earth = parse_tag(tag)
    run = {"tag": tag, "name": name, "n_star": n_star, "n_earth": n_earth,
           "t": t, "v": v, "e": e, "table": table}
    folder = os.path.join(outputs, obj, tag)
    if os.path.exists(os.path.join(folder, "fit.npz")):
        run["fit"] = np.load(os.path.join(folder, "fit.npz"), allow_pickle=True)
    components = os.path.join(folder, "twoframe_components.fits")
    if os.path.exists(components):
        c = fits.getdata(components, "COEFFS")
        if "vrad_fit" in c.columns.names:
            keep = ~np.asarray(c["rejected"], bool) & np.isfinite(np.asarray(c["vrad_fit"], float))
            run["vrad_fit"] = np.asarray(c["vrad_fit"], float)[keep]
            # the BERV itself: with orbits, berv is the star's frame
            berv = np.asarray(c["berv"], float)
            if "berv_bary" in c.columns.names:
                bary = np.asarray(c["berv_bary"], float)
                berv = np.where(np.isfinite(bary), bary, berv)
            run["berv"] = berv[keep]
    return run


def on_common(runs, delivered):
    """Every run, and the delivered spectra, cut to the exposures all have."""
    key = lambda t: np.round(t, 6)
    common = key(delivered["t"])
    for run in runs:
        common = np.intersect1d(common, key(run["t"]))
    for run in [delivered] + runs:
        keep = np.isin(key(run["t"]), common)
        for name in ("t", "v", "e"):
            run[name] = run[name][keep]
        run["table"] = run["table"][keep]
        run["stats"] = velocity_stats(run["t"], run["v"], run["e"])
    return len(common)


def star_conditions(run):
    """(labels, names, rho): Spearman rho of every star coefficient of a run's
    fit against the observing conditions the fit recorded."""
    from .plotting import rank_correlations
    fit = run.get("fit")
    if fit is None:
        return None
    keep = ~np.asarray(fit["rejected"], bool)
    a = np.asarray(fit["a"], float)[keep]
    comps = [("a%d" % (k + 1), a[:, k]) for k in range(a.shape[1])]
    rho = np.asarray(rank_correlations(comps, np.asarray(fit["anc_values"])[:, keep]))
    return [str(x) for x in fit["anc_labels"]], [n for n, _ in comps], rho


def trust(run):
    """(rms of vrad_fit in m/s, its rho with BERV, the star coefficient that
    follows BERV most closely and its rho)."""
    out = [np.nan, np.nan, "", np.nan]
    if "vrad_fit" in run:
        out[0] = float(np.std(run["vrad_fit"]))
        out[1] = float(spearmanr(run["vrad_fit"], run["berv"])[0])
    sc = star_conditions(run)
    if sc is not None and "BERV" in sc[0]:
        col = sc[2][:, sc[0].index("BERV")]
        k = int(np.argmax(np.abs(col)))
        out[2], out[3] = sc[1][k], float(col[k])
    return out


def strpca_pairs(run):
    """[(key, the fit's a_k less its mean, LBL's STRPCAk, its error)] of a run,
    exposure by exposure, matched by file name."""
    fit, table = run.get("fit"), run["table"]
    keys = sorted((c for c in table.colnames if re.fullmatch(r"STRPCA\d+", c)),
                  key=lambda c: int(c[6:]))
    if fit is None or not keys:
        return []
    rejected = np.asarray(fit["rejected"], bool)
    first = {}
    for r, name in enumerate(str(x) for x in fit["filename"]):
        first.setdefault(name.split("t.fits")[0], r)
    stems = [os.path.basename(str(f)).split("t_")[0] for f in table["FILENAME"]]
    hit = np.array([s in first and not rejected[first[s]] for s in stems])
    rows = np.array([first[s] for s, h in zip(stems, hit) if h], dtype=int)
    a = np.asarray(fit["a"], float)
    out = []
    for key in keys:
        k = int(key[6:]) - 1
        x = a[rows, k] - a[~rejected, k].mean()
        y = np.asarray(table[key], float)[hit]
        e = np.asarray(table["s" + key], float)[hit]
        ok = np.isfinite(x) & np.isfinite(y)
        out.append((key, x[ok], y[ok], e[ok]))
    return out


# ------------------------------------------------------------------- the pages
def summary_page(delivered, runs, n_common, title):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.95, title, fontsize=13, weight="bold")
    fig.text(0.04, 0.915, "%d exposures that every run has; velocities in m/s, each"
             " about its median; robust sigma = 1.4826 MAD" % n_common, fontsize=9)
    head = ("%-10s %5s %5s %8s %8s %9s %9s   %12s %9s   %s"
            % ("run", "star", "obs", "rms", "robust", "nightly", "median",
               "vrad_fit rms", "vs BERV", "star coef. most like BERV"))
    lines = [head, "-" * len(head)]
    s = delivered["stats"]
    lines.append("%-10s %5s %5s %8.2f %8.2f %9.2f %9.2f" % (
        "delivered", "", "", s["rms"], s["robust"], s["nightly_rms"], s["median_error"]))
    for run in runs:
        s = run["stats"]
        vr, vb, which, rho = trust(run)
        lines.append("%-10s %5d %5d %8.2f %8.2f %9.2f %9.2f   %12.1f %+9.2f   %s"
                     % (run["tag"], run["n_star"], run["n_earth"], s["rms"], s["robust"],
                        s["nightly_rms"], s["median_error"], vr, vb,
                        "%s %+.2f" % (which, rho) if which else ""))
    fig.text(0.04, 0.86, "\n".join(lines), family="monospace", fontsize=8.6, va="top")
    fig.text(0.04, 0.18,
             "vrad_fit is the star shift each fit's velocity term took out of the observer"
             " block's reach, written to every corrected file as PCASTR_V. A fit whose\n"
             "velocity term moves the star by far more than LBL measures, in step with the"
             " barycentric velocity, or whose star coefficient follows the barycentric\n"
             "velocity, the seeing or the sun, has let the star and the sky trade places, and"
             " its correction leaves the difference in the flux as velocity.",
             fontsize=8.5, va="top")
    return fig


def group_name(kind, fixed):
    """'observer scan at 2 star components', 'star scan at 3 observer components'."""
    return ("observer scan at %d star component%s" % (fixed, "" if fixed == 1 else "s")
            if kind == "observer" else
            "star scan at %d observer component%s" % (fixed, "" if fixed == 1 else "s"))


def metrics_page(delivered, groups):
    """One row of metric panels per scan, against the count it varies."""
    fig, axes = plt.subplots(len(groups), len(METRICS),
                             figsize=(11, 0.9 + 2.5 * len(groups)), squeeze=False)
    for row, (kind, fixed, runs) in enumerate(groups):
        count = "n_earth" if kind == "observer" else "n_star"
        for col, (key, name) in enumerate(METRICS):
            ax = axes[row, col]
            x = [r[count] for r in runs]
            ax.plot(x, [r["stats"][key] for r in runs], "o-", color=AFTER, lw=1.2, ms=5,
                    label="corrected")
            ax.set_xticks(x)
            ax.axhline(delivered["stats"][key], color=BEFORE, ls="--", lw=1.0,
                       label="delivered")
            ax.set_xlabel("%s components" % kind, fontsize=8)
            ax.set_title("%s (m/s)" % name if row == 0 else "", fontsize=9)
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.2)
        axes[row, 0].set_ylabel(group_name(kind, fixed).replace(" at ", "\nat "), fontsize=8)
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("the numbers against the component count, one scan per row", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def matrix_page(delivered, runs):
    """rms and nightly rms on the grid of star by observer counts."""
    stars = sorted({r["n_star"] for r in runs})
    earths = sorted({r["n_earth"] for r in runs})
    if len(stars) < 2 or len(earths) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.2 + 0.55 * len(stars)))
    for ax, (key, name) in zip(axes, (("rms", "rms"), ("nightly_rms", "nightly rms"))):
        grid = np.full((len(stars), len(earths)), np.nan)
        for r in runs:
            grid[stars.index(r["n_star"]), earths.index(r["n_earth"])] = r["stats"][key]
        im = ax.imshow(np.ma.masked_invalid(grid), cmap="cividis", aspect="auto",
                       vmin=np.nanmin(grid), vmax=max(np.nanmax(grid), delivered["stats"][key]))
        for i in range(len(stars)):
            for j in range(len(earths)):
                if np.isfinite(grid[i, j]):
                    dark = grid[i, j] < np.nanmin(grid) + 0.5 * (np.nanmax(grid) - np.nanmin(grid))
                    ax.text(j, i, "%.1f" % grid[i, j], ha="center", va="center", fontsize=9,
                            color="white" if dark else "black")
        ax.set_xticks(range(len(earths)))
        ax.set_xticklabels(earths)
        ax.set_yticks(range(len(stars)))
        ax.set_yticklabels(stars)
        ax.set_xlabel("observer components", fontsize=8.5)
        ax.set_ylabel("star components", fontsize=8.5)
        ax.set_title("%s (m/s); delivered %.1f, blank not run" % (name, delivered["stats"][key]),
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("the matrix: every run on the grid of component counts", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def series_page(delivered, runs, which):
    panels = [delivered] + runs
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.2 + 1.75 * len(panels)),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for k, (ax, run) in enumerate(zip(axes, panels)):
        colour = BEFORE if k == 0 else AFTER
        d = run["v"] - np.median(run["v"])
        ax.errorbar(run["t"], d, yerr=run["e"], fmt="o", ms=2.2, lw=0.5,
                    color=colour, ecolor=colour, alpha=0.55, capsize=0)
        nt = nightly(run["t"], run["v"], run["e"])
        ax.plot(nt[0], nt[1] - np.median(run["v"]), "o", ms=3.2, mfc="white",
                mec="0.15", mew=0.8, label="nightly mean")
        ax.axhline(0, color="0.6", lw=0.6)
        s = run["stats"]
        ax.set_title("%s: rms %.2f, robust sigma %.2f, nightly rms %.2f, median error"
                     " %.2f m/s" % (run.get("tag", "delivered spectra"), s["rms"], s["robust"],
                                    s["nightly_rms"], s["median_error"]), fontsize=8.5)
        ax.set_ylabel("m/s", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.15)
    lim = np.percentile(np.abs(np.concatenate([r["v"] - np.median(r["v"]) for r in panels])), 99.5)
    axes[0].set_ylim(-1.15 * lim, 1.15 * lim)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[-1].set_xlabel("rjd (BJD - 2400000)", fontsize=9)
    fig.suptitle("the %s: LBL velocities of each run, and of the delivered spectra"
                 " on the same exposures" % which, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def conditions_page(runs):
    rows, names, labels = [], [], None
    for run in runs:
        sc = star_conditions(run)
        if sc is None:
            continue
        labels = sc[0]
        for name, r in zip(sc[1], sc[2]):
            rows.append(r)
            names.append("%s  %s" % (run["tag"], name))
    if not rows:
        return None
    rho = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.36 * len(rows)))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if abs(rho[i, j]) >= 0.3:
                ax.text(j, i, "%+.2f" % rho[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if abs(rho[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Spearman rho")
    ax.set_title("every star coefficient against the observing conditions, |rho| printed"
                 " above 0.30: a star coefficient has no business following these",
                 fontsize=9.5)
    fig.tight_layout()
    return fig


def strpca_page(runs, aliased):
    pairs = [(run["tag"], p) for run in runs for p in strpca_pairs(run)]
    if not pairs:
        return None
    ncol = min(4, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.6 + 2.7 * ncol, 0.8 + 2.9 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    for ax, (tag, (key, x, y, e)) in zip(axes.ravel(), pairs):
        r = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else np.nan
        slope = float(np.sum(x * y) / np.sum(x * x)) if x.size else np.nan
        ax.errorbar(x, y, yerr=e, fmt="o", ms=2, lw=0.4, color=AFTER, ecolor=AFTER,
                    alpha=0.5, capsize=0)
        lim = 1.1 * np.percentile(np.abs(np.r_[x, y]), 99.5) if x.size else 1
        ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=0.8)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_aspect("equal", adjustable="box")
        late = aliased and int(key[6:]) >= 3
        ax.set_title("%s %s: r %.3f, slope %.2f%s" % (tag, key, r, slope,
                                                       "\nLBL divides this residual twice"
                                                       if late else ""), fontsize=8)
        ax.set_xlabel("the fit's a%s" % key[6:], fontsize=7.5)
        ax.set_ylabel("LBL's " + key, fontsize=7.5)
        ax.tick_params(labelsize=7)
    fig.suptitle("LBL's STRPCA amplitudes against the fit's own coefficients, exposure by"
                 " exposure", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def variant_colour(k, n):
    """The k-th of n variants' colour: the categorical palette, or a
    sequential map once there are more variants than it has colours."""
    if n <= len(VARIANTS):
        return VARIANTS[k]
    return plt.get_cmap("viridis")(k / max(n - 1, 1))


def dv_histogram(ax, runs, colours):
    """Every series' velocities as a histogram, on ONE set of bins.

    The time panels say where the two differ; this says by how much, which is
    the number a correction is judged on. Shared bins, because two histograms
    binned differently are two pictures and not a comparison, and the range is
    the 99th percentile of everything drawn, so one wild exposure does not
    squeeze the distribution into the middle bin.
    """
    values = [run["v"] - np.median(run["v"]) for run in runs]
    lim = float(np.percentile(np.abs(np.concatenate(values)), 99)) * 1.25
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    bins = np.linspace(-lim, lim, 41)
    for run, value, colour in zip(runs, values, colours):
        s = run.get("stats") or velocity_stats(run["t"], run["v"], run["e"])
        ax.hist(value, bins=bins, histtype="stepfilled", color=colour,
                alpha=0.18, lw=0.0)
        ax.hist(value, bins=bins, histtype="step", color=colour, lw=1.3,
                label="%s: rms %.2f, robust %.2f m/s"
                      % (run["label"], s["rms"], s["robust"]))
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_xlabel("velocity - median (m/s)", fontsize=8.5)
    ax.set_ylabel("exposures", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=7.5)
    return ax


def periodogram_panel(ax, runs, colours, pmin=1.1, samples=4000):
    """Lomb-Scargle of every series, overplotted on one axis.

    Overplotted and not stacked: the question is whether a peak SURVIVED the
    correction or was made by it, and two panels answer that by eye and badly.
    Every exposure, weighted by LBL's own error bar, over periods from `pmin`
    days to the campaign's baseline; the power is the standard normalisation,
    so the two curves are on one scale. The peak period of each is in the
    legend, which is the number one reads off a plot like this anyway.
    """
    from astropy.timeseries import LombScargle

    t = runs[0]["t"]
    baseline = float(np.max(t) - np.min(t))
    if not np.isfinite(baseline) or baseline <= pmin:
        ax.text(0.5, 0.5, "the campaign is too short for a periodogram",
                fontsize=8, ha="center", va="center", transform=ax.transAxes,
                color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    freq = np.linspace(1.0 / baseline, 1.0 / pmin, samples)
    period = 1.0 / freq
    for run, colour in zip(runs, colours):
        power = LombScargle(run["t"], run["v"], run["e"]).power(freq)
        ax.plot(period, power, lw=0.9, color=colour, alpha=0.9,
                label="%s: peak %.3g d" % (run["label"],
                                           period[int(np.argmax(power))]))
    # annual and half-annual: the BERV, the water column and the solar
    # elevation all live there, so a peak at either is expected rather than
    # alarming. Drawn only when the campaign is long enough to reach them.
    for p in (365.25, 182.6):
        if baseline > p:
            ax.axvline(p, color="#1f4e9c", lw=0.7, ls="-.", alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlim(pmin, baseline)
    ax.set_xlabel("period (d)", fontsize=8.5)
    ax.set_ylabel("Lomb-Scargle power", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=7.5)
    return ax


def compilation_figure(original, variants, title=None):
    """RV time series of every variant against the original, in one figure.

    `original` and each of `variants` carry t, v, e, stats and a "label", on
    the same exposures (on_common). Top: the nightly means of all of them,
    each about its own median, the original in grey and every variant in its
    colour, with its rms and nightly rms in the legend. Below: one panel per
    variant, its exposures in its colour over the original's in light grey.
    Last row, the two ways of looking at the same velocities without time:
    the distribution of all of them, and their periodograms overplotted.
    """
    n = len(variants)
    runs = [original] + list(variants)
    colours = [ORIGINAL] + [variant_colour(k, n) for k in range(n)]
    fig = plt.figure(figsize=(11, 6.1 + 1.9 * n))
    grid = fig.add_gridspec(n + 2, 1,
                            height_ratios=[2.3] + [1.0] * n + [1.5])
    axes = [fig.add_subplot(grid[0])]
    axes += [fig.add_subplot(grid[k + 1], sharex=axes[0]) for k in range(n)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    last = grid[n + 1].subgridspec(1, 2, wspace=0.22)
    dv_histogram(fig.add_subplot(last[0]), runs, colours)
    periodogram_panel(fig.add_subplot(last[1]), runs, colours)
    centred = lambda run: run["v"] - np.median(run["v"])
    top = axes[0]
    for run, colour in zip(runs, colours):
        nt = nightly(run["t"], run["v"], run["e"])
        s = run["stats"]
        first = run is original
        # points and no line: a line between two nightly means draws a slope
        # across a gap of a fortnight that nothing measured, and on a campaign
        # with gaps that is most of what one ends up looking at
        top.plot(nt[0], nt[1] - np.median(run["v"]), "o", color=colour,
                 ms=4.0 if first else 3.2, lw=0,
                 alpha=1.0 if first else 0.9, zorder=3 if first else 2,
                 label="%s: rms %.1f, nightly rms %.1f m/s"
                       % (run["label"], s["rms"], s["nightly_rms"]))
    top.axhline(0, color="0.7", lw=0.6)
    top.set_ylabel("nightly mean - median (m/s)", fontsize=8.5)
    top.legend(fontsize=7.5, frameon=False, loc="upper left")
    top.grid(alpha=0.15)
    top.tick_params(labelsize=8)
    lim = np.percentile(np.abs(np.concatenate([centred(r) for r in runs])), 99.5)
    for k, (ax, run) in enumerate(zip(axes[1:], variants)):
        colour = colours[k + 1]
        ax.errorbar(original["t"], centred(original), yerr=original["e"], fmt="o", ms=2,
                    lw=0.4, color="0.72", ecolor="0.8", capsize=0, zorder=1)
        ax.errorbar(run["t"], centred(run), yerr=run["e"], fmt="o", ms=2.4, lw=0.5,
                    color=colour, ecolor=colour, alpha=0.8, capsize=0, zorder=2)
        s = run["stats"]
        ax.set_title("%s: rms %.2f, robust sigma %.2f, nightly rms %.2f, median error"
                     " %.2f m/s   (the original in grey)"
                     % (run["label"], s["rms"], s["robust"], s["nightly_rms"],
                        s["median_error"]), fontsize=8.2)
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("m/s", fontsize=8)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7.5)
    axes[-1].set_xlabel("rjd (BJD - 2400000)", fontsize=9)
    fig.suptitle(title or "LBL velocities: the original and every variant, on the same"
                 " %d exposures" % original["t"].size, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def berv_rho(run):
    """Spearman rho between a run's velocities and its BERV, or None.

    The number the BERV page exists for: a residual that follows the Earth's
    own motion is the instrument or the tellurics, not the star.
    """
    berv = run.get("berv")
    if berv is None or np.asarray(berv).size < 3:
        return None
    berv = np.asarray(berv, float)
    good = np.isfinite(berv) & np.isfinite(run["v"])
    if good.sum() < 3 or np.ptp(berv[good]) == 0:
        return None
    return float(spearmanr(run["v"][good], berv[good])[0])


def berv_figure(original, variants, title=None):
    """The same velocities, against BERV instead of time.

    Time says whether a correction removed scatter; BERV says where the scatter
    came from. A residual that rides the Earth's velocity is telluric or
    instrumental, and on a time axis it hides inside a year: the same points
    against BERV put it on a line. Every panel is points alone, no line joining
    them, since neighbours in BERV are nights apart and a line between them
    would be a shape nothing measured.

    Returns None when the rdb carried no BERV column, so a caller can simply
    not add the page.
    """
    runs = [original] + list(variants)
    if any(run.get("berv") is None for run in runs):
        return None
    n = len(variants)
    colours = [ORIGINAL] + [variant_colour(k, n) for k in range(n)]
    fig = plt.figure(figsize=(11, 3.4 + 1.9 * n))
    grid = fig.add_gridspec(n + 1, 1, height_ratios=[2.0] + [1.0] * n)
    axes = [fig.add_subplot(grid[0])]
    axes += [fig.add_subplot(grid[k + 1], sharex=axes[0]) for k in range(n)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    centred = lambda run: run["v"] - np.median(run["v"])       # noqa: E731

    top = axes[0]
    for run, colour in zip(runs, colours):
        rho = berv_rho(run)
        first = run is original
        top.errorbar(run["berv"], centred(run), yerr=run["e"], fmt="o",
                     ms=3.4 if first else 2.8, lw=0.4, color=colour,
                     ecolor=colour, alpha=0.9 if first else 0.8,
                     capsize=0, zorder=3 if first else 2,
                     label="%s: rms %.1f m/s%s"
                           % (run["label"], run["stats"]["rms"],
                              "" if rho is None else ", rho with BERV %+.2f" % rho))
    top.axhline(0, color="0.7", lw=0.6)
    top.set_ylabel("velocity - median (m/s)", fontsize=8.5)
    top.legend(fontsize=7.5, frameon=False, loc="upper left")
    top.grid(alpha=0.15)
    top.tick_params(labelsize=8)

    lim = np.percentile(np.abs(np.concatenate([centred(r) for r in runs])), 99.5)
    for k, (ax, run) in enumerate(zip(axes[1:], variants)):
        colour = colours[k + 1]
        ax.errorbar(original["berv"], centred(original), yerr=original["e"],
                    fmt="o", ms=2, lw=0.4, color="0.72", ecolor="0.8",
                    capsize=0, zorder=1)
        ax.errorbar(run["berv"], centred(run), yerr=run["e"], fmt="o", ms=2.4,
                    lw=0.5, color=colour, ecolor=colour, alpha=0.8, capsize=0,
                    zorder=2)
        rho, was = berv_rho(run), berv_rho(original)
        ax.set_title("%s: rho with BERV %s (the original in grey, %s)"
                     % (run["label"],
                        "n/a" if rho is None else "%+.2f" % rho,
                        "n/a" if was is None else "%+.2f" % was), fontsize=8.2)
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("m/s", fontsize=8)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7.5)
    axes[-1].set_xlabel("BERV (km/s)", fontsize=9)
    fig.suptitle(title or "the same velocities against BERV", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def compare_main(args):
    """--compare: the compilation of named LBL objects against the original."""
    rdb = lambda name: os.path.join(args.lbl_dir, "lblrdb", "lbl_%s_%s.rdb" % (name, name))
    t, v, e, table = rdb_rows(rdb(args.object))
    original = {"t": t, "v": v, "e": e, "table": table, "label": args.original_label}
    variants = []
    for item in args.compare:
        name, _, label = item.partition("=")
        t, v, e, table = rdb_rows(rdb(name))
        variants.append({"t": t, "v": v, "e": e, "table": table, "label": label or name})
    n_common = on_common(variants, original)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig = compilation_figure(original, variants, args.title)
    fig.savefig(args.out)
    plt.close(fig)
    log("wrote %s: %d variants on %d common exposures" % (args.out, len(variants), n_common),
        "info")
    for run in [original] + variants:
        s = run["stats"]
        log("%-44s rms %6.2f  robust sigma %6.2f  nightly rms %6.2f  median error %5.2f"
            % (run["label"], s["rms"], s["robust"], s["nightly_rms"], s["median_error"]),
            "value")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    p.add_argument("--tags", nargs="+", default=None, help="the runs, e.g. s2-o3v s3-o3v")
    p.add_argument("--compare", nargs="+", default=None, metavar="NAME=LABEL",
                   help="instead of a scan report, the compilation of these LBL"
                        " objects against the original (the object's own)")
    p.add_argument("--original-label", default="original, no PCA cleanup")
    p.add_argument("--suffix", default="_PCA2D_{tag}",
                   help="the corrected objects' LBL names, as lbl.suffix spells them")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--lbl-dir", default="lbl")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True, help="the PDF to write")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        return compare_main(args)
    if not args.tags:
        raise SystemExit("--tags, the runs of a scan, or --compare, named LBL objects")
    tags = list(dict.fromkeys(args.tags))
    delivered_rdb = os.path.join(args.lbl_dir, "lblrdb", "lbl_%s_%s.rdb"
                                 % (args.object, args.object))
    if not os.path.exists(delivered_rdb):
        raise SystemExit("no LBL velocities for the delivered spectra: %s" % delivered_rdb)
    t, v, e, table = rdb_rows(delivered_rdb)
    delivered = {"t": t, "v": v, "e": e, "table": table}
    loaded = {tag: load_run(args.object, tag, args.suffix, args.outputs, args.lbl_dir)
              for tag in tags}
    runs = sorted((loaded[tag] for tag in tags if loaded[tag] is not None),
                  key=lambda r: (r["n_star"], r["n_earth"]))
    if not runs:
        raise SystemExit("none of the runs has LBL velocities yet")
    n_common = on_common(runs, delivered)
    groups = [(kind, fixed, [loaded[t] for t in members if loaded[t] is not None])
              for kind, fixed, members in scan_groups([r["tag"] for r in runs])]
    from .lbltemplate import resproj_divides_in_place
    title = args.title or "%s: LBL velocities across component counts" % args.object
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    delivered["label"] = args.original_label
    for run in runs:
        run["label"] = run["tag"]
    pages = [lambda: summary_page(delivered, runs, n_common, title),
             lambda: compilation_figure(delivered, runs),
             lambda: metrics_page(delivered, groups) if groups else None,
             lambda: matrix_page(delivered, runs)]
    pages += [lambda g=g: series_page(delivered, g[2], group_name(g[0], g[1])) for g in groups]
    pages += [lambda: conditions_page(runs),
              lambda: strpca_page(runs, resproj_divides_in_place())]
    with PdfPages(args.out) as pdf:
        for page in pages:
            fig = page()
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
    log("wrote %s: %d runs on %d common exposures" % (args.out, len(runs), n_common), "info")
    for run in [delivered] + runs:
        s = run["stats"]
        log("%-10s rms %6.2f  robust sigma %6.2f  nightly rms %6.2f  median error %5.2f"
            % (run.get("tag", "delivered"), s["rms"], s["robust"], s["nightly_rms"],
               s["median_error"]), "value")


if __name__ == "__main__":
    main()

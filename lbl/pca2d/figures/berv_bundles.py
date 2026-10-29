"""The rows of the fit: which exposures were coadded, night by night, and the
BERV bundles each night was cut into. Per star and per night: a joint fit's
stars each have their own nights, even the nights they share (each star's
cube coadds its own exposures).

    python -m lbl.pca2d.figures.berv_bundles --cube <cube> --out bundles.pdf \
        [--title NAME] [--berv 0.1] [--minutes M] [--max N]

Read from the cube's members.csv (filename, row, group, bjd, berv), which the
cube build writes. One page. Top: every night of the campaign, each exposure a
point at its BERV less the night's median, consecutive bundles of a night
alternating in colour, each star's nights together under its name; the dashed lines are the widest range the limit allows
around the median. Bottom: the nights that were cut, BERV against time, a
vertical line at each cut. A cube of single exposures says so on the page.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from astropy.table import Table                                   # noqa: E402

from lbl.pca2d.logger import log                                  # noqa: E402

#: consecutive bundles of a night alternate between the first two slots of the
#: validated categorical palette, the pair the sequence figure uses; the cuts
#: are also drawn as lines, so colour is never the only cue
COLOURS = ("#2a78d6", "#eb6834")
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#e3e3e3"
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}
#: the most cut nights drawn one by one under the overview
MAX_DETAIL = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--berv", type=float, default=None,
                   help="input.bundle_berv, km/s")
    p.add_argument("--minutes", type=float, default=None,
                   help="input.bundle_minutes")
    p.add_argument("--max", type=float, default=None, help="input.bundle_max")
    return p.parse_args(argv)


def night_of(group: str) -> str:
    """The night a group belongs to: its label up to the bundle's #."""
    return str(group).split("#")[0]


def limits_text(args) -> str:
    said = []
    if args.berv:
        said.append("%.0f m/s of BERV" % (1000 * args.berv))
    if args.minutes:
        said.append("%g minutes" % args.minutes)
    if args.max:
        said.append("%d exposures" % int(args.max))
    return " or ".join(said) if said else "no limit (whole nights)"


def read_members(cube):
    path = os.path.join(cube, "members.csv")
    if not os.path.exists(path):
        return None
    table = Table.read(path, format="ascii.csv")
    if "berv" not in table.colnames:
        return None
    return table


def frame_label(cube):
    """What the BERVs of this page are: the BERV, or, when the cube's star
    frame follows known orbits, the BERV and the Keplerian velocity
    together (sequence.order_label, from the cube's meta.fits)."""
    try:
        from .sequence import order_label
        return order_label(Table.read(os.path.join(cube, "meta.fits")))
    except Exception:                                         # noqa: BLE001
        return "BERV"


def row_objects(cube):
    """{row name: star} of a joint cube (meta.fits names each row by its first
    exposure, and says whose it is); {} for a cube of one star."""
    try:
        meta = Table.read(os.path.join(cube, "meta.fits"))
    except Exception:                                         # noqa: BLE001
        return {}
    if "object" not in meta.colnames:
        return {}
    out = {os.path.basename(str(f)): str(o)
           for f, o in zip(meta["filename"], meta["object"])}
    return out if len(set(out.values())) > 1 else {}


def note_page(pdf_path, title, text):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(8.5, 3.0))
        fig.text(0.5, 0.62, title, ha="center", fontsize=12, color=INK,
                 weight="bold")
        fig.text(0.5, 0.40, text, ha="center", fontsize=10, color=MUTED,
                 wrap=True)
        fig.savefig(pdf_path)
        plt.close(fig)


def draw(members, args):
    # PER STAR AND PER NIGHT. A joint cube's members are every star's, and two
    #   stars observed the same night share its label: taken by the night
    #   alone they made one "night" whose BERV median mixed both stars
    #   (2026-10-24). Each star's own cube coadds its own exposures, so a
    #   night here is a star's night
    stars_of = row_objects(args.cube)
    stars = np.array([stars_of.get(os.path.basename(str(r)), "")
                      for r in members["row"]])
    groups = np.array(["%s|%s" % (s, g) for s, g in zip(stars, members["group"])])
    nights = np.array([night_of(g) for g in groups])
    berv = np.asarray(members["berv"], dtype=float)
    bjd = np.asarray(members["bjd"], dtype=float)
    # with orbits the members' "berv" is the star's frame
    by = frame_label(args.cube)
    order = np.argsort(bjd)
    groups, nights, berv, bjd, stars = (groups[order], nights[order],
                                        berv[order], bjd[order], stars[order])
    # each star's nights together, the stars in the order they come
    first = {}
    for s in stars:
        first.setdefault(s, len(first))
    names = sorted(set(nights), key=lambda n: (first[n.split("|")[0]],
                                               bjd[nights == n].min()))
    shown = [n.split("|", 1)[1] for n in names]
    n_stars = len(first) if stars_of else 1
    n_rows = len(set(groups))
    cut = [n for n in names if len(set(groups[nights == n])) > 1]
    widest = max((1000 * np.ptp(berv[nights == n]) for n in names), default=0.0)

    n_detail = min(len(cut), MAX_DETAIL)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(8.5, 4.2 + 2.4 * ((n_detail + 1) // 2)))
        grid = fig.add_gridspec(1 + (n_detail + 1) // 2, 2,
                                height_ratios=[1.6] + [1.0] * ((n_detail + 1) // 2),
                                hspace=0.55, wspace=0.25)
        top = fig.add_subplot(grid[0, :])
        for i, night in enumerate(names):
            sel = nights == night
            median = np.median(berv[sel])
            for k, group in enumerate(dict.fromkeys(groups[sel])):
                one = sel & (groups == group)
                top.plot(np.full(one.sum(), i), 1000 * (berv[one] - median), "o",
                         ms=4.5, color=COLOURS[k % 2], mec="white", mew=0.6,
                         zorder=3)
        if args.berv:
            for sign in (-1, 1):
                top.axhline(sign * 500 * args.berv, color=MUTED, lw=1.0,
                            ls="--", zorder=1)
        top.axhline(0, color=GRID, lw=1.0, zorder=0)
        # each star's block of nights, under its name, a line between two
        if n_stars > 1:
            starts = [i for i, n in enumerate(names)
                      if i == 0 or n.split("|")[0] != names[i - 1].split("|")[0]]
            for k, i0 in enumerate(starts):
                i1 = starts[k + 1] if k + 1 < len(starts) else len(names)
                if k:
                    top.axvline(i0 - 0.5, color=MUTED, lw=0.9, zorder=1)
                top.text(0.5 * (i0 + i1 - 1), 1.0, names[i0].split("|")[0],
                         transform=top.get_xaxis_transform(), ha="center",
                         va="bottom", fontsize=8, color=INK, weight="bold")
        top.set_xlim(-0.8, len(names) - 0.2)
        step = max(1, len(names) // 12)
        top.set_xticks(range(0, len(names), step))
        top.set_xticklabels([shown[i] for i in range(0, len(names), step)],
                            rotation=35, ha="right", fontsize=7, color=MUTED)
        top.set_ylabel("%s - night median (m/s)" % by, fontsize=9, color=INK)
        top.tick_params(labelsize=8, colors=MUTED)
        top.grid(axis="y", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            top.spines[spine].set_visible(False)
        top.set_title(
            "%d exposures, %s, %d rows of the fit; bundles of at most %s"
            "\nwidest night: %.0f m/s of BERV; %d night%s cut, consecutive bundles"
            " alternate in colour" % (
                len(berv), ("%d nights of %d stars (each star's own)"
                            % (len(names), n_stars)) if n_stars > 1
                else "%d nights" % len(names), n_rows, limits_text(args),
                widest, len(cut), "" if len(cut) == 1 else "s"),
            fontsize=9, color=INK, loc="left", pad=16 if n_stars > 1 else 6)
        # the most cut nights, one by one
        detail = sorted(cut, key=lambda n: -len(set(groups[nights == n])))
        for j, night in enumerate(detail[:n_detail]):
            ax = fig.add_subplot(grid[1 + j // 2, j % 2])
            sel = nights == night
            hours = 24.0 * (bjd[sel] - bjd[sel].min())
            g = groups[sel]
            b = berv[sel]
            for k, group in enumerate(dict.fromkeys(g)):
                one = g == group
                ax.plot(hours[one], b[one], "o", ms=4, color=COLOURS[k % 2],
                        mec="white", mew=0.5, zorder=3)
                if k:
                    ax.axvline(hours[one].min() - 0.5 * np.diff(hours).min()
                               if len(hours) > 1 else hours[one].min(),
                               color=INK, lw=0.8, zorder=2)
            star, label = night.split("|", 1)
            ax.set_title("%s%s: %d exposures, %d bundles, %.0f m/s"
                         % ("%s, " % star if n_stars > 1 else "", label,
                            sel.sum(), len(set(g)), 1000 * np.ptp(b)),
                         fontsize=8.5, color=INK, loc="left")
            ax.set_xlabel("hours since the first exposure", fontsize=8,
                          color=INK)
            ax.set_ylabel("%s (km/s)" % by, fontsize=8, color=INK)
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.grid(color=GRID, lw=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        if args.title:
            fig.suptitle(args.title, fontsize=13, weight="bold", color=INK)
        fig.savefig(args.out, bbox_inches="tight")
        plt.close(fig)
    return len(names), n_rows, len(cut), widest


def main(argv=None):
    args = parse_args(argv)
    members = read_members(args.cube)
    title = "The rows of the fit: nights and BERV bundles"
    if members is None:
        note_page(args.out, title, "this cube keeps no list of the exposures in"
                  " each row (it was built before members.csv existed)")
        log("  not drawn: no members.csv with BERVs in %s" % args.cube, "warn")
        return 0
    groups = [str(g) for g in members["group"]]
    if len(set(groups)) == len(groups):
        note_page(args.out, title, "every exposure is a row of its own:"
                  " nothing was coadded, so there is no bundle to show"
                  " (%d exposures)" % len(groups))
        log("  every exposure is its own row: no bundle", "info")
        return 0
    nights, rows, cut, widest = draw(members, args)
    log("wrote %s: %d nights, %d rows, %d nights cut, widest %.0f m/s"
        % (args.out, nights, rows, cut, widest), "info")
    return 0


if __name__ == "__main__":
    sys.exit(main())

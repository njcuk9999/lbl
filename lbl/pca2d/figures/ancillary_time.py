"""Every quantity of the correlation matrix, against time.

The correlation page says which ancillary quantity a component follows; this
page shows the quantities themselves over the campaign, one panel each, so a
season, a trend or a bad night is seen before anything is read into a
coefficient. The list is the matrix's own (fit.npz anc_labels, from
plotting.ancillary_table), time aside. Each exposure, or night in a stacked
cube, is one point, marked by what the fit did with it: used it, rejected its
coefficients, or never saw it because its band SNR was below the cut.

    python ancillary_time.py --cube CUBE --fit FIT.npz --out PAGE.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from lbl.pca2d.logger import log  # noqa: E402
from lbl.pca2d.plotting import ANCILLARY  # noqa: E402

#: panels per page: nine stacked time series still read on a portrait page
PER_PAGE = 9
RJD0 = 2400000.0
STYLE = {
    "fitted": dict(color="C0", marker="o", ms=3, ls="none", label="fitted"),
    "rejected": dict(color="C3", marker="x", ms=5, ls="none",
                     label="rejected by the coefficient cut"),
    "not fitted": dict(color="0.55", marker="o", ms=3, ls="none", mfc="none",
                       label="not fitted: band SNR below the cut"),
}
#: with several objects in one cube the colour says WHICH STAR and the marker
#: says what the fit did with the exposure: a joint run's page held three
#: campaigns of airmass and seeing with nothing to tell them apart.
MARKER = {"fitted": dict(marker="o", ms=3, ls="none"),
          "rejected": dict(marker="x", ms=5, ls="none"),
          "not fitted": dict(marker="o", ms=3, ls="none", mfc="none")}
#: a categorical palette that survives being printed and colour-blind readers
OBJECT_COLOURS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9",
                  "#e69f00")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True, help="the cube the fit was made from")
    p.add_argument("--fit", required=True, help="the fit's fit.npz")
    p.add_argument("--title", default="", help="the object, for the page title")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _column(meta, key):
    """A metadata column as floats, NaN where masked."""
    return np.ma.filled(np.ma.masked_invalid(np.ma.asarray(meta[key], dtype=float)),
                        np.nan)


def exposure_table(meta, fit):
    """(rjd, {label: values}, status), one entry per exposure of the cube.

    The labels are the correlation matrix's, in its order, without `time`.
    Values come from the cube's metadata; a quantity only the fit has (the
    water column, read from the t.fits headers) comes from the fit, NaN for
    an exposure it did not fit. status is "fitted", "rejected" (the
    coefficient cut) or "not fitted" (dropped before the fit, band SNR below
    the cut).
    """
    names = [os.path.basename(str(v)) for v in meta["filename"]]
    first = {}
    for i, name in enumerate(names):
        first.setdefault(name, i)
    order = sorted(first.values())
    exposures = [names[i] for i in order]

    fitted = [os.path.basename(str(v)) for v in fit["filename"]]
    rejected = (np.asarray(fit["rejected"], dtype=bool) if "rejected" in fit.files
                else np.zeros(len(fitted), dtype=bool))
    row_of = {}
    for i, name in enumerate(fitted):
        row_of.setdefault(name, i)
    status = np.array(["not fitted" if name not in row_of
                       else "rejected" if rejected[row_of[name]] else "fitted"
                       for name in exposures])

    column = {label: key for key, label in ANCILLARY}
    all_labels = [str(x) for x in fit["anc_labels"]]
    anc = np.asarray(fit["anc_values"], dtype=float)
    values = {}
    for k, label in enumerate(all_labels):
        if label == "time":
            continue
        key = column.get(label)
        if key is not None and key in meta.colnames:
            values[label] = _column(meta, key)[order]
        else:
            values[label] = np.array([anc[k][row_of[name]] if name in row_of
                                      else np.nan for name in exposures])
    rjd = _column(meta, "bjd")[order] - RJD0
    objects = (np.asarray([str(v) for v in meta["object"]])[order]
               if "object" in meta.colnames else np.array([""] * len(order)))
    return rjd, values, status, objects


def draw(rjd, values, status, title="", unit="exposures", objects=None):
    """One portrait page per PER_PAGE quantities, time on a shared axis.

    One colour per object when the cube holds several, the marker keeping what
    the fit did with the exposure. A joint run drew three campaigns' airmass,
    seeing and water on one axis with nothing to tell the stars apart.
    """
    labels = list(values)
    counts = ", ".join("%d %s" % ((status == s).sum(), s) for s in STYLE
                       if (status == s).any())
    names = ([] if objects is None
             else [n for n in dict.fromkeys(objects) if n])
    several = len(names) > 1
    colour = {n: OBJECT_COLOURS[i % len(OBJECT_COLOURS)]
              for i, n in enumerate(names)}
    figs = []
    for start in range(0, len(labels), PER_PAGE):
        chunk = labels[start:start + PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 1, sharex=True, squeeze=False,
                                 figsize=(8.5, 1.0 * len(chunk) + 1.8))
        for n, (ax, label) in enumerate(zip(axes[:, 0], chunk)):
            if several:
                for name in names:
                    for s, mark in MARKER.items():
                        sel = (status == s) & (objects == name)
                        if sel.any():
                            ax.plot(rjd[sel], values[label][sel],
                                    color=colour[name], **mark)
            else:
                for s, style in STYLE.items():
                    sel = status == s
                    if sel.any():
                        kw = dict(style)
                        if n:
                            kw.pop("label")
                        ax.plot(rjd[sel], values[label][sel], **kw)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        if several:
            # two legends: the colours are the stars, the markers are the fit
            stars = [plt.Line2D([], [], color=colour[n], marker="o", ms=4,
                                ls="none", label=n) for n in names]
            marks = [plt.Line2D([], [], color="0.35", ls="none",
                                label=STYLE[s]["label"],
                                **{k: v for k, v in MARKER[s].items()
                                   if k != "ls"}) for s in MARKER]
            first = axes[0, 0].legend(handles=stars, fontsize=7, ncol=len(stars),
                                      loc="upper left", framealpha=0.8)
            axes[0, 0].add_artist(first)
            axes[0, 0].legend(handles=marks, fontsize=6.5, ncol=3,
                              loc="upper right", framealpha=0.8)
        else:
            axes[0, 0].legend(fontsize=7, loc="upper right", ncol=3,
                              framealpha=0.8)
        axes[-1, 0].set_xlabel("BJD - 2400000")
        fig.suptitle("%severy quantity of the correlation matrix, against time"
                     % (title + ": " if title else ""), fontsize=11)
        fig.text(0.5, 0.945, "%d %s: %s" % (len(rjd), unit, counts),
                 ha="center", fontsize=8, color="0.3")
        fig.tight_layout(rect=(0, 0, 1, 0.935))
        figs.append(fig)
    return figs


def main(argv=None):
    args = parse_args(argv)
    fit = np.load(args.fit)
    if "anc_labels" not in fit.files or not len(fit["anc_labels"]):
        log("  no ancillary quantity in %s: nothing to draw" % args.fit, "warn")
        return None
    meta = Table.read(os.path.join(args.cube, "meta.fits"))
    rjd, values, status, objects = exposure_table(meta, fit)
    stacked = ("n_exposures" in meta.colnames
               and int(np.nanmax(_column(meta, "n_exposures"))) > 1)
    figs = draw(rjd, values, status, args.title,
                "nights" if stacked else "exposures", objects)
    with PdfPages(args.out) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    stars = [n for n in dict.fromkeys(objects) if n]
    log("wrote %s: %d quantities over %d %s%s"
        % (args.out, len(values), len(rjd), "nights" if stacked else "exposures",
           ", one colour per star (%s)" % ", ".join(stars) if len(stars) > 1
           else ""))
    return None


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""The velocities a run produced, added to that run's own report.

A run ends with LBL, whose velocities are written to an rdb under the LBL
tree's lblrdb folder. A correction is worth what it does to the velocities,
so the numbers belong in the document the run leaves behind, beside
the river plots that show what was subtracted. Every star of the run gets a
page, joint runs included, and when lbl.before was on (the delivered spectra
measured too) each page is the comparison rather than one curve on its own.

Appended to the report after the LBL stage, because that is when the rdb
exists; the report itself is bound at the figures stage, long before.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .logger import log
from .lblscan import (AFTER, BEFORE, berv_figure, compilation_figure,
                      nightly, rdb_rows, velocity_stats)


def rdb_path(data_dir, name):
    """Where LBL keeps the rdb of one object."""
    return os.path.join(data_dir, "lblrdb", "lbl_%s_%s.rdb" % (name, name))


def load_series(data_dir, name, label):
    """One rdb as the dict the figures take, or None when it is not there."""
    path = rdb_path(data_dir, name)
    if not os.path.exists(path):
        return None
    t, v, e, table = rdb_rows(path)
    if t.size == 0:
        log("  %s has no finite velocity in it" % os.path.basename(path), "warn")
        return None
    return {"t": t, "v": v, "e": e, "table": table, "label": label,
            "name": name, "path": path, "berv": berv_of(table),
            "stats": velocity_stats(t, v, e)}


def berv_of(table):
    """The BERV column of an rdb, or None when it has none.

    LBL spells it BERV; a hand-made rdb, and the ones the tests write, may not
    carry it at all, and that is not an error: it costs the BERV page and
    nothing else.
    """
    for column in table.colnames:
        if column.strip().lower() == "berv":
            berv = np.asarray(table[column], float)
            return berv if np.isfinite(berv).any() else None
    return None


def on_common(series):
    """Every series cut to the exposures they all have.

    Two runs compared on different exposures are not compared: a corrected set
    that lost its three worst nights would read as an improvement made of
    nothing. Returns a NEW list, so the callers' own stats stay intact.
    """
    key = lambda t: np.round(t, 6)                            # noqa: E731
    common = key(series[0]["t"])
    for run in series[1:]:
        common = np.intersect1d(common, key(run["t"]))
    out = []
    for run in series:
        keep = np.isin(key(run["t"]), common)
        cut = dict(run)
        cut["t"], cut["v"], cut["e"] = run["t"][keep], run["v"][keep], run["e"][keep]
        cut["table"] = run["table"][keep]
        if run.get("berv") is not None:
            cut["berv"] = run["berv"][keep]
        cut["stats"] = velocity_stats(cut["t"], cut["v"], cut["e"])
        out.append(cut)
    return out, int(common.size)


def solo_figure(run, title):
    """One set of velocities on its own: the exposures, and the nightly means.

    What a run looks like when lbl.before was off. It says the same numbers as
    a comparison page, without the thing to compare against, and says so.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0]})
    centred = run["v"] - np.median(run["v"])
    s = run["stats"]
    axes[0].errorbar(run["t"], centred, yerr=run["e"], fmt="o", ms=2.6, lw=0.5,
                     color=AFTER, ecolor=AFTER, alpha=0.85, capsize=0)
    axes[0].set_title("every exposure: rms %.2f, robust sigma %.2f, median"
                      " error %.2f m/s" % (s["rms"], s["robust"],
                                           s["median_error"]), fontsize=8.5)
    nt = nightly(run["t"], run["v"], run["e"])
    # points and no line joining them: consecutive nights can be a fortnight
    # apart, and the line drawn across the gap is a shape nothing measured
    axes[1].errorbar(nt[0], nt[1] - np.median(run["v"]), yerr=nt[2], fmt="o",
                     ms=4, lw=0, color=AFTER, ecolor=AFTER, capsize=0)
    axes[1].set_title("%d nightly weighted means: nightly rms %.2f m/s"
                      % (s["nights"], s["nightly_rms"]), fontsize=8.5)
    for ax in axes:
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_ylabel("m/s", fontsize=8.5)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("rjd (BJD - 2400000)", fontsize=9)
    fig.suptitle(title + "   (no uncorrected velocities: lbl.before was off)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def numbers(star, series, n_common):
    """The stats of one star's series, as text for the report's number page."""
    head = "%s   %d exposures in common, %d nights" % (
        star, n_common, series[-1]["stats"]["nights"])
    rows = ["  %-26s %8s %8s %8s %8s %6s"
            % ("", "rms", "robust", "nightly", "med err", "n")]
    for run in series:
        s = run["stats"]
        rows.append("  %-26s %8.2f %8.2f %8.2f %8.2f %6d"
                    % (run["label"], s["rms"], s["robust"], s["nightly_rms"],
                       s["median_error"], s["n"]))
    if len(series) == 2:
        before, after = series[0]["stats"], series[1]["stats"]
        for key, what in (("rms", "rms"), ("nightly_rms", "nightly rms")):
            if before[key] > 0:
                rows.append("  %-26s %8.2fx   %s"
                            % ("gain in " + what, before[key] / after[key],
                               "above 1 is better"))
    return head + "\n" + "\n".join(rows)


def stars_of(plan):
    """[(star, object name in LBL before, after)] for every star of a run.

    A joint run corrects several stars off one observer basis and LBL measures
    each of them on its own, so a joint report needs one page per star. The
    names are built exactly as the LBL stage built them (lbl.object_names), so
    a suffix changed in the config cannot make this look somewhere else.
    """
    from .lbl import object_names
    config, tag = plan["config"], plan["tag"]
    members = plan.get("members")
    stars = [m["object"] for m in members] if members \
        else [config["input"]["object"]]
    return [(star,) + object_names(config, star, tag) for star in stars]


def report_path(plan):
    """The bundle this run left, or None when the figures stage did not run."""
    from .naming import report_name

    config = plan["config"]
    stem = report_name(config["input"].get("object"), plan["tag"],
                       (config.get("provenance") or {}).get("run_hash"))
    path = os.path.join(plan["outdir"], stem + ".pdf")
    return path if os.path.exists(path) else None


def velocity_pages(plan, out=None):
    """Draw the RV pages of a run and append them to its report.

    Returns the path written, or None when LBL has measured nothing yet.
    """
    config = plan["config"]
    block = config.get("lbl") or {}
    data_dir = block.get("directory") or "lbl"
    asked_before = bool(block.get("before", True))

    pages, text = [], []
    for star, before_name, after_name in stars_of(plan):
        after = load_series(data_dir, after_name, "corrected")
        if after is None:
            log("no LBL velocities for %s yet (%s): no RV page for it"
                % (star, rdb_path(data_dir, after_name)), "warn")
            continue
        series = [after]
        if asked_before:
            before = load_series(data_dir, before_name, "delivered")
            if before is None:
                log("lbl.before is on but %s has no rdb: the page for %s shows"
                    " the corrected velocities alone"
                    % (rdb_path(data_dir, before_name), star), "warn")
            else:
                series = [before, after]
        if len(series) == 2:
            cut, n_common = on_common(series)
            fig = compilation_figure(
                cut[0], cut[1:],
                title="%s: LBL velocities, delivered and corrected, on the"
                      " same %d exposures" % (star, n_common))
        else:
            cut, n_common = series, series[0]["stats"]["n"]
            fig = solo_figure(cut[0], "%s: LBL velocities of the corrected"
                                      " spectra" % star)
        pages.append(fig)
        # the same points against BERV: time says whether the scatter went
        # down, BERV says whether what is left follows the Earth
        against_berv = berv_figure(
            cut[0], cut[1:],
            title="%s: the same velocities against BERV" % star)
        if against_berv is None:
            log("  %s has no BERV column: no page against BERV for %s"
                % (os.path.basename(cut[-1]["path"]), star), "warn")
        else:
            pages.append(against_berv)
        text.append(numbers(star, cut, n_common))

    if not pages:
        return None

    out = out or report_path(plan) or os.path.join(
        plan["outdir"], "%s_%s_velocities.pdf"
        % (config["input"].get("object"), plan["tag"]))
    tmp = out + ".rv.pdf"
    with PdfPages(tmp) as pdf:
        from .figures.bundle import text_pages
        text_pages(pdf, "The velocities, in numbers (m/s)",
                   "\n\n".join(text) + "\n\n"
                   + "rms and robust sigma are about the median; nightly rms is"
                     " the scatter of the weighted nightly means; med err is the"
                     " median of LBL's own error bar. Every series of a star is"
                     " cut to the exposures they share before anything is"
                     " measured.", size=8.0)
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)

    if os.path.exists(out) and out.endswith(".pdf") and os.path.getsize(out):
        _append(out, tmp)
        os.remove(tmp)
    else:
        os.replace(tmp, out)
    log("%d RV page%s added to %s" % (len(pages), "" if len(pages) == 1 else "s",
                                      out), "value")
    return out


def _append(report, extra):
    """Put `extra`'s pages at the end of `report`, keeping its outline."""
    from pypdf import PdfReader, PdfWriter
    base, add = PdfReader(report), PdfReader(extra)
    writer = PdfWriter()
    writer.append(base)
    start = len(base.pages)
    for page in add.pages:
        writer.add_page(page)
    writer.add_outline_item("The velocities LBL measured", start)
    with open(report, "wb") as fh:
        writer.write(fh)

#!/usr/bin/env python
"""Everything a run produced, as ONE multipage PDF.

    python -m lbl.pca2d.figures.bundle --config outputs/TOI2120/1-3v/resolved_config.yaml \
        --cube cache/cube_tfits_<key> --outdir outputs/TOI2120/1-3v

Scattering forty PDFs across three directories makes them hard to look at in
order, hard to send, and easy to compare across runs by accident: two files
with the same name in two directories are one careless click apart. A run now
leaves one document.

It opens with what was actually run, not what a config file says today: the
resolved parameters, after the package defaults are merged in, as plain
selectable text you can copy back into a YAML. Then the figures, grouped, with
PDF bookmarks so the sections are one click away.

The individual figures are built into a temporary directory and removed once
they are bound in. Pass --keep to leave them.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:                                           # pragma: no cover
    # PYPDF ONLY BINDS. Without it every figure is still drawn and the LaTeX
    # report (texreport) still holds them all; only the bound copy is missing.
    # LBL installed before pypdf was one of its dependencies has none, and
    # the figures stage stopped on this import (2026-10-21).
    PdfReader = PdfWriter = None

HERE = os.path.dirname(os.path.abspath(__file__))
# No sys.path edit here. There was one, pointing at the pca2d/ directory rather
# than at the repository, so it never made `import pca2d` work (the install
# does that); what it did do was put pca2d/ at the head of the path of the whole
# pipeline, since the CLI imports this module in-process. From then on the name
# `lbl` meant pca2d/lbl.py and not the LBL package, and a full run died at its
# last stage with "attempted relative import with no known parent package".

from lbl.pca2d.config import cache_key, load_config, spectra_dir  # noqa: E402
from lbl.pca2d.logger import log                              # noqa: E402

#: what a figure says when it drew less than it was asked for. Reported even
#: though the figure succeeded: a page that is not there is not an error anybody
#: notices, and three of them went unseen on the joint run of 2026-09-12.
SKIPPED = re.compile(r"\bskipped\b|\bnot drawn\b|\bnothing to draw\b")

#: default windows, the ones this campaign looks at
WINDOWS = ["1200.3:2", "1220:4", "1267:2", "1593.6:2", "1669.5:5", "1700.5:3",
           "2196.2:2", "2450:5"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--cube", required=True)
    p.add_argument("--outdir", required=True,
                   help="the fit's output directory; fit.npz is read from it"
                        " and the bundle is written into it")
    p.add_argument("--windows", nargs="+", default=WINDOWS)
    p.add_argument("--source-dir", default=None)
    p.add_argument("--out", default=None,
                   help="default: <outdir>/<object>_<tag>.pdf")
    p.add_argument("--keep", action="store_true",
                   help="leave the individual figures on disk as well")
    return p.parse_args(argv)


def wrap(text, width=96):
    """Break long lines so nothing runs off the page."""
    out = []
    for line in text.splitlines():
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > width // 2 else width
            out.append(line[:cut])
            line = "  " + line[cut:].lstrip()
        out.append(line)
    return out


def text_pages(pdf, title, body, per_page=62, size=7.0):
    """`body` as selectable monospace text, paginated."""
    lines = wrap(body)
    for start in range(0, len(lines), per_page):
        chunk = lines[start:start + per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.06, 0.965, title if start == 0 else title + " (continued)",
                 fontsize=11, weight="bold", va="top")
        fig.text(0.06, 0.93, "\n".join(chunk), fontsize=size, va="top",
                 family="monospace", linespacing=1.35)
        pdf.savefig(fig)
        plt.close(fig)


def clear_figures(folder):
    """Remove the report's figures of a previous run of this stage.

    Not the AppleDouble files macOS writes beside every file on an exFAT
    disk: `._name.pdf` goes when `name.pdf` does, and removing it afterwards
    raised FileNotFoundError on a joint run that had done nothing wrong
    (2026-09-17, on irrisor). A file that another process took in the
    meantime is not a reason to lose a stage either: the point is an empty
    folder, and what is already gone is gone.
    """
    if not os.path.isdir(folder):
        return 0
    removed = 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".pdf") or name.startswith("."):
            continue
        try:
            os.remove(os.path.join(folder, name))
            removed += 1
        except OSError:
            continue
    return removed


def object_name(config, args):
    """What to call this run.

    `input.object` when it is set, and otherwise the config file's own stem,
    which is how these are named. NOT the source directory's basename: that is
    `tfiles` for every target here, so a bundle would come out called tfiles.
    """
    named = config["input"].get("object")
    if named:
        return str(named)
    # outputs/<object>/<tag> is where this is being written, so the parent of
    # the output directory names the object. Tried before the config file's
    # stem, which is "config" whenever the generic config.yaml is used
    # directly, and before the source directory's basename, which is "tfiles"
    # for every target.
    parent = os.path.basename(os.path.dirname(os.path.normpath(args.outdir)))
    if parent and parent not in ("", ".", "outputs"):
        return parent
    stem = os.path.splitext(os.path.basename(args.config))[0]
    if stem and stem != "config":
        return stem
    return os.path.basename(os.path.normpath(spectra_dir(config)))


def summary(config, args, fit):
    """The few numbers a reader wants before anything else, as text."""
    rows = summary_rows(config, args, fit)
    width = max(len(k) for k, _ in rows)
    head = "\n".join("%-*s   %s" % (width, k, v) for k, v in rows)
    return head + settings_block(config)


def summary_rows(config, args, fit):
    """[(what, value)] of the run: the report's table and the text above."""
    tw = config["twoframe"]
    rows = []
    rows.append(("object", object_name(config, args)))
    rows.append(("source", spectra_dir(config)))
    key = cache_key(config)
    here = os.path.basename(os.path.normpath(args.cube))
    rows.append(("cube", here))
    rows.append(("this config's cube key", key + (
        "" if key in here else
        "   <-- DOES NOT MATCH THE CUBE ABOVE. The cache key hashes the"
        " configuration, not the data, so a config edited after the cube was"
        " built points somewhere else. Every figure below came from the cube"
        " named above.")))
    rows.append(("domain", "%.1f to %.1f nm at dv = %.2f km/s"
                 % (config["domain"]["wave_min"], config["domain"]["wave_max"],
                    config["domain"]["dv"])))
    rows.append(("high-pass", "savgol, %d samples (%.1f km/s), order %d, %s"
                 % (config["highpass"]["window"],
                    config["highpass"]["window"] * float(config["domain"]["dv"]),
                    config["highpass"]["polyorder"], config["highpass"]["mode"])))
    rows.append(("components", "%d star + %d observer"
                 % (tw["n_star"], tw["n_earth"])))
    rows.append(("nightly stacking", str(config["input"].get("nightly_stack"))))
    if fit is not None:
        n = int(np.asarray(fit["a"]).shape[0])
        rows.append(("rows fitted", "%d" % n))
        chi2n = float(fit["chi2_null"]) if "chi2_null" in fit.files else np.nan
        chi2b = float(fit["chi2_best"]) if "chi2_best" in fit.files else np.nan
        if np.isfinite(chi2n) and chi2n > 0:
            rows.append(("weighted variance removed", "%.2f%%"
                         % (100 * (1 - chi2b / chi2n))))
        ps = np.asarray(fit["power_star"]) if "power_star" in fit.files else None
        pe = np.asarray(fit["power_earth"]) if "power_earth" in fit.files else None
        if ps is not None and np.isfinite(chi2n) and chi2n > 0:
            rows.append(("star block shares",
                         np.array2string(100 * ps / chi2n, precision=2)))
            rows.append(("observer block shares",
                         np.array2string(100 * pe / chi2n, precision=2)))
        if "rejected" in fit.files:
            rows.append(("rejected by the MAD cut",
                         "%d" % int(np.asarray(fit["rejected"]).sum())))
    return rows


def settings_block(config):
    """Every window setting and the command, as the text pages print them."""
    # Every setting the window can change, all of them, whether or not it was
    # touched. A report that names only the settings that differ from some
    # default makes the reader guess what the default was; a report that names
    # only the ones above made a (dF/dv)^2 run indistinguishable from a flux
    # run on its own front page.
    from ..config import WINDOW_SETTINGS, setting_value
    keys = [path for path, _what in WINDOW_SETTINGS]
    kw = max(len(k) for k in keys)
    said = []
    for path, what in WINDOW_SETTINGS:
        value = setting_value(config, path)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        said.append("  %-*s  %-9s   %s" % (kw, path, value, what))
    block = ("\n\nEVERY SETTING THE WINDOW CAN SET, AND WHAT THIS RUN USED\n"
             + "\n".join(said))

    # and the command it was actually given, which the settings above cannot
    # show: they are the outcome, not the request
    command = (config.get("provenance") or {}).get("command")
    if command:
        block += "\n\nTHE COMMAND THIS RUN WAS GIVEN\n  " + command
    else:
        block += ("\n\nTHE COMMAND THIS RUN WAS GIVEN\n  not recorded: the run"
                  " predates provenance.command (2026-09-15)")
    return block


def periodogram_args(py, script, fit_path, out, config):
    """The periodogram's command, carrying the periods a component must not
    vary at.

    That is what makes the figure a check rather than a picture: a basis
    component varying at a planet's period subtracts that planet out of the
    spectra, the velocities come back cleaner BECAUSE the signal is gone, and
    nothing in the residuals looks wrong. No period was ever passed before
    2026-09-12, so the question the figure exists to answer was never put to it.
    The periods are published values, per object, under objects.<NAME>.target.
    """
    known = config.get("target") or {}
    argv = [py, script, "--fit", fit_path, "--out", out]
    if known.get("planets"):
        # every digit the archive gives: truncating a period is how a
        # marked line lands beside the peak instead of on it
        argv += ["--planets", *["%.10g" % float(p) for p in known["planets"]]]
    if known.get("prot"):
        argv += ["--prot", "%.10g" % float(known["prot"])]
    return argv


def run(cmd, failures):
    """Run one diagnostic, and say so if it fails rather than dying.

    The second argument is the list of failures to append to. It was called
    `log` until it shadowed the logger of that name, which every call in here
    then went through, so a failing figure raised TypeError instead of being
    written to the bundle's last page.
    """
    log("   " + " ".join(os.path.basename(c) if c.endswith(".py") else c
                         for c in cmd[:4]) + " ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        failures.append((" ".join(cmd),
                         (r.stderr or r.stdout or "").strip()[-400:], False))
        log("      failed, see the bundle's last page")
    else:
        # A figure that SUCCEEDS can still have drawn less than it was asked
        # for, and its own report of that was thrown away with the rest of its
        # output, since only a non-zero exit was ever shown. Three windows of
        # the joint run were skipped for "too few rows" and nobody saw it: the
        # report simply had three pages fewer than windows (2026-09-13).
        for line in (r.stdout or "").splitlines():
            if SKIPPED.search(line):
                log("   " + line.split("| ", 1)[-1].strip(), "warn")
                failures.append((" ".join(cmd[:4]) + " ...", line.strip(), True))
    return r.returncode == 0


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    fit_path = os.path.join(args.outdir, "fit.npz")
    fit = np.load(fit_path) if os.path.exists(fit_path) else None
    obj = object_name(config, args)
    tag = os.path.basename(os.path.normpath(args.outdir))
    from lbl.pca2d.naming import report_name
    out = args.out or os.path.join(args.outdir, report_name(
        obj, tag, (config.get("provenance") or {}).get("run_hash")) + ".pdf")
    tmp = tempfile.mkdtemp(prefix="bundle_")
    failures = []

    # ---- the figures ---------------------------------------------------
    py = sys.executable
    d = lambda name: os.path.join(HERE, name)                 # noqa: E731
    log("  building the figures")
    # the correct stage's own options, so panels 3 and 6 show what the files
    # actually have divided out, and panels 3 and 5 in yellow what they will
    # have masked (imported here: cli imports this module)
    from ..cli import clip_args, shrink_args
    run([py, d("sequence.py"), "--cube", args.cube, "--fit", fit_path,
         "--windows", *args.windows, "--source-dir",
         args.source_dir or spectra_dir(config),
         "--out", os.path.join(tmp, "sequence.pdf"), *shrink_args(config),
         *clip_args(config, quiet=True)], failures)
    run([py, d("weight_spectrum.py"), "--cube", args.cube, "--fit", fit_path,
         "--out", os.path.join(tmp, "weights.pdf")], failures)
    # which exposures each row of the fit holds: the nights, and the BERV
    # bundles a night was cut into (input.bundle_*)
    inp = config.get("input") or {}
    limits = []
    for flag, key in (("--berv", "bundle_berv"), ("--minutes", "bundle_minutes"),
                      ("--max", "bundle_max")):
        if inp.get(key) not in (None, "", False):
            limits += [flag, str(inp[key])]
    run([py, d("berv_bundles.py"), "--cube", args.cube, "--title", str(obj),
         "--out", os.path.join(tmp, "bundles.pdf"), *limits], failures)
    run(periodogram_args(py, d("coeff_periodogram.py"), fit_path,
                         os.path.join(tmp, "periodogram.pdf"), config), failures)
    run([py, d("ancillary_time.py"), "--cube", args.cube, "--fit", fit_path,
         "--title", str(obj), "--out", os.path.join(tmp, "ancillary.pdf")], failures)
    # Only scripts that take --windows are used, so the cube is loaded once per
    # script and not once per window. The per-window ones cost a full cube read
    # each: on 321 exposures that is 4.1 GB and a minute and a half, and with
    # three of them over eight windows the bundle spent an hour and a half
    # reading the same file twenty-four times. plot_snippet, star_frame_envelope
    # and compare_template are what went; sequence.py shows the whole chain for
    # every window in one read, and the stacks show the rest.
    per_window = []

    # ---- the front matter ----------------------------------------------
    front = os.path.join(tmp, "_front.pdf")
    with PdfPages(front) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.62, obj, fontsize=34, ha="center", weight="bold")
        fig.text(0.5, 0.565, "two-frame weighted PCA, %s" % tag, fontsize=13,
                 ha="center", color="0.3")
        fig.text(0.5, 0.52, "%d star + %d observer components"
                 % (config["twoframe"]["n_star"], config["twoframe"]["n_earth"]),
                 fontsize=11, ha="center", color="0.3")
        # the code that drew this report; the run's own is in its parameters
        from ..provenance import stamp as code_stamp
        fig.text(0.5, 0.49, "drawn by pca2d %s" % code_stamp(), fontsize=9,
                 ha="center", color="0.5")
        pdf.savefig(fig)
        plt.close(fig)
        text_pages(pdf, "What was run", summary(config, args, fit), size=8.5)
    params = os.path.join(tmp, "_parameters.pdf")
    with PdfPages(params) as pdf:
        text_pages(pdf, "Parameters, as resolved (copy this back into a YAML)",
                   yaml.safe_dump(config, sort_keys=False, default_flow_style=False,
                                  width=94))

    # ---- bind ------------------------------------------------------------
    order = [("Front matter", front)]
    # then the campaign itself, before any model: every quantity the
    # correlations are measured against, over time
    order.append(("Every quantity of the correlations, against time",
                  os.path.join(tmp, "ancillary.pdf")))
    order.append(("Parameters", params))
    # the whole chain first, one page per window: it is the overview every
    # other figure is a detail of
    order.append(("The rows of the fit: nights and BERV bundles",
                  os.path.join(tmp, "bundles.pdf")))
    order.append(("The sequence, step by step", os.path.join(tmp, "sequence.pdf")))
    order.append(("Variance", os.path.join(args.outdir, "variance.pdf")))
    # not the basis vectors: nine curves over half a million samples is a page
    # nobody can read, and what each component DOES is on the two pages that
    # follow, the coefficients against time and against everything measurable
    order.append(("Coefficients against time",
                  os.path.join(args.outdir, "coefficients_vs_time.pdf")))
    order.append(("Correlations", os.path.join(args.outdir, "correlations.pdf")))
    order.append(("Coefficient periodogram", os.path.join(tmp, "periodogram.pdf")))
    order.append(("Weight spectrum", os.path.join(tmp, "weights.pdf")))
    # no "one exposure, before and after" pages any more: the sequence's own
    # flux panel shows exposures before and after, in every window
    order.extend(per_window)

    page = 0
    bound = 0
    broke = [f for f in failures if not (len(f) > 2 and f[2])]
    skipped = [f for f in failures if len(f) > 2 and f[2]]
    if PdfWriter is None:
        log("pypdf is not installed (pip install pypdf): the figures are"
            " not bound into %s, and the report written by LaTeX holds"
            " them all" % out, "warn")
    else:
        writer = PdfWriter()
        for title, path in order:
            if not os.path.exists(path):
                continue
            try:
                reader = PdfReader(path)
            except Exception as exc:                              # noqa: BLE001
                failures.append((path, str(exc)))
                continue
            writer.add_outline_item(title, page)
            for p in reader.pages:
                writer.add_page(p)
                page += 1
            bound += 1
        broke = [f for f in failures if not (len(f) > 2 and f[2])]
        skipped = [f for f in failures if len(f) > 2 and f[2]]
        if failures:
            note = os.path.join(tmp, "_failures.pdf")
            body = []
            if broke:
                body.append("FAILED\n" + "\n\n".join("%s\n    %s" % (c, e)
                                                      for c, e, *_ in broke))
            if skipped:
                # not errors: a figure deciding it had nothing to draw there. Said
                # out loud all the same, because a page that is not in the report is
                # not something anybody notices missing.
                body.append("NOT DRAWN, AND WHY\n"
                            + "\n\n".join("%s\n    %s" % (c, e)
                                           for c, e, *_ in skipped))
            with PdfPages(note) as pdf:
                text_pages(pdf, "What did not build", "\n\n".join(body))
            reader = PdfReader(note)
            writer.add_outline_item("What did not build", page)
            for p in reader.pages:
                writer.add_page(p)
                page += 1

        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as fh:
            writer.write(fh)

    # The report's own copy of every figure, and what it needs to be written
    # again after LBL, when the velocities exist (texreport). Taken before the
    # loose figures below are removed, since some of them are among these.
    from .. import texreport
    folder = os.path.join(args.outdir, "report")
    old = os.path.join(folder, "figures")
    clear_figures(old)
    texreport.write_manifest(folder, {
        "object": obj, "tag": tag, "cube": args.cube,
        "windows": [str(w) for w in args.windows],
        "figures": texreport.keep_figures(order, folder),
        "failures": [{"what": f[0], "why": f[1],
                      "skipped": bool(len(f) > 2 and f[2])} for f in failures],
        "run": [[k, str(v)] for k, v in summary_rows(config, args, fit)
                if str(v).strip() not in ("[]", "")],
    })

    # A run leaves ONE document. The figures the fit itself wrote into the
    # output directory are now pages of it, and leaving them beside it is how
    # a directory fills with forty files that disagree about which run they
    # came from. The CSVs stay: they are data, not pages.
    if not args.keep:
        removed = 0
        for _title, path in order:
            if (path.startswith(os.path.abspath(args.outdir))
                    or os.path.dirname(os.path.abspath(path))
                    == os.path.abspath(args.outdir)) and path != out:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        if removed:
            log("  folded %d loose figures into the bundle and removed them"
                  % removed)
    if args.keep:
        keep = os.path.join(args.outdir, "figures")
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(tmp, keep)
        log("  individual figures kept in %s" % keep)
    shutil.rmtree(tmp, ignore_errors=True)
    if PdfWriter is None:
        log("figures drawn and kept for the report in %s, not bound (no pypdf)"
            % folder)
        return None
    log("wrote %s: %d pages from %d figures%s%s"
          % (out, page, bound,
             ", %d failed" % len(broke) if broke else "",
             ", %d window%s not drawn" % (len(skipped), "" if len(skipped) == 1
                                          else "s") if skipped else ""))
    return None


if __name__ == "__main__":
    main()

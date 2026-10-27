#!/usr/bin/env python
"""A run's report as a document: LaTeX, a table of contents, sections, numbers.

    python -m lbl.pca2d.texreport --run <run folder>       # write it again, now
    python -m lbl.pca2d.texreport --run <run folder> --lbl-dir <LBL tree>

The report a run left was the figures it drew, bound one after another with a
page of monospace text in front: a compilation of matplotlib outputs, which is
what it looked like (2026-09-16). This writes a report instead, from
templates/report.tex:

    abstract          what was run and what it did to the velocities, in words
    lists             the contents, and every figure by a short name
    1 Summary         the gains and losses of every star, one table each,
                      every row saying which way it went
    2 Velocities      per star: the time series (points, no lines), the same
                      points against BERV, d2v as an activity indicator, what
                      the correction changed, and the periodograms
    3 The run         every setting, the command, the code, the fit's numbers
    4 The correction  the figures of the figures stage, each with a caption
    5 Not drawn       what did not build, and why
    A Parameters      the resolved configuration, verbatim

It is written twice in a run's life: at the figures stage, when there are no
velocities yet, and after LBL, when there are. Both times from what is on disk:
the figures the figures stage keeps in <run>/report/figures, listed in
<run>/report/manifest.json, and LBL's rdb files. A run from before the report
existed has neither folder nor manifest; its bound PDF is cut along its own
bookmarks instead, so `--run` works on it too.

Nothing here is fatal to a run. Without pdflatex, or when it fails, the bound
PDF the figures stage wrote stays the report and the run says why.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FuncFormatter, MaxNLocator

from . import bervbias
from .lblscan import AFTER, BEFORE, amplitude_at, nightly, rdb_rows, velocity_stats
from .logger import log

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "report.tex")
MANIFEST = "manifest.json"
#: the bound PDF's own text sections. They are written again here, as text,
#: so they are not taken as figures when an old bundle is cut up
TEXT_SECTIONS = ("Front matter", "Parameters", "What did not build",
                 "The velocities LBL measured")
#: where a TeX installation puts pdflatex when the window's PATH does not
#: know it: a program started from the Dock does not read the shell's profile
TEX_HOMES = ("/Library/TeX/texbin", "/usr/texbin", "/opt/homebrew/bin",
             "/usr/local/bin", "/usr/bin")
#: ink, for everything that is not data (dataviz: text never wears a series
#: colour); the two series colours are lblscan's, validated as a pair
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#e3e3e3"
#: a figure is drawn at the width it is printed at, so its text prints at the
#: size it was written at: A4 less two 2.2 cm margins
WIDTH = 6.6
#: how the report's figures write their text. Matplotlib's default is a
#: Type 3 font and a Unicode minus sign, and a viewer showed the amp axis of
#: the posterior as 20, 10, 0, 10, 20 (2026-09-16): the minus signs were in
#: the drawing but not in the PDF's text. Embedded TrueType, and the minus
#: sign every keyboard has
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}
#: tick labels that carry their sign, for the axes where the sign is the point
SIGNED = FuncFormatter(lambda value, _pos: "0" if abs(value) < 1e-9
                       else "%+g" % value)

#: what each figure of the figures stage shows, said under it. Keyed by the
#: bundle's own section titles
CAPTIONS = {
    "Every quantity of the correlations, against time":
        "Every quantity the correlations are measured against, over the"
        " campaign: the conditions the observer components are compared with.",
    "The rows of the fit: nights and BERV bundles":
        "Which exposures each row of the fit holds. Top: every night, each"
        " exposure at its BERV less the night's median; consecutive bundles of"
        " a night alternate in colour, and the dashed lines are the widest"
        " range the bundle limit allows. Bottom: the nights that were cut,"
        " BERV against time, a vertical line at each cut.",
    "The sequence, step by step":
        "The whole chain in one window: the spectra as read, the fitted"
        " observer model, what was divided out, and what is left. The thick"
        " black line is where a feature fixed in the observer's frame falls in"
        " each row: the residuals of the observer's frame run parallel to it,"
        " the star's lines stand vertical.",
    "Variance":
        "The weighted variance each component removes, star and observer"
        " blocks apart.",
    "Coefficients against time":
        "Every component's coefficient per exposure, over the campaign.",
    "Correlations":
        "Every coefficient against every measured quantity, the strongest"
        " correlations first.",
    "Coefficient periodogram":
        "Periodogram of every coefficient. A component varying at a known"
        " planet's period would subtract that planet: it is marked when the"
        " configuration gives the period.",
    "Weight spectrum":
        "The weights the fit gave each sample, which is where the model is"
        " trusted and where it is not.",
}


# ================================================================= LaTeX ===
_TEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
        "&": r"\&", "#": r"\#", "^": r"\^{}", "_": r"\_", "%": r"\%",
        "~": r"\textasciitilde{}", "<": r"\textless{}", ">": r"\textgreater{}",
        "|": r"\textbar{}"}
#: characters inputenc does not take in text mode, and what they become. The
#: dashes become hyphens on purpose: no em dash in anything written here
_UNICODE = {"\u00b2": r"\textsuperscript{2}", "\u00b3": r"\textsuperscript{3}",
            "\u03c3": r"$\sigma$", "\u00d7": r"$\times$", "\u00b1": r"$\pm$",
            "\u2265": r"$\geq$", "\u2264": r"$\leq$", "\u2013": "-",
            "\u2014": "-", "\u2212": "-", "\u2026": r"\ldots{}",
            "\u00b0": r"$^\circ$", "\u2192": r"$\rightarrow$",
            "\u0394": r"$\Delta$", "\u00b7": r"$\cdot$", "\u2019": "'",
            "\u2018": "`", "\u201c": "``", "\u201d": "''", "\u00a0": "~"}


#: a terminal's colour and cursor sequences, the whole of each
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def tex(text) -> str:
    """`text` as LaTeX prints it, whatever it holds.

    Object names carry underscores (SMETHELLS_20), run names plus signs and
    paths every character there is, and one unescaped one stops the document.
    A character nothing here knows becomes a question mark rather than an
    error: a report with one odd glyph is still a report. Terminal colours
    and other control characters are dropped: a skipped figure's reason came
    from a coloured log line, and its escape stopped pdflatex on a joint run
    (2026-09-17, "Unicode character ^^[ (U+001B)").
    """
    out = []
    for ch in ANSI_ESCAPE.sub("", str(text)):
        if ch in "\t\n":
            out.append(" " if ch == "\t" else ch)
        elif ord(ch) < 32 or 127 <= ord(ch) < 160:
            continue
        elif ch in _TEX:
            out.append(_TEX[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        elif ord(ch) < 256:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def tex_break(text) -> str:
    """tex(), with a place to break after every _ / + - . in it.

    A run's suffix (_PCA2D_{tag}_GL406_3b7955) is one word to LaTeX, and one
    word wider than its column runs over the next one.
    """
    out = tex(text)
    for mark in (r"\_", "/", "+", "-", "."):
        out = out.replace(mark, mark + r"\allowbreak{}")
    return out


def find_pdflatex():
    """The pdflatex this machine has, or None."""
    found = shutil.which("pdflatex")
    if found:
        return found
    for home in TEX_HOMES:
        path = os.path.join(home, "pdflatex")
        if os.access(path, os.X_OK):
            return path
    for path in sorted(glob.glob("/usr/local/texlive/*/bin/*/pdflatex"),
                       reverse=True):
        if os.access(path, os.X_OK):
            return path
    return None


def slug(text) -> str:
    """A file name LaTeX reads without complaint."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower() or "x"


def number(value, digits=2, unit=""):
    """A number as the tables print it, or a dash-free 'n/a'."""
    if value is None or not np.isfinite(value):
        return "n/a"
    text = "%.*f" % (digits, value)
    return text + (" " + unit if unit else "")


def wrap_lines(text, width=96):
    """Hard-wrapped, since verbatim never breaks a line itself."""
    out = []
    for line in str(text).splitlines():
        while len(line) > width:
            out.append(line[:width])
            line = "    " + line[width:]
        out.append(line)
    return "\n".join(out) + "\n"


# ============================================================ the figures ===
def pdf_pages(path) -> int:
    """How many pages a figure's PDF has: by pypdf, or, without it, by
    counting its page objects, which is exact for what matplotlib writes."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            with open(path, "rb") as handle:
                return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])",
                                      handle.read()))
        except OSError:
            return 0
    try:
        return len(PdfReader(path).pages)
    except Exception:                                          # noqa: BLE001
        return 0


def read_manifest(folder):
    path = os.path.join(folder, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def write_manifest(folder, manifest):
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, MANIFEST + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(manifest, handle, indent=1)
    os.replace(tmp, os.path.join(folder, MANIFEST))


def keep_figures(order, folder):
    """Copy the figures stage's figures where the report can find them again.

    `order` is the bundle's [(section title, path)]. Returns the manifest's
    figure list, in that order, text sections left out.
    """
    figures = os.path.join(folder, "figures")
    os.makedirs(figures, exist_ok=True)
    kept = []
    for i, (title, path) in enumerate(order):
        if title in TEXT_SECTIONS or not os.path.exists(path):
            continue
        name = "%02d-%s.pdf" % (i, slug(title))
        shutil.copyfile(path, os.path.join(figures, name))
        kept.append({"title": title, "file": "figures/" + name,
                     "pages": pdf_pages(path)})
    return kept


def split_bundle(bundle, folder):
    """Cut a bound PDF from before the report along its own bookmarks.

    Returns the manifest's figure list. The bookmarks were written with page
    indices, so each section runs from its own index to the next one's.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(bundle)
    refs = {page.indirect_reference.idnum: i
            for i, page in enumerate(reader.pages)}
    marks = []
    for item in reader.outline:
        if isinstance(item, list):
            continue
        page = item.get("/Page")
        index = (int(page) if isinstance(page, (int, float))
                 else refs.get(getattr(page, "idnum", None)))
        if index is None:
            try:
                index = reader.get_destination_page_number(item)
            except Exception:                                   # noqa: BLE001
                index = None
        if index is not None:
            marks.append((int(index), str(item.title)))
    marks.sort()
    figures = os.path.join(folder, "figures")
    os.makedirs(figures, exist_ok=True)
    kept = []
    for n, (start, title) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(reader.pages)
        if title in TEXT_SECTIONS or end <= start:
            continue
        writer = PdfWriter()
        for page in reader.pages[start:end]:
            writer.add_page(page)
        name = "%02d-%s.pdf" % (n, slug(title))
        with open(os.path.join(figures, name), "wb") as handle:
            writer.write(handle)
        kept.append({"title": title, "file": "figures/" + name,
                     "pages": end - start})
    return kept


# ========================================================= the velocities ===
def rdb_file(tree, name, comparison=None):
    """LBL's velocities of `name` measured against `comparison`'s template
    (its own when None): lblrdb/lbl_<name>_<comparison>.rdb."""
    return os.path.join(tree, "lblrdb", "lbl_%s_%s.rdb"
                        % (name, comparison or name))


def comparisons(config, stars):
    """{star: the template LBL measured it against}. Another star's when the
    fit took one as the star's spectrum (twoframe.star_templates, one entry
    per member in the members' order, LBL_Template_<name>_<instrument>.fits),
    else the star's own. Read as the star's own, a star measured against a
    vetted template had no velocities in its report: lbl_TOI3750_TOI3750.rdb
    for lbl_TOI3750_VETTED_GL846.rdb (2026-10-24)."""
    entries = list((config.get("twoframe") or {}).get("star_templates") or [])
    instrument = str((config.get("input") or {}).get("instrument") or "")
    tail = "_%s.fits" % instrument.lower()
    out = {}
    for i, star in enumerate(stars):
        entry = entries[i] if i < len(entries) else None
        name = None
        if isinstance(entry, dict) and entry.get("file"):
            base = os.path.basename(str(entry["file"]))
            if base.startswith("LBL_Template_"):
                base = base[len("LBL_Template_"):]
                if instrument and base.endswith(tail):
                    name = base[:-len(tail)]
                else:
                    name = re.sub(r"_[^_]+\.fits$", "", base)
        out[star] = name or star
    return out


def column(table, *names):
    """A column of an rdb as floats, whatever its case, or None."""
    lower = {c.strip().lower(): c for c in table.colnames}
    for name in names:
        if name.lower() in lower:
            values = np.asarray(table[lower[name.lower()]], float)
            return values if np.isfinite(values).any() else None
    return None


def text_column(table, name):
    """A column of an rdb as strings, or None."""
    lower = {c.strip().lower(): c for c in table.colnames}
    if name.lower() not in lower:
        return None
    return np.asarray([str(x).strip() for x in table[lower[name.lower()]]])


def snr_column(table):
    """(name, values) of the SNR the headers carry, or (None, None).

    APERO writes the extracted SNR of one order, EXTSN060 for NIRPS and
    EXTSN035 for SPIRou, and that measured one is quoted first; a column
    called SNR only when there is none. Never SNRGOAL: SPIRou's headers carry
    the SNR the observation was asked to reach, and a GL725B report quoted
    its 150.0 as the median SNR.
    """
    names = [n.strip() for n in table.colnames]
    for pattern in (r"EXTSN\d+", r"SNR(?!GOAL)\w*"):
        for name in names:
            if re.fullmatch(pattern, name, re.IGNORECASE):
                values = column(table, name)
                if values is not None:
                    return name, values
    return None, None


def vtot_of(run):
    """V_tot = vrad/1000 - BERV of a series, in km/s, or None without BERV.

    The separation of the star's lines from the telluric ones, which is what
    the bias follows (bervbias.total_velocity).
    """
    if run.get("vtot") is not None:
        return run["vtot"]
    if run.get("berv") is None:
        return None
    return bervbias.total_velocity(run["v"], run["berv"])


def load(tree, name, label, comparison=None):
    """One LBL object as the figures take it, or None when it has no rdb."""
    path = rdb_file(tree, name, comparison)
    if not os.path.exists(path):
        return None
    t, v, e, table = rdb_rows(path)
    if t.size < 3:
        return None
    # LBL's temperature projection, DTEMP<T>, when the run asked for one
    dtemp = next((c for c in table.colnames if re.fullmatch(r"DTEMP\d+", c)),
                 None)
    berv = column(table, "BERV")
    snr_name, snr = snr_column(table)
    return {"label": label, "name": name, "path": path, "t": t, "v": v,
            "e": e, "berv": berv,
            "vtot": (bervbias.total_velocity(v, berv)
                     if berv is not None else None),
            "snr": snr, "snr_name": snr_name,
            "date_obs": text_column(table, "DATE-OBS"),
            "d2v": column(table, "d2v"), "sd2v": column(table, "sd2v"),
            "dtemp": column(table, dtemp) if dtemp else None,
            "sdtemp": column(table, "s" + dtemp) if dtemp else None,
            "dtemp_name": dtemp, "mtime": os.path.getmtime(path)}


def common(before, after):
    """Both series cut to the exposures they share, in time order."""
    key = lambda t: np.round(t, 6)                            # noqa: E731
    shared = np.intersect1d(key(before["t"]), key(after["t"]))
    out = []
    for run in (before, after):
        keep = np.isin(key(run["t"]), shared)
        order = np.argsort(run["t"][keep])
        cut = dict(run)
        for name in ("t", "v", "e", "berv", "vtot", "snr", "date_obs", "d2v",
                     "sd2v", "dtemp", "sdtemp"):
            if cut.get(name) is not None:
                cut[name] = np.asarray(run[name])[keep][order]
        out.append(cut)
    return out[0], out[1]


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def inliers(x, nsig=5.0):
    """Where `x` is within nsig robust sigmas of its median."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return ok
    sig = robust_sigma(x[ok])
    if not np.isfinite(sig) or sig == 0:
        return ok
    return ok & (np.abs(x - np.median(x[ok])) < nsig * sig)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def binned(x, y, width=2.0, least=3):
    """(centres, medians, errors of the medians) of y in bins of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < least:
        return np.array([]), np.array([]), np.array([])
    edges = np.arange(np.floor(x[ok].min() / width) * width,
                      x[ok].max() + width, width)
    centres, medians, errors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = ok & (x >= lo) & (x < hi)
        if inside.sum() >= least:
            centres.append(0.5 * (lo + hi))
            medians.append(np.median(y[inside]))
            errors.append(1.2533 * robust_sigma(y[inside])
                          / np.sqrt(inside.sum()))
    return np.array(centres), np.array(medians), np.array(errors)


#: false-alarm probabilities whose power levels the periodograms draw
FAP_LEVELS = (1e-2, 1e-3, 1e-4)


def periodogram(t, y, e, pmin=1.1, samples=6000):
    """(periods, power, best period, its power, false-alarm probability,
    the powers of FAP_LEVELS).

    The false alarms are Baluev's (2008) approximation over the frequencies
    searched, astropy's default: the power a peak anywhere in them needs to
    reach that probability.
    """
    from astropy.timeseries import LombScargle

    t, y, e = (np.asarray(a, float) for a in (t, y, e))
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    baseline = float(np.ptp(t[ok])) if ok.sum() > 3 else 0.0
    if baseline <= 2 * pmin or ok.sum() < 8:
        return None
    fmin, fmax = 1.0 / baseline, 1.0 / pmin
    freq = np.linspace(fmin, fmax, samples)
    model = LombScargle(t[ok], y[ok], e[ok])
    power = model.power(freq)
    best = int(np.nanargmax(power))
    try:
        fap = float(model.false_alarm_probability(
            power[best], minimum_frequency=fmin, maximum_frequency=fmax))
    except Exception:                                          # noqa: BLE001
        fap = np.nan
    try:
        levels = np.asarray(model.false_alarm_level(
            FAP_LEVELS, minimum_frequency=fmin, maximum_frequency=fmax), float)
    except Exception:                                          # noqa: BLE001
        levels = np.full(len(FAP_LEVELS), np.nan)
    return (1.0 / freq, power, float(1.0 / freq[best]), float(power[best]),
            fap, levels)


def bias_row(fitted):
    """The table's view of one BERV-bias fit."""
    if fitted is None:
        return {}
    peak, lo, hi = fitted["peak"]
    return {"bias": fitted, "bias_peak": float(peak),
            "bias_err": float(0.5 * (hi - lo)),
            "bias_significance": fitted["significance"],
            "bias_detected": fitted["detected"],
            "bias_upper": fitted["upper"],
            "bias_fwhm": float(fitted["fwhm"][0]),
            "bias_fwhm_err": float(0.5 * (fitted["fwhm"][2]
                                          - fitted["fwhm"][1])),
            "bias_amp_fwhm_r": fitted["amp_fwhm_r"],
            "bias_amp_slope_r": fitted.get("amp_slope_r", np.nan),
            "bias_p_positive": fitted["p_positive"],
            "bias_c": float(fitted["c"][0]),
            "bias_slope": float(fitted["slope"][0]),
            "bias_slope_err": float(0.5 * (fitted["slope"][2]
                                           - fitted["slope"][1])),
            "bias_drift": bool(fitted.get("drift")),
            "bias_delta_bic": fitted.get("delta_bic", np.nan),
            "jitter": float(fitted["jitter"][0])}


def star_numbers(before, after, planets=(), seed=0):
    """Everything the summary table says about one star, both ways."""
    out = {"n": int(before["t"].size),
           "nights": int(velocity_stats(after["t"], after["v"],
                                        after["e"])["nights"])}
    for key, run in (("before", before), ("after", after)):
        s = velocity_stats(run["t"], run["v"], run["e"])
        row = {"rms": s["rms"], "robust": s["robust"],
               "nightly_rms": s["nightly_rms"],
               "median_error": s["median_error"]}
        vtot = vtot_of(run)
        if vtot is not None:
            # against V_tot, the separation of the star's lines from the
            # tellurics, which is what the bias follows; BERV alone is that
            # only for a star at rest
            _, medians, _ = binned(vtot, run["v"] - np.median(run["v"]))
            row.update(berv_binned=float(np.std(medians))
                       if medians.size >= 3 else np.nan)
            # the bias's own shape, by MCMC; a straight line has no meaning
            # the DC level free, and free to drift over the campaign
            row.update(bias_row(bervbias.fit(vtot, run["v"], run["e"],
                                             t=run["t"], seed=seed)))
            row["bias_near"] = int(np.sum(np.abs(vtot) < bervbias.FWHM_MAX))
        if run.get("d2v") is not None:
            good = inliers(run["d2v"])
            row.update(d2v_sigma=robust_sigma(run["d2v"]),
                       d2v_error=float(np.nanmedian(run["sd2v"]))
                       if run.get("sd2v") is not None else np.nan,
                       d2v_r=pearson(run["d2v"][good], run["v"][good]))
        if run.get("dtemp") is not None:
            good = inliers(run["dtemp"])
            _, medians, _ = binned(vtot, run["dtemp"] - np.nanmedian(
                run["dtemp"])) if vtot is not None else (0, [], 0)
            row.update(dtemp_name=run["dtemp_name"],
                       dtemp_sigma=robust_sigma(run["dtemp"]),
                       dtemp_error=float(np.nanmedian(run["sdtemp"]))
                       if run.get("sdtemp") is not None else np.nan,
                       dtemp_r=pearson(run["dtemp"][good], run["v"][good]),
                       dtemp_berv_binned=float(np.std(medians))
                       if len(medians) >= 3 else np.nan)
        found = periodogram(run["t"], run["v"], run["e"])
        if found:
            row.update(peak_period=found[2], peak_power=found[3],
                       peak_fap=found[4])
        if run.get("d2v") is not None:
            good = inliers(run["d2v"])
            err = run.get("sd2v")
            spot = periodogram(run["t"][good], run["d2v"][good],
                               (err if err is not None
                                else np.ones_like(run["d2v"]))[good])
            if spot:
                row.update(d2v_peak_period=spot[2], d2v_peak_fap=spot[4])
        row["planets"] = [amplitude_at(run["t"], for_phase(run, p), run["e"],
                                       p) for p in planets]
        row["planet_bic"] = [planet_bic(run["t"], for_phase(run, p),
                                        run["e"], p) for p in planets]
        out[key] = row
    change = after["v"] - before["v"]
    out["change_rms"] = float(np.std(change - np.median(change)))
    rb, ra = out["before"]["rms"], out["after"]["rms"]
    out["removed"] = float(np.sqrt(rb ** 2 - ra ** 2)) if rb > ra else np.nan
    out["planet_periods"] = [float(p) for p in planets]
    out["baseline"] = float(np.ptp(after["t"]))
    out["observations"] = observations(before, after, out["nights"])
    return out


def utc_dates(run):
    """The UT date of each exposure: DATE-OBS's, or the rjd's."""
    if run.get("date_obs") is not None:
        return np.asarray([d[:10] for d in run["date_obs"]])
    from astropy.time import Time

    return np.asarray(Time(np.asarray(run["t"], float) + 2400000.0,
                           format="jd").isot.astype("U10"))


def day_of_year(dates):
    """Day of a common year, 1 to 365, of 'YYYY-MM-DD' dates; 29 Feb is 28."""
    out = []
    for date in dates:
        month, day = int(date[5:7]), int(date[8:10])
        if (month, day) == (2, 29):
            day = 28
        out.append(datetime.date(2001, month, day).timetuple().tm_yday)
    return np.asarray(out, int)


def calendar_windows(dates, gap=15):
    """The calendar dates, year left out, that `dates` fall on, as windows.

    V_tot follows the BERV, which repeats every year, so the dates a star's
    lines sit on the tellurics are the same every year. The days of the year
    are joined into one window unless `gap` days or more separate them, and
    a window may run across the new year. Each is (first day, last day), as
    days of a common year.
    """
    days = np.unique(day_of_year(dates))
    if days.size == 0:
        return []
    # start after the widest gap around the year, so a window running
    # through 31 December is one window
    following = np.roll(days, -1)
    gaps = (following - days) % 365
    gaps[gaps == 0] = 365                          # one day only
    widest = int(np.argmax(gaps))
    order = np.roll(days, -(widest + 1))
    windows, first = [], order[0]
    for previous, day in zip(order[:-1], order[1:]):
        if (day - previous) % 365 >= gap:
            windows.append((int(first), int(previous)))
            first = day
    windows.append((int(first), int(order[-1])))
    return windows


def in_windows(dates, windows):
    """Whether each date's day of the year is inside one of the windows."""
    days = day_of_year(dates)
    inside = np.zeros(days.size, bool)
    for first, last in windows:
        inside |= ((days - first) % 365) <= ((last - first) % 365)
    return inside


def calendar_day(day):
    """'17 Jun' for a day of a common year."""
    return (datetime.date(2001, 1, 1)
            + datetime.timedelta(days=int(day) - 1)).strftime("%-d %b")


def observations(before, after, nights):
    """What the headers say about the exposures both series share."""
    dates = utc_dates(after)
    out = {"n": int(after["t"].size), "nights": int(nights),
           "first": str(dates[0]), "last": str(dates[-1]),
           "span": float(np.ptp(after["t"]))}
    run = before if before.get("snr") is not None else after
    if run.get("snr") is not None:
        out["snr_name"] = run["snr_name"]
        out["snr"] = float(np.nanmedian(run["snr"]))
    if after.get("berv") is not None:
        berv = np.asarray(after["berv"], float)
        out["berv"] = (float(np.nanmin(berv)), float(np.nanmax(berv)))
    vtot = vtot_of(after)
    if vtot is not None:
        out["systemic"] = float(np.median(after["v"]) / 1000.0)
        out["vtot"] = (float(np.nanmin(vtot)), float(np.nanmax(vtot)))
        close = np.abs(vtot) < bervbias.CLOSE_KMS
        out["close_n"] = int(close.sum())
        # the calendar dates, every year alike, and every exposure that
        # falls on them, whatever its own V_tot
        out["windows"] = calendar_windows(dates[close])
        out["window_n"] = int(in_windows(dates, out["windows"]).sum())
    return out


def same_period(p1, p2, baseline):
    """Whether two periods are one peak: within a resolution element."""
    if not (p1 and p2 and baseline and np.isfinite(p1) and np.isfinite(p2)):
        return False
    return abs(1.0 / p1 - 1.0 / p2) < 1.0 / baseline


def verdict(before, after, better="lower", tolerance=0.02):
    """'gain', 'loss' or 'same' for one row of the table."""
    if not (np.isfinite(before) and np.isfinite(after)):
        return "same"
    if better == "lower":
        if after < before * (1 - tolerance):
            return "gain"
        if after > before * (1 + tolerance):
            return "loss"
        return "same"
    # closer to zero is better: a correlation, a slope
    if abs(after) < abs(before) - tolerance:
        return "gain"
    if abs(after) > abs(before) + tolerance:
        return "loss"
    return "same"


# --------------------------------------------------------------- drawing --
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5, labelcolor=INK)


def _limits(*arrays, pad=1.25, q=99.0):
    values = np.concatenate([np.abs(np.asarray(a, float)) for a in arrays])
    values = values[np.isfinite(values)]
    top = float(np.percentile(values, q)) * pad if values.size else 1.0
    return (-top, top) if top > 0 else (-1.0, 1.0)


def _points(ax, x, y, e, colour, label, size=3.2, alpha=0.55, zorder=2):
    """Exposures as points: a marker and its bar, never a line between."""
    ax.errorbar(x, y, yerr=e, fmt="o", ls="none", ms=size, mfc=colour,
                mec="white", mew=0.4, ecolor=colour, elinewidth=0.5,
                alpha=alpha, capsize=0, label=label, zorder=zorder)


#: inches added above a figure for the star's name
NAME_BAND = 0.34


def _named(fig, name):
    """The star's name above everything in the figure.

    A joint report holds the same figures once per star, and a figure seen
    on its own, or on a page whose caption is on the next, said nothing of
    whose velocities it drew (2026-09-17). The figure is made taller by a
    band, everything already in it keeps its size and moves down, and the
    name goes in the band.
    """
    width, height = fig.get_size_inches()
    total = height + NAME_BAND
    scale = height / total
    fig.set_size_inches(width, total)
    for ax in fig.axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 * scale, box.width,
                         box.height * scale])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, y * scale))
    fig.text(0.012, 1.0 - 0.5 * NAME_BAND / total, name, ha="left",
             va="center", fontsize=12, fontweight="bold", color=INK)


def _save(fig, path, name=None):
    """Write the figure, with the star's name above it when one is given."""
    if name:
        _named(fig, name)
    with plt.rc_context(STYLE):
        fig.savefig(path)
    plt.close(fig)
    return path


def straight_line(t, v, e):
    """(slope in m/s per year, the line at each t): a straight line in time,
    weighted by the error bars, about the middle of the span."""
    t, v, e = (np.asarray(x, float) for x in (t, v, e))
    mid = 0.5 * (t.min() + t.max())
    tau = (t - mid) / 365.25
    if np.ptp(tau) <= 0:
        return 0.0, np.full(t.size, np.median(v))
    slope, level = np.polyfit(tau, v, 1, w=1.0 / e)
    return float(slope), level + slope * tau


def _histogram(ax, values, bins, colour, label):
    """One series' spread, as a step outline over a light fill."""
    ax.hist(values, bins=bins, color=colour, alpha=0.18, lw=0)
    ax.hist(values, bins=bins, color=colour, histtype="step", lw=1.3,
            label=label)


def figure_time(before, after, path):
    """The velocities over the campaign, each less its median, with the
    straight line in time drawn over them; below, their spread, delivered
    against corrected, as they are and with that line taken out."""
    fig = plt.figure(figsize=(WIDTH, 8.0))
    cells = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85],
                             hspace=0.42, wspace=0.08)
    axes = [fig.add_subplot(cells[0, :])]
    axes.append(fig.add_subplot(cells[1, :], sharex=axes[0], sharey=axes[0]))
    lim = _limits(before["v"] - np.median(before["v"]),
                  after["v"] - np.median(after["v"]))
    spreads = []
    for ax, run, colour in zip(axes, (before, after), (BEFORE, AFTER)):
        middle = np.median(run["v"])
        nt = nightly(run["t"], run["v"], run["e"])
        # one exposure a night makes the nightly means the exposures again,
        # drawn over them: one layer then, not two that hide each other
        if nt.shape[1] < 0.8 * run["t"].size:
            _points(ax, run["t"], run["v"] - middle, run["e"], colour,
                    "exposures")
            label = "nightly means"
        else:
            label = "exposures (about one a night)"
        ax.errorbar(nt[0], nt[1] - middle, yerr=nt[2], fmt="o", ls="none",
                    ms=5.0, mfc=colour, mec=INK, mew=0.5, ecolor=INK,
                    elinewidth=0.6, capsize=0, label=label, zorder=3)
        slope, line = straight_line(run["t"], run["v"], run["e"])
        order = np.argsort(run["t"])
        ax.plot(run["t"][order], line[order] - middle, color=INK, lw=1.0,
                ls="--", zorder=4,
                label="straight line, %+.2f m/s/yr" % slope)
        s = velocity_stats(run["t"], run["v"], run["e"])
        ax.set_title("%s: rms %.2f m/s, nightly rms %.2f m/s, median error"
                     " %.2f m/s" % (run["label"], s["rms"], s["nightly_rms"],
                                    s["median_error"]),
                     fontsize=8.5, color=INK, loc="left")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel("velocity - median (m/s)\nmedian = %.1f m/s" % middle,
                      fontsize=8, color=INK)
        ax.set_ylim(*lim)
        ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)
        _style(ax)
        spreads.append((run["v"] - middle, run["v"] - line, slope, colour,
                        run["label"]))
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[-1].set_xlabel("RJD (BJD - 2400000)", fontsize=8, color=INK)

    # the spread, before and after the correction, and before and after the
    # straight line: one set of bins for all four, so they compare
    bins = np.linspace(lim[0], lim[1], 41)
    left = fig.add_subplot(cells[2, 0])
    right = fig.add_subplot(cells[2, 1], sharex=left, sharey=left)
    for raw, flat, slope, colour, name in spreads:
        _histogram(left, raw, bins, colour, "%s: rms %.2f, robust %.2f"
                   % (name, np.std(raw), robust_sigma(raw)))
        _histogram(right, flat, bins, colour, "%s: rms %.2f, robust %.2f"
                   % (name, np.std(flat), robust_sigma(flat)))
    left.set_title("as measured, less the median", fontsize=8, color=INK,
                   loc="left")
    right.set_title("less the straight line in time", fontsize=8, color=INK,
                    loc="left")
    left.set_ylabel("exposures", fontsize=8, color=INK)
    for ax in (left, right):
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("velocity (m/s)", fontsize=8, color=INK)
        ax.set_xlim(*lim)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        _style(ax)
    plt.setp(right.get_yticklabels(), visible=False)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.96)
    return _save(fig, path, before.get("star"))


def _bias_band(ax, fitted, grid, colour, label=None, offset=0.0):
    """The fitted bias: the band its 16th-84th percentiles span over `grid`,
    and its median when the bias is detected.

    Not the median of one that is not: the median of curves of every width
    the prior allows is a spike nobody fitted, and it reads as a feature.
    """
    lo, mid, hi = bervbias.envelope(fitted, grid)
    ax.fill_between(grid, lo + offset, hi + offset, color=colour,
                    alpha=0.22 if fitted["detected"] else 0.14, lw=0,
                    zorder=3, label=None if fitted["detected"] else label)
    if fitted["detected"]:
        ax.plot(grid, mid + offset, color=colour, lw=1.6, zorder=4,
                label=label)


VTOT_LABEL = "$V_\\mathrm{tot}$ = vrad - BERV (km/s)"


def _close_band(ax):
    """|V_tot| < 4 km/s, where the star's lines sit on the tellurics."""
    ax.axvspan(-bervbias.CLOSE_KMS, bervbias.CLOSE_KMS, color=GRID, alpha=0.6,
               lw=0, zorder=0)


def without_dc(run, fitted):
    """The velocities less the fitted DC level, drift included."""
    if fitted.get("drift"):
        tau = (np.asarray(run["t"], float) - fitted["t_mid"]) / 365.25
        return run["v"] - (fitted["c"][0] + fitted["slope"][0] * tau)
    return run["v"] - fitted["c"][0]


def figure_berv(before, after, path):
    """The velocities against V_tot with the fitted bias and its 1 sigma
    envelope, each series on its own, then both envelopes on one axis."""
    fb = (before.get("fits") or {})
    vb, va = vtot_of(before), vtot_of(after)
    if vb is None or va is None \
            or fb.get("before") is None or fb.get("after") is None:
        return None
    fits = {"before": fb["before"], "after": fb["after"]}
    fig = plt.figure(figsize=(WIDTH, 6.4))
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    top = [fig.add_subplot(grid_spec[0, 0])]
    top.append(fig.add_subplot(grid_spec[0, 1], sharex=top[0], sharey=top[0]))
    both = fig.add_subplot(grid_spec[1, :], sharex=top[0])
    span = np.concatenate([vb, va])
    grid = np.linspace(np.nanmin(span), np.nanmax(span), 300)
    residuals = []
    for ax, run, x, key, colour in zip(top, (before, after), (vb, va),
                                       ("before", "after"), (BEFORE, AFTER)):
        fitted = fits[key]
        y = without_dc(run, fitted)
        residuals.append(y)
        _close_band(ax)
        _points(ax, x, y, run["e"], colour, "exposures", alpha=0.4)
        centres, medians, errors = binned(x, y)
        if centres.size:
            ax.errorbar(centres, medians, yerr=errors, fmt="s", ls="none",
                        ms=4.5, mfc="white", mec=INK, mew=0.8, ecolor=INK,
                        elinewidth=0.7, capsize=0, zorder=5,
                        label="2 km/s bins, median")
        # in the series' colour, as below: the grey is |V_tot| < 4 km/s
        _bias_band(ax, fitted, grid, colour, label="fitted bias, 1$\\sigma$")
        ax.set_title("%s\n%s" % (run["label"], bervbias.summary(fitted)),
                     fontsize=7, color=INK, loc="left")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel(VTOT_LABEL, fontsize=8, color=INK)
        _style(ax)
    lim = _limits(*residuals)
    top[0].set_ylim(*lim)
    top[0].set_ylabel("velocity - fitted DC level (m/s)", fontsize=8,
                      color=INK)
    top[0].legend(fontsize=6, frameon=False, loc="upper left")
    _close_band(both)
    for run, x, key, colour in ((before, vb, "before", BEFORE),
                                (after, va, "after", AFTER)):
        fitted = fits[key]
        centres, medians, errors = binned(x, without_dc(run, fitted))
        if centres.size:
            both.errorbar(centres, medians, yerr=errors, fmt="s", ls="none",
                          ms=4.5, mfc="white", mec=colour, mew=0.9,
                          ecolor=colour, elinewidth=0.7, capsize=0, zorder=5)
        _bias_band(both, fitted, grid, colour,
                   label="%s: %s" % (run["label"], bervbias.summary(fitted)))
    both.axhline(0, color=MUTED, lw=0.6)
    both.set_xlabel(VTOT_LABEL, fontsize=8, color=INK)
    both.set_ylabel("fitted bias (m/s)", fontsize=8, color=INK)
    # the title is the legend's own: set apart, the two overlapped
    legend = both.legend(fontsize=6.5, frameon=False, loc="lower left",
                         bbox_to_anchor=(0.0, 1.02), ncol=1,
                         borderaxespad=0.0,
                         title="both fits, 1$\\sigma$ envelopes, over the"
                               " binned medians", title_fontsize=7.5)
    legend._legend_box.align = "left"
    _style(both)
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


def amp_range(*fits):
    """One amp axis for every fit shown side by side.

    Each posterior on its own axis made the delivered one look as wide as
    the corrected one, which spans four times more: two panels meant to be
    compared are drawn on the same scale.
    """
    ends = np.array([np.percentile(f["samples"][:, 0], [0.5, 99.5])
                     for f in fits])
    # symmetric about zero: amp is signed, and the sign is part of the answer
    reach = 1.08 * float(np.max(np.abs(ends)))
    reach = reach if reach > 0 else 1.0
    return -reach, reach


def slope_range(*fits):
    """One drift axis for every fit shown side by side, m/s per year."""
    ends = np.array([np.percentile(f["samples"][:, 4], [0.5, 99.5])
                     for f in fits])
    lo, hi = float(ends[:, 0].min()), float(ends[:, 1].max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


#: the posterior mass the shading encloses, 1, 2 and 3 sigma in two
#: dimensions, and how dark each is: the core darkest, so two posteriors laid
#: over each other still show where each one is dense
MASSES = (0.393, 0.865, 0.989)
SHADES = (0.55, 0.3, 0.12)


def _joint(ax, x, y, xr, yr, colour):
    """Two parameters' joint posterior, drawn so another can lie over it:
    filled regions holding 39%, 86% and 99% of it (1, 2 and 3 sigma in two
    dimensions), lighter outwards, on a histogram smoothed by one bin, and
    the 1 and 2 sigma outlines."""
    from scipy.ndimage import gaussian_filter

    counts, xe, ye = np.histogram2d(x, y, bins=50, range=[xr, yr])
    counts = gaussian_filter(counts, 1.0)
    ordered = np.sort(counts.ravel())[::-1]
    if ordered.sum() <= 0:
        return
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = [float(ordered[min(np.searchsorted(cumulative, q),
                                ordered.size - 1)]) for q in MASSES]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    top = float(ordered[0]) * 1.01
    for level, shade in zip(levels, SHADES):
        if 0 < level < top:
            ax.contourf(xc, yc, counts.T, levels=[level, top],
                        colors=[colour], alpha=shade * 0.6)
    lines = sorted({lv for lv in levels[:2] if 0 < lv < top})
    if lines:
        ax.contour(xc, yc, counts.T, levels=lines, colors=[colour],
                   linewidths=[0.8, 1.3][:len(lines)])
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)


def _marginal(ax, x, xr, colour, label=None):
    """One parameter alone: a light fill under a step outline."""
    ax.hist(x, bins=50, range=xr, color=colour, alpha=0.2, lw=0,
            density=True)
    ax.hist(x, bins=50, range=xr, color=colour, histtype="step", lw=1.3,
            density=True, label=label)


def figure_corner(before, after, path):
    """The two posteriors laid over each other, one triangle: the FWHM, the
    amplitude and, when the DC level drifts, its slope, each pair's joint
    posterior below the diagonal and each parameter alone on it, the FWHM
    with its prior."""
    fb = (before.get("fits") or {})
    if fb.get("before") is None or fb.get("after") is None:
        return None
    fits = [(fb["before"], BEFORE, before["label"]),
            (fb["after"], AFTER, after["label"])]
    drift = all(f.get("drift") for f, _c, _l in fits)
    params = [("FWHM (km/s)", 1, (bervbias.FWHM_MIN, bervbias.FWHM_MAX)),
              ("amp ((m/s)/(km/s))", 0, amp_range(fb["before"], fb["after"]))]
    if drift:
        params.append(("drift (m/s/yr)", 4,
                       slope_range(fb["before"], fb["after"])))
    n = len(params)
    fig = plt.figure(figsize=(WIDTH, 0.86 * WIDTH if n == 3 else 0.62 * WIDTH))
    cells = fig.add_gridspec(n, n, wspace=0.08, hspace=0.08)
    for row in range(n):
        for col in range(row + 1):
            ax = fig.add_subplot(cells[row, col])
            xname, xi, xr = params[col]
            if row == col:
                for fitted, colour, label in fits:
                    _marginal(ax, fitted["samples"][:, xi], xr, colour,
                              label=label)
                if xi == 1:
                    # the prior, as a density, for what the data added to it
                    mean, spread = bervbias.FWHM_PRIOR
                    xs = np.linspace(*xr, 200)
                    ax.plot(xs, np.exp(-0.5 * ((xs - mean) / spread) ** 2)
                            / (spread * np.sqrt(2 * np.pi)), color=MUTED,
                            lw=1.0, ls="--", label="FWHM prior")
                ax.set_xlim(*xr)
                ax.set_yticks([])
                if row == 0:
                    ax.legend(fontsize=7.5, frameon=False, loc="upper left",
                              bbox_to_anchor=(1.1, 1.0))
            else:
                yname, yi, yr = params[row]
                # the wider posterior first, so the narrower lies on top
                for fitted, colour, _label in sorted(
                        fits, key=lambda f: -np.std(f[0]["samples"][:, yi])):
                    _joint(ax, fitted["samples"][:, xi],
                           fitted["samples"][:, yi], xr, yr, colour)
                if yi == 0:
                    ax.axhline(0, color=MUTED, lw=0.6)
                    ax.yaxis.set_major_formatter(SIGNED)
                if xi == 0:
                    ax.axvline(0, color=MUTED, lw=0.6)
                if col == 0:
                    ax.set_ylabel(yname, fontsize=8, color=INK)
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if yi == 1:
                    ax.set_yticks([2, 5, 8])
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
            if xi == 1:
                ax.set_xticks([2, 5, 8])
            else:
                ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
            if row == n - 1:
                ax.set_xlabel(xname, fontsize=8, color=INK)
                if xi == 0:
                    ax.xaxis.set_major_formatter(SIGNED)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            _style(ax)
            ax.tick_params(labelsize=7)
    # the numbers, in the empty upper right
    for k, (fitted, colour, label) in enumerate(fits):
        said = ["r(amp, FWHM) %.2f" % fitted["amp_fwhm_r"]]
        if drift:
            said += ["r(amp, drift) %.2f" % fitted["amp_slope_r"],
                     "r(FWHM, drift) %.2f" % fitted["fwhm_slope_r"]]
        said.append("P(amp > 0) %.2f" % fitted["p_positive"])
        fig.text(0.98, 0.72 - 0.13 * k, "%s\n%s" % (label, "\n".join(said)),
                 ha="right", va="top", fontsize=7.5, color=colour)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.97)
    return _save(fig, path, before.get("star"))


def figure_d2v(before, after, path):
    if before.get("d2v") is None or after.get("d2v") is None:
        return None
    scale = 1e3                              # (m/s)^2 -> 10^3 (m/s)^2
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.6))
    both = np.concatenate([before["d2v"], after["d2v"]]) / scale
    good = inliers(both)
    lo, hi = np.percentile(both[good], [0.5, 99.5]) if good.sum() > 4 \
        else (np.nanmin(both), np.nanmax(both))
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    dlim = (lo - pad, hi + pad)
    vlim = _limits(before["v"] - np.median(before["v"]),
                   after["v"] - np.median(after["v"]))
    for col, (run, colour) in enumerate(((before, BEFORE), (after, AFTER))):
        d2v = run["d2v"] / scale
        err = run["sd2v"] / scale if run.get("sd2v") is not None else None
        outside = int(np.sum((d2v < dlim[0]) | (d2v > dlim[1])))
        top = axes[0, col]
        _points(top, run["t"], d2v, err, colour, "exposures")
        top.set_ylim(*dlim)
        top.set_title("%s d2v\nrobust sigma %.0f, median error %.0f%s"
                      % (run["label"], robust_sigma(d2v),
                         np.nanmedian(err) if err is not None else np.nan,
                         "; %d off the scale" % outside if outside else ""),
                      fontsize=7.5, color=INK, loc="left")
        top.set_xlabel("RJD (BJD - 2400000)", fontsize=7.5, color=INK)
        _style(top)
        bottom = axes[1, col]
        keep = inliers(run["d2v"])
        _points(bottom, d2v[keep], (run["v"] - np.median(run["v"]))[keep],
                run["e"][keep], colour, "exposures")
        bottom.set_xlim(*dlim)
        bottom.set_ylim(*vlim)
        bottom.set_title("%s velocity against d2v\nr = %.2f"
                         % (run["label"],
                            pearson(run["d2v"][keep], run["v"][keep])),
                         fontsize=7.5, color=INK, loc="left")
        bottom.set_xlabel("d2v ($10^3$ m$^2$/s$^2$)", fontsize=7.5, color=INK)
        _style(bottom)
    axes[0, 0].set_ylabel("d2v ($10^3$ m$^2$/s$^2$)", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("velocity - median (m/s)", fontsize=7.5, color=INK)
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


def figure_change(before, after, path):
    change = after["v"] - before["v"]
    change = change - np.median(change)
    vtot = vtot_of(before)
    panels = 2 if vtot is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(WIDTH, 3.0), sharey=True,
                             squeeze=False)
    lim = _limits(change)
    ax = axes[0, 0]
    _points(ax, before["t"], change, None, INK, "exposures", alpha=0.6)
    ax.set_xlabel("RJD (BJD - 2400000)", fontsize=8, color=INK)
    ax.set_ylabel("corrected - delivered (m/s)", fontsize=8, color=INK)
    ax.set_title("what the correction moved: rms %.2f m/s"
                 % np.std(change), fontsize=8, color=INK, loc="left")
    ax.set_ylim(*lim)
    ax.axhline(0, color=MUTED, lw=0.6)
    _style(ax)
    if panels == 2:
        ax = axes[0, 1]
        _close_band(ax)
        _points(ax, vtot, change, None, INK, "exposures", alpha=0.6)
        centres, medians, errors = binned(vtot, change)
        if centres.size:
            ax.errorbar(centres, medians, yerr=errors, fmt="s", ls="none",
                        ms=5, mfc="white", mec=INK, mew=0.9, ecolor=INK,
                        elinewidth=0.8, capsize=0, zorder=4)
        ax.set_xlabel(VTOT_LABEL, fontsize=8, color=INK)
        ax.set_title("against $V_\\mathrm{tot}$, 2 km/s bins", fontsize=8,
                     color=INK, loc="left")
        ax.axhline(0, color=MUTED, lw=0.6)
        _style(ax)
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


def figure_dtemp(before, after, path):
    """LBL's temperature projection, delivered and corrected, over time and
    against BERV, all four panels on one scale.

    A correction of the observer frame has no business changing the star's
    temperature; what it may change is a telluric residual that DTEMP picked
    up, and that shows against BERV.
    """
    if before.get("dtemp") is None or after.get("dtemp") is None:
        return None
    name = before["dtemp_name"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.4), sharey=True)
    both = np.concatenate([before["dtemp"] - np.nanmedian(before["dtemp"]),
                           after["dtemp"] - np.nanmedian(after["dtemp"])])
    lim = _limits(both[inliers(both)])
    for col, (run, colour) in enumerate(((before, BEFORE), (after, AFTER))):
        y = run["dtemp"] - np.nanmedian(run["dtemp"])
        e = run.get("sdtemp")
        outside = int(np.sum(np.abs(y) > lim[1]))
        top, bottom = axes[0, col], axes[1, col]
        _points(top, run["t"], y, e, colour, "exposures")
        top.set_title("%s %s\nrobust sigma %.1f K, median error %.1f K%s"
                      % (run["label"], name, robust_sigma(y),
                         np.nanmedian(e) if e is not None else np.nan,
                         "; %d off the scale" % outside if outside else ""),
                      fontsize=7.5, color=INK, loc="left")
        top.set_xlabel("RJD (BJD - 2400000)", fontsize=7.5, color=INK)
        vtot = vtot_of(run)
        if vtot is not None:
            _close_band(bottom)
            _points(bottom, vtot, y, e, colour, "exposures", alpha=0.45)
            centres, medians, errors = binned(vtot, y)
            if centres.size:
                bottom.errorbar(centres, medians, yerr=errors, fmt="s",
                                ls="none", ms=4.5, mfc="white", mec=INK,
                                mew=0.8, ecolor=INK, elinewidth=0.7,
                                capsize=0, zorder=5, label="2 km/s bins, median")
            bottom.set_title("against $V_\\mathrm{tot}$: binned medians"
                             " scatter %.1f K"
                             % (np.std(medians) if centres.size >= 3
                                else np.nan),
                             fontsize=7.5, color=INK, loc="left")
        bottom.set_xlabel(VTOT_LABEL, fontsize=7.5, color=INK)
        for ax in (top, bottom):
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_ylim(*lim)
            _style(ax)
    axes[0, 0].set_ylabel("%s - median (K)" % name, fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("%s - median (K)" % name, fontsize=7.5, color=INK)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


def _mark_peak(ax, periods, power, colour, label, above, fap=np.nan):
    """The highest peak of one curve: a marker on it, and beside it its
    period, its power and its false-alarm probability. No line across the
    panel at its level: it ran through the other curve and the FAP levels."""
    best = int(np.nanargmax(power))
    period, level = periods[best], power[best]
    ax.plot([period], [level], marker="v", ms=7, mfc=colour, mec="white",
            mew=0.8, ls="none", zorder=6)
    # on the side with room: a peak in the right third of a log axis is
    # labelled leftwards, or the label runs off the figure
    lo, hi = np.log10(np.nanmin(periods)), np.log10(np.nanmax(periods))
    right = (np.log10(period) - lo) > 0.66 * (hi - lo)
    ax.annotate("%s %.5g d, power %.2f, FAP %.2g"
                % (label, period, level, fap),
                (period, level), xytext=(-5 if right else 5,
                                         6 if above else -12),
                textcoords="offset points", fontsize=6.5, color=INK,
                ha="right" if right else "left", va="bottom", zorder=7)
    return period, level


#: the periods the Earth puts in a velocity: the year and its first two
#: harmonics (the BERV, the water column, the seasons), and the synodic month
#: (the Moon's light in the sky)
FAP_NAMES = {1e-2: "1%", 1e-3: "0.1%", 1e-4: "$10^{-4}$"}
REFERENCE_PERIODS = ((365.25, "1 yr"), (365.25 / 2, "1/2 yr"),
                     (365.25 / 3, "1/3 yr"), (29.53, "month"))


def planet_lines(planets):
    """[(label, period)] from planet dictionaries or bare periods."""
    out = []
    for i, planet in enumerate(planets or []):
        if isinstance(planet, dict):
            out.append((str(planet.get("short") or planet["name"]),
                        float(planet["period"])))
        else:
            out.append(("bcdefgh"[i] if i < 7 else "", float(planet)))
    return out


#: the colour the known planets are drawn in
PLANET = "#1a7f37"


def for_phase(run, period):
    """The velocities to fold at `period`: less their straight line in time,
    unless the period is long enough for the line to take the planet with
    it (a third of the baseline or more)."""
    v = np.asarray(run["v"], float)
    if period >= np.ptp(run["t"]) / 3.0:
        return v - np.median(v)
    return v - straight_line(run["t"], v, run["e"])[1]


def phase_of(t, planet):
    """Phase in [-0.5, 0.5): 0 at the transit when the archive gives one."""
    t = np.asarray(t, float)
    zero = planet.get("t0")
    zero = zero - 2400000.0 if zero else float(np.min(t))
    return ((t - zero) / planet["period"] + 0.5) % 1.0 - 0.5


def sine_fit(phase, y, e):
    """(K, its error, the curve's function of phase): y = c + A cos + B sin."""
    w = 1.0 / np.asarray(e, float) ** 2
    ang = 2 * np.pi * np.asarray(phase, float)
    A = np.column_stack([np.cos(ang), np.sin(ang), np.ones_like(ang)])
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ ((A * w[:, None]).T @ y)
    resid = y - A @ p
    scale = float(np.sum(w * resid ** 2) / max(len(y) - 3, 1))
    k = float(np.hypot(p[0], p[1]))
    var = (p[0] ** 2 * cov[0, 0] + p[1] ** 2 * cov[1, 1]
           + 2 * p[0] * p[1] * cov[0, 1])
    err = float(np.sqrt(max(scale * var, 0.0)) / max(k, 1e-9))
    return k, err, (lambda x: p[0] * np.cos(2 * np.pi * x)
                    + p[1] * np.sin(2 * np.pi * x) + p[2])


def _profile_lnl(A, y, e):
    """ln L_max of a linear model with a free jitter: for each jitter the
    weighted least squares, and the jitter that does best (the 2 pi left
    out, as bervbias counts it)."""
    from scipy.optimize import minimize_scalar

    from .bervbias import JITTER_MAX, JITTER_MIN

    def minus(ljit):
        var = e ** 2 + np.exp(2 * ljit)
        root = 1.0 / np.sqrt(var)
        coef, *_ = np.linalg.lstsq(A * root[:, None], y * root, rcond=None)
        resid = y - A @ coef
        return 0.5 * float(np.sum(resid ** 2 / var + np.log(var)))

    found = minimize_scalar(minus, bounds=(np.log(JITTER_MIN),
                                           np.log(JITTER_MAX)),
                            method="bounded", options={"xatol": 1e-4})
    # the bounded search can miss a minimum at its edge: try both
    best = min(found.fun, minus(np.log(JITTER_MIN)), minus(np.log(JITTER_MAX)))
    return -float(best)


def planet_bic(t, y, e, period):
    """Delta BIC of a sine at `period`: BIC(constant) - BIC(constant + sine),
    both with a free jitter (k = 2 against 4). Positive when the data prefer
    the sine, on Kass and Raftery's scale as the bias's is."""
    t, y, e = (np.asarray(x, float) for x in (t, y, e))
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    if t.size < 6:
        return np.nan
    ang = 2 * np.pi * t / float(period)
    ones = np.ones_like(t)
    null = _profile_lnl(ones[:, None], y, e)
    sine = _profile_lnl(np.column_stack([ones, np.cos(ang), np.sin(ang)]),
                        y, e)
    return float(2.0 * (sine - null) - 2.0 * np.log(t.size))


def figure_phase(before, after, planets, path, most=4):
    """The velocities folded at each known planet's period, delivered and
    corrected side by side on one scale, with the sine fitted at that period
    and, for a transiting planet with a published K, the orbit it implies."""
    planets = [p for p in (planets or []) if isinstance(p, dict)][:most]
    if not planets:
        return None
    fig, axes = plt.subplots(len(planets), 2, figsize=(WIDTH,
                             2.35 * len(planets) + 0.3), sharex=True,
                             sharey="row", squeeze=False)
    grid = np.linspace(-0.5, 0.5, 400)
    for row, planet in zip(axes, planets):
        folded = [(run, colour, phase_of(run["t"], planet),
                   for_phase(run, planet["period"]))
                  for run, colour in ((before, BEFORE), (after, AFTER))]
        # the bulk, not the few exposures far off it: a planet of a few m/s
        # disappears on an axis stretched to the outliers
        lim = _limits(*[y for _r, _c, _x, y in folded], q=97.0)
        for ax, (run, colour, x, y) in zip(row, folded):
            _points(ax, x, y, run["e"], colour, "exposures", size=2.6,
                    alpha=0.45)
            edges = np.linspace(-0.5, 0.5, 11)
            centres, means, errors = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                inside = (x >= lo) & (x < hi)
                if inside.sum() >= 3:
                    w = 1.0 / run["e"][inside] ** 2
                    centres.append(0.5 * (lo + hi))
                    means.append(np.sum(w * y[inside]) / np.sum(w))
                    errors.append(max(1.0 / np.sqrt(np.sum(w)),
                                      np.std(y[inside])
                                      / np.sqrt(inside.sum())))
            ax.errorbar(centres, means, yerr=errors, fmt="s", ls="none",
                        ms=4.5, mfc="white", mec=INK, mew=0.8, ecolor=INK,
                        elinewidth=0.7, capsize=0, zorder=5,
                        label="phase bins, weighted mean")
            k, err, curve = sine_fit(x, y, run["e"])
            dbic = planet_bic(run["t"], y, run["e"], planet["period"])
            # what the sine leaves: the rms of the residuals to it, beside
            #   the median error bar that says what noise alone would leave
            ok = np.isfinite(x) & np.isfinite(y)
            resid = y[ok] - curve(x[ok])
            rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan
            typical = float(np.nanmedian(run["e"][ok])) if resid.size else np.nan
            ax.plot(grid, curve(grid), color=colour, lw=1.6, zorder=6,
                    label="sine at P: K = %.2f $\\pm$ %.2f m/s" % (k, err))
            if planet.get("k") and planet.get("t0"):
                # circular and transiting: v = -K sin(2 pi phase)
                ax.plot(grid, -planet["k"] * np.sin(2 * np.pi * grid),
                        color=MUTED, lw=1.0, ls="--", zorder=4,
                        label="published K = %.2f m/s" % planet["k"])
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_ylim(*lim)
            ax.set_xlim(-0.5, 0.5)
            ax.set_title("%s: K = %.2f $\\pm$ %.2f m/s\n\u0394BIC %+.1f (%s)\n"
                         "residuals: %.2f m/s rms (errors %.2f)"
                         % (run["label"], k, err, dbic,
                            bervbias.bic_words(dbic, "sine"), rms, typical),
                         fontsize=7.5, color=INK, loc="left")
            ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=1)
            _style(ax)
        row[0].set_ylabel("%s\nP = %s d\nvelocity (m/s)"
                          % (planet["name"], ("%.6f" % planet["period"])
                             if planet["period"] < 100
                             else "%.3f" % planet["period"]),
                          fontsize=7.5, color=PLANET)
    for ax in axes[-1]:
        ax.set_xlabel("phase%s" % (" (0 = transit)" if all(
            p.get("t0") for p in planets) else ""), fontsize=8, color=INK)
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


def figure_periodograms(before, after, planets, path):
    lines = planet_lines(planets)
    # down to the shortest known period, which may be under a day
    # (TOI-4552 b, 0.301 d), with the grid made finer to match
    pmin = min([1.1] + [0.8 * p for _l, p in lines])
    samples = 6000 if pmin >= 1.1 else int(6000 * 1.1 / pmin)
    rows = [("velocity", "v", "e")]
    if before.get("d2v") is not None and after.get("d2v") is not None:
        rows.append(("d2v", "d2v", "sd2v"))
    if before.get("dtemp") is not None and after.get("dtemp") is not None:
        rows.append((before["dtemp_name"], "dtemp", "sdtemp"))
    fig, axes = plt.subplots(len(rows), 1, figsize=(WIDTH, 2.5 * len(rows) + 0.4),
                             sharex=True, squeeze=False)
    drawn = False
    for ax, (what, value, error) in zip(axes[:, 0], rows):
        found_both = []
        fap_levels = []
        for run, colour in ((before, BEFORE), (after, AFTER)):
            y, e = run[value], run.get(error)
            if e is None:
                e = np.ones_like(y)
            keep = inliers(y) if value != "v" else np.isfinite(y)
            found = periodogram(run["t"][keep], y[keep], e[keep],
                                pmin=pmin, samples=samples)
            if not found:
                continue
            drawn = True
            periods, power, best, level, fap, levels = found
            fap_levels.append(levels)
            # see-through, both: the two series overlap nearly everywhere,
            # and at full ink the one drawn last hid the other
            ax.plot(periods, power, color=colour, lw=0.9, alpha=0.55,
                    label="%s: peak %.5g d at %.2f, FAP %.2g"
                          % (run["label"], best, level, fap))
            found_both.append((periods, power, colour, run["label"], fap))
        # each curve's highest peak, where it is and how high, the higher
        # one's label above its marker and the other's below, so they part
        order = sorted(range(len(found_both)),
                       key=lambda i: -np.nanmax(found_both[i][1]))
        for rank, i in enumerate(order):
            periods, power, colour, label, fap = found_both[i]
            _mark_peak(ax, periods, power, colour, label, above=rank == 0,
                       fap=fap)
        # the power a peak needs for each false-alarm probability: the two
        # series share their dates, so their levels agree to a fraction of a
        # per cent, and the higher of the two is drawn once
        levels = (np.nanmax(np.vstack(fap_levels), axis=0) if fap_levels
                  else np.full(len(FAP_LEVELS), np.nan))
        ceiling = ax.get_ylim()[1]
        if np.isfinite(levels).any():
            ceiling = max(ceiling, 1.05 * float(np.nanmax(levels)))
        ax.set_ylim(0, ceiling * 1.12)
        for probability, power_level in zip(FAP_LEVELS, levels):
            if not np.isfinite(power_level):
                continue
            ax.axhline(power_level, color=MUTED, lw=0.7, ls=":", zorder=1)
            ax.annotate("FAP %s" % FAP_NAMES[probability],
                        (1.0, power_level), xycoords=("axes fraction", "data"),
                        xytext=(-2, 1), textcoords="offset points",
                        fontsize=6, color=MUTED, ha="right", va="bottom")
        top = ax.get_ylim()[1] if drawn else 1.0
        for label, period in lines:
            ax.axvline(period, color=PLANET, lw=1.1, zorder=0)
            ax.text(period, top, " " + label, fontsize=7, color=PLANET,
                    va="top", ha="left", fontweight="bold")
        ax.set_xscale("log")
        shown = (np.nanmin([np.nanmin(f[0]) for f in found_both]),
                 np.nanmax([np.nanmax(f[0]) for f in found_both])) \
            if found_both else (0, np.inf)
        for period, name in REFERENCE_PERIODS:
            if not shown[0] <= period <= shown[1]:
                continue
            ax.axvline(period, color=MUTED, lw=0.7, ls="--", zorder=0,
                       alpha=0.8)
            ax.text(period, top, name + " ", fontsize=6, color=MUTED,
                    va="top", ha="right", rotation=90)
        ax.set_ylabel("%s power" % what, fontsize=8, color=INK)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
        _style(ax)
    if not drawn:
        plt.close(fig)
        return None
    axes[-1, 0].set_xlabel("period (d)", fontsize=8, color=INK)
    fig.tight_layout()
    return _save(fig, path, before.get("star"))


# ============================================================ the context ===
def find_tree(config, outdir, given=None):
    """The LBL tree this run's rdb files are in, or None.

    lbl.directory in full since 2026-09-16. Before, it was `lbl`, relative to
    wherever the run was started, which for the window is the configuration's
    folder: that one is read off the recorded command, then the working
    directory, then the output root. The first that holds an lblrdb folder
    wins.
    """
    if given:
        return os.path.abspath(given)
    chosen = str((config.get("lbl") or {}).get("directory") or "lbl")
    if os.path.isabs(chosen):
        return chosen
    starts = []
    command = (config.get("provenance") or {}).get("command") or ""
    found = re.search(r"--config\s+(\S+)", command)
    if found:
        starts.append(os.path.dirname(os.path.abspath(found.group(1))))
    starts.append(os.getcwd())
    root = (config.get("output") or {}).get("directory")
    if root:
        starts.append(os.path.abspath(root))
    for start in starts:
        tree = os.path.join(start, chosen)
        if os.path.isdir(os.path.join(tree, "lblrdb")):
            return tree
    return os.path.join(starts[0], chosen)


def stars_of(outdir, config):
    """The stars a run corrected: one, or the members of a joint run."""
    parent = os.path.dirname(os.path.normpath(outdir))
    if os.path.basename(os.path.dirname(parent)) == "joint":
        return os.path.basename(parent).split("+")
    return [str(config["input"]["object"])]


def planets_of(outdir, config, star, joint=False):
    """The published periods of one star, from its OWN configuration.

    A joint run's configuration is its first member's, target included, so
    it answers for that member only: read for the others, it gave GJ1
    Proxima's four planets. A member's own is its cube_config beside the run,
    else its `objects:` block in the configuration file the run was given.
    Nothing, rather than another star's periods, when neither says.
    """
    member = os.path.join(outdir, "cube_config_%s.yaml" % star)
    if os.path.exists(member):
        with open(member) as handle:
            target = (yaml.safe_load(handle) or {}).get("target") or {}
        return [float(p) for p in target.get("planets") or []]
    if not joint or str(star) == str(stars_first(config)):
        target = config.get("target") or {}
        return [float(p) for p in target.get("planets") or []]
    command = (config.get("provenance") or {}).get("command") or ""
    found = re.search(r"--config\s+(\S+)", command)
    if found and os.path.exists(found.group(1)):
        with open(found.group(1)) as handle:
            layered = yaml.safe_load(handle) or {}
        block = ((layered.get("objects") or {}).get(star) or {})
        target = block.get("target") or {}
        return [float(p) for p in target.get("planets") or []]
    return []


def known_planets(outdir, config, star, joint, facts, folder):
    """(planets, status) of one star: the NASA Exoplanet Archive's, and any
    period its configuration gives that the archive does not.

    Each planet is a dictionary (name, short, period in days, t0 in BJD or
    None, k in m/s or None, source, disposition), shortest period first. The
    archive's answer is kept beside the report, and used again when the
    archive cannot be reached; `status` says which it was.
    """
    from . import archive

    kept = os.path.join(folder, "planets_%s.json" % slug(star))
    got = archive.planets(facts or {}, folder=star)
    if got.get("ok") and not got.get("stale"):
        with open(kept, "w") as handle:
            json.dump(got, handle, indent=1)
    elif not got.get("planets") and os.path.exists(kept):
        with open(kept) as handle:
            got = dict(json.load(handle), stale=True)
    planets = [dict(p) for p in got.get("planets") or []]
    for period in planets_of(outdir, config, star, joint):
        if not any(abs(p["period"] - period) < archive.SAME_PERIOD * period
                   for p in planets):
            planets.append({"name": "%.6g d" % period,
                            "short": "%.4g d" % period, "period": period,
                            "t0": None, "k": None,
                            "source": "configuration", "disposition": ""})
    planets.sort(key=lambda p: p["period"])
    return planets, got


def planet_labels(numbers):
    """The short name of each planet the numbers were measured at."""
    names = numbers.get("planet_names")
    if names:
        return list(names)
    return list("bcdefgh")[:len(numbers.get("planet_periods") or [])]


def stars_first(config):
    """The member a joint configuration was copied from."""
    command = (config.get("provenance") or {}).get("command") or ""
    found = re.search(r"--objects\s+(\S+)", command)
    if found:
        return found.group(1).split(",")[0]
    return str(config["input"].get("object") or "").split("+")[0]


def newest_input(outdir, star, joint):
    """When what LBL should have measured last changed: the fit, or the
    corrected spectra, whichever is newer. None when neither is there.

    Velocities older than that are from a previous fit of the same run name,
    which is what the report says of them at the figures stage of a run
    done again, before its LBL stage has caught up.
    """
    folder = os.path.join(outdir, "corrected", star) if joint \
        else os.path.join(outdir, "corrected")
    times = [os.path.getmtime(p) for p in glob.glob(os.path.join(folder, "*.fits"))]
    fit = os.path.join(outdir, "fit.npz")
    if os.path.exists(fit):
        times.append(os.path.getmtime(fit))
    return max(times) if times else None


# ============================================================== the stars ===
def first_header(config, star):
    """What one of this star's spectra says of it, as a dictionary, or {}.

    The primary header, and over it the first extension that carries APERO's
    PP_ keys: the object as APERO's database has it, each value with its
    source. PP_RV is in m/s whatever its comment says: NIRPS files label it
    [km/s] and hold GJ 1 at 25534 and Proxima at -22400, which are SIMBAD's
    25.534 and -22.4 km/s. It is given in km/s as PP_RV_KMS.
    """
    from astropy.io import fits

    from .config import spectra_dir

    root = (config.get("input") or {}).get("directory") or ""
    for folder in (os.path.join(root, star), spectra_dir(config)):
        files = sorted(glob.glob(os.path.join(folder, "*.fits")))
        files = [f for f in files if not os.path.basename(f).startswith(".")]
        if not files:
            continue
        try:
            with fits.open(files[0]) as hdulist:
                out = dict(hdulist[0].header)
                for hdu in hdulist[1:]:
                    if "PP_OBJN" in hdu.header:
                        out.update(dict(hdu.header))
                        rv = hdu.header.get("PP_RV")
                        if isinstance(rv, (int, float)):
                            out["PP_RV_KMS"] = rv / 1000.0
                        break
            return out
        except Exception:                                      # noqa: BLE001
            return {}
    return {}


def star_facts(star, config, folder):
    """SIMBAD's view of one star, kept beside the report for next time."""
    from . import simbad

    header = first_header(config, star)
    kept = os.path.join(folder, "simbad_%s.json" % slug(star))
    facts = simbad.star(star, header)
    if facts.get("ok"):
        with open(kept, "w") as handle:
            json.dump(facts, handle, indent=1)
    elif os.path.exists(kept):
        with open(kept) as handle:
            facts = json.load(handle)
        facts["stale"] = True
    facts["header"] = {k: header.get(k) for k in header
                       if (k.startswith("PP_") or k in (
                           "OBJECT", "DRSOBJN", "ESO OCS TARG SPTYPE",
                           "ESO OCS TARG JMAG", "OBJTEMP", "OBJMAG"))
                       and not isinstance(header.get(k), bytes)}
    return facts


def _sexa(value, hours):
    if value is None:
        return "n/a"
    value = value / 15.0 if hours else value
    sign = "-" if value < 0 else ("" if hours else "+")
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    sec = (value - d - m / 60.0) * 3600
    return "%s%02d %02d %0*.*f" % (sign, d, m, 5 if hours else 4,
                                   2 if hours else 1, sec)


def distance_text(facts):
    plx, err = facts.get("plx_value"), facts.get("plx_err")
    if not plx or plx <= 0:
        return "n/a"
    text = "%.3f mas" % plx
    if err:
        text += " $\\pm$ %.3f" % err
    return text + ", %.2f pc" % (1000.0 / plx)


def _pp(header, key, fmt="%s", unit=""):
    """An APERO value with its source, or ''."""
    value = header.get(key)
    if value in (None, "", "None"):
        return ""
    text = (fmt % value) if not isinstance(value, str) else value
    source = header.get(key + "S")
    return tex(text + (" " + unit if unit else "")) + (
        " \\muted{(%s)}" % tex(source)
        if source not in (None, "", "None") else "")


def star_table(star, facts, planets):
    """One star, SIMBAD's description beside the one APERO worked with."""
    header = facts.get("header") or {}
    rows = []

    def add(what, simbad_text, apero_text=""):
        rows.append((what, simbad_text, apero_text))

    apero_name = header.get("PP_OBJN") or header.get("DRSOBJN")
    add("APERO name", "", "\\textbf{%s}%s" % (
        tex(apero_name), " \\muted{(PP\\_OBJN)}" if header.get("PP_OBJN")
        else " \\muted{(DRSOBJN)}") if apero_name else "n/a")
    ok = facts.get("ok")
    if ok:
        add("name", "%s \\muted{(%s)}" % (
            tex(facts["main_id"].replace("NAME ", "")), tex(facts.get("otype"))),
            _pp(header, "PP_OBJNS"))
        found = "from %s" % tex(facts.get("resolved_from"))
        if facts.get("offset") is not None:
            found += (", %.1f arcsec from %s position at %.1f"
                      % (facts["offset"], facts.get("offset_from", "the"),
                         facts.get("offset_epoch", 2000)))
        add("found", found)
        if facts.get("ids"):
            add("identifiers", tex_break(", ".join(facts["ids"])))
    else:
        add("SIMBAD", "\\watch{not reached: %s}"
            % tex(facts.get("error") or "no answer"))
        add("name in the headers", tex(header.get("OBJECT", "n/a")))
    ra_apero = header.get("PP_RA")
    add("position (deg)",
        "%.6f, %+.6f \\muted{(ICRS, J2000)}" % (facts["ra"], facts["dec"])
        if ok and facts.get("ra") is not None else "",
        ("%.6f, %+.6f" % (ra_apero, header.get("PP_DEC"))
         + (" \\muted{(%s, JD %.1f)}" % (tex(header.get("PP_RAS")),
                                           header.get("PP_EPOCH"))
            if header.get("PP_EPOCH") else ""))
        if isinstance(ra_apero, (int, float)) else "")
    if ok:
        sptype = tex(facts.get("sp_type") or "n/a")
    else:
        sptype = tex(header.get("ESO OCS TARG SPTYPE", ""))
    add("spectral type", sptype, _pp(header, "PP_SPT"))
    add("parallax, distance", distance_text(facts) if ok else "",
        _pp(header, "PP_PLX", "%.3f", "mas") + (
            ", %.2f pc" % (1000.0 / header["PP_PLX"])
            if isinstance(header.get("PP_PLX"), (int, float))
            and header["PP_PLX"] > 0 else ""))
    add("proper motion (mas/yr)",
        "%+.2f, %+.2f" % (facts["pmra"], facts.get("pmdec") or 0.0)
        if ok and facts.get("pmra") is not None else "",
        ("%+.2f, %+.2f" % (header["PP_PMRA"], header.get("PP_PMDE", 0.0))
         + (" \\muted{(%s)}" % tex(header.get("PP_PMRAS"))
            if header.get("PP_PMRAS") else ""))
        if isinstance(header.get("PP_PMRA"), (int, float)) else "")
    rv = facts.get("rvz_radvel") if ok else None
    rv_apero = header.get("PP_RV_KMS")
    add("systemic velocity (km/s)",
        ("%.3f" % rv + (" $\\pm$ %.3f" % facts["rvz_err"]
                        if facts.get("rvz_err") else ""))
        if rv is not None else ("not in SIMBAD" if ok else ""),
        ("%.3f" % rv_apero + (" \\muted{(%s)}" % tex(header.get("PP_RVS"))
                              if header.get("PP_RVS") not in (None, "None")
                              else " \\muted{(no source: a placeholder)}"
                              if rv_apero == 0 else ""))
        if isinstance(rv_apero, (int, float)) else "")
    teff_simbad = ""
    measures = facts.get("teff") or [] if ok else []
    if measures:
        latest = measures[0]
        teff_simbad = "%.0f K" % latest["teff"]
        if latest.get("log_g") is not None:
            teff_simbad += ", log g %.2f" % latest["log_g"]
        if latest.get("fe_h") is not None:
            teff_simbad += ", [Fe/H] %+.2f" % latest["fe_h"]
        teff_simbad += " \\muted{(%s)}" % tex(latest["bibcode"])
        if len(measures) > 1:
            values = [m["teff"] for m in measures]
            teff_simbad += ("; %d values, %.0f to %.0f K"
                            % (len(values), min(values), max(values)))
    elif header.get("OBJTEMP"):
        teff_simbad = "%s K \\muted{(OBJTEMP)}" % tex(header["OBJTEMP"])
    add("Teff, latest", teff_simbad, _pp(header, "PP_TEFF", "%.0f", "K"))
    mags = (facts.get("mags") or {}) if ok else {}
    shown = ["%s %.2f" % (band, mags[band]) for band in
             ("B", "V", "G", "J", "H", "K") if mags.get(band) is not None]
    add("magnitudes", ", ".join(shown) or (
        "J %s \\muted{(headers)}" % tex(header["ESO OCS TARG JMAG"])
        if header.get("ESO OCS TARG JMAG") else "n/a"))
    known, status = planets if isinstance(planets, tuple) else (
        [{"name": "%.6g d" % p, "period": p, "source": "configuration",
          "disposition": ""} for p in planets], {"ok": True})
    said = "; ".join(
        "%s, %s d \\muted{(%s%s)}" % (
            tex(p["name"]), number(p["period"], 6 if p["period"] < 10 else 4),
            tex(p["source"]), ", " + tex(p["disposition"])
            if p.get("disposition") and p["source"] == "TOI" else "")
        for p in known)
    where = ("the NASA Exoplanet Archive, read on %s" % tex(status["retrieved"])
             if status.get("retrieved") else "")
    if status.get("stale"):
        where += " \\watch{(kept from an earlier report: the archive did not"\
                 " answer this time)}"
    elif status.get("ok") is False:
        where = "\\watch{the NASA Exoplanet Archive did not answer (%s)}" % tex(
            str(status.get("error"))[:80])
    add("known planets", said or "none known", where)
    add("read on", tex(facts.get("retrieved", "")) + (
        " \\watch{(kept from an earlier report: SIMBAD did not answer this"
        " time)}" if facts.get("stale") else ""),
        tex(str(header.get("PP_DDATE", ""))[:10]) + (
            " \\muted{(APERO's database)}" if header.get("PP_DDATE") else ""))
    body = "\n".join("%s & %s & %s \\\\" % (tex(what), a, b)
                     for what, a, b in rows)
    ragged = ">{\\raggedright\\arraybackslash}"
    columns = "".join(ragged + "p{%s\\linewidth}" % w
                      for w in ("0.2", "0.41", "0.28"))
    return ("\\subsection{%s}\n\\begin{longtable}{%s}\n\\toprule\n"
            " & SIMBAD & APERO (PP\\_ keys) \\\\\n\\midrule\n"
            "%s\n\\bottomrule\n\\end{longtable}\n"
            % (tex(star), columns, body))


def star_label(facts):
    """'Smethells 20, M1Ve, 50.7 pc' for the abstract, or ''."""
    if not facts.get("ok"):
        return ""
    parts = [facts["main_id"].replace("NAME ", "")]
    if facts.get("sp_type"):
        parts.append(facts["sp_type"])
    if facts.get("plx_value"):
        parts.append("%.1f pc" % (1000.0 / facts["plx_value"]))
    return ", ".join(parts)


# ============================================================== sections ===
def figure_block(path, caption, short=None, label=None, width=r"\linewidth",
                 page=None):
    """One figure. `short` is what the list of figures says of it: a caption
    is a paragraph, and a list of paragraphs is not a list anybody reads."""
    options = "width=%s,height=0.82\\textheight,keepaspectratio" % width
    if page is not None:
        options += ",page=%d" % page
    return ("\\begin{figure}\n\\centering\n\\includegraphics[%s]{%s}\n"
            "\\caption%s{%s}%s\n\\end{figure}\n"
            % (options, path, "[%s]" % short if short else "", caption,
               "\\label{%s}" % label if label else ""))


def mark(word):
    return {"gain": "\\gain{gain}", "loss": "\\loss{loss}",
            "same": "\\muted{same}", "watch": "\\watch{changed}"}[word]


def summary_table(star, numbers):
    b, a = numbers["before"], numbers["after"]
    rows = []

    def row(name, key, unit="m/s", digits=2, better="lower", tolerance=0.02):
        vb, va = b.get(key, np.nan), a.get(key, np.nan)
        if not (np.isfinite(vb) or np.isfinite(va)):
            return
        change = ("%.2f$\\times$" % (vb / va) if better == "lower"
                  and np.isfinite(vb) and np.isfinite(va) and va > 0 else "")
        rows.append("%s & %s & %s & %s & %s \\\\"
                    % (name, number(vb, digits), number(va, digits), change,
                       mark(verdict(vb, va, better, tolerance))))

    row("rms (m/s)", "rms")
    row("robust sigma (m/s)", "robust")
    row("nightly rms (m/s)", "nightly_rms")
    row("median error (m/s)", "median_error")
    if "bias_peak" in b:
        def said(side):
            if side["bias_detected"]:
                return "%.1f $\\pm$ %.1f (%.1f$\\sigma$)" % (
                    side["bias_peak"], side["bias_err"],
                    side["bias_significance"])
            return "$<$ %.1f" % side["bias_upper"]
        rows.append("BERV bias at its peak (m/s) & %s & %s & & %s \\\\"
                    % (said(b), said(a), mark(bias_verdict(b, a))))
        rows.append("its FWHM (km/s) & %s & %s & & \\\\" % tuple(
            ("%.1f $\\pm$ %.1f" % (side["bias_fwhm"], side["bias_fwhm_err"])
             if side["bias_detected"] else "\\muted{the prior's}")
            for side in (b, a)))
        rows.append("amp-FWHM correlation (posterior) & %s & %s & & \\\\"
                    % (number(b["bias_amp_fwhm_r"]),
                       number(a["bias_amp_fwhm_r"])))
        if b.get("bias_drift"):
            rows.append("amp-drift correlation (posterior) & %s & %s & & \\\\"
                        % (number(b["bias_amp_slope_r"]),
                           number(a["bias_amp_slope_r"])))
        rows.append("P(amp $>$ 0) (posterior) & %s & %s & & \\\\"
                    % (number(b["bias_p_positive"]),
                       number(a["bias_p_positive"])))
        if np.isfinite(b.get("bias_delta_bic", np.nan)):
            def bic(side):
                value = side.get("bias_delta_bic", np.nan)
                return "%s (%s)" % (tex("%+.1f" % value) if np.isfinite(value)
                                    else "n/a", bervbias.bic_words(value))
            rows.append("$\\Delta$BIC, no bias $-$ bias & %s & %s & & %s \\\\"
                        % (bic(b), bic(a),
                           mark(bic_verdict(b["bias_delta_bic"],
                                            a["bias_delta_bic"]))))
        row("jitter beyond LBL's errors (m/s)", "jitter")
        if b.get("bias_drift"):
            rows.append("DC level, its drift (m/s/yr) & %s & %s & & \\\\"
                        % tuple("%s $\\pm$ %.2f" % (tex("%+.2f" % side["bias_slope"]),
                                                   side["bias_slope_err"])
                                for side in (b, a)))
    if "berv_binned" in b:
        row("scatter of $V_\\mathrm{tot}$-binned medians (m/s)",
            "berv_binned")
    if "d2v_sigma" in b:
        vb, va = b["d2v_sigma"] / 1e3, a["d2v_sigma"] / 1e3
        # lower is better: a scatter that falls is telluric noise the
        # correction took out of the line width, one that rises is noise it
        # put in (asked for on 2026-09-17; it was flagged as a change either
        # way until then)
        word = verdict(vb, va, tolerance=0.1)
        rows.append("d2v robust sigma ($10^3$ m$^2$/s$^2$) & %s & %s & & %s \\\\"
                    % (number(vb, 1), number(va, 1), mark(word)))
        rows.append("velocity-d2v correlation r & %s & %s & & \\muted{activity}"
                    " \\\\" % (number(b["d2v_r"], 3), number(a["d2v_r"], 3)))
    if "dtemp_sigma" in b and "dtemp_sigma" in a:
        name = tex(b["dtemp_name"])
        word = ("watch" if verdict(b["dtemp_sigma"], a["dtemp_sigma"],
                                   tolerance=0.1) != "same" else "same")
        rows.append("%s robust sigma (K) & %s & %s & & %s \\\\"
                    % (name, number(b["dtemp_sigma"]), number(a["dtemp_sigma"]),
                       mark(word)))
        rows.append("%s median error (K) & %s & %s & & \\\\"
                    % (name, number(b["dtemp_error"]), number(a["dtemp_error"])))
        rows.append("velocity-%s correlation r & %s & %s & & \\muted{activity}"
                    " \\\\" % (name, number(b["dtemp_r"], 3),
                                 number(a["dtemp_r"], 3)))
        rows.append("scatter of %s's $V_\\mathrm{tot}$-binned medians (K)"
                    " & %s & %s & & %s"
                    " \\\\" % (name, number(b["dtemp_berv_binned"]),
                                 number(a["dtemp_berv_binned"]),
                                 mark(verdict(b["dtemp_berv_binned"],
                                              a["dtemp_berv_binned"],
                                              tolerance=0.05))))
    if "peak_period" in b:
        rows.append("highest velocity peak (d) & %s & %s & & \\\\"
                    % (number(b["peak_period"], 3), number(a["peak_period"], 3)))
        rows.append("its false-alarm probability & %s & %s & & \\\\"
                    % (tex("%.2g" % b["peak_fap"]), tex("%.2g" % a["peak_fap"])))
    for letter, period, kb, ka in zip(planet_labels(numbers),
                                      numbers["planet_periods"],
                                      b["planets"], a["planets"]):
        moved = abs(ka[0] - kb[0]) > 2 * np.hypot(ka[1], kb[1])
        rows.append("K at %s d (%s, m/s) & %s $\\pm$ %s & %s $\\pm$ %s"
                    " & & %s \\\\"
                    % (number(period, 4), tex(letter), number(kb[0]),
                       number(kb[1]),
                       number(ka[0]), number(ka[1]),
                       mark("watch" if moved else "same")))
        bics = (b.get("planet_bic") or [], a.get("planet_bic") or [])
        index = numbers["planet_periods"].index(period)
        if index < len(bics[0]) and index < len(bics[1]):
            vb, va = bics[0][index], bics[1][index]
            # higher is better here: a correction that took noise away makes
            # the planet stand out more, one that took the planet makes it less
            word = ("gain" if va > vb + 2 else "loss" if va < vb - 2
                    else "same")
            rows.append("$\\Delta$BIC of a sine at %s d (%s) & %s & %s & & %s"
                        " \\\\" % (number(period, 4), tex(letter),
                                    "%s (%s)" % (tex("%+.1f" % vb),
                                                 bervbias.bic_words(vb, "sine")),
                                    "%s (%s)" % (tex("%+.1f" % va),
                                                 bervbias.bic_words(va, "sine")),
                                    mark(word)))
    rows.append("\\midrule")
    rows.append("rms of corrected - delivered (m/s) & \\multicolumn{2}{c}{%s}"
                " & & \\\\" % number(numbers["change_rms"]))
    rows.append("noise removed in quadrature (m/s) & \\multicolumn{2}{c}{%s}"
                " & & \\\\" % (number(numbers["removed"])
                               if np.isfinite(numbers["removed"])
                               else "none: the rms grew"))
    # FIXED WIDTHS (asked 2026-10-23): with `rrr` a cell such as "726.5
    #   $\\pm$ 141.9 (3.3$\\sigma$)" set its column's width, and TOI-904's
    #   table ran 73 pt past the margin. Each column now has its share of the
    #   line and its cells wrap: 0.86 of \\linewidth, plus 12 pt of padding
    #   per column, is less than the line
    left = ">{\\raggedright\\arraybackslash}p{%s\\linewidth}"
    right = ">{\\raggedleft\\arraybackslash}p{%s\\linewidth}"
    columns = (left % "0.33" + right % "0.19" + right % "0.19"
               + right % "0.065" + left % "0.085")
    return ("\\begin{longtable}{%s}\n"
            "\\caption{%s: the %d exposures both series share, over %d"
            " nights.}\\\\\n\\toprule\n & delivered & corrected & gain & \\\\\n"
            "\\midrule\n\\endhead\n%s\n\\bottomrule\n\\end{longtable}\n"
            % (columns, tex(star), numbers["n"], numbers["nights"],
               "\n".join(rows)))


def observations_table(star, numbers):
    """What was observed, from the headers: the table before the numbers."""
    o = numbers.get("observations")
    if not o:
        return ""
    rows = ["exposures (both series) & %d \\\\" % o["n"],
            "nights & %d \\\\" % o["nights"],
            "first and last observation (UT) & %s to %s, %s d \\\\"
            % (o["first"], o["last"], number(o["span"], 0))]
    if "snr" in o:
        rows.append("median SNR (%s) & %s \\\\" % (tex(o["snr_name"]),
                                                   number(o["snr"], 1)))
    if "berv" in o:
        rows.append("BERV (km/s) & %s to %s \\\\"
                    % (signed(o["berv"][0]), signed(o["berv"][1])))
    if "vtot" in o:
        rows.append("systemic velocity, median vrad (km/s) & %s \\\\"
                    % signed(o["systemic"], 3))
        rows.append("$V_\\mathrm{tot}$ = vrad $-$ BERV (km/s) & %s to %s"
                    " \\\\" % (signed(o["vtot"][0]), signed(o["vtot"][1])))
        limit = "%g" % bervbias.CLOSE_KMS
        if o["windows"]:
            spans = ["%s%s" % (calendar_day(first), "" if first == last
                               else " to " + calendar_day(last))
                     for first, last in o["windows"]]
            said = ("%s: %d of the %d exposures (%.1f\\%%) fall on these"
                    " dates, %d of them within %s km/s"
                    % ("; ".join(spans), o["window_n"], o["n"],
                       100.0 * o["window_n"] / max(o["n"], 1), o["close_n"],
                       limit))
        else:
            said = ("none: $V_\\mathrm{tot}$ never comes within %s km/s of"
                    " zero" % limit)
        rows.append("calendar dates with $|V_\\mathrm{tot}| < %s$ km/s, every"
                    " year (UT), the star's lines on the tellurics & %s \\\\"
                    % (limit, said))
    return ("\\begin{longtable}{p{0.36\\linewidth}p{0.58\\linewidth}}\n"
            "\\caption{%s: the observations, from the headers.}\\\\\n"
            "\\toprule\n\\endhead\n%s\n\\bottomrule\n\\end{longtable}\n"
            % (tex(star), "\n".join(rows)))


def signed(value, digits=1):
    """'+2.6' or '-23.2', with a minus sign LaTeX sets as one."""
    if not np.isfinite(value):
        return "n/a"
    return ("$%+.*f$" % (digits, value))


def bic_verdict(before, after, step=2.0):
    """gain when the evidence for a bias fell by more than `step` on the BIC
    scale, loss when it rose by more and is now worth a mention."""
    if not (np.isfinite(before) and np.isfinite(after)):
        return "same"
    if after < before - step:
        return "gain"
    if after > before + step and after > step:
        return "loss"
    return "same"


def bias_verdict(b, a):
    """gain, loss or same for the fitted BERV bias.

    Measured against the errors, not a fixed tolerance: a bias that was
    detected and is gone is a gain, one that appeared is a loss, and two
    detections are compared by their difference in units of its error.
    """
    if not ("bias_detected" in b and "bias_detected" in a):
        return "same"
    if b["bias_detected"] and not a["bias_detected"]:
        return "gain"
    if a["bias_detected"] and not b["bias_detected"]:
        return "loss"
    if not b["bias_detected"]:
        return "same"
    change = abs(b["bias_peak"]) - abs(a["bias_peak"])
    error = np.hypot(b["bias_err"], a["bias_err"])
    return "gain" if change > error else "loss" if -change > error else "same"


def bias_words(side):
    bic = side.get("bias_delta_bic", np.nan)
    said = (", $\\Delta$BIC %s" % tex("%+.0f" % bic)) if np.isfinite(bic) else ""
    if side.get("bias_detected"):
        return "%.0f $\\pm$ %.0f m/s%s" % (side["bias_peak"], side["bias_err"],
                                          said)
    if "bias_upper" in side:
        return "none detected ($<$ %.0f m/s%s)" % (side["bias_upper"], said)
    return "not fitted"


def verdict_lines(numbers):
    """(better, worse, moved) as lists of what, for the summary's bullets."""
    b, a = numbers["before"], numbers["after"]
    better, worse, moved = [], [], []
    for key, what, how in (("rms", "rms", "lower"),
                           ("robust", "robust sigma", "lower"),
                           ("nightly_rms", "nightly rms", "lower"),
                           ("median_error", "LBL's error bars", "lower"),
                           ("jitter", "the jitter beyond LBL's errors",
                            "lower"),
                           ("berv_binned", "the structure against $V_\\mathrm{tot}$",
                            "lower")):
        if key not in b:
            continue
        word = verdict(b[key], a[key], how, 0.05 if how == "zero" else 0.02)
        (better if word == "gain" else worse if word == "loss"
         else []).append(what)
    word = bias_verdict(b, a)
    (better if word == "gain" else worse if word == "loss"
     else []).append("the fitted BERV bias")
    if "bias_delta_bic" in b and "bias_delta_bic" in a:
        word = bic_verdict(b["bias_delta_bic"], a["bias_delta_bic"])
        (better if word == "gain" else worse if word == "loss"
         else []).append("the evidence for it ($\\Delta$BIC %s to %s)"
                         % (tex("%+.0f" % b["bias_delta_bic"]),
                            tex("%+.0f" % a["bias_delta_bic"])))
    if "dtemp_sigma" in b and "dtemp_sigma" in a:
        if verdict(b["dtemp_sigma"], a["dtemp_sigma"], tolerance=0.1) != "same":
            moved.append("%s's scatter (%.1f to %.1f K)"
                         % (tex(b["dtemp_name"]), b["dtemp_sigma"],
                            a["dtemp_sigma"]))
        word = verdict(b["dtemp_berv_binned"], a["dtemp_berv_binned"],
                       tolerance=0.05)
        (better if word == "gain" else worse if word == "loss"
         else []).append("%s's structure against $V_\\mathrm{tot}$" % tex(b["dtemp_name"]))
    if "d2v_sigma" in b:
        word = verdict(b["d2v_sigma"], a["d2v_sigma"], tolerance=0.1)
        (better if word == "gain" else worse if word == "loss"
         else []).append("d2v's scatter (%.0f to %.0f, in $10^3$ m$^2$/s$^2$)"
                         % (b["d2v_sigma"] / 1e3, a["d2v_sigma"] / 1e3))
    for letter, period, kb, ka in zip(planet_labels(numbers),
                                      numbers["planet_periods"],
                                      b["planets"], a["planets"]):
        if abs(ka[0] - kb[0]) > 2 * np.hypot(ka[1], kb[1]):
            moved.append("the amplitude of %s (%.4g d)"
                         % (tex(letter), period))
    return better, worse, moved


def activity_note(numbers):
    """A sentence when the velocity's strongest period is also d2v's."""
    a = numbers["after"]
    if same_period(a.get("peak_period"), a.get("d2v_peak_period"),
                   numbers.get("baseline")):
        return ("The corrected velocity's strongest peak, at %.3g d, is also"
                " d2v's: activity (the rotation or its alias) rather than a"
                " planet." % a["peak_period"])
    return ""


def star_headline(star, numbers):
    """The abstract's sentence for one star: short, and a planet only when
    the correction changed its amplitude."""
    b, a = numbers["before"], numbers["after"]
    ratio = b["rms"] / a["rms"] if a["rms"] > 0 else np.nan
    words = ("%s: rms %.2f to %.2f m/s (%s), nightly rms %.2f to %.2f m/s"
             % (tex(star), b["rms"], a["rms"],
                ("%.2f times lower" % ratio) if ratio >= 1
                else ("%.2f times higher" % (1 / ratio)),
                b["nightly_rms"], a["nightly_rms"]))
    if "bias_peak" in b:
        words += ", BERV bias %s to %s" % (bias_words(b), bias_words(a))
    words += "."
    _better, _worse, moved = verdict_lines(numbers)
    if moved:
        words += " \\watch{Changed where nothing should: %s.}" % "; ".join(moved)
    return words


def star_sentence(star, numbers):
    b, a = numbers["before"], numbers["after"]
    ratio = b["rms"] / a["rms"] if a["rms"] > 0 else np.nan
    words = ("%s: on the %d exposures both series share, the rms goes from"
             " %.2f to %.2f m/s (%s) and the nightly rms from %.2f to %.2f m/s."
             % (tex(star), numbers["n"], b["rms"], a["rms"],
                ("%.2f times lower" % ratio) if ratio >= 1
                else ("%.2f times higher" % (1 / ratio)),
                b["nightly_rms"], a["nightly_rms"]))
    if "bias_peak" in b:
        words += (" The bias that follows BERV, fitted, goes from %s to %s."
                  % (bias_words(b), bias_words(a)))
    for letter, period, kb, ka in zip(planet_labels(numbers),
                                      numbers["planet_periods"],
                                      b["planets"], a["planets"]):
        words += (" At the %.4g d period of %s, K goes from %.2f to"
                  " %.2f m/s." % (period, tex(letter), kb[0], ka[0]))
    return words


def series_section(series, folder):
    """The report's first section: each star's velocity series as LBL's
    report draws it (lbl.report.figures.page_sequence), every epoch coloured
    by the star's velocity in the telluric frame when SIMBAD has its
    systemic velocity, by its BERV otherwise, and how its scatter is
    distributed. [] when no star has velocities yet."""
    if not series:
        return []
    from lbl.report import figures as lbl_figures
    from lbl.report import series as lbl_series
    parts = ["\\section{The velocity series}\n"
             "Every epoch coloured by the star's velocity in the telluric"
             " frame, $V_\\mathrm{tot}$, when SIMBAD has its systemic"
             " velocity: a residual of the tellurics is largest near"
             " $V_\\mathrm{tot} = 0$, where the colours are darkest, and"
             " changes sign with it. Without it, by the BERV: a residual that"
             " follows the BERV is not the star's, and shows as colours that"
             " line up.\n"]
    for star, run in series:
        name = "series-%s.pdf" % slug(star)
        try:
            table = lbl_series.read_rdb(run["path"])
            fig = lbl_figures.page_sequence(
                plt, table, star, lbl_series.systemic_velocity(star))
            coloured = fig.coloured_by
            fig.savefig(os.path.join(folder, "figures", name))
            plt.close(fig)
        except Exception as exc:                          # noqa: BLE001
            parts.append("\\muted{The velocity series of %s could not be"
                         " drawn (%s).}\n" % (tex(star), tex(str(exc))))
            continue
        parts.append(figure_block(
            "figures/" + name,
            "The %s velocities of %s that LBL measured (%s), each epoch"
            " coloured by %s; below, how their scatter is distributed,"
            " dashed lines at plus and minus one MAD."
            % (tex(run["label"]), tex(star),
               tex(os.path.basename(run["path"])),
               "$V_\\mathrm{tot}$, the star's velocity in the telluric frame"
               " (outlined: within %g km/s of the tellurics)"
               % lbl_figures.VTOT_CLOSE if coloured == "V_tot"
               else "its BERV"),
            short="Velocity series, %s" % tex(star)))
    return parts


def velocity_section(star, before, after, numbers, folder, stale):
    base = "rv-" + slug(star)
    figures = os.path.join(folder, "figures")
    parts = ["\\subsection{%s}\n" % tex(star)]
    if stale:
        parts.append("\\watch{These velocities are older than this run's"
                     " fit or its corrected spectra: they are from an earlier"
                     " run under the same name, and LBL has not measured this"
                     " one yet.}\n\n")
    parts.append(star_sentence(star, numbers) + " " + activity_note(numbers)
                 + "\n")
    near = numbers["before"].get("bias_near")
    if near is not None and near < bervbias.MIN_NEAR:
        parts.append("\n\\watch{The bias against $V_\\mathrm{tot}$ is not"
                     " fitted: %d exposure%s bring%s $V_\\mathrm{tot}$ within"
                     " %g km/s of zero, where the star's lines meet the"
                     " tellurics, and a fit needs %d.}\n"
                     % (near, "" if near == 1 else "s",
                        "s" if near == 1 else "", bervbias.FWHM_MAX,
                        bervbias.MIN_NEAR))
    if before.get("dtemp") is None:
        parts.append("\n\\muted{No temperature projection in these"
                     " velocities: this run's LBL had no DTEMP table. Runs from"
                     " 2026-09-16 measure the one nearest the star's Teff"
                     " (lbl.dtemp).}\n")
    name = tex(star)
    drawn = [
        (figure_time(before, after, os.path.join(figures, base + "-time.pdf")),
         "%s: the velocities over the campaign" % name,
         "The velocities over the campaign, delivered above and corrected"
         " below, on the same scale, each less its own median, which its"
         " axis gives. %s; no line joins them. The dashed line is a straight"
         " line in time, fitted with the error bars as weights. Below, the"
         " spread of the exposures, delivered and corrected on the same"
         " bins: as measured on the left, less that straight line on the"
         " right, with the rms and the robust sigma (1.4826 MAD) of each, in"
         " m/s."
         % ("Small points are exposures, large ones the weighted nightly"
            " means" if numbers["nights"] < 0.8 * numbers["n"]
            else "One point per exposure, about one a night")),
        (figure_berv(before, after, os.path.join(figures, base + "-berv.pdf")),
         "%s: the BERV bias, fitted" % name,
         "The velocities against the total velocity"
         " $V_\\mathrm{tot} = v_\\mathrm{rad} - \\mathrm{BERV}$, the star's"
         " velocity in the telluric frame (the systemic velocity less the"
         " BERV, composed relativistically), with the bias a"
         " telluric line blended with the stellar lines produces, fitted by"
         " MCMC: $v = c + s\\,(t - t_\\mathrm{mid}) + a\\,V_\\mathrm{tot}\\,"
         "e^{-V_\\mathrm{tot}^2/2\\sigma^2}$, the DC level $c$ free and free"
         " to drift by $s$ over the campaign (the zero of the velocities is"
         " not their median when the exposures sit more on one side of"
         " $V_\\mathrm{tot} = 0$, and a drift passes for a bias when"
         " $V_\\mathrm{tot}$ follows the season), with a jitter added"
         " to LBL's error bars, a flat prior on $a$ and a Gaussian one on the"
         " Gaussian's FWHM, $2\\sqrt{2\\ln 2}\\,\\sigma$, of $5 \\pm 1.5$ km/s"
         " within 1 to 10 km/s. The line is the posterior median, the band"
         " its 1$\\sigma$ envelope, in each series' colour (the band alone"
         " for a bias that is not"
         " detected); squares are medians in 2 km/s bins, and the grey band"
         " is $|V_\\mathrm{tot}| < 4$ km/s, where the star's lines sit on the"
         " tellurics. Above,"
         " each series less its fitted DC level, drift included; below,"
         " both envelopes on one"
         " axis. $\\Delta$BIC is BIC(no bias) $-$ BIC(bias), $k \\ln n - 2\\ln"
         " L_\\mathrm{max}$ with $k = 3$ (DC level, its drift, jitter)"
         " against 5, the"
         " likelihood maximised with the FWHM free within 1 to 10 km/s:"
         " positive when the data prefer the bias, above 2, 6 and 10 positive,"
         " strong and very strong evidence for it (Kass \\& Raftery 1995); it"
         " asks whether a curve of this shape improves the fit, whatever put"
         " it there, and on a narrow $V_\\mathrm{tot}$ span a seasonal signal"
         " can be that curve."
         " The peak is the bias at $V_\\mathrm{tot} = \\pm\\sigma$,"
         " $a\\sigma e^{-1/2}$; below 3$\\sigma$ from zero, only an upper"
         " limit on it is quoted."),
        (figure_corner(before, after, os.path.join(figures, base + "-corner.pdf")),
         "%s: the covariance of amp, FWHM and the drift" % name,
         "The joint posterior of the bias's amplitude $a$, the FWHM of the"
         " Gaussian whose derivative it is (from 1 to 10 km/s) and the DC"
         " level's drift $s$, delivered and"
         " corrected laid over each other in their colours: below the"
         " diagonal, each pair's joint posterior, shaded darkest where it"
         " holds 39\\% of the mass (1$\\sigma$ in two dimensions), lighter"
         " out to 86\\% and 99\\% (2 and 3$\\sigma$), with the 1 and"
         " 2$\\sigma$ outlines; on the diagonal, each parameter alone, the"
         " FWHM with its prior dashed. $a$ is signed: its"
         " prior is flat, the same on both sides, half the"
         " walkers start on each, and the amp axis is symmetric about zero;"
         " P($a > 0$) is the posterior's share on the positive side. The two trade against each other,"
         " since a narrower bias needs a larger amplitude to reach the same"
         " points. The FWHM's prior is $5 \\pm 1.5$ km/s: a marginal that is"
         " the prior is a width the data said nothing about, which is what a"
         " bias that is not there looks like."),
        (figure_d2v(before, after, os.path.join(figures, base + "-d2v.pdf")),
         "%s: d2v, the activity indicator" % name,
         "d2v, LBL's second-derivative term, which follows the line width and"
         " is an activity indicator. Above, over the campaign; below, the"
         " velocity against it. A scatter that falls is telluric noise the"
         " correction took out of it, a gain; one that rises, noise it put"
         " in. A velocity that correlates with it is activity rather than"
         " noise."),
        (figure_dtemp(before, after, os.path.join(figures, base + "-dtemp.pdf")),
         "%s: %s, the temperature projection" % (name, tex(
             before.get("dtemp_name") or "DTEMP")),
         "LBL's projection of every line's residual on the temperature"
         " gradient of the model nearest the star's Teff, delivered and"
         " corrected, on one scale. Above, over the campaign; below, against"
         " $V_\\mathrm{tot}$, with medians in 2 km/s bins. The star's temperature is not"
         " the observer frame's business: a scatter that changes is a"
         " correction reaching into the star, and a structure against BERV"
         " that goes away is a telluric residual DTEMP had picked up."
         " The grey band is $|V_\\mathrm{tot}| < 4$ km/s."),
        (figure_change(before, after, os.path.join(figures, base + "-change.pdf")),
         "%s: what the correction moved" % name,
         "What the correction changed, exposure by exposure: the corrected"
         " velocity less the delivered one, over time and against"
         " $V_\\mathrm{tot}$, the grey band being"
         " $|V_\\mathrm{tot}| < 4$ km/s."),
        (figure_periodograms(before, after,
                             numbers.get("planets_known")
                             or numbers["planet_periods"],
                             os.path.join(figures, base + "-periods.pdf")),
         "%s: periodograms of the velocity and its indicators" % name,
         "Lomb-Scargle periodograms of the velocity, of d2v%s, delivered and"
         " corrected. Each curve's"
         " highest peak is marked by a triangle, with its period, power and"
         " false-alarm probability beside it. Dashed grey lines, named, are a"
         " year, half and a third of a year, and a synodic month (29.53 d);"
         " dotted grey ones, the power a peak needs for a false-alarm"
         " probability of 1\\%%, 0.1\\%% and $10^{-4}$ (Baluev's approximation"
         " over the periods shown, the higher of the two series' levels);"
         " green ones, named, the known planets and TESS candidates"
         " (NASA Exoplanet Archive), and any period the configuration gives."
         " A known planet should keep its peak; a peak at a"
         " year or its harmonics is the Earth's; one shared with d2v or the"
         " temperature is the star's activity."
         % (" and of the temperature projection"
            if before.get("dtemp") is not None else "")),
        (figure_phase(before, after, numbers.get("planets_known"),
                      os.path.join(figures, base + "-phase.pdf")),
         "%s: the velocities folded at the known planets' periods" % name,
         "The velocities folded at the period of each known planet or TESS"
         " candidate (NASA Exoplanet Archive, or the configuration), delivered"
         " on the left and corrected on the right, on one scale; phase 0 is"
         " the transit when the archive gives one, the first exposure"
         " otherwise. Each series is less its straight line in time (fitted"
         " with the error bars as weights), except at a period of a third of"
         " the campaign or more, where the line would take the planet with"
         " it. The squares are weighted means in tenths of the phase; the"
         " line is a sine at that period, its amplitude $K$ in the title with"
         " $\\Delta$BIC = BIC(constant) $-$ BIC(constant + sine), both with a"
         " free jitter ($k$ = 2 against 4), positive when the data prefer the"
         " sine (above 2, 6 and 10: positive, strong, very strong);"
         " dashed, where the archive gives a transit time and a"
         " semi-amplitude, the circular orbit they imply. Under it, the rms"
         " of the residuals to the sine, beside the median error bar: what"
         " the fit leaves, against what the noise alone would. A correction"
         " that left the planet alone keeps $K$."),
    ]
    for path, short, caption in drawn:
        if path:
            # the star first in every caption too: a joint report repeats
            # these figures once per star
            parts.append(figure_block("figures/" + os.path.basename(path),
                                      "\\textbf{%s.} %s" % (name, caption),
                                      short=short))
    parts.append("\\clearpage\n")
    return "".join(parts)


def run_section(config, manifest, folder):
    from .config import WINDOW_SETTINGS, setting_value

    parts = ["\\section{The run}\n"]
    rows = (manifest or {}).get("run") or []
    if rows:
        parts.append("\\begin{longtable}{>{\\raggedright\\arraybackslash}"
                     "p{0.34\\linewidth}>{\\raggedright\\arraybackslash}"
                     "p{0.6\\linewidth}}\n\\toprule\n")
        for key, value in rows:
            parts.append("%s & %s \\\\\n" % (tex(key), tex_break(value)))
        parts.append("\\bottomrule\n\\end{longtable}\n")
    parts.append("\\subsection{Every setting the window can set}\n"
                 "\\begin{longtable}{>{\\raggedright\\arraybackslash}"
                 "p{0.28\\linewidth}>{\\raggedright\\arraybackslash}"
                 "p{0.2\\linewidth}>{\\raggedright\\arraybackslash}"
                 "p{0.44\\linewidth}}\n\\toprule\nsetting & value & what it"
                 " does \\\\\n\\midrule\n\\endhead\n")
    for path, what in WINDOW_SETTINGS:
        value = setting_value(config, path)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, bool) or value is None:
            # as the configuration spells them, not as Python prints them
            value = {True: "true", False: "false", None: "null"}[value]
        parts.append("\\texttt{%s} & %s & %s \\\\\n"
                     % (tex_break(path), tex_break(value), tex(what)))
    parts.append("\\bottomrule\n\\end{longtable}\n")
    provenance = config.get("provenance") or {}
    command = provenance.get("command") or "not recorded (before 2026-09-15)"
    with open(os.path.join(folder, "command.txt"), "w") as handle:
        handle.write(wrap_lines(command, 90))
    parts.append("\\subsection{The command and the code}\n"
                 "The command this run was given:\n{\\footnotesize"
                 "\\verbatiminput{command.txt}}\n"
                 "Code: pca2d %s%s.\n"
                 % (tex((provenance.get("commit") or "unknown")[:12]),
                    tex(" with local changes") if provenance.get("dirty")
                    else ""))
    return "".join(parts)


def run_rows(outdir, cube=None):
    """The run's summary rows, for a run whose report folder has none.

    The figures stage records them since 2026-09-16; before, they were only
    printed on the bound PDF's first page. They are worked out again from the
    run's configuration and its fit, less the cube, which nothing recorded.
    """
    import types

    from .config import load_config
    from .figures.bundle import summary_rows

    path = os.path.join(outdir, "resolved_config.yaml")
    fit_path = os.path.join(outdir, "fit.npz")
    try:
        config = load_config(path)
        fit = np.load(fit_path) if os.path.exists(fit_path) else None
        rows = summary_rows(config, types.SimpleNamespace(
            outdir=outdir, config=path, cube=cube or "not recorded"), fit)
    except Exception as exc:                                   # noqa: BLE001
        log("  the run's summary could not be worked out again (%s)" % exc,
            "warn")
        return []
    if cube is None:
        rows = [(k, v) for k, v in rows
                if k not in ("cube", "this config's cube key")]
    # a block with no component has no shares to print
    rows = [(k, v) for k, v in rows if str(v).strip() not in ("[]", "")]
    return [[str(k), str(v)] for k, v in rows]


#: figures of the figures stage whose every page names its star already
NAMES_ITSELF = ("The sequence, step by step",
                "The rows of the fit: nights and BERV bundles",
                "Every quantity of the correlations, against time")


def _band_page(width, band, name):
    """A page `width` x `band` points holding `name`, as the velocity
    figures carry theirs, for pypdf to lay over another."""
    import io

    from pypdf import PdfReader

    fig = plt.figure(figsize=(width / 72.0, band / 72.0))
    fig.patch.set_alpha(0.0)
    fig.text(0.012 * WIDTH * 72.0 / width, 0.5, name, ha="left", va="center",
             fontsize=12, fontweight="bold", color=INK)
    buffer = io.BytesIO()
    with plt.rc_context(STYLE):
        fig.savefig(buffer, format="pdf", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return PdfReader(buffer).pages[0]


def named_figure(folder, file, name):
    """A figure PDF with `name` in a band above every page.

    Written beside the figure as named-<file>, from the figure itself each
    time, so the band is never laid twice; the figure the figures stage kept
    stays as it was. Returns the new file, relative to `folder` as `file` is.
    """
    from pypdf import PageObject, PdfReader, PdfWriter, Transformation

    head, tail = os.path.split(file)
    out = os.path.join(head, "named-" + tail)
    band = NAME_BAND * 72.0
    writer = PdfWriter()
    stamps = {}
    for page in PdfReader(os.path.join(folder, file)).pages:
        box = page.mediabox
        width, height = float(box.width), float(box.height)
        if round(width, 1) not in stamps:
            stamps[round(width, 1)] = _band_page(width, band, name)
        blank = PageObject.create_blank_page(width=width, height=height + band)
        blank.merge_transformed_page(
            page, Transformation().translate(-float(box.left),
                                             -float(box.bottom)))
        blank.merge_transformed_page(stamps[round(width, 1)],
                                     Transformation().translate(0, height))
        writer.add_page(blank)
    with open(os.path.join(folder, out), "wb") as handle:
        writer.write(handle)
    return out


def correction_section(manifest, windows=(), folder=None, name=None):
    """The figures stage's figures, each with the run's target above it.

    `name` is the target, or the joint set; the figures that name their star
    on every page already (NAMES_ITSELF) are left alone.
    """
    figures = (manifest or {}).get("figures") or []
    if not figures:
        return ("\\section{The correction}\nThe figures stage has not drawn"
                " anything for this run yet.\n")
    parts = ["\\section{The correction}\n"]
    for entry in figures:
        title, path, pages = entry["title"], entry["file"], entry.get("pages", 1)
        if name and folder and title not in NAMES_ITSELF \
                and os.path.exists(os.path.join(folder, path)):
            try:
                path = named_figure(folder, path, name)
            except Exception as exc:                          # noqa: BLE001
                log("could not name %s (%s); it goes in as it is"
                    % (path, exc), "warn")
        parts.append("\\subsection{%s}\n" % tex(title))
        caption = tex(CAPTIONS.get(title, title))
        if pages <= 1:
            parts.append(figure_block(path, caption, short=tex(title)))
        else:
            # the sequence draws one page per window, in the config's order
            named = (list(windows) if title.startswith("The sequence")
                     and len(windows) == pages else [None] * pages)
            for page, window in zip(range(1, pages + 1), named):
                where = (" Window %s (centre:width, nm)." % tex(window)
                         if window else "")
                parts.append(figure_block(
                    path, "%s%s (%d of %d)" % (caption, where, page, pages),
                    short=("%s, window %s" % (tex(title), tex(window))
                           if window else "%s, %d of %d"
                           % (tex(title), page, pages)),
                    page=page))
        parts.append("\\clearpage\n")
    return "".join(parts)


def failure_section(manifest):
    failures = (manifest or {}).get("failures") or []
    if not failures:
        return ""
    parts = ["\\section{What did not build}\n\\begin{itemize}\n"]
    for item in failures:
        parts.append("\\item %s\\\\ \\muted{%s}\n"
                     % (("Not drawn: " if item.get("skipped") else "Failed: ")
                        + tex(item.get("what", "")),
                        tex(item.get("why", ""))[:600]))
    parts.append("\\end{itemize}\n")
    return "".join(parts)


def parameters_appendix(config, folder):
    body = yaml.safe_dump({k: v for k, v in config.items()},
                          sort_keys=False, default_flow_style=False, width=90)
    with open(os.path.join(folder, "parameters.yaml"), "w") as handle:
        handle.write(wrap_lines(body, 96))
    return ("\\appendix\n\\section{The parameters, as resolved}\n"
            "Every key of the configuration this run used, after the defaults,"
            " the instrument, the object and the command line were merged.\n"
            "{\\scriptsize\\verbatiminput{parameters.yaml}}\n")


# =============================================================== the build ===
def compile_tex(folder, name, pdflatex):
    """Run pdflatex twice (the contents need the second pass). True if done."""
    for _ in range(2):
        try:
            done = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 name + ".tex"], cwd=folder, capture_output=True, text=True,
                timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            log("pdflatex could not run: %s" % exc, "warn")
            return False
        if done.returncode != 0:
            lines = [line for line in (done.stdout or "").splitlines()
                     if line.startswith("!") or line.startswith("l.")]
            log("pdflatex stopped on %s.tex: %s"
                % (os.path.join(folder, name), " | ".join(lines[:4])
                   or (done.stdout or "")[-300:]), "warn")
            return False
    return os.path.exists(os.path.join(folder, name + ".pdf"))


def render(outdir, config=None, lbl_dir=None, out=None, stars=None):
    """Write the run's LaTeX report and compile it. Returns the PDF, or None.

    `outdir` is the run's folder. The report is written in <outdir>/report
    and copied to the run's own PDF name, <object>_<tag>.pdf, where the window
    and the run's last line look for it.
    """
    outdir = os.path.abspath(outdir)
    pdflatex = find_pdflatex()
    if pdflatex is None:
        log("no pdflatex on this machine, so the report stays the bound PDF of"
            " the figures. A TeX installation (MacTeX, or BasicTeX) gives the"
            " full report", "warn")
        return None
    if config is None:
        with open(os.path.join(outdir, "resolved_config.yaml")) as handle:
            config = yaml.safe_load(handle)
    folder = os.path.join(outdir, "report")
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    tag = os.path.basename(outdir)
    from .naming import report_name
    name = report_name(config["input"].get("object") or "run", tag,
                       (config.get("provenance") or {}).get("run_hash"))
    out = out or os.path.join(outdir, name + ".pdf")

    manifest = read_manifest(folder)
    if manifest is None and os.path.exists(out):
        log("no report folder in %s: cutting the bound PDF along its"
            " bookmarks to write one" % outdir, "info")
        # the report is about to take its name: the old one is kept whole,
        # so that writing a run's report again can always be undone
        kept = os.path.join(folder, "bound_before_the_report.pdf")
        if not os.path.exists(kept):
            shutil.copyfile(out, kept)
            log("  the bound PDF as it was is kept: %s" % kept, "info")
        manifest = {"figures": split_bundle(out, folder), "failures": [],
                    "run": run_rows(outdir)}
        write_manifest(folder, manifest)

    stars = stars or stars_of(outdir, config)
    # <root>[/_name]/<object>/<tag>, or <root>[/_name]/joint/<A+B>/<tag>
    above = os.path.dirname(os.path.dirname(outdir))
    joint = os.path.basename(above) == "joint"
    run_name = os.path.basename(os.path.dirname(above) if joint else above)
    tree = find_tree(config, outdir, lbl_dir)
    from .lbl import object_names
    results, sections, missing = [], [], []
    facts = {star: star_facts(star, config, folder) for star in stars}
    planets = {}
    # the template each star was measured against: the corrected object
    #   follows the star when it was its own, else keeps the other star's
    against = comparisons(config, stars)
    series = []
    for star in stars:
        before_name, after_name = object_names(config, star, tag)
        own = against[star] == star
        after_against = after_name if own else against[star]
        before = load(tree, before_name, "delivered", against[star])
        after = load(tree, after_name, "corrected", after_against)
        # the report opens on the velocity series, the corrected one when
        #   LBL has measured it
        if after is not None or before is not None:
            series.append((star, after if after is not None else before))
        if before is None or after is None:
            missing.append((star, rdb_file(tree, before_name, against[star])
                            if before is None else
                            rdb_file(tree, after_name, after_against)))
            continue
        before, after = common(before, after)
        # whose velocities these are, for the figures' own titles
        before["star"] = after["star"] = star
        if before["t"].size < 4:
            missing.append((star, "fewer than four exposures in common"))
            continue
        known, status = known_planets(outdir, config, star, joint,
                                      facts[star], folder)
        planets[star] = (known, status)
        import zlib
        numbers = star_numbers(before, after, [p["period"] for p in known],
                               seed=zlib.crc32(star.encode()) & 0xffffffff)
        numbers["planet_names"] = [p.get("short") or p["name"] for p in known]
        numbers["planets_known"] = known
        before["fits"] = {"before": numbers["before"].get("bias"),
                          "after": numbers["after"].get("bias")}
        newest = newest_input(outdir, star, joint)
        stale = newest is not None and after["mtime"] < newest
        results.append((star, numbers, stale))
        sections.append(velocity_section(star, before, after, numbers, folder,
                                         stale))

    for star in stars:
        if star not in planets:
            planets[star] = known_planets(outdir, config, star, joint,
                                          facts[star], folder)
    # THE VELOCITY SERIES FIRST, right after the contents: what one opens a
    #   report for (2026-10-24)
    body = series_section(series, folder)
    body.append("\\section{The star%s}\n" % ("s" if len(stars) > 1 else ""))
    body.append("As SIMBAD describes %s, found from the names in the"
                " spectra's headers and the folder's own.\n"
                % ("them" if len(stars) > 1 else "it"))
    for star in stars:
        body.append(star_table(star, facts[star], planets[star]))
    body.append("\\section{Summary}\n")
    if results:
        body.append("One table per star: the delivered and corrected velocities"
                    " LBL measured, on the exposures both have. The last"
                    " column says which way each number went: \\gain{gain} or"
                    " \\loss{loss} where lower is better, \\watch{changed}"
                    " where the correction should have changed nothing.\n")
        for star, numbers, _stale in results:
            better, worse, moved = verdict_lines(numbers)
            items = []
            if better:
                items.append("\\item \\gain{Better}: %s." % ", ".join(better))
            if worse:
                items.append("\\item \\loss{Worse}: %s." % ", ".join(worse))
            if moved:
                items.append("\\item \\watch{Changed where nothing should}:"
                             " %s." % "; ".join(moved))
            note = activity_note(numbers)
            if note:
                items.append("\\item %s" % note)
            body.append("\\subsection*{%s}\n" % tex(star))
            if items:
                body.append("\\begin{itemize}\n%s\n\\end{itemize}\n"
                            % "\n".join(items))
            body.append(observations_table(star, numbers))
            body.append(summary_table(star, numbers))
    for star, why in missing:
        body.append("\\muted{%s: no velocities to compare yet (%s).}\n\n"
                    % (tex(star), tex(why)))
    if sections:
        body.append("\\clearpage\n\\section{Velocities}\n")
        body.extend(sections)
    body.append(run_section(config, manifest, folder))
    body.append("\\clearpage\n")
    body.append(correction_section(
        manifest, (manifest or {}).get("windows")
        or (config.get("output") or {}).get("windows") or (),
        folder=folder, name=" + ".join(stars)))
    body.append(failure_section(manifest))
    body.append(parameters_appendix(config, folder))

    tw = config.get("twoframe") or {}
    from .config import describe_star
    frozen = ""
    if tw.get("earth_basis"):
        try:
            from .basis import Basis
            frozen = ", FROZEN: those of %s" % Basis(tw["earth_basis"]).describe()
        except Exception:                                 # noqa: BLE001
            frozen = ", FROZEN: those of %s" % tw["earth_basis"]
    what = ("%s; %d observer components divided out of the spectra%s; %s"
            % (describe_star(tw.get("n_star", 0)), tw.get("n_earth", 0),
               frozen, "one observer basis for %s" % ", ".join(stars)
               if len(stars) > 1 else "on %s alone" % stars[0]))
    named = ["%s (%s)" % (star, star_label(facts[star]))
             if star_label(facts[star]) else star for star in stars]
    abstract = [tex("Two-frame PCA correction of %s: %s."
                    % (" and ".join(named), what))]
    abstract.extend(star_headline(star, numbers) for star, numbers, _ in results)
    if not results:
        abstract.append("LBL has not measured this run yet, so the report"
                        " shows the correction and not its effect on the"
                        " velocities.")
    if any(stale for _, _, stale in results):
        abstract.append("\\watch{Some velocities predate the corrected spectra;"
                        " see their section.}")

    with open(TEMPLATE) as handle:
        document = handle.read()
    now = datetime.datetime.now().astimezone()
    fill = {
        "TITLE": tex(" + ".join(stars)),
        "SUBTITLE": tex("pca2d-preclean run %s%s" % (
            tag, ", %s" % run_name.lstrip("_") if run_name.startswith("_")
            else "")),
        # the date, and under it the time to the second: two reports of one
        # day are told apart by it
        "DATE": "%s\\\\\n{\\small written at %s}" % (
            tex(now.date().isoformat()),
            tex(now.strftime("%H:%M:%S %Z (UTC%z)"))),
        "RUNHEAD": tex("%s, %s" % (" + ".join(stars), tag)),
        "ABSTRACT": "\n\n".join(abstract),
        "BODY": "".join(body),
    }
    for key, value in fill.items():
        document = document.replace("<<%s>>" % key, value)
    with open(os.path.join(folder, name + ".tex"), "w") as handle:
        handle.write(document)
    if not compile_tex(folder, name, pdflatex):
        return None
    shutil.copyfile(os.path.join(folder, name + ".pdf"), out)
    log("report written: %s (%d star%s with velocities)"
        % (out, len(results), "" if len(results) == 1 else "s"), "value")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m lbl.pca2d.texreport",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="the run's folder: <output root>/<object>/<tag>")
    p.add_argument("--lbl-dir", default=None,
                   help="LBL's tree, when the run's configuration does not say"
                        " where it is")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    done = render(args.run, lbl_dir=args.lbl_dir)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

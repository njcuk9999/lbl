"""The runs of a data folder, read back from disk, never remembered by the
window: what the window ran (DATA_DIR/gui/runs/*/run.json), the PCA2D
corrections on disk (DATA_DIR/pca2d/**/resolved_config.yaml, the record
lbl_pca2d writes with every fit) and LBL's reports (DATA_DIR/reports/*.pdf).
A run made from a terminal shows up as well as one made from the window.

When the data folder is one of a data root's (ROOT/{INSTRUMENT}_{SOURCE},
lbl.gui.dispatch), the runs and reports of all its folders are listed
(homes), each with its folder: a run does not vanish from the list when
the window moves to another instrument.
"""
from __future__ import annotations

import glob
import json
import os

#: where the window keeps its runs, under DATA_DIR
GUI_RUNS = os.path.join("gui", "runs")
#: THE WINDOW'S LOGS, apart from its runs' folders: each run's
#:   (<run>.log) and the archive search's and fetch's (archives.log).
#:   LBL's own log/ folder stays LBL's.
GUI_LOGS = os.path.join("gui", "logs")
#: LBL's reports, under DATA_DIR (lbl.report.main.FOLDER)
LBL_REPORTS = "reports"
#: what the objects of the spectra PCA2D corrected end with
#:   (lbl.recipes.lbl_pca2d.SUFFIX), and so the names of LBL's reports on them
PCA2D_MARK = "_pca2d"
#: PDFs of a PCA2D run folder that are not its report: the components'
#:   figures, and the report's copy in its report/ folder
PCA2D_EXTRA = ("components.pdf",)
PCA2D_COPIES = "report"


def run_log(folder: str) -> str:
    """The log of the run whose folder is gui/runs/<run>: gui/logs/<run>.log
    (a run.log found in the run's own folder is moved there)."""
    logs = os.path.join(os.path.dirname(os.path.dirname(folder)), "logs")
    path = os.path.join(logs, os.path.basename(folder.rstrip(os.sep)) + ".log")
    old = os.path.join(folder, "run.log")
    if os.path.exists(old) and not os.path.exists(path):
        try:
            os.makedirs(logs, exist_ok=True)
            os.replace(old, path)
        except OSError:
            return old
    return path


def window_runs(data_dir: str) -> list:
    out = []
    for path in glob.glob(os.path.join(data_dir, GUI_RUNS, "*", "run.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        record["folder"] = os.path.dirname(path)
        record["log"] = run_log(record["folder"])
        record["reports"] = final_reports(record.get("reports") or [])
        out.append(record)
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return out


def final_reports(paths: list) -> list:
    """The report a run is shown with, among the PDFs it wrote: LBL's report
    of the spectra PCA2D corrected (reports/<object>_pca2d..._<template>.pdf),
    one per star; when it corrected none, LBL's report of the spectra as
    delivered; when LBL wrote none, PCA2D's own report. Never the
    components' figures (components.pdf), nor the copy of PCA2D's report in
    its report/ folder."""
    lbl = [p for p in paths
           if os.path.basename(os.path.dirname(p)) == LBL_REPORTS]
    if lbl:
        return [p for p in lbl if PCA2D_MARK in os.path.basename(p)] or lbl
    return [p for p in paths if os.path.basename(p) not in PCA2D_EXTRA
            and os.path.basename(os.path.dirname(p)) != PCA2D_COPIES]


def homes(data_dir: str) -> list:
    """The data folders whose runs are listed: DATA_DIR, and when it is one
    of a root's, the root's other folders and the root itself (the old
    layout), as (folder, its name in the root: '' for the root); DATA_DIR's
    first."""
    from .dispatch import readers, split
    root, mine = split(data_dir)
    if not mine:
        return [(os.path.abspath(data_dir or ""), "")]
    out = [(os.path.join(root, name), name) for name in sorted(readers())
           if name == mine or os.path.isdir(os.path.join(root, name, GUI_RUNS))
           or glob.glob(os.path.join(root, name, "reports", "*.pdf"))]
    if os.path.isdir(os.path.join(root, GUI_RUNS)):
        out.append((root, ""))
    return sorted(out, key=lambda h: h[1] != mine)


def root_runs(data_dir: str) -> list:
    """The runs of every folder of homes(data_dir), newest first, each with
    its data_dir and home (its folder's name in the root)."""
    out = []
    for folder, home in homes(data_dir):
        for record in window_runs(folder):
            record.update(data_dir=folder, home=home)
            out.append(record)
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return out


def root_reports(data_dir: str) -> list:
    """LBL's reports of every folder of homes(data_dir), newest first, each
    with its home."""
    out = []
    for folder, home in homes(data_dir):
        for report in reports(folder):
            report["home"] = home
            out.append(report)
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def reports(data_dir: str) -> list:
    """LBL's reports: dict(name, path, mtime, size)."""
    out = []
    for path in glob.glob(os.path.join(data_dir, "reports", "*.pdf")):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out.append(dict(name=os.path.basename(path), path=path,
                        mtime=stat.st_mtime, size=stat.st_size))
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out

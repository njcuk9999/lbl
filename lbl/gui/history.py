"""The runs of a data folder, read back from disk, never remembered by the
window: what the window ran (DATA_DIR/gui/runs/*/run.json), the PCA2D
corrections on disk (DATA_DIR/pca2d/**/resolved_config.yaml, the record
lbl_pca2d writes with every fit) and LBL's reports (DATA_DIR/reports/*.pdf).
A run made from a terminal shows up as well as one made from the window.
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
        out.append(record)
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
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

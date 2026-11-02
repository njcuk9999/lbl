"""Every run an output root holds, read from what the runs left in it.

    python -m lbl.pca2d.runs /path/to/output_root

A run writes `resolved_config.yaml` into its own folder before its first
stage: every setting as it ended up, the command it was given word for word,
the code it ran, and the time it started. That file is therefore the record
of the run, and this walks an output root for them, newest first, so that
the window can list what has been tried, with what, and where each one's
compilation PDF is.

Nothing here reads a cube or a spectrum: a root of a hundred runs is read in
a fraction of a second, and a folder still being written answers with what it
has.
"""

from __future__ import annotations

import os
import sys

#: folders a run fills and no run is ever written inside
SKIP = {"corrected", "report", "figures", "lbl", "cache", "spill", "science",
        "lblrv", "lblrdb", "lblreftable", "templates", "masks", "models",
        "calib", "log", "plots", "tmp", "__pycache__"}
NAME = "resolved_config.yaml"
#: how deep under the root a run can sit: <root>/_<name>/joint/<A+B>/<tag>
DEPTH = 5


def folders(root, depth=DEPTH):
    """Every folder under `root` that holds a resolved configuration."""
    root = os.path.abspath(os.path.expanduser(str(root or "")))
    if not os.path.isdir(root):
        return []
    found = []
    for here, subs, files in os.walk(root):
        level = here[len(root):].count(os.sep)
        subs[:] = [] if level >= depth else [
            s for s in subs if s not in SKIP and not s.startswith(".")]
        if NAME in files:
            found.append(here)
            subs[:] = []                 # a run holds no other run
    return found


def report_pdf(folder, config):
    """The compilation PDF of a run, or None.

    `<object>_<tag>_<hash>.pdf` as cli and texreport name it, the object
    being the joint set for a joint run; `<object>_<tag>.pdf` for a run whose
    name carries no hash.
    """
    from .naming import report_name

    name = ((config.get("input") or {}).get("object")
            or os.path.basename(os.path.dirname(folder)))
    tag = os.path.basename(folder)
    stamp = (config.get("provenance") or {}).get("run_hash")
    for stem in (report_name(name, tag, stamp), report_name(name, tag)):
        path = os.path.join(folder, stem + ".pdf")
        if os.path.exists(path):
            return path
    return None


def run_hash(provenance):
    """Which run this is: the hash its name carries, or, for a run whose
    name carries none, the one its own command gives."""
    from .naming import run_hash as hashed

    recorded = (provenance or {}).get("run_hash")
    if recorded:
        return str(recorded)
    command = (provenance or {}).get("command")
    if not command:
        return ""
    import shlex
    try:
        return hashed(shlex.split(command))
    except ValueError:
        return ""


def one(folder):
    """What a run's folder says about it, as a dictionary."""
    import yaml

    path = os.path.join(folder, NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:                                          # noqa: BLE001
        config = {}
    provenance = config.get("provenance") or {}
    twoframe = config.get("twoframe") or {}
    objects = str((config.get("input") or {}).get("object") or "")
    return {"folder": folder, "config": config, "hash": run_hash(provenance),
            "objects": objects, "targets": objects.split("+"),
            "tag": os.path.basename(folder),
            "name": os.path.basename(os.path.dirname(folder)),
            "started": provenance.get("started")
            or _from_disk(path),
            "command": provenance.get("command") or "",
            "commit": provenance.get("commit"),
            "dirty": bool(provenance.get("dirty")),
            # n_star counts the star's spectra with the template; the
            #   configuration counts the free components
            "components": "n_star %s + %s observer" % (
                int(twoframe["n_star"]) + 1 if "n_star" in twoframe else "?",
                twoframe.get("n_earth", "?")),
            "report": report_pdf(folder, config)}


def _from_disk(path):
    """When a run started, for one with no provenance.started: when its
    resolved configuration was written, which is the same moment."""
    import datetime

    try:
        when = os.path.getmtime(path)
    except OSError:
        return ""
    return datetime.datetime.fromtimestamp(when).astimezone().isoformat(
        timespec="seconds")


def listed(root, depth=DEPTH):
    """Every run under `root`, the most recent first."""
    out = [one(folder) for folder in folders(root, depth)]
    out.sort(key=lambda run: (run["started"] or "", run["folder"]),
             reverse=True)
    return out


def settings(run):
    """[(setting, value)] of one run: every setting the window can set, as
    that run resolved it, then what it was given and what code ran."""
    from .config import WINDOW_SETTINGS, setting_value

    config = run.get("config") or {}
    rows = [("run hash", run.get("hash") or "not recorded"),
            ("targets", run["objects"]),
            ("components (star + observer)", run["components"]),
            ("started", run["started"] or "not recorded"),
            ("folder", run["folder"])]
    for path, _what in WINDOW_SETTINGS:
        value = setting_value(config, path)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        rows.append((path, str(value)))
    if run.get("commit"):
        rows.append(("code", "%s%s" % (run["commit"][:10],
                                       ", edited" if run["dirty"] else "")))
    rows.append(("compilation PDF", run.get("report") or "not written yet"))
    return rows


def main(argv=None):
    for root in (argv if argv is not None else sys.argv[1:]):
        for run in listed(root):
            print("%-8s %-20s %-26s %-8s %s"
                  % (run["hash"], run["started"][:19], run["objects"],
                     run["tag"], run["report"] or "(no PDF)"))


if __name__ == "__main__":
    main()

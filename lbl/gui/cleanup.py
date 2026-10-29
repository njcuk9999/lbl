"""What a data folder's products weigh, and deleting those that can be made
again.

Every product is put in one kind, which says what deleting it costs:

    scratch      remade without being asked, by the next run that needs it
                 (PCA2D's registered cubes)
    rebuildable  remade by running again, in minutes (superseded
                 corrections, corrected and telluric-cleaned spectra, LBL's
                 per-exposure velocities, templates, masks)
    results      what a run is for, and what is kept: never deleted here
                 (rdb tables, reports, observer bases, the fits' own
                 folders, models/: LBL's models, the vetted templates, the
                 kept PCA2D bases)
    data         the spectra themselves: never touched

Only scratch and rebuildable can be deleted from the window, and only inside
the data folder. A link is removed as a link: what it points to (spectra
kept on another disk, a scratch volume) is left where it is.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

from .targets import DERIVED

#: the kinds that can be deleted
DELETABLE = ("scratch", "rebuildable")


def size_of(path: str) -> tuple:
    """(bytes, files) under a path, links not followed."""
    if os.path.islink(path) or os.path.isfile(path):
        try:
            return os.lstat(path).st_size, 1
        except OSError:
            return 0, 0
    total = count = 0
    for base, _, names in os.walk(path):
        for name in names:
            full = os.path.join(base, name)
            try:
                total += os.lstat(full).st_size
                count += 1
            except OSError:
                pass
    return total, count


#: what each product is, by its code (the page says it in its language:
#:    i18n's what_<code>)
WHAT = dict(
    cache="PCA2D's registered cubes, remade by the next fit",
    superseded="a correction replaced by a newer one, and what LBL made"
               " from it",
    spectra="the spectra of a target",
    corrected="corrected spectra (remade from the fit; tar them first if"
              " they are someone's)",
    cleaned="telluric-cleaned spectra",
    lblrv="LBL's per-exposure velocities (about ten minutes)",
    template="an LBL template",
    mask="an LBL mask",
    rdb="LBL's velocity tables",
    reports="the reports",
    components="observer bases and stars' components",
    models="LBL's models: stellar models, TAPAS, vetted templates, kept"
           " PCA2D bases")


def _item(path: str, kind: str, code: str) -> dict:
    nbytes, files = size_of(path)
    link = os.path.islink(path)
    return dict(path=path, kind=kind, code=code, what=WHAT[code],
                bytes=nbytes, files=files,
                link=os.path.realpath(path) if link else None)


def survey(data_dir: str) -> list:
    """Everything the window can say about a data folder's products, the
    heaviest first in each kind."""
    data_dir = os.path.abspath(data_dir)
    out = []
    cache = os.path.join(data_dir, "pca2d", "cache")
    if os.path.exists(cache):
        out.append(_item(cache, "scratch", "cache"))
    for path in glob.glob(os.path.join(data_dir, "pca2d", "**", "superseded",
                                       "*"), recursive=True):
        out.append(_item(path, "rebuildable", "superseded"))
    science = os.path.join(data_dir, "science")
    for path in glob.glob(os.path.join(science, "*")):
        name = os.path.basename(path)
        if not DERIVED.search(name):
            out.append(_item(path, "data", "spectra"))
        elif "_pca2d" in name:
            out.append(_item(path, "rebuildable", "corrected"))
        else:
            out.append(_item(path, "rebuildable", "cleaned"))
    for folder, code in (("lblrv", "lblrv"), ("templates", "template"),
                         ("masks", "mask")):
        for path in glob.glob(os.path.join(data_dir, folder, "*")):
            out.append(_item(path, "rebuildable", code))
    for folder, code in (("lblrdb", "rdb"), ("reports", "reports"),
                         (os.path.join("pca2d", "components"),
                          "components"), ("models", "models")):
        path = os.path.join(data_dir, folder)
        if os.path.exists(path):
            out.append(_item(path, "results", code))
    order = {"scratch": 0, "rebuildable": 1, "results": 2, "data": 3}
    out.sort(key=lambda r: (order[r["kind"]], -r["bytes"]))
    return out


def busy(data_dir: str) -> list:
    """The Python processes working in this data folder (their current
    folder in it), the window's own server aside: a run started from a
    terminal reads the cubes and the corrected spectra the cleanup would
    delete, and nothing else says it is going. [dict(pid, command)].

    When this system gives no way to tell (neither psutil nor lsof), one
    dict(pid=None, command="", unknown=True): nothing found is not nothing
    going, and the cleanup refuses to delete on it."""
    found = _working_in(data_dir)
    if found is None:
        return [dict(pid=None, command="", unknown=True)]
    procs = []
    for pid, command in sorted(found.items()):
        # the multiprocessing helpers of a run are the run, and a window
        #   is not one; nor is an interpreter someone works in
        if "resource_tracker" in command or "lbl_gui" in command \
                or "lbl-gui" in command:
            continue
        if not is_run(command):
            continue
        procs.append(dict(pid=pid, command=command[:200]))
    return procs


def _inside(path: str, root: str) -> bool:
    """Whether `path` is `root` or in it, links resolved (and case ignored
    where the system ignores it, Windows)."""
    path = os.path.normcase(os.path.realpath(path))
    return path == root or path.startswith(os.path.join(root, ""))


def _working_in(data_dir: str):
    """{pid: command line} of the Python processes, this one aside, whose
    current folder is `data_dir` or in it; None when this system gives no
    way to tell. psutil where it is installed (every system), lsof and ps
    otherwise (macOS, Linux)."""
    root = os.path.normcase(os.path.realpath(data_dir))
    try:
        import psutil
    except ImportError:
        psutil = None
    found = {}
    if psutil is not None:
        for proc in psutil.process_iter(["name"]):
            name = str(proc.info.get("name") or "").lower()
            if proc.pid == os.getpid() or not name.startswith("python"):
                continue
            try:
                if _inside(proc.cwd(), root):
                    found[proc.pid] = " ".join(proc.cmdline())
            except (psutil.Error, OSError):
                # ended meanwhile, or another user's
                continue
        return found
    try:
        out = subprocess.run(["lsof", "-a", "-d", "cwd", "-c", "python",
                              "-Fpn"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pid = None
    for line in out.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None:
            if pid != os.getpid() and _inside(line[1:], root):
                found[pid] = ""
    for pid in found:
        try:
            found[pid] = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return found


#: interpreters a person works in: their current folder says where they
#:    are, not that something is being computed there
INTERACTIVE = ("ipython", "jupyter", "ipykernel", "bpython", "ptpython")


def is_run(command: str) -> bool:
    """Whether a Python process runs something (a script, a module, code
    given with -c) rather than waits for someone (a bare python, ipython,
    a Jupyter kernel). An ipython left open in a target's folder would
    otherwise hold back every cleanup for as long as it stays open."""
    words = str(command or "").split()
    if not words:
        return False
    if any(any(name in os.path.basename(word) for name in INTERACTIVE)
           for word in words[:3]):
        return False
    rest = words[1:]
    if "-i" in rest:
        return False
    return any(word.endswith(".py") or word in ("-m", "-c")
               for word in rest)


def delete(data_dir: str, paths: list) -> list:
    """Delete these paths if each is a scratch or rebuildable product of
    this data folder, and nothing is running in it; returns [dict(path,
    done, why)]."""
    running = busy(data_dir)
    if any(p.get("unknown") for p in running):
        why = ("whether a run is going in this folder cannot be told on this"
               " system (install psutil): nothing is deleted")
        return [dict(path=path, done=False, why=why) for path in paths]
    if running:
        why = "a run is going in this folder (pid %s): nothing is deleted" \
            % ", ".join(str(p["pid"]) for p in running)
        return [dict(path=path, done=False, why=why) for path in paths]
    allowed = {item["path"]: item for item in survey(data_dir)
               if item["kind"] in DELETABLE}
    root = os.path.abspath(data_dir) + os.sep
    out = []
    for path in paths:
        item = allowed.get(path)
        if item is None or not os.path.abspath(path).startswith(root):
            out.append(dict(path=path, done=False,
                            why="not a product the window may delete"))
            continue
        try:
            if os.path.islink(path) or os.path.isfile(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
            out.append(dict(path=path, done=True, why=""))
        except OSError as exc:
            out.append(dict(path=path, done=False, why=str(exc)))
    return out

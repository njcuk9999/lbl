"""Which code made a run: the git commit, and whether it had local changes.

Written into every run's resolved configuration, its fit.npz, the headers of
its corrected spectra and its LBL template, so that a result can be traced to
the code that produced it and made again.
"""
import functools
import os
import subprocess

from . import __version__

HERE = os.path.dirname(os.path.abspath(__file__))


def _git(*args):
    try:
        r = subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                           text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def code_version():
    """{"commit": the full hash, None outside a git checkout; "dirty": True
    when a tracked file differs from that commit; "version": the package's}.

    Asked once per process: the code that started a run is the code that made
    it, and a correct stage writing hundreds of files asks for every one."""
    commit = _git("rev-parse", "HEAD")
    dirty = None
    if commit:
        # tracked files only: a scratch file beside the code changes nothing
        status = _git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status) if status is not None else None
    return {"commit": commit, "dirty": dirty, "version": __version__}


def stamp(version=None):
    """The short form, for a FITS card: 12 hex digits, '+' when a tracked file
    had changed, 'unknown' outside a git checkout."""
    v = version or code_version()
    if not v["commit"]:
        return "unknown"
    return v["commit"][:12] + ("+" if v["dirty"] else "")


#: modules a run never needs: the command itself, which is already running, and
#: the window, which would bring tkinter into a run that has no screen
#: the package's own dotted name, lbl.pca2d since it moved into LBL
PACKAGE = __name__.rsplit(".", 1)[0]
NOT_A_STAGE = tuple("%s.%s" % (PACKAGE, name) for name in ("cli", "gui", "__main__"))


def freeze(package=PACKAGE, skip=NOT_A_STAGE):
    """Load every module of the package NOW. Returns the source fingerprint.

    A stage imports what it needs when it starts, and a run lasts hours. On
    2026-09-16 the code was edited while a joint run was fitting: config.py had
    been loaded at 10:06, lbl.py was loaded for the first time at 10:52, and
    the new lbl.py asked the old config.py for a function it did not have. The
    run died at the LBL stage with a traceback whose lines did not match its
    line numbers, since Python prints the file as it is on disk NOW.

    Loaded together at the start, the run is one version of the code from its
    first line to its last, whatever happens on disk meanwhile. About a second
    for the package's forty modules. A module that fails to load here is left
    to fail where it is used, as it did before.
    """
    import importlib

    # the files, not pkgutil: figures/ has no __init__.py, so pkgutil never
    # walks into it, and figures.bundle is exactly what a later stage imports
    for path in sorted(fingerprint()):
        parts = path[:-len(".py")].split(os.sep)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join([package] + parts)
        if name in skip or name == package:
            continue
        try:
            importlib.import_module(name)
        except Exception:                                       # noqa: BLE001
            pass
    return fingerprint()


def fingerprint(root=HERE):
    """{path under `root`: (size, mtime)} of every Python source in it."""
    out = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            full = os.path.join(base, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            out[os.path.relpath(full, root)] = (stat.st_size, stat.st_mtime_ns)
    return out


def drift(snapshot, root=HERE):
    """The sources that changed, appeared or went since `snapshot`, sorted."""
    now = fingerprint(root)
    return sorted(path for path in set(snapshot) | set(now)
                  if snapshot.get(path) != now.get(path))

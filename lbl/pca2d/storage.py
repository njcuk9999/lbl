"""Where a run's products are kept, when that is not the internal disk.

With output.fits_directory set, a run's folder is not a folder on the internal
disk but a link to its place under that directory, at the same relative path:

    outputs/TOI2120/2-7v                   -> <fits_directory>/outputs/TOI2120/2-7v
    lbl/lblrv, lbl/lblrdb, lbl/models, ... -> <fits_directory>/lbl/...

The link is made before the run writes anything, so everything it writes (the
corrected spectra, the fit's components, the figures) lands on that disk
directly and never passes through the internal one. Whole folders are linked,
never single files: a link to one FITS would not survive astropy, which, asked
to overwrite, removes the link and writes a real file where it was.

lbl/ itself stays a real folder, because lbl/science is made of symlinks and
an exFAT disk cannot store one. Every other folder LBL writes in is linked.
"""

from __future__ import annotations

import filecmp
import os
import shutil

from .logger import log

# Every folder LBL writes in (its *_SUBDIR parameters, and the ones it makes
# for models, plots and logs), except science: see above.
LBL_FOLDERS = ("lblrv", "lblrdb", "lblreftable", "templates", "masks",
               "models", "calib", "plots", "log")


def root(config: dict):
    """output.fits_directory, or None when everything stays local."""
    return (config.get("output") or {}).get("fits_directory") or None


def check(config: dict, dry_run: bool = False):
    """Stop the run if a FITS directory is set and cannot be written to.

    Stopping is the point: falling back to the local tree would do, quietly,
    exactly what this setting exists to prevent. The directory itself is
    created when its parent is there. The parent never is: a missing
    /Volumes/<disk> means the disk is not mounted, and creating it would put
    the files on the internal disk under a name that looks external. A dry run
    only warns, and creates nothing.
    """
    where = root(config)
    if not where:
        return None
    parent = os.path.dirname(os.path.normpath(where))
    if not os.path.isdir(where) and not os.path.isdir(parent):
        why = ("output.fits_directory is %s, but %s does not exist: is the disk"
               " mounted?" % (where, parent))
        if dry_run:
            log(why + " A real run would stop here.", "warn")
            return None
        log(why + " Stopping here rather than write the run on the internal"
            " disk.", "error")
        raise SystemExit(2)
    if not dry_run:
        os.makedirs(where, exist_ok=True)
        if not os.access(where, os.W_OK):
            log("output.fits_directory %s is not writable" % where, "error")
            raise SystemExit(2)
    return where


def external(config: dict, local: str) -> str:
    """Where `local` lives under the FITS directory: the same relative path."""
    full = os.path.abspath(local)
    rel = os.path.relpath(full)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        rel = full.lstrip(os.sep)
    return os.path.join(root(config), rel)


def _within(path: str, top: str) -> bool:
    """Whether `path`, or the nearest part of it that exists, resolves into `top`."""
    probe = os.path.abspath(path)
    while not os.path.lexists(probe):
        probe = os.path.dirname(probe)
    real, top = os.path.realpath(probe), os.path.realpath(top)
    return real == top or real.startswith(top + os.sep)


def _inventory(local: str, target: str):
    """The real files under `local`, and the links in it already pointing out.

    A link is accepted only when it points exactly to its own place under
    `target`, which is what the first move of these FITS left behind: that
    file is already where it is going. Any other link stops everything, since
    the disk it would have to move to cannot store it.
    """
    files, placed = {}, []
    for base, dirs, names in os.walk(local):
        for name in dirs + names:
            path = os.path.join(base, name)
            if not os.path.islink(path):
                continue
            rel = os.path.relpath(path, local)
            home = os.path.join(target, rel)
            if not (os.path.exists(path)
                    and os.path.realpath(path) == os.path.realpath(home)):
                log("%s is a link to %s, and the disk behind"
                    " output.fits_directory cannot store a link, so %s is left"
                    " as it is and the run stops" % (path, os.readlink(path),
                                                     local), "error")
                raise SystemExit(2)
            placed.append(rel)
        for name in names:
            path = os.path.join(base, name)
            if not os.path.islink(path):
                files[os.path.relpath(path, local)] = os.path.getsize(path)
    return files, placed


def link_dir(config: dict, local: str) -> str:
    """Make `local` a link to its place under the FITS directory.

    Returns where writes into `local` will land. A no-op without a FITS
    directory, and when `local` sits in a folder that is already out there. A
    folder that exists for real, from before the setting, is moved first:
    every file copied and compared byte for byte, then the folder swapped for
    the link, and the swap undone if the link does not show every file.
    """
    if not root(config):
        return local
    target = external(config, local)
    if os.path.islink(local):
        if os.path.realpath(local) != os.path.realpath(target):
            log("%s already links to %s, not to %s; left as it is, and the run"
                " stops rather than guess which is meant"
                % (local, os.readlink(local), target), "error")
            raise SystemExit(2)
        os.makedirs(target, exist_ok=True)
        return target
    if _within(os.path.dirname(os.path.abspath(local)), root(config)):
        return local
    os.makedirs(target, exist_ok=True)
    if not os.path.isdir(local):
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        os.symlink(target, local)
        log("%s is a link to %s" % (local, target), "info")
        return target

    files, placed = _inventory(local, target)
    for rel, nbytes in files.items():
        src, dest = os.path.join(local, rel), os.path.join(target, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        if os.path.getsize(dest) != nbytes or not filecmp.cmp(src, dest,
                                                              shallow=False):
            log("the copy of %s at %s does not match; nothing moved"
                % (src, dest), "error")
            raise SystemExit(2)
    moving = local + ".__moving__"
    os.rename(local, moving)
    try:
        os.symlink(target, local)
        missing = [rel for rel in list(files) + placed
                   if not os.path.exists(os.path.join(local, rel))]
        if missing:
            raise RuntimeError("not there through the link: %s" % missing[:3])
    except Exception:
        if os.path.islink(local):
            os.remove(local)
        os.rename(moving, local)
        raise
    # removes the links inside without following them, so what they point to
    # (the files already out there) is untouched
    shutil.rmtree(moving)
    log("%s moved to %s (%d files, %.1f MB, and %d already there) and linked"
        % (local, target, len(files), sum(files.values()) / 1e6, len(placed)),
        "value")
    return target

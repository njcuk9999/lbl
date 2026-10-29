"""Everything pca2d-preclean keeps between runs, in one labelled place.

    <output.cache_directory>/          `cache/` by default
        README.txt                     what this is, and that deleting it is safe
        cube_<format>_<key>/           one registered cube per configuration
            grid.npy data.npy sigma.npy trans.npy meta.fits cube_config.yaml
            snippets/                  ln(flux) around each figure window
        telluric_<key>.npz             telluric optical depth per unit airmass

Nothing here is a result. Every file is rebuilt from the spectra when it is
missing, so the whole folder can be deleted at any time, and --clean-cache does
exactly that, saying what it removes and how much room that gives back.
"""

from __future__ import annotations

import os
import shutil

from .logger import log
from .progress import size

README = """\
This is pca2d-preclean's cache: intermediates it rebuilds from the spectra
whenever they are missing. Nothing in it is a result.

    cube_<format>_<key>/      one registered cube per configuration that
                              defines one; <key> hashes that configuration
        snippets/             ln(flux) around each figure window, so that
                              drawing the figures does not reopen every spectrum
    telluric_<key>.npz        telluric optical depth per unit airmass

Deleting any of it is safe. `pca2d-preclean --object NAME --clean-cache` does
it for you and lists what it removed; with --dry-run it only lists.
"""

#: What this package writes here. Anything else in the folder is left alone.
OURS = ("cube_", "telluric_")


def root(config: dict) -> str:
    return config["output"]["cache_directory"]


def ensure(config: dict) -> str:
    """The cache directory, created with its README the first time."""
    directory = root(config)
    os.makedirs(directory, exist_ok=True)
    readme = os.path.join(directory, "README.txt")
    if not os.path.exists(readme):
        with open(readme, "w") as handle:
            handle.write(README)
    return directory


def _bytes(path: str) -> int:
    if os.path.islink(path) or os.path.isfile(path):
        return os.lstat(path).st_size
    total = 0
    for base, _, names in os.walk(path):
        for name in names:
            full = os.path.join(base, name)
            if not os.path.islink(full):
                total += os.path.getsize(full)
    return total


def entries(config: dict) -> list:
    """(path, bytes) of everything in the cache that this package wrote."""
    directory = root(config)
    if not os.path.isdir(directory):
        return []
    return [(os.path.join(directory, name), _bytes(os.path.join(directory, name)))
            for name in sorted(os.listdir(directory)) if name.startswith(OURS)]


def clean(config: dict, dry_run: bool = False) -> tuple:
    """Remove what entries() lists and say so. Returns (count, bytes).

    Only names this package writes are touched, never the folder itself and
    never anything else a person may have put in it, and a symlink is removed
    as a link rather than followed.
    """
    found = entries(config)
    total = sum(n for _, n in found)
    if not found:
        log("cache %s holds nothing of ours to remove" % root(config), "info")
        return 0, 0
    for path, nbytes in found:
        log("  %s %s (%s)" % ("would remove" if dry_run else "removing", path,
                              size(nbytes)), "warn")
        if dry_run:
            continue
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    log("%s %d cache entries, %s" % ("would free" if dry_run else "freed",
                                     len(found), size(total)), "value")
    return len(found), total


# ------------------------------------------------------------ snippets -----
# The raw ln(flux) of every exposure on the few grid columns around a figure
# window, kept next to the cube they belong to. Keyed by the block's columns
# rather than by the window as written, so two ways of writing one window share
# a file, and the cube's own key already pins the grid and the resampling.

def snippet_path(cube_dir: str, a0: int, b0: int) -> str:
    return os.path.join(cube_dir, "snippets", "cols_%07d_%07d.npz" % (a0, b0))


def write_snippet(cube_dir: str, a0: int, b0: int, by_file: dict) -> str:
    """{filename: (2, b0-a0)} to disk, atomically. Returns the path."""
    import numpy as np

    path = snippet_path(cube_dir, a0, b0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = sorted(by_file)
    values = (np.stack([by_file[k] for k in names]).astype(np.float32) if names
              else np.zeros((0, 2, b0 - a0), dtype=np.float32))
    tmp = path[:-len(".npz")] + ".partial.npz"
    np.savez(tmp, names=np.asarray(names), values=values, a0=a0, b0=b0)
    os.replace(tmp, path)
    return path


def read_snippet(cube_dir: str, a0: int, b0: int):
    """{filename: (2, b0-a0)} for this block, or None if there is none yet."""
    import numpy as np

    path = snippet_path(cube_dir, a0, b0)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as blob:
        if int(blob["a0"]) != a0 or int(blob["b0"]) != b0:
            return None
        values = blob["values"]
        return {str(name): values[i] for i, name in enumerate(blob["names"])}

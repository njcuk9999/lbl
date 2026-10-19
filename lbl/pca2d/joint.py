"""One observer basis for several stars.

The atmosphere and the instrument belong to the night, not to the target: two
stars observed in the same campaign share them. Only the star differs, and
with it the BERV coverage and the systemic velocity, so fitting several
objects together buys a cleaner observer basis. What is common to all of them
cannot be a star, and what follows one star cannot be common.

Each object keeps its own star spectrum, per order parity; the observer
components are shared by construction, because there is one basis and every
row of every object sees it.

The joint cube is the objects' own cubes, concatenated row for row on the one
grid they share, with three labels rewritten:

    parity     2 * (the object's index) + the order parity. The star spectra
               are estimated per group, so each object gets its own, while
               `parity % 2` still gives the order parity, which means the same
               thing for every object and is what the rest of the package
               reads.
    exposure   offset per object, so the two rows of an exposure stay tied and
               no two objects share an id
    object     the object's name, which the metadata already carries

Nothing else changes: the fit, the figures and the correction read a cube.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from astropy.table import Table, vstack

from .logger import log

#: ids are offset by this per object, far above any campaign's exposure count
EXPOSURE_STRIDE = 1000000


def joint_name(objects):
    """PROXIMA+GJ1+GJ3090: the folder and the cache key a joint run is under."""
    return "+".join(str(o) for o in objects)


def cube_path(cache_dir, fmt, key, objects):
    """Where the joint cube lives, keyed by the configuration and the objects."""
    import hashlib
    stamp = hashlib.sha1(joint_name(objects).encode()).hexdigest()[:6]
    return os.path.join(cache_dir, "cube_%s_%s_j%s" % (fmt, key, stamp))


def merge_snippets(cubes, target):
    """Carry the members' raw-flux snippets into the joint cube.

    Each cube keeps the raw flux around every figure window, written while the
    build had the spectra open anyway, so that a figure never reopens hundreds
    of files to draw five of them. They are keyed by file NAME, and no two
    objects share one, so the joint cube's snippet for a block is simply the
    union of its members'.

    Without this the joint cube had none, and every window of the report fell
    back to re-reading every spectrum of every object from the shared disk and
    resampling it: 31 minutes for ONE window against a fraction of a second,
    and an hour and a half for the figures of a three-star run (2026-09-13).
    """
    from . import cache as _cache

    blocks = {}
    for cube in cubes:
        folder = os.path.join(cube, "snippets")
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.startswith("cols_") or not name.endswith(".npz"):
                continue
            try:
                a0, b0 = (int(v) for v in name[5:-4].split("_"))
            except ValueError:
                continue
            stored = _cache.read_snippet(cube, a0, b0)
            if stored:
                blocks.setdefault((a0, b0), {}).update(stored)
    for (a0, b0), by_file in sorted(blocks.items()):
        _cache.write_snippet(target, a0, b0, by_file)
    if blocks:
        log("  carried %d snippet blocks, %d spectra in all, into the joint cube"
            % (len(blocks), len(next(iter(blocks.values())))))
    return len(blocks)


def build(cubes, objects, path, config_file=None, chunk=64):
    """Write the joint cube of `cubes`, in that order, at `path`.

    The rows are copied a chunk at a time through memory maps: three arrays of
    a thousand rows over half a million columns do not belong in memory at
    once. Returns the number of rows written.
    """
    gone = [(name, c) for name, c in zip(objects, cubes)
            if not os.path.exists(os.path.join(c, "grid.npy"))]
    if gone:
        # the caller builds what is missing before coming here (cli.
        # run_joint_cube), so reaching this means it could not. Said in words,
        # because np.load's own error names a grid.npy and nothing else
        raise SystemExit(
            "the joint cube is made of the objects' own cubes and %s not"
            " there: %s. Each is written by the cube stage, so run this"
            " configuration with the cube stage in it; if it was there a"
            " moment ago, something emptied the cache while the run was going."
            % ("is" if len(gone) == 1 else "are",
               ", ".join("%s (%s)" % (name, path) for name, path in gone)))
    grids = [np.load(os.path.join(c, "grid.npy")) for c in cubes]
    for name, grid in zip(objects, grids):
        if grid.shape != grids[0].shape or not np.allclose(grid, grids[0]):
            raise SystemExit("%s was built on another grid: a joint fit needs"
                             " one domain, one step and one high pass for every"
                             " object" % name)
    metas = [Table.read(os.path.join(c, "meta.fits")) for c in cubes]
    rows = [len(m) for m in metas]
    total = int(sum(rows))
    has_trans = all(os.path.exists(os.path.join(c, "trans.npy")) for c in cubes)

    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "grid.npy"), grids[0])
    n_pixels = grids[0].size
    names = ["data.npy", "sigma.npy"] + (["trans.npy"] if has_trans else [])
    out = {name: np.lib.format.open_memmap(
        os.path.join(tmp, name), mode="w+", dtype=np.float32,
        shape=(total, n_pixels)) for name in names}
    start = 0
    for cube, name, n in zip(cubes, objects, rows):
        for what in names:
            source = np.load(os.path.join(cube, what), mmap_mode="r")
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                out[what][start + a:start + b] = source[a:b]
            del source
        log("  %-10s %4d rows from %s" % (name, n, os.path.basename(cube)))
        start += n
    for handle in out.values():
        handle.flush()
    del out

    stacked = []
    for k, (meta, name) in enumerate(zip(metas, objects)):
        meta = meta.copy()
        parity = (np.asarray(meta["parity"], dtype=int)
                  if "parity" in meta.colnames else np.zeros(len(meta), dtype=int))
        meta["parity"] = 2 * k + (parity % 2)
        exposure = (np.asarray(meta["exposure"], dtype=int)
                    if "exposure" in meta.colnames else np.arange(len(meta)))
        meta["exposure"] = exposure + k * EXPOSURE_STRIDE
        meta["object"] = [str(name)] * len(meta)
        stacked.append(meta)
    vstack(stacked, join_type="outer").write(os.path.join(tmp, "meta.fits"),
                                             overwrite=True)
    if config_file and os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(tmp, "cube_config.yaml"))
    merge_snippets(cubes, tmp)
    shutil.rmtree(path, ignore_errors=True)
    os.rename(tmp, path)
    log("joint cube %s: %d rows, %d objects, %d columns"
        % (path, total, len(objects), n_pixels), "value")
    return total


def star_group(objects, name):
    """The first star-spectrum group of an object: 2 * its index.

    The correction of that object's files adds the order parity to it, so the
    group is 2 * index + parity, exactly what the cube's labels say.
    """
    return 2 * list(objects).index(name)

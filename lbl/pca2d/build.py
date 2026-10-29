#!/usr/bin/env python
"""Build and cache a registered data cube. No PCA, no frame choice made here.

This is the only way new data enters the project, and it is the `cube` stage
of `pca2d-preclean`. The two-frame fit in `pca2d/twoframe.py` reads a cube from
`cache/` and cannot make one.

The cube is written to `cache/cube_<format>_<hash>/`, the hash covering every
configuration entry that changes the cube. Running twice with the same config
is free; the second run finds the cache and returns.

**Always the observer frame.** The two-frame model does its own BERV shifting,
carrying the star basis into each spectrum's frame with an exact operator on the
log-uniform grid, so the cube must not be pre-shifted. `registration.frame` is
forced to `observer` here rather than trusted from the YAML: a barycentric cube
would be silently wrong rather than loudly broken, since the fit would still
converge, onto a star block anchored to a frame moving at twice the BERV.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

from . import cube, grids, wizard
from .config import cache_key, load_config
from .logger import log


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore any cached cube and rebuild it")
    parser.add_argument("--no-prompt", action="store_true",
                        help="never ask; fail with the list of problems instead."
                             " Use this in scripts and CI")
    parser.add_argument("--check", action="store_true",
                        help="validate the config against the data and exit,"
                             " without building anything")
    parser.add_argument("--allow-any-frame", action="store_true",
                        help="do not force registration.frame to observer. Only"
                             " for a deliberate control experiment; the two-frame"
                             " fit expects an observer-frame cube")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config if os.path.exists(args.config) else None)
    if args.no_cache:
        config["output"]["use_cache"] = False

    # Check before spending twenty minutes, not after. The checks are concrete:
    # the glob is expanded and counted, the first file is opened, the domain is
    # compared against the wavelengths that file actually covers.
    ok = wizard.report(config)
    if args.check:
        log("configuration is usable" if ok else "configuration is NOT usable",
            "value" if ok else "error")
        return 0 if ok else 1
    if not ok:
        if args.no_prompt or not sys.stdin.isatty():
            raise SystemExit("configuration is not usable; fix the entries above"
                             " in %s, or run without --no-prompt to be asked"
                             % args.config)
        config = wizard.run_wizard(config, args.config)

    frame = config["registration"].get("frame")
    if frame != "observer" and not args.allow_any_frame:
        log("registration.frame was %r; forcing 'observer' (--allow-any-frame"
            " to override)" % frame, "warn")
        config["registration"]["frame"] = "observer"

    fmt = str(config["input"].get("format", "tfits"))
    key = cache_key(config)
    log("input format: %s, frame: %s, highpass: %s"
        % (fmt, config["registration"]["frame"], config["highpass"]["mode"]), "value")
    log("cube cache key: cache/cube_%s_%s" % (fmt, key), "value")

    grid, data, sigma, trans, meta = cube.build_cube(config)
    dv = grids.grid_dv(grid)
    log("cube: %d rows x %d columns, %.1f - %.1f nm, dv = %.4f km/s"
        % (data.shape[0], data.shape[1], grid.min(), grid.max(), dv), "value")

    if "parity" in meta.colnames:
        parities = np.unique(meta["parity"])
        log("%d rows = %d exposures x %d order parities"
            % (len(meta), len(meta) // parities.size, parities.size), "value")

    cache_dir = os.path.join(config["output"].get("cache_directory", "cache"),
                             "cube_%s_%s" % (fmt, key))
    if os.path.isdir(cache_dir) and os.path.exists(args.config):
        # the recipe travels with the cube, so a directory listing of cache/ is
        # not the only thing standing between a reader and its provenance
        shutil.copy(args.config, os.path.join(cache_dir, "cube_config.yaml"))
    log("cube ready: %s" % cache_dir, "value")
    return cache_dir


if __name__ == "__main__":
    main()

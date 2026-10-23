#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The outlier masks of the correct stage on a range of the grid
(outliers.residual_outliers, `within`): what the sequence figure draws in
yellow has to be what the whole grid gives the corrected files.

Created on 2026-10-23

@author: artigau
"""
import os

import numpy as np
from astropy.table import Table

from lbl.pca2d.grids import pixel_shift
from lbl.pca2d.outliers import residual_outliers


def _cube(tmp_path, n_files=12, m=4000, dv=0.5, seed=3):
    """A t.fits-like cube of noise, two rows per exposure, with a feature
    that stands still in the STAR's frame (so slanted on the detector) and a
    few single hot samples; and a fit that explains nothing of it."""
    rng = np.random.default_rng(seed)
    n = 2 * n_files
    berv = np.repeat(np.linspace(-20.0, 20.0, n_files), 2)
    grid = 1500.0 * np.exp(np.arange(m) * dv / 299792.458)
    data = rng.normal(0.0, 0.01, (n, m)).astype(np.float32)
    shift = np.rint(-pixel_shift(berv, dv)).astype(int)
    for r in range(n):
        if rng.random() < 0.7:
            data[r, 1990 + shift[r]:1994 + shift[r]] += 0.06
    data[5, 2101] += 0.2
    data[17, 1877] -= 0.2
    path = os.path.join(str(tmp_path), "cube")
    os.makedirs(path)
    np.save(os.path.join(path, "grid.npy"), grid)
    np.save(os.path.join(path, "data.npy"), data)
    np.save(os.path.join(path, "sigma.npy"),
            np.full((n, m), 0.01, dtype=np.float32))
    names = ["e%02d.fits" % (r // 2) for r in range(n)]
    Table(dict(filename=names, berv=berv, parity=np.arange(n) % 2,
               exposure=np.arange(n) // 2, snr_band=np.full(n, 100.0))
          ).write(os.path.join(path, "meta.fits"))
    fit = os.path.join(str(tmp_path), "fit.npz")
    np.savez(fit, P=np.zeros((1, m)), a=np.zeros((n, 1)),
             Q=np.zeros((1, m)), b=np.zeros((n, 1)), berv=berv, dv=dv)
    return path, fit


def test_a_range_of_the_grid_gets_the_whole_grid_s_masks(tmp_path):
    cube, fit_path = _cube(tmp_path)
    options = dict(column_frac=0.1, column_chi2=1.5, bias_nsig=10.0,
                   bias_samples=9, star_frac=0.1, star_chi2=1.5,
                   star_bias_nsig=10.0, block=700)
    fit = np.load(fit_path)
    whole, report = residual_outliers(cube, fit, 3.0, 51, **options)
    first, last = 1500, 2500
    part, _ = residual_outliers(cube, fit, 3.0, 51, within=(first, last),
                                **options)
    # the star-frame feature is found, and the two hot samples
    assert report["star"]["n_columns"] > 0
    assert whole["e02.fits"][1, 2101] and whole["e08.fits"][1, 1877]
    # far enough from the range's edges for every test, the same masks
    for name in whole:
        assert part[name].shape == (2, last - first)
        assert np.array_equal(whole[name][:, 1800:2200],
                              part[name][:, 1800 - first:2200 - first])

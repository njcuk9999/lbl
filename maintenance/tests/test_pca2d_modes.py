#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The instrument modes PCA2D keeps (quality.drs_modes): HARPS gives two
groups, its two fibres (HAM and EGGS, each alone or with NIRPS), and a run
keeps the group most of its files are in, chosen before the cube's cache
key is taken; and a cube left with no more rows than the fit has
components stops, saying what rejected the others, rather than fitting
and correcting one exposure.

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_pca2d_modes.py

Created on 2026-11-12

@author: artigau
"""
from collections import Counter

import numpy as np
import pytest
from astropy.io import fits

from lbl.core import base
from lbl.pca2d import cube, readers
from lbl.pca2d.config import instrument_layer


def _files(tmp_path, modes):
    paths = []
    for i, mode in enumerate(modes):
        head = fits.Header()
        if mode is not None:
            head["HIERARCH ESO INS MODE"] = mode
        path = tmp_path / ("HARPS.%03d_e2ds_A.fits" % i)
        fits.PrimaryHDU(np.ones((2, 4), "f4"), head).writeto(path)
        paths.append(str(path))
    return paths


def _config():
    layer = instrument_layer("HARPS", "ORIG")
    return dict(quality=dict(layer["quality"]), input=dict(layer["input"]),
                lbl=dict(instrument="HARPS", data_source="ORIG"))


def test_harps_gives_its_two_fibres():
    assert instrument_layer("HARPS", "ORIG")["quality"]["drs_modes"] == [
        list(modes) for modes in base.HARPS_MODES]


def test_the_fibre_most_files_are_in(tmp_path):
    files = _files(tmp_path, ["EGGS+NIRPS"] * 5 + ["HARPS+NIRPS", "EGGS"])
    config = _config()
    modes, said = readers.choose_drs_modes(config, files)
    assert modes == ["EGGS", "EGGS+NIRPS"]
    assert "5 EGGS+NIRPS" in said and "6 files" in said and "other 1" in said
    # settled in the configuration, so the cache key holds the modes kept
    readers.settle_drs_modes(config, files)
    assert config["quality"]["drs_modes"] == ["EGGS", "EGGS+NIRPS"]
    # HAM: the plain list HARPS gave before its groups, so the cube's cache
    #   key (made from the configuration) is the one of the cubes made then
    ham = _config()
    (tmp_path / "ham").mkdir()
    readers.settle_drs_modes(ham, _files(tmp_path / "ham", ["HARPS"] * 3))
    assert ham["quality"]["drs_modes"] == ["HARPS", "HARPS+NIRPS"]


def test_a_plain_list_is_left_alone(tmp_path):
    files = _files(tmp_path, ["HE", "HA"])
    config = dict(quality=dict(drs_modes=["HE"]), input={})
    assert readers.choose_drs_modes(config, files) == (["HE"], "")
    config = dict(quality=dict(drs_modes=None), input={})
    assert readers.choose_drs_modes(config, files) == (None, "")


def test_too_few_rows_for_the_fit_stop_and_say_why():
    config = dict(twoframe=dict(n_earth=7, n_star=0))
    reasons = Counter({"instrument mode 'EGGS+NIRPS' is not one of the 2"
                       " asked for (HARPS, HARPS+NIRPS)": 56})
    with pytest.raises(RuntimeError) as caught:
        cube._require_rows(config, 1, 1, 57, reasons)
    text = str(caught.value)
    assert "1 of the 57 exposures" in text and "56 instrument mode" in text
    assert "at least 8" in text
    # every exposure rejected: the same stop
    with pytest.raises(RuntimeError, match="0 of the 3"):
        cube._require_rows(config, 0, 0, 3, Counter({"x": 3}))
    # one row more than components: fitted
    cube._require_rows(config, 8, 8, 8, Counter())
    assert cube._reasons(Counter({"a": 5, "b": 3, "c": 2, "d": 1}),
                         most=2) == "5 a; 3 b; 3 for other reasons"


def test_one_object_of_a_joint_fit_needs_one_row():
    config = dict(twoframe=dict(n_earth=7, n_star=0))
    cube._require_rows(config, 2, 2, 9, Counter({"x": 7}), own_fit=False)
    with pytest.raises(RuntimeError, match="none of the 9 exposures"):
        cube._require_rows(config, 0, 0, 9, Counter({"x": 9}), own_fit=False)


def test_a_common_mask_left_empty_stops_the_correction():
    """A few exposures that weight almost nothing make the common mask (the
    samples every exposure weighted) all but empty: the correction stops and
    names them, rather than writing NaN files LBL then fails on"""
    from lbl.pca2d import reconstruct

    rng = np.random.default_rng(1)
    bright = {"b%d" % i: rng.uniform(size=(2, 1000)) < 0.9 for i in range(8)}
    faint = {"f%d" % i: rng.uniform(size=(2, 1000)) < 0.02 for i in range(2)}
    alive = dict(bright, **faint)
    shared = np.ones((2, 1000), dtype=bool)
    for value in alive.values():
        shared &= value
    with pytest.raises(RuntimeError) as caught:
        reconstruct._enough_in_common(shared, set(alive), alive, 1.0)
    assert "f0" in str(caught.value) and "nothing is written" in str(
        caught.value)
    # in a joint fit, said and gone on
    reconstruct._enough_in_common(shared, set(alive), alive, 1.0, joint=True)
    # the bright ones alone: kept
    shared = np.ones((2, 1000), dtype=bool)
    for value in bright.values():
        shared &= value
    reconstruct._enough_in_common(shared, set(bright), bright, 1.0)


def test_only_the_sky_s_lines_drown_the_star():
    """max_sky_ratio masks where the sky's emission LINES exceed the star,
    not a smooth thermal background that does along a whole stretch"""
    from lbl.pca2d import tfits

    n = 2000
    flux = np.full((2, n), 100.0)
    sky = np.zeros((2, n))
    sky[:, 500:504] = 1000.0                     # an OH line, 10x the star
    sky[1] += np.linspace(0.0, 800.0, n)         # a thermal ramp to 8x
    drowned = tfits.sky_drowned(sky, flux, 4.0)
    assert drowned[0, 500:504].all() and drowned[0].sum() == 4
    # the line still masked over the ramp, the ramp itself not
    assert drowned[1, 500:504].all() and drowned[1].sum() <= 6
    assert not drowned[1, 1500:].any()

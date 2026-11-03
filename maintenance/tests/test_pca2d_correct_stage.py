"""What the correct stage reads, and how a joint fit's members are kept apart.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_correct_stage.py
"""
import copy
import os

import numpy as np
import pytest
from astropy.table import Table

from lbl.pca2d import cli
from lbl.pca2d import config as spconfig
from lbl.pca2d import reconstruct
from lbl.pca2d import readers


def _coeffs():
    return Table(rows=[("a_S2D_A.fits", 2460600.60, 0.0, 0.1),
                       ("a_S2D_A.fits", 2460600.60, 0.0, 0.1),
                       ("b_S2D_A.fits", 2460601.60, 0.0, 0.3),
                       ("b_S2D_A.fits", 2460601.60, 0.0, 0.3)],
                 names=("filename", "bjd", "berv", "b1"))


def _bridged(tmp_path):
    source = tmp_path / "science"
    source.mkdir()
    for name in ("a_S2D_A.fits", "c_S2D_A.fits", "b_S2D_A.fits", "other.txt"):
        (source / name).write_bytes(b"")

    class Args:
        source_dir = str(source)
        cube = None
    return Args, {"input": {"format": "s2d", "pattern": "*_S2D_A.fits"}}


def test_stacked_rows_read_a_bridged_instrument(tmp_path, monkeypatch):
    """Files of the configuration's pattern, by the configuration's reader."""
    args, config = _bridged(tmp_path)
    bjd = {"a_S2D_A.fits": 2460600.60, "c_S2D_A.fits": 2460600.62,
           "b_S2D_A.fits": 2460601.60}
    seen = []

    def read_any(path, cfg):
        seen.append((os.path.basename(path), cfg))
        return {"meta": {"bjd": bjd[os.path.basename(path)], "berv": 2.5}}
    monkeypatch.setattr(readers, "read_any", read_any)
    pairs, missing = reconstruct.rows_by_night({"coeffs": _coeffs()}, args, config)
    assert missing == 0
    assert all(cfg is config for _, cfg in seen)
    got = {os.path.basename(p): float(r["b1"]) for p, r in pairs}
    assert got == {"a_S2D_A.fits": 0.1, "c_S2D_A.fits": 0.1, "b_S2D_A.fits": 0.3}
    assert all(float(r["berv"]) == 2.5 for _, r in pairs)


def test_a_file_that_cannot_be_read_is_counted_and_said(tmp_path, monkeypatch):
    args, config = _bridged(tmp_path)
    said = []
    monkeypatch.setattr(reconstruct, "log", lambda *a, **k: said.append(a[0]))

    def broken(path, cfg):
        raise OSError("no such extension")
    monkeypatch.setattr(readers, "read_any", broken)
    pairs, missing = reconstruct.rows_by_night({"coeffs": _coeffs()}, args, config)
    assert (pairs, missing) == ([], 3)
    # once, with the reason
    assert len([s for s in said if "no such extension" in s]) == 1


def _masks():
    kept = np.ones((2, 10), dtype=bool)
    return {"a1": ~kept, "a2": ~kept, "b1": kept, "b2": kept}


def test_each_star_of_a_joint_fit_gets_its_own_common_mask():
    masks = _masks()
    assert reconstruct.mask_for("common", None, masks, "a2").sum() == 0
    assert reconstruct.mask_for("common", None, masks, "b1").sum() == 20
    # and even when the exposure has a refit's mask of its own
    own = np.zeros((2, 10), dtype=bool)
    assert reconstruct.mask_for("common", own, masks, "b2").sum() == 20


def test_on_a_stacked_cube_a_file_is_found_by_its_row():
    masks = _masks()
    assert reconstruct.mask_for("common", None, masks, "b9", "b1").sum() == 20
    assert reconstruct.mask_for("common", None, masks, "a9", "a1").sum() == 0
    assert reconstruct.mask_for("exposure", None, masks, "b9", "b1").sum() == 20


def test_the_other_mask_modes_are_as_they_were():
    masks = _masks()
    own = np.zeros((2, 10), dtype=bool)
    assert reconstruct.mask_for("none", own, masks, "b1") is None
    assert reconstruct.mask_for("exposure", own, masks, "b1") is own
    assert reconstruct.mask_for("exposure", None, masks, "b1") is masks["b1"]
    assert reconstruct.mask_for("exposure", None, None, "b1") is None
    # one campaign: the one set, also for a file that is not in the cube
    one = {"a1": masks["b1"], "a2": masks["b1"]}
    assert reconstruct.mask_for("common", None, one, "z") is masks["b1"]

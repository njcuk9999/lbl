#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LBL's full reset from the window's Cleanup tab: what lbl_reset deletes is
said before (lbl_reset.reset_plan, io.files_to_clean, cleanup.reset_homes),
the same list the recipe empties; the reset runs lbl_reset itself, one
instrument folder at a time, never while a run works in the root.

Created on 2026-11-12

@author: artigau
"""
import os
import subprocess

import pytest

from lbl.core import io
from lbl.gui import cleanup, lblinfo
from lbl.recipes import lbl_reset

GRID = lblinfo.instrument_object("NIRPS_HE", "CADC").params[
    "SAMPLE_WAVE_GRID_FILE"]
GONE = ["science/TOI1_pca2d/a.fits", "science/TOI1_pca2d_v2/a.fits",
        "science/TOI1_residuals/a.fits", "science/TOI1_tc/a.fits",
        "lblrv/TOI1_TOI1/x_lbl.fits", "lblrdb/lbl_TOI1_TOI1.rdb",
        "lblreftable/ref.csv", "masks/m.fits", "templates/t.fits",
        "plots/p.pdf", "calib/" + GRID]
KEPT = ["science/TOI1/a.fits", "calib/blaze_A.fits", "reports/r.pdf",
        "pca2d/cache/c.npy", "models/tellurics/tapas.fits"]


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "NIRPS_HE_CADC"
    for name in GONE + KEPT:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    return tmp_path


def test_the_plan_is_what_the_reset_empties(root):
    folder = str(root / "NIRPS_HE_CADC")
    params = lblinfo.instrument_object("NIRPS_HE", "CADC").params
    listed = sorted(os.path.relpath(name, folder)
                    for _what, path, kwargs in lbl_reset.reset_plan(
                        folder, params)
                    for name in io.files_to_clean(path, **kwargs))
    assert listed == sorted(os.path.normpath(name) for name in GONE)
    # the recipe's own loop, on the same plan: those files and no other
    for _what, path, kwargs in lbl_reset.reset_plan(folder, params):
        io.clean_directory(path, logmsg=False, **kwargs)
    for name in GONE:
        assert not (root / "NIRPS_HE_CADC" / name).exists()
    for name in KEPT:
        assert (root / "NIRPS_HE_CADC" / name).exists()


def test_the_window_says_it_per_instrument_folder(root):
    homes = cleanup.reset_homes(str(root))
    row = next(r for r in homes if r["home"] == "NIRPS_HE_CADC")
    assert (row["instrument"], row["source"]) == ("NIRPS_HE", "CADC")
    assert row["files"] == len(GONE)
    assert {p["what"] for p in row["parts"] if p["files"]} == {
        "derived", "reftable", "rdb", "lblrv", "masks", "templates", "plots",
        "wavegrid"}


def test_the_reset_runs_lbl_reset_on_that_folder(root, monkeypatch):
    folder = str(root / "NIRPS_HE_CADC")
    monkeypatch.setattr(cleanup, "busy", lambda data_dir: [])
    ran = []

    def fake(cmd, **kwargs):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(cleanup.subprocess, "run", fake)
    got = cleanup.reset(str(root), folder)
    assert got["done"] and len(ran) == 1
    assert "lbl_reset.main(instrument='NIRPS_HE', data_source='CADC'," \
        " data_dir=%r)" % folder in ran[0][-1]
    # not a folder of this root: nothing run
    assert not cleanup.reset(str(root), str(root / "elsewhere"))["done"]
    assert len(ran) == 1


def test_nothing_is_reset_while_a_run_works(root, monkeypatch):
    monkeypatch.setattr(cleanup, "busy",
                        lambda data_dir: [dict(pid=42, command="python w.py")])
    got = cleanup.reset(str(root), str(root / "NIRPS_HE_CADC"))
    assert not got["done"] and "42" in got["why"]
    assert (root / "NIRPS_HE_CADC" / "masks" / "m.fits").exists()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Names of files every disk can hold: ESO names its calibrations by the time
they were taken (r.NIRPS.2023-04-11T12:56:58.825_BLAZE_A.fits) and no
Windows disk (nor exFAT) takes a ':'. The window writes such a blaze as
lbl.core.base.portable_name gives it, fetched (fetch.spectra) or made at
the dispatch, and LBL finds it from the name the header gives
(Instrument.blaze_path).

Created on 2026-11-12

@author: artigau
"""
import io
import os
import types

import numpy as np
import pytest
from astropy.io import fits

from lbl.core.base import portable_name
from lbl.gui.fetch import spectra

ESO_BLAZE = "r.NIRPS.2023-04-11T12:56:58.825_BLAZE_A.fits"


def test_a_portable_name_has_no_character_windows_refuses():
    assert portable_name(ESO_BLAZE) == \
        "r.NIRPS.2023-04-11T12-56-58.825_BLAZE_A.fits"
    assert portable_name('a<b>c"d|e?f*g\th') == "a-b-c-d-e-f-g-h"
    # what every disk takes is left as it is, separators included
    for name in ("GJ699_e2ds_A.fits", "r.NIRPS.ADP.2025-04-11T12-47-13.918"
                 "_S2D_BLAZE_TCORR_A.fits", os.path.join("calib", "x.fits")):
        assert portable_name(name) == name


def _inst(tmp_path):
    from lbl.instruments.default import Instrument
    inst = types.SimpleNamespace(params=dict(
        DATA_DIR=str(tmp_path), INSTRUMENT="NIRPS_HE", DATA_SOURCE="ESO"))
    inst.default_blaze_file = lambda: Instrument.default_blaze_file(inst)
    return inst


def test_lbl_finds_a_blaze_under_its_portable_name(tmp_path):
    from lbl.core.base_classes import LblException
    from lbl.instruments.default import Instrument
    inst = _inst(tmp_path)
    calib = tmp_path / "calib"
    calib.mkdir()
    with pytest.raises(LblException):
        Instrument.blaze_path(inst, str(calib), ESO_BLAZE)
    (calib / portable_name(ESO_BLAZE)).write_text("")
    assert Instrument.blaze_path(inst, str(calib), ESO_BLAZE) == \
        str(calib / portable_name(ESO_BLAZE))
    # the header's own name first, where a disk holds it
    if os.name != "nt":
        (calib / ESO_BLAZE).write_text("")
        assert Instrument.blaze_path(inst, str(calib), ESO_BLAZE) == \
            str(calib / ESO_BLAZE)


def _archive(monkeypatch, files):
    """ESO's archive, its tar served from `files` {name: bytes}"""
    names = list(files)
    monkeypatch.setattr(spectra, "tar_members", lambda url: [
        (name, k, len(files[name])) for k, name in enumerate(names)])
    monkeypatch.setattr(spectra, "_range",
                        lambda url, start, size, *args: files[names[start]])


def _fits(hdus):
    out = io.BytesIO()
    fits.HDUList(hdus).writeto(out)
    return out.getvalue()


def _nirps_tar(blaze=ESO_BLAZE):
    flux = _fits([fits.PrimaryHDU(),
                  fits.ImageHDU(np.ones((2, 8), "f4"), name="SCIDATA"),
                  fits.ImageHDU(np.ones((2, 8), "f4"), name="ERRDATA")])
    tell = _fits([fits.PrimaryHDU(),
                  fits.ImageHDU(np.full((2, 8), 0.9, "f4"), name="SCIDATA")])
    return {"x/r.NIRPS.2023-04-12T01:00:00.000_S2D_BLAZE_A.fits": flux,
            "x/r.NIRPS.2023-04-12T01:00:00.000_S2D_TELL_SPECTRUM_A.fits":
                tell,
            "x/" + blaze: _fits([fits.PrimaryHDU(np.ones((2, 8), "f4"))])}


def test_the_blaze_of_nirps_at_eso_is_fetched_under_a_portable_name(
        tmp_path, monkeypatch):
    science, calib = tmp_path / "science", tmp_path / "calib"
    science.mkdir()
    calib.mkdir()
    _archive(monkeypatch, _nirps_tar())
    one = dict(id="ADP.2025-04-11T12:47:13.918", name=spectra._safe(
        "r.NIRPS.ADP.2025-04-11T12:47:13.918_S2D_BLAZE_TCORR_A.fits"))
    assert spectra._get_nirps(one, str(science), str(calib)) == ""
    assert os.listdir(calib) == [portable_name(ESO_BLAZE)]
    assert os.listdir(science) == [one["name"]]
    assert not any(":" in name for name in os.listdir(science))


def test_a_blaze_already_here_under_eso_s_name_is_not_fetched_again(
        tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("no ':' in a name on Windows")
    science, calib = tmp_path / "science", tmp_path / "calib"
    science.mkdir()
    calib.mkdir()
    (calib / ESO_BLAZE).write_text("fetched before")
    _archive(monkeypatch, _nirps_tar())
    one = dict(id="ADP.1", name="a_S2D_BLAZE_TCORR_A.fits")
    assert spectra._get_nirps(one, str(science), str(calib)) == ""
    assert os.listdir(calib) == [ESO_BLAZE]


def test_the_blaze_made_for_harps_at_eso_has_a_portable_name(
        tmp_path, monkeypatch):
    science, calib = tmp_path / "science", tmp_path / "calib"
    science.mkdir()
    calib.mkdir()
    head = fits.Header()
    head["INSTRUME"] = "HARPS"
    named = "HARPS.2010-01-01T20:00:00.000_blaze_A.fits"
    head["HIERARCH ESO DRS BLAZE FILE"] = named
    rng = np.random.default_rng(3)
    e2ds = _fits([fits.PrimaryHDU(1000.0 + rng.random((3, 1024)), head)])
    _archive(monkeypatch, {"x/HARPS.2010-01-02T01:00:00.000_e2ds_A.fits":
                           e2ds})
    one = dict(id="ADP.2", name="HARPS.2010-01-02T01-00-00.000_e2ds_A.fits")
    assert spectra._get_harps(one, str(science), str(calib)) == ""
    assert os.listdir(calib) == [portable_name(named)]
    from lbl.instruments.default import Instrument
    assert Instrument.blaze_path(_inst(tmp_path), str(calib), named) == \
        str(calib / portable_name(named))


def test_the_dispatch_makes_a_blaze_under_a_portable_name(tmp_path):
    from lbl.gui import dispatch
    calib = tmp_path / "HARPS_ORIG" / "calib"
    named = "HARPS.2014-01-01T20:00:00.000_blaze_A.fits"
    moves = [dict(kind="science", blaze=named, family="HARPS",
                  dest=str(tmp_path / "HARPS_ORIG" / "science" / "GJ1"
                           / "a_e2ds_A.fits"))]
    assert dispatch._blazes(moves) == [
        (moves[0]["dest"], str(calib / portable_name(named)))]
    # one here under its own name is that blaze
    if os.name != "nt":
        calib.mkdir(parents=True)
        (calib / named).write_text("")
        assert dispatch._blazes(moves) == []

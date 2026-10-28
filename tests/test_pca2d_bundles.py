"""Bundles: a night cut into several rows of the fit (input.bundle_*).

A transit sequence is sixty exposures over five hours, and its night's mean is
not one sky. The bundles cut a night by time, BERV range or count; the cube
records which exposure went into which row, and the correct stage matches an
exposure to its row by that, since its night alone no longer says.

    PYTHONPATH=<this checkout> python -m pytest tests/test_pca2d_bundles.py
"""
import datetime as dt
import os

import numpy as np
import pytest
from astropy.table import Table

from lbl.pca2d import cube as spcube
from lbl.pca2d import nights
from lbl.pca2d import reconstruct


def _stamps(start, n, step_min):
    t0 = dt.datetime.fromisoformat(start)
    return [t0 + dt.timedelta(minutes=step_min * i) for i in range(n)]


def test_no_bundle_keeps_whole_nights():
    stamps = _stamps("2024-10-12T02:00:00", 6, 5)
    labels = [nights.night_label(s) for s in stamps]
    assert nights.split_nights(stamps, labels, None) == labels


def test_a_night_cut_by_minutes_from_each_bundle_s_start():
    stamps = _stamps("2024-10-12T02:00:00", 10, 5)      # 0, 5, ... 45 min
    labels = [nights.night_label(s) for s in stamps]
    out = nights.split_nights(stamps, labels, {"minutes": 20})
    # 0..20 min is five exposures, then 25..45
    assert out == ["2024-10-11#00"] * 5 + ["2024-10-11#01"] * 5


def test_a_night_that_fits_keeps_its_plain_label():
    stamps = (_stamps("2024-10-12T02:00:00", 3, 5)
              + _stamps("2024-10-13T02:00:00", 10, 5))
    labels = [nights.night_label(s) for s in stamps]
    out = nights.split_nights(stamps, labels, {"minutes": 20})
    assert out[:3] == ["2024-10-11"] * 3
    assert set(out[3:]) == {"2024-10-12#00", "2024-10-12#01"}


def test_a_night_cut_by_berv_range_and_by_count():
    stamps = _stamps("2024-10-12T02:00:00", 8, 5)
    labels = [nights.night_label(s) for s in stamps]
    bervs = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    out = nights.split_nights(stamps, labels, {"berv": 0.1}, bervs)
    assert out == ["2024-10-11#00"] * 4 + ["2024-10-11#01"] * 4
    out = nights.split_nights(stamps, labels, {"max": 3})
    assert out == (["2024-10-11#00"] * 3 + ["2024-10-11#01"] * 3
                   + ["2024-10-11#02"] * 2)


def test_group_files_reads_the_times_from_the_names(tmp_path):
    names = ["NIRPS.2024-10-12T02:%02d:00.000t.fits" % m for m in range(0, 50, 5)]
    files = [str(tmp_path / n) for n in reversed(names)]   # out of order
    out, labels = nights.group_files(files, True, bundle={"minutes": 20})
    assert [os.path.basename(f) for f in out] == names
    assert labels == ["2024-10-11#00"] * 5 + ["2024-10-11#01"] * 5


def test_the_settings_are_checked():
    assert nights.bundle_settings({}) is None
    assert nights.bundle_settings({"bundle_minutes": None}) is None
    assert nights.bundle_settings({"bundle_minutes": 30}) == {"minutes": 30.0}
    with pytest.raises(ValueError):
        nights.bundle_settings({"bundle_berv": -1})


def _config(**inp):
    base = {"nightly_stack": "auto"}
    base.update(inp)
    return {"input": base, "output": {"max_memory_gb": 1000.0},
            "twoframe": {"dtype": "float32"}}


def test_the_bundles_never_decide_the_stacking():
    """They cut stacked nights; with a default of 100 m/s on every
    configuration, letting them turn stacking on would stack every
    spectrograph."""
    for stacking, expected in (("auto", False), (True, True), (False, False)):
        stack, _ = spcube.resolve_stacking(
            _config(nightly_stack=stacking, bundle_berv=0.1), 100, 1000)
        assert stack is expected, stacking


def test_the_cube_records_which_exposure_went_into_which_row(tmp_path):
    coadd = spcube._Coadder(2, 2, 4, np.float32, False)
    good = np.ones((2, 4), bool)
    ones = np.ones((2, 4))
    for name, label in (("a.fits", "n#00"), ("b.fits", "n#00"),
                        ("c.fits", "n#01")):
        coadd.add(label, ones, ones, None, good, {"filename": name, "bjd": 1.0})
    data, sigma, trans, meta = coadd.result()
    assert data.shape[0] == 4
    path = str(tmp_path / "cube")
    spcube._save_cache(path, np.arange(4.0), data, sigma, trans, meta)
    members = reconstruct.member_row_names(path)
    assert members == {"a.fits": "a.fits", "b.fits": "a.fits",
                       "c.fits": "c.fits"}
    assert reconstruct.cut_key("/x/b.fits", None, None, members) == "a.fits"
    assert nights.row_unit(path) == "bundles"
    spcube.set_pending_members(None)


def _exposure_tree(tmp_path, members):
    source = tmp_path / "science"
    source.mkdir()
    for name in members:
        (source / name).write_bytes(b"")
    cube = tmp_path / "cube"
    cube.mkdir()
    Table(rows=[(f, r, "g") for f, r in members.items()],
          names=("filename", "row", "group")).write(
        str(cube / spcube.MEMBERS_FILE), format="ascii.csv")
    return str(source), str(cube)


def test_two_bundles_of_one_night_are_two_rows(tmp_path, monkeypatch):
    """The night alone would hand both bundles the same row."""
    members = {"e1t.fits": "e1t.fits", "e2t.fits": "e1t.fits",
               "e3t.fits": "e3t.fits", "e4t.fits": "e3t.fits"}
    source, cube = _exposure_tree(tmp_path, members)
    bjd = {"e1t.fits": 2460600.60, "e2t.fits": 2460600.61,
           "e3t.fits": 2460600.70, "e4t.fits": 2460600.71}
    monkeypatch.setattr(reconstruct.sptf, "read_tfits", lambda path: {
        "meta": {"bjd": bjd[os.path.basename(path)], "berv": 1.0}})
    # one coefficient row per (row, parity), as the fit writes them
    coeffs = Table(rows=[("e1t.fits", 2460600.605, 0.0, 0.1),
                         ("e1t.fits", 2460600.605, 0.0, 0.1),
                         ("e3t.fits", 2460600.705, 0.0, 0.9),
                         ("e3t.fits", 2460600.705, 0.0, 0.9)],
                   names=("filename", "bjd", "berv", "b1"))

    class Args:
        source_dir = source
    Args.cube = cube
    pairs, missing = reconstruct.rows_by_night({"coeffs": coeffs}, Args)
    got = {os.path.basename(p): float(r["b1"]) for p, r in pairs}
    assert missing == 0
    assert got == {"e1t.fits": 0.1, "e2t.fits": 0.1, "e3t.fits": 0.9,
                   "e4t.fits": 0.9}


def test_whole_nights_match_as_they_always_did(tmp_path, monkeypatch):
    """By membership or by night, a cube of whole nights gives the same row."""
    members = {"e1t.fits": "e1t.fits", "e2t.fits": "e1t.fits",
               "f1t.fits": "f1t.fits"}
    source, cube = _exposure_tree(tmp_path, members)
    bjd = {"e1t.fits": 2460600.60, "e2t.fits": 2460600.62,
           "f1t.fits": 2460601.60}
    monkeypatch.setattr(reconstruct.sptf, "read_tfits", lambda path: {
        "meta": {"bjd": bjd[os.path.basename(path)], "berv": 1.0}})
    coeffs = Table(rows=[("e1t.fits", 2460600.61, 0.0, 0.1),
                         ("e1t.fits", 2460600.61, 0.0, 0.2),
                         ("f1t.fits", 2460601.60, 0.0, 0.3),
                         ("f1t.fits", 2460601.60, 0.0, 0.4)],
                   names=("filename", "bjd", "berv", "b1"))

    class Args:
        source_dir = source
    Args.cube = cube
    by_members, _ = reconstruct.rows_by_night({"coeffs": coeffs}, Args)
    Args.cube = None
    by_night, _ = reconstruct.rows_by_night({"coeffs": coeffs}, Args)
    assert [float(r["b1"]) for _, r in by_members] == \
        [float(r["b1"]) for _, r in by_night] == [0.2, 0.2, 0.4]


def _nominal(instrument):
    from lbl.pca2d import config as spconfig
    path = os.path.join(os.path.dirname(spconfig.__file__), "nominal.yaml")
    return spconfig, spconfig.load_config(path, instrument=instrument)


def test_the_default_is_100_m_per_s_of_berv():
    _, cfg = _nominal("NIRPS")
    assert cfg["input"]["bundle_berv"] == 0.1
    assert cfg["input"]["bundle_minutes"] is None
    assert cfg["input"]["bundle_max"] is None


def test_unset_bundles_do_not_rekey_the_cubes():
    spconfig, cfg = _nominal("NIRPS")
    for name in ("bundle_minutes", "bundle_berv", "bundle_max"):
        cfg["input"][name] = None
    key = spconfig.cache_key(cfg)
    for name in ("bundle_minutes", "bundle_berv", "bundle_max"):
        cfg["input"].pop(name)
    assert spconfig.cache_key(cfg) == key
    cfg["input"]["bundle_berv"] = 0.1
    assert spconfig.cache_key(cfg) != key


def test_bundles_do_not_rekey_a_cube_of_single_exposures():
    spconfig, cfg = _nominal("NIRPS")
    cfg["input"]["nightly_stack"] = False
    key = spconfig.cache_key(cfg)
    cfg["input"]["bundle_berv"] = 0.05
    assert spconfig.cache_key(cfg) == key


def test_the_berv_line_runs_where_the_observer_frame_goes():
    """A feature fixed in the observer's frame sits at +BERV in the star's
    (the barycentric correction adds it), relativistically: the line of the
    river plots, checked against WASP-76's telluric residuals."""
    from lbl.pca2d.figures.sequence import berv_track
    c = 299792.458
    berv = np.array([-10.0, 0.0, 10.0])
    track = berv_track(1000.0, berv)
    assert track[1] == pytest.approx(1000.0)
    assert np.all(np.diff(track) > 0)
    beta = 10.0 / c
    assert track[2] == pytest.approx(1000.0 * np.sqrt((1 + beta) / (1 - beta)),
                                     rel=1e-12)


def test_the_report_page_of_the_bundles(tmp_path):
    from lbl.pca2d.figures import berv_bundles
    cube = tmp_path / "cube"
    cube.mkdir()
    rows = []
    for k in range(12):                     # one night, 12 exposures, 2 bundles
        rows.append(("e%02dt.fits" % k, "e00t.fits" if k < 6 else "e06t.fits",
                     "2024-10-11#%02d" % (k // 6), 2460600.6 + k / 288.0,
                     5.0 - 0.02 * k))
    rows.append(("f00t.fits", "f00t.fits", "2024-10-12", 2460601.6, 4.0))
    Table(rows=rows, names=("filename", "row", "group", "bjd", "berv")).write(
        str(cube / "members.csv"), format="ascii.csv")
    out = tmp_path / "bundles.pdf"
    assert berv_bundles.main(["--cube", str(cube), "--out", str(out),
                              "--berv", "0.1"]) == 0
    assert out.stat().st_size > 1000
    # a cube without the list still gives a page, which says why
    bare = tmp_path / "bare"
    bare.mkdir()
    assert berv_bundles.main(["--cube", str(bare), "--out",
                              str(tmp_path / "note.pdf")]) == 0
    assert (tmp_path / "note.pdf").exists()

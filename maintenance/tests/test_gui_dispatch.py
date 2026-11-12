"""The dispatch of a root's Downloads (lbl.gui.dispatch): the DATA_DIR of
each instrument and source, where a file's star is in J2000, one folder for
one star whatever its names, the two stars of a binary apart, the folders
already there kept, nothing overwritten, and the models of every DATA_DIR
the root's. SIMBAD is never asked: its answers are stood in for."""
import json
import os
import time

import numpy as np
import pytest
from astropy.io import fits

from lbl.gui import dispatch

#: eps Eri as ESO's observers and APERO (Gaia EDR3, epoch 2016.0) write it
ESO_EPS = dict(alpha=33255.844, delta=-92729.739, pma=-0.974758,
               pmd=0.020876)
APERO_EPS = dict(PP_RA=53.22829342, PP_DEC=-9.458168216, PP_EPOCH=2457388.5,
                 PP_PMRA=-974.7581452, PP_PMDE=20.87584009)


def _harps(path, name, alpha, delta, pma=0.0, pmd=0.0, mjd=56900.0):
    """An e2ds of the HARPS DRS: one image in the primary."""
    head = fits.Header()
    head["INSTRUME"] = "HARPS"
    head["OBJECT"] = name
    head["MJD-OBS"] = mjd
    for key, value in (("ESO DPR CATG", "SCIENCE"), ("ESO INS MODE", "HARPS"),
                       ("ESO TEL TARG ALPHA", alpha),
                       ("ESO TEL TARG DELTA", delta),
                       ("ESO TEL TARG EPOCH", 2000.0),
                       ("ESO TEL TARG PMA", pma), ("ESO TEL TARG PMD", pmd),
                       ("ESO OBS TARG NAME", name)):
        head["HIERARCH " + key] = value
    fits.PrimaryHDU(np.ones((2, 8), "f4"), head).writeto(path)


def _harpsn(path, name):
    """An S2D_A of the TNG's espdr: eps Eri under a programme's nickname."""
    head = fits.Header()
    head["INSTRUME"] = "HARPN"
    head["MJD-OBS"] = 57240.0
    for key, value in (("ESO PRO CATG", "S2D_A"), ("TNG DPR CATG", "SCIENCE"),
                       ("TNG OBS TARG NAME", name),
                       ("TNG TEL TARG ALPHA", "03h32m55.900"),
                       ("TNG TEL TARG DELTA", "-09:27:29.700"),
                       ("TNG TEL TARG EPOCH", 2000.0),
                       ("TNG TEL TARG PMA", -0.97517),
                       ("TNG TEL TARG PMD", 0.01948)):
        head["HIERARCH " + key] = value
    fits.HDUList([fits.PrimaryHDU(header=head), fits.ImageHDU(
        np.ones((2, 8), "f4"), name="SCIDATA")]).writeto(path)


def _nirps(path, drsobjn="EPSILONERI", obj="Eps_eri"):
    """A t.fits of APERO: the flux in the primary, Gaia's place in PP_*."""
    head = fits.Header()
    head.update(dict(INSTRUME="NIRPS", DRSMODE="HE", DRSOBJN=drsobjn,
                     OBJECT=obj, DPRTYPE="OBJ_SKY", WAVEORDN=1,
                     MJDMID=60586.3, **APERO_EPS))
    fits.PrimaryHDU(np.ones((2, 8), "f4"), head).writeto(path)


@pytest.fixture(autouse=True)
def no_apero(tmp_path_factory, monkeypatch):
    """No database of APERO's, and none fetched: the copy this machine may
    keep says nothing here (a test gives its own)."""
    empty = tmp_path_factory.mktemp("no_apero")
    monkeypatch.setenv("LBL_APERO_ASTROMETRICS", str(empty))
    return empty


@pytest.fixture
def root(tmp_path):
    down = tmp_path / "Downloads"
    (down / "unpacked").mkdir(parents=True)
    _harps(down / "HARPS.2014-01-01T00-00-00.000_e2ds_A.fits", "HR1084",
           ESO_EPS["alpha"], ESO_EPS["delta"])
    _harpsn(down / "unpacked" / "r.HARPN.2015-08-06T06-01-36_S2D_A.fits",
            "K3_01")
    _nirps(down / "NIRPS.2024-10-03T07:04:44.397t.fits")
    # GJ 65 A and B: 2 arcsec apart, one proper motion
    _harps(down / "HARPS.2019-01-01T00-00-00.000_e2ds_A.fits", "GJ 65 A",
           13900.10, -175701.0, 3.32, 0.59)
    _harps(down / "HARPS.2019-01-02T00-00-00.000_e2ds_A.fits", "GJ65B",
           13900.24, -175701.0, 3.32, 0.59)
    _harps(down / "HARPS.2014-01-01T00-00-00.000_ccf_K5_A.fits", "HR1084",
           ESO_EPS["alpha"], ESO_EPS["delta"])
    (down / "notes.txt").write_text("mine")
    return tmp_path


def test_the_data_dirs_of_a_root():
    assert dispatch.folder_of("NIRPS_HE", "APERO") == "NIRPS_HE_APERO"
    # one source only: the instrument's name alone
    assert dispatch.folder_of("ESPRESSO", "None") == "ESPRESSO"
    assert dispatch.folder_of("NEID", "ESSP") == "NEID"
    assert dispatch.reader_of("/a/b/SPIROU_CADC") == ("SPIROU", "CADC")
    assert dispatch.reader_of("/a/b/demo") is None
    assert dispatch.split("/a/b/HARPS_ORIG") == ("/a/b", "HARPS_ORIG")
    assert dispatch.split("/a/b/demo") == ("/a/b/demo", "")


def test_a_place_brought_back_to_2000():
    # ESO's J2000 place and APERO's Gaia place of 2016 moved back by its
    #   proper motion: one place
    eso = dispatch.position([fits.Header({
        "HIERARCH ESO TEL TARG ALPHA": ESO_EPS["alpha"],
        "HIERARCH ESO TEL TARG DELTA": ESO_EPS["delta"],
        "HIERARCH ESO TEL TARG EPOCH": 2000.0,
        "HIERARCH ESO TEL TARG PMA": ESO_EPS["pma"],
        "HIERARCH ESO TEL TARG PMD": ESO_EPS["pmd"]})])
    apero = dispatch.position([fits.Header(APERO_EPS)])
    assert eso["how"] == apero["how"] == "catalogue"
    assert apero["pm"] == pytest.approx((-974.758, 20.876), abs=1e-3)
    assert dispatch.separation((eso["ra"], eso["dec"]),
                               (apero["ra"], apero["dec"])) < 1.0
    # TNG's text, and a place of no proper motion: as loose as the star
    #   may have moved since 2000
    tng = dispatch.position([fits.Header({
        "HIERARCH TNG TEL TARG ALPHA": "10h19m36.281",
        "HIERARCH TNG TEL TARG DELTA": "19:52:12.130",
        "HIERARCH TNG TEL TARG EPOCH": 2000.0})], year=2018.0)
    assert tng["ra"] == pytest.approx(154.90117, abs=1e-5)
    assert tng["pm"] is None and tng["loose"] == pytest.approx(18.0)


def test_names_compared_and_the_components_they_say():
    assert dispatch.norm("Gliese 849") == dispatch.norm("GL849") == "GJ849"
    # a planet's letter or number: its star
    assert dispatch.norm("GJ 1214 b") == "GJ1214"
    assert dispatch.norm("TOI-1452.01") == "TOI1452"
    assert dispatch.component("GJ 15 A") == ("GJ15", "A")
    assert dispatch.component("AlphaCenB") == ("ALPHACEN", "B")
    # a name ending in a capital that is not a component's
    assert dispatch.component("BARNARD") is None
    assert dispatch.component("HD189733") is None
    assert dispatch.clean("KIC 12060710") == "KIC_12060710"


def _sighting(name, ra, dec, pm=(3320.0, 590.0), how="catalogue"):
    return dict(names={name: 1}, norms={dispatch.norm(name)},
                component=dispatch.component(name),
                place=dict(ra=ra, dec=dec, pm=pm, loose=0.0, how=how),
                pm=pm, loose=0.0, files=[0])


def test_a_binary_apart_by_simbad_and_a_fast_star_found():
    a = _sighting("BL Cet", 24.7559, -17.9503)
    b = _sighting("UV Cet", 24.7559 + 2.0 / 3600, -17.9503)
    # two names at one place: one star, until SIMBAD knows them apart
    assert dispatch.stars([a, b]) == [[0, 1]]
    idents = {"BL Cet": dict(main="BL Cet", ra=a["place"]["ra"],
                             dec=-17.9503, pmra=3320.0, pmdec=590.0),
              "UV Cet": dict(main="UV Cet", ra=b["place"]["ra"],
                             dec=-17.9503, pmra=3320.0, pmdec=590.0)}
    apart = dispatch.apart_by_simbad([a, b], [[0, 1]], idents)
    assert dispatch.stars([a, b], frozenset(apart)) == [[0], [1]]
    # Barnard's star, 10.4 arcsec/yr: its place of 2020 written as if of
    #   2000, without its proper motion, is still the star's
    fast = _sighting("GJ 699", 269.4521, 4.6934, pm=(-801.6, 10362.4))
    late = _sighting("BARNARD", 269.4521 - 0.0045, 4.6934 + 0.0576, pm=None)
    late["loose"] = 20.0
    assert dispatch.stars([fast, late]) == [[0, 1]]


def test_one_folder_a_star_whatever_its_names(root):
    plan = dispatch.plan(str(root), ask_simbad=False)
    where = {(r["home"], r["folder"]) for r in plan["folders"]}
    # eps Eri by three names in three DATA_DIRs: APERO's name in each
    assert {("HARPS_ORIG", "EPSILONERI"), ("HARPSN_ESO", "EPSILONERI"),
            ("NIRPS_HE_APERO", "EPSILONERI")} <= where
    # GJ 65 A and B: their names say two stars
    binary = sorted(r["folder"] for r in plan["folders"]
                    if r["names"][0] in ("GJ 65 A", "GJ65B"))
    assert binary == ["GJ65B", "GJ_65_A"]
    assert [u["name"] for u in plan["unused"]] == [
        "HARPS.2014-01-01T00-00-00.000_ccf_K5_A.fits"]
    assert [u["name"] for u in plan["left"]] == ["notes.txt"]


def test_the_files_moved_recorded_and_the_models_shared(root):
    (root / "HARPS_ORIG" / "models" / "tellurics").mkdir(parents=True)
    (root / "HARPS_ORIG" / "models" / "tellurics" / "tapas.fits").write_text(
        "tapas")
    result = dispatch.go(dispatch.plan(str(root), ask_simbad=False))
    assert result["moved"] == 5 and result["unused"] == 1
    assert (root / "NIRPS_HE_APERO" / "science" / "EPSILONERI"
            / "NIRPS.2024-10-03T07:04:44.397t.fits").is_file()
    down = root / "Downloads"
    # what LBL does not read put aside, what is no spectrum left, the
    #   folder an archive was unpacked in taken away once empty
    assert sorted(os.listdir(down)) == [".dispatch_headers.json",
                                        ".dispatched.jsonl", "notes.txt",
                                        "unused"]
    lines = (down / ".dispatched.jsonl").read_text().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["how"]
    # a models/ of a DATA_DIR's own merged into the root's, then linked
    assert (root / "models" / "tellurics" / "tapas.fits").read_text() \
        == "tapas"
    for home in ("HARPS_ORIG", "HARPSN_ESO", "NIRPS_HE_APERO"):
        assert os.readlink(root / home / "models") == os.path.join(
            os.pardir, "models")


def test_a_star_keeps_its_folder_and_nothing_is_overwritten(root):
    dispatch.go(dispatch.plan(str(root), ask_simbad=False))
    down = root / "Downloads"
    # another night of eps Eri under another name: into its folder
    _harps(down / "HARPS.2015-01-01T00-00-00.000_e2ds_A.fits", "eps Eri",
           ESO_EPS["alpha"], ESO_EPS["delta"], ESO_EPS["pma"], ESO_EPS["pmd"])
    # a copy of a spectrum there: taken out; a different one of that name:
    #   left
    _nirps(down / "NIRPS.2024-10-03T07:04:44.397t.fits")
    _harps(down / "HARPS.2019-01-01T00-00-00.000_e2ds_A.fits", "GJ 65 A",
           13900.10, -175701.0, 3.32, 0.59, mjd=58484.5)
    plan = dispatch.plan(str(root), ask_simbad=False)
    eps = [r for r in plan["folders"] if r["home"] == "HARPS_ORIG"
           and r["names"] == ["eps Eri"]]
    assert eps and eps[0]["folder"] == "EPSILONERI" and not eps[0]["new"]
    result = dispatch.go(plan)
    assert result["moved"] == 1 and result["copies"] == 1
    assert [k["name"] for k in result["kept"]] == [
        "HARPS.2019-01-01T00-00-00.000_e2ds_A.fits"]
    assert (down / "HARPS.2019-01-01T00-00-00.000_e2ds_A.fits").is_file()


def test_the_window_plans_then_moves(root):
    box = dispatch.Dispatcher()
    box.idents = {name: None for name in ("HR1084", "K3_01", "EPSILONERI",
                                          "Eps_eri", "GJ 65 A", "GJ65B")}
    box.start_plan(str(root))
    for _ in range(200):
        if box.status()["state"] != "planning":
            break
        time.sleep(0.05)
    status = box.status()
    assert status["state"] == "planned"
    assert status["plan"]["science"] == 5 and "moves" not in status["plan"]
    # a plan for another root is not moved
    assert "error" in box.start_go(str(root / "elsewhere"))
    assert box.start_go(str(root))["state"] == "moving"


def _nirps_eso(path, catg, mode="HE", name="TOI-206", arcfile=None):
    """A product of ESO's NIRPS pipeline: an S2D, or the night's blaze."""
    head = fits.Header()
    head["INSTRUME"] = "NIRPS"
    head["OBJECT"] = name
    head["MJD-OBS"] = 60928.34
    if arcfile:
        head["ARCFILE"] = arcfile
    for key, value in (("ESO PRO CATG", catg), ("ESO INS MODE", mode),
                       ("ESO TEL TARG ALPHA", 45456.935),
                       ("ESO TEL TARG DELTA", -623120.465),
                       ("ESO TEL TARG EPOCH", 2000.0),
                       ("ESO TEL TARG PMA", 0.105946),
                       ("ESO TEL TARG PMD", 0.110888),
                       ("ESO PRO REC1 CAL24 NAME",
                        "r.NIRPS.2025-09-09T12:12:29.357_BLAZE_A.fits")):
        head["HIERARCH " + key] = value
    hdus = [fits.PrimaryHDU(header=head)]
    if catg == "BLAZE_A":
        hdus.append(fits.ImageHDU(np.ones((2, 8), "f4")))
    else:
        hdus.append(fits.ImageHDU(np.ones((2, 8), "f4"), name="SCIDATA"))
    fits.HDUList(hdus).writeto(path)


def test_nirps_at_eso_its_telluric_corrected_s2d_and_its_blaze(tmp_path):
    down = tmp_path / "Downloads"
    down.mkdir()
    _nirps_eso(down / "r.NIRPS.2026-02-27T02:15:29.476_S2D_BLAZE_TELL_CORR_A"
                      ".fits", "S2D_BLAZE_TELL_CORR_A")
    _nirps_eso(down / "r.NIRPS.2026-02-27T02:15:29.476_S2D_A.fits", "S2D_A")
    # the night's blaze, ':' written '_' by the archive
    _nirps_eso(down / "r.NIRPS.2025-09-09T12_12_29.357_BLAZE_A.fits",
               "BLAZE_A", arcfile="NIRPS.2025-09-09T12:12:29.357.fits")
    plan = dispatch.plan(str(tmp_path), ask_simbad=False)
    # HE from ESO INS MODE; LBL's NIRPS ESO reader keeps the blaze in and
    #   does not clean the tellurics: the S2D_BLAZE_TELL_CORR_A it reads
    assert [(r["home"], r["folder"], r["science"]) for r in plan["folders"]] \
        == [("NIRPS_HE_ESO", "TOI-206", 1)]
    assert [u["name"] for u in plan["unused"]] == [
        "r.NIRPS.2026-02-27T02:15:29.476_S2D_A.fits"]
    assert plan["defaults"] == ["NIRPS_HE_ESO"]
    result = dispatch.go(plan)
    # the blaze under the name the spectra's headers give it, as every
    #   disk can hold it (':' as '-', where LBL looks for it too), and the
    #   setup's default blaze, in the root's models
    calib = tmp_path / "NIRPS_HE_ESO" / "calib"
    assert os.listdir(calib) == [
        "r.NIRPS.2025-09-09T12-12-29.357_BLAZE_A.fits"]
    assert result["defaults"] == ["NIRPS_HE_ESO"]
    assert (tmp_path / "models" / "default-blaze"
            / "blaze_NIRPS_HE_ESO.fits").is_file()


def test_a_header_with_a_wrong_place_does_not_split_a_star():
    # 430 files of one name at eps Eri, one of them 8 arcsec off (an
    #   observer's typo), and 280 of another name at eps Eri: one star
    pm = (-974.758, 20.876)
    many = _sighting("HD22049", 53.2327, -9.4582, pm=pm)
    many["files"] = list(range(430))
    typo = _sighting("HD22049", 53.2350, -9.4582, pm=pm)
    other = _sighting("Epsilon Eridani", 53.2327, -9.4583, pm=pm)
    other["files"] = list(range(280))
    assert dispatch.stars([many, typo, other]) == [[0, 1, 2]]
    # two stars 20 arcsec apart, a pointing between them: never one
    a = _sighting("STAR1", 10.0, 20.0)
    b = _sighting("STAR2", 10.0, 20.0 + 20.0 / 3600)
    p = _sighting("P", 10.0, 20.0 + 10.0 / 3600, pm=None, how="pointing")
    assert sorted(map(len, dispatch.stars([a, b, p]))) == [1, 2]


def test_what_the_window_shows_of_a_root(root):
    dispatch.go(dispatch.plan(str(root), ask_simbad=False))
    # a science/ of the root's own with empty folders only: no targets, not
    #   offered as a data folder
    (root / "science" / "OLD").mkdir(parents=True)
    _harps(root / "Downloads" / "HARPS.2016-01-01T00-00-00.000_e2ds_A.fits",
           "eps Eri", ESO_EPS["alpha"], ESO_EPS["delta"])
    shown = dispatch.layout(str(root))
    assert not shown["legacy"] and shown["science"]["empty"] == ["OLD"]
    harps = next(f for f in shown["folders"] if f["name"] == "HARPS_ORIG")
    assert (harps["targets"], harps["spectra"], harps["models"]) == (
        3, 3, "linked")
    # what waits in Downloads, by kind of name; what cannot be dispatched
    #   apart, with why
    assert shown["downloads"]["kinds"] == [dict(kind="*_e2ds_A.fits", n=1)]
    assert shown["downloads"]["left"] == [dict(
        why="not a spectrum (*.txt)", n=1, example="notes.txt")]
    assert shown["downloads"]["unused"] == 1


def test_files_from_elsewhere_added_then_planned(root, tmp_path_factory):
    # spectra elsewhere on the machine: copied into a folder of Downloads
    #   of their own (the originals stay), then the plan of Downloads, as
    #   Dispatch makes it
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    _harps(elsewhere / "HARPS.2016-01-01T00-00-00.000_e2ds_A.fits",
           "eps Eri", ESO_EPS["alpha"], ESO_EPS["delta"])
    _nirps(elsewhere / "NIRPS.2025-01-01T00:00:00.000t.fits")
    picked = [str(p) for p in sorted(elsewhere.iterdir())]
    box = dispatch.Dispatcher()
    box.idents = {name: None for name in ("HR1084", "K3_01", "EPSILONERI",
                                          "Eps_eri", "GJ 65 A", "GJ65B",
                                          "eps Eri")}
    assert box.start_add(str(root), picked)["state"] == "adding"
    for _ in range(200):
        if box.status()["state"] not in ("adding", "planning"):
            break
        time.sleep(0.05)
    status = box.status()
    assert status["state"] == "planned" and status["added"]["added"] == 2
    added = status["added"]["folder"]
    assert os.path.dirname(added) == str(root / "Downloads")
    assert sorted(os.listdir(added)) == sorted(os.path.basename(p)
                                               for p in picked)
    assert all(os.path.exists(p) for p in picked)
    assert status["plan"]["science"] == 7
    # moved rather than copied: the original leaves its folder
    one = elsewhere / "HARPS.2016-01-02T00-00-00.000_e2ds_A.fits"
    _harps(one, "eps Eri", ESO_EPS["alpha"], ESO_EPS["delta"])
    moved = dispatch.add(str(root), [str(one)], move=True)
    assert moved["added"] == 1 and not one.exists()


def test_templates_of_the_same_mode_across_the_root(tmp_path):
    from astropy.table import Table
    from lbl.gui import lblinfo

    def template(path, instrument, mode):
        path.parent.mkdir(parents=True, exist_ok=True)
        table = Table(dict(wavelength=[1.0, 2.0], flux=[1.0, 1.0]))
        table.meta.update(LBLINSTR=instrument, LBLIMODE=mode)
        table.write(path)

    # the root's own templates/ (old layout): one of NIRPS HE CADC, one of
    #   APERO under the same name pattern; an instrument folder's own, and
    #   another mode's, whose name LBL writes otherwise (_nirps_he_eso)
    template(tmp_path / "templates" / "LBL_Template_GL1_nirps_he.fits",
             "NIRPS_HE", "CADC")
    template(tmp_path / "templates" / "LBL_Template_GL2_nirps_he.fits",
             "NIRPS_HE", "APERO")
    template(tmp_path / "NIRPS_HE_CADC" / "templates"
             / "LBL_Template_OWN_nirps_he.fits", "NIRPS_HE", "CADC")
    template(tmp_path / "NIRPS_HE_ESO" / "templates"
             / "LBL_Template_X_nirps_he_eso.fits", "NIRPS_HE", "ESO")
    home = str(tmp_path / "NIRPS_HE_CADC")
    rows = lblinfo.templates(home, "NIRPS_HE", "CADC")
    assert [(r["name"], r["here"], r["home"]) for r in rows] == [
        ("GL1", False, ""), ("OWN", True, None)]
    eso = lblinfo.templates(str(tmp_path / "NIRPS_HE_ESO"), "NIRPS_HE", "ESO")
    assert [r["name"] for r in eso] == ["X"]
    # a run naming GL1: linked where LBL reads it, nothing copied
    assert lblinfo.link_templates(home, "NIRPS_HE", "CADC", ["GL1"]) == ["GL1"]
    link = tmp_path / "NIRPS_HE_CADC" / "templates" / \
        "LBL_Template_GL1_nirps_he.fits"
    assert link.is_symlink() and link.resolve() == (
        tmp_path / "templates" / "LBL_Template_GL1_nirps_he.fits")
    assert lblinfo.templates(home, "NIRPS_HE", "CADC")[0]["here"]


def test_headers_read_once_and_kept_when_moved(root, monkeypatch):
    # a plan made again reads no header it has read (Downloads/HEADERS),
    #   even after the window started again; moved, a file keeps its record
    calls = []
    real = dispatch.read

    def counted(path):
        calls.append(os.path.basename(path))
        return real(path)

    monkeypatch.setattr(dispatch, "read", counted)
    first = dispatch.plan(str(root), ask_simbad=False)
    read_first = len(calls)
    assert read_first == 7
    calls.clear()
    again = dispatch.plan(str(root), ask_simbad=False)
    assert calls == [] and again["folders"] == first["folders"]
    dispatch.go(again)
    kept = json.loads((root / "Downloads" / dispatch.HEADERS).read_text())
    moved = str(root / "NIRPS_HE_APERO" / "science" / "EPSILONERI"
                / "NIRPS.2024-10-03T07:04:44.397t.fits")
    assert moved in kept["files"]
    # a new file, and only it, read: the folder listed again by its date
    _harps(root / "Downloads" / "HARPS.2017-01-01T00-00-00.000_e2ds_A.fits",
           "HR1084", ESO_EPS["alpha"], ESO_EPS["delta"])
    calls.clear()
    dispatch.plan(str(root), ask_simbad=False)
    assert "HARPS.2017-01-01T00-00-00.000_e2ds_A.fits" in calls
    assert not [c for c in calls if c.startswith("NIRPS")]
    # notes.txt is no spectrum: left, not counted as waiting
    assert dispatch.downloads(str(root))["files"] == 1


def test_the_primary_header_says_enough():
    # ESO's products, APERO's t.fits, the CADC's, the DRS's e2ds: by the
    #   primary header alone; the others' extensions are read
    assert dispatch._primary_source("NIRPS", fits.Header(
        {"HIERARCH ESO PRO CATG": "S2D_A"})) == "ESO"
    assert dispatch._primary_source("SPIROU", fits.Header(
        dict(NAXIS=2, WAVEORDN=1))) == "APERO"
    assert dispatch._primary_source("SPIROU", fits.Header(dict(NAXIS=0))) \
        == "CADC"
    assert dispatch._primary_source("HARPS", fits.Header(dict(NAXIS=2))) \
        == "ORIG"
    assert dispatch._primary_source("HARPS", fits.Header(dict(NAXIS=0))) \
        is None
    assert dispatch._primary_source("EXPRES", fits.Header()) is None


def test_what_cannot_be_dispatched_is_not_counted_as_waiting(tmp_path):
    down = tmp_path / "Downloads"
    down.mkdir()
    _harps(down / "HARPS.2014-01-01T00-00-00.000_e2ds_A.fits", "HR1084",
           ESO_EPS["alpha"], ESO_EPS["delta"])
    (down / "index.csv").write_text("a,b\n")
    # a download that did not finish
    (down / "NIRPS.2023-10-30T00:32:50.266t.fits").write_bytes(b"")
    (down / "garbled.fits").write_text("not fits at all")
    before = dispatch.downloads(str(tmp_path))
    # the garbled one is known only once a plan has read it
    assert before["files"] == 2 and before["n_left"] == 2
    plan = dispatch.plan(str(tmp_path), ask_simbad=False)
    whys = {row["why"] for row in plan["left"]}
    assert dispatch.EMPTY in whys
    after = dispatch.downloads(str(tmp_path))
    assert after["files"] == 1 and after["n_left"] == 3
    assert {row["why"] for row in after["left"]} == {
        "not a spectrum (*.csv)", dispatch.EMPTY,
        "not a FITS file astropy reads (OSError)"}


# -----------------------------------------------------------------------------
# How a new star is named, and a name given by hand
# -----------------------------------------------------------------------------
#: AD Leo where ESO's observers point (J2000, arcsec/yr)
ESO_ADLEO = dict(alpha=101936.28, delta=195212.0, pma=-0.4986, pmd=-0.0434)


def _apero_db(folder):
    """A database of APERO's with two stars, as its YAML files are."""
    (folder / "verified").mkdir(parents=True)
    entries = dict(
        EPSILONERI=("* eps Eri", ["HR 1084", "HD 22049", "* eps Eri"],
                    53.2326854, -9.4582610, -974.758, 20.876),
        GL388=("V* AD Leo", ["GJ 388", "V* AD Leo", "BD+20 2465"],
               154.9011700, 19.8700029, -498.6, -43.4))
    for name, (simbad, aliases, ra, dec, pmra, pmde) in entries.items():
        lines = ["APERO_NAME: %s" % name, "SIMBAD_NAME: '%s'" % simbad,
                 "APERO_CLASS: STAR", "RA_J2000_DEG: %r" % ra,
                 "DEC_J2000_DEG: %r" % dec,
                 "PMRA:\n  value: %r" % pmra, "PMDE:\n  value: %r" % pmde,
                 "ALIASES:"] + ["- '%s'" % one for one in aliases]
        (folder / "verified" / (name + ".yaml")).write_text(
            "\n".join(lines) + "\n")
    return folder


def test_the_database_of_apero(tmp_path, monkeypatch):
    from lbl.gui.fetch import apero_names
    monkeypatch.setenv("LBL_APERO_ASTROMETRICS",
                       str(_apero_db(tmp_path / "db")))
    # found as APERO finds a name: its spaces, signs and case aside
    assert apero_names.lookup("hr1084")["apero"] == "EPSILONERI"
    assert apero_names.lookup("GJ388")["apero"] == "GL388"
    assert apero_names.lookup("BD+20 2465")["apero"] == "GL388"
    assert apero_names.lookup("YO17") is None
    # and by its place
    near = apero_names.near(154.9012, 19.8701, 30.0)
    assert [one["apero"] for one in near] == ["GL388"]
    assert near[0]["sep"] < 1.0
    assert apero_names.near(0.0, 0.0, 30.0) == []


def test_new_stars_named_each_way(tmp_path, monkeypatch):
    monkeypatch.setenv("LBL_APERO_ASTROMETRICS",
                       str(_apero_db(tmp_path / "db")))
    down = tmp_path / "root" / "Downloads"
    down.mkdir(parents=True)
    _harps(down / "HARPS.2014-01-01T00-00-00.000_e2ds_A.fits", "HR1084",
           ESO_EPS["alpha"], ESO_EPS["delta"], ESO_EPS["pma"], ESO_EPS["pmd"])
    # AD Leo under a programme's nickname nobody knows
    _harps(down / "HARPS.2016-01-01T00-00-00.000_e2ds_A.fits", "YO17",
           **ESO_ADLEO)
    # SIMBAD's answers, as asked before
    idents = dict(HR1084=dict(main="* eps Eri", ra=53.2326854,
                              dec=-9.4582610, pmra=-974.758, pmdec=20.876),
                  YO17=None)
    named = {}
    for naming in dispatch.NAMINGS:
        plan = dispatch.plan(str(tmp_path / "root"), dict(idents),
                             ask_simbad=False, naming=naming)
        named[naming] = {r["names"][0]: (r["folder"], r["how"])
                         for r in plan["folders"]}
        assert plan["naming"] == naming
    # APERO's name, by a name of the headers or by the place
    assert named["apero"] == dict(HR1084=("EPSILONERI", "apero"),
                                  YO17=("GL388", "apero"))
    # SIMBAD's main name made a folder's; a name SIMBAD does not know
    #   (SIMBAD not asked where the star is): the headers'
    assert named["simbad"] == dict(HR1084=("eps_Eri", "simbad"),
                                   YO17=("YO17", "headers"))
    assert named["headers"] == dict(HR1084=("HR1084", "headers"),
                                    YO17=("YO17", "headers"))


def test_a_star_already_in_the_root_keeps_its_name(root):
    dispatch.go(dispatch.plan(str(root), ask_simbad=False))
    _harps(root / "Downloads" / "HARPS.2015-01-01T00-00-00.000_e2ds_A.fits",
           "eps Eri", ESO_EPS["alpha"], ESO_EPS["delta"], ESO_EPS["pma"],
           ESO_EPS["pmd"])
    plan = dispatch.plan(str(root), ask_simbad=False, naming="headers")
    row = [r for r in plan["folders"] if r["names"] == ["eps Eri"]][0]
    assert (row["folder"], row["how"]) == ("EPSILONERI", "existing")


def test_a_name_given_by_hand_is_a_folder_name():
    problem = dispatch.name_problem
    assert problem("Gl_699") is None and problem("BD+20_2465") is None
    assert problem("")[0] == "name_empty"
    assert problem(" GJ1")[0] == "name_space_ends"
    assert problem("GJ1 ")[0] == "name_space_ends"
    assert problem("GJ*1") == ("name_chars", ["*", "GJ_1"])
    assert problem("GJ 1") == ("name_chars", ["(space)", "GJ_1"])
    assert problem("_GJ1")[0] == "name_ends"
    assert problem("GJ1_pca2d")[0] == "name_suffix"
    assert problem("A" * 65)[0] == "name_long"


def test_a_planned_folder_named_by_hand(root):
    plan = dispatch.plan(str(root), ask_simbad=False)
    star = [r for r in plan["folders"] if r["names"] == ["GJ 65 A"]][0]
    assert star["folder"] == "GJ_65_A"
    # the name of the other star of the plan, or a bad one: refused
    assert dispatch.rename_planned(plan, star["star"], "GJ65B")[0] \
        == "name_taken"
    assert dispatch.rename_planned(plan, star["star"], "GJ 65A")[0] \
        == "name_chars"
    assert dispatch.rename_planned(plan, star["star"], "GJ65A") is None
    assert (star["folder"], star["how"]) == ("GJ65A", "hand")
    dispatch.go(plan)
    assert (root / "HARPS_ORIG" / "science" / "GJ65A"
            / "HARPS.2019-01-01T00-00-00.000_e2ds_A.fits").is_file()
    assert not (root / "HARPS_ORIG" / "science" / "GJ_65_A").exists()


def test_the_window_renames_a_planned_folder(root):
    box = dispatch.Dispatcher()
    box.idents = {name: None for name in ("HR1084", "K3_01", "EPSILONERI",
                                          "Eps_eri", "GJ 65 A", "GJ65B")}
    box.start_plan(str(root), naming="headers")
    for _ in range(200):
        if box.status()["state"] != "planning":
            break
        time.sleep(0.05)
    rows = box.status()["plan"]["folders"]
    star = [r for r in rows if r["names"] == ["GJ65B"]][0]["star"]
    said = box.rename(str(root), star, "GJ65B_")
    assert said["problem"]["key"] == "name_ends"
    shown = box.rename(str(root), star, "UV_Cet")["plan"]["folders"]
    assert [r["folder"] for r in shown if r["star"] == star] == ["UV_Cet"]


def test_a_folder_renamed_in_every_instrument_folder(root,
                                                     monkeypatch):
    from lbl.gui import cleanup
    monkeypatch.setattr(cleanup, "busy", lambda data_dir: [])
    dispatch.go(dispatch.plan(str(root), ask_simbad=False))
    # what LBL made of EPSILONERI in HARPS_ORIG, and of another star
    rdb = root / "HARPS_ORIG" / "lblrdb"
    rdb.mkdir()
    (rdb / "lbl2_EPSILONERI_EPSILONERI.rdb").write_text("x")
    (rdb / "lbl2_EPSILONERI2_EPSILONERI2.rdb").write_text("x")
    (root / "HARPS_ORIG" / "science" / "EPSILONERI_pca2d").mkdir()
    dry = dispatch.rename_folder(str(root), "HARPS_ORIG", "EPSILONERI",
                                 "eps_Eri", dry=True)
    assert dry["ok"] and not dry["done"]
    assert [one["home"] for one in dry["folders"]] == [
        "HARPS_ORIG", "HARPSN_ESO", "NIRPS_HE_APERO"]
    assert dry["products"] == [dict(home="HARPS_ORIG", n=2, examples=[
        "lblrdb/lbl2_EPSILONERI_EPSILONERI.rdb",
        "science/EPSILONERI_pca2d"])]
    assert (root / "HARPS_ORIG" / "science" / "EPSILONERI").is_dir()
    # a name taken in one of them: nothing renamed
    (root / "HARPSN_ESO" / "science" / "eps_Eri").mkdir()
    taken = dispatch.rename_folder(str(root), "HARPS_ORIG", "EPSILONERI",
                                   "eps_Eri")
    assert taken["problem"] == ("name_taken", ["HARPSN_ESO/science/eps_Eri"])
    assert not taken["done"]
    (root / "HARPSN_ESO" / "science" / "eps_Eri").rmdir()
    done = dispatch.rename_folder(str(root), "HARPS_ORIG", "EPSILONERI",
                                  "eps_Eri")
    assert done["done"]
    for home in ("HARPS_ORIG", "HARPSN_ESO", "NIRPS_HE_APERO"):
        assert (root / home / "science" / "eps_Eri").is_dir()
        assert not (root / home / "science" / "EPSILONERI").exists()
    lines = (root / "Downloads" / ".dispatched.jsonl").read_text()
    assert lines.count("renamed by hand") == 3
    # one instrument folder alone, and a folder that is not there
    one = dispatch.rename_folder(str(root), "HARPS_ORIG", "eps_Eri", "EPSERI",
                                 everywhere=False)
    assert [f["home"] for f in one["folders"]] == ["HARPS_ORIG"]
    assert (root / "HARPSN_ESO" / "science" / "eps_Eri").is_dir()
    assert dispatch.rename_folder(str(root), "HARPS_ORIG", "nope", "X")[
        "problem"][0] == "rename_missing"
    assert dispatch.rename_folder(str(root), "HARPS_ORIG", "../x", "X")[
        "problem"][0] == "rename_missing"


def test_the_blazes_made_are_told_as_they_go(tmp_path):
    down = tmp_path / "Downloads"
    down.mkdir()
    # three nights of HARPS without their blazes (ESO does not give them)
    for night in range(3):
        path = down / ("HARPS.2014-01-0%dT00-00-00.000_e2ds_A.fits"
                       % (night + 1))
        _harps(path, "HR1084", ESO_EPS["alpha"], ESO_EPS["delta"])
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["HIERARCH ESO DRS BLAZE FILE"] = \
                "HARPS.2014-01-0%dT20-00-00.000_blaze_A.fits" % (night + 1)
    said = []
    result = dispatch.go(dispatch.plan(str(tmp_path), ask_simbad=False),
                         lambda *args: said.append(args))
    assert len(result["made"]) == 3 and not result["failed"]
    made = sorted((tmp_path / "HARPS_ORIG" / "calib").glob("*_blaze_A.fits"))
    assert len(made) == 3
    # each says its instrument, so that it can be dispatched again
    assert fits.getheader(made[0])["INSTRUME"] == "HARPS"
    # every step told, the moves done before the blazes begin
    steps = [one[2] for one in said]
    assert (3, 3, "moving") in said and (3, 3, "blazes") in said
    assert steps.index("blazes") > max(i for i, s in enumerate(steps)
                                       if s == "moving")
    assert steps[-1] == "done"


def test_a_disk_without_links_keeps_the_models_where_they_are(
        tmp_path, monkeypatch):
    # a DATA_DIR with models of its own (a kept PCA2D basis), on a disk
    #   that refuses links (FAT, exFAT under Linux, Windows without the
    #   right): nothing of it leaves, it stays a folder of its own
    mine = tmp_path / "HARPS_ORIG" / "models" / "pca2d_HARPS_ORIG"
    mine.mkdir(parents=True)
    (mine / "basis_2026a.npz").write_text("kept")

    def refused(*args, **kwargs):
        raise OSError(1, "Operation not permitted")
    monkeypatch.setattr(os, "symlink", refused)
    monkeypatch.setattr(dispatch, "_junction", lambda target, link: False)
    done = dispatch.share(str(tmp_path), "HARPS_ORIG")
    assert "no link" in done["models"]
    assert (mine / "basis_2026a.npz").read_text() == "kept"
    assert not (tmp_path / "models" / "pca2d_HARPS_ORIG").exists()


def test_a_link_seen_as_one_however_it_is_made(tmp_path):
    # a junction (Windows) is no symbolic link to os.path.islink: the
    #   folder that leads to the root's models is linked all the same
    (tmp_path / "models").mkdir()
    (tmp_path / "HARPS_ORIG").mkdir()
    os.symlink(str(tmp_path / "models"), str(tmp_path / "HARPS_ORIG"
                                             / "models"))
    assert dispatch.sharing(str(tmp_path), "HARPS_ORIG") == dict(
        models="linked")
    assert dispatch.share(str(tmp_path), "HARPS_ORIG") == dict(
        models="linked")


def test_templates_copied_where_no_link_can_be_made(tmp_path, monkeypatch):
    from lbl.gui import lblinfo
    other = tmp_path / "HARPS_ORIG" / "templates"
    other.mkdir(parents=True)
    tpl = other / "LBL_Template_GJ1_harps_orig.fits"
    tpl.write_text("template")
    monkeypatch.setattr(lblinfo, "templates", lambda *args: [dict(
        name="GJ1", here=False, path=str(tpl))])

    def refused(*args, **kwargs):
        raise OSError(1, "Operation not permitted")
    monkeypatch.setattr(os, "symlink", refused)
    data_dir = tmp_path / "HARPS_ESO"
    assert lblinfo.link_templates(str(data_dir), "HARPS", "ESO",
                                  ["GJ1"]) == ["GJ1"]
    copy = data_dir / "templates" / tpl.name
    assert copy.read_text() == "template" and not copy.is_symlink()


def test_blazes_the_window_made_are_dispatched_again(tmp_path):
    """A blaze the window made (HIERARCH LBL BLAZE) before it wrote the
    instrument in it, put back in Downloads: placed by its name's prefix in
    the calib/ of its instrument; a copy of one already there removed, one
    that differs kept in Downloads, calib's untouched."""
    down = tmp_path / "Downloads"
    down.mkdir()
    calib = tmp_path / "HARPS_ORIG" / "calib"
    calib.mkdir(parents=True)

    def blaze(path, value):
        head = fits.Header()
        head["HIERARCH LBL BLAZE"] = ("empirical", "envelope of an e2ds")
        fits.PrimaryHDU(np.full((2, 8), value, "f4"), head).writeto(path)
    blaze(down / "HARPS.2005-01-02T20:03:59.870_blaze_A.fits", 1.0)
    blaze(down / "HARPS.2005-01-04T20:18:16.757_blaze_A.fits", 1.0)
    blaze(calib / "HARPS.2005-01-04T20:18:16.757_blaze_A.fits", 1.0)
    blaze(down / "HARPS.2005-01-05T19:58:50.363_blaze_A.fits", 1.0)
    blaze(calib / "HARPS.2005-01-05T19:58:50.363_blaze_A.fits", 2.0)
    # an unknown file without an instrument stays unknown
    fits.PrimaryHDU(np.ones((2, 8), "f4")).writeto(down / "mystery.fits")
    record = dispatch.read(str(down / "HARPS.2005-01-02T20:03:59.870_blaze_A.fits"))
    assert (record["instrument"], record["source"], record["kind"]) == (
        "HARPS", "ORIG", "calib")
    assert dispatch.read(str(down / "mystery.fits"))["why"] == \
        "no instrument in its header"
    result = dispatch.go(dispatch.plan(str(tmp_path), ask_simbad=False))
    assert (calib / "HARPS.2005-01-02T20:03:59.870_blaze_A.fits").is_file()
    assert not (down / "HARPS.2005-01-02T20:03:59.870_blaze_A.fits").exists()
    assert not (down / "HARPS.2005-01-04T20:18:16.757_blaze_A.fits").exists()
    assert result["copies"] == 1
    assert (down / "HARPS.2005-01-05T19:58:50.363_blaze_A.fits").is_file()
    assert fits.getdata(calib / "HARPS.2005-01-05T19:58:50.363_blaze_A.fits"
                        )[0, 0] == 2.0
    assert [k["name"] for k in result["kept"]] == [
        "HARPS.2005-01-05T19:58:50.363_blaze_A.fits"]


def test_harps_eggs_with_nirps_goes_with_eggs():
    """EGGS+NIRPS is HARPS's EGGS fibre observed together with NIRPS: its
    files go to <star>_EGGS, as EGGS's do, and never with HAM's"""
    from lbl.core.base import harps_eggs

    for mode, eggs in (("EGGS", True), ("EGGS+NIRPS", True),
                       ("eggs+nirps ", True), ("HARPS", False),
                       ("HARPS+NIRPS", False), (None, False), ("", False)):
        assert harps_eggs(mode) is eggs
        head = {} if mode is None else {"HIERARCH ESO INS MODE": mode}
        assert dispatch._setting("HARPS", [head]) == ("EGGS" if eggs else "")

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
    assert sorted(os.listdir(down)) == [".dispatched.jsonl", "notes.txt",
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
    # the blaze under the name the spectra's headers give it, and the
    #   setup's default blaze, in the root's models
    calib = tmp_path / "NIRPS_HE_ESO" / "calib"
    assert os.listdir(calib) == [
        "r.NIRPS.2025-09-09T12:12:29.357_BLAZE_A.fits"]
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

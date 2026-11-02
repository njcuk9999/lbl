"""lbl_pca2d, the pca2d precleaning as a step of lbl_wrap.

No spectrum is read here: what is checked is what the PCA2D dict becomes,
which LBL products a new correction moves aside, and that a wrap without
PCA2D does not even import it.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_lbl_pca2d.py
"""
import os
import subprocess
import sys

import pytest

from lbl.core.base_classes import LblException
from lbl.recipes import lbl_pca2d


def test_true_is_the_nominal_set():
    switches, overrides = lbl_pca2d.settings(True)
    assert switches == lbl_pca2d.DEFAULTS and overrides == {}
    # n_earth is the instrument's (PCA2D_N_EARTH), resolved once it is known
    assert (switches["n_star"], switches["n_earth"]) == (1, None)


def test_the_instrument_gives_its_own_settings():
    from lbl.pca2d.config import instrument_layer
    nirps = instrument_layer("NIRPS_HE", "CADC")
    assert nirps["twoframe"]["n_earth"] == 7
    assert nirps["input"]["format"] == "tfits"
    assert nirps["input"]["extensions"]["flux"] == "FluxA"
    harps = instrument_layer("HARPS", "ESO")
    assert harps["twoframe"]["n_earth"] == 2
    assert harps["input"]["to_observer"] is True
    assert harps["correct"]["mask_share"] == 0.5
    # the bands are LBL's, those that overlap the domain
    assert "k" in instrument_layer("SPIROU", "APERO")["domain"]["bands"]
    assert "k" not in nirps["domain"]["bands"]
    assert "g" in harps["domain"]["bands"]


def test_the_figures_follow_the_report_unless_asked():
    assert lbl_pca2d.settings(True, report=False)[0]["figures"] is False
    assert lbl_pca2d.settings(True, report=True)[0]["figures"] is True
    assert lbl_pca2d.settings(dict(figures=True), report=False)[0]["figures"]


def test_the_delivered_object_is_measured_too_by_default():
    assert lbl_pca2d.compare(True) and lbl_pca2d.compare(dict(n_star=1))
    assert not lbl_pca2d.compare(dict(compare=False))


def test_short_dotted_and_nested_keys_land_in_their_sections():
    switches, overrides = lbl_pca2d.settings(dict(
        n_star=2, nsig_cut=0, correct=dict(mask_share=0.5),
        **{"twoframe.iters": 8}))
    assert switches["n_star"] == 2
    assert overrides == {"correct": {"nsig_cut": 0, "mask_share": 0.5},
                         "twoframe": {"iters": 8}}


def test_an_unknown_key_stops_before_anything_is_read():
    with pytest.raises(LblException, match="nsigcut"):
        lbl_pca2d.settings(dict(nsigcut=3))
    with pytest.raises(LblException, match="correct.nonsense"):
        lbl_pca2d.settings(dict(correct=dict(nonsense=1)))


def test_the_paths_are_lbl_s_to_decide():
    with pytest.raises(LblException, match="input.directory"):
        lbl_pca2d.settings({"input.directory": "/tmp"})


def test_the_nirps_mode_follows_the_lbl_instrument():
    from lbl.pca2d.config import instrument_layer
    # the mode is the instrument class's (PCA2D_DRS_MODES)
    assert instrument_layer("NIRPS_HE", "CADC")["quality"]["drs_modes"] == \
        ["HE"]
    assert instrument_layer("NIRPS_HA", "CADC")["quality"]["drs_modes"] == \
        ["HA"]
    assert instrument_layer("SPIROU", "APERO")["quality"]["drs_modes"] is None
    block = lbl_pca2d.request_block({}, "NIRPS_HE", "CADC", None)
    assert block["lbl"] == {"instrument": "NIRPS_HE", "data_source": "CADC"}
    # what the PCA2D dict asks for is kept, over the instrument's
    block = lbl_pca2d.request_block({"quality": {"drs_modes": ["HA"]}},
                                    "NIRPS_HE", "CADC", "*t.fits")
    assert block["quality"]["drs_modes"] == ["HA"]
    assert block["input"]["pattern"] == "*t.fits"


def test_without_input_file_the_files_say_their_pattern():
    # HARPS read as ORIG: e2ds files, not the ESO archive's S2D that PCA2D
    #   takes for HARPS by default
    names = ["/s/HARPS.ADP.2014-09-23T11-00-52.687_e2ds_A.fits",
             "/s/HARPS.ADP.2014-09-23T11-01-18.413_e2ds_A.fits"]
    assert lbl_pca2d.files_pattern(names) == "*_e2ds_A.fits"
    assert lbl_pca2d.files_pattern(["/s/2950000t.fits",
                                    "/s/2950011t.fits"]) == "*t.fits"
    # names that share only .fits say nothing: PCA2D's default then
    assert lbl_pca2d.files_pattern(["/s/a.fits", "/s/b.fits"]) is None
    assert lbl_pca2d.files_pattern([]) is None


def _tree(root):
    entries = [
        "science/OBJ/a.fits", "science/OBJ_pca2d/a.fits",
        "science/OBJ_pca2d_test/a.fits",
        "lblrv/OBJ_pca2d_OBJ_pca2d/a_lbl.fits",
        "lblrv/OBJ_pca2d_test_OBJ_pca2d_test/a_lbl.fits",
        "lblrv/OBJ_OBJ/a_lbl.fits",
        "lblrdb/lbl_OBJ_pca2d_OBJ_pca2d.rdb",
        "lblrdb/lbl_OBJ_pca2d_test_OBJ_pca2d_test.rdb",
        "lblrdb/lbl_OBJ_OBJ.rdb",
        "templates/Template_s1d_OBJ_pca2d_sc1d_v_file_A.fits",
        "masks/OBJ_pca2d_pos.fits",
        "pca2d/OBJ/request.yaml",
        # a folder of the person's own, with copies named after the object
        "comparison/OBJ_pca2d_report.pdf",
    ]
    for entry in entries:
        path = os.path.join(root, entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


def test_what_lbl_made_from_the_corrected_object_and_nothing_else(tmp_path):
    _tree(str(tmp_path))
    found = sorted(os.path.relpath(p, str(tmp_path))
                   for p in lbl_pca2d.lbl_products(str(tmp_path), "OBJ_pca2d"))
    assert found == ["lblrdb/lbl_OBJ_pca2d_OBJ_pca2d.rdb",
                     "lblrv/OBJ_pca2d_OBJ_pca2d",
                     "masks/OBJ_pca2d_pos.fits",
                     "templates/Template_s1d_OBJ_pca2d_sc1d_v_file_A.fits"]


def test_a_new_correction_moves_the_old_one_aside_and_deletes_nothing(tmp_path):
    root = str(tmp_path)
    _tree(root)
    work = os.path.join(root, "pca2d", "OBJ")
    lbl_pca2d.supersede(root, os.path.join(root, "science", "OBJ_pca2d"),
                        "OBJ_pca2d", work, lambda *a, **k: None)
    assert not os.path.exists(os.path.join(root, "science", "OBJ_pca2d"))
    assert not os.path.exists(os.path.join(root, "lblrv", "OBJ_pca2d_OBJ_pca2d"))
    # the delivered object and the labelled correction are untouched
    for kept in ("science/OBJ/a.fits", "science/OBJ_pca2d_test/a.fits",
                 "comparison/OBJ_pca2d_report.pdf",
                 "lblrv/OBJ_OBJ/a_lbl.fits", "lblrdb/lbl_OBJ_OBJ.rdb",
                 "lblrdb/lbl_OBJ_pca2d_test_OBJ_pca2d_test.rdb"):
        assert os.path.exists(os.path.join(root, kept)), kept
    (aside,) = os.listdir(os.path.join(work, "superseded"))
    moved = os.path.join(work, "superseded", aside)
    assert os.path.exists(os.path.join(moved, "science", "OBJ_pca2d", "a.fits"))
    assert os.path.exists(os.path.join(moved, "lblrdb",
                                       "lbl_OBJ_pca2d_OBJ_pca2d.rdb"))


def test_the_stamp_is_the_settings_and_the_files(tmp_path):
    path = tmp_path / "a.fits"
    path.write_bytes(b"x" * 10)
    switches, _ = lbl_pca2d.settings(True)
    one = lbl_pca2d.request_stamp({"a": 1}, switches, [str(path)])
    assert one == lbl_pca2d.request_stamp({"a": 1}, switches, [str(path)])
    assert one != lbl_pca2d.request_stamp({"a": 2}, switches, [str(path)])
    path.write_bytes(b"x" * 11)
    assert one != lbl_pca2d.request_stamp({"a": 1}, switches, [str(path)])
    # the figures change nothing in the spectra, and not the stamp either
    other = dict(switches, figures=False)
    path.write_bytes(b"x" * 10)
    os.utime(str(path), (0, 0))
    assert lbl_pca2d.request_stamp({"a": 1}, switches, [str(path)]) == \
        lbl_pca2d.request_stamp({"a": 1}, other, [str(path)])


def test_a_wrap_without_pca2d_never_imports_it(tmp_path):
    """A wrap without a PCA2D key: no lbl_pca2d, no lbl.pca2d."""
    code = (
        "import sys\n"
        "from lbl.recipes import lbl_wrap\n"
        "lbl_wrap.main(dict(INSTRUMENT='SPIROU', DATA_SOURCE='APERO',"
        " DATA_DIR=%r, DATA_TYPES=['SCIENCE'], OBJECT_SCIENCE=['X'],"
        " OBJECT_COMPARISON=['X'], OBJECT_TEFF=[3000], REPORT=False))\n"
        "print(sorted(m for m in sys.modules if m.startswith('lbl.pca2d')\n"
        "             or m == 'lbl.recipes.lbl_pca2d'))\n"
        % str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_n_star_counts_the_star_spectra_with_the_template():
    from lbl.pca2d.config import describe_star, free_star_components
    from lbl.pca2d.lblscan import parse_tag
    # 1: the template alone, not adjusted in depth; 2: one free component
    assert free_star_components(1) == 0 and free_star_components(2) == 1
    with pytest.raises(ValueError, match='template'):
        free_star_components(0)
    with pytest.raises(LblException, match='n_star = 0'):
        lbl_pca2d.settings(dict(n_star=0))
    with pytest.raises(LblException, match='give n_star'):
        lbl_pca2d.settings({'twoframe.n_star': 1})
    assert 'not adjusted in depth' in describe_star(0)
    assert 'free amplitude' in describe_star(1)
    # run folders: s<n_star>-o<n_earth>, and <n_star - 1>-<n_earth> as older
    #   versions named them
    assert parse_tag('s1-o7') == (1, 7) and parse_tag('0-7') == (1, 7)
    assert parse_tag('s2-o3v') == (2, 3) and parse_tag('1-3') == (2, 3)


def test_the_metadata_keywords_are_the_instrument_s():
    from astropy.io import fits
    from lbl.pca2d import tfits
    from lbl.pca2d.config import instrument_layer
    keys = instrument_layer("NIRPS_HE", "CADC")["input"]["header_keys"]
    assert keys["airmass_end"][0] == "HIERARCH ESO TEL AIRM END"
    assert keys["teff"] == ["OBJTEMP", "PP_TEFF"]
    science, primary = fits.Header(), fits.Header()
    science["BJD"], science["BERV"] = 2460000.5, 12.3
    primary["HIERARCH ESO TEL AIRM START"] = 1.2
    primary["HIERARCH ESO TEL AIRM END"] = 1.4
    primary["DRSMODE"], primary["OBJECT"] = "HE", "STAR"
    meta = tfits.header_meta((science, primary), keys)
    assert (meta["bjd"], meta["berv"]) == (2460000.5, 12.3)
    assert abs(meta["airmass"] - 1.3) < 1e-12
    assert (meta["drs_mode"], meta["object"]) == ("HE", "STAR")
    assert meta["humidity"] != meta["humidity"]          # not written: NaN
    # an instrument reporting one airmass gives it for both ends
    spirou = instrument_layer("SPIROU", "APERO")["input"]["header_keys"]
    assert spirou["airmass_end"] == spirou["airmass_start"] == ["AIRMASS"]
    harps = instrument_layer("HARPS", "ESO")["input"]["header_keys"]
    assert harps["drs_mode"] == ["HIERARCH ESO INS MODE"]


def test_pdflatex_can_be_named(monkeypatch, tmp_path):
    import os
    from lbl.core import io
    fake = tmp_path / "pdflatex"
    fake.write_text("")
    os.chmod(fake, 0o755)
    monkeypatch.setenv("LBL_PDFLATEX", str(fake))
    io.find_pdflatex.cache_clear()
    try:
        assert io.find_pdflatex() == str(fake)
    finally:
        io.find_pdflatex.cache_clear()

"""lbl_pca2d, the pca2d precleaning as a step of lbl_wrap.

No spectrum is read here: what is checked is what the PCA2D dict becomes,
which LBL products a new correction moves aside, and that a wrap without
PCA2D does not even import it.

    PYTHONPATH=<this checkout> python -m pytest tests/test_lbl_pca2d.py
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
    assert (switches["n_star"], switches["n_earth"]) == (1, 3)


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
    block = lbl_pca2d.request_block({}, "NIRPS_HE", "CADC", None)
    assert block["quality"]["drs_modes"] == ["HE"]
    assert block["lbl"] == {"instrument": "NIRPS_HE", "data_source": "CADC"}
    block = lbl_pca2d.request_block({"quality": {"drs_modes": ["HA"]}},
                                    "NIRPS_HE", "CADC", "*t.fits")
    assert block["quality"]["drs_modes"] == ["HA"]
    assert block["input"]["pattern"] == "*t.fits"
    assert "quality" not in lbl_pca2d.request_block({}, "SPIROU", "APERO", "*")


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
    """LBL as it was: no PCA2D key, no lbl_pca2d, no lbl.pca2d."""
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

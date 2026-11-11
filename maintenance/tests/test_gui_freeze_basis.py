"""The Runs tab's Freeze basis (lbl/gui/lblinfo.py: run_basis,
freeze_basis): the observer basis a run fitted, found by its name and its
date, kept under a name of one's own in models/pca2d_<INSTRUMENT>_<SOURCE>,
never over a basis or a target already there, never twice."""
import datetime
import os

import numpy as np
import pytest
from astropy.io import fits

from lbl.gui import lblinfo
from lbl.pca2d import basis as pbasis

#: when the run ran (Unix time) and when its fit wrote the basis
START = datetime.datetime(2026, 11, 10, 16, 1, 12).timestamp()
END = START + 300.0


def _basis(path, name, made, ident="f0b6ec1e745e", rundir="s1-o7"):
    head = fits.Header()
    for key, value in (("BASISNAM", name), ("BASISID", ident),
                       ("INSTRUME", "NIRPS_HE"), ("DATASRC", "APERO"),
                       ("NEARTH", 7), ("RUNDIR", rundir),
                       ("DATE", datetime.datetime.fromtimestamp(made)
                        .isoformat(timespec="seconds"))):
        head[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=head), fits.ImageHDU(
        np.arange(6.0).reshape(2, 3), name="COMPONENTS")]).writeto(path)


def _run(pca2d=None, state="done", targets=("TOI-206",)):
    return dict(id="261110_160112_TOI-206_s1-o7", state=state, started=START,
                ended=END, meta=dict(targets=list(targets),
                                     instrument="NIRPS_HE",
                                     pca2d=pca2d if pca2d is not None
                                     else dict(n_star=1, n_earth=7)))


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "science" / "TOI-206").mkdir(parents=True)
    _basis(pbasis.component_file("observer", "TOI-206", "NIRPS_HE",
                                 str(tmp_path)), "TOI-206", END - 1)
    return str(tmp_path)


def test_the_basis_a_run_fitted_and_where_it_stands(data_dir):
    got = lblinfo.run_basis(data_dir, _run())
    assert got["state"] == "available" and got["label"] == "TOI-206"
    assert got["file"] == "PCA2D_Observer_TOI-206_nirps_he.fits"
    # no PCA2D; a frozen basis read; a run not over
    assert lblinfo.run_basis(data_dir, _run(pca2d={})) is None
    assert lblinfo.run_basis(data_dir, _run(pca2d=dict(
        input_name="2026a")))["state"] == "used"
    assert lblinfo.run_basis(data_dir, _run(state="running"))["state"] \
        == "running"
    # the fit of other settings, or a later one, wrote over it
    assert lblinfo.run_basis(data_dir, _run(pca2d=dict(
        n_star=1, n_earth=5)))["state"] == "replaced"
    later = dict(_run(), started=START - 3600, ended=END - 3600)
    assert lblinfo.run_basis(data_dir, later)["state"] == "replaced"
    # another target: no basis of its name
    assert lblinfo.run_basis(data_dir, _run(targets=("GJ1",)))["state"] \
        == "missing"


def test_a_basis_frozen_under_a_name(data_dir):
    run = _run()
    dry = lblinfo.freeze_basis(data_dir, run, "nirps_2026a", dry=True)
    assert dry["ok"] and not dry["done"] and not os.path.exists(dry["path"])
    done = lblinfo.freeze_basis(data_dir, run, "nirps_2026a")
    assert done["done"]
    assert done["path"] == os.path.join(
        data_dir, "models", "pca2d_NIRPS_HE_APERO",
        "PCA2D_Observer_nirps_2026a_nirps_he.fits")
    head = fits.getheader(done["path"])
    assert head["BASISNAM"] == "nirps_2026a"
    assert head["BASISID"] == "f0b6ec1e745e" and head["FROMRUN"] == run["id"]
    # the run's own file stays; the PCA2D settings now offer the frozen one
    assert os.path.exists(pbasis.component_file(
        "observer", "TOI-206", "NIRPS_HE", data_dir))
    offered = [b for b in lblinfo.bases(data_dir, "NIRPS_HE", "APERO")
               if b["kept"]]
    assert [(b["name"], b["compatible"]) for b in offered] == [
        ("nirps_2026a", True)]
    # the run says it is frozen, and it is not frozen twice
    assert lblinfo.run_basis(data_dir, run)["frozen"] == ["nirps_2026a"]
    again = lblinfo.freeze_basis(data_dir, run, "other_name")
    assert again["problem"] == dict(key="freeze_already",
                                    args=["nirps_2026a"])


def test_the_names_refused(data_dir):
    run = _run()
    _basis(os.path.join(data_dir, "models", "pca2d_SPIROU_CADC",
                        "PCA2D_Observer_spirou_SNR15-25_spirou.fits"),
           "spirou_SNR15-25", END, ident="0123456789ab")

    def refused(name):
        got = lblinfo.freeze_basis(data_dir, run, name, dry=True)
        assert not got["ok"]
        return got["problem"]["key"]

    # no file name
    assert refused("nirps 2026") == "name_chars"
    assert refused(" a") == "name_space_ends"
    assert refused("") == "name_empty"
    # a basis already there, kept or in components, any instrument, any case
    assert refused("SPIROU_snr15-25") == "freeze_taken"
    assert refused("toi-206") == "freeze_taken"
    said = lblinfo.freeze_basis(data_dir, run, "toi-206", dry=True)["problem"]
    assert said["args"] == ["toi-206", os.path.join(
        "pca2d", "components", "PCA2D_Observer_TOI-206_nirps_he.fits")]
    # a target of the folder: its next fit would write over it
    os.makedirs(os.path.join(data_dir, "science", "GJ1"))
    assert refused("GJ1") == "freeze_target"
    # names LBL gives its folders are fine for a basis
    assert lblinfo.freeze_basis(data_dir, run, "nirps_pca2d",
                                dry=True)["ok"]

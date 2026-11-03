"""A resolved configuration reads back as itself.

The order is wrap over instrument over nominal, and every stage of a run reads
the flat file the first one wrote. If the instrument's class were applied over
that file, what the PCA2D dict said about anything the class also sets would be
lost on read back, and the cube key could differ between two stages.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_fixed_point.py
"""
import os

import pytest
import yaml

from lbl.pca2d import config as spconfig
from lbl.pca2d import cli


NOMINAL = os.path.join(os.path.dirname(spconfig.__file__), "nominal.yaml")

#: one of each kind of key the instrument's layer also sets
ASKED = dict(
    input=dict(pattern="*X.fits", nightly_stack=False, to_observer=True),
    twoframe=dict(n_earth=3),
    correct=dict(mask_share=0.25),
    output=dict(windows=["1267:2"]),
    domain=dict(wave_min=1000.0, wave_max=1800.0),
    quality=dict(drs_modes=["HA"]),
)


def _request(tmp_path, object_name, block, instrument, source):
    """A request as the recipe writes it: nominal, with the object's block."""
    with open(NOMINAL, encoding="utf-8") as handle:
        nominal = yaml.safe_load(handle)
    nominal["objects"] = {object_name: dict(
        block, lbl=dict(instrument=instrument, data_source=source))}
    path = os.path.join(str(tmp_path), "request.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(nominal, handle, sort_keys=False)
    return path


def _resolved(tmp_path, config):
    path = os.path.join(str(tmp_path), "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False,
                       default_flow_style=False)
    return path


@pytest.mark.parametrize("instrument, source", [("NIRPS_HE", "CADC"),
                                                ("HARPS", "ORIG"),
                                                ("SPIROU", "APERO")])
def test_a_resolved_file_reads_back_as_itself(tmp_path, instrument, source):
    request = _request(tmp_path, "STAR", ASKED, instrument, source)
    first = spconfig.load_config(request, object_name="STAR")
    # the object's block won over the instrument's class
    assert first["twoframe"]["n_earth"] == 3
    assert first["input"]["pattern"] == "*X.fits"
    assert first["correct"]["mask_share"] == 0.25
    back = spconfig.load_config(_resolved(tmp_path, first))
    assert back == first
    assert spconfig.cache_key(back) == spconfig.cache_key(first)
    # and again: it is a fixed point, not one more step towards one
    assert spconfig.load_config(_resolved(tmp_path, back)) == first


def test_instrument_wins_over_nominal_and_object_over_instrument(tmp_path):
    request = _request(tmp_path, "STAR", dict(twoframe=dict(n_earth=3)),
                       "HARPS", "ESO")
    layer = spconfig.instrument_layer("HARPS", "ESO")
    plain = spconfig.load_config(NOMINAL, instrument="HARPS", data_source="ESO")
    assert plain["twoframe"]["n_earth"] == layer["twoframe"]["n_earth"]
    assert plain["input"]["pattern"] == layer["input"]["pattern"]
    # another object of the same file gets the class's number
    other = spconfig.load_config(request, object_name="OTHER",
                                 instrument="HARPS", data_source="ESO")
    assert other["twoframe"]["n_earth"] == layer["twoframe"]["n_earth"]
    own = spconfig.load_config(request, object_name="STAR")
    assert own["twoframe"]["n_earth"] == 3


def test_a_class_changed_after_the_run_does_not_rewrite_it(tmp_path, monkeypatch):
    """A run keeps its own cube key when the class's domain moves afterwards."""
    request = _request(tmp_path, "STAR", {}, "NIRPS_HE", "CADC")
    first = spconfig.load_config(request, object_name="STAR")
    path = _resolved(tmp_path, first)
    real = spconfig.instrument_layer

    def moved(instrument, source):
        layer = real(instrument, source)
        layer["domain"]["wave_max"] += 1.5
        layer["twoframe"]["n_earth"] += 4
        return layer
    monkeypatch.setattr(spconfig, "instrument_layer", moved)
    back = spconfig.load_config(path)
    assert back == first
    assert spconfig.cache_key(back) == spconfig.cache_key(first)


def test_a_flat_file_with_only_some_keys_gets_the_rest_from_the_class(tmp_path):
    path = os.path.join(str(tmp_path), "flat.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(lbl=dict(instrument="HARPS", data_source="ESO"),
                            twoframe=dict(n_earth=5)), handle)
    cfg = spconfig.load_config(path)
    layer = spconfig.instrument_layer("HARPS", "ESO")
    assert cfg["twoframe"]["n_earth"] == 5
    assert cfg["input"]["pattern"] == layer["input"]["pattern"]
    assert cfg["input"]["instrument"] == "HARPS"


def test_a_flat_file_of_another_instrument_stops(tmp_path):
    path = _resolved(tmp_path, dict(lbl=dict(instrument="HARPS",
                                             data_source="ESO")))
    with pytest.raises(SystemExit, match="NIRPS_HE"):
        spconfig.load_config(path, instrument="NIRPS_HE", data_source="ESO")
    with pytest.raises(SystemExit, match="ORIG"):
        spconfig.load_config(path, instrument="HARPS", data_source="ORIG")
    # the same names are no contradiction
    assert spconfig.load_config(path, instrument="HARPS")["input"][
        "instrument"] == "HARPS"


def test_harps_with_another_pattern_gets_one_cube_key(tmp_path):
    """The file's pattern against the class's: one key in every stage."""
    request = _request(tmp_path, "STAR",
                       dict(input=dict(pattern="*_e2ds_A.fits")),
                       "HARPS", "ORIG")
    first = spconfig.load_config(request, object_name="STAR")
    layer = spconfig.instrument_layer("HARPS", "ORIG")
    assert layer["input"]["pattern"] != "*_e2ds_A.fits"
    back = spconfig.load_config(_resolved(tmp_path, first))
    assert spconfig.cache_key(back) == spconfig.cache_key(first)


def test_the_run_tag_and_the_fit_agree_after_read_back(tmp_path):
    request = _request(tmp_path, "STAR", dict(twoframe=dict(n_earth=3)),
                       "NIRPS_HE", "CADC")
    first = spconfig.load_config(request, object_name="STAR")
    assert cli.run_tag(first) == "s1-o3"
    back = spconfig.load_config(_resolved(tmp_path, first))
    assert cli.run_tag(back) == "s1-o3"

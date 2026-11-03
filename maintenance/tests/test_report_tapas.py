"""The report finds LBL's TAPAS file where LBL keeps it: its copy in other/,
else models/tellurics/ (lbl.report.main._tapas_file).

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_report_tapas.py
"""
import os

from lbl.report import main as report_main


def test_the_tapas_file_is_found_where_lbl_keeps_it(tmp_path):
    assert report_main._tapas_file(str(tmp_path)) is None
    models = tmp_path / "models" / "tellurics"
    models.mkdir(parents=True)
    (models / "tapas_lbl.fits").write_bytes(b"")
    assert report_main._tapas_file(str(tmp_path)) == str(
        models / "tapas_lbl.fits")
    # LBL's copy in other/ first, as tellu_clean.get_tapas_lbl reads it
    other = tmp_path / "other"
    other.mkdir()
    (other / "tapas_lbl.fits").write_bytes(b"")
    assert report_main._tapas_file(str(tmp_path)) == str(other / "tapas_lbl.fits")


def test_a_tapas_file_left_at_the_top_of_models_is_found(tmp_path):
    """Older versions of LBL kept the models at the top of models/."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tapas_lbl.fits").write_bytes(b"")
    found = report_main._tapas_file(str(tmp_path))
    assert found == os.path.join(str(tmp_path), "models", "tellurics",
                                 "tapas_lbl.fits")


def test_the_zero_point_figure_is_handed_that_file(tmp_path, monkeypatch):
    from lbl.instruments import espadons
    models = tmp_path / "models" / "tellurics"
    models.mkdir(parents=True)
    (models / "tapas_lbl.fits").write_bytes(b"")
    seen = {}

    def figure(plt, path, tapas_file=None, title=""):
        seen["tapas"] = tapas_file
    monkeypatch.setattr(espadons, "telluric_figure", figure)
    report_main._zero_point_figure(None, "x.fits", str(tmp_path), "t")
    assert seen["tapas"] == str(models / "tapas_lbl.fits")

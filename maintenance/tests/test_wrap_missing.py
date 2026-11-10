"""What lbl_compute needs and is not there, found by lbl_wrap before it runs
(lbl_wrap._missing_for_compute): the templates of the science object and of
the comparison, the comparison's mask, by the paths LBL's class gives them.
The wrap makes them then, said, rather than lbl_compute stopping on them
(RUN_LBL_TEMPLATE off, a template a PCA2D correction made again moved to
superseded/)."""
from astropy.table import Table

from lbl.recipes import lbl_wrap


def _ask(data_dir, science, comparison):
    return lbl_wrap._missing_for_compute(
        {}, instrument="NIRPS_HE", data_dir=str(data_dir), data_source="ESO",
        data_type="SCIENCE", object_science=science,
        object_comparison=comparison, blaze_corrected=False, blaze_file=None)


def test_the_templates_and_mask_compute_needs(tmp_path):
    (tmp_path / "templates").mkdir()
    (tmp_path / "masks").mkdir()
    table = Table(dict(wavelength=[1.0], flux=[1.0]))
    table.write(tmp_path / "templates" / "LBL_Template_GJ581_nirps_he_eso.fits")
    # TOI-206_pca2d's template is not there, the comparison's is, its mask
    #   is not
    assert _ask(tmp_path, "TOI-206_pca2d", "GJ581") == (["TOI-206_pca2d"],
                                                        True)
    table.write(tmp_path / "masks" / "LBL_Mask_GJ581_full_nirps_he_eso.fits")
    table.write(tmp_path / "templates"
                / "LBL_Template_TOI-206_pca2d_nirps_he_eso.fits")
    assert _ask(tmp_path, "TOI-206_pca2d", "GJ581") == ([], False)
    # one object, its own comparison: asked once
    assert _ask(tmp_path, "GJ2", "GJ2") == (["GJ2"], True)

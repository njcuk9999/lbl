"""ESO's pipelines (espdr: NIRPS, HARPS, HARPS-N at the TNG, ESPRESSO) write
0 where they have no data: LBL's FITS reader gives them as NaN, for these
products only (lbl.core.io.eso_zeros_as_nan); and the header of an rdb
combines a value too large to be squared without a warning."""
import warnings

import numpy as np
import pytest
from astropy.io import fits

from lbl.core import io
from lbl.science import general


def _s2d(path, eso=True):
    head = fits.Header()
    if eso:
        head["HIERARCH ESO PRO CATG"] = "S2D_BLAZE_TELL_CORR_A"
    flux = np.array([[0.0, 2.0, 3.0], [4.0, 0.0, np.nan]])
    fits.HDUList([fits.PrimaryHDU(header=head),
                  fits.ImageHDU(flux, name="SCIDATA"),
                  fits.ImageHDU(np.array([[0, 1, 0]], dtype=np.uint16),
                                name="QUALDATA")]).writeto(path)


def test_an_eso_product_s_zeros_are_nan(tmp_path):
    _s2d(tmp_path / "eso.fits")
    flux = io.load_fits(str(tmp_path / "eso.fits"), extname="SCIDATA")
    assert np.isnan(flux).sum() == 3 and not (flux == 0).any()
    # by number, and the first extension with data when none is named
    assert np.isnan(io.load_fits(str(tmp_path / "eso.fits"),
                                 extnum=1)).sum() == 3
    assert np.isnan(io.load_fits(str(tmp_path / "eso.fits"))).sum() == 3
    # integers (QUALDATA) as they are
    qual = io.load_fits(str(tmp_path / "eso.fits"), extname="QUALDATA")
    assert qual.dtype == np.uint16 and (qual == 0).sum() == 2
    datas, _ = io.load_fits_multi(str(tmp_path / "eso.fits"), ["SCIDATA"],
                                  "SCIDATA")
    assert np.isnan(datas[0]).sum() == 3


def test_another_pipeline_s_zeros_stay(tmp_path):
    _s2d(tmp_path / "other.fits", eso=False)
    flux = io.load_fits(str(tmp_path / "other.fits"), extname="SCIDATA")
    assert (flux == 0).sum() == 2 and np.isnan(flux).sum() == 1


def test_a_header_value_too_large_to_square(tmp_path):
    key = "HIERARCH ESO INS PRES102 VAL"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = general.combine_rv_headers({key: [2.45e198, 1.0, 2.0]},
                                         {key: ""})
    std = next(v for k, v in out.items() if k.startswith("HIERARCH CSTD"))
    mean = next(v for k, v in out.items() if k.startswith("HIERARCH CAVG"))
    assert std[0] == "NaN" and mean[0] == pytest.approx(2.45e198 / 3)

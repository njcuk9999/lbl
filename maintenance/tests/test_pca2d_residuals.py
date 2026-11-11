#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA2D's residuals (save_residuals): the fit's residual of every row over the
whole grid (outliers.residual_rows), put back on each order's own pixels
(reconstruct.on_pixels) in a copy of the source whose flux it replaces
(reconstruct.residual_file); the switch of lbl_pca2d, its folder, and the
window's view of it (an option, a product, not a target).

Created on 2026-11-11

@author: artigau
"""
import os

import numpy as np
import pytest
from astropy import constants
from astropy.io import fits
from astropy.table import Table

from lbl.pca2d import outliers, reconstruct
from lbl.pca2d import tfits as sptf
from lbl.pca2d.grids import pixel_shift

C_KMS = constants.c.to("km/s").value


def _cube(tmp_path, n_files=6, m=1200, dv=0.5, seed=7, star=0.0):
    """A t.fits-like cube, two rows per exposure: noise, an observer
    component with its amplitudes, a star component carried by the BERV
    when `star`; and the fit that holds them, as twoframe saves it."""
    rng = np.random.default_rng(seed)
    n = 2 * n_files
    berv = np.repeat(np.linspace(-15.0, 15.0, n_files), 2)
    grid = 1500.0 * np.exp(np.arange(m) * dv / C_KMS)
    x = np.arange(m)
    Q = np.exp(-0.5 * ((x - 400) / 3.0) ** 2)[None, :]
    b = rng.normal(0.0, 0.05, (n, 1))
    P = star * np.exp(-0.5 * ((x - 800) / 4.0) ** 2)[None, :]
    a = np.ones((n, 1))
    data = (rng.normal(0.0, 0.01, (n, m)) + b @ Q).astype(np.float32)
    shift = np.rint(-pixel_shift(berv, dv)).astype(int)
    for r in range(n):
        data[r] += np.roll(P[0], shift[r]).astype(np.float32)
    sigma = np.full((n, m), 0.01, dtype=np.float32)
    sigma[3, 100] = np.nan                       # a sample with no weight
    path = os.path.join(str(tmp_path), "cube")
    os.makedirs(path)
    np.save(os.path.join(path, "grid.npy"), grid)
    np.save(os.path.join(path, "data.npy"), data)
    np.save(os.path.join(path, "sigma.npy"), sigma)
    names = ["e%02d.fits" % (r // 2) for r in range(n)]
    Table(dict(filename=names, berv=berv, parity=np.arange(n) % 2,
               exposure=np.arange(n) // 2, snr_band=np.full(n, 100.0))
          ).write(os.path.join(path, "meta.fits"))
    fit = os.path.join(str(tmp_path), "fit.npz")
    np.savez(fit, P=P, a=a, Q=Q, b=b, berv=berv, dv=dv)
    return path, fit, data, b @ Q


def test_the_residual_is_the_data_less_the_model(tmp_path):
    cube, fit_path, data, observer = _cube(tmp_path)
    rows, names, parity = outliers.residual_rows(cube, np.load(fit_path))
    assert rows.shape == data.shape
    assert names[:4] == ["e00.fits", "e00.fits", "e01.fits", "e01.fits"]
    assert list(parity[:4]) == [0, 1, 0, 1]
    expected = data - observer
    ok = np.isfinite(rows)
    assert np.allclose(rows[ok], expected[ok], atol=1e-6)
    # no weight, no residual: where the cube weighs nothing (the sample
    #   without a sigma, the grid's ends), and only there
    from lbl.pca2d.twoframe import fit_weight_settings, load_cube
    weights = load_cube(cube, dtype=np.float32, **fit_weight_settings(
        np.load(fit_path)))[2]
    assert np.isnan(rows[3, 100])
    assert np.array_equal(ok, weights > 0)
    assert ok.mean() > 0.8


def test_the_residual_is_the_same_block_by_block(tmp_path):
    """Column blocks with the margin the star's shift needs: the same as
    the whole grid at once, a star component carried by the BERV in."""
    cube, fit_path, _, _ = _cube(tmp_path, star=0.05)
    fit = np.load(fit_path)
    whole, _, _ = outliers.residual_rows(cube, fit, block=None)
    blocks, _, _ = outliers.residual_rows(cube, fit, block=250)
    assert np.allclose(whole, blocks, atol=1e-6, equal_nan=True)
    # the star is taken out with the observer: what is left is the noise
    assert np.nanstd(whole) < 0.012
    # into an array on disk, of the cube's shape only
    out = np.lib.format.open_memmap(str(tmp_path / "rows.npy"), mode="w+",
                                    dtype=np.float32, shape=whole.shape)
    outliers.residual_rows(cube, fit, out, block=400)
    assert np.allclose(np.asarray(out), whole, atol=1e-6, equal_nan=True)
    with pytest.raises(ValueError):
        outliers.residual_rows(cube, fit, np.empty((2, 3), np.float32))


def test_on_pixels_is_linear_and_says_nothing_beside_a_gap():
    grid = np.linspace(1000.0, 1010.0, 11)
    values = 2.0 * np.arange(11.0)
    values[7] = np.nan
    wave = np.array([999.0, 1000.0, 1002.5, 1006.5, 1007.5, 1010.0, 1011.0])
    got = reconstruct.on_pixels(grid, values, wave)
    assert np.isnan(got[0]) and np.isnan(got[-1])          # off the grid
    assert got[1] == pytest.approx(0.0) and got[2] == pytest.approx(5.0)
    # beside the NaN sample on either side: nothing
    assert np.isnan(got[3]) and np.isnan(got[4])
    assert got[5] == pytest.approx(20.0)


def _tfits(path, wave):
    head = fits.Header()
    head["OBJECT"] = "X"
    hdus = [fits.PrimaryHDU(header=head),
            fits.ImageHDU(np.full(wave.shape, 100.0), name="FluxA"),
            fits.ImageHDU(wave, name="WaveA"),
            fits.ImageHDU(np.full(wave.shape, 0.5), name="BlazeA"),
            fits.ImageHDU(np.ones(wave.shape), name="Recon")]
    fits.HDUList(hdus).writeto(path)


def test_a_residual_file_is_its_source_with_the_residual_as_flux(tmp_path):
    sptf.use_extensions(dict(flux="FluxA", wave="WaveA", blaze="BlazeA",
                             recon="Recon"))
    try:
        grid = np.linspace(1000.0, 1100.0, 1001)
        rows = np.vstack([0.001 * (grid - 1000.0), -0.002 * (grid - 1000.0)])
        wave = np.vstack([np.linspace(1010.0, 1020.0, 50),
                          np.linspace(1015.0, 1025.0, 50),
                          np.linspace(1090.0, 1110.0, 50)])
        src = tmp_path / "src" / "NIRPS.2024-01-01T00:00:00.000t.fits"
        src.parent.mkdir()
        _tfits(str(src), wave)
        model = dict(n_star=0, n_earth=7)
        new = reconstruct.residual_file(model, str(src), rows, {0: 0, 1: 1},
                                        grid, str(tmp_path / "res"), berv=3.5)
        assert os.path.basename(new) == src.name
        with fits.open(new) as out, fits.open(str(src)) as before:
            resid = out["FluxA"].data
            # each order from the row of its parity, on its own pixels
            assert np.allclose(resid[0], 0.001 * (wave[0] - 1000.0))
            assert np.allclose(resid[1], -0.002 * (wave[1] - 1000.0))
            # past the grid's end: NaN
            assert np.isnan(resid[2][wave[2] > 1100.0]).all()
            for name in ("WaveA", "BlazeA", "Recon"):
                assert np.array_equal(out[name].data, before[name].data)
            head = out[0].header
            assert head["PCA2RES"] and head["PCA2RNEA"] == 7
            assert head["PCA2BERV"] == 3.5 and head["OBJECT"] == "X"
        # a cube of one row per exposure: that row for every order
        new = reconstruct.residual_file(model, str(src), rows, {0: 1}, grid,
                                        str(tmp_path / "one"))
        with fits.open(new) as out:
            assert np.allclose(out["FluxA"].data[0],
                               -0.002 * (wave[0] - 1000.0))
    finally:
        sptf.use_extensions(None)


def test_put_flux_replaces_rather_than_multiplies():
    from lbl.pca2d import readers
    image = fits.HDUList([fits.PrimaryHDU(np.full((2, 3), 7.0))])
    readers.put_flux(image, (0, None), np.zeros((2, 3)))
    assert not image[0].data.any()
    table = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(
        [fits.Column(name="spectrum", format="3D",
                     array=np.full((2, 3), 7.0))], name="optimal")])
    readers.put_flux(table, ("optimal", "spectrum"), -np.ones((2, 3)))
    assert (table["optimal"].data["spectrum"] == -1).all()


# =============================================================================
# The switch, its folder, the window
# =============================================================================
def test_the_switch_is_off_unless_asked():
    from lbl.recipes import lbl_pca2d
    assert lbl_pca2d.DEFAULTS["save_residuals"] is False
    switches = lbl_pca2d.settings(dict(save_residuals=True), report=True)[0]
    assert switches["save_residuals"] is True
    assert lbl_pca2d.settings(dict(), report=True)[0]["save_residuals"] \
        is False


def test_a_new_correction_moves_the_old_residuals_aside(tmp_path):
    from lbl.recipes import lbl_pca2d
    data_dir = tmp_path / "data"
    corrected = data_dir / "science" / "GJ1_pca2d"
    residuals = data_dir / "science" / "GJ1_residuals"
    for folder in (corrected, residuals):
        folder.mkdir(parents=True)
        (folder / "a.fits").write_bytes(b"x")
    work = tmp_path / "work"
    said = []
    lbl_pca2d.supersede(str(data_dir), str(corrected), "GJ1_pca2d",
                        str(work), lambda *a: said.append(a),
                        extra=[str(residuals)])
    assert not residuals.exists() and not corrected.exists()
    aside = list((work / "superseded").glob("*/science/GJ1_residuals/a.fits"))
    assert len(aside) == 1 and said


def test_the_window_offers_it_and_files_it_as_a_product(tmp_path):
    from lbl.core.simbad import SUFFIXES
    from lbl.gui import cleanup, i18n, lblinfo
    from lbl.gui.targets import DERIVED
    row = next(r for r in lblinfo.pca2d_options("NIRPS_HE")
               if r["key"] == "save_residuals")
    assert (row["default"], row["kind"], row["group"]) == (False, "bool",
                                                           "runs")
    assert "save_residuals" in i18n.OPTIONS
    assert i18n.EN["what_residuals"] and i18n.FR["what_residuals"]
    # a folder of residuals is no target, and names its star
    for name in ("GJ1_residuals", "GJ1_residuals_SNR50-60"):
        assert DERIVED.search(name) and SUFFIXES.sub("", name) == "GJ1"
    (tmp_path / "science" / "GJ1").mkdir(parents=True)
    (tmp_path / "science" / "GJ1" / "a.fits").write_bytes(b"x")
    (tmp_path / "science" / "GJ1_residuals").mkdir()
    (tmp_path / "science" / "GJ1_residuals" / "a.fits").write_bytes(b"x")
    kinds = {os.path.basename(item["path"]): item for item in
             cleanup.survey(str(tmp_path))}
    assert kinds["GJ1_residuals"]["code"] == "residuals"
    assert kinds["GJ1"]["code"] == "spectra"

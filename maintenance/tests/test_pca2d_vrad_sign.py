"""The velocity the fit's shift term absorbs (vrad_fit, PCASTR_V) has LBL's
sign: a receding star is positive.

A cube of a star of Gaussian lines moving on a known receding series, under
a BERV, with noise and no sky, is fitted by the real twoframe.main with the
velocity term on, and vrad_fit is compared with the truth.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_vrad_sign.py
"""
import contextlib
import io
import os

import numpy as np
import yaml
from astropy import constants
from astropy.table import Table

from lbl.pca2d import grids, twoframe

C_KMS = constants.c.to("km/s").value


def _lines(x, centres, depths, sigma):
    out = np.zeros(x.size)
    for c, d in zip(centres, depths):
        lo, hi = max(0, int(c - 7 * sigma)), min(x.size, int(c + 7 * sigma) + 1)
        u = (x[lo:hi] - c) / sigma
        out[lo:hi] += d * np.exp(-0.5 * u * u)
    return out


def _cube(path, n=60, m=3000, dv=0.5, amplitude=0.1, sigma=0.004):
    """ln f of a receding sine of `amplitude` km/s, and the truth in km/s."""
    rng = np.random.default_rng(3)
    star_rng = np.random.default_rng(1000)
    grid = 1500.0 * np.exp(np.arange(m) * dv / C_KMS)
    t = np.sort(rng.uniform(0, 400.0, n))
    berv = 20.0 * np.sin(2 * np.pi * t / 365.25)
    v = amplitude * np.sin(2 * np.pi * t / 17.3)
    count = int(m * dv / 25.0)
    centres = star_rng.uniform(0, m, count)
    depths = -0.15 * star_rng.lognormal(-0.7, 0.7, count)
    width = 6.0 / 2.3548 / dv
    x = np.arange(m, dtype=float)
    # a line at rest at c is at c + p(v) - p(BERV) on the observer's grid
    shift = grids.pixel_shift(v, dv) - grids.pixel_shift(berv, dv)
    data = np.array([_lines(x, centres + s, depths, width) for s in shift])
    data += rng.normal(0.0, sigma, data.shape)
    os.makedirs(path)
    np.save(os.path.join(path, "grid.npy"), grid)
    np.save(os.path.join(path, "data.npy"), data.astype(np.float32))
    np.save(os.path.join(path, "sigma.npy"),
            np.full(data.shape, sigma, dtype=np.float32))
    meta = Table()
    meta["filename"] = ["e%03d.fits" % i for i in range(n)]
    meta["bjd"] = t + 2460000.0
    meta["berv"] = berv
    meta["airmass"] = np.ones(n)
    meta["snr_band"] = np.full(n, 100.0)
    meta["object"] = ["STAR"] * n
    meta["n_exposures"] = np.ones(n, dtype=int)
    meta["exposure"] = np.arange(n)
    meta["parity"] = np.zeros(n, dtype=int)
    meta.write(os.path.join(path, "meta.fits"))
    return v


def test_a_receding_star_has_a_positive_vrad_fit(tmp_path, monkeypatch):
    cube = str(tmp_path / "cube")
    truth = _cube(cube)
    outdir = str(tmp_path / "fit")
    os.makedirs(outdir)
    config = dict(domain=dict(bands=dict(H=[1490.0, 1780.0])),
                  twoframe=dict(n_star=0, n_earth=2, mean="star",
                                star_basis="spline", template=False,
                                velocity_term=True,
                                velocity_min_transmission=0.0,
                                template_berv_bin=3000, leakage=False,
                                dtype="float64", iters=6))
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    # the fit's figures and its FITS copy are output only
    for name in ("plot_coeffs", "plot_variance", "plot_components",
                 "correlate_and_plot", "write_components_fits",
                 "write_variance_table"):
        monkeypatch.setattr(twoframe, name, lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        twoframe.main(["--config", path, "--cube", cube, "--outdir", outdir])
    table = Table.read(os.path.join(outdir, "coefficients.csv"))
    fitted = np.asarray(table["vrad_fit"], dtype=float)
    slope = np.polyfit(truth * 1000.0, fitted, 1)[0]
    assert 0.9 < slope < 1.1, slope
    assert np.corrcoef(truth, fitted)[0, 1] > 0.95
    # and the shift in samples is the other way: the column moves the star
    #   towards the blue for a positive alpha
    assert np.corrcoef(truth, np.asarray(table["alpha"], float))[0, 1] < -0.95

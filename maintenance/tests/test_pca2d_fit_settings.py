"""Settings of the configuration that have to reach the fit: the weights'
(weights.min_pixel_snr, weights.telluric_ramp_zero, weights.ln_clip_low) and
twoframe.template_berv_min_entries. The fit records the weights' settings in
fit.npz, and every later reader of the cube weighs the samples as it did.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_fit_settings.py
"""
import contextlib
import io
import os

import numpy as np
import yaml
from astropy import constants
from astropy.table import Table

from lbl.pca2d import reconstruct, twoframe

C_KMS = constants.c.to("km/s").value


def _cube(path, n=24, m=2400, dv=0.5):
    """Noise and a static pattern; the right half at a pixel SNR of 50, the
    left half at 250."""
    rng = np.random.default_rng(8)
    grid = 1500.0 * np.exp(np.arange(m) * dv / C_KMS)
    sigma = np.full((n, m), 0.004)
    sigma[:, m // 2:] = 0.02
    pattern = np.sin(np.arange(m) / 7.0) * 0.02
    data = rng.normal(1.0, 0.3, (n, 1)) * pattern + rng.normal(0.0, sigma)
    os.makedirs(path)
    np.save(os.path.join(path, "grid.npy"), grid)
    np.save(os.path.join(path, "data.npy"), data.astype(np.float32))
    np.save(os.path.join(path, "sigma.npy"), sigma.astype(np.float32))
    meta = Table()
    meta["filename"] = ["e%03d.fits" % i for i in range(n)]
    meta["bjd"] = 2460000.0 + np.arange(n) * 3.1
    meta["berv"] = np.linspace(-10.0, 10.0, n)
    meta["airmass"] = np.ones(n)
    meta["snr_band"] = np.full(n, 100.0)
    meta["object"] = ["STAR"] * n
    meta["n_exposures"] = np.ones(n, dtype=int)
    meta["exposure"] = np.arange(n)
    meta["parity"] = np.zeros(n, dtype=int)
    meta.write(os.path.join(path, "meta.fits"))
    return m


def _fit(tmp_path, monkeypatch, cube, name, weights=None, noise=False):
    outdir = str(tmp_path / name)
    os.makedirs(outdir)
    config = dict(twoframe=dict(n_star=0, n_earth=1, mean="star",
                                star_basis="spline", template=False,
                                leakage=False, dtype="float64", iters=3,
                                noise_model=noise))
    if weights:
        config["weights"] = weights
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    for figure in ("plot_coeffs", "plot_variance", "plot_components",
                   "correlate_and_plot", "write_components_fits",
                   "write_variance_table"):
        monkeypatch.setattr(twoframe, figure, lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        twoframe.main(["--config", path, "--cube", cube, "--outdir", outdir])
    return np.load(os.path.join(outdir, "fit.npz"))


def test_the_pixel_snr_of_the_configuration_reaches_the_fit(tmp_path,
                                                            monkeypatch):
    cube = str(tmp_path / "cube")
    m = _cube(cube)
    plain = _fit(tmp_path, monkeypatch, cube, "plain")
    strict = _fit(tmp_path, monkeypatch, cube, "strict",
                  dict(min_pixel_snr=100.0), noise=True)
    right = slice(m // 2 + 10, m - twoframe.EDGE)
    assert np.any(plain["Q"][:, right])
    assert not np.any(strict["Q"][:, right])
    # recorded, and read back by the later readers of the cube
    settings = twoframe.fit_weight_settings(strict)
    assert {k: settings[k] for k in ("min_pixel_snr", "ln_clip_low",
                                     "ramp_zero")} == dict(
        min_pixel_snr=100.0, ln_clip_low=None, ramp_zero=0.5)
    # and the noise model the fit measured on its residuals, when asked
    assert settings["noise_model"].alpha >= 0.5
    assert "noise_model" not in twoframe.fit_weight_settings(plain)
    assert twoframe.fit_weight_settings(plain)["min_pixel_snr"] == 5.0
    masks = reconstruct.fit_weights_mask(
        cube, "common", 1.0, twoframe.fit_weight_settings(strict))
    assert not np.any(next(iter(masks.values()))[0, right])
    masks = reconstruct.fit_weights_mask(cube, "common", 1.0)
    assert np.all(next(iter(masks.values()))[0, right])


def test_a_fit_that_records_nothing_was_made_with_the_defaults(tmp_path):
    path = str(tmp_path / "fit.npz")
    np.savez(path, b=np.zeros((2, 1)))
    assert twoframe.fit_weight_settings(path) == twoframe.weight_settings()
    assert twoframe.fit_weight_settings(str(tmp_path / "none.npz")) == \
        twoframe.weight_settings()
    assert twoframe.weight_settings(dict(weights=dict(
        min_pixel_snr=None, ln_clip_low=-0.5, telluric_ramp_zero=0.8))) == \
        dict(min_pixel_snr=None, ln_clip_low=-0.5, ramp_zero=0.8)


def test_template_berv_min_entries_is_read_from_the_yaml(tmp_path):
    path = str(tmp_path / "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(twoframe=dict(template_berv_min_entries=50)), handle)
    args = twoframe.parse_args(["--cube", "c", "--outdir", "o", "--config", path])
    assert args.template_berv_min_entries == 50
    # the command line still wins, and without either it is the default
    args = twoframe.parse_args(["--cube", "c", "--outdir", "o", "--config", path,
                                "--template-berv-min-entries", "7"])
    assert args.template_berv_min_entries == 7
    args = twoframe.parse_args(["--cube", "c", "--outdir", "o"])
    assert args.template_berv_min_entries == 3

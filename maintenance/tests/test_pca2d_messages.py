"""Messages of PCA2D that have to say what happened: the overlap of two orders
of the same parity, and a t.fits without the instrument's wavelength
extension.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_messages.py
"""
import os

import numpy as np
from astropy.io import fits

from lbl.pca2d import config as spconfig
from lbl.pca2d import grids, tfits, wizard


def _config():
    path = os.path.join(os.path.dirname(spconfig.__file__), "nominal.yaml")
    cfg = spconfig.load_config(path, instrument="NIRPS_HE", data_source="CADC")
    cfg["registration"]["frame"] = "observer"
    return cfg


def test_an_overlap_of_two_orders_of_one_parity_is_counted(monkeypatch):
    """Orders 0 and 2 overlap; the earlier one has the higher blaze there and
    keeps the zone. The log gives the zone's size, not what is left of it."""
    said = []
    monkeypatch.setattr(tfits, "log", lambda msg, *a, **k: said.append(msg))
    cfg = _config()
    grid = grids.magic_grid(1500.0, 0.5, 1500.0, 1520.0)
    n_pix = 4000
    wave = np.vstack([np.linspace(1500.5, 1510.0, n_pix),
                      np.linspace(1505.0, 1515.0, n_pix),
                      np.linspace(1509.0, 1519.0, n_pix)])
    blaze = np.ones_like(wave)
    blaze[2] = np.linspace(0.05, 1.0, n_pix)    # order 2 starts on its edge
    payload = dict(flux=1000.0 * blaze, wave=wave, blaze=blaze,
                   recon=np.ones_like(wave), sky=None,
                   snr=np.full(3, 100.0),
                   meta=dict(berv=0.0, bjd=2460000.5, filename="x.fits"))
    monkeypatch.setattr(tfits, "order_quality",
                        lambda p, c: (np.ones(wave.shape, bool), [0, 1, 2]))
    tfits.resample_exposure(payload, grid, cfg)
    overlap = [m for m in said if "same parity overlap" in m]
    assert overlap, said
    samples = int(overlap[0].split(" over ")[1].split()[0])
    # 1509 to 1510 nm, less the eroded edges: about 400 samples at 0.5 km/s
    assert 300 < samples < 450, overlap[0]


def test_a_missing_wavelength_extension_is_named(tmp_path):
    folder = tmp_path / "STAR"
    folder.mkdir()
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((2, 5)),
                                                   name="FluxA")]).writeto(
        folder / "at.fits")
    cfg = _config()
    cfg["input"]["directory"] = str(tmp_path)
    cfg["input"]["object"] = "STAR"
    info = wizard.probe_files(cfg)
    assert info["no_wave"] == cfg["input"]["extensions"]["wave"]


def test_the_parity_agreement_compares_noise_not_continua():
    """With the high pass deferred, the two parities hold ln f, each order
    at its own level: their difference is noise only once each order is
    high-passed."""
    from lbl.pca2d import cube
    rng = np.random.default_rng(3)
    m = 3000
    x = np.arange(m)
    lines = -0.2 * np.exp(-0.5 * ((x[:, None] - rng.uniform(0, m, 60)[None, :])
                                  / 3.0) ** 2).sum(axis=1)
    values = np.vstack([lines + 0.4 + 1e-4 * x, lines - 0.3 - 2e-4 * x])
    values += rng.normal(0.0, 0.01, values.shape)
    sigma = np.full(values.shape, 0.01)
    good = np.ones(values.shape, dtype=bool)
    segments = [(0, 0, m), (1, 0, m)]
    hp = dict(window=101, polyorder=2)
    deferred = cube._parity_agreement(values, sigma, good, segments, hp)
    assert 0.7 < deferred < 1.2, deferred
    # without the segments (already high-passed values), as it was
    flat = values - np.vstack([0.4 + 1e-4 * x, -0.3 - 2e-4 * x])
    assert 0.8 < cube._parity_agreement(flat, sigma, good) < 1.2

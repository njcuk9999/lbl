"""What the report's pages read: a NaN velocity does not flatten the
periodogram, and the telluric page reads the spectra by the instrument's class
or says why it cannot.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_report_inputs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from lbl.report import figures, series, stats  # noqa: E402
from lbl.report.periodogram import frequency_grid, gls  # noqa: E402


def _series(n=80, seed=2):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 300.0, n))
    value = 5.0 * np.sin(2 * np.pi * time / 13.7) + rng.normal(0.0, 1.0, n)
    return time, value, np.ones(n)


def test_one_nan_velocity_does_not_flatten_the_periodogram():
    time, value, err = _series()
    freq = frequency_grid(time)
    clean = gls(time, value, err, freq)
    value[7] = np.nan
    holed = gls(time, value, err, freq)
    assert holed.max() > 0.5
    # the same as without that epoch
    keep = np.arange(time.size) != 7
    assert np.allclose(holed, gls(time[keep], value[keep], err[keep], freq))
    assert abs(1.0 / freq[np.argmax(holed)] - 1.0 / freq[np.argmax(clean)]) < 0.1
    err[3] = np.nan
    err[4] = 0.0
    assert gls(time, value, err, freq).max() > 0.5


def test_the_summary_leaves_out_an_epoch_without_velocity():
    time, value, err = _series()
    value[5] = np.nan
    # as series.read_rdb gives it: a dict of columns
    table = {series.TIME: time, series.VELOCITY: value, series.ERROR: err}
    # the nights cut at local noon at La Silla
    rows = {label: text for label, text, _ in stats.summarise(table, -70.73)}
    assert rows["Epochs"] == "%d" % (time.size - 1)
    assert rows["rms"] != "nan"


def _s2d(path, flux, wave):
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(flux), fits.ImageHDU(),
                  fits.ImageHDU(), fits.ImageHDU(wave * 10.0)]).writeto(path)


def _tree(tmp_path):
    science, cleaned = tmp_path / "science", tmp_path / "tc"
    science.mkdir()
    cleaned.mkdir()
    names = ["X.2024-01-%02dT03:00:00.000_S2D_A.fits" % d for d in (1, 2, 3)]
    table = {series.FILENAME: np.array(["%s_STAR_STAR_lbl.fits" % n[:-5]
                                        for n in names]),
             series.WATER: np.array([0.5, 1.0, 2.0])}
    return science, cleaned, names, table


def test_the_telluric_page_reads_by_the_instrument_s_class(tmp_path,
                                                           monkeypatch):
    from lbl.pca2d import readers
    science, cleaned, names, table = _tree(tmp_path)
    for name in names:
        # one image, the layout the page alone could not read
        fits.PrimaryHDU(np.ones((2, 50))).writeto(science / name)
        fits.PrimaryHDU(np.ones((2, 50))).writeto(cleaned / name)
    wave = np.vstack([np.linspace(645.0, 650.0, 400),
                      np.linspace(649.0, 654.0, 400)])
    flux = 1.0 - 0.3 * np.exp(-0.5 * ((wave - 647.76) / 0.01) ** 2)
    seen = []

    def read(path, instrument, source):
        seen.append((os.path.basename(path), instrument, source))
        return dict(flux=flux if "science" in str(path) else np.ones_like(flux),
                    wave=wave)
    monkeypatch.setattr(readers, "read", read)
    fig = figures.page_telluric(plt, table, "t", str(science), str(cleaned),
                                instrument="HARPS", data_source="ORIG")
    assert fig is not None
    assert {s[1:] for s in seen} == {("HARPS", "ORIG")}
    plt.close(fig)


def test_without_the_instrument_a_file_it_cannot_read_says_why(tmp_path):
    science, cleaned, names, table = _tree(tmp_path)
    for name in names:
        fits.PrimaryHDU(np.ones((2, 50))).writeto(science / name)
        fits.PrimaryHDU(np.ones((2, 50))).writeto(cleaned / name)
    fig = figures.page_telluric(plt, table, "t", str(science), str(cleaned))
    assert fig is not None
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("not an ESO S2D file" in t for t in texts)
    plt.close(fig)


def test_an_eso_s2d_file_is_read_as_before(tmp_path):
    science, cleaned, names, table = _tree(tmp_path)
    wave = np.vstack([np.linspace(645.0, 650.0, 400),
                      np.linspace(649.0, 654.0, 400)])
    flux = 1.0 - 0.3 * np.exp(-0.5 * ((wave - 647.76) / 0.01) ** 2)
    for name in names:
        _s2d(science / name, flux, wave)
        _s2d(cleaned / name, np.ones_like(flux), wave)
    before, got, after = figures._telluric_pair(
        str(science / names[0]), str(cleaned / names[0]))
    assert np.allclose(got, wave) and np.allclose(before, flux)
    fig = figures.page_telluric(plt, table, "t", str(science), str(cleaned))
    assert fig is not None and not any(
        "cannot read" in t.get_text() for ax in fig.axes for t in ax.texts)
    plt.close(fig)

"""lbl.science.activity_lines on synthetic spectra: a line of known
equivalent width added to the spectrum comes back as EW_DIFF.

    PYTHONPATH=<this checkout> python -m pytest tests/test_activity_lines.py
"""
import numpy as np
import pytest

from lbl.core import math as mp
from lbl.science import activity_lines

# He I 10830, as in the catalog
LAM0 = 1083.3273


def _gauss(wave, centre, fwhm):
    return np.exp(-4 * np.log(2) * ((wave - centre) / fwhm) ** 2)


def _orders(velocity=0.0, ew_added=0.0, fwhm_added=0.01, noise=0.0):
    """
    Two orders around He I 10830 (the line near the centre of the second),
    a template with a photospheric line on the window, a sloped blaze, and
    the spectrum = the template x blaze x 0.8, plus an absorption of
    equivalent width ew_added [mA], all shifted by velocity
    """
    npix = 4000
    wave_t = np.array([np.linspace(1079.5, 1084.5, npix),
                       np.linspace(1081.0, 1086.0, npix)])
    # the observer's wavelengths of these template-frame wavelengths
    wavegrid = mp.doppler_shift(wave_t, -velocity)
    template = 1 - 0.3 * _gauss(wave_t, LAM0 + 0.004, 0.02)
    blaze = 1 - 0.5 * ((np.arange(npix) - npix / 2) / npix) ** 2
    blaze = np.array([blaze, blaze]) * 1e4
    depth = ew_added / activity_lines.NM_TO_MA / (
            fwhm_added * np.sqrt(np.pi / (4 * np.log(2))))
    spectrum = 0.8 * blaze * (template - depth * _gauss(wave_t, LAM0,
                                                        fwhm_added))
    rng = np.random.default_rng(1)
    rms = np.full_like(spectrum, noise)
    spectrum = spectrum + rng.normal(0, 1, spectrum.shape) * noise
    return wavegrid, spectrum, template, blaze, rms


@pytest.mark.parametrize('ew_added', [25.0, -40.0])
def test_an_added_line_comes_back(ew_added):
    wavegrid, spectrum, template, blaze, rms = _orders(
        velocity=15000.0, ew_added=ew_added, noise=1.0)
    table = activity_lines.measure(wavegrid, spectrum, template, blaze, rms,
                                   velocity=15000.0, berv=0.0)
    row = table[list(table['NAME']).index('He 1.08333')]
    # the 0.025 nm box holds 99.7% of a 0.01 nm FWHM line
    assert row['dEW'] == pytest.approx(0.9967 * ew_added, abs=0.3)
    assert row['sdEW'] < 0.3
    assert row['FRACVALID'] == 1


def test_no_change_gives_zero_and_lines_outside_are_flagged():
    wavegrid, spectrum, template, blaze, rms = _orders(noise=1.0)
    table = activity_lines.measure(wavegrid, spectrum, template, blaze, rms,
                                   velocity=0.0, berv=0.0)
    row = table[list(table['NAME']).index('He 1.08333')]
    assert abs(row['dEW']) < 4 * row['sdEW']
    # H alpha is not in these orders
    halpha = table[list(table['NAME']).index('H 0.65646')]
    assert np.isnan(halpha['dEW']) and np.isnan(halpha['FRACVALID'])


def test_a_gaussian_window_weights_the_line():
    wavegrid, spectrum, template, blaze, rms = _orders(ew_added=30.0,
                                                       fwhm_added=0.02)
    line = dict(NAME='test', WAVE=LAM0, WIDTH=0.03, FUNCTION='gaussian')
    out = activity_lines._measure_line(line, wavegrid[1], spectrum[1],
                                       template[1], blaze[1],
                                       np.ones_like(spectrum[1]))
    # two gaussians: sigma_w / sqrt(sigma_w^2 + sigma_line^2)
    expected = 30.0 * 0.03 / np.sqrt(0.03 ** 2 + 0.02 ** 2)
    assert out['dEW'] == pytest.approx(expected, rel=0.01)


def test_the_rdb_gets_one_column_per_measure_and_line(tmp_path):
    from astropy.io import fits
    from astropy.table import Table

    # two lblrv files, the second one from before the activity lines
    wavegrid, spectrum, template, blaze, rms = _orders(ew_added=25.0,
                                                       noise=1.0)
    act = activity_lines.measure(wavegrid, spectrum, template, blaze, rms,
                                 velocity=0.0, berv=0.0)
    files = [str(tmp_path / 'a_lbl.fits'), str(tmp_path / 'b_lbl.fits')]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(dict(x=[1]))),
                  fits.BinTableHDU(act, name='ACTIVITY')]).writeto(files[0])
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU(Table(dict(x=[1])))]).writeto(files[1])
    rdb = Table(dict(FILENAME=['b_lbl.fits', 'a_lbl.fits'], vrad=[1., 2.]))
    # the two He I 10830 lines are in these orders
    assert activity_lines.add_to_rdb(rdb, files) == 2
    assert rdb.colnames == ['FILENAME', 'vrad'] + [
        '{0}_{1}'.format(col, key) for key in ['He_1.08321', 'He_1.08333']
        for col in ['dEW', 'sdEW', 'FRACVALID', 'CHI2']]
    assert np.isnan(rdb['dEW_He_1.08333'][0])
    assert rdb['dEW_He_1.08333'][1] == pytest.approx(0.9967 * 25, abs=0.3)


def test_the_report_page_finds_and_draws_the_lines():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lbl.report import figures

    rng = np.random.default_rng(2)
    nspec = 30
    table = {'rjd': np.linspace(60000, 60300, nspec)}
    for key in ['He_1.08333', 'H_1.28216', 'H_2.16612']:
        table['dEW_' + key] = rng.normal(0, 2, nspec)
        table['sdEW_' + key] = np.full(nspec, 2.0)
    # a flare, and a line never measured
    table['dEW_He_1.08333'][5] = -40
    table['dEW_H_2.16612'][:] = np.nan
    labels = {'He 1.08333': 'He I 10830', 'H 1.28216': 'Pa beta'}
    lines = figures.activity_lines(table, labels)
    assert [line[1:] for line in lines] == [('He 1.08333', 'He I 10830'),
                                            ('H 1.28216', 'Pa beta')]
    good, mean, chi2 = figures.activity_stats(table, 'He_1.08333')
    assert good.sum() == nspec and chi2 > 10
    fig = figures.page_activity(plt, table, lines, 'test')
    assert fig is not None
    plt.close(fig)

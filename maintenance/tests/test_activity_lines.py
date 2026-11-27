"""lbl.science.activity_lines on synthetic spectra: the formalism of Schofer
et al. (2019): the template's equivalent width in the catalog's window
against its pseudo-continuum (EWT), a line added to the spectrum coming
back as dEW, and EW = EWT + dEW.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_activity_lines.py
"""
import numpy as np
import pytest
from scipy.special import erf

from lbl.core import math as mp
from lbl.science import activity_lines

# He I 10830, as in the catalog (its NIST centre)
LAM0 = 1083.3273
NAME = 'He 1.08333'


def _boxcar_share(lo, hi, centre, fwhm):
    """The share of a gaussian line of this FWHM and centre inside [lo, hi]"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return 0.5 * (erf((hi - centre) / sigma / np.sqrt(2)) -
                  erf((lo - centre) / sigma / np.sqrt(2)))


def _gauss(wave, centre, fwhm):
    return np.exp(-4 * np.log(2) * ((wave - centre) / fwhm) ** 2)


#: the template's own line under He I 10830: depth, centre, FWHM [nm]
T_DEPTH, T_CENTRE, T_FWHM = 0.3, LAM0 + 0.004, 0.02


def _template(wave, slope=0.0, clutter=False):
    """A template of continuum 1 + slope (lam - LAM0), a line under He I,
    and with clutter, narrow lines every 0.07 nm (a pseudo-continuum's
    90th percentile is above them)"""
    flux = 1 - T_DEPTH * _gauss(wave, T_CENTRE, T_FWHM)
    if clutter:
        for centre in np.arange(1075.0, 1095.0, 0.07):
            if abs(centre - LAM0) > 0.2:
                flux = flux - 0.4 * _gauss(wave, centre, 0.003)
    return flux * (1 + slope * (wave - LAM0))


def _orders(velocity=0.0, ew_added=0.0, fwhm_added=0.01, noise=0.0,
            slope=0.0, clutter=False):
    """
    Two orders around He I 10830, the second holding its window and both
    pseudo-continuum ranges (1082.0 to 1087.57 nm), a sloped blaze, and the
    spectrum = the template x blaze x 0.8, plus an absorption of equivalent
    width ew_added [mA, in units of the continuum], all shifted by velocity
    """
    npix = 8000
    wave_t = np.array([np.linspace(1076.0, 1084.0, npix),
                       np.linspace(1079.0, 1091.0, npix)])
    # the observer's wavelengths of these template-frame wavelengths
    wavegrid = mp.doppler_shift(wave_t, -velocity)
    template = _template(wave_t, slope, clutter)
    blaze = 1 - 0.5 * ((np.arange(npix) - npix / 2) / npix) ** 2
    blaze = np.array([blaze, blaze]) * 1e4
    depth = ew_added / activity_lines.NM_TO_MA / (
            fwhm_added * np.sqrt(np.pi / (4 * np.log(2))))
    continuum = 1 + slope * (wave_t - LAM0)
    spectrum = 0.8 * blaze * (template - continuum * depth *
                              _gauss(wave_t, LAM0, fwhm_added))
    rng = np.random.default_rng(1)
    rms = np.full_like(spectrum, noise)
    spectrum = spectrum + rng.normal(0, 1, spectrum.shape) * noise
    return wavegrid, spectrum, template, blaze, rms


def _template_1d(slope=0.0, clutter=False):
    """the template itself, merged, finely sampled"""
    wave = np.linspace(1070.0, 1100.0, 120001)
    return wave, _template(wave, slope, clutter)


def _measure(wavegrid, spectrum, template, blaze, rms, velocity=0.0,
             slope=0.0, clutter=False):
    return activity_lines.measure(wavegrid, spectrum, template, blaze, rms,
                                  velocity=velocity, berv=0.0,
                                  template_1d=_template_1d(slope, clutter))


def _row(table, name=NAME):
    return table[list(table['NAME']).index(name)]


def _line(name=NAME):
    return next(line for line in activity_lines.load_lines()
                if line['NAME'] == name)


def _expected_ewt(name=NAME):
    """The template's equivalent width in the window, integrated finely"""
    lo, hi = activity_lines.window(_line(name))
    wave = np.linspace(lo, hi, 20001)
    depth = T_DEPTH * _gauss(wave, T_CENTRE, T_FWHM)
    return np.trapezoid(depth, wave) * activity_lines.NM_TO_MA


@pytest.mark.parametrize('ew_added', [25.0, -40.0])
def test_an_added_line_comes_back(ew_added):
    wavegrid, spectrum, template, blaze, rms = _orders(
        velocity=15000.0, ew_added=ew_added, noise=1.0)
    table = _measure(wavegrid, spectrum, template, blaze, rms,
                     velocity=15000.0)
    row = _row(table)
    lo, hi = activity_lines.window(_line())
    share = _boxcar_share(lo, hi, LAM0, 0.01)
    assert row['dEW'] == pytest.approx(share * ew_added, abs=0.3)
    assert row['sdEW'] < 0.3 and row['sEW'] == row['sdEW']
    assert row['FRACVALID'] == 1
    # the template's own, and the spectrum's
    assert row['EWT'] == pytest.approx(_expected_ewt(), rel=2e-3)
    assert row['EW'] == pytest.approx(row['EWT'] + row['dEW'])


def test_the_pseudo_continuum_is_the_90th_percentile_of_its_ranges():
    # a sloped continuum and lines crowding the ranges: the template's
    #   equivalent width is that of its own line all the same
    wavegrid, spectrum, template, blaze, rms = _orders(
        ew_added=20.0, noise=0.5, slope=0.01, clutter=True)
    row = _row(_measure(wavegrid, spectrum, template, blaze, rms,
                        slope=0.01, clutter=True))
    assert row['EWT'] == pytest.approx(_expected_ewt(), rel=0.02)
    lo, hi = activity_lines.window(_line())
    assert row['dEW'] == pytest.approx(
        _boxcar_share(lo, hi, LAM0, 0.01) * 20.0, abs=0.3)
    # the other continuum (95th percentile within 2.5 nm, on the slope)
    #   is somewhere else: a systematic error, positive
    assert row['EWTSYS'] > 0


def test_no_change_gives_zero_and_lines_outside_are_flagged():
    wavegrid, spectrum, template, blaze, rms = _orders(noise=1.0)
    table = _measure(wavegrid, spectrum, template, blaze, rms)
    row = _row(table)
    assert abs(row['dEW']) < 4 * row['sdEW']
    # H alpha is not in these orders
    halpha = _row(table, 'H 0.65646')
    assert np.isnan(halpha['EW']) and np.isnan(halpha['FRACVALID'])


def test_an_order_holding_one_range_measures_with_its_level():
    # the first order holds the window and the blue range only: the
    #   template holds both, the spectrum's level comes from the one
    wavegrid, spectrum, template, blaze, rms = _orders(ew_added=25.0)
    terms = activity_lines.template_terms(_line(), *_template_1d())
    one = activity_lines._measure_line(_line(), terms, wavegrid[0],
                                       spectrum[0], template[0], blaze[0],
                                       rms[0] + 1)
    both = activity_lines._measure_line(_line(), terms, wavegrid[1],
                                        spectrum[1], template[1], blaze[1],
                                        rms[1] + 1)
    assert one['EWT'] == both['EWT']
    assert one['dEW'] == pytest.approx(both['dEW'], abs=0.1)
    # an order with the window but neither range: the spectrum's level
    #   from its valid pixels nearest the ranges, on each side
    def cut(a, b):
        part = slice(np.searchsorted(wavegrid[1], a),
                     np.searchsorted(wavegrid[1], b))
        return activity_lines._measure_line(
            _line(), terms, wavegrid[1][part], spectrum[1][part],
            template[1][part], blaze[1][part], rms[1][part] + 1)
    narrow = cut(1082.5, 1083.9)
    assert narrow['dEW'] == pytest.approx(both['dEW'], abs=0.1)
    # nothing beyond the window's wings: not measured there
    assert cut(1083.29, 1083.37) is None


def test_the_catalog_s_schofer_lines_are_their_table_1():
    # Schofer et al. 2019, Table 1: lambda0 +- dlambda, ranges lambda_l/r
    #   +- dlambda_PC, in A, vacuum: dlambda is HALF the window
    table = {'He 1.08333': (10833.31, 0.25, 10821.0, 10874.7, 1.00),
             'H 1.28216': (12821.57, 0.25, 12817.0, 12826.0, 1.00),
             'H 0.65646': (6564.60, 5.00, 6552.50, 6578.50, 2.50),
             'Ca 0.85004': (8500.35, 2.80, 8494.00, 8505.00, 1.00),
             'Ca 0.85444': (8544.44, 2.80, 8538.50, 8555.00, 1.00),
             'Ca 0.86645': (8664.52, 2.40, 8658.00, 8670.00, 1.00),
             'He 0.58773': (5877.25, 2.50, 5872.75, 5882.75, 1.25),
             'Na 0.58916': (5891.58, 2.99, 5882.75, 5908.25, 1.25),
             'Na 0.58976': (5897.56, 2.99, 5882.75, 5908.25, 1.25)}
    for name, (lam0, dlam, lam_l, lam_r, dpc) in table.items():
        line = _line(name)
        assert activity_lines.window(line) == pytest.approx(
            ((lam0 - dlam) / 10, (lam0 + dlam) / 10)), name
        ranges = [edge for r in activity_lines.continuum(line) for edge in r]
        assert ranges == pytest.approx(
            [(lam_l - dpc) / 10, (lam_l + dpc) / 10,
             (lam_r - dpc) / 10, (lam_r + dpc) / 10]), name


def test_the_catalog_s_paschen_lines_are_fuhrmeister_s_table_1():
    # Fuhrmeister et al. 2023, Table 1: centre, the FULL width of the
    #   window, the two reference intervals, in A, vacuum
    table = {'H 1.00521': (10052.60, 1.5, (10045.0, 10047.0),
                           (10076.0, 10078.0)),
             'H 1.09411': (10941.17, 1.5, (10902.0, 10904.0),
                           (10964.7, 10966.7))}
    for name, (lam0, full, left, right) in table.items():
        line = _line(name)
        assert activity_lines.window(line) == pytest.approx(
            ((lam0 - full / 2) / 10, (lam0 + full / 2) / 10)), name
        ranges = [edge for r in activity_lines.continuum(line) for edge in r]
        assert ranges == pytest.approx([v / 10 for v in left + right]), name


def test_every_line_has_a_window_between_its_two_ranges():
    lines = activity_lines.load_lines()
    assert len(lines) == len({line['NAME'] for line in lines})
    for line in lines:
        (c1lo, c1hi), (c2lo, c2hi) = activity_lines.continuum(line)
        lo, hi = activity_lines.window(line)
        assert c1lo < c1hi <= lo < hi <= c2lo < c2hi, line['NAME']
        assert line['REF'], line['NAME']


def test_a_file_of_another_catalog_version_is_measured_again(tmp_path):
    from astropy.io import fits
    from astropy.table import Table

    wavegrid, spectrum, template, blaze, rms = _orders(noise=1.0)
    act = _measure(wavegrid, spectrum, template, blaze, rms)
    header = activity_lines.extension_header('x.fits')
    good = str(tmp_path / 'good_lbl.fits')
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(dict(x=[1]))),
                  fits.BinTableHDU(act, header=header, name='ACTIVITY')]
                 ).writeto(good)
    assert activity_lines.has_activity(good)
    header['ACTVERS'] = activity_lines.catalog_version() - 1
    old = str(tmp_path / 'old_lbl.fits')
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(dict(x=[1]))),
                  fits.BinTableHDU(act, header=header, name='ACTIVITY')]
                 ).writeto(old)
    assert not activity_lines.has_activity(old)


def test_the_rdb_gets_one_column_per_measure_and_line(tmp_path):
    from astropy.io import fits
    from astropy.table import Table

    # two lblrv files, the second one without an ACTIVITY extension (as
    #   older versions of LBL wrote them)
    wavegrid, spectrum, template, blaze, rms = _orders(ew_added=25.0,
                                                       noise=1.0)
    act = _measure(wavegrid, spectrum, template, blaze, rms)
    measured = [name for name, ew in zip(act['NAME'], act['EW'])
                if np.isfinite(ew)]
    assert NAME in measured
    files = [str(tmp_path / 'a_lbl.fits'), str(tmp_path / 'b_lbl.fits')]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(dict(x=[1]))),
                  fits.BinTableHDU(act, name='ACTIVITY')]).writeto(files[0])
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU(Table(dict(x=[1])))]).writeto(files[1])
    rdb = Table(dict(FILENAME=['b_lbl.fits', 'a_lbl.fits'], vrad=[1., 2.]))
    assert activity_lines.add_to_rdb(rdb, files) == len(measured)
    assert rdb.colnames == ['FILENAME', 'vrad'] + [
        '{0}_{1}'.format(col, name.replace(' ', '_')) for name in measured
        for col in activity_lines.MEASURED]
    key = NAME.replace(' ', '_')
    assert np.isnan(rdb['EW_' + key][0])
    assert rdb['EW_' + key][1] == pytest.approx(
        rdb['EWT_' + key][1] + rdb['dEW_' + key][1])


def test_the_report_page_finds_and_draws_the_lines():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lbl.report import figures

    rng = np.random.default_rng(2)
    nspec = 30
    table = {'rjd': np.linspace(60000, 60300, nspec)}
    for key in ['He_1.08333', 'H_1.28216', 'H_2.16612']:
        table['EWT_' + key] = np.full(nspec, 80.0)
        table['EWTSYS_' + key] = np.full(nspec, 3.0)
        table['dEW_' + key] = rng.normal(0, 2, nspec)
        table['sdEW_' + key] = np.full(nspec, 2.0)
        table['EW_' + key] = table['EWT_' + key] + table['dEW_' + key]
        table['sEW_' + key] = table['sdEW_' + key]
    # a flare, and a line never measured
    table['EW_He_1.08333'][5] = 40
    table['EW_H_2.16612'][:] = np.nan
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
    # an rdb of before: its dEW
    old = {k: v for k, v in table.items() if not k.startswith(('EW', 'sEW'))}
    assert [line[1] for line in figures.activity_lines(old)] == [
        'He 1.08333', 'H 1.28216']

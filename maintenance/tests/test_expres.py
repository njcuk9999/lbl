"""The EXPRES class on files shaped like the pipeline's: the flux in electrons
over the pixels Excalibur calibrates, a barycentric wave map LBL can spline,
the time and the BERV from the other extensions, the extraction-only files
left out."""
import numpy as np
import pytest
from astropy.io import fits

from lbl.core import math as mp
from lbl.pca2d import readers

NPIX = 7920
NORD = 3
REDSHIFT = 7.3e-5
BARYMJD = 58710.4625
MIDPOINT = 2458710.9595


def _write(path, calibrated=True):
    pix = np.arange(NPIX, dtype=float)
    wave = np.array([5000.0 + 80 * order + pix * 0.01 for order in range(NORD)])
    bary = wave * (1 + REDSHIFT)
    excalibur = wave + 0.001
    bary_excalibur = excalibur * (1 + REDSHIFT)
    # the classical solution misses the edges, Excalibur misses a whole order
    #   and more of the edges of the others
    bary[:, :50] = np.nan
    wave[:, :50] = np.nan
    for column in (excalibur, bary_excalibur):
        column[:, :200] = np.nan
        column[:, -200:] = np.nan
        column[2] = np.nan
    columns = [('spectrum', np.full((NORD, NPIX), 0.5)),
               ('blaze', np.full((NORD, NPIX), 1000.0))]
    if calibrated:
        columns += [('wavelength', wave), ('bary_wavelength', bary),
                    ('excalibur', excalibur),
                    ('bary_excalibur', bary_excalibur)]
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name=name, format='%dD' % NPIX, array=values)
         for name, values in columns], name='optimal')
    table.header['BARYMJD'] = BARYMJD
    hdus = [fits.PrimaryHDU(), table]
    if calibrated:
        meter = fits.BinTableHDU.from_columns(
            [fits.Column(name='midpoints', format='D', array=np.zeros(2))],
            name='EXPOSURE METER + BARY_CORR')
        meter.header['wtd_mdpt'] = MIDPOINT
        meter.header['wtd_mdpt_bc'] = REDSHIFT
        hdus.append(meter)
    fits.HDUList(hdus).writeto(path)
    return str(path)


@pytest.fixture
def inst():
    return readers.instrument('EXPRES', 'ORIG')


def test_flux_in_electrons_only_where_excalibur_calibrates(inst, tmp_path):
    flux, _ = inst.load_science_file(_write(tmp_path / 'a.fits'))
    assert flux[0, 1000] == 500.0
    assert np.isnan(flux[0, 100]) and np.isnan(flux[0, -100])
    assert np.isnan(flux[2]).all()


def test_wave_map_finite_increasing_and_excalibur_where_it_exists(inst,
                                                                  tmp_path):
    wave = inst.get_wave_solution(_write(tmp_path / 'a.fits'))
    assert np.isfinite(wave).all()
    assert (np.diff(wave, axis=1) > 0).all()
    expected = (5000.0 + 1000 * 0.01 + 0.001) * (1 + REDSHIFT) / 10.0
    assert wave[0, 1000] == pytest.approx(expected, rel=1e-12)


def test_times_and_berv_from_the_other_extensions(inst, tmp_path):
    header = inst.load_header(_write(tmp_path / 'a.fits'))
    assert header['MJDMID'] == pytest.approx(MIDPOINT - 2400000.5)
    assert header['BJD'] == pytest.approx(BARYMJD + 2400000.5)
    # relativistic: the velocity whose Doppler ratio is 1 + z
    ratio = 1 + REDSHIFT
    expected = mp.speed_of_light_ms * (ratio ** 2 - 1) / (ratio ** 2 + 1)
    # the grid is barycentric: nothing to apply, the tellurics still move
    assert inst.get_berv(header) == 0.0
    inst.params.set('OBJECT_SCIENCE', 'HD10700')
    assert inst.get_berv_transmission(header) == pytest.approx(expected)


def test_blaze_read_from_the_science_file(inst, tmp_path):
    path = _write(tmp_path / 'a.fits')
    flux, header = inst.load_science_file(path)
    blaze, is_flat = inst.load_blaze_from_science(path, flux, header, '',
                                                  normalize=False)
    assert not is_flat
    assert blaze[0, 1000] == 1000.0


def test_extraction_only_files_left_out(inst, tmp_path):
    folder = tmp_path / 'STAR'
    folder.mkdir()
    good = _write(folder / 'good.fits')
    _write(folder / 'bare.fits', calibrated=False)
    inst.params.set('OBJECT_SCIENCE', 'STAR')
    inst.params.set('INPUT_FILE', '*.fits')
    assert list(inst.science_files(str(tmp_path))) == [good]


def test_the_window_reads_an_expres_file(tmp_path):
    """The window's scan (lbl.gui.targets.read_file) names EXPRES from its
    INSTRMNT and takes the date and the BERV as LBL's class does: the
    flux-weighted midpoint and its barycentric redshift."""
    from lbl.core.math import speed_of_light
    from lbl.gui import targets
    from lbl.instruments import expres
    primary = fits.PrimaryHDU()
    primary.header['INSTRMNT'] = 'EXPRES'
    primary.header['OBJECT'] = '10700'
    primary.header['AEXPTIME'] = 80.0
    bary = fits.BinTableHDU.from_columns(
        [fits.Column(name='midpoints', format='D', array=np.zeros(2))],
        name=expres.BARY_EXTENSION)
    bary.header[expres.BARY_MIDPOINT_KEY] = 2458710.959567354
    bary.header[expres.BARY_REDSHIFT_KEY] = 7.313910006701e-05
    path = str(tmp_path / '10700_190815.1114.fits')
    fits.HDUList([primary, bary]).writeto(path)
    record = targets.read_file(path)
    assert record['instrument'] == 'EXPRES'
    assert record['object'] == '10700'
    assert record['exptime'] == 80.0
    assert abs(record['mjd'] - 58710.459567354) < 1e-6
    ratio = 1 + 7.313910006701e-05
    berv = speed_of_light * (ratio ** 2 - 1) / (ratio ** 2 + 1)
    assert abs(record['berv'] - berv) < 1e-4
    assert abs(record['berv'] - 21.9257) < 1e-3
    assert targets.night_of(record) is not None

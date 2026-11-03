"""A corrected copy of a spectrum whose flux is a column of a table (EXPRES).

PCA2D writes its corrected spectra in the format of the originals, a copy
whose flux alone has changed, and LBL measures them with the same class. An
EXPRES file holds its flux as the `spectrum` column of the `optimal`
extraction, multiplied by the `blaze` column on reading: there is no image
extension to replace. These tests write a corrected copy through
reconstruct.correct_file on files shaped like the pipeline's, read it back
with LBL's EXPRES class and check that only the flux changed, by the
correction, at the right wavelengths: the EXPRES grid is barycentric, the
correction lives in the observer's frame.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_table_flux.py
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lbl.core import math as mp
from lbl.pca2d import grids
from lbl.pca2d import readers
from lbl.pca2d import reconstruct
from lbl.pca2d.config import instrument_layer

# the EXPRES files of test_expres.py, the same shape as the pipeline's
_SPEC = importlib.util.spec_from_file_location(
    "_expres_files", os.path.join(os.path.dirname(__file__), "test_expres.py"))
_FILES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FILES)

#: the telluric line of the correction, in the OBSERVER's frame [nm], its
#: width [nm] and its depth in ln flux
LINE_NM, LINE_WIDTH, LINE_DEPTH = 504.0, 0.01, 0.3


def _expres_file(tmp_path, name='10700_190815.1114.fits'):
    """An EXPRES file with a flux that varies, so that a column copied in
    the wrong place shows"""
    folder = tmp_path / 'HD10700'
    folder.mkdir(exist_ok=True)
    path = _FILES._write(folder / name)
    with fits.open(path, mode='update') as hdulist:
        table = hdulist['optimal'].data
        pix = np.arange(_FILES.NPIX)
        table['spectrum'][:] = 0.5 + 0.1 * np.sin(pix / 37.0)[None, :]
        table['blaze'][:] = 1000.0 + 300 * np.cos(pix / 900.0)[None, :]
    return path


def _config():
    """The configuration a wrap gives pca2d for EXPRES: its instrument
    layer, from LBL's class, with no models folder (no TAPAS download)"""
    layer = instrument_layer('EXPRES', 'ORIG')
    return dict(lbl=layer['lbl'], input=layer['input'], output=dict(),
                registration=dict())


def _model(n_earth=1):
    """A fit with one observer component, a telluric line at LINE_NM in the
    observer's frame, and no star component"""
    grid = grids.magic_grid(499.0, 0.5, 499.0, 526.0)
    line = -LINE_DEPTH * np.exp(-0.5 * ((grid - LINE_NM) / LINE_WIDTH) ** 2)
    zero = np.zeros(grid.size)
    return dict(grid=grid, template=zero, P=np.zeros((0, grid.size)),
                Q=np.array([line] * n_earth), dv=0.5, n_star=0,
                n_earth=n_earth, content='test', means=dict(even=zero,
                                                             odd=zero),
                support=np.ones(grid.size, dtype=bool), max_mad=0.0)


def _row(berv):
    return dict(b1=1.0, berv=berv, rejected=False)


def _berv_kms():
    ratio = 1 + _FILES.REDSHIFT
    return mp.speed_of_light_ms * (ratio ** 2 - 1) / (ratio ** 2 + 1) / 1000


def _read(inst, path):
    flux, header = inst.load_science_file(path)
    wave = inst.get_wave_solution(path, flux, header)
    blaze, _ = inst.load_blaze_from_science(path, flux, header, '',
                                            normalize=False)
    return flux, header, wave, blaze


@pytest.fixture
def inst():
    inst = readers.instrument('EXPRES', 'ORIG')
    inst.params.set('OBJECT_SCIENCE', 'HD10700')
    return inst


def test_the_class_names_the_column_and_it_is_found(inst, tmp_path):
    path = _expres_file(tmp_path)
    flux, _ = inst.load_science_file(path)
    assert readers.flux_place(inst, path, flux) == ('optimal', 'spectrum')
    # not an image of the flux's shape: what flux_extension looks for
    assert readers.flux_extension(path, flux) is None


def test_the_cube_can_write_what_the_reader_returns(tmp_path):
    """the metadata of an exposure go into the cube's meta.fits: none of it
    may be None (the flux_extension of a table used to be)"""
    path = _expres_file(tmp_path)
    payload = readers.read_any(path, _config())
    out = str(tmp_path / 'meta.fits')
    Table(rows=[payload['meta'], payload['meta']]).write(out)
    assert len(Table.read(out)) == 2


def test_expres_is_fitted_in_the_observer_frame(tmp_path):
    """the grid is barycentric: PCA2D brings it back to the observer's frame
    by the BERV the telluric cleaning uses (get_berv_transmission)"""
    layer = instrument_layer('EXPRES', 'ORIG')
    assert layer['input']['to_observer'] is True
    assert layer['correct']['mask_share'] == 0.5
    path = _expres_file(tmp_path)
    payload = readers.read_any(path, _config())
    berv = _berv_kms()
    assert payload['meta']['berv'] == pytest.approx(berv, rel=1e-9)
    assert payload['meta']['berv_obs'] == pytest.approx(berv, rel=1e-9)
    bary = readers.instrument('EXPRES', 'ORIG').get_wave_solution(path)
    # relativistic: the observer's wavelength is the barycentric one over
    #   sqrt((1 + v/c) / (1 - v/c)), here 1 + z exactly
    assert payload['wave'] == pytest.approx(
        bary / (1 + _FILES.REDSHIFT), rel=1e-12)


def test_round_trip_only_the_flux_changes(inst, tmp_path):
    path = _expres_file(tmp_path)
    flux0, header0, wave0, blaze0 = _read(inst, path)
    outdir = str(tmp_path / 'HD10700_pca2d')
    new, touched, _ = reconstruct.correct_file(
        _model(), _row(_berv_kms()), path, outdir, n_star=0, n_earth=1,
        config=_config(), keep_name=True)
    assert os.path.basename(new) == os.path.basename(path)
    assert touched > 0
    flux1, header1, wave1, blaze1 = _read(inst, new)
    # the wavelengths, the blaze and the header LBL reads are the same
    assert np.array_equal(wave1, wave0)
    assert np.array_equal(blaze1, blaze0, equal_nan=True)
    assert {'MJDMID', 'BJD', 'BERV'} <= set(header0.data)
    for key in header0.data:
        assert header1[key] == header0[key], key
    assert header1['PCA2REF'] is True
    # the flux: NaN where it was (outside Excalibur's range), and divided
    #   by the line where it is measured
    assert np.array_equal(np.isnan(flux1), np.isnan(flux0))
    good = np.isfinite(flux0)
    ratio = np.where(good, flux1 / np.where(good, flux0, 1.0), np.nan)
    observer = wave0 / (1 + _FILES.REDSHIFT)
    expected = np.exp(LINE_DEPTH * np.exp(
        -0.5 * ((observer - LINE_NM) / LINE_WIDTH) ** 2))
    assert ratio[good] == pytest.approx(expected[good], rel=1e-5)
    # the line is corrected where it falls in the OBSERVER's frame, 37
    #   pixels from where it would fall on the barycentric grid
    peak = int(np.nanargmax(ratio[0]))
    assert abs(observer[0, peak] - LINE_NM) < 0.0011
    assert abs(wave0[0, peak] - LINE_NM) > 0.03
    # far from the line the flux is the delivered flux
    far = good & (np.abs(observer - LINE_NM) > 20 * LINE_WIDTH)
    assert np.allclose(flux1[far], flux0[far], rtol=1e-12, atol=0)


def test_round_trip_the_rest_of_the_file_is_untouched(inst, tmp_path):
    path = _expres_file(tmp_path)
    new, _, _ = reconstruct.correct_file(
        _model(), _row(_berv_kms()), path, str(tmp_path / 'out'), n_star=0,
        n_earth=1, config=_config(), keep_name=True)
    with fits.open(path) as before, fits.open(new) as after:
        assert [h.name for h in after] == [h.name for h in before]
        for hdu_b, hdu_a in zip(before[1:], after[1:]):
            # the same cards (a copy of a table HDU may move EXTNAME up)
            assert dict(hdu_a.header) == dict(hdu_b.header)
            for name in hdu_b.columns.names:
                if hdu_b.name.lower() == 'optimal' and name == 'spectrum':
                    continue
                assert np.array_equal(hdu_a.data[name], hdu_b.data[name],
                                      equal_nan=True), name
        # the primary header: the original cards, plus pca2d's own
        for key in before[0].header:
            assert after[0].header[key] == before[0].header[key]
        # the spectrum column, where LBL does not read it (no Excalibur
        #   wavelengths), is the original one, to the bit
        spec_b = before['optimal'].data['spectrum']
        spec_a = after['optimal'].data['spectrum']
        outside = ~np.isfinite(before['optimal'].data['bary_excalibur'])
        assert np.array_equal(spec_a[outside], spec_b[outside],
                              equal_nan=True)
        assert not np.array_equal(spec_a, spec_b)


def test_round_trip_blanked_samples_are_nan(inst, tmp_path):
    """what the fit gave no weight comes back NaN through the class"""
    path = _expres_file(tmp_path)
    flux0, _, wave0, _ = _read(inst, path)
    model = _model()
    alive = np.ones((2, model['grid'].size), dtype=bool)
    blank = (model['grid'] > 502.0) & (model['grid'] < 502.2)
    alive[0, blank] = False
    new, _, _ = reconstruct.correct_file(
        model, _row(_berv_kms()), path, str(tmp_path / 'out'), n_star=0,
        n_earth=1, config=_config(), keep_name=True, alive=alive)
    flux1, _, _, _ = _read(inst, new)
    observer = wave0 / (1 + _FILES.REDSHIFT)
    inside = (observer[0] > 502.01) & (observer[0] < 502.19)
    assert inside.sum() > 100
    assert np.isnan(flux1[0][inside]).all()
    assert np.isfinite(flux0[0][inside]).all()
    # the other parity's orders keep their samples
    assert np.array_equal(np.isnan(flux1[1]), np.isnan(flux0[1]))


def test_an_image_flux_is_still_found_and_scaled(tmp_path):
    """the image path the other bridged instruments take: the extension of
    the flux's shape, multiplied in place"""
    path = str(tmp_path / 'e2ds.fits')
    flux = np.arange(12, dtype=float).reshape(3, 4) + 1
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones(5), name='X'),
                  fits.ImageHDU(flux, name='SCIDATA')]).writeto(path)

    class Image:
        def science_flux_column(self):
            return None
    place = readers.flux_place(Image(), path, flux)
    assert place == ('SCIDATA', None)
    factor = np.full(flux.shape, 2.0)
    factor[0, 0] = np.nan
    with fits.open(path) as hdulist:
        out = fits.HDUList([h.copy() for h in hdulist])
    readers.scale_flux(out, place, factor)
    assert np.isnan(out['SCIDATA'].data[0, 0])
    assert np.array_equal(out['SCIDATA'].data.ravel()[1:],
                          2 * flux.ravel()[1:])
    assert np.array_equal(out['X'].data, np.ones(5))

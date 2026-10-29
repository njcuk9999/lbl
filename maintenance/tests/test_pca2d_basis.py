"""lbl.pca2d.basis: the components of a fit saved in DATA_DIR/pca2d/
components and read back, and the checks made before a fit uses an observer
basis frozen.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_basis.py
"""
import os

import numpy as np
import pytest
from astropy import constants
from astropy.io import fits
from astropy.table import Table

from lbl.pca2d import basis

C_KMS = constants.c.to('km/s').value


def _fake_run(folder, n_rows=6, n_pix=50, n_earth=3):
    """A fit folder with what basis.save reads: two stars, two parity rows
    per exposure"""
    rng = np.random.default_rng(0)
    grid = 965.0 * np.exp(np.arange(n_pix) * 0.5 / C_KMS)
    q = np.linalg.qr(rng.normal(size=(n_pix, n_earth)))[0].T
    files = np.repeat(['f%d.fits' % i for i in range(n_rows // 2)], 2)
    np.savez_compressed(
        os.path.join(folder, 'fit.npz'), grid=grid, Q=q,
        P=np.zeros((1, n_pix)), b=rng.normal(size=(n_rows, n_earth)),
        bjd=2461000.5 + np.repeat(np.arange(n_rows // 2), 2),
        berv=np.repeat([-20.0, 5.0, 25.0], 2), parity=np.array([0, 1, 0, 1, 2, 3]),
        filename=files, rejected=np.zeros(n_rows, dtype=bool),
        power_earth=np.array([3.0, 2.0, 1.0]), dv=0.5,
        pca2d_code='abc', anc_labels=np.array(['airmass']),
        anc_values=np.ones((1, n_rows)), a=rng.normal(size=(n_rows, 1)),
        templates=np.arange(4 * n_pix, dtype=float).reshape(4, n_pix))
    table = Table(dict(wavelength=grid, constrained=np.ones(n_pix, bool)))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, name='BASIS')]
                 ).writeto(os.path.join(folder, 'twoframe_components.fits'))
    return grid, q


def test_a_basis_comes_back_as_it_was_saved(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    grid, q = _fake_run(str(run))
    path = basis.save(str(run), 'test_basis', str(tmp_path), ['A', 'B'],
                      instrument='NIRPS_HE')
    assert path == str(tmp_path / 'pca2d' / 'components' /
                       'PCA2D_Observer_test_basis_nirps_he.fits')
    # by its output name (this instrument's), its file name, its path
    loaded = basis.load('test_basis', str(tmp_path), 'NIRPS_HE')
    assert basis.load('PCA2D_Observer_test_basis_nirps_he',
                      str(tmp_path)).path == path == basis.load(path).path
    assert loaded.ident == basis.basis_id(grid, q)
    assert np.array_equal(loaded.on_grid(grid), q)
    assert loaded.on_grid(grid, 2).shape == (2, grid.size)
    # one row per exposure, its star from the parity group
    assert list(loaded.amplitudes['object']) == ['A', 'A', 'B']
    assert list(loaded.training['nrows']) == [2, 1]
    assert loaded.header['NEARTH'] == 3 and loaded.header['OBJ2'] == 'B'


def test_a_cube_on_another_grid_or_too_many_components_is_refused(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    grid, _ = _fake_run(str(run))
    loaded = basis.Basis(basis.save(str(run), 'b', str(tmp_path), ['A', 'B']))
    with pytest.raises(ValueError, match='not on the grid'):
        loaded.on_grid(grid * (1 + 1e-9))
    with pytest.raises(ValueError, match='has 3'):
        loaded.on_grid(grid, 4)
    assert loaded.check_dates(np.array([61000.0, 70000.0])) == 1
    # saved without an instrument or a DATA_SOURCE: it cannot be checked,
    #   and stops
    with pytest.raises(ValueError, match='does not say which INSTRUMENT'):
        loaded.check_compatible('SPIROU', 'CADC')


def test_a_basis_of_another_instrument_is_refused(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _fake_run(str(run))
    loaded = basis.Basis(basis.save(str(run), 'b', str(tmp_path), ['A', 'B'],
                                    instrument='NIRPS_HE'))
    with pytest.raises(ValueError, match='made with NIRPS_HE spectra, this'
                                         ' run is SPIROU'):
        loaded.check_compatible('SPIROU', 'CADC')
    # no DATA_SOURCE given when saved: refused
    with pytest.raises(ValueError, match='does not say which DATA_SOURCE'):
        loaded.check_compatible('NIRPS_HE', 'CADC')


def test_a_basis_of_another_data_source_is_refused(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _fake_run(str(run))
    loaded = basis.Basis(basis.save(str(run), 'b', str(tmp_path), ['A', 'B'],
                                    instrument='NIRPS_HE',
                                    data_source='CADC'))
    loaded.check_compatible('nirps_he', 'cadc')
    with pytest.raises(ValueError) as err:
        loaded.check_compatible('NIRPS_HE', 'ESO', others='none here')
    text = str(err.value)
    # both sources, the file, why and what to do
    assert 'made from CADC spectra, this run reads ESO spectra' in text
    assert loaded.path in text and 'Why:' in text and 'What to do:' in text
    assert 'none here' in text


def test_each_star_of_a_joint_fit_gets_its_own_file(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    grid, _ = _fake_run(str(run))
    paths = basis.save_stars(str(run), str(tmp_path), ['A', 'B'],
                             instrument='NIRPS_HE', name='v1')
    assert [os.path.basename(p) for p in paths] == [
        'PCA2D_Star_A_v1_nirps_he.fits', 'PCA2D_Star_B_v1_nirps_he.fits']
    with fits.open(paths[1]) as hdul:
        # star B is the third and fourth parity groups of the joint fit
        assert np.array_equal(hdul['TEMPLATES'].data[0],
                              np.arange(2 * grid.size, 3 * grid.size))
        assert hdul[0].header['OBJECT'] == 'B'
        assert len(hdul['AMPLITUDES'].data) == 1

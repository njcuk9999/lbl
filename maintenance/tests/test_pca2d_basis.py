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
    with pytest.raises(ValueError, match='NOT ON THE GRID'):
        loaded.on_grid(grid * (1 + 1e-9))
    with pytest.raises(ValueError, match='has 3'):
        loaded.on_grid(grid, 4)
    assert loaded.check_dates(np.array([61000.0, 70000.0])) == 1
    # saved without an instrument or a DATA_SOURCE: it cannot be checked,
    #   and stops
    with pytest.raises(ValueError, match='does not say which INSTRUMENT'):
        loaded.check_compatible('SPIROU', 'CADC')


def test_a_basis_on_another_grid_says_what_differs_and_to_rebuild_it(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    grid, _ = _fake_run(str(run))
    loaded = basis.Basis(basis.save(str(run), 'b', str(tmp_path), ['A', 'B']))
    step = 0.5 / C_KMS
    assert loaded.grid_problem(grid) == ''
    # the same comb cut to another domain (an older LBL's): what this run
    #   covers and the basis does not is named, and what to do
    text = loaded.grid_problem(965.0 * np.exp(np.arange(5, 60) * step))
    assert text.startswith('THE OBSERVER BASIS b IS NOT ON THE GRID OF THIS'
                           ' RUN: THE BASIS MUST BE REBUILT.')
    assert 'the wavelength domain' in text
    assert 'the basis has no components from' in text
    assert "REBUILD THE BASIS ON THIS GRID" in text
    assert "output_name='b', on A, B" in text and loaded.path in text
    # another step, another anchor
    assert 'what differs: the step' in loaded.grid_problem(
        965.0 * np.exp(np.arange(50) * 0.6 / C_KMS))
    assert 'what differs: the anchor' in loaded.grid_problem(
        965.1 * np.exp(np.arange(50) * step))


def test_the_grid_of_a_run_is_checked_before_it_runs(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    grid, _ = _fake_run(str(run))
    loaded = basis.Basis(basis.save(str(run), 'b', str(tmp_path), ['A', 'B']))
    domain = dict(wave0=965.0, dv=0.5, wave_min=965.0,
                  wave_max=grid[-1] * (1 + 1e-12), smart_dv=False,
                  grid_source='magic')
    assert loaded.check_domain(domain) is True
    # dv measured from the spectra, or the grid of an s1d: only the fit can
    #   tell
    assert loaded.check_domain(dict(domain, smart_dv=True)) is False
    assert loaded.check_domain(dict(domain, grid_source='native')) is False
    with pytest.raises(ValueError, match='THE BASIS MUST BE REBUILT'):
        loaded.check_domain(dict(domain, wave_max=grid[-1] * 1.0001))


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


def test_an_exposure_s_snr_is_the_median_of_its_orders(tmp_path):
    """tfits.exposure_snr: every order's extraction SNR (APERO's EXTSNxxx),
    the SNR the LBL window shows; header_snr the same from the headers"""
    from lbl.pca2d import tfits
    assert tfits.exposure_snr(dict(snr=[10.0, np.nan, 30.0, 20.0, 0.0])) \
        == 20.0
    assert np.isnan(tfits.exposure_snr(dict(snr=[np.nan])))
    head = fits.Header()
    for order, value in enumerate([12.0, 40.0, 33.0]):
        head['EXTSN%03d' % order] = value
    path = str(tmp_path / 'x_t.fits')
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(np.ones((3, 4)), name=n,
                                                header=head if n == 'FluxA'
                                                else None)
                                  for n in ('FluxA', 'WaveA', 'BlazeA',
                                            'Recon')]
    fits.HDUList(hdus).writeto(path)
    assert tfits.header_snr(path) == 33.0
    assert np.isnan(tfits.header_snr(str(tmp_path / 'none.fits')))


def _cube(folder, members=True, snr=True):
    """A cube folder: three rows (one per exposure, or the bundles of four
    exposures)"""
    os.makedirs(folder)
    Table(dict(filename=['f0.fits', 'f1.fits', 'f2.fits'],
               bjd=[2461000.5, 2461001.5, 2461002.5],
               snr_orders=[50.0, 60.0, 70.0], snr_band=[55.0, 65.0, 75.0])
          ).write(os.path.join(folder, 'meta.fits'))
    if members:
        table = Table(dict(filename=['f0.fits', 'g0.fits', 'f1.fits',
                                     'f2.fits'],
                           row=['f0.fits', 'f0.fits', 'f1.fits', 'f2.fits'],
                           group=['n0', 'n0', 'n1', 'n2'],
                           bjd=[2461000.5, 2461000.6, 2461001.5, 2461002.5],
                           berv=[1.0, 1.0, 2.0, 3.0]))
        if snr:
            table['snr'] = [40.0, 42.0, 60.0, 70.0]
            table['snr_band'] = [41.0, 43.0, 61.0, 71.0]
        table.write(os.path.join(folder, 'members.csv'), format='ascii.csv')


def test_a_fit_lists_its_input_exposures_and_their_snr(tmp_path):
    """basis.write_inputs: one row per exposure, from the members of a cube
    of bundles or the rows of a cube of exposures; an SNR the cube did not
    keep is read from the file's header"""
    cube = str(tmp_path / 'cube')
    _cube(cube)
    out = basis.write_inputs(cube, str(tmp_path))
    table = Table.read(out, format='ascii.csv')
    assert list(table['filename']) == ['f0.fits', 'g0.fits', 'f1.fits',
                                       'f2.fits']
    assert list(table['row']) == ['f0.fits', 'f0.fits', 'f1.fits', 'f2.fits']
    assert list(table['snr']) == [40.0, 42.0, 60.0, 70.0]
    plain = str(tmp_path / 'plain')
    _cube(plain, members=False)
    table = Table.read(basis.write_inputs(plain, str(tmp_path)),
                       format='ascii.csv')
    assert list(table['snr']) == [50.0, 60.0, 70.0]
    assert list(table['row']) == list(table['filename'])
    old = str(tmp_path / 'old')
    _cube(old, snr=False)
    science = tmp_path / 'science'
    science.mkdir()
    head = fits.Header()
    head['EXTSN000'] = 25.0
    fits.HDUList([fits.PrimaryHDU()] + [
        fits.ImageHDU(np.ones((1, 2)), name=n, header=head)
        for n in ('FluxA', 'WaveA', 'BlazeA', 'Recon')]).writeto(
        str(science / 'g0.fits'))
    table = Table.read(basis.write_inputs(old, str(tmp_path),
                                          [str(science)]),
                       format='ascii.csv')
    assert table['snr'][1] == 25.0
    assert np.isnan(table['snr'][0])


def test_a_basis_keeps_the_snr_of_its_input_exposures(tmp_path):
    """INPUTS: each input exposure, its star and whether its row was fitted;
    SNRMED and its percentiles in the header, over the fitted ones"""
    run = tmp_path / 'run'
    run.mkdir()
    _fake_run(str(run))
    Table(dict(filename=['f0.fits', 'h0.fits', 'f1.fits', 'f2.fits', 'x.fits'],
               row=['f0.fits', 'f0.fits', 'f1.fits', 'f2.fits', 'x.fits'],
               bjd=[1.0, 1.1, 2.0, 3.0, 4.0],
               snr=[30.0, 50.0, 40.0, 80.0, 500.0],
               snr_band=[31.0, 51.0, 41.0, 81.0, 501.0])).write(
        str(run / basis.INPUTS_FILE), format='ascii.csv')
    path = basis.save(str(run), 'snr_basis', str(tmp_path), ['A', 'B'],
                      instrument='NIRPS_HE')
    loaded = basis.load(path)
    assert list(loaded.inputs['object']) == ['A', 'A', 'A', 'B', '']
    assert list(loaded.inputs['used']) == [True, True, True, True, False]
    assert loaded.header['SNRMED'] == 45.0 and loaded.header['SNRN'] == 4
    # each training star's, in TRAINING
    assert list(loaded.training['snr']) == [40.0, 80.0]
    assert list(loaded.training['nexp']) == [3, 1]
    # each star's own file: its exposures and their SNR
    stars = basis.save_stars(str(run), str(tmp_path), ['A', 'B'],
                             instrument='NIRPS_HE')
    with fits.open(stars[0]) as hdul:
        assert list(hdul['INPUTS'].data['filename']) == [
            'f0.fits', 'h0.fits', 'f1.fits']
        assert hdul[0].header['SNRMED'] == 40.0
    assert fits.getheader(stars[1])['SNRMED'] == 80.0
    # a fit that did not list them: no INPUTS, no SNRMED
    os.remove(str(run / basis.INPUTS_FILE))
    path = basis.save(str(run), 'plain_basis', str(tmp_path), ['A', 'B'],
                      instrument='NIRPS_HE')
    assert basis.load(path).inputs is None
    assert 'SNRMED' not in basis.load(path).header


def test_a_basis_saved_before_gets_its_inputs_snr(tmp_path):
    """basis.add_inputs: a basis without INPUTS (saved before 2026-11-25)
    gets them from the cube its fit was made on, its components and
    BASISID as they were; a star's file its own exposures only"""
    run = tmp_path / 'run'
    run.mkdir()
    _fake_run(str(run))
    path = basis.save(str(run), 'old_basis', str(tmp_path), ['A', 'B'],
                      instrument='NIRPS_HE')
    stars = basis.save_stars(str(run), str(tmp_path), ['A', 'B'],
                             instrument='NIRPS_HE')
    before = basis.load(path)
    assert before.inputs is None
    cube = str(tmp_path / 'cube')
    _cube(cube)
    assert basis.add_inputs(path, cube) == 4
    after = basis.load(path)
    assert after.ident == before.ident
    assert np.array_equal(after.components, before.components)
    assert list(after.inputs['object']) == ['A', 'A', 'A', 'B']
    assert after.header['SNRMED'] == 51.0
    assert list(after.training['snr']) == [42.0, 70.0]
    assert basis.add_inputs(stars[1], cube) == 1
    with fits.open(stars[1]) as hdul:
        assert list(hdul['INPUTS'].data['filename']) == ['f2.fits']
        assert hdul[0].header['SNRMED'] == 70.0

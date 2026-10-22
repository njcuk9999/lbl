"""The components of PCA2D fits, saved and read back: the observer-frame
basis (tellurics left by the DRS, airglow, the instrument), which another
star can use FROZEN, and each star's own spectrum and components.

WHY. A star whose BERV barely moves over its run (TOI-904: 2.3 km/s in 75
days) cannot tell its own frame from the observer's: whatever is fixed on the
detector is fixed on the star as well. A basis learned on stars whose BERV
does move, where the two frames separate, is brought in frozen: the new star
fits only its own spectrum and components, and the amplitudes of the basis.

THE FOLDER, DATA_DIR/pca2d/components/, where every fit writes and where
a run reads (or where one puts a file to be read), named as LBL names its
templates, the instrument last:

    PCA2D_Observer_<output>_<instrument>.fits
                         the observer basis of a fit; <output> is the PCA2D
                         output_name, or the object(s) of the fit
    PCA2D_Star_<object>_<output>_<instrument>.fits
                         each star of the fit: its spectrum per order
                         parity, its star-frame components and their
                         amplitudes

PCA2D input_name=<output> reads the observer basis of that name, frozen.

THE OBSERVER FILE:

    PRIMARY      header only: what the basis is and what it depends on
                 (BASISNAM, BASISID, INSTRUME, NEARTH, WAVE0, DV, NPIX, the
                 high-pass, the MJD range, the training stars, the code)
    WAVE         the observer-frame grid the components live on [nm, vacuum]
    COMPONENTS   (NEARTH, NPIX): the observer-frame components, in the fit's
                 own space (high-passed log flux), by decreasing power
    MASK         (NPIX) 1 where the training constrained the basis
    VARIANCE     one row per component: its weighted power and its share
    TRAINING     one row per training star: rows, MJD and BERV ranges
    AMPLITUDES   one row per training row: star, file, BJD, BERV, the
                 ancillary quantities the fit kept (airmass, humidity...) and
                 the amplitude on each component

BASISID is a hash of the grid and the components: it enters the request
stamp of a correction made with the basis, so that another basis makes the
correction again, and it is written in the corrected spectra's headers.

Created on 2026-10-22

@author: artigau
"""
import datetime
import hashlib
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table

#: where the components are, under DATA_DIR
FOLDER = os.path.join('pca2d', 'components')


def component_file(kind: str, label: str, instrument: str,
                   data_dir: str) -> str:
    """<DATA_DIR>/pca2d/components/PCA2D_<Kind>_<label>_<instrument>.fits,
    kind 'observer' or 'star', the instrument in lower case as in LBL's
    templates (nirps_he, spirou)"""
    return os.path.join(data_dir, FOLDER, 'PCA2D_{0}_{1}_{2}.fits'.format(
        kind.capitalize(), label, (instrument or 'unknown').lower()))


def basis_path(name: str, data_dir: str = None, instrument: str = '') -> str:
    """The file of an observer basis: name may be a path, a file of the
    components folder (PCA2D_Observer_2026a_nirps_he), or the output name
    alone (2026a, for this instrument)"""
    if os.path.exists(name) or name.endswith('.fits') and os.sep in name:
        return os.path.abspath(name)
    if data_dir is None:
        raise ValueError('basis %r: not a file, and no DATA_DIR to look in'
                         % name)
    stem = name[:-len('.fits')] if name.endswith('.fits') else name
    if stem.startswith('PCA2D_Observer_'):
        return os.path.join(data_dir, FOLDER, stem + '.fits')
    return component_file('observer', stem, instrument, data_dir)


def basis_id(grid: np.ndarray, components: np.ndarray) -> str:
    """12 hex digits that change with the grid or any component"""
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(grid, dtype='<f8').tobytes())
    digest.update(np.ascontiguousarray(components, dtype='<f8').tobytes())
    return digest.hexdigest()[:12]


def save(run_dir: str, name: str, data_dir: str, objects: list,
         instrument: str = '', config: dict = None,
         data_source: str = '') -> str:
    """
    Save the observer-frame basis of a finished fit

    :param run_dir: str, the fit's folder (fit.npz, twoframe_components.fits)
    :param name: str, the basis' name
    :param data_dir: str, LBL's DATA_DIR
    :param objects: list of str, the stars of the fit, in its order (a joint
                    fit numbers them 0, 1, ... in its parity groups)
    :param instrument: str, LBL's instrument (NIRPS_HE, SPIROU...)
    :param config: dict, the fit's configuration (for the high-pass)
    :param data_source: str, LBL's DATA_SOURCE (the pipeline that reduced
                        the training spectra)

    :return: str, the file written
    """
    fit = np.load(os.path.join(run_dir, 'fit.npz'), allow_pickle=False)
    grid = np.asarray(fit['grid'], dtype=float)
    components = np.asarray(fit['Q'], dtype=float)
    n_earth = components.shape[0]
    with fits.open(os.path.join(run_dir, 'twoframe_components.fits')) as hdul:
        mask = np.asarray(hdul['BASIS'].data['constrained'], dtype=np.uint8)
        head = hdul[0].header
    keep = ~np.asarray(fit['rejected'], dtype=bool)
    bjd = np.asarray(fit['bjd'], dtype=float)
    berv = np.asarray(fit['berv'], dtype=float)
    parity = np.asarray(fit['parity'], dtype=int)
    # the star of each row: a joint fit has two parity groups per star
    member = parity // 2 if len(objects) > 1 else np.zeros_like(parity)
    star = np.array([objects[m] if m < len(objects) else '?'
                     for m in member])
    # one row per exposure (the parity rows share their amplitudes)
    first = np.ones(len(parity), dtype=bool)
    first[1:] = np.asarray(fit['filename'])[1:] != \
        np.asarray(fit['filename'])[:-1]
    rows = first & keep
    mjd = bjd - 2400000.5
    ident = basis_id(grid, components)
    hpass = (config or {}).get('highpass', {})

    primary = fits.Header()
    cards = [('BASISNAM', name, 'name of this basis'),
             ('BASISID', ident, 'hash of the grid and the components'),
             ('INSTRUME', instrument, 'LBL instrument of the training data'),
             ('DATASRC', data_source, 'LBL DATA_SOURCE of the training data'),
             ('INFORMAT', head.get('INFORMAT', ''), 'input format of the cube'),
             ('CONTENT', head.get('CONTENT', ''), 'space of the components'),
             ('NEARTH', n_earth, 'observer-frame components'),
             ('NSTAR', int(fit['P'].shape[0]), 'star components of the fit'),
             ('WAVE0', float(grid[0]), 'nm, first sample of the grid'),
             ('DV', float(fit['dv']), 'km/s per sample'),
             ('NPIX', int(grid.size), 'samples of the grid'),
             ('HPMODE', str(hpass.get('mode', head.get('HIGHPASS', ''))),
              'high-pass mode'),
             ('HPWIDTH', float(hpass.get('width_kms', np.nan)),
              'km/s, high-pass width'),
             ('MJDMIN', float(np.nanmin(mjd[rows])), 'first training row'),
             ('MJDMAX', float(np.nanmax(mjd[rows])), 'last training row'),
             ('NOBJ', len(objects), 'training stars'),
             ('NROWS', int(rows.sum()), 'training rows (exposures, nights)'),
             ('CODE', str(fit['pca2d_code']), 'pca2d code of the fit'),
             ('RUNDIR', os.path.basename(os.path.normpath(run_dir)),
              'the fit it comes from'),
             ('DATE', datetime.datetime.now().isoformat(timespec='seconds'),
              'when the basis was saved')]
    for key, value, comment in cards:
        # a quantity the fit did not record is left out, not written as NaN
        if isinstance(value, float) and not np.isfinite(value):
            continue
        primary[key] = (value, comment)
    for index, obj in enumerate(objects):
        primary['OBJ%d' % (index + 1)] = (obj, 'training star')

    power = np.asarray(fit['power_earth'], dtype=float)
    variance = Table(dict(component=np.arange(1, n_earth + 1), power=power,
                          share=power / np.sum(power)))
    training = Table(dict(
        object=list(objects),
        nrows=[int(np.sum(rows & (star == obj))) for obj in objects],
        mjd_min=[float(np.nanmin(mjd[rows & (star == obj)]))
                 if np.any(rows & (star == obj)) else np.nan
                 for obj in objects],
        mjd_max=[float(np.nanmax(mjd[rows & (star == obj)]))
                 if np.any(rows & (star == obj)) else np.nan
                 for obj in objects],
        berv_min=[float(np.nanmin(berv[rows & (star == obj)]))
                  if np.any(rows & (star == obj)) else np.nan
                  for obj in objects],
        berv_max=[float(np.nanmax(berv[rows & (star == obj)]))
                  if np.any(rows & (star == obj)) else np.nan
                  for obj in objects]))
    amplitudes = Table(dict(object=star[rows],
                            filename=np.asarray(fit['filename'])[rows],
                            bjd=bjd[rows], berv=berv[rows]))
    if 'anc_labels' in fit.files:
        for label, values in zip(fit['anc_labels'], fit['anc_values']):
            column = str(label).replace(' ', '_').replace('.', '')
            if column not in amplitudes.colnames:
                amplitudes[column] = np.asarray(values, dtype=float)[rows]
    b = np.asarray(fit['b'], dtype=float)
    for index in range(n_earth):
        amplitudes['b%d' % (index + 1)] = b[rows, index]

    out = component_file('observer', name, instrument, data_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    hdus = [fits.PrimaryHDU(header=primary),
            fits.ImageHDU(grid, name='WAVE'),
            fits.ImageHDU(components, name='COMPONENTS'),
            fits.ImageHDU(mask, name='MASK'),
            fits.BinTableHDU(variance, name='VARIANCE'),
            fits.BinTableHDU(training, name='TRAINING'),
            fits.BinTableHDU(amplitudes, name='AMPLITUDES')]
    fits.HDUList(hdus).writeto(out, overwrite=True)
    return out


class Basis:
    """A saved basis, read back: .grid, .components, .mask, .header, .ident,
    .name, .training (Table), .amplitudes (Table)"""

    def __init__(self, path: str):
        self.path = path
        with fits.open(path) as hdul:
            self.header = hdul[0].header.copy()
            self.grid = np.asarray(hdul['WAVE'].data, dtype=float)
            self.components = np.asarray(hdul['COMPONENTS'].data, dtype=float)
            self.mask = np.asarray(hdul['MASK'].data, dtype=bool)
            self.training = Table(hdul['TRAINING'].data)
            self.amplitudes = Table(hdul['AMPLITUDES'].data)
        self.name = self.header.get('BASISNAM', os.path.basename(path))
        self.ident = self.header.get('BASISID', '')
        self.n_earth = self.components.shape[0]

    def describe(self) -> str:
        """One line: name, id, components, stars and dates"""
        return ('%s (%s): %d observer components, trained on %d stars (%s),'
                ' MJD %.0f to %.0f' % (
                    self.name, self.ident, self.n_earth,
                    self.header.get('NOBJ', 0),
                    ', '.join(str(o) for o in self.training['object']),
                    self.header.get('MJDMIN', np.nan),
                    self.header.get('MJDMAX', np.nan)))

    def on_grid(self, grid: np.ndarray, n_earth: int = None) -> np.ndarray:
        """
        The first n_earth components, after checking that the cube is on the
        basis' grid (same samples, same wavelengths)

        :param grid: np.ndarray, the cube's grid
        :param n_earth: int, how many (all by default)

        :return: np.ndarray (n_earth, n_pix)
        """
        n_earth = self.n_earth if n_earth is None else int(n_earth)
        if n_earth > self.n_earth:
            raise ValueError('%d observer components asked of basis %s, which'
                             ' has %d' % (n_earth, self.name, self.n_earth))
        if grid.size != self.grid.size or not np.allclose(
                grid, self.grid, rtol=1e-12, atol=0):
            raise ValueError(
                'the cube is not on the grid of basis %s (%d samples from'
                ' %.4f nm here, %d from %.4f nm there): same instrument and'
                ' domain settings are needed' % (
                    self.name, grid.size, grid[0], self.grid.size,
                    self.grid[0]))
        return self.components[:n_earth].copy()

    def check_dates(self, mjd: np.ndarray, margin: float = 60.0) -> int:
        """How many of these dates fall more than margin days outside the
        training's"""
        mjd = np.asarray(mjd, dtype=float)
        low = self.header.get('MJDMIN', -np.inf) - margin
        high = self.header.get('MJDMAX', np.inf) + margin
        return int(np.sum((mjd < low) | (mjd > high)))


def load(name: str, data_dir: str = None, instrument: str = '') -> Basis:
    """A saved observer basis, by label (the instrument's, in DATA_DIR/pca2d/
    components), file name or path"""
    path = basis_path(name, data_dir, instrument)
    if not os.path.exists(path):
        raise FileNotFoundError('no observer basis %s (%s)' % (name, path))
    return Basis(path)


def save_stars(run_dir: str, data_dir: str, objects: list,
               instrument: str = '', name: str = None) -> list:
    """
    Save each star's part of a finished fit, one file per star:
    PCA2D_Star_<object>[_<name>]_<instrument>.fits

        PRIMARY      header only: the star, the instrument, the grid, NSTAR,
                     the MJD range, the code
        WAVE         the grid [nm], in the star's frame
        TEMPLATES    (2, NPIX): the star's spectrum, even and odd orders
        COMPONENTS   (NSTAR, NPIX): the star-frame components (shared by the
                     stars of a joint fit)
        AMPLITUDES   one row per exposure: file, BJD, BERV, the amplitudes

    :param run_dir: str, the fit's folder
    :param data_dir: str, LBL's DATA_DIR
    :param objects: list of str, the stars of the fit, in its order
    :param instrument: str, LBL's instrument
    :param name: str, the fit's output name, if it has one

    :return: list of str, the files written
    """
    fit = np.load(os.path.join(run_dir, 'fit.npz'), allow_pickle=False)
    grid = np.asarray(fit['grid'], dtype=float)
    star_pcs = np.asarray(fit['P'], dtype=float)
    templates = (np.asarray(fit['templates'], dtype=float)
                 if 'templates' in fit.files else
                 np.zeros((2 * len(objects), grid.size)))
    parity = np.asarray(fit['parity'], dtype=int)
    member = parity // 2 if len(objects) > 1 else np.zeros_like(parity)
    keep = ~np.asarray(fit['rejected'], dtype=bool)
    filename = np.asarray(fit['filename'])
    first = np.ones(len(parity), dtype=bool)
    first[1:] = filename[1:] != filename[:-1]
    written = []
    for index, obj in enumerate(objects):
        rows = first & keep & (member == index)
        mjd = np.asarray(fit['bjd'], dtype=float)[rows] - 2400000.5
        primary = fits.Header()
        cards = [('OBJECT', obj, 'the star'),
                 ('INSTRUME', instrument, 'LBL instrument'),
                 ('NSTAR', int(star_pcs.shape[0]), 'star-frame components'),
                 ('WAVE0', float(grid[0]), 'nm, first sample of the grid'),
                 ('DV', float(fit['dv']), 'km/s per sample'),
                 ('NPIX', int(grid.size), 'samples of the grid'),
                 ('MJDMIN', float(np.min(mjd)) if mjd.size else np.nan,
                  'first row'),
                 ('MJDMAX', float(np.max(mjd)) if mjd.size else np.nan,
                  'last row'),
                 ('NROWS', int(rows.sum()), 'rows (exposures, nights)'),
                 ('NOBJ', len(objects), 'stars of the fit (joint if > 1)'),
                 ('CODE', str(fit['pca2d_code']), 'pca2d code of the fit'),
                 ('RUNDIR', os.path.basename(os.path.normpath(run_dir)),
                  'the fit it comes from'),
                 ('DATE', datetime.datetime.now().isoformat(
                     timespec='seconds'), 'when it was saved')]
        for key, value, comment in cards:
            if isinstance(value, float) and not np.isfinite(value):
                continue
            primary[key] = (value, comment)
        if 'earth_basis' in fit.files and str(fit['earth_basis']):
            primary['EARTHBAS'] = (str(fit['earth_basis']),
                                   'observer basis, frozen')
            primary['EARTHBID'] = (str(fit['earth_basis_id']),
                                   'its BASISID')
        amplitudes = Table(dict(filename=filename[rows],
                                bjd=np.asarray(fit['bjd'])[rows],
                                berv=np.asarray(fit['berv'])[rows]))
        a = np.asarray(fit['a'], dtype=float)
        for comp in range(a.shape[1]):
            amplitudes['a%d' % (comp + 1)] = a[rows, comp]
        label = obj if not name else '{0}_{1}'.format(obj, name)
        out = component_file('star', label, instrument, data_dir)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fits.HDUList([fits.PrimaryHDU(header=primary),
                      fits.ImageHDU(grid, name='WAVE'),
                      fits.ImageHDU(templates[2 * index:2 * index + 2],
                                    name='TEMPLATES'),
                      fits.ImageHDU(star_pcs, name='COMPONENTS'),
                      fits.BinTableHDU(amplitudes, name='AMPLITUDES')]
                     ).writeto(out, overwrite=True)
        written.append(out)
    return written

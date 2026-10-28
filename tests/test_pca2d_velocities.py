#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA2D's star frame follows the velocities LBL's template fitted freely
(registration.velocities, lbl_pca2d.object_velocities), file by file, as it
follows known orbits.

Created on 2026-10-24

@author: artigau
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table

from lbl.pca2d import nights, readers
from lbl.recipes import lbl_pca2d

C_KMS = 299792.458


def test_the_star_frame_takes_each_file_its_velocity():
    config = dict(registration=dict(velocities={'a.fits': 0.05}))
    payload = dict(meta=dict(berv=10.0, bjd=2460000.5))
    readers.star_frame(payload, config, '/data/science/STAR/a.fits')
    meta = payload['meta']
    assert meta['berv_bary'] == 10.0 and meta['vstar'] == 0.05
    assert abs(meta['berv'] - (10.0 - 0.05) / (1 - 10.0 * 0.05 / C_KMS ** 2)) < 1e-12
    # a file it does not name, and no orbits: the BERV alone
    other = dict(meta=dict(berv=10.0, bjd=2460000.5))
    readers.star_frame(other, config, '/data/science/STAR/b.fits')
    assert other['meta']['berv'] == 10.0 and 'vstar' not in other['meta']
    # the bundles of a night are cut in the same frame
    bervs = nights.frame_bervs(['/x/a.fits', '/x/b.fits'], [10.0, 10.0], [],
                               {'a.fits': 0.05})
    assert bervs[1] == 10.0 and abs(bervs[0] - meta['berv']) < 1e-12


def _template(data_dir, name, vsys, kepler=None):
    folder = data_dir / 'templates'
    folder.mkdir(exist_ok=True)
    sci = Table(dict(FILENAME=['/d/science/%s/%d.fits' % (name, i)
                               for i in range(len(vsys))],
                     VSYS=np.asarray(vsys, dtype=float)))
    if kepler is not None:
        sci['VKEP'] = kepler
    hdus = [fits.PrimaryHDU(),
            fits.BinTableHDU(Table(dict(wavelength=[1.0, 2.0],
                                        flux=[1.0, 1.0])), name='TEMPLATE'),
            fits.BinTableHDU(sci, name='SCI_TABLE')]
    fits.HDUList(hdus).writeto(folder / ('LBL_Template_%s_spirou.fits' % name),
                               overwrite=True)


def test_the_free_fit_of_the_template_and_only_it(tmp_path):
    def quiet(*args, **kwargs):
        pass
    _template(tmp_path, 'STAR', [100.0, 130.0, 70.0, 100.0])
    found = lbl_pca2d.object_velocities('STAR', 'STAR', str(tmp_path),
                                        'SPIROU', quiet)
    assert found == {'0.fits': 0.0, '1.fits': 0.03, '2.fits': -0.03,
                     '3.fits': 0.0}
    # another star's template knows other spectra
    assert lbl_pca2d.object_velocities('STAR', 'GL699', str(tmp_path),
                                       'SPIROU', quiet) == {}
    # a template made on known orbits did not fit its velocities
    _template(tmp_path, 'STAR', [100.0, 130.0, 70.0, 100.0],
              kepler=[1.0, 2.0, 3.0, 4.0])
    assert lbl_pca2d.object_velocities('STAR', None, str(tmp_path),
                                       'SPIROU', quiet) == {}

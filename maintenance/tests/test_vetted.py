#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LBL's vetted templates (lbl.core.vetted): the list, the best match by Teff,
and a depot file written as an LBL template the first time a wrap's
VETTED_<object> is looked for.

Created on 2026-10-23

@author: artigau
"""
import os
import shutil

import numpy as np
from astropy import constants
from astropy.io import fits
from astropy.table import Table

from lbl.core import base_classes
from lbl.core import vetted
from lbl.instruments import select

C_MS = constants.c.value


def _inst(data_dir, instrument='SPIROU', source='CADC'):
    args = base_classes.ParamDict()
    args.set('INSTRUMENT', instrument, source='test')
    args.set('DATA_SOURCE', source, source='test')
    args.set('DATA_DIR', str(data_dir), source='test')
    return select.load_instrument(args)


def _depot(path, nfiles=40):
    """A depot file as make_vetted.py writes it: a grid uniform in
    velocity (1 km/s), a few lines, n_valid low at the ends"""
    npts = 6000
    wave = 1500.0 * np.exp(np.arange(npts) * 1000.0 / C_MS)
    flux = np.ones(npts)
    for centre in np.linspace(1502, 1508, 30):
        flux -= 0.4 * np.exp(-0.5 * ((wave - centre) / 0.01) ** 2)
    n_valid = np.full(npts, nfiles)
    n_valid[:100] = 3
    table = Table(dict(wavelength=wave, flux=flux.astype(np.float32),
                       eflux=np.full(npts, 1e-3, np.float32),
                       rms=np.full(npts, 1e-3, np.float32),
                       n_valid=n_valid.astype(np.int32)))
    sci = Table(dict(Filename=['a.fits', 'b.fits'], BERV=[-12.3, 20.1],
                     BJD=[2460000.5, 2460100.5], USED=[1, 1]))
    header = fits.Header()
    header['OBJECT'] = 'GL699'
    header['VETTED'] = True
    header['VETNFIL'] = nfiles
    header['VETTEFF'] = 3311.0
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.BinTableHDU(table, name='TEMPLATE'),
                  fits.BinTableHDU(sci, name='SCI_TABLE')]).writeto(path)


def test_the_list_and_the_best_match():
    found = vetted.entries('SPIROU')
    assert found and all(e['name'] == 'VETTED_' + e['object'] for e in found)
    teffs = [e['teff'] for e in found]
    assert teffs == sorted(teffs)
    assert vetted.entries('NIRPS_HE') and not vetted.entries('ESPADONS')
    best = vetted.best_match('SPIROU', 3320)
    assert best['object'] == 'GL699'
    # outside 2900-4300 K, or for an instrument without them: none
    assert vetted.best_match('SPIROU', 5000) is None
    assert vetted.best_match('SPIROU', 2800) is None
    assert vetted.best_match('HARPS', 3300) is None
    assert vetted.is_vetted('VETTED_GL699') and not vetted.is_vetted('GL699')


def test_a_depot_file_becomes_an_lbl_template(tmp_path):
    inst = _inst(tmp_path)
    depot = str(tmp_path / 'vetted_GL699_spirou.fits')
    _depot(depot)
    out = str(tmp_path / 'LBL_Template_VETTED_GL699_spirou.fits')
    vetted.write_template(inst, depot, out)
    table = inst.load_template(out)
    for col in ('flux', 'flux_odd', 'flux_even', 'flux_savgol_d0',
                'flux_savgol_d3', 'flux_odd_savgol_d1',
                'flux_even_savgol_d2'):
        assert col in table.colnames, col
    # where fewer than half the spectra contributed, no flux
    assert np.all(np.isnan(table['flux'][:100]))
    assert np.isfinite(table['flux'][200:-200]).all()
    header = fits.getheader(out)
    assert header['VETTED'] and header['VETNFIL'] == 40


def test_a_wrap_naming_a_vetted_template_gets_it(tmp_path, monkeypatch):
    inst = _inst(tmp_path)
    repo = tmp_path / 'repo'
    repo.mkdir()
    entry = vetted.entry_of('SPIROU', 'VETTED_GL699')
    _depot(str(repo / entry['file']))
    fetched = []

    def download(url, name, savepath, required=True):
        fetched.append(url)
        shutil.copy(str(repo / os.path.basename(url)), savepath)
    monkeypatch.setattr(vetted.io, 'get_urlfile', download)
    inst.params.set('OBJECT_SCIENCE', 'STAR', source='test')
    inst.params.set('OBJECT_COMPARISON', 'VETTED_GL699', source='test')
    templates = tmp_path / 'templates'
    templates.mkdir()
    path = inst.template_file(str(templates), 'comparison')
    assert os.path.basename(path) == 'LBL_Template_VETTED_GL699_spirou.fits'
    assert os.path.exists(path) and fetched[0].endswith(entry['file'])
    # there now: not fetched again
    inst.template_file(str(templates), 'comparison')
    assert len(fetched) == 1

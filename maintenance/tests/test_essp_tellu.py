#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A telluric-cleaned file of the ESSP classes is written in the layout their reader
reads. No real ESSP file was at hand: the layout is what the reader asks for.

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest
from astropy.io import fits

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _inst(instrument):
    args = base_classes.ParamDict()
    args.set('OBJECT_SCIENCE', 'TEST', source='test')
    args.set('OBJECT_COMPARISON', 'TEST', source='test')
    klass = select.InstDict[instrument]['ESSP']
    return klass(parameters.params.copy(), args)


def _essp_file(path, inst):
    flux = np.full((3, 50), 100.0)
    wave = 5000.0 + np.arange(50.0)[None, :] + np.arange(3.0)[:, None] * 100
    blaze = np.full((3, 50), 2.0)
    hdr = fits.Header()
    hdr['OBJECT'] = 'TEST'
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(flux, name='FLUX'),
                  fits.ImageHDU(wave, name='WAVELENGTH'),
                  fits.ImageHDU(blaze, name='BLAZE')]).writeto(path)
    return flux, wave, blaze


@pytest.mark.parametrize('instrument', ['HARPS', 'HARPSN', 'EXPRES', 'NEID'])
def test_write_tellu_cleaned_reads_back(tmp_path, instrument):
    inst = _inst(instrument)
    path = str(tmp_path / 'in.fits')
    flux, wave, blaze = _essp_file(path, inst)
    sci, hdr = inst.load_science_file(path)
    cleaned = sci * 1.5
    props = dict(FILENAME=path, pre_cleaned_flux=cleaned,
                 pre_cleaned_exponent_water=1.1,
                 pre_cleaned_exponent_others=1.2)
    out = str(tmp_path / 'out.fits')
    inst.write_tellu_cleaned(out, props, hdr)
    sci2, hdr2 = inst.load_science_file(out)
    assert sci2 == pytest.approx(cleaned)
    assert inst.get_wave_solution(out) == pytest.approx(wave / 10)
    assert inst.load_blaze('', out, normalize=False) == pytest.approx(blaze)
    assert hdr2[inst.params['KW_TAU_H2O']] == 1.1
    assert hdr2[inst.params['KW_TAU_OTHERS']] == 1.2


def test_no_flux_extension_stops_with_a_message(tmp_path):
    inst = _inst('HARPS')
    path = str(tmp_path / 'in.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.ones((2, 2)), name='OTHER')]).writeto(path)
    hdr = inst.load_header(path)
    props = dict(FILENAME=path, pre_cleaned_flux=np.ones((2, 2)),
                 pre_cleaned_exponent_water=1.0,
                 pre_cleaned_exponent_others=1.0)
    with pytest.raises(base_classes.LblException, match='FLUX'):
        inst.write_tellu_cleaned(str(tmp_path / 'out.fits'), props, hdr)

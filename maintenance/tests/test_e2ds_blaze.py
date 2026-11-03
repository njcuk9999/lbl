#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The blaze of an e2ds of the HARPS and HARPS-N DRS: spectrum and blaze are both per
pixel, so the blaze reaches the recipes without a wavelength gradient.

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest
from astropy.io import fits

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _setup(tmp_path, instrument):
    """A small e2ds with its wave solution in the header (a steep dispersion so
    that a gradient would show), and its blaze in calib"""
    klass = select.InstDict[instrument]['ORIG']
    inst = klass(parameters.params.copy(), base_classes.ParamDict())
    nby, nbx = 4, 200
    hdr = fits.Header()
    hdr[inst.params['KW_WAVEORDN']] = nby
    hdr[inst.params['KW_WAVEDEGN']] = 2
    for order in range(nby):
        coeffs = [5000.0 + 100 * order, 0.05, 1e-4]
        for deg, coeff in enumerate(coeffs):
            num = order * 3 + deg
            hdr[inst.params['KW_WAVECOEFFS'].format(num)] = coeff
    calib = tmp_path / 'calib'
    calib.mkdir()
    blaze = 100.0 + np.sin(np.arange(nbx) / 40.0)[None, :] * np.ones((nby, 1))
    fits.PrimaryHDU(blaze).writeto(calib / 'blaze.fits')
    hdr[inst.params['KW_BLAZE_FILE']] = 'blaze.fits'
    path = tmp_path / 'e2ds.fits'
    fits.PrimaryHDU(np.ones((nby, nbx)), header=hdr).writeto(path)
    return inst, str(path), str(calib), blaze


@pytest.mark.parametrize('instrument', ['HARPS', 'HARPSN'])
def test_blaze_from_science_is_the_file_itself(tmp_path, instrument):
    inst, path, calib, blaze = _setup(tmp_path, instrument)
    sci, hdr = inst.load_science_file(path)
    out, corrected = inst.load_blaze_from_science(path, sci, hdr, calib,
                                                  normalize=False)
    assert not corrected
    assert out == pytest.approx(blaze)


@pytest.mark.parametrize('instrument', ['HARPS', 'HARPSN'])
def test_load_blaze_is_the_file_itself(tmp_path, instrument):
    inst, path, calib, blaze = _setup(tmp_path, instrument)
    out = inst.load_blaze(calib + '/blaze.fits', science_file=path,
                          normalize=False)
    assert out == pytest.approx(blaze)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The bands and cross-order regions of the compil step (get_binned_parameters): the
'' region holds every line, the halves follow the detector's real width, and the
bands the binned band plot asks for exist (or the plot does without them).

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _registered():
    for instrument, sources in select.InstDict.items():
        if instrument == 'Generic':
            continue
        for source, klass in sources.items():
            yield instrument, source, klass


def _inst(klass, **extra):
    args = base_classes.ParamDict()
    for key, value in extra.items():
        args.set(key, value, source='test')
    return klass(parameters.params.copy(), args)


def _regions_of(binned, xpix):
    """the regions a line at xpix falls in (as general.py masks them)"""
    out = []
    for name, low, high in zip(binned['region_names'], binned['region_low'],
                               binned['region_high']):
        if low < xpix < high:
            out.append(name)
    return out


@pytest.mark.parametrize('instrument,source,klass', list(_registered()))
def test_every_line_is_in_the_unnamed_region(instrument, source, klass):
    binned = _inst(klass).get_binned_parameters()
    assert binned['region_names'][0] == ''
    for xpix in (1, 4095, 4100, 9000, 20000):
        assert '' in _regions_of(binned, xpix)


@pytest.mark.parametrize('instrument,source,klass', list(_registered()))
def test_binned_plot_bands_exist(instrument, source, klass):
    inst = _inst(klass)
    bands = inst.get_binned_parameters()['bands']
    for num in (1, 2, 3):
        name = inst.params['COMPILE_BINNED_BAND{0}'.format(num)]
        assert name is None or name in bands, (num, name, bands)
    # a colour needs two different bands
    band2 = inst.params['COMPILE_BINNED_BAND2']
    band3 = inst.params['COMPILE_BINNED_BAND3']
    if band2 is not None or band3 is not None:
        assert band2 != band3


def test_espresso_halves_follow_its_9111_pixels():
    inst = _inst(select.InstDict['ESPRESSO']['None'])
    binned = inst.get_binned_parameters()
    assert binned['region_names'] == ['', '_0-4555', '_4555-9111']
    assert _regions_of(binned, 9000) == ['', '_4555-9111']


def test_harps_keeps_its_column_names():
    inst = _inst(select.InstDict['HARPS']['ORIG'])
    binned = inst.get_binned_parameters()
    assert binned['region_names'] == ['', '_0-2044', '_2044-4088']
    assert _regions_of(binned, 3000) == ['', '_2044-4088']


def test_width_from_the_sample_wave_grid(tmp_path):
    """SOPHIE has no width in the class: the sample wave grid gives it"""
    klass = select.InstDict['SOPHIE']['None']
    calib = tmp_path / 'calib'
    calib.mkdir()
    inst = _inst(klass, DATA_DIR=str(tmp_path))
    grid = np.ones((3, 5000))
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(grid)]).writeto(
        calib / inst.params['SAMPLE_WAVE_GRID_FILE'])
    binned = inst.get_binned_parameters()
    assert binned['region_names'] == ['', '_0-2500', '_2500-5000']
    assert _regions_of(binned, 4500) == ['', '_2500-5000']


def test_without_a_grid_the_width_is_4096(tmp_path):
    inst = _inst(select.InstDict['SOPHIE']['None'], DATA_DIR=str(tmp_path))
    assert inst.detector_npixel() == 4096


def test_binned_band_plot_without_its_bands(monkeypatch):
    """HARPS has only g: the plot shows g and the whole, with no colour, and
    raises nothing for the bands it does not have"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lbl.science import plot
    monkeypatch.setattr(plot, 'PLT_MOD', plt)
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    inst = _inst(select.InstDict['HARPS']['ORIG'], PLOT=True,
                 PLOT_COMPIL_BINNED=True, OBJECT_SCIENCE='A',
                 OBJECT_COMPARISON='A')
    rdb = Table(dict(rjd=np.arange(5.0), vrad=np.zeros(5), svrad=np.ones(5),
                     vrad_g=np.zeros(5), svrad_g=np.ones(5)))
    for bands in (('r', 'g', 'r'), ('g', None, None), (None, None, None)):
        for num, name in enumerate(bands, start=1):
            inst.params.set('COMPILE_BINNED_BAND{0}'.format(num), name,
                            source='test')
        plot.compil_binned_band_plot(inst, rdb)

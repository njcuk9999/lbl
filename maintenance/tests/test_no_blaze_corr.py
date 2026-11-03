#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
no_blaze_corr, the artificial blaze for files without one: the diffraction order
comes from the spacing of the order centres. ESPRESSO has two slices per order,
so it takes every other centre; the others have one row per order.

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _inst(instrument, source='None'):
    klass = select.InstDict[instrument][source]
    return klass(parameters.params.copy(), base_classes.ParamDict())


def _wave_map(orders, slices=1, npix=4096):
    """A wave map by the grating equation: lambda m = const, one free spectral
    range wide per order; with two slices each order is repeated, 1e-3 nm off"""
    const = 500.0 * 120
    rows = []
    for order in orders:
        centre = const / order
        fsr = centre / order
        wave = centre + fsr * np.linspace(-0.5, 0.5, npix)
        for slice_num in range(slices):
            rows.append(wave + 1e-3 * slice_num)
    return np.array(rows)


def _edge(blaze):
    good = np.arange(3, blaze.shape[0] - 3)
    return np.median(0.5 * (blaze[good, 20] + blaze[good, -20]))


@pytest.mark.parametrize('instrument,source', [('HARPS', 'ESO'),
                                               ('HARPSN', 'ESO'),
                                               ('NIRPS_HE', 'ESO'),
                                               ('NIRPS_HA', 'ESO'),
                                               ('HARPS', 'ESSP'),
                                               ('EXPRES', 'ORIG')])
def test_order_per_row_blaze_is_a_sinc_of_one_order(instrument, source):
    """Half a free spectral range from the centre sinc**2 is (2/pi)**2 = 0.405;
    a stride of two takes a period twice too long, which gives 0.81"""
    wave = _wave_map(np.arange(150, 90, -1))
    sci, blaze = _inst(instrument, source).no_blaze_corr(np.ones_like(wave),
                                                         wave)
    assert sci.shape == wave.shape
    assert _edge(blaze) == pytest.approx((2 / np.pi) ** 2, abs=0.02)


def test_espresso_takes_every_other_centre():
    wave = _wave_map(np.arange(150, 90, -1), slices=2)
    _, blaze = _inst('ESPRESSO').no_blaze_corr(np.ones_like(wave), wave)
    # without its stride the pairs of slices give an absurd order, and a
    #   period far too short: near zero at the ends
    assert _edge(blaze) == pytest.approx((2 / np.pi) ** 2, abs=0.02)


def test_generic_has_one_centre_per_row():
    """The generic class has no fixed parameters to instantiate it through
    the registry: the body is compared"""
    import inspect

    from lbl.instruments import generic
    source = inspect.getsource(generic.Generic.no_blaze_corr)
    assert '::2' not in source

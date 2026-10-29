#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The correction's stamp reads the configuration a correction depends on
(lbl.recipes.lbl_pca2d.nominal_digest): the general block of nominal.yaml
and the instrument's own settings (the PCA2D_* parameters of its LBL class),
parsed. A window, a comment or another instrument's setting does not make a
correction again.

Created on 2026-10-23

@author: artigau
"""
import copy

from lbl.pca2d import config as spconfig
from lbl.recipes import lbl_pca2d

NOMINAL = dict(
    general=dict(correct=dict(nsig_cut=3.0), output=dict(model_directory='m')))
LAYERS = dict(
    HARPS=dict(quality=dict(drs_modes=['HARPS']),
               output=dict(windows=['628.0:4'])),
    NIRPS_HE=dict(input=dict(nightly_stack=True),
                  output=dict(windows=['1267:2'])))


def _digest(monkeypatch, nominal, layers, instrument):
    monkeypatch.setattr(lbl_pca2d, 'read_nominal', lambda: nominal)
    monkeypatch.setattr(spconfig, 'instrument_layer',
                        lambda name, source: copy.deepcopy(layers[name]))
    return lbl_pca2d.nominal_digest(instrument, 'ANY')


def test_a_window_or_another_instrument_changes_nothing(monkeypatch):
    base = _digest(monkeypatch, NOMINAL, LAYERS, 'NIRPS_HE')
    moved = copy.deepcopy(LAYERS)
    moved['HARPS']['output']['windows'] = ['630.75:2']
    moved['NIRPS_HE']['output']['windows'] = ['1264.25:2']
    general = copy.deepcopy(NOMINAL)
    general['general']['output']['model_directory'] = 'elsewhere'
    assert _digest(monkeypatch, general, moved, 'NIRPS_HE') == base
    # HARPS's own setting: the NIRPS corrections stay, HARPS's are made again
    other = copy.deepcopy(LAYERS)
    other['HARPS']['quality']['drs_modes'] = ['HARPS', 'HARPS+NIRPS']
    assert _digest(monkeypatch, NOMINAL, other, 'NIRPS_HE') == base
    assert _digest(monkeypatch, NOMINAL, other, 'HARPS') != \
        _digest(monkeypatch, NOMINAL, LAYERS, 'HARPS')
    # a general setting: every instrument
    changed = copy.deepcopy(NOMINAL)
    changed['general']['correct']['nsig_cut'] = 2.5
    assert _digest(monkeypatch, changed, LAYERS, 'NIRPS_HE') != base

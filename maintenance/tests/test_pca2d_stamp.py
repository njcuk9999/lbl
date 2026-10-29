#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The correction's stamp reads the nominal configuration as a correction
depends on it (lbl.recipes.lbl_pca2d.nominal_digest): a window, a comment
or another instrument's setting does not make a correction again.

Created on 2026-10-23

@author: artigau
"""
import copy

from lbl.recipes import lbl_pca2d

NOMINAL = dict(
    general=dict(correct=dict(nsig_cut=3.0), output=dict(windows=['1267:2'])),
    instruments=dict(
        HARPS=dict(lbl=dict(instrument='HARPS'),
                   quality=dict(drs_modes=['HARPS']),
                   output=dict(windows=['628.0:4'])),
        NIRPS=dict(lbl=dict(instrument='NIRPS_HE'),
                   input=dict(nightly_stack=True))))


def _digest(monkeypatch, nominal, instrument):
    monkeypatch.setattr(lbl_pca2d, 'read_nominal', lambda: nominal)
    return lbl_pca2d.nominal_digest(instrument)


def test_a_window_or_another_instrument_changes_nothing(monkeypatch):
    base = _digest(monkeypatch, NOMINAL, 'NIRPS_HE')
    moved = copy.deepcopy(NOMINAL)
    moved['instruments']['HARPS']['output']['windows'] = ['630.75:2']
    moved['general']['output']['windows'] = ['1264.25:2']
    assert _digest(monkeypatch, moved, 'NIRPS_HE') == base
    # HARPS's own setting: the NIRPS corrections stay, HARPS's are made again
    other = copy.deepcopy(NOMINAL)
    other['instruments']['HARPS']['quality']['drs_modes'] = ['HARPS',
                                                             'HARPS+NIRPS']
    assert _digest(monkeypatch, other, 'NIRPS_HE') == base
    assert _digest(monkeypatch, other, 'HARPS') != \
        _digest(monkeypatch, NOMINAL, 'HARPS')
    # a general setting: every instrument
    general = copy.deepcopy(NOMINAL)
    general['general']['correct']['nsig_cut'] = 2.5
    assert _digest(monkeypatch, general, 'NIRPS_HE') != base

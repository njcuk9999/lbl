#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The order in which lbl_wrap calls the recipes, with recorders in their place:
the reset once before the first object, a string OBJECT_SCIENCE as one object,
RUN_LBL_RESMAP accepted, and the templates of the telluric cleaning.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes
from lbl.recipes import lbl_wrap


@pytest.fixture
def calls(monkeypatch):
    """every recipe of the wrap replaced by a recorder of its arguments"""
    record = []
    for name in ('lbl_reset', 'lbl_telluclean', 'lbl_template', 'lbl_mask',
                 'lbl_compute', 'lbl_compile'):
        module = getattr(lbl_wrap, name)

        def main(_name=name, **kwargs):
            record.append((_name, kwargs))
        monkeypatch.setattr(module, 'main', main)
    return record


def _params(data_dir, **extra):
    rparams = dict(INSTRUMENT='HARPS', DATA_SOURCE='ORIG',
                   DATA_DIR=str(data_dir), DATA_TYPES=['SCIENCE', 'SCIENCE'],
                   OBJECT_SCIENCE=['STAR1', 'STAR2'],
                   OBJECT_COMPARISON=['STAR1', 'STAR2'],
                   OBJECT_TEFF=[3500, 3600], RUN_LBL_TEMPLATE=True,
                   RUN_LBL_COMPUTE=True, REPORT=False)
    rparams.update(extra)
    return rparams


def test_reset_once_before_every_object(tmp_path, calls):
    lbl_wrap.main(_params(tmp_path, RUN_LBL_RESET=True))
    names = [name for name, _ in calls]
    assert names.count('lbl_reset') == 1
    assert names[0] == 'lbl_reset'
    computed = [kw['object_science'] for name, kw in calls
                if name == 'lbl_compute']
    assert computed == ['STAR1', 'STAR2']


def test_resmap_does_not_stop_the_run(tmp_path, calls):
    lbl_wrap.main(_params(tmp_path, RUN_LBL_RESMAP=True))
    for _, kwargs in calls:
        assert 'RUN_LBL_RESMAP' not in kwargs


def test_a_string_is_one_object(tmp_path, calls):
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                          OBJECT_COMPARISON='GJ1', DATA_TYPES='SCIENCE',
                          OBJECT_TEFF=3500))
    computed = [kw['object_science'] for name, kw in calls
                if name == 'lbl_compute']
    assert computed == ['GJ1']


def test_not_a_name_stops_with_a_message(tmp_path, calls):
    with pytest.raises(base_classes.LblException, match='OBJECT_SCIENCE'):
        lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE=12))


def _cleaning(calls):
    """the templates the second cleaning pass uses, and what compute gets"""
    second = [kw for name, kw in calls if name == 'lbl_telluclean'
              and kw['telluclean_use_template']]
    compute = [kw for name, kw in calls if name == 'lbl_compute']
    return second, compute


def test_cleaning_with_itself(tmp_path, calls):
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                          OBJECT_COMPARISON='GJ1', DATA_TYPES='SCIENCE',
                          OBJECT_TEFF=3500, RUN_LBL_TELLUCLEAN=True))
    second, compute = _cleaning(calls)
    assert second[0]['object_comparison'] == 'GJ1_tc'
    assert compute[0]['object_science'] == 'GJ1_tc'
    assert compute[0]['object_comparison'] == 'GJ1_tc'


def test_cleaning_with_a_vetted_template(tmp_path, calls):
    """the cleaning uses the star's own template; the velocities are measured
    against the vetted template, whose name gets no _tc"""
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                          OBJECT_COMPARISON='VETTED_GL699',
                          DATA_TYPES='SCIENCE', OBJECT_TEFF=3500,
                          RUN_LBL_TELLUCLEAN=True))
    second, compute = _cleaning(calls)
    assert second[0]['object_comparison'] == 'GJ1_tc'
    assert compute[0]['object_comparison'] == 'VETTED_GL699'
    templates = [kw['object_comparison'] for name, kw in calls
                 if name == 'lbl_template']
    assert 'VETTED_GL699_tc' not in templates


def test_cleaning_with_another_star_cleaned(tmp_path, calls):
    (tmp_path / 'science' / 'GJ2_tc').mkdir(parents=True)
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                          OBJECT_COMPARISON='GJ2', DATA_TYPES='SCIENCE',
                          OBJECT_TEFF=3500, RUN_LBL_TELLUCLEAN=True))
    second, compute = _cleaning(calls)
    # the cleaning of GJ1 divides GJ1's own lines, not GJ2's
    assert second[0]['object_comparison'] == 'GJ1_tc'
    assert compute[0]['object_comparison'] == 'GJ2_tc'


def test_cleaning_with_another_star_not_cleaned_stops(tmp_path, calls):
    with pytest.raises(base_classes.LblException, match='Clean GJ2 first'):
        lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                              OBJECT_COMPARISON='GJ2', DATA_TYPES='SCIENCE',
                              OBJECT_TEFF=3500, RUN_LBL_TELLUCLEAN=True))
    # it stops before cleaning anything
    assert not [name for name, _ in calls if name == 'lbl_telluclean']


def test_a_pca2d_run_writes_one_report(tmp_path, calls, monkeypatch):
    """The delivered object is measured, for the report to set its
    velocities beside the corrected ones, but gets no report of its own:
    <object>_<template>.pdf beside <object>_pca2d_<template>.pdf confused
    (TOI1452 against VETTED_GJ1289, 2026-11-13). Its velocities are in the
    corrected object's report, the only one."""
    import lbl.report
    from lbl.recipes import lbl_pca2d
    reports, merged = [], []
    monkeypatch.setattr(lbl.report, 'make_report',
                        lambda **kw: reports.append(kw['title'])
                        or '%s.pdf' % kw['title'])
    monkeypatch.setattr(lbl_pca2d, 'frozen_basis', lambda *a, **k: None)
    monkeypatch.setattr(lbl_pca2d, 'main',
                        lambda object_science, object_comparison, **kw:
                        (object_science + '_pca2d', object_comparison))
    monkeypatch.setattr(lbl_pca2d, 'report',
                        lambda **kw: merged.append(kw))
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE=['TOI1452'],
                          OBJECT_COMPARISON=['VETTED_GJ1289'],
                          DATA_TYPES=['SCIENCE'], OBJECT_TEFF=[3200],
                          RUN_LBL_COMPILE=True, REPORT=True,
                          PCA2D=dict(n_star=1)))
    compiled = [kw['object_science'] for name, kw in calls
                if name == 'lbl_compile']
    assert compiled == ['TOI1452', 'TOI1452_pca2d']
    assert reports == ['TOI1452_pca2d_VETTED_GJ1289']
    assert len(merged) == 1
    assert merged[0]['delivered'] == 'TOI1452'
    assert merged[0]['lbl_report'] == 'TOI1452_pca2d_VETTED_GJ1289.pdf'

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


def test_one_template_call_and_two_masks_against_another_star(tmp_path,
                                                             calls):
    """lbl_template makes the science's template and the comparison's in one
    call: a second call with the science as its comparison made the same
    file again. lbl_mask makes one mask per call: the comparison's, then
    the science's own"""
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE='GJ1',
                          OBJECT_COMPARISON='VETTED_GL699',
                          DATA_TYPES='SCIENCE', OBJECT_TEFF=3500,
                          RUN_LBL_MASK=True))
    # the steps themselves (overwrite as SKIP_LBL_* say); those made with
    #   overwrite=False are the net for what compute finds missing, here
    #   where the recipes write nothing
    templates = [kw for name, kw in calls
                 if name == 'lbl_template' and kw['overwrite']]
    assert len(templates) == 1
    assert (templates[0]['object_science'],
            templates[0]['object_comparison']) == ('GJ1', 'VETTED_GL699')
    masks = [kw['object_comparison'] for name, kw in calls
             if name == 'lbl_mask' and kw['overwrite']]
    assert masks == ['VETTED_GL699', 'GJ1']


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
    # each measured once; the corrected object's table kept for its report
    #   (the last of PCA2D's passes measured it already)
    compiled = [(kw['object_science'], kw['skip_done']) for name, kw in calls
                if name == 'lbl_compile']
    assert [name for name, skip in compiled if not skip] == [
        'TOI1452', 'TOI1452_pca2d']
    assert compiled[-1] == ('TOI1452_pca2d', True)
    assert reports == ['TOI1452_pca2d_VETTED_GJ1289']
    assert len(merged) == 1
    assert merged[0]['delivered'] == 'TOI1452'
    assert merged[0]['lbl_report'] == 'TOI1452_pca2d_VETTED_GJ1289.pdf'


def _passes_wrap(tmp_path, calls, monkeypatch, changes, small_rv=True,
                 pca2d=None, kept=None, skip=None):
    """A PCA2D run whose LBL velocities of the corrected spectra move by
    `changes[k]` [m/s] from pass k to pass k + 1 (err 2 m/s): the star
    frame each pass was given, and the passes recorded"""
    import json
    from lbl.recipes import lbl_pca2d
    from lbl.science import velocity_box
    corrections, marked, npass = [], [], [0]
    folder = tmp_path / 'science' / 'STAR_pca2d'
    folder.mkdir(parents=True, exist_ok=True)
    files = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
    for name in files:
        (folder / name).write_text('')
    signal = {'a.fits': -100.0, 'b.fits': 0.0, 'c.fits': 100.0,
              'd.fits': 50.0}

    def main(object_science, object_comparison, velocities_from=None,
             first=None, discard_previous=False, **kw):
        corrections.append(dict(velocities_from or {}))
        npass[0] += 1
        if kept and npass[0] == 1:
            lbl_pca2d.SETTLED['STAR_pca2d'] = kept
        state = str(tmp_path / 'done.json')
        with open(state, 'w') as handle:
            json.dump(dict(stamp='s%d' % npass[0]), handle)
        lbl_pca2d.STATES['STAR_pca2d'] = state
        return object_science + '_pca2d', object_comparison + '_pca2d'

    def compute(**kw):
        k = npass[0] - 1
        shift = sum(changes[:k])
        velocity_box.update(str(tmp_path), kw['object_science'], {
            name: dict(v=v * (1 + shift / 100.0), err=2.0)
            for name, v in signal.items()}, 'lbl')
    monkeypatch.setattr(lbl_pca2d, 'frozen_basis', lambda *a, **k: None)
    monkeypatch.setattr(lbl_pca2d, 'main', main)
    monkeypatch.setattr(lbl_pca2d, 'report', lambda **kw: None)
    monkeypatch.setattr(lbl_pca2d, 'mark_passes',
                        lambda names, n, settled: marked.append((n, settled)))
    monkeypatch.setattr(lbl_wrap.lbl_compute, 'main', compute)
    lbl_wrap.main(_params(tmp_path, OBJECT_SCIENCE=['STAR'],
                          OBJECT_COMPARISON=['STAR'], DATA_TYPES=['SCIENCE'],
                          OBJECT_TEFF=[3200], RUN_LBL_COMPILE=True,
                          REPORT=False, SMALL_RV_SHIFT=small_rv,
                          SKIP_LBL_PCA2D=skip,
                          PCA2D=dict(n_star=1, **(pca2d or {}))))
    return corrections, marked


def test_pca2d_in_two_passes_at_least(tmp_path, calls, monkeypatch):
    """the first pass on the BERV alone, the second on LBL's velocities of
    the first's corrected spectra; settled at once (they did not move):
    two passes, whatever SMALL_RV_SHIFT says"""
    for small_rv in (True, False):
        corrections, marked = _passes_wrap(tmp_path / str(small_rv), calls,
                                           monkeypatch, [0, 0, 0],
                                           small_rv=small_rv)
        assert len(corrections) == 2
        assert corrections[0] == {'STAR': None}
        frame = corrections[1]['STAR']
        assert frame['c.fits'] - frame['a.fits'] == pytest.approx(0.2)
        assert marked == [(2, True)]
    # LBL measured each pass's corrected spectra
    computed = [kw['object_science'] for name, kw in calls
                if name == 'lbl_template']
    assert computed.count('STAR_pca2d') >= 2


def test_passes_go_on_until_lbl_s_velocities_settle(tmp_path, calls,
                                                    monkeypatch):
    """a pass whose velocities still moved (5 % of 100 m/s, against 2 m/s
    errors) is followed by another; at most max_passes, the last kept"""
    corrections, marked = _passes_wrap(tmp_path / 'a', calls, monkeypatch,
                                       [5, 0, 0])
    assert len(corrections) == 3 and marked == [(3, True)]
    corrections, marked = _passes_wrap(tmp_path / 'b', calls, monkeypatch,
                                       [5, 5, 5, 5],
                                       pca2d=dict(max_passes=3))
    assert len(corrections) == 3 and marked == [(3, False)]


def test_a_correction_kept_is_measured_once(tmp_path, calls, monkeypatch):
    """the settled last pass of an earlier run's passes, or SKIP_LBL_PCA2D
    True: no other pass"""
    corrections, _ = _passes_wrap(tmp_path / 'a', calls, monkeypatch,
                                  [5, 5], kept=3)
    assert len(corrections) == 1
    corrections, _ = _passes_wrap(tmp_path / 'b', calls, monkeypatch,
                                  [5, 5], skip=True)
    assert len(corrections) == 1
